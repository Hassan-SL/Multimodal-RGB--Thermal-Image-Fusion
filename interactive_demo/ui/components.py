"""
Reusable UI component renderers: dynamic pipeline diagrams, metrics cards,
sensor status pills, and model information cards.
"""

import streamlit as st
from typing import Dict, Any, Optional


def render_header():
    """Renders top research demonstrator banner."""
    st.markdown("""
    <div class="fyp-header">
        <h1 class="fyp-title">🔬 MULTIMODAL RGB–IR OBJECT DETECTION</h1>
        <div class="fyp-subtitle">Live Deployment & Research Demonstrator • M3FD Benchmark Pipeline</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_row(
    fps: float,
    latency_ms: float,
    num_objs: int,
    gpu_mem: str,
    active_model: str
):
    """Renders persistent 5-card performance and telemetry bar."""
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{fps:.1f}</div>
            <div class="metric-label">Throughput (FPS)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{latency_ms:.1f} <span style="font-size:14px;color:#64748b;font-weight:600;">ms</span></div>
            <div class="metric-label">End-to-End Latency</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{num_objs}</div>
            <div class="metric-label">Objects Detected</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:18px;margin-top:6px;font-weight:800;">{gpu_mem}</div>
            <div class="metric-label">GPU Memory (VRAM)</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:15px;color:#1e40af;font-weight:800;margin-top:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{active_model}</div>
            <div class="metric-label">Active Architecture</div>
        </div>
        """, unsafe_allow_html=True)


def render_pipeline_diagram(model_key: str, rgb_active: bool = True, ir_active: bool = True, w_rgb: float = 0.60, w_ir: float = 0.40):
    """Renders an adaptive dynamic pipeline architecture diagram."""
    if model_key == "rgb":
        diag = """
  Visible Optical (Vis) [640×640×3]
          │
          ▼
  ┌───────────────────────────────────┐
  │  YOLOv5su Direct Optical (9.12M)  │ ──► Detections [People, Car, ...]
  └───────────────────────────────────┘
        """
    elif model_key == "ir":
        diag = """
  Thermal Infrared (IR) [640×640×1]
          │
          ▼ (3-Channel Luminance Replication)
  ┌───────────────────────────────────┐
  │  YOLOv5su Direct Thermal (9.12M)  │ ──► Detections [People, Car, ...]
  └───────────────────────────────────┘
        """
    elif model_key == "stage3":
        diag = """
  Visible Optical (Vis) ──┐
                          ├─► [Y_vis, Y_ir] ──► TarDAL Generator ──► YCrCb Reconstruct ──► YOLOv5su ──► Detections
  Thermal Infrared (IR) ──┘   (Normalized [0,1])   (296K Params)     (Tanh Clamping)     (9.12M)
        """
    elif model_key == "stage6":
        diag = """
  Visible Optical (Vis) ──┐
                          ├─► [Y_vis, Y_ir] ──► TarDAL Generator ──► YCrCb Reconstruct ──► YOLO11s (C3k2/C2PSA) ──► Detections
  Thermal Infrared (IR) ──┘   (Normalized [0,1])   (296K Frozen)     (Tanh Clamping)     (9.43M Modern Head)
        """
    elif model_key == "stage5":
        rgb_tag = "[ONLINE]" if rgb_active else "[OFFLINE ⚠️]"
        ir_tag = "[ONLINE]" if ir_active else "[OFFLINE ⚠️]"
        if rgb_active and ir_active:
            diag = f"""
  Visible Optical (Vis) {rgb_tag} ──► YOLOv5su (RGB) ────┐ (w_rgb={w_rgb:.2f})
                                                        ├─► Weighted Boxes Fusion (WBF) ──► Final Detections
  Thermal Infrared (IR) {ir_tag} ──► YOLOv5su (IR) ─────┘ (w_ir={w_ir:.2f})
            """
        elif rgb_active and not ir_active:
            diag = """
  Visible Optical (Vis) [ONLINE]   ──► YOLOv5su (RGB) ──► Final Detections (Graceful Fallback)
  Thermal Infrared (IR) [OFFLINE]  ──► [BYPASSED ❌]
            """
        elif ir_active and not rgb_active:
            diag = """
  Visible Optical (Vis) [OFFLINE]  ──► [BYPASSED ❌]
  Thermal Infrared (IR) [ONLINE]   ──► YOLOv5su (IR) ──► Final Detections (Graceful Fallback)
            """
        else:
            diag = """
  Visible Optical (Vis) [OFFLINE]  ──► [DISABLED ❌]
  Thermal Infrared (IR) [OFFLINE]  ──► [DISABLED ❌] ──► ZERO DETECTIONS
            """
    else:
        diag = "Pipeline: Standard Multi-Modal Ingestion"

    st.markdown(f'<div class="pipeline-box"><pre style="margin:0;color:#0f172a;font-weight:600;">{diag}</pre></div>', unsafe_allow_html=True)


def render_model_card(model_key: str):
    """Renders concise research information card for the selected architecture."""
    cards = {
        "stage5": {
            "title": "Stage 5 — Decision-Level Late Fusion (WBF)",
            "type": "Decision-Level Consensus (Box Merging)",
            "params": "18.24M (Dual YOLOv5su Detectors)",
            "val_map50": "74.01%",
            "test_map50_op": "30.29% (Operational, conf=0.25, iou=0.50)",
            "test_map50_bm": "34.10% (Benchmark, conf=0.001, iou=0.60)",
            "latency": "~19.41 ms (Tesla T4)",
            "fps": "~51.5 FPS (Real-Time Automotive Safe >30 FPS)",
            "strength": "100% Graceful Degradation under sensor failure; zero generative latency bottleneck.",
            "limitation": "Dual detector VRAM footprint (~1.4 GB)."
        },
        "stage6": {
            "title": "Stage 6 — TarDAL + YOLO11s Modern Feature Fusion",
            "type": "Feature-Level Fusion (Learned Generator + Modern Attention)",
            "params": "9.43M Detector + 296K Generator",
            "val_map50": "81.08% (Val Live) / 80.40% (Checkpoint)",
            "test_map50_op": "41.65% (Operational, conf=0.25, iou=0.50)",
            "test_map50_bm": "45.72% (Benchmark, conf=0.001, iou=0.60)",
            "latency": "79.42 ms (Tesla T4)",
            "fps": "12.6 FPS (Generator bounded)",
            "strength": "Peak accuracy on core target classes: People 77.49%, Car 85.26% (Project Records).",
            "limitation": "Dense-block generator takes 61.61 ms, violating the 30 FPS automotive constraint."
        },
        "stage3": {
            "title": "Stage 3 — Task-Driven Joint Feature Fusion (YOLOv5su)",
            "type": "Feature-Level Fusion (Joint Backprop Adaptation)",
            "params": "9.12M Detector + 296K Generator",
            "val_map50": "74.57%",
            "test_map50_op": "40.12% (Operational, conf=0.25, iou=0.50)",
            "test_map50_bm": "48.56% (Benchmark, conf=0.001, iou=0.60)",
            "latency": "77.95 ms (Tesla T4)",
            "fps": "12.8 FPS",
            "strength": "Highest unweighted 6-class mAP@50 (48.56%) and highest Retention Rate (65.1%).",
            "limitation": "Generative dense-block creates non-real-time throughput bottleneck."
        },
        "rgb": {
            "title": "Stage 4A — Direct Optical Baseline (RGB Only)",
            "type": "Unimodal Optical (No Fusion)",
            "params": "9.12M (YOLOv5su)",
            "val_map50": "76.40% (50 Epochs Converged)",
            "test_map50_op": "25.59% (Operational, conf=0.25, iou=0.50)",
            "test_map50_bm": "29.33% (Benchmark, conf=0.001, iou=0.60)",
            "latency": "9.42 ms (Tesla T4)",
            "fps": "106.2 FPS",
            "strength": "Exceptional throughput (106 FPS); rich surface texture and color details.",
            "limitation": "Severe pedestrian deficit in dark/shadowed environments (People recall drops to 28.1%)."
        },
        "ir": {
            "title": "Stage 4B — Direct Thermal Baseline (IR Only)",
            "type": "Unimodal Thermal Infrared (No Fusion)",
            "params": "9.12M (YOLOv5su)",
            "val_map50": "72.00% (50 Epochs Converged)",
            "test_map50_op": "23.54% (Operational, conf=0.25, iou=0.50)",
            "test_map50_bm": "28.38% (Benchmark, conf=0.001, iou=0.60)",
            "latency": "9.59 ms (Tesla T4)",
            "fps": "104.3 FPS",
            "strength": "Outstanding thermal body contrast on pedestrians (People 75.43% test).",
            "limitation": "Completely blind to cold unheated obstacles (Lamp 1.87% test)."
        }
    }

    c = cards.get(model_key, cards["stage5"])
    st.markdown(f"""
    <div style="background-color:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #1e40af;border-radius:8px;padding:16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
        <h4 style="margin:0 0 10px 0;color:#0f172a;font-weight:800;font-size:16px;">{c['title']}</h4>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;line-height:1.5;">
            <div><b style="color:#64748b;font-weight:700;">Paradigm:</b> <span style="color:#1e293b;font-weight:600;">{c['type']}</span></div>
            <div><b style="color:#64748b;font-weight:700;">Parameters:</b> <span style="color:#1e293b;font-weight:600;">{c['params']}</span></div>
            <div><b style="color:#64748b;font-weight:700;">Val mAP@50:</b> <span style="color:#15803d;font-weight:800;">{c['val_map50']}</span></div>
            <div><b style="color:#64748b;font-weight:700;">Test mAP (Operational):</b> <span style="color:#1e40af;font-weight:800;">{c['test_map50_op']}</span></div>
            <div><b style="color:#64748b;font-weight:700;">Tesla T4 Latency:</b> <span style="color:#1e293b;font-weight:600;">{c['latency']}</span></div>
            <div><b style="color:#64748b;font-weight:700;">Throughput:</b> <b style="color:#0f172a;font-weight:800;">{c['fps']}</b></div>
        </div>
        <div style="margin-top:12px;padding-top:10px;border-top:1px solid #f1f5f9;font-size:13px;line-height:1.6;">
            <b style="color:#15803d;font-weight:800;">Key Strength:</b> <span style="color:#1e293b;">{c['strength']}</span><br>
            <b style="color:#b91c1c;font-weight:800;">Key Limitation:</b> <span style="color:#1e293b;">{c['limitation']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
