"""
Interactive FYP Architectural Evolution Walkthrough ("How It Works").
Details Stages 1 through 6, scientific motivations, and architectural milestones.
"""

import streamlit as st


def render_how_it_works_page():
    st.markdown("### 🧬 The 6-Stage Research Evolution")
    st.markdown("""
    This project followed a systematic, decoupled systems-engineering methodology to determine:
    1. *Does multi-modal fusion justify its computational cost?*
    2. *At what architectural stage (feature-level vs. decision-level) should fusion occur?*
    3. *Can a modern detector architecture extract higher utility from fixed generative representations?*
    """)

    stages = [
        {
            "num": "Stage 1",
            "name": "Zero-Shot Pre-Trained TarDAL Fusion",
            "desc": "Off-the-shelf TarDAL generative fusion with frozen pretrained detection head.",
            "finding": "Catastrophic failure (16.97% Val mAP@50). Proved that off-the-shelf vision heads cannot interpret novel fused pixel distributions without domain adaptation.",
            "badge": "Baseline Milestone (16.97% Val)"
        },
        {
            "num": "Stage 2",
            "name": "Adapted Detection Head (Frozen Generator)",
            "desc": "Froze the TarDAL generator and fully adapted the YOLOv5su detection head on fused images.",
            "finding": "Performance soared by +56.33 pp to 73.30% Val mAP@50. Proved that the generative representation was rich in spatial features, but required dedicated head adaptation.",
            "badge": "Head Adaptation (+56.3 pp gain)"
        },
        {
            "num": "Stage 3",
            "name": "Task-Driven Joint Adaptation",
            "desc": "Jointly trained generator and detection head using end-to-end task-specific detection loss gradients.",
            "finding": "Achieved 74.57% Val mAP@50 and peak 48.56% Test Benchmark mAP@50. Established the highest generalization retention rate (65.1%) under distribution shift.",
            "badge": "Feature Fusion Leader (48.56% Test)"
        },
        {
            "num": "Stage 4",
            "name": "Direct Unimodal Baselines (RGB & IR)",
            "desc": "Trained standalone YOLOv5su detectors directly on raw Visible Optical (RGB) and Thermal Infrared (IR) inputs.",
            "finding": "Revealed critical physical sensor trade-offs: Direct RGB scored 76.40% on Val, but missed pedestrians in dark test scenes (28.1% recall). Direct IR scored 75.43% on pedestrians, but was blind to cold lamps (1.87%). Proved multi-modal necessity.",
            "badge": "Unimodal Baselines (50 Epochs)"
        },
        {
            "num": "Stage 5",
            "name": "Decision-Level Late Fusion (WBF)",
            "desc": "Decoupled the sensor pipelines: independent unimodal detectors combined via class-constrained Weighted Boxes Fusion.",
            "finding": "Bypassed the TarDAL 61.6 ms generative bottleneck entirely. Delivers ~51.5 FPS real-time throughput, outperforming unimodal models by +4.7 pp with 100% graceful degradation under sensor dropout.",
            "badge": "Real-Time Leader (~51.5 FPS / 19.4 ms)"
        },
        {
            "num": "Stage 6",
            "name": "Modern Detector Feature Fusion (YOLO11s)",
            "desc": "Fine-tuned a state-of-the-art YOLO11s detector (with C3k2 extractors and C2PSA attention) on the frozen Stage 3 generative representation.",
            "finding": "Achieved project-record validation accuracy (81.08%) and peak target accuracy on core traffic classes: Pedestrians (77.49%) and Vehicles (85.26%). Proved modern attention blocks exploit fused features with higher precision.",
            "badge": "Peak Target Accuracy (81.08% Val)"
        }
    ]

    for s in stages:
        st.markdown(f"""
        <div style="background-color:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #1e40af;padding:16px 20px;margin-bottom:16px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <h4 style="margin:0;color:#0f172a;font-weight:800;font-size:16px;">{s['num']}: {s['name']}</h4>
                <span style="background-color:#eff6ff;border:1px solid #bfdbfe;color:#1e40af;padding:3px 10px;border-radius:6px;font-size:11.5px;font-weight:700;font-family:monospace;">{s['badge']}</span>
            </div>
            <p style="margin:8px 0 6px 0;font-size:13.5px;color:#1e293b;line-height:1.5;"><b style="color:#0f172a;font-weight:700;">Architecture:</b> {s['desc']}</p>
            <div style="font-size:13px;color:#334155;line-height:1.5;margin-top:8px;padding-top:8px;border-top:1px solid #f1f5f9;">
                <b style="color:#1e40af;font-weight:800;">Key Scientific Discovery:</b> {s['finding']}
            </div>
        </div>
        """, unsafe_allow_html=True)
