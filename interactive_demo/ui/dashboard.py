"""
Main Live Deployment & Research Demonstration Interface.
Supports Image Pairs, Curated Samples, Streaming Video, Webcam fallback,
Real-time sensor dropout simulation, and live latency decomposition.
"""

import cv2
import time
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Tuple, Optional

from preprocessing.image_loader import load_and_align_pair, validate_image_file, ALLOWED_EXTENSIONS
from pipelines.rgb_pipeline import RGBPipeline
from pipelines.ir_pipeline import IRPipeline
from pipelines.stage3_pipeline import Stage3Pipeline
from pipelines.stage5_pipeline import Stage5Pipeline
from pipelines.stage6_pipeline import Stage6Pipeline
from ui.components import (
    render_header,
    render_metric_row,
    render_pipeline_diagram,
    render_model_card
)
from ui.benchmark_view import run_live_benchmark
from utils.device import get_device_info, get_memory_info


PIPELINES = {
    "stage5": Stage5Pipeline,
    "stage6": Stage6Pipeline,
    "stage3": Stage3Pipeline,
    "rgb": RGBPipeline,
    "ir": IRPipeline
}

def resolve_app_path(rel_path: str) -> Path:
    """Robust multi-environment path resolver."""
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return p
    app_dir = Path(__file__).resolve().parent.parent
    project_root = app_dir.parent
    candidates = [
        app_dir / rel_path,
        project_root / rel_path,
        Path.cwd() / rel_path,
        Path("E:/My Drive/FYP/code") / rel_path,
        Path("/content/drive/MyDrive/FYP/code") / rel_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    return app_dir / rel_path


SAMPLE_PAIRS = {
    "Sample 1: Daytime Street (M3FD_00000)": ("assets/sample_pairs/rgb_00000.png", "assets/sample_pairs/ir_00000.png"),
    "Sample 2: Nighttime Pedestrians (M3FD_00001)": ("assets/sample_pairs/rgb_00001.png", "assets/sample_pairs/ir_00001.png"),
    "Sample 3: Vehicle Cluster (M3FD_00003)": ("assets/sample_pairs/rgb_00003.png", "assets/sample_pairs/ir_00003.png"),
    "Sample 4: Low-Contrast Shadows (M3FD_00004)": ("assets/sample_pairs/rgb_00004.png", "assets/sample_pairs/ir_00004.png")
}


def render_dashboard():
    # --------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # --------------------------------------------------------------------------
    st.sidebar.markdown("### 🎛️ System Controls")

    ui_mode = st.sidebar.radio(
        "Interface Mode",
        ["Demo Mode (Simple)", "Research Mode (Engineering Telemetry)"],
        index=0
    )
    is_research = "Research" in ui_mode

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 1. Architecture / Model")

    model_options = {
        "Stage 5: RGB + IR Late Fusion (WBF) [Real-Time Leader]": "stage5",
        "Stage 6: TarDAL + YOLO11s [Peak Target Accuracy]": "stage6",
        "Stage 3: TarDAL + YOLOv5su [Feature Fusion]": "stage3",
        "Stage 4A: RGB — YOLOv5su [Direct Optical Baseline]": "rgb",
        "Stage 4B: IR — YOLOv5su [Direct Thermal Baseline]": "ir"
    }

    selected_model_label = st.sidebar.selectbox("Active Pipeline", list(model_options.keys()), index=0)
    active_model_key = model_options[selected_model_label]

    # Early pipeline instantiation for video streaming and live inference
    pipeline_cls = PIPELINES.get(active_model_key, Stage5Pipeline)
    pipeline = pipeline_cls()

    st.sidebar.markdown("#### 2. Detection Parameters")
    conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    iou_thresh = st.sidebar.slider("NMS / Match IoU Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

    # Late fusion specific controls
    w_rgb, w_ir = 0.60, 0.40
    if active_model_key == "stage5":
        st.sidebar.markdown("#### 3. WBF Sensor Weights")
        st.sidebar.caption("Validated Research Optimum: **RGB 0.60 / IR 0.40**")
        w_rgb = st.sidebar.slider("Optical Weight (w_RGB)", min_value=0.10, max_value=0.90, value=0.60, step=0.05)
        w_ir = round(1.0 - w_rgb, 2)
        st.sidebar.text(f"Thermal Weight (w_IR): {w_ir:.2f}")

    # Sensor Availability / Dropout Simulation
    st.sidebar.markdown("#### 4. Sensor Availability (Fault Tolerance)")
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        rgb_active = st.checkbox("Visible Optical", value=True, help="Toggle to simulate optical camera failure/blinding")
    with col_s2:
        ir_active = st.checkbox("Thermal IR", value=True, help="Toggle to simulate thermal sensor failure")

    if not rgb_active and not ir_active:
        st.sidebar.error("⚠️ All sensors offline! Enable at least one sensor.")

    # Hardware probe info
    st.sidebar.markdown("---")
    dev_info = get_device_info()
    st.sidebar.markdown(f"**Hardware Device:** `{dev_info['display']}`")
    mem_info = get_memory_info()
    if mem_info['available']:
        st.sidebar.markdown(f"**GPU VRAM Allocated:** `{mem_info['display']}`")

    # --------------------------------------------------------------------------
    # INPUT SOURCE SELECTION
    # --------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 5. Input Stream")
    input_source = st.sidebar.selectbox("Source Type", ["Curated M3FD Samples", "Upload Image Pair", "Upload Video Pair (RGB + IR)", "Live Camera (Webcam)"])

    rgb_frame, ir_frame = None, None
    source_error = None

    if input_source == "Curated M3FD Samples":
        chosen_sample = st.selectbox("Select Synchronized M3FD Scene", list(SAMPLE_PAIRS.keys()), index=0)
        rgb_rel, ir_rel = SAMPLE_PAIRS[chosen_sample]
        rgb_p = resolve_app_path(rgb_rel)
        ir_p = resolve_app_path(ir_rel)
        if rgb_p.exists() and ir_p.exists():
            rgb_frame, ir_frame, source_error = load_and_align_pair(rgb_p, ir_p)
        else:
            source_error = f"Sample files not found at {rgb_p}. Please verify assets directory."

    elif input_source == "Upload Image Pair":
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            up_rgb = st.file_uploader("Upload Visible Optical (RGB)", type=["png", "jpg", "jpeg", "bmp"])
        with col_u2:
            up_ir = st.file_uploader("Upload Thermal Infrared (IR)", type=["png", "jpg", "jpeg", "bmp"])

        if up_rgb is not None and up_ir is not None:
            rgb_bytes = up_rgb.read()
            ir_bytes = up_ir.read()
            rgb_frame, ir_frame, source_error = load_and_align_pair(rgb_bytes, ir_bytes)
        elif up_rgb is not None or up_ir is not None:
            st.info("ℹ️ Please upload BOTH matching visible and infrared frames for paired multi-modal inference.")

    elif "Video" in input_source:
        col_v_opt1, col_v_opt2 = st.columns([3, 1])
        with col_v_opt1:
            st.markdown("##### 📹 Paired Multi-Modal Video Ingestion")
        with col_v_opt2:
            use_sample_vid = st.checkbox("Use Demo Video Pair", value=False)

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            up_rgb_vid = st.file_uploader(
                "Upload Visible Optical Video (RGB)",
                type=["mp4", "avi", "mov", "mkv"],
                key="up_rgb_vid",
                disabled=use_sample_vid
            )
        with col_v2:
            up_ir_vid = st.file_uploader(
                "Upload Thermal Infrared Video (IR)",
                type=["mp4", "avi", "mov", "mkv"],
                key="up_ir_vid",
                disabled=use_sample_vid
            )

        t_rgb_path, t_ir_path = None, None
        if use_sample_vid:
            s_rgb = resolve_app_path("assets/sample_videos/sample_rgb.mp4")
            s_ir = resolve_app_path("assets/sample_videos/sample_ir.mp4")
            if s_rgb.exists() and s_ir.exists():
                t_rgb_path = s_rgb
                t_ir_path = s_ir
            else:
                source_error = "Sample video files not found in assets/sample_videos."
        elif up_rgb_vid is not None and up_ir_vid is not None:
            t_dir = Path("temp_videos")
            t_dir.mkdir(exist_ok=True)
            t_rgb_path = t_dir / f"temp_rgb_{up_rgb_vid.name}"
            t_ir_path = t_dir / f"temp_ir_{up_ir_vid.name}"
            t_rgb_path.write_bytes(up_rgb_vid.read())
            t_ir_path.write_bytes(up_ir_vid.read())
        elif up_rgb_vid is not None and up_ir_vid is None:
            st.info("ℹ️ Please upload the matching Thermal Infrared Video (IR) on the right to complete the synchronized pair.")
        elif up_ir_vid is not None and up_rgb_vid is None:
            st.info("ℹ️ Please upload the matching Visible Optical Video (RGB) on the left to complete the synchronized pair.")

        if t_rgb_path is not None and t_ir_path is not None:
            cap_rgb = cv2.VideoCapture(str(t_rgb_path))
            cap_ir = cv2.VideoCapture(str(t_ir_path))

            total_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
            total_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = max(1, min(total_rgb, total_ir))
            vid_fps = cap_rgb.get(cv2.CAP_PROP_FPS) or 25.0

            cap_rgb.release()
            cap_ir.release()

            st.markdown("###### 🎬 Video Playback & Detection Mode")
            play_mode = st.radio(
                "Select Interaction Mode",
                [
                    "▶️ Live Streaming Detection (Interactive Playback)",
                    "🔍 Frame-by-Frame Scrubber (Detailed Analysis)",
                    "🎬 Render Full Video Player (HTML5 MP4 with Controls)"
                ],
                horizontal=True
            )

            if play_mode == "▶️ Live Streaming Detection (Interactive Playback)":
                st.markdown("""
                <div style="background-color:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #1e40af;padding:10px 14px;border-radius:6px;margin:8px 0;">
                    <b style="color:#1e40af;font-size:13px;">▶️ Live Streaming Player:</b>
                    <span style="color:#334155;font-size:13px;">Streams synchronized frames through the active detection pipeline in real-time with live bounding box overlay.</span>
                </div>
                """, unsafe_allow_html=True)

                c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([2, 2, 2])
                with c_ctrl1:
                    max_stream_frames = st.slider("Max Frames to Stream", 10, total_frames, min(total_frames, 60))
                with c_ctrl2:
                    frame_step = st.selectbox("Frame Stride / Speed", [1, 2, 3], index=0, format_func=lambda x: f"{x}x ({'Every frame' if x==1 else f'Every {x} frames'})")
                with c_ctrl3:
                    st.write("")
                    st.write("")
                    start_stream = st.button("▶️ Start Live Playback", type="primary", use_container_width=True)

                if start_stream:
                    cap_rgb = cv2.VideoCapture(str(t_rgb_path))
                    cap_ir = cv2.VideoCapture(str(t_ir_path))

                    pbar = st.progress(0)
                    stream_status = st.empty()

                    c_stream1, c_stream2 = st.columns(2)
                    with c_stream1:
                        st.markdown("**Visible Optical Stream (RGB)**")
                        ph_rgb = st.empty()
                    with c_stream2:
                        st.markdown("**Thermal Infrared Stream (IR)**")
                        ph_ir = st.empty()

                    st.markdown("#### 🎯 Live Multi-Modal Detections")
                    ph_det = st.empty()
                    ph_telemetry = st.empty()

                    processed = 0
                    for f_idx in range(0, max_stream_frames, frame_step):
                        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                        cap_ir.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                        r_ret, f_rgb = cap_rgb.read()
                        i_ret, f_ir = cap_ir.read()
                        if not (r_ret and i_ret and f_rgb is not None and f_ir is not None):
                            break

                        raw_rgb = cv2.cvtColor(f_rgb, cv2.COLOR_BGR2RGB)
                        raw_ir = cv2.cvtColor(f_ir, cv2.COLOR_BGR2RGB)
                        if raw_ir.shape[:2] != raw_rgb.shape[:2]:
                            raw_ir = cv2.resize(raw_ir, (raw_rgb.shape[1], raw_rgb.shape[0]))

                        t0 = time.perf_counter()
                        eff_rgb = raw_rgb if rgb_active else None
                        eff_ir = raw_ir if ir_active else None
                        res = pipeline.run(eff_rgb, eff_ir, conf=conf_thresh, iou=iou_thresh, w_rgb=w_rgb, w_ir=w_ir, rgb_active=rgb_active, ir_active=ir_active)
                        t1 = time.perf_counter()
                        fps_inst = 1.0 / max(1e-4, (t1 - t0))

                        # Update stream frames
                        if rgb_active:
                            ph_rgb.image(raw_rgb, use_container_width=True)
                        else:
                            ph_rgb.warning("Visible sensor OFFLINE ⚠️")

                        if ir_active:
                            ph_ir.image(raw_ir, use_container_width=True)
                        else:
                            ph_ir.warning("Thermal sensor OFFLINE ⚠️")

                        if res.annotated_image is not None:
                            ph_det.image(res.annotated_image, use_container_width=True)

                        ph_telemetry.markdown(f"""
                        <div style="background-color:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:12px 18px;margin-top:10px;display:flex;justify-content:space-between;box-shadow:0 1px 3px rgba(0,0,0,0.05);font-size:13px;">
                            <div><b style="color:#64748b;">Frame:</b> <span style="font-weight:800;color:#0f172a;">{f_idx + 1} / {max_stream_frames}</span></div>
                            <div><b style="color:#64748b;">Timestamp:</b> <span style="font-weight:800;color:#0f172a;">{f_idx / vid_fps:.2f}s</span></div>
                            <div><b style="color:#64748b;">Detections:</b> <span style="font-weight:800;color:#1e40af;">{len(res.boxes)} targets</span></div>
                            <div><b style="color:#64748b;">Latency:</b> <span style="font-weight:800;color:#0f172a;">{res.latency_ms:.1f} ms</span></div>
                            <div><b style="color:#64748b;">Live Throughput:</b> <span style="font-weight:800;color:#15803d;">{fps_inst:.1f} FPS</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                        processed += 1
                        pbar.progress(min(1.0, (f_idx + 1) / max_stream_frames))

                    cap_rgb.release()
                    cap_ir.release()
                    stream_status.success(f"✅ Playback completed: Streamed {processed} synchronized frames.")
                    return

            elif play_mode == "🎬 Render Full Video Player (HTML5 MP4 with Controls)":
                st.markdown("""
                <div style="background-color:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #1e40af;padding:10px 14px;border-radius:6px;margin:8px 0;">
                    <b style="color:#1e40af;font-size:13px;">🎬 Native Video Player:</b>
                    <span style="color:#334155;font-size:13px;">Processes the video through the active pipeline and outputs a playable MP4 video with standard play, pause, seek, and download controls.</span>
                </div>
                """, unsafe_allow_html=True)

                c_ren1, c_ren2 = st.columns([3, 1])
                with c_ren1:
                    max_render_frames = st.slider("Frames to Render", 10, total_frames, min(total_frames, 60), key="render_frames_slider")
                with c_ren2:
                    st.write("")
                    st.write("")
                    start_render = st.button("🎬 Render Annotated Video", type="primary", use_container_width=True)

                out_vid_path = Path("temp_videos") / "annotated_detection_output.mp4"
                if start_render:
                    cap_rgb = cv2.VideoCapture(str(t_rgb_path))
                    cap_ir = cv2.VideoCapture(str(t_ir_path))

                    r_pbar = st.progress(0)
                    r_status = st.empty()
                    r_status.info("Rendering annotated detection video...")

                    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    r_ok, sample_frame = cap_rgb.read()
                    if r_ok and sample_frame is not None:
                        vh, vw = sample_frame.shape[:2]
                        Path("temp_videos").mkdir(exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(str(out_vid_path), fourcc, vid_fps, (vw, vh))

                        for fi in range(max_render_frames):
                            cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, fi)
                            cap_ir.set(cv2.CAP_PROP_POS_FRAMES, fi)
                            ret_r, f_r = cap_rgb.read()
                            ret_i, f_i = cap_ir.read()
                            if not (ret_r and ret_i and f_r is not None and f_i is not None):
                                break

                            rw_rgb = cv2.cvtColor(f_r, cv2.COLOR_BGR2RGB)
                            rw_ir = cv2.cvtColor(f_i, cv2.COLOR_BGR2RGB)
                            if rw_ir.shape[:2] != (vh, vw):
                                rw_ir = cv2.resize(rw_ir, (vw, vh))

                            eff_r = rw_rgb if rgb_active else None
                            eff_i = rw_ir if ir_active else None
                            res = pipeline.run(eff_r, eff_i, conf=conf_thresh, iou=iou_thresh, w_rgb=w_rgb, w_ir=w_ir, rgb_active=rgb_active, ir_active=ir_active)

                            if res.annotated_image is not None:
                                out_bgr = cv2.cvtColor(res.annotated_image, cv2.COLOR_RGB2BGR)
                                writer.write(out_bgr)
                            else:
                                writer.write(f_r)

                            r_pbar.progress((fi + 1) / max_render_frames)

                        writer.release()
                        cap_rgb.release()
                        cap_ir.release()
                        r_status.success("✅ Video rendering complete!")

                if out_vid_path.exists():
                    st.markdown("#### 📺 Playable Multi-Modal Detection Video")
                    st.video(str(out_vid_path))
                    with open(str(out_vid_path), "rb") as f_down:
                        st.download_button(
                            label="⬇️ Download Annotated Detection Video (MP4)",
                            data=f_down.read(),
                            file_name="annotated_detection_output.mp4",
                            mime="video/mp4"
                        )
                    return

            else:
                # Synchronized Frame Scrubber Controls
                st.markdown("###### ⏱️ Synchronized Frame Scrubber")
                sc1, sc2 = st.columns([3, 1])
                with sc1:
                    frame_idx = st.slider(
                        "Video Frame Index",
                        min_value=0,
                        max_value=total_frames - 1,
                        value=0,
                        key="vid_frame_slider",
                        help="Slide to analyze any synchronized timestamp across both sensors"
                    )
                with sc2:
                    timestamp = frame_idx / vid_fps
                    st.metric("Timestamp", f"{timestamp:.2f}s", f"Frame {frame_idx + 1}/{total_frames}")

                cap_rgb = cv2.VideoCapture(str(t_rgb_path))
                cap_ir = cv2.VideoCapture(str(t_ir_path))
                cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                cap_ir.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ret_rgb, f_rgb = cap_rgb.read()
                ret_ir, f_ir = cap_ir.read()

                cap_rgb.release()
                cap_ir.release()

                if ret_rgb and ret_ir and f_rgb is not None and f_ir is not None:
                    raw_rgb = cv2.cvtColor(f_rgb, cv2.COLOR_BGR2RGB)
                    raw_ir = cv2.cvtColor(f_ir, cv2.COLOR_BGR2RGB)

                    if raw_ir.shape[:2] != raw_rgb.shape[:2]:
                        raw_ir = cv2.resize(raw_ir, (raw_rgb.shape[1], raw_rgb.shape[0]))

                    rgb_frame, ir_frame, source_error = load_and_align_pair(raw_rgb, raw_ir)
                else:
                    source_error = f"Could not decode synchronized frames at index {frame_idx}."

    elif input_source == "Live Camera (Webcam)":
        st.info("Webcam mode: Captures single frame from host camera and applies thermal replication.")
        pic = st.camera_input("Capture Frame")
        if pic is not None:
            nparr = np.frombuffer(pic.read(), np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is not None:
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                ir_frame = cv2.cvtColor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

    if source_error:
        st.error(f"⚠️ {source_error}")
        return

    if rgb_frame is None and ir_frame is None:
        st.warning("Please select or upload input sensor data to begin live detection.")
        return

    # --------------------------------------------------------------------------
    # PIPELINE EXECUTION
    # Pipeline already instantiated early
    pass

    # Apply sensor dropout simulation to inputs
    effective_rgb = rgb_frame if rgb_active else None
    effective_ir = ir_frame if ir_active else None

    # Run inference
    with st.spinner(f"Running inference ({pipeline.name})..."):
        try:
            result = pipeline.run(
                rgb_img=effective_rgb,
                ir_img=effective_ir,
                conf=conf_thresh,
                iou=iou_thresh,
                w_rgb=w_rgb,
                w_ir=w_ir,
                rgb_active=rgb_active,
                ir_active=ir_active
            )
        except Exception as e:
            st.error(f"Pipeline execution error: {e}")
            return

    # --------------------------------------------------------------------------
    # TOP PERFORMANCE METRIC BAR
    # --------------------------------------------------------------------------
    render_metric_row(
        fps=result.fps,
        latency_ms=result.latency_ms,
        num_objs=result.total_detections,
        gpu_mem=dev_info.get("display", "N/A"),
        active_model=result.model_name
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SENSOR STATUS & DYNAMIC ARCHITECTURE DIAGRAM
    # --------------------------------------------------------------------------
    st.markdown("#### ⚡ Active Perception Pipeline Architecture")
    render_pipeline_diagram(active_model_key, rgb_active, ir_active, w_rgb, w_ir)

    # --------------------------------------------------------------------------
    # MAIN LIVE DETECTION PANELS
    # --------------------------------------------------------------------------
    st.markdown("#### 🖥️ Multi-Modal Sensor Streams & Detection Output")

    # Inputs Row
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        st.markdown(f"**Visible Optical Input** " + ("`[ONLINE]`" if rgb_active else "`[OFFLINE ⚠️]`"))
        if rgb_active and rgb_frame is not None:
            st.image(rgb_frame, use_container_width=True)
        else:
            st.warning("Visible Optical sensor disabled / offline.")

    with col_in2:
        st.markdown(f"**Thermal Infrared Input** " + ("`[ONLINE]`" if ir_active else "`[OFFLINE ⚠️]`"))
        if ir_active and ir_frame is not None:
            st.image(ir_frame, use_container_width=True)
        else:
            st.warning("Thermal Infrared sensor disabled / offline.")

    # Intermediate Display (Fused image for TarDAL or Dual Detections for Stage 5)
    if active_model_key in ["stage3", "stage6"] and result.fused_image is not None:
        st.markdown("#### 🎨 TarDAL Generative Feature-Fused Image")
        st.caption("Synthesized by TarDAL Generator (296K parameters, Tanh remapping -> YCrCb color reconstruction)")
        st.image(result.fused_image, use_container_width=True)

    elif active_model_key == "stage5" and is_research:
        st.markdown("#### 🔀 Stage 5 Decision-Level Consensus Streams")
        c_rgb_det, c_ir_det = st.columns(2)
        with c_rgb_det:
            st.markdown(f"**RGB Stream Detections:** `{result.intermediate_data.get('rgb_detections', 0)} candidates`")
            if result.intermediate_data.get("annotated_rgb") is not None:
                st.image(result.intermediate_data["annotated_rgb"], use_container_width=True)
        with c_ir_det:
            st.markdown(f"**IR Stream Detections:** `{result.intermediate_data.get('ir_detections', 0)} candidates`")
            if result.intermediate_data.get("annotated_ir") is not None:
                st.image(result.intermediate_data["annotated_ir"], use_container_width=True)

    # Final Output Display
    st.markdown("#### 🎯 Final Detections (Multi-Modal Consensus)")
    if result.annotated_image is not None:
        st.image(result.annotated_image, use_container_width=True)

    # Class Counts breakdown
    if result.class_counts:
        st.markdown("##### Detected Class Frequencies")
        cols = st.columns(len(result.class_counts))
        for col, (cname, cnt) in zip(cols, result.class_counts.items()):
            col.metric(cname, cnt)

    # --------------------------------------------------------------------------
    # RESEARCH TELEMETRY & LATENCY DECOMPOSITION
    # --------------------------------------------------------------------------
    if is_research:
        st.markdown("---")
        st.markdown("### ⏱️ Latency & Systems Decomposition")
        render_model_card(active_model_key)

        st.markdown("##### Live Measured Timing Breakdown (This Frame)")
        t_cols = st.columns(len(result.timing))
        for col, (stage_name, val_ms) in zip(t_cols, result.timing.items()):
            label_clean = stage_name.replace("_ms", "").replace("_", " ").title()
            unit = "ms" if "fps" not in stage_name else "FPS"
            col.metric(label_clean, f"{val_ms:.2f} {unit}")

        # Live Benchmark Runner Section
        st.markdown("##### 🚀 Live Repeated-Pass Benchmark")
        num_bench_frames = st.slider("Benchmark Sample Size (Frames)", min_value=10, max_value=50, value=20)
        if st.button("Run Live Latency Benchmark"):
            run_live_benchmark(pipeline, effective_rgb, effective_ir, num_frames=num_bench_frames, conf=conf_thresh, iou=iou_thresh)
