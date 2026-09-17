"""
Live Performance Benchmark Runner.
Executes repeated inference passes on active hardware, computes P50/P95 latency percentiles,
and explicitly separates 'LIVE DEMO MEASUREMENT' from 'OFFICIAL RESEARCH BENCHMARK'.
"""

import time
import numpy as np
import streamlit as st
from utils.device import get_device_info, get_memory_info
from pipelines.base_pipeline import BasePipeline


def run_live_benchmark(pipeline: BasePipeline, rgb_img: np.ndarray, ir_img: np.ndarray, num_frames: int = 30, conf: float = 0.25, iou: float = 0.50):
    """Executes repeated inference to collect latency distribution statistics."""
    latencies = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Warm-up pass
    status_text.text("Warming up pipeline & GPU...")
    pipeline.run(rgb_img, ir_img, conf=conf, iou=iou)

    status_text.text(f"Benchmarking {num_frames} frames...")
    for i in range(num_frames):
        t0 = time.perf_counter()
        res = pipeline.run(rgb_img, ir_img, conf=conf, iou=iou)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        progress_bar.progress((i + 1) / num_frames)

    status_text.empty()
    progress_bar.empty()

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    st.markdown("""
    <div style="background-color:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #1e40af;padding:12px 16px;border-radius:6px;margin:12px 0;">
        <b style="color:#1e40af;font-weight:800;font-size:13px;">ℹ️ LIVE DEMO MEASUREMENT</b> 
        <span style="font-size:13px;color:#334155;font-weight:500;">(Reflects current host processor & browser overhead; separate from official Tesla T4 thesis benchmarks).</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Mean Latency", f"{mean_lat:.2f} ms")
    with c2:
        st.metric("Median (P50)", f"{median_lat:.2f} ms")
    with c3:
        st.metric("P95 Latency", f"{p95_lat:.2f} ms")
    with c4:
        st.metric("Throughput", f"{fps:.1f} FPS")

    mem = get_memory_info()
    st.caption(f"Host Device: {get_device_info()['display']} • Peak Allocated VRAM: {mem['display']}")
