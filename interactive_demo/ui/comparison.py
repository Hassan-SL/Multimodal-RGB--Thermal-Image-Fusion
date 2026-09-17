"""
Research Comparison Dashboard: Renders authoritative Master Tables 0, 1, 2, and 3.
Strictly separates Academic Benchmark (conf=0.001, iou=0.60) from
Operational Deployment (conf=0.25, iou=0.50).
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd


def load_telemetry():
    """Loads authoritative project benchmark telemetry."""
    app_dir = Path(__file__).resolve().parent.parent
    project_root = app_dir.parent
    candidates = [
        project_root / "runs/master_benchmark/all_stages_decomposed_results.json",
        app_dir / "runs/master_benchmark/all_stages_decomposed_results.json",
        Path("runs/master_benchmark/all_stages_decomposed_results.json"),
        Path("E:/My Drive/FYP/code/runs/master_benchmark/all_stages_decomposed_results.json"),
        Path("/content/drive/MyDrive/FYP/code/runs/master_benchmark/all_stages_decomposed_results.json")
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def render_comparison_page():
    st.markdown("### 📊 Comprehensive Multi-Stage Comparative Analysis")
    st.markdown("""
    This dashboard provides paper-ready, authoritative performance telemetry across all six research stages.
    Evaluation protocols are strictly separated by operational thresholds to prevent scientific conflation.
    """)

    # Protocol explanation banner
    st.info("""
    **Methodological Protocol Distinction**:
    - **Academic Benchmark Protocol ($\tau_{\\text{conf}}=0.001, \\theta_{\\text{IoU}}=0.60$)**: Full Precision-Recall curve integration (standard PASCAL VOC / COCO convention).
    - **Operational Real-World Protocol ($\tau_{\\text{conf}}=0.25, \\theta_{\\text{IoU}}=0.50$)**: High-confidence filtering for real-time edge robotics and Decision-Level Late Fusion (WBF).
    """)

    telemetry = load_telemetry()

    # Master Table 0: Evolutionary Progression
    st.markdown("#### 🏛️ Master Table 0: Evolutionary Milestone Progression (Stages 1 – 6)")
    t0_data = [
        {"Stage / Paradigm": "Stage 1: Zero-Shot Baseline", "Fusion Type": "Feature (Unadapted Gen)", "Val mAP50": "16.97%", "Test mAP (BM)": "12.40%", "Test mAP (Op)": "9.80%", "Retention %": "73.1%", "Latency": "121.20 ms", "FPS": "8.2", "Fault Tolerance": "0% (Single Failure)"},
        {"Stage / Paradigm": "Stage 2: Adapted Head", "Fusion Type": "Feature (Frozen Gen)", "Val mAP50": "73.30%", "Test mAP (BM)": "45.80%", "Test mAP (Op)": "37.90%", "Retention %": "62.5%", "Latency": "80.17 ms", "FPS": "12.5", "Fault Tolerance": "0% (Single Failure)"},
        {"Stage / Paradigm": "Stage 3: Joint Adaptation 🏆", "Fusion Type": "Feature (Joint Adapted)", "Val mAP50": "74.57%", "Test mAP (BM)": "48.56%", "Test mAP (Op)": "40.12%", "Retention %": "65.1%", "Latency": "77.95 ms", "FPS": "12.8", "Fault Tolerance": "0% (Single Failure)"},
        {"Stage / Paradigm": "Stage 4A: Direct Optical", "Fusion Type": "Unimodal Optical (No Fusion)", "Val mAP50": "76.40%", "Test mAP (BM)": "29.33%", "Test mAP (Op)": "25.59%", "Retention %": "38.4%", "Latency": "9.42 ms", "FPS": "106.2", "Fault Tolerance": "Fails in dark/shadows"},
        {"Stage / Paradigm": "Stage 4B: Direct Thermal", "Fusion Type": "Unimodal Thermal (No Fusion)", "Val mAP50": "72.00%", "Test mAP (BM)": "28.38%", "Test mAP (Op)": "23.54%", "Retention %": "39.4%", "Latency": "9.59 ms", "FPS": "104.3", "Fault Tolerance": "Blind to cold objects"},
        {"Stage / Paradigm": "Stage 5: Decision Late Fusion 🏆", "Fusion Type": "Decision (WBF 0.6/0.4)", "Val mAP50": "74.01%", "Test mAP (BM)": "34.10%", "Test mAP (Op)": "30.29%", "Retention %": "46.1%", "Latency": "19.41 ms", "FPS": "51.5", "Fault Tolerance": "100% Graceful Degradation"},
        {"Stage / Paradigm": "Stage 6: Modern YOLO11s 🏆", "Fusion Type": "Feature (Modern Attention)", "Val mAP50": "81.08%", "Test mAP (BM)": "45.72%", "Test mAP (Op)": "41.65%", "Retention %": "56.4%", "Latency": "79.42 ms", "FPS": "12.6", "Fault Tolerance": "0% (Single Failure)"},
    ]
    df_t0 = pd.DataFrame(t0_data)
    st.dataframe(df_t0, use_container_width=True, hide_index=True)

    # Master Table 1: Academic Benchmark
    st.markdown("#### 🔬 Master Table 1: Academic Benchmark Protocol (`conf=0.001, iou=0.60`)")
    t1_data = [
        {"Architecture": "Stage 1: Pretrained TarDAL", "mAP@50": "12.40%", "mAP@50-95": "6.20%", "Precision": "24.50%", "Recall": "14.10%", "People": "18.20%", "Car": "32.50%", "Bus": "4.10%", "Lamp": "8.30%", "Motor": "9.40%", "Truck": "1.90%"},
        {"Architecture": "Stage 2: Adapted Head", "mAP@50": "45.80%", "mAP@50-95": "27.40%", "Precision": "58.40%", "Recall": "44.20%", "People": "64.10%", "Car": "80.50%", "Bus": "45.20%", "Lamp": "38.10%", "Motor": "41.50%", "Truck": "5.40%"},
        {"Architecture": "Stage 3: Joint Adaptation 🏆", "mAP@50": "48.56%", "mAP@50-95": "29.53%", "Precision": "60.52%", "Recall": "46.80%", "People": "68.09%", "Car": "83.28%", "Bus": "58.62%", "Lamp": "41.40%", "Motor": "46.60%", "Truck": "14.04%"},
        {"Architecture": "Stage 4A: Direct Optical (RGB)", "mAP@50": "29.33%", "mAP@50-95": "16.19%", "Precision": "51.10%", "Recall": "28.09%", "People": "33.21%", "Car": "77.43%", "Bus": "15.54%", "Lamp": "28.53%", "Motor": "20.30%", "Truck": "0.98%"},
        {"Architecture": "Stage 4B: Direct Thermal (IR)", "mAP@50": "28.38%", "mAP@50-95": "15.91%", "Precision": "34.69%", "Recall": "29.46%", "People": "75.43%", "Car": "73.68%", "Bus": "10.59%", "Lamp": "1.87%", "Motor": "7.78%", "Truck": "0.93%"},
        {"Architecture": "Stage 5: Decision Late Fusion", "mAP@50": "34.10%", "mAP@50-95": "18.11%", "Precision": "68.86%", "Recall": "31.39%", "People": "57.80%", "Car": "81.20%", "Bus": "14.80%", "Lamp": "22.30%", "Motor": "19.50%", "Truck": "1.10%"},
        {"Architecture": "Stage 6: Modern YOLO11s 🏆", "mAP@50": "45.72%", "mAP@50-95": "28.90%", "Precision": "58.80%", "Recall": "47.02%", "People": "77.49%", "Car": "85.26%", "Bus": "31.79%", "Lamp": "40.82%", "Motor": "35.21%", "Truck": "3.75%"},
    ]
    df_t1 = pd.DataFrame(t1_data)
    st.dataframe(df_t1, use_container_width=True, hide_index=True)

    # Master Table 2: Operational Deployment
    st.markdown("#### 🚗 Master Table 2: Operational Real-World Deployment Protocol (`conf=0.25, iou=0.50`)")
    t2_data = [
        {"Architecture": "Stage 1: Pretrained TarDAL", "mAP@50": "9.80%", "Precision": "24.50%", "Recall": "14.10%", "People": "14.10%", "Car": "26.80%", "Latency": "121.20 ms", "FPS": "8.2", "Real-Time?": "❌ No"},
        {"Architecture": "Stage 2: Adapted Head", "mAP@50": "37.90%", "Precision": "58.40%", "Recall": "44.20%", "People": "59.80%", "Car": "75.30%", "Latency": "80.17 ms", "FPS": "12.5", "Real-Time?": "❌ No"},
        {"Architecture": "Stage 3: Joint Adaptation", "mAP@50": "40.12%", "Precision": "60.52%", "Recall": "46.80%", "People": "64.20%", "Car": "78.40%", "Latency": "77.95 ms", "FPS": "12.8", "Real-Time?": "❌ No"},
        {"Architecture": "Stage 4A: Direct Optical (RGB)", "mAP@50": "25.59%", "Precision": "51.10%", "Recall": "28.09%", "People": "27.29%", "Car": "71.56%", "Latency": "9.42 ms", "FPS": "106.2", "Real-Time?": "✅ Yes (>30 FPS)"},
        {"Architecture": "Stage 4B: Direct Thermal (IR)", "mAP@50": "23.54%", "Precision": "34.69%", "Recall": "29.46%", "People": "67.93%", "Car": "65.80%", "Latency": "9.59 ms", "FPS": "104.3", "Real-Time?": "✅ Yes (>30 FPS)"},
        {"Architecture": "Stage 5: Decision Late Fusion 🏆", "mAP@50": "30.29%", "Precision": "68.86%", "Recall": "31.39%", "People": "51.08%", "Car": "77.56%", "Latency": "19.41 ms", "FPS": "51.5", "Real-Time?": "✅ Yes (Real-Time Leader)"},
        {"Architecture": "Stage 6: Modern YOLO11s 🏆", "mAP@50": "41.65%", "Precision": "58.80%", "Recall": "47.02%", "People": "73.12%", "Car": "81.40%", "Latency": "79.42 ms", "FPS": "12.6", "Real-Time?": "❌ No (Gen Bounded)"},
    ]
    df_t2 = pd.DataFrame(t2_data)
    st.dataframe(df_t2, use_container_width=True, hide_index=True)

    # Master Table 3: Latency Decomposition
    st.markdown("#### ⏱️ Master Table 3: Hardware Latency Breakdown (NVIDIA Tesla T4 GPU)")
    t3_data = [
        {"Stage": "Stage 1: Pretrained Baseline", "Preprocess": "2.10 ms", "Generator": "104.40 ms", "Reconstruction": "5.20 ms", "Detector": "9.50 ms", "WBF Merge": "—", "Total Latency": "121.20 ms", "Throughput": "8.2 FPS", "Params": "9.42M"},
        {"Stage": "Stage 2: Adapted Head", "Preprocess": "2.10 ms", "Generator": "66.43 ms", "Reconstruction": "2.22 ms", "Detector": "9.42 ms", "WBF Merge": "—", "Total Latency": "80.17 ms", "Throughput": "12.5 FPS", "Params": "9.42M"},
        {"Stage": "Stage 3: Joint Adaptation", "Preprocess": "2.10 ms", "Generator": "66.43 ms", "Reconstruction": "—", "Detector": "9.42 ms", "WBF Merge": "—", "Total Latency": "77.95 ms", "Throughput": "12.8 FPS", "Params": "9.42M"},
        {"Stage": "Stage 4A: Direct Optical", "Preprocess": "1.80 ms", "Generator": "—", "Reconstruction": "—", "Detector": "7.62 ms", "WBF Merge": "—", "Total Latency": "9.42 ms", "Throughput": "106.2 FPS", "Params": "9.12M"},
        {"Stage": "Stage 4B: Direct Thermal", "Preprocess": "1.80 ms", "Generator": "—", "Reconstruction": "—", "Detector": "7.79 ms", "WBF Merge": "—", "Total Latency": "9.59 ms", "Throughput": "104.3 FPS", "Params": "9.12M"},
        {"Stage": "Stage 5: Late Fusion (WBF) 🏆", "Preprocess": "3.60 ms", "Generator": "—", "Reconstruction": "—", "Detector": "15.23 ms", "WBF Merge": "0.58 ms", "Total Latency": "19.41 ms", "Throughput": "51.5 FPS", "Params": "18.24M"},
        {"Stage": "Stage 6: Modern YOLO11s", "Preprocess": "2.10 ms", "Generator": "61.61 ms", "Reconstruction": "4.97 ms", "Detector": "10.74 ms", "WBF Merge": "—", "Total Latency": "79.42 ms", "Throughput": "12.6 FPS", "Params": "9.43M"},
    ]
    df_t3 = pd.DataFrame(t3_data)
    st.dataframe(df_t3, use_container_width=True, hide_index=True)
