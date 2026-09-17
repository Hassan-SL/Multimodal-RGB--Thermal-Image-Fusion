# Multimodal RGB-IR Object Detection Interactive Research Demonstrator
### Final Year Project (FYP) — Live Demonstration & Deployment Interface

---

## 🌟 Executive Summary

This application represents the **final deployment and demonstration interface** of the Final Year Project on **Multimodal RGB-Thermal Object Detection**. 

It transforms the multi-stage research pipeline into a live, production-grade interactive system enabling academic supervisors, thesis examiners, and engineers to:
1. **Directly interact** with raw paired RGB and Infrared (IR) sensor data.
2. **Visually inspect** intermediate task-driven feature fusion (TarDAL luminance fusion and chrominance reconstruction).
3. **Compare all 5 detector architectures** side-by-side with real-time bounding boxes and confidence scores.
4. **Stress-test sensor resilience** via live RGB/IR sensor dropout simulation (verifying graceful degradation).
5. **Profile millisecond-level inference latency** across Preprocessing, TarDAL Generator, Detectors, and Late Fusion.
6. **Validate research conclusions** through interactive Master Benchmarking Tables and empirical trade-off analyzers.

---

## 🏗️ System Architecture

The application is engineered strictly around modular, decoupled components:

```
c:\Users\Admin\Desktop\fyp\code\
├── run_app.py                  # Root convenience Python launcher
├── run_demo.bat                # 1-click Windows batch launcher
├── checkpoints/                # FYP model weights (YOLO, TarDAL, Stage 3, Stage 5, Stage 6)
├── runs/                       # Benchmark logs & empirical evaluation telemetry
└── interactive_demo/           # 📦 Self-Contained Interactive Research Package
    ├── app.py                  # Main Streamlit application entrypoint and tab routing
    ├── README.md               # Application manual & defense demonstration script
    ├── config/
    │   └── app_config.yaml     # Centralized configuration (models, classes, thresholds)
    ├── utils/
    │   ├── device.py           # Hardware detection (CUDA, GPU VRAM profiling)
    │   ├── timing.py           # High-precision Stopwatches with hardware CUDA sync
    │   ├── visualization.py    # M3FD class color palette, bounding boxes, side-by-side panels
    │   └── logging_utils.py    # Structured logging
    ├── preprocessing/
    │   ├── tardal_preprocessor.py # YCrCb color space decomposition & luminance fusion math
    │   └── image_loader.py     # Multi-format image loader with dimensional alignment validation
    ├── models/
    │   ├── tardal_wrapper.py   # TarDAL Generator PyTorch wrapper (eval mode, frozen gradients)
    │   ├── yolo_wrapper.py     # Ultralytics detector wrapper with torch.inference_mode()
    │   ├── late_fusion.py      # Class-constrained Weighted Boxes Fusion (WBF)
    │   └── loader.py           # Thread-safe cached ModelManager with multi-path resolution
    ├── pipelines/
    │   ├── base_pipeline.py    # Abstract Base Pipeline & DetectionResult dataclass
    │   ├── rgb_pipeline.py     # Mode A: Direct Optical (RGB -> YOLOv5su)
    │   ├── ir_pipeline.py      # Mode B: Direct Thermal (IR -> YOLOv5su)
    │   ├── stage3_pipeline.py  # Mode C: TarDAL + YOLOv5su Feature Fusion
    │   ├── stage6_pipeline.py  # Mode D: TarDAL + YOLO11s Modern Detector Fusion
    │   └── stage5_pipeline.py  # Mode E: Decision-Level Late Fusion (Dual YOLO + WBF)
    ├── ui/
    │   ├── styles.py           # Clean dark research engineering theme (#0d1117)
    │   ├── components.py       # Header, metric cards, pipeline DAG diagrams, model specs
    │   ├── dashboard.py        # Primary live perception dashboard & sensor dropout controls
    │   ├── comparison.py       # Comprehensive Master Comparison Tables (Tables 0, 1, 2, 3)
    │   ├── recommendations.py  # Strategic deployment recommendations & 3 Decoupled Conclusions
    │   ├── how_it_works.py     # 6-Stage FYP architectural evolution deep-dive
    │   └── benchmark_view.py   # Automated live latency benchmarking suite (P50/P95 profiles)
    ├── tests/
    │   └── test_interactive_app.py # Automated 9-test smoke test suite (100% pass)
    └── assets/
        └── sample_pairs/       # Curated M3FD paired test images (Day, Night, Adverse)
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- **Python**: 3.10 to 3.14 (Virtual environment configured in `.venv`)
- **PyTorch**: Compatible with CUDA or CPU
- **Ultralytics**: YOLOv8/YOLO11 engine
- **Streamlit**: Web application framework

### 2. Launching the Application
You can launch the demonstrator using any of the following convenient methods from the repository root:

**Method A: 1-Click Batch File (Windows)**
Double click `run_demo.bat` or run:
```powershell
.\run_demo.bat
```

**Method B: Root Python Launcher**
```powershell
python run_app.py
```

**Method C: Direct Streamlit CLI**
```powershell
.\.venv\Scripts\streamlit run interactive_demo/app.py
```

The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 🎯 The 5 Operational Modes

| Mode | Pipeline Architecture | Fusion Level | Primary Strength | Checkpoints Used |
| :--- | :--- | :--- | :--- | :--- |
| **Mode A** | RGB $\to$ YOLOv5su | Unimodal | Optimal chromatic resolution in daylight | `best.pt` |
| **Mode B** | IR $\to$ YOLOv5su | Unimodal | High thermal contrast in total darkness | `best.pt` |
| **Mode C** | TarDAL $\to$ YOLOv5su | Feature-Level (Joint) | Illumination-invariant structural fusion | `stage3_gen_best.pt` + `stage3_best.pt` |
| **Mode D** | TarDAL $\to$ YOLO11s | Feature-Level (Modern) | **Peak Target Precision** (Highest Val mAP) | `stage3_gen_best.pt` + `stage6_yolo11s_best.pt` |
| **Mode E** | Dual YOLOv5su $\to$ WBF | Decision-Level (Late) | **Peak OOD Generalization** & Sensor Fault Tolerance | `yolov5su_rgb_best.pt` + `yolov5su_ir_best.pt` |

---

## ⚡ Live Sensor Failure & Dropout Simulation

In real-world autonomous driving, sensors degrade or fail:
- **RGB camera failures**: Headlight glare, lens occlusion, heavy fog, total nighttime darkness.
- **Thermal camera failures**: Sensor saturation, thermal crossover, hardware disconnection.

### Testing Sensor Dropout:
1. Select **Mode E: Decision-Level Late Fusion (Stage 5)** in the sidebar.
2. In the **Sensor Dropout Simulation** control panel:
   - Toggle **RGB Sensor Dropout (Simulate Darkness/Occlusion)**: The RGB detector is disabled; late fusion falls back gracefully to the IR detector.
   - Toggle **IR Sensor Dropout (Simulate Thermal Blindness)**: The IR detector is disabled; late fusion falls back gracefully to the RGB detector.
   - Observe the live latency profiling, detection counts, and the **Sensor Status Alert** reflecting true fail-safe degradation.

---

## 🔬 Protocol Stratification

The application strictly distinguishes between two evaluation protocols:

1. **Academic Research Benchmark** (`conf = 0.001`, `iou = 0.60`):
   - Computes standard COCO/PASCAL VOC PR curves and maximum theoretical recall.
   - Used for the official thesis results reported in Master Tables 1 and 2.
2. **Operational Deployment Threshold** (`conf = 0.25`, `iou = 0.50`):
   - Suppresses low-confidence background clutter for real-time robotic systems.
   - Used by default in the live perception dashboard to present actionable detections.

---

## 📊 The Three Decoupled Conclusions

A core contribution of this thesis is resolving the apparent contradiction between validation accuracy, test robustness, and real-time inference:

1. **Conclusion 1: Peak Target In-Domain Precision**
   - **Winner**: Stage 6 (TarDAL + YOLO11s)
   - **Validation mAP@50**: **82.3%**
   - *Rationale*: C3k2 building blocks and SPPF attention capture fine-grained features when test distributions match training conditions.
2. **Conclusion 2: Peak Robustness Under Distribution Shift**
   - **Winner**: Stage 5 (Decision-Level Late Fusion with WBF)
   - **Test mAP@50**: **50.2%** (vs. 48.6% for Stage 3 and 46.5% for Stage 6)
   - *Rationale*: Independent unimodal representations prevent pixel-level fusion artifacts from corrupting detector heads on out-of-distribution scenes.
3. **Conclusion 3: Real-Time Edge Autonomy**
   - **Winner**: Stage 1 Unimodal Baseline (51.2 FPS) / Stage 6 Feature Fusion (28.4 FPS on CUDA)
   - *Rationale*: While Stage 5 provides maximal accuracy on shifted test sets, it requires dual forward passes + WBF overhead (16.2 FPS on CUDA), making Stage 6 the balanced single-pass fusion architecture.

---

## ⏱️ 5-Minute FYP Defense Demonstration Script

When demonstrating this project to thesis examiners or supervisors, follow this structured script:

- **Minute 1: Introduction & The Sensor Complementarity Problem (Dashboard)**
  - Load sample `00000.png` (or `00001.png`).
  - Show raw RGB (washed out / dark) and raw IR (thermal signature evident).
  - Select **Mode A (RGB)** $\to$ show missed thermal targets.
  - Select **Mode B (IR)** $\to$ show lost texture and color boundaries.
- **Minute 2: Feature-Level Fusion via TarDAL (Dashboard & How It Works)**
  - Select **Mode C (Stage 3)** and **Mode D (Stage 6)**.
  - Show the intermediate **Fused Luminance Image** generated live by TarDAL.
  - Highlight how TarDAL preserves both thermal heat signatures and optical high-frequency edges.
- **Minute 3: Sensor Failure & Resilience (Dropout Simulation)**
  - Switch to **Mode E (Stage 5 Decision-Level Fusion)**.
  - Enable **RGB Sensor Dropout**. Show how detections survive via the IR stream with zero crashes.
  - Re-enable RGB and toggle **IR Dropout**. Show how optical perception maintains vehicle localization.
- **Minute 4: Empirical Benchmark Analysis (Comparison Dashboard Tab)**
  - Navigate to the **Research Comparison Dashboard** tab.
  - Walk the examiners through **Master Table 1 (Validation)** vs **Master Table 2 (Test)**.
  - Explain why Stage 6 wins on Validation (82.3%) while Stage 5 wins on Test (50.2%).
- **Minute 5: Deployment Recommendations & Q&A (Strategic Deployment Tab)**
  - Switch to the **Strategic Deployment Recommendations** tab.
  - Summarize the Three Decoupled Conclusions based on the target operational envelope.

---

## 🧪 Automated Smoke Testing

To verify end-to-end pipeline integrity, model weights, and preprocessors:

```powershell
# From project root:
.\.venv\Scripts\python.exe interactive_demo/tests/test_interactive_app.py

# Or from inside interactive_demo:
cd interactive_demo
..\.venv\Scripts\python.exe tests/test_interactive_app.py
```

**Test Suite Coverage**:
- `test_config_exists`: Verifies YAML syntax and model paths.
- `test_image_loader_synthetic`: Tests dimension alignment and type checks.
- `test_tardal_preprocessor_shape`: Verifies YCrCb math and reconstruction dimensions.
- `test_wbf_fusion_logic`: Validates Weighted Boxes Fusion coordinate and score fusion.
- `test_rgb_pipeline_runs`: End-to-end Mode A execution.
- `test_ir_pipeline_runs`: End-to-end Mode B execution.
- `test_stage3_pipeline_runs`: End-to-end Mode C execution (TarDAL + YOLOv5su).
- `test_stage6_pipeline_runs`: End-to-end Mode D execution (TarDAL + YOLO11s).
- `test_stage5_dropout_resilience`: Validates sensor failure handling in Mode E.

---
*Developed for the Final Year Project in Computer Engineering & Vision — 2026.*
