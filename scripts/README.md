# End-to-End Pipeline Execution Guide 📋

This directory contains the sorted, modular scripts for executing every phase of the **MultiSensorFusion_RGB_IR** research pipeline—from initial environment verification to modern model training, late fusion ensembling, and unified benchmarking.

> **Path Resolution Guaranteed:** All scripts leverage [`path_config.py`](path_config.py) to dynamically locate project root, dataset folders, weights, and run outputs. There are **zero hardcoded defective paths**. The scripts run identically on Windows, Linux, and Google Colab.

---

## 🚦 Stage-by-Stage Workflow

### Stage 0: Environment & Dataset Setup (`scripts/00_setup/`)
Verifies hardware, packages, dataset formatting, and pipeline sanity:
1. `python scripts/00_setup/00_preflight_check.py` — Verifies PyTorch, CUDA, dependencies, and raw M3FD files.
2. `python scripts/00_setup/01_verify_and_link_dataset.py` — Validates paired RGB-IR file names and formats.
3. `python scripts/00_setup/02_convert_xml_to_yolo.py` — Converts M3FD XML bounding boxes to YOLO normalized coordinates.
4. `python scripts/00_setup/03_smoke_test.py` — Fast 5-sample dry run through the pipeline.

---

### Stage 1 & 2: Feature-Level Fusion via TarDAL (`scripts/01_feature_fusion_tardal/`)
Processes Visible-Infrared pairs into synthesized fused images in $YCrCb$ color space:
1. `python scripts/01_feature_fusion_tardal/01_run_tardal_feature_fusion.py` — Runs the TarDAL Generator on all pairs.
2. `python scripts/01_feature_fusion_tardal/02_prepare_fused_dataset.py` — Organizes fused images and labels into official Train (3,360), Val (420), and Test (420) splits.
3. `python scripts/01_feature_fusion_tardal/03_visualize_fused_samples.py` — Generates qualitative side-by-side RGB vs IR vs Fused comparisons.

---

### Stage 3: Baseline Detector Fine-Tuning (YOLOv5su) (`scripts/02_stage3_yolov5su_fusion/`)
Fine-tunes the baseline YOLOv5su architecture on TarDAL-fused imagery:
1. `python scripts/02_stage3_yolov5su_fusion/01_train_stage3_yolov5su.py` — Trains YOLOv5su backbone and head.
2. `python scripts/02_stage3_yolov5su_fusion/02_eval_stage3_benchmark.py` — Evaluates validation and test mAP@50.

---

### Stage 4: Direct Input vs Fusion Validation (`scripts/03_stage4_unimodal_baselines/`)
Establishes unimodal performance baselines to isolate the true contribution of multimodal fusion:
1. `python scripts/03_stage4_unimodal_baselines/01_train_stage4_unimodal_baselines.py` — Trains Visible-only and IR-only models.
2. `python scripts/03_stage4_unimodal_baselines/02_eval_stage4_comparative.py` — Directly compares RGB vs IR vs TarDAL fusion.
3. `python scripts/03_stage4_unimodal_baselines/03_visualize_stage4_comparison.py` — Plots comparative visual detection panels.

---

### Stage 5: Decision-Level Late Fusion (`scripts/04_stage5_late_fusion/`)
Combines predictions from independent unimodal heads in bounding-box space:
1. `python scripts/04_stage5_late_fusion/01_generate_raw_predictions.py` — Caches raw unimodal detections.
2. `python scripts/04_stage5_late_fusion/02_late_fusion_engine.py` — Merges detections using Weighted Boxes Fusion (WBF) and NMS.
3. `python scripts/04_stage5_late_fusion/03_late_fusion_diagnostics.py` — Runs modality weight sweeps ($w_{rgb}, w_{ir}$) and sensor dropout tests.
4. `python scripts/04_stage5_late_fusion/04_visualize_late_fusion.py` — Renders ensembled bounding boxes against individual modalities.

---

### Stage 6: Modern Detector SOTA Fine-Tuning (YOLO11s) (`scripts/05_stage6_modern_detector_yolo11s/`)
Replaces legacy YOLOv5su with state-of-the-art YOLO11s on the frozen TarDAL fusion:
1. `python scripts/05_stage6_modern_detector_yolo11s/01_train_stage6_yolo11s.py` — 50-epoch training with AdamW and patience 15.
2. `python scripts/05_stage6_modern_detector_yolo11s/02_eval_stage6_test.py` — Computes official COCO test metrics (53.27% mAP@50).
3. `python scripts/05_stage6_modern_detector_yolo11s/03_visualize_stage6_results.py` — Generates high-resolution qualitative detection figures.
4. `python scripts/05_stage6_modern_detector_yolo11s/04_compare_stage6_results.py` — Evaluates YOLO11s vs YOLOv5su accuracy, parameter efficiency, and FPS.

---

### Stage 7: Master Unified Synthesis (`scripts/06_master_synthesis/`)
Comprehensive comparative benchmarks across all research stages:
1. `python scripts/06_master_synthesis/01_master_unified_test_benchmark.py` — Unified evaluation table across all stages.
2. `python scripts/06_master_synthesis/02_eval_all_stages_decomposed.py` — Decomposed per-class metrics (People, Car, Bus, Motorcycle, Lamp, Truck).
3. `python scripts/06_master_synthesis/03_diagnose_val_test_gap.py` — Statistical distribution shift and domain gap analysis between Validation and Test splits.

---

### Live Demonstrator Deployment (`scripts/07_deployment/`)
Interactive Streamlit web application:
```bash
python scripts/07_deployment/run_demo.py
```
Or:
```bash
streamlit run interactive_demo/app.py
```
