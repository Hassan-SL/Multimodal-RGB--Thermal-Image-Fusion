# MultiSensorFusion_RGB_IR: Master Results & Artifacts Index 📁

This directory contains the organized results, evaluation curves, confusion matrices, benchmark charts, and qualitative detection figures from the Final Year Project.

Designed specifically for automated ingestion and review by **Prism OpenAI**, thesis supervisors, and academic defense examiners.

---

## 🏛️ Directory Structure & Stage Organization

```text
results/
├── 01_Stage1_Zero_Shot_Pretrained/
│   ├── stage1_summary_metrics.json
│   └── README.md
│
├── 02_Stage2_Adapted_Detection_Head/
│   ├── stage2_training_metrics_curves.png
│   ├── stage2_training_loss_and_metrics.csv
│   ├── stage2_confusion_matrix_normalized.png
│   ├── stage2_confusion_matrix_raw.png
│   ├── stage2_precision_recall_curve.png
│   ├── stage2_f1_confidence_curve.png
│   ├── stage2_precision_confidence_curve.png
│   ├── stage2_recall_confidence_curve.png
│   ├── stage2_val_predictions_batch0.jpg
│   ├── stage2_val_predictions_batch1.jpg
│   └── stage2_val_predictions_batch2.jpg
│
├── 03_Stage3_Task_Driven_Feature_Fusion_YOLOv5su/
│   ├── stage3_training_history.log
│   ├── stage3_val_fused_sample_*.png
│   ├── stage3_summary_metrics.json
│   └── README.md
│
├── 04_Stage4_Unimodal_Baselines_RGB_and_IR/
│   ├── optical_rgb/
│   │   ├── stage4_rgb_training_metrics_curves.png
│   │   ├── stage4_rgb_training_loss_and_metrics.csv
│   │   ├── stage4_rgb_confusion_matrix_normalized.png
│   │   ├── stage4_rgb_precision_recall_curve.png
│   │   └── stage4_rgb_val_predictions_*.jpg
│   └── thermal_ir/
│       ├── stage4_ir_training_metrics_curves.png
│       ├── stage4_ir_training_loss_and_metrics.csv
│       ├── stage4_ir_confusion_matrix_normalized.png
│       ├── stage4_ir_precision_recall_curve.png
│       └── stage4_ir_val_predictions_*.jpg
│
├── 05_Stage5_Decision_Level_Late_Fusion/
│   ├── stage5_optimal_modality_weights.json
│   ├── stage5_test_benchmark_metrics.json
│   └── qualitative_comparisons/
│       ├── case1_ir_rescues_rgb_darkness_ex*.jpg
│       ├── case2_rgb_rescues_ir_cold_target_ex*.jpg
│       ├── case3_consensus_boost_high_confidence_ex*.jpg
│       ├── case4_failure_modes_both_sensors_blind_ex*.jpg
│       └── case5_general_scene_comparison_ex*.jpg
│
├── 06_Stage6_Modern_Detector_Fusion_YOLO11s/
│   ├── stage6_yolo11s_training_curves.png
│   ├── stage6_yolo11s_training_history.csv
│   ├── stage6_confusion_matrix_normalized.png
│   ├── stage6_precision_recall_curve.png
│   ├── stage6_f1_confidence_curve.png
│   ├── stage6_master_comparison_plot.png
│   ├── stage6_test_benchmark_metrics.json
│   ├── stage6_final_comparison_report.md
│   ├── stage6_final_comparison_report.json
│   └── qualitative_comparisons/
│       ├── case1_yolo11s_catches_missed_pedestrians_ex*.jpg
│       ├── case2_yolov5su_catches_missed_low_freq_ex*.jpg
│       ├── case3_tighter_localization_bounding_box_ex*.jpg
│       ├── case4_false_positive_suppression_ex*.jpg
│       └── case5_small_distant_targets_ex*.jpg
│
└── 07_Master_Comparative_Synthesis/
    ├── chart1_per_class_accuracy_histogram.png
    ├── chart2_pareto_accuracy_vs_latency.png
    ├── chart3_multimodal_radar_chart.png
    ├── chart4_latency_decomposition_stacked.png
    ├── chart5_val_to_test_generalization_drop.png
    ├── master_unified_benchmark_results.json
    └── all_stages_decomposed_telemetry.json
```

---

## 📊 Summary of Master Performance Telemetry

| Stage / Paradigm | Fusion Type | Val mAP@50 | Test mAP@50 (Academic) | Test mAP@50 (Operational) | Retention % | Latency | FPS | Key Strength |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1: Zero-Shot Baseline** | Feature (Unadapted) | 16.97% | 12.40% | 9.80% | 73.1% | 121.20 ms | 8.2 | Off-the-shelf control baseline |
| **Stage 2: Adapted Head** | Feature (Frozen TarDAL) | 73.30% | 45.80% | 37.90% | 62.5% | 80.17 ms | 12.5 | Fast detector adaptation |
| **Stage 3: Joint Adaptation** 🏆 | Feature (Joint Adapted) | 74.57% | **48.56%** 🏆 | 40.12% | **65.1%** 🏆 | 77.95 ms | 12.8 | **Highest 6-class unweighted test mAP** |
| **Stage 4A: Direct Optical** | Unimodal Optical | 76.40% | 29.33% | 25.59% | 38.4% | **9.42 ms** | **106.2** | Optical texture (fails at night) |
| **Stage 4B: Direct Thermal** | Unimodal Thermal | 72.00% | 28.38% | 23.54% | 39.4% | **9.59 ms** | 104.3 | Heat signatures (blind to cold objects) |
| **Stage 5: Decision Late Fusion** 🏆 | Decision (WBF) | 74.01% | 34.10% | 30.29% | 46.1% | **19.41 ms** | **51.5** 🏆 | **Real-time + 100% Graceful Degradation** |
| **Stage 6: Modern YOLO11s** 🏆 | Feature (TarDAL + YOLO11s) | **81.08%** 🏆 | 45.72% | **41.65%** 🏆 | 56.4% | 79.42 ms | 12.6 | **Peak Pedestrian (77.5%) & Car (85.3%)** |
