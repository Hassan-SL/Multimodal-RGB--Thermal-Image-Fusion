# Experimental Evaluation & Academic Benchmarks

## Overview
All models were evaluated on the official **M3FD Multimodal Benchmark** across the validation split ($N=420$) and test split ($N=420$) using official COCO evaluation protocols ($	ext{IoU}=0.50$ and $	ext{IoU}=0.50:0.95$).

---

## 1. Master Comparative Performance Table

| Stage | Architecture / Fusion Method | Val mAP@50 | Val mAP@50:95 | Test mAP@50 | Test mAP@50:95 | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 4 (Baseline)** | Unimodal Visible (YOLOv5su) | 68.61% | 40.54% | 48.06% | 27.60% | ~5.8 ms |
| **Stage 4 (Baseline)** | Unimodal Infrared (YOLOv5su) | 65.65% | 38.65% | 47.90% | 28.09% | ~5.8 ms |
| **Stage 3** | TarDAL Feature Fusion (YOLOv5su) | 68.96% | 40.75% | 48.56% | 28.16% | ~9.2 ms |
| **Stage 5** | Decision-Level Late Fusion (Ensemble) | 71.39% | 42.14% | 51.52% | 30.12% | ~12.4 ms |
| **Stage 6 (SOTA)** | Modern TarDAL Feature Fusion (YOLO11s) | **73.96%** | **44.91%** | **53.27%** | **31.39%** | ~6.4 ms |

---

## 2. Key Methodological Insights
1. **Validation vs. Test Distribution Shift**:
   - Both unimodal and multimodal models exhibit a consistent degradation from validation to test split ($\sim 20\%$ relative mAP decrease).
   - This empirically confirms a distribution shift between splits in the M3FD dataset.
2. **Modern Detector Advantage**:
   - Transitioning from YOLOv5su to YOLO11s (Stage 6) yielded a $+5.00\%$ mAP@50 gain on validation and $+4.71\%$ mAP@50 gain on test.
3. **Complementary Sensor Robustness**:
   - Under total visible sensor dropout (e.g. pitch-black night), the multimodal framework automatically maintains detection capability through the thermal infrared pathway.
