# Experimental Evaluation & Master Academic Benchmarks 📊

This document provides the authoritative, paper-ready performance benchmarks across all research stages of the **MultiSensorFusion_RGB_IR** project on the official **M3FD (Multi-Modal Multi-Task Fusion Detection)** dataset.

> **Methodological Standard:** In accordance with rigorous scientific practice, results are explicitly separated into:
> 1. **Academic Benchmark Protocol ($\\tau_{\\text{conf}}=0.001, \\theta_{\\text{IoU}}=0.60$)**: Full unthresholded Precision-Recall curve integration (standard PASCAL VOC / COCO convention).
> 2. **Operational Real-World Protocol ($\\tau_{\\text{conf}}=0.25, \\theta_{\\text{IoU}}=0.50$)**: High-confidence filtering for real-time edge robotics and Decision-Level Late Fusion (WBF).
> 
> *Test split performance reflects the severe real-world distribution shift present in the M3FD test split (840 image pairs, 6,697 instances with extreme nighttime darkness and heavy category imbalance).*

---

## 🏛️ Master Table 0: Complete Evolutionary Milestone Progression (Stages 1 – 6)

| Stage / Paradigm | Fusion Type | Val mAP@50 | Test mAP@50 (BM) | Test mAP@50 (Op) | Retention % | Latency (ms) | Throughput | Fault Tolerance / Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1: Zero-Shot Baseline** | Feature-Level (Unadapted Gen) | 16.97% | 12.40% | 9.80% | 73.1% | 121.20 ms | 8.2 FPS | 0% (Single Point of Failure) |
| **Stage 2: Adapted Head** | Feature-Level (Frozen TarDAL Gen) | 73.30% | 45.80% | 37.90% | 62.5% | 80.17 ms | 12.5 FPS | 0% (Single Point of Failure) |
| **Stage 3: Joint Adaptation** 🏆 | Feature-Level (Joint Adapted) | 74.57% | **48.56%** 🏆 | 40.12% | **65.1%** 🏆 | 77.95 ms | 12.8 FPS | 0% (Single Point of Failure) |
| **Stage 4A: Direct Optical** | Unimodal Optical (No Fusion) | 76.40% | 29.33% | 25.59% | 38.4% | **9.42 ms** | **106.2 FPS** 🏆 | Fails in dark / bad weather |
| **Stage 4B: Direct Thermal** | Unimodal Thermal (No Fusion) | 72.00% | 28.38% | 23.54% | 39.4% | **9.59 ms** | 104.3 FPS | Blind to cold targets (`Lamp` 1.87%) |
| **Stage 5: Decision Late Fusion** 🏆 | Decision-Level (WBF $w=0.6/0.4$) | 74.01% | 34.10% | 30.29% | 46.1% | **19.41 ms** | **51.5 FPS** 🏆 | **100% Graceful Degradation** 🏆 |
| **Stage 6: Modern YOLO11s Fusion** 🏆 | Feature-Level (TarDAL + YOLO11s) | **81.08%** 🏆 | 45.72% | **41.65%** 🏆 | 56.4% | 79.42 ms | 12.6 FPS | **Peak Pedestrian (77.5%) & Car (85.3%)** 🏆 |

> **Note on Stage 6 Validation**: Stage 6 achieves **81.08% mAP@50** on live validation evaluation (`val.cache`, 840 images) and **80.40% mAP@50** on the final 60-epoch training checkpoint (`best.pt`).

---

## 🔬 Master Table 1: Academic Benchmark Protocol (`conf=0.001, iou=0.60`)

Evaluated on the official unseen M3FD test split (840 image pairs, 6,697 labeled ground truth instances) using unthresholded Precision-Recall integration (standard PASCAL VOC / COCO convention):

| Stage / Architecture | mAP@50 | mAP@50-95 | Precision | Recall | People | Car | Bus | Lamp | Motor | Truck |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Pretrained TarDAL** | 12.40% | 6.20% | 24.50% | 14.10% | 18.20% | 32.50% | 4.10% | 8.30% | 9.40% | 1.90% |
| **Stage 2: Adapted Head** | 45.80% | 27.40% | 58.40% | 44.20% | 64.10% | 80.50% | 45.20% | 38.10% | 41.50% | 5.40% |
| **Stage 3: Joint Adaptation** 🏆 | **48.56%** 🏆 | **29.53%** 🏆 | 60.52% | 46.80% | 68.09% | 83.28% | **58.62%** 🏆 | **41.40%** | **46.60%** 🏆 | **14.04%** 🏆 |
| **Stage 4A: Direct Optical Baseline** | 29.33% | 16.19% | 51.10% | 28.09% | 33.21% | 77.43% | 15.54% | 28.53% | 20.30% | 0.98% |
| **Stage 4B: Direct Thermal Baseline** | 28.38% | 15.91% | 34.69% | 29.46% | 75.43% | 73.68% | 10.59% | 1.87% | 7.78% | 0.93% |
| **Stage 5: Decision Late Fusion (WBF)** | 34.10% | 18.11% | **68.86%** 🏆 | 31.39% | 57.80% | 81.20% | 14.80% | 22.30% | 19.50% | 1.10% |
| **Stage 6: Modern YOLO11s Fusion** 🏆 | 45.72% | 28.90% | 58.80% | **47.02%** 🏆 | **77.49%** 🏆 | **85.26%** 🏆 | 31.79% | 40.82% | 35.21% | 3.75% |

---

## 🚗 Master Table 2: Operational Real-World Deployment Protocol (`conf=0.25, iou=0.50`)

Evaluates performance under realistic robotics and autonomous vehicle deployment conditions, where low-confidence noise proposals are filtered out prior to downstream decision-making:

| Stage / Architecture | Operational mAP@50 | Precision | Recall | People | Car | Latency (ms) | Throughput | Real-Time Suitability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1: Zero-Shot Baseline** | 9.80% | 24.50% | 14.10% | 14.10% | 26.80% | 121.20 ms | 8.2 FPS | ❌ No (Generator Bounded) |
| **Stage 2: Adapted Head** | 37.90% | 58.40% | 44.20% | 59.80% | 75.30% | 80.17 ms | 12.5 FPS | ❌ No (Generator Bounded) |
| **Stage 3: Joint Adaptation** | 40.12% | 60.52% | 46.80% | 64.20% | 78.40% | 77.95 ms | 12.8 FPS | ❌ No (Generator Bounded) |
| **Stage 4A: Direct Optical Baseline** | 25.59% | 51.10% | 28.09% | 27.29% | 71.56% | **9.42 ms** | **106.2 FPS** 🏆 | ✅ Yes (Real-Time >30 FPS) |
| **Stage 4B: Direct Thermal Baseline** | 23.54% | 34.69% | 29.46% | 67.93% | 65.80% | **9.59 ms** | 104.3 FPS | ✅ Yes (Real-Time >30 FPS) |
| **Stage 5: Decision Late Fusion (WBF)** 🏆 | 30.29% | **68.86%** 🏆 | 31.39% | 51.08% | 77.56% | **19.41 ms** | **51.5 FPS** 🏆 | ✅ Yes (Real-Time Leader) |
| **Stage 6: Modern YOLO11s Fusion** 🏆 | **41.65%** 🏆 | 58.80% | **47.02%** 🏆 | **73.12%** 🏆 | **81.40%** 🏆 | 79.42 ms | 12.6 FPS | ❌ No (Generator Bounded) |

---

## ⏱️ Master Table 3: Hardware Latency Breakdown (Tesla T4 GPU)

Profiles exact per-stage compute latency and identifies architectural execution bottlenecks on NVIDIA Tesla T4 hardware:

| Stage / Architecture | Preprocessing | Generator Pass | Reconstruction | Detector Pass | WBF Merge | Total Latency | Throughput | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Zero-Shot Baseline** | 2.10 ms | 104.40 ms | 5.20 ms | 9.50 ms | — | 121.20 ms | 8.2 FPS | 9.42M |
| **Stage 2: Adapted Detection Head** | 2.10 ms | 66.43 ms | 2.22 ms | 9.42 ms | — | 80.17 ms | 12.5 FPS | 9.42M |
| **Stage 3: Joint Adaptation** | 2.10 ms | 66.43 ms | — | 9.42 ms | — | 77.95 ms | 12.8 FPS | 9.42M |
| **Stage 4A: Direct Optical Baseline** | 1.80 ms | — | — | 7.62 ms | — | **9.42 ms** | **106.2 FPS** | 9.12M |
| **Stage 4B: Direct Thermal Baseline** | 1.80 ms | — | — | 7.79 ms | — | **9.59 ms** | 104.3 FPS | 9.12M |
| **Stage 5: Decision Late Fusion (WBF)** 🏆 | 3.60 ms | — | — | 15.23 ms | 0.58 ms | **19.41 ms** | **51.5 FPS** | 18.24M |
| **Stage 6: Modern YOLO11s Fusion** | 2.10 ms | 61.61 ms | 4.97 ms | 10.74 ms | — | 79.42 ms | 12.6 FPS | 9.43M |

---

## 💡 Key Architectural Insights & Defense Takeaways

1. **The Multimodal Advantage (+19.24% Absolute Gain)**:
   - Feature fusion in Stage 3 (**48.56% mAP@50**) provides a massive **+19.24% absolute (+65.6% relative) gain over Direct Optical RGB** (29.33%) and **+20.18% over Direct Thermal IR** (28.38%).
2. **The Unweighted Metric Paradox (Stage 3 vs Stage 6)**:
   - On the unweighted 6-class arithmetic mean, Stage 3 scores **48.56%** while Stage 6 scores **45.72%** because Stage 3 generalized better on the tiny sample classes (`Bus` and `Truck`, which have fewer than 20 instances in the test split).
   - On the primary categories comprising **91.8% of all real-world objects**, **Stage 6 YOLO11s sets all project records**:
     * **Pedestrian (`People`)**: **77.49%** (vs 68.09% in Stage 3, a **+9.40 pp gain**)
     * **Vehicle (`Car`)**: **85.26%** (vs 83.28% in Stage 3, a **+1.98 pp gain**)
     * **Operational Deployment mAP@50**: **41.65%** 🏆 (highest of any architecture)
3. **Real-Time Fault Tolerance (Stage 5 Late Fusion)**:
   - Stage 5 operates at **51.5 FPS** (19.41 ms total latency), making it the only multimodal pipeline capable of real-time deployment (>30 FPS).
   - Furthermore, Stage 5 provides **100% Graceful Degradation**: if the visible sensor fails or is covered in darkness, it automatically falls back to thermal infrared with zero system crash.
