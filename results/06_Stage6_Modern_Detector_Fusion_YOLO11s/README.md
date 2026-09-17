# Stage 6: Modern Detector Fusion (TarDAL + YOLO11s) 🏆

- **Architecture:** TarDAL Generator + YOLO11s (featuring C3k2 and C2PSA attention mechanisms)
- **Validation mAP@50:** **81.08%** 🏆 (Highest validation score in project; 80.40% final checkpoint)
- **Test mAP@50 (Academic):** 45.72% (+16.39 pp over Direct RGB)
- **Test mAP@50 (Operational $\\tau=0.25$):** **41.65%** 🏆 (Project Record!)
- **Primary Category Project Records:**
  - **Pedestrian (`People`):** **77.49%** 🏆 (+9.40 pp gain over Stage 3!)
  - **Vehicle (`Car`):** **85.26%** 🏆 (+1.98 pp gain over Stage 3!)
  *(These two categories account for **91.8%** of all real-world test targets).*
- **Latency / FPS (Tesla T4):** 79.42 ms / 12.6 FPS
