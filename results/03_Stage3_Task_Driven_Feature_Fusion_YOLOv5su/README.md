# Stage 3: Task-Driven Feature Fusion (TarDAL + YOLOv5su) 🏆

- **Architecture:** Jointly-adapted TarDAL Generator + YOLOv5su
- **Validation mAP@50:** 74.57%
- **Test mAP@50 (Academic $\\tau=0.001$):** **48.56%** 🏆 (Project Record for unweighted 6-class mean)
- **Test mAP@50 (Operational $\\tau=0.25$):** 40.12%
- **Retention Rate:** **65.1%** 🏆 (Highest generalization retention of any model)
- **Latency / FPS (Tesla T4):** 77.95 ms / 12.8 FPS
- **Key Takeaway:** Adapting generative weights using downstream task gradients aligns thermal-visible representations specifically for machine vision, delivering **+19.24 pp gain over Direct RGB**.
