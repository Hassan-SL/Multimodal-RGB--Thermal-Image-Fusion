# Stage 1: Zero-Shot Pretrained Baseline (TarDAL)

- **Architecture:** Unadapted TarDAL Generator + YOLOv5su
- **Validation mAP@50:** 16.97%
- **Test mAP@50 (Academic $\\tau=0.001$):** 12.40%
- **Test mAP@50 (Operational $\\tau=0.25$):** 9.80%
- **Latency / FPS (Tesla T4):** 121.20 ms / 8.2 FPS
- **Key Takeaway:** Proves that out-of-the-box pretrained generative fusion fails for downstream machine perception without fine-tuning (only 12.4% test mAP).
