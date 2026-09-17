# Stage 2: Adapted Detection Head (Frozen TarDAL Generator)

- **Architecture:** Frozen TarDAL Generator (296K params) + Fine-tuned YOLOv5su Detection Head
- **Validation mAP@50:** 73.30%
- **Test mAP@50 (Academic $\\tau=0.001$):** 45.80%
- **Test mAP@50 (Operational $\\tau=0.25$):** 37.90%
- **Latency / FPS (Tesla T4):** 80.17 ms / 12.5 FPS
- **Key Takeaway:** Fine-tuning the detection head on fused images dramatically boosts test accuracy from 12.40% to 45.80% (+33.40 pp gain).
