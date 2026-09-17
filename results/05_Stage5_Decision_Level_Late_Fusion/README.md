# Stage 5: Decision-Level Late Fusion (Weighted Boxes Fusion) 🏆

- **Architecture:** Independent Dual YOLOv5su streams + Weighted Boxes Fusion (WBF)
- **Optimal Modality Weights:** $w_{\\text{RGB}} = 0.60$, $w_{\\text{IR}} = 0.40$ (IoU match: 0.50)
- **Validation mAP@50:** 74.01%
- **Test mAP@50 (Academic):** 34.10%
- **Test mAP@50 (Operational $\\tau=0.25$):** **30.29%** (Highest operational score among unimodal/WBF)
- **Total Latency / FPS (Tesla T4):** **19.41 ms / 51.5 FPS** 🏆 (Real-Time Leader, >30 FPS)
- **Fault Tolerance:** **100% Graceful Degradation** — If visible camera is obscured or fails, system seamlessly operates on thermal with zero downtime.
