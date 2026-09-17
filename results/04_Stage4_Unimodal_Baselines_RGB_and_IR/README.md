# Stage 4: Unimodal Direct-Input Baselines (Visible Optical & Thermal Infrared)

Establishes unimodal single-sensor control baselines to rigorously evaluate the necessity of multimodal fusion:
- **Direct Optical (RGB):**
  - Validation mAP@50: **76.40%**
  - Test mAP@50 (Academic): **29.33%** (Retention: **38.4%** 🚨)
  - Failure Mode: Severe blindness in nighttime scenes, dense shadows, and low-contrast optical backgrounds.
- **Direct Thermal (IR):**
  - Validation mAP@50: **72.00%**
  - Test mAP@50 (Academic): **28.38%** (Retention: **39.4%** 🚨)
  - Failure Mode: Blind to cold objects (`Lamp` mAP is only 1.87%), complete loss of color and texture.
- **Throughput:** ~9.5 ms / 106 FPS (high-speed unimodal processing).
