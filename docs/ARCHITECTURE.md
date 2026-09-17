# MultiSensorFusion Architecture & Methodology

## 1. System Pipeline Overview
The system provides two complementary multimodal fusion paradigms:
1. **Feature-Level Fusion (Pixel/Feature Space)**: Uses a Target-Aware Dual Adversarial Network (TarDAL) to generate a synthesized image combining thermal emissivity and optical chrominance.
2. **Decision-Level Late Fusion (Bounding Box Space)**: Runs independent unimodal detectors on RGB and IR streams, ensembling predictions via confidence-weighted bounding box aggregation.

---

## 2. Feature-Level Fusion: TarDAL Generator
- **Color Decomposition:** Visible RGB image is decomposed into Luminance ($Y_{\text{vis}}$) and Chrominance ($Cr_{\text{vis}}, Cb_{\text{vis}}$) via $YCrCb$ transformation.
- **Thermal Mapping:** Infrared is extracted as thermal luminance $Y_{\text{ir}}$.
- **Generator Processing:** A dual-stream CNN fuses $Y_{\text{vis}}$ and $Y_{\text{ir}}$ while preserving high-contrast thermal signatures and background details.
- **Tanh Remapping:** The generator output in $[-1, 1]$ is normalized:
  $$Y_{\text{fused}} = \text{clamp}\left(\frac{Y + 1}{2}, 0, 1\right)$$
- **Color Recomposition:** $Y_{\text{fused}}$ is combined with original $Cr_{\text{vis}}$ and $Cb_{\text{vis}}$ to produce full-color fused RGB.

---

## 3. Decision-Level Fusion: Late Box Ensembling
In scenarios where sensor alignment is imperfect or illumination varies dynamically:
- Separate detectors infer on RGB and IR.
- Bounding boxes are filtered by confidence ($c \ge \tau_{\text{conf}}$).
- Overlapping predictions across sensors are resolved using Weighted Boxes Fusion (WBF) or Class-Aware NMS, dynamically scaling box weights $(w_{\text{rgb}}, w_{\text{ir}})$.
