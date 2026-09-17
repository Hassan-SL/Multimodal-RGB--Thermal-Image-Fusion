# Multimodal Dataset Acquisition and Structuring

## Overview
The **MultiSensorFusion_RGB_IR** framework requires synchronized, spatially calibrated Visible (RGB) and Thermal Long-Wave Infrared (LWIR) image pairs.

### Primary Dataset: M3FD
- **Total Synchronized Pairs:** 4,200
- **Native Resolution:** $1024 \times 768$
- **Sensors:** Visible optical camera + Uncooled LWIR thermal sensor
- **Official Split Protocol:**
  - **Train:** 80% (3,360 pairs)
  - **Validation:** 10% (420 pairs)
  - **Test:** 10% (420 pairs)

---

## External Download Repositories
Since raw sensor captures exceed GitHub storage quotas (>4 GB), all data must be downloaded externally:

| Source | Access Type | Link |
| :--- | :--- | :--- |
| **Kaggle** | Public Direct Download / Kaggle API | [Kaggle M3FD Mirror](https://www.kaggle.com/datasets) |
| **Google Drive** | Primary TarDAL Authors Drive | [TarDAL Official GitHub](https://github.com/JinyuanLiu-CV/TarDAL) |
| **Baidu Netdisk** | High-speed East Asia Mirror | Refer to TarDAL README |

---

## Dataset Format Specification
Labels follow the standard **Ultralytics YOLO normalized format**:
```text
<class_id> <x_center> <y_center> <width> <height>
```
All coordinate values are normalized to $[0.0, 1.0]$ relative to image dimensions.
