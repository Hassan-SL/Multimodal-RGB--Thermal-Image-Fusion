# Pretrained Model Checkpoints 📦

This directory houses the weights for the TarDAL fusion generator and fine-tuned YOLO detectors.

> **Official Release:** Pre-trained model weights are hosted on [GitHub Release v1.0.0](https://github.com/Hassan-SL/Multimodal-RGB--Thermal-Image-Fusion/releases/tag/v1.0.0). Binary checkpoint files (`*.pt`, `*.pth`) are excluded from direct Git tracking to keep the repository lightweight.

---

## 📥 Model Inventory & Download Links

You can download individual model checkpoints directly from the official release assets:

| Model Stage | Architecture | Target Modality | File Name | Size | Direct Download Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TarDAL Generator** | Dual-adversarial CNN | YCrCb Feature Fusion | `tardal_generator.pth` | ~2.5 MB | [Download from Release v1.0.0](https://github.com/Hassan-SL/Multimodal-RGB--Thermal-Image-Fusion/releases/tag/v1.0.0) |
| **Stage 3 Detector** | YOLOv5su | Fused-domain Detection | `yolov5su_stage3_best.pt` | ~18.5 MB | [Download from Release v1.0.0](https://github.com/Hassan-SL/Multimodal-RGB--Thermal-Image-Fusion/releases/tag/v1.0.0) |
| **Stage 4 Visible** | YOLOv5su | Unimodal RGB Baseline | `yolov5su_rgb_best.pt` | ~18.5 MB | [Download from Release v1.0.0](https://github.com/Hassan-SL/Multimodal-RGB--Thermal-Image-Fusion/releases/tag/v1.0.0) |
| **Stage 4 Infrared** | YOLOv5su | Unimodal IR Baseline | `yolov5su_ir_best.pt` | ~18.5 MB | [Download from Release v1.0.0](https://github.com/Hassan-SL/Multimodal-RGB--Thermal-Image-Fusion/releases/tag/v1.0.0) |
| **Stage 6 Detector** | YOLO11s | Modern Architecture | `yolo11s_stage6_best.pt` | ~19.0 MB | [Download from Release v1.0.0](https://github.com/Hassan-SL/Multimodal-RGB--Thermal-Image-Fusion/releases/tag/v1.0.0) |

> 🔗 **All-in-One Release Page:** You can view and download all binary assets in one place at:  
> 👉 **[GitHub Release v1.0.0 Assets](https://github.com/Hassan-SL/Multimodal-RGB--Thermal-Image-Fusion/releases/tag/v1.0.0)**

---

## 📂 Where to Place Downloaded Weights

Place all downloaded `.pt` and `.pth` files directly into this directory:

```text
MultiSensorFusion_RGB_IR/
└── weights/
    ├── README.md
    ├── .gitignore
    ├── tardal_generator.pth
    ├── yolov5su_stage3_best.pt
    ├── yolov5su_rgb_best.pt
    ├── yolov5su_ir_best.pt
    └── yolo11s_stage6_best.pt
```

*(Note: The dynamic path resolver in `scripts/path_config.py` will automatically detect them once placed here.)*
