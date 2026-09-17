# Pretrained Model Checkpoints

This directory houses the weights for the TarDAL fusion generator and fine-tuned YOLO detectors.

> **Note:** Binary checkpoint files (`*.pt`, `*.pth`) are excluded from Git via `.gitignore`. You can download pre-trained checkpoints from the links below or train your own using `scripts/`.

---

## Model Inventory & Download Links

| Model Stage | Architecture | Target | File Name | Size | Download |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TarDAL Generator** | Dual-adversarial CNN | YCrCb Feature Fusion | `tardal_generator.pth` | ~2.5 MB | [GitHub Releases / Drive Link](#) |
| **Stage 3 Detector** | YOLOv5su | Fused-domain Detection | `yolov5su_stage3_best.pt` | ~18.5 MB | [GitHub Releases / Drive Link](#) |
| **Stage 4 Visible** | YOLOv5su | Unimodal RGB Baseline | `yolov5su_rgb_best.pt` | ~18.5 MB | [GitHub Releases / Drive Link](#) |
| **Stage 4 Infrared** | YOLOv5su | Unimodal IR Baseline | `yolov5su_ir_best.pt` | ~18.5 MB | [GitHub Releases / Drive Link](#) |
| **Stage 6 Detector** | YOLO11s | Modern Architecture | `yolo11s_stage6_best.pt` | ~19.0 MB | [GitHub Releases / Drive Link](#) |

---

## Where to Place Weights
Place all downloaded `.pt` and `.pth` files directly into this directory:
```text
MultiSensorFusion_RGB_IR/
└── weights/
    ├── README.md
    ├── tardal_generator.pth
    ├── yolov5su_stage3_best.pt
    ├── yolov5su_rgb_best.pt
    ├── yolov5su_ir_best.pt
    └── yolo11s_stage6_best.pt
```
