# Dataset Setup & External Download Guide

> **Important Note:** In accordance with GitHub repository size guidelines and dataset licensing, raw image files and annotation labels are **not** tracked in this repository.

---

## 1. Supported Benchmark: M3FD
This project evaluates multimodal object detection using the **M3FD (Multi-Modal Multi-Task Fusion Detection)** dataset.
M3FD contains **4,200 synchronized, pixel-aligned RGB and Thermal Infrared (IR)** image pairs captured across diverse challenging conditions (day, night, overcast, smoke/fog).

### Classes Annotated (6 Classes):
- `0`: **People**
- `1`: **Car**
- `2`: **Bus**
- `3`: **Motorcycle**
- `4`: **Lamp**
- `5`: **Truck**

---

## 2. External Download Links

You can obtain the dataset from any of the following verified external mirrors:

1. **Kaggle Mirror (Recommended - Direct Download & API access)**:
   - [Kaggle M3FD Dataset Link](https://www.kaggle.com/datasets/jinyuanliu/m3fd-dataset) *(or search `M3FD Dataset` on Kaggle)*
   - CLI Download:
     ```bash
     kaggle datasets download -d jinyuanliu/m3fd-dataset
     unzip m3fd-dataset.zip -d data/
     ```

2. **Official Source (TarDAL Research Team)**:
   - [Official TarDAL Repository](https://github.com/JinyuanLiu-CV/TarDAL)
   - [Baidu Netdisk / Google Drive Links provided by TarDAL authors](https://github.com/JinyuanLiu-CV/TarDAL#dataset)

---

## 3. Expected Local Directory Structure

After downloading and extracting, arrange the dataset inside `data/` as follows:

```text
MultiSensorFusion_RGB_IR/
└── data/
    ├── README.md               <-- (This guide)
    ├── .gitignore              <-- (Protects against accidental commits)
    └── m3fd/
        ├── images/
        │   ├── train/          # 3,360 training images (or unimodal/fused pairs)
        │   ├── val/            # 420 validation images
        │   └── test/           # 420 test images
        └── labels/
            ├── train/          # YOLO-formatted .txt labels (class x_c y_c w h)
            ├── val/
            └── test/
```

For raw paired sensor inputs (before fusion):
```text
data/
└── raw_pairs/
    ├── visible/                # RGB optical images (.png / .jpg)
    ├── infrared/               # Thermal infrared images (.png / .jpg)
    └── labels/                 # Corresponding annotations
```

---

## 4. Verification Script

To verify that your dataset is correctly downloaded and formatted according to YOLO conventions:

```bash
python scripts/01_verify_dataset.py --data_dir data/m3fd
```
