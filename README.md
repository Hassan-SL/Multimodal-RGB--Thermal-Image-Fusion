# MultiSensorFusion_RGB_IR 🚀
### Multimodal Target-Aware Feature and Decision Fusion for Robust RGB-Thermal Object Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v5su%20%7C%2011s-00FFFF.svg)](https://docs.ultralytics.com/)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Executive Summary

Autonomous perception systems operating in real-world environments face significant visual degradations: total darkness, glare, dense fog, smoke, and camouflage. While optical **Visible (RGB)** sensors capture rich chrominance and spatial textures, they fail under illumination loss. Conversely, **Thermal Infrared (IR)** sensors detect heat signatures regardless of lighting, but lack color discrimination and fine background texture.

**MultiSensorFusion_RGB_IR** is a comprehensive, production-grade multimodal object detection framework that synergizes Visible and Thermal Infrared imaging through:
1. **Feature-Level Fusion**: Target-Aware Dual Adversarial Learning (**TarDAL**) preserving thermal saliency and optical chrominance in $YCrCb$ space.
2. **Modern Deep Detectors**: Benchmarking and fine-tuning across **YOLOv5su** and state-of-the-art **YOLO11s**.
3. **Decision-Level Late Fusion**: Dynamic modality weighting and box ensembling (WBF / NMS) with simulated sensor dropout robustness.
4. **Live Interactive Demonstrator**: A full-featured, accessible Streamlit UI for real-time paired image and synchronized dual-video playback.

---

## 📊 Benchmark Results on M3FD

Evaluated on the official 4,200-pair **M3FD (Multi-Modal Multi-Task Fusion Detection)** benchmark across the standard Validation ($N=420$) and Test ($N=420$) splits under COCO evaluation protocols ($	ext{IoU}=0.50$ and $	ext{IoU}=0.50:0.95$):

| Method / Architecture | Fusion Level | Val mAP@50 | Val mAP@50:95 | Test mAP@50 | Test mAP@50:95 | Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RGB Unimodal Baseline** | None | 68.61% | 40.54% | 48.06% | 27.60% | 9.12 M |
| **IR Unimodal Baseline** | None | 65.65% | 38.65% | 47.90% | 28.09% | 9.12 M |
| **TarDAL + YOLOv5su** | Feature-Level | 68.96% | 40.75% | 48.56% | 28.16% | ~10.1 M |
| **Decision-Level Late Fusion** | Decision-Level | 71.39% | 42.14% | 51.52% | 30.12% | 18.24 M |
| **TarDAL + YOLO11s (Ours)** | **Feature-Level** | **73.96%** | **44.91%** | **53.27%** | **31.39%** | **9.41 M** |

> For comprehensive per-class breakdowns (People, Car, Bus, Motorcycle, Lamp, Truck) and distribution shift analyses, see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

## 📁 Repository Structure

```text
MultiSensorFusion_RGB_IR/
├── .github/                   # GitHub workflows and issue templates
├── configs/                   # Hyperparameter and dataset configuration YAMLs
├── data/                      # Dataset guidance (Data is hosted externally)
│   ├── .gitignore             # Strictly ignores raw data files
│   └── README.md              # Download instructions (Kaggle, Drive)
├── docs/                      # Technical reports, architecture, and benchmarks
│   ├── ARCHITECTURE.md        # Mathematical formulation of TarDAL & late fusion
│   ├── BENCHMARKS.md          # Full validation & test mAP breakdown tables
│   ├── DATASET_GUIDE.md       # External dataset acquisition guide
│   └── assets/                # Visual diagrams and comparison figures
├── interactive_demo/          # Production-grade Streamlit live deployment interface
│   ├── app.py                 # Application entrypoint
│   ├── core/                  # Inference pipeline, latency benchmarks, video engine
│   ├── models/                # YOLO & TarDAL wrappers, late fusion logic
│   ├── preprocessing/         # YCrCb color decomposition & Tanh remapping
│   └── ui/                    # WCAG-compliant high-contrast dashboard
├── notebooks/                 # Academic exploration and synthesis notebooks
├── scripts/                   # CLI execution scripts (dataset verification, demo runner)
├── src/                       # Core reusable Python package
│   ├── fusion/                # Feature-level fusion operators
│   ├── detection/             # Detection heads and bounding box aggregators
│   └── utils/                 # Metrics and visualization tools
├── weights/                   # Checkpoint storage guidance & download links
├── .gitignore                 # Standard open-source gitignore
├── CITATION.cff               # Citation metadata
├── LICENSE                    # MIT License
├── README.md                  # Main repository README
├── environment.yml            # Conda environment file
└── requirements.txt           # Pip dependencies
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Hassan-SL/Multimodal-RGB--Thermal-Image-Fusion.git
cd MultiSensorFusion_RGB_IR
```

### 2. Environment Setup

**Using Conda (Recommended):**
```bash
conda env create -f environment.yml
conda activate multisensor_fusion
```

**Or using Pip:**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Download Dataset & Checkpoints
- Follow [data/README.md](data/README.md) to download the M3FD dataset from Kaggle or the official repository.
- Download pre-trained weights directly from [GitHub Release v1.0.0](https://github.com/Hassan-SL/Multimodal-RGB--Thermal-Image-Fusion/releases/tag/v1.0.0) and place them into weights/ (see [weights/README.md](weights/README.md)).

### 4. Run the Interactive Web Demonstrator
Launch the live interactive research demonstration dashboard:
```bash
python scripts/run_demo.py
```
Or directly with Streamlit:
```bash
streamlit run interactive_demo/app.py
```
*Open `http://localhost:8501` in your browser to interact with live image fusion, synchronized dual-video playback, real-time FPS profiling, and sensor dropout stress-testing.*

---

## 📖 Citation

If you find this repository or research helpful in your work, please cite:

```bibtex
@misc{multisensorfusion2026,
  title={MultiSensorFusion_RGB_IR: Multimodal Target-Aware Feature and Decision Fusion for Robust Object Detection},
  author={MultiSensorFusion Research Team},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/Hassan-SL/Multimodal-RGB--Thermal-Image-Fusion}}
}
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
