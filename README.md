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

## 📊 Authoritative Benchmark Results on M3FD

Evaluated on the official 4,200-pair **M3FD (Multi-Modal Multi-Task Fusion Detection)** benchmark across the standard Validation split ($N=420$) and unseen Test split ($N=840$ image pairs, 6,697 labeled instances) under official academic protocols ($\\tau=0.001, \\theta=0.60$) and operational deployment protocols ($\\tau=0.25, \\theta=0.50$):

| Architecture / Stage | Fusion Paradigm | Val mAP@50 | Test mAP@50 (Academic) | Test mAP@50 (Operational) | Latency (T4) | Throughput | Fault Tolerance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 4A: Direct Optical (RGB)** | Unimodal Optical | 76.40% | 29.33% | 25.59% | **9.42 ms** | **106.2 FPS** | Fails in darkness |
| **Stage 4B: Direct Thermal (IR)** | Unimodal Thermal | 72.00% | 28.38% | 23.54% | **9.59 ms** | 104.3 FPS | Blind to cold objects |
| **Stage 3: TarDAL Feature Fusion** 🏆 | Feature-Level (YOLOv5su) | 74.57% | **48.56%** 🏆 | 40.12% | 77.95 ms | 12.8 FPS | Single Point of Failure |
| **Stage 5: Decision-Level Late Fusion** 🏆 | Decision-Level (WBF) | 74.01% | 34.10% | 30.29% | **19.41 ms** | **51.5 FPS** 🏆 | **100% Graceful Degradation** 🏆 |
| **Stage 6: Modern YOLO11s Fusion** 🏆 | Feature-Level (YOLO11s) | **81.08%** 🏆 | 45.72% | **41.65%** 🏆 | 79.42 ms | 12.6 FPS | **Peak People (77.5%) & Car (85.3%)** 🏆 |

### 🏆 Key Findings
- **Multimodal Superiority**: Feature fusion in Stage 3 achieves **48.56% mAP@50**, delivering a **+19.24% absolute gain (+65.6% relative boost)** over Direct Optical RGB.
- **Safety-Critical Performance**: Stage 6 YOLO11s sets all project records for primary real-world categories (**People: 77.49%**, **Car: 85.26%**, and Operational Deployment: **41.65%**).
- **Edge Deployment**: Stage 5 Late Fusion provides real-time throughput (**51.5 FPS**) and **100% sensor fault tolerance**.

> For detailed per-class breakdowns, latency decompositions, and domain shift analyses, see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

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
