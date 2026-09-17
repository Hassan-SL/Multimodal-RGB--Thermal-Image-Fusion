"""
Central Dynamic Path Configuration for MultiSensorFusion_RGB_IR.
Resolves all project, dataset, weights, and run paths dynamically.
Works out of the box across Windows, Linux, and Google Colab with ZERO hardcoded path defects.
"""

import os
import sys
from pathlib import Path

# 1. Dynamically locate project root by locating anchor files
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = None

for parent in [CURRENT_FILE.parent, CURRENT_FILE.parent.parent, CURRENT_FILE.parent.parent.parent]:
    if (parent / "requirements.txt").exists() and (parent / "interactive_demo").exists():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "scripts" else CURRENT_FILE.parent

CODE_ROOT = PROJECT_ROOT
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Ensure in sys.path
for p in [PROJECT_ROOT, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# 2. Dynamic dataset discovery (Prioritizes raw M3FD_Detection with vi/ir/labels)
POSSIBLE_DATA_DIRS = [
    PROJECT_ROOT.parent / "dataset" / "M3FD_Detection",
    PROJECT_ROOT / "data" / "M3FD_Detection",
    PROJECT_ROOT / "data" / "m3fd",
    PROJECT_ROOT.parent / "dataset" / "M3fd_yolo",
    PROJECT_ROOT.parent / "dataset",
    PROJECT_ROOT.parent / "code" / "m3fd_yolo_official_split",
    PROJECT_ROOT.parent / "code" / "data" / "m3fd",
    Path(os.environ.get("M3FD_DIR", ""))
]

DATASET_ROOT = next((p for p in POSSIBLE_DATA_DIRS if p.exists() and p.is_dir()), PROJECT_ROOT / "data" / "m3fd")

def find_subfolder(root: Path, candidates: list[str], fallback: str) -> Path:
    if root and root.exists():
        for cand in candidates:
            p = root / cand
            if p.exists() and p.is_dir():
                return p
    return root / fallback if root else Path(fallback)

IR_DIR = find_subfolder(DATASET_ROOT, ["ir", "IR", "Ir"], "ir")
VI_DIR = find_subfolder(DATASET_ROOT, ["vi", "vis", "VI", "Vis", "VIS"], "vi")
LABELS_DIR = find_subfolder(DATASET_ROOT, ["labels", "Labels", "LABELS"], "labels")
META_DIR = find_subfolder(DATASET_ROOT, ["meta", "Meta", "META"], "meta")
ANNOTATION_DIR = find_subfolder(DATASET_ROOT, ["Annotation", "annotation", "ANNOTATION"], "Annotation")

# 3. Weights and Checkpoints
POSSIBLE_WEIGHTS_DIRS = [
    PROJECT_ROOT / "weights",
    PROJECT_ROOT / "checkpoints",
    PROJECT_ROOT.parent / "code" / "checkpoints",
    PROJECT_ROOT.parent / "code" / "TarDAL-main" / "weights" / "v1",
    Path(os.environ.get("WEIGHTS_DIR", ""))
]
WEIGHTS_ROOT = next((p for p in POSSIBLE_WEIGHTS_DIRS if p.exists() and p.is_dir()), PROJECT_ROOT / "weights")
CHECKPOINTS_DIR = WEIGHTS_ROOT

def get_weights_path(filename: str) -> Path:
    for wdir in POSSIBLE_WEIGHTS_DIRS:
        if wdir.exists():
            candidate = wdir / filename
            if candidate.exists():
                return candidate
    return WEIGHTS_ROOT / filename

# 4. TarDAL Models & Configs
TARDAL_ROOT = PROJECT_ROOT.parent / "code" / "TarDAL-main"
if not TARDAL_ROOT.exists():
    TARDAL_ROOT = PROJECT_ROOT / "TarDAL-main"
TARDAL_CONFIG_TT = TARDAL_ROOT / "config" / "official" / "infer" / "tardal-tt.yaml"

# Find TarDAL-tt weights across multiple possible locations
_tardal_candidates = [
    WEIGHTS_ROOT / "tardal_generator.pth",
    PROJECT_ROOT.parent / "code" / "TarDAL-main" / "weights" / "v1" / "tardal-tt.pth",
    PROJECT_ROOT.parent / "code" / "checkpoints" / "stage3_gen_best.pt",
    WEIGHTS_ROOT / "tardal-tt.pth"
]
TARDAL_WEIGHTS_TT = next((w for w in _tardal_candidates if w.exists()), WEIGHTS_ROOT / "tardal-tt.pth")

# 5. Core directories
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

def print_paths_summary():
    print("=" * 70)
    print(" MultiSensorFusion_RGB_IR Dynamic Environment Resolution ".center(70))
    print("=" * 70)
    print(f"Project Root   : {PROJECT_ROOT}")
    print(f"Dataset Root   : {DATASET_ROOT} (Exists: {DATASET_ROOT.exists()})")
    print(f"  - IR Dir     : {IR_DIR} (Exists: {IR_DIR.exists()})")
    print(f"  - VI Dir     : {VI_DIR} (Exists: {VI_DIR.exists()})")
    print(f"  - Labels Dir : {LABELS_DIR} (Exists: {LABELS_DIR.exists()})")
    print(f"Weights Root   : {WEIGHTS_ROOT} (Exists: {WEIGHTS_ROOT.exists()})")
    print(f"TarDAL Weights : {TARDAL_WEIGHTS_TT} (Exists: {TARDAL_WEIGHTS_TT.exists()})")
    print(f"Runs Directory : {RUNS_DIR}")
    print(f"Configs Dir    : {CONFIGS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    print_paths_summary()
