"""
01_verify_and_link_dataset.py
Verifies dataset integrity and automatically establishes portable directory junctions for TarDAL.
"""

import sys
import subprocess
from pathlib import Path

# Import portable path resolver

import sys
from pathlib import Path

# Add scripts directory to sys.path so path_config is always discoverable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import path_config
from path_config import *


def link_junction(target: Path, src: Path):
    if target.exists():
        try:
            target.rmdir()
        except Exception:
            pass

    if not target.exists() and src and src.exists():
        cmd = f'cmd /c mklink /J "{target}" "{src}"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"[Link] Junction created: {target} -> {src}")
    elif target.exists():
        print(f"[Link] Junction target already exists: {target}")
    else:
        print(f"[Error] Source directory {src} does not exist.")

def verify_dataset():
    print(f"\n--- Verifying Dataset at {DATASET_ROOT} ---")
    if not DATASET_ROOT.exists():
        print(f"[Error] Dataset directory not found: {DATASET_ROOT}")
        return False

    subdirs = [("ir", IR_DIR), ("vi", VI_DIR), ("labels", LABELS_DIR), ("meta", META_DIR)]
    all_ok = True
    for name, path in subdirs:
        if path.exists() and path.is_dir():
            count = len(list(path.iterdir()))
            print(f"  [OK] {name:<8}: {count} files found")
        else:
            print(f"  [MISSING] {name:<8}: folder does not exist at {path}")
            all_ok = False

    return all_ok

def main():
    print("=== Portable Dataset Setup & Linker ===")
    if verify_dataset():
        # Setup links for code/data/m3fd and TarDAL-main/data/m3fd
        target_code_data = CODE_ROOT / "data" / "m3fd"
        link_junction(target_code_data, DATASET_ROOT)
        link_junction(TARDAL_DATA_DIR, DATASET_ROOT)
        print("\nDataset verified and linked successfully!")
    else:
        print("\nDataset verification failed. Please check folder contents.")

if __name__ == "__main__":
    main()