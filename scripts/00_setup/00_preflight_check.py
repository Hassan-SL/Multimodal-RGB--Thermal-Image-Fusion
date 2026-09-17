"""
00_preflight_check.py
Comprehensive pre-flight verification script for TarDAL + M3FD project.
Checks all dependencies, configuration parameters, weight files, and dataset structure.
"""

import sys
import yaml
from pathlib import Path

# Import path resolver

import sys
from pathlib import Path

# Add scripts directory to sys.path so path_config is always discoverable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import path_config
from path_config import *


def check_dependencies():
    print("\n--- 1. Verifying Python Package Dependencies ---")
    required = [
        "torch", "torchvision", "kornia", "cv2", "yaml", 
        "pandas", "matplotlib", "tqdm", "thop", "tabulate", "scipy"
    ]
    all_ok = True
    for lib in required:
        try:
            m = __import__(lib)
            ver = getattr(m, '__version__', 'installed')
            print(f"  [OK] {lib:<15}: {ver}")
        except ImportError as e:
            if lib == "thop":
                print(f"  [OPTIONAL] {lib:<15}: {e} (Will auto-install in Colab)")
            else:
                print(f"  [MISSING] {lib:<15}: {e}")
                all_ok = False
    return all_ok

def check_weights():
    print("\n--- 2. Verifying Model Weight Checkpoint ---")
    if TARDAL_WEIGHTS_TT.exists():
        size_mb = TARDAL_WEIGHTS_TT.stat().st_size / (1024 * 1024)
        print(f"  [OK] Weights file found : {TARDAL_WEIGHTS_TT.name} ({size_mb:.2f} MB)")
        try:
            import torch
            ckpt = torch.load(TARDAL_WEIGHTS_TT, map_location="cpu")
            num_keys = len(ckpt.keys()) if isinstance(ckpt, dict) else len(ckpt)
            print(f"  [OK] PyTorch Checkpoint  : Successfully loaded ({num_keys} tensor keys)")
            return True
        except Exception as e:
            print(f"  [Error] Failed to load checkpoint: {e}")
            return False
    else:
        print(f"  [MISSING] Weights file not found at {TARDAL_WEIGHTS_TT}")
        return False

def check_config():
    print("\n--- 3. Verifying TarDAL-TT Configuration File ---")
    if not TARDAL_CONFIG_TT.exists():
        print(f"  [MISSING] Config file not found at {TARDAL_CONFIG_TT}")
        return False

    try:
        with open(TARDAL_CONFIG_TT, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        print(f"  [OK] Strategy          : {cfg.get('strategy')}")
        print(f"  [OK] Fuse Pretrained   : {cfg.get('fuse', {}).get('pretrained')}")
        print(f"  [OK] Detect Pretrained : {cfg.get('detect', {}).get('pretrained')}")
        print(f"  [OK] Dataset Name      : {cfg.get('dataset', {}).get('name')}")
        print(f"  [OK] Dataset Root      : {cfg.get('dataset', {}).get('root')}")
        print(f"  [OK] Save Text Labels  : {cfg.get('inference', {}).get('save_txt')}")
        return True
    except Exception as e:
        print(f"  [Error] Failed to parse config YAML: {e}")
        return False

def check_dataset():
    print("\n--- 4. Verifying Dataset Integrity & Link Structure ---")
    if not DATASET_ROOT.exists():
        print(f"  [MISSING] Dataset directory not found at {DATASET_ROOT}")
        return False

    subdirs = [("ir/IR", IR_DIR), ("vi/vis", VI_DIR), ("labels", LABELS_DIR), ("meta", META_DIR)]
    all_ok = True
    for name, path in subdirs:
        if path.exists() and path.is_dir() and len(list(path.iterdir())) > 0:
            count = len(list(path.iterdir()))
            print(f"  [OK] {name:<10}: {count} files")
        elif name == "labels":
            ann_dir = next((DATASET_ROOT / d for d in ["Annotation", "annotation"] if (DATASET_ROOT / d).exists()), None)
            if ann_dir and ann_dir.exists():
                print(f"  [AUTO-CONVERTING] 'labels/' missing. Found XML annotations in {ann_dir.name}. Auto-converting to YOLO format...")
                path.mkdir(parents=True, exist_ok=True)
                try:
                    import xml.etree.ElementTree as ET
                    CLASS_TO_ID = {"People": 0, "Car": 1, "Bus": 2, "Motorcycle": 3, "Lamp": 4, "Truck": 5}
                    xml_files = list(ann_dir.glob("*.xml"))
                    for xf in xml_files:
                        tree = ET.parse(xf)
                        root = tree.getroot()
                        sz = root.find("size")
                        w = float(sz.find("width").text)
                        h = float(sz.find("height").text)
                        out_lines = []
                        for obj in root.findall("object"):
                            c_name = obj.find("name").text
                            if c_name in CLASS_TO_ID:
                                cid = CLASS_TO_ID[c_name]
                                bnd = obj.find("bndbox")
                                xmin = float(bnd.find("xmin").text)
                                ymin = float(bnd.find("ymin").text)
                                xmax = float(bnd.find("xmax").text)
                                ymax = float(bnd.find("ymax").text)
                                xc = ((xmin + xmax) / 2.0) / w
                                yc = ((ymin + ymax) / 2.0) / h
                                bw = (xmax - xmin) / w
                                bh = (ymax - ymin) / h
                                out_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                        (path / f"{xf.stem}.txt").write_text("\n".join(out_lines) + "\n", encoding='utf-8')
                    print(f"  [OK] labels    : {len(xml_files)} files auto-converted from XML!")
                except Exception as ex:
                    print(f"  [MISSING] labels    : directory missing at {path} ({ex})")
                    all_ok = False
            else:
                print(f"  [MISSING] labels    : directory missing at {path}")
                all_ok = False
        else:
            print(f"  [MISSING] {name:<10}: directory missing at {path}")
            all_ok = False
    return all_ok

def run_preflight():
    print("==================================================")
    print("      TAR_DAL + M3FD PRE-FLIGHT SYSTEM REVIEW     ")
    print("==================================================")
    
    dep_ok = check_dependencies()
    w_ok = check_weights()
    cfg_ok = check_config()
    ds_ok = check_dataset()
    
    print("\n--------------------------------------------------")
    if dep_ok and w_ok and cfg_ok and ds_ok:
        print(" [PASSED] ALL PRE-FLIGHT CHECKS PASSED SUCCESSFULLY!")
        print(" You are 100% ready to run inference/training.")
    else:
        print(" [WARNING] SOME PRE-FLIGHT CHECKS FAILED OR NEED ATTENTION.")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    run_preflight()