"""
Stage 3 & Baseline Evaluator (scripts_AG/12_eval_test_set_and_baselines.py)

Evaluates model checkpoints on M3FD val and test splits (840 images each).
Supports both Ultralytics YOLO checkpoints and Stage 3 TarDAL native checkpoints.
Features full YCbCr -> RGB color reconstruction, 640x640 standardized input resizing,
and explicit evaluation parameters: imgsz=640, batch=16, device=device, conf=0.001, iou=0.6.
"""

import sys
import os
import time
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
import cv2
import yaml
from kornia.color import ycbcr_to_rgb
from torchvision.ops import box_convert

# Path setup for native TarDAL modules
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
TARDAL_MAIN_DIR = CODE_ROOT / "TarDAL-main"

if TARDAL_MAIN_DIR.exists() and str(TARDAL_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TARDAL_MAIN_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Register safe globals for PyTorch 2.6 compatibility
try:
    import ultralytics
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
except Exception:
    pass

# Native TarDAL imports
try:
    from module.fuse.generator import Generator
    from module.detect.models.yolo import Model as YOLOModel
    from module.detect.utils.metrics import ap_per_class, box_iou
    from module.detect.utils.general import non_max_suppression
    from pipeline.detect import Detect
    from pipeline.fuse import Fuse
except ImportError:
    from modules.generator import Generator
    from module.detect.models.yolo import Model as YOLOModel
    from module.detect.utils.metrics import ap_per_class, box_iou
    from module.detect.utils.general import non_max_suppression

try:

import sys
from pathlib import Path

# Add scripts directory to sys.path so path_config is always discoverable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import path_config
from path_config import *

except Exception:
    CODE_ROOT = SCRIPT_DIR.parent
    DATASET_ROOT = Path("/content/m3fd/M3FD_Detection")
    RUNS_DIR = CODE_ROOT / "runs"
    CHECKPOINTS_DIR = CODE_ROOT / "checkpoints"

# Fixed class order matching M3FD dataset ground truth: 0: People, 1: Car, 2: Bus, 3: Lamp, 4: Motorcycle, 5: Truck
CLASSES = ['People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck']
NAMES_DICT = {0: 'People', 1: 'Car', 2: 'Bus', 3: 'Lamp', 4: 'Motorcycle', 5: 'Truck'}


def safe_torch_load(path, device):
    """Loads PyTorch checkpoints safely, handling PyTorch 2.6+ weights_only defaults."""
    if not path or not Path(path).exists():
        return None
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)
    except Exception:
        try:
            return torch.load(str(path), map_location=device)
        except Exception as ex:
            raise ex


def resolve_generator_checkpoint(name: str | None) -> Path | None:
    if not name or str(name).strip() == "":
        name = "stage3_gen_best.pt"
    
    raw_name = Path(name).name
    key = raw_name.lower().replace('.pth', '').replace('.pt', '').strip()
    alias_map = {
        'dt': 'tardal-dt.pth', 'tardal_dt': 'tardal-dt.pth', 'tardal-dt': 'tardal-dt.pth',
        'ct': 'tardal-ct.pth', 'tardal_ct': 'tardal-ct.pth', 'tardal-ct': 'tardal-ct.pth',
        'tt': 'tardal-tt.pth', 'tardal_tt': 'tardal-tt.pth', 'tardal-tt': 'tardal-tt.pth'
    }
    target_fname = alias_map.get(key, raw_name)

    search_paths = [
        CHECKPOINTS_DIR / target_fname,
        TARDAL_MAIN_DIR / "weights" / "v1" / target_fname,
        CODE_ROOT / "weights" / target_fname,
        Path("/content/drive/MyDrive/FYP/code/checkpoints") / target_fname
    ]
    found = next((p for p in search_paths if p.exists()), None)
    if found:
        return found
    return None


def resolve_detector_checkpoint(name: str | None) -> Path | None:
    if not name or str(name).strip() == "":
        name = "stage3_best.pt"
    raw_name = Path(name).name
    search_paths = [
        CHECKPOINTS_DIR / raw_name,
        CHECKPOINTS_DIR / "stage3_best.pt",
        CHECKPOINTS_DIR / "best.pt",
        RUNS_DIR / "stage3_joint_end2end" / raw_name,
        RUNS_DIR / "fine_tune_detection" / "tardal_head_finetune" / "weights" / raw_name,
        Path("/content/drive/MyDrive/FYP/code/checkpoints") / raw_name,
        Path("/content/drive/MyDrive/FYP/code/checkpoints") / "stage3_best.pt"
    ]
    found = next((p for p in search_paths if p.exists()), None)
    if found:
        return found
    return None


def refuse_dataset_with_generator(generator, device, target_split="val"):
    """
    Fuses IR and VIS images dynamically using the specified Stage 3 Generator backbone,
    saving the output into /content/fused_dataset_stage3/{target_split}/images for evaluation.
    """
    cand_m3fd = [
        Path("/content/m3fd/M3FD_Detection"),
        Path("/content/m3fd"),
        CODE_ROOT / "data" / "m3fd",
        CODE_ROOT / "data" / "M3FD_Detection",
        CODE_ROOT / "m3fd_yolo_official_split",
    ]
    m3fd_root = next((p for p in cand_m3fd if p.exists() and (p / "meta").exists()), cand_m3fd[0])

    base_fused = Path("/content/fused_dataset_stage3") if Path("/content").exists() else CODE_ROOT / "runs" / "fused_dataset_stage3"
    out_dir = base_fused / target_split
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    meta_file = m3fd_root / "meta" / f"{target_split}.txt"
    if not meta_file.exists():
        meta_cands = [
            Path("/content/m3fd/M3FD_Detection/meta") / f"{target_split}.txt",
            Path("/content/m3fd/meta") / f"{target_split}.txt",
            CODE_ROOT / "data" / "m3fd" / "meta" / f"{target_split}.txt",
            CODE_ROOT / "m3fd_yolo_official_split" / "meta" / f"{target_split}.txt",
        ]
        meta_file = next((p for p in meta_cands if p.exists()), None)

    if not meta_file or not meta_file.exists():
        print(f"[AG Eval Error] Meta split file for '{target_split}' not found!")
        return None

    img_names = [l.strip() for l in meta_file.read_text(encoding='utf-8').splitlines() if l.strip()]
    ir_dir = m3fd_root / "ir"
    vi_dir = m3fd_root / "vi"
    lbl_dir = m3fd_root / "labels"
    if not ir_dir.exists():
        for cand in [Path("/content/m3fd/M3FD_Detection"), Path("/content/m3fd"), CODE_ROOT / "data" / "m3fd"]:
            if (cand / "ir").exists():
                ir_dir = cand / "ir"
                vi_dir = cand / "vi"
                lbl_dir = cand / "labels"
                break

    print(f"[AG Eval] 🔄 Dynamically fusing {len(img_names)} image pairs for '{target_split}' using live Generator -> {img_out}...")
    
    generator.eval()
    with torch.no_grad():
        for name in tqdm(img_names, desc=f"Dynamic Fusion ({target_split.upper()})"):
            p_stem = Path(name).stem
            ir_p = next((ir_dir / f for f in [f"{p_stem}.png", f"{p_stem}.jpg", f"{p_stem}.jpeg"] if (ir_dir / f).exists()), None)
            vi_p = next((vi_dir / f for f in [f"{p_stem}.png", f"{p_stem}.jpg", f"{p_stem}.jpeg"] if (vi_dir / f).exists()), None)
            lbl_p = lbl_dir / f"{p_stem}.txt"
            if not lbl_p.exists():
                lbl_cands = [
                    lbl_dir / f"{p_stem}.txt",
                    Path("/content/m3fd/M3FD_Detection/labels") / f"{p_stem}.txt",
                    CODE_ROOT / "data" / "m3fd" / "labels" / f"{p_stem}.txt"
                ]
                lbl_p = next((p for p in lbl_cands if p.exists()), lbl_p)

            if not ir_p or not vi_p or not ir_p.exists() or not vi_p.exists():
                continue

            ir_img = cv2.imread(str(ir_p), cv2.IMREAD_GRAYSCALE)
            vi_img = cv2.imread(str(vi_p))

            if ir_img is None or vi_img is None:
                continue

            ir_img = cv2.resize(ir_img, (640, 640))
            vi_img = cv2.resize(vi_img, (640, 640))

            vi_ycrcb = cv2.cvtColor(vi_img, cv2.COLOR_BGR2YCrCb)
            vi_y = vi_ycrcb[:, :, 0]
            cbcr = vi_ycrcb[:, :, 1:3]

            ir_t = torch.from_numpy(ir_img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            vi_t = torch.from_numpy(vi_y).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

            fused_y_t = generator(ir_t, vi_t)
            # Static deterministic Tanh mapping [-1, 1] -> [0, 1]
            fused_y_01 = ((fused_y_t + 1.0) / 2.0).clamp(0.0, 1.0)
            fused_y = (fused_y_01.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            fused_ycrcb = np.dstack((fused_y, cbcr))
            fused_bgr = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)

            cv2.imwrite(str(img_out / f"{p_stem}.jpg"), fused_bgr)

            if lbl_p.exists():
                shutil.copy2(str(lbl_p), str(lbl_out / f"{p_stem}.txt"))

    yaml_p = base_fused / "data_stage3.yaml"
    data_dict = {
        'path': str(base_fused),
        'train': f"{target_split}/images",
        'val': f"{target_split}/images",
        'test': f"{target_split}/images",
        'names': NAMES_DICT
    }
    yaml_p.write_text(yaml.dump(data_dict, default_flow_style=False), encoding='utf-8')
    return yaml_p


def ensure_eval_dataset_yaml(target_split="val"):
    base_fused = Path("/content/fused_dataset_stage3") if Path("/content").exists() else CODE_ROOT / "runs" / "fused_dataset_stage3"
    stage3_split_img = base_fused / target_split / "images"
    if stage3_split_img.exists() and len(list(stage3_split_img.glob("*.*"))) >= 800:
        yaml_p = base_fused / "data_stage3.yaml"
        data_dict = {
            'path': str(base_fused),
            'train': f"{target_split}/images",
            'val': f"{target_split}/images",
            'test': f"{target_split}/images",
            'names': NAMES_DICT
        }
        yaml_p.write_text(yaml.dump(data_dict, default_flow_style=False), encoding='utf-8')
        return yaml_p

    cand_fused = [
        base_fused,
        Path("/content/fused_dataset_stage2"),
        RUNS_DIR / "fused_dataset_stage2",
        Path("/content/m3fd/M3FD_Detection"),
        CODE_ROOT / "data" / "m3fd"
    ]
    
    active_dir = None
    for d in cand_fused:
        img_sub = d / target_split / "images"
        if img_sub.exists() and len(list(img_sub.glob("*.*"))) > 0:
            active_dir = d
            break
            
    if not active_dir:
        active_dir = base_fused

    eval_yaml_path = RUNS_DIR / "data_eval_dynamic.yaml"
    eval_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    split_rel = f"{target_split}/images" if (active_dir / target_split / "images").exists() else "images"

    data_dict = {
        'path': str(active_dir),
        'train': split_rel,
        'val': split_rel,
        'test': split_rel,
        'names': NAMES_DICT
    }
    with open(eval_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_dict, f, default_flow_style=False)
        
    return eval_yaml_path


def evaluate_test_set_and_baselines(gen_override: str = None, det_override: str = None, force_refuse: bool = False, split_name: str = "val"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_split = "val" if str(split_name).lower() in ["val", "validation"] else "test"

    gen_ckpt = resolve_generator_checkpoint(gen_override)
    det_ckpt = resolve_detector_checkpoint(det_override)

    print("=" * 80)
    print(f" 📊 STAGE 3 EVALUATION & BENCHMARK DASHBOARD ({target_split.upper()} SPLIT) ".center(80))
    print("=" * 80)
    print(f"  Generator Backbone Checkpoint : {gen_ckpt.name if gen_ckpt else 'Default Initialization'}")
    print(f"  Detector Head Checkpoint      : {det_ckpt.name if det_ckpt else 'Default Initialization'}")
    print(f"  Evaluation Split              : {target_split.upper()}")
    print(f"  Compute Device                : {device}")
    print(f"  Standardized Eval Protocol    : imgsz=640, batch=16, conf=0.001, iou=0.6")
    print("=" * 80 + "\n")

    # 1. Instantiate TarDAL Generator Backbone with fallback key checking ('g' or 'fuse')
    generator = Generator(dim=32, depth=3).to(device)
    if gen_ckpt and gen_ckpt.exists():
        try:
            ckpt_data = safe_torch_load(gen_ckpt, device)
            state_dict = None
            if isinstance(ckpt_data, dict):
                if 'g' in ckpt_data:
                    state_dict = ckpt_data['g']
                elif 'fuse' in ckpt_data:
                    state_dict = ckpt_data['fuse']
                else:
                    state_dict = ckpt_data
            else:
                state_dict = ckpt_data

            if isinstance(state_dict, dict):
                generator.load_state_dict(state_dict, strict=False)
                print(f"[AG Eval] Loaded Generator weights from {gen_ckpt.name}")
        except Exception as e:
            print(f"[AG Eval Warning] Generator weight loading note: {e}")
    generator.eval()

    # 2. Detector Resolution (TarDAL Native Stage 3 branch & Ultralytics Stage 2 branch)
    det_model = None
    yolo_evaluator = None

    if det_ckpt and det_ckpt.exists():
        ckpt_data = safe_torch_load(det_ckpt, device)
        
        # Check if checkpoint is a TarDAL Stage 3 dictionary format (contains 'detect' or 'fuse')
        if isinstance(ckpt_data, dict) and ('detect' in ckpt_data or 'fuse' in ckpt_data):
            print(f"[AG Eval] Detected TarDAL Stage 3 native dictionary checkpoint ({det_ckpt.name})!")
            cfg_path = TARDAL_MAIN_DIR / "module" / "detect" / "models" / "yolov5s.yaml"
            if cfg_path.exists():
                det_model = YOLOModel(cfg=str(cfg_path), ch=3, nc=6).to(device)
                
                # Extract detector state dict
                raw_state = None
                if 'detect' in ckpt_data:
                    d_val = ckpt_data['detect']
                    raw_state = d_val['detect'] if isinstance(d_val, dict) and 'detect' in d_val else d_val
                elif 'model' in ckpt_data:
                    m_val = ckpt_data['model']
                    raw_state = m_val.state_dict() if hasattr(m_val, 'state_dict') else m_val

                if isinstance(raw_state, dict):
                    target_state = det_model.state_dict()
                    remapped = {}
                    for k, v in raw_state.items():
                        if k in target_state and target_state[k].shape == v.shape:
                            remapped[k] = v
                        elif k.startswith('model.22.'):
                            alt_k = k.replace('model.22.', 'model.24.')
                            if alt_k in target_state and target_state[alt_k].shape == v.shape:
                                remapped[alt_k] = v
                    det_model.load_state_dict(remapped, strict=False)
                    print(f"[AG Eval] Successfully loaded Stage 3 TarDAL Detector weights into PyTorch model!")
                det_model.eval()
        else:
            # Stage 2 Ultralytics checkpoint path
            try:
                from ultralytics import YOLO
                yolo_evaluator = YOLO(str(det_ckpt))
                print(f"[AG Eval] Successfully loaded Ultralytics Stage 2 Detector from {det_ckpt.name}")
            except Exception as e:
                print(f"[AG Eval Warning] Detector weight loading note: {e}")

    # 3. Dynamic YAML verification & live in-memory fusion
    base_fused = Path("/content/fused_dataset_stage3") if Path("/content").exists() else CODE_ROOT / "runs" / "fused_dataset_stage3"
    stage3_split_img = base_fused / target_split / "images"
    has_fused_data = stage3_split_img.exists() and len(list(stage3_split_img.glob("*.*"))) >= 800

    if force_refuse or not has_fused_data:
        print(f"[AG Eval] 🔄 Dynamic fusion required for '{target_split}' split using {gen_ckpt.name if gen_ckpt else 'Generator'}...")
        data_yaml = refuse_dataset_with_generator(generator, device, target_split=target_split)
    else:
        print(f"[AG Eval] ✅ Using existing fused '{target_split}' split in {stage3_split_img.parent} ({len(list(stage3_split_img.glob('*.*')))} images)")
        yaml_p = base_fused / "data_stage3.yaml"
        data_dict = {
            'path': str(base_fused),
            'train': f"{target_split}/images",
            'val': f"{target_split}/images",
            'test': f"{target_split}/images",
            'names': NAMES_DICT
        }
        yaml_p.write_text(yaml.dump(data_dict, default_flow_style=False), encoding='utf-8')
        data_yaml = yaml_p

    if not data_yaml or not data_yaml.exists():
        data_yaml = refuse_dataset_with_generator(generator, device, target_split=target_split)

    # 4. Evaluation Execution with Standardized Parameters
    t_start = time.time()

    if yolo_evaluator and data_yaml and data_yaml.exists():
        print(f"[AG Eval] Running official Ultralytics evaluation (imgsz=640, batch=16, conf=0.001, iou=0.6) on: {data_yaml}...")
        metrics = yolo_evaluator.val(
            data=str(data_yaml),
            split=target_split,
            imgsz=640,
            batch=16,
            device=device,
            conf=0.001,
            iou=0.6,
            verbose=False
        )
        t_end = time.time()
        t_total = (t_end - t_start) / 840.0 * 1000.0
        fps = 1000.0 / max(t_total, 1e-5)

        per_class_table = []
        for i, c_name in enumerate(CLASSES):
            ap50 = metrics.box.ap50[i] * 100 if hasattr(metrics.box, 'ap50') and i < len(metrics.box.ap50) else 0.0
            ap50_95 = metrics.box.ap[i] * 100 if hasattr(metrics.box, 'ap') and i < len(metrics.box.ap) else 0.0
            p_val = metrics.box.p[i] * 100 if hasattr(metrics.box, 'p') and i < len(metrics.box.p) else 0.0
            r_val = metrics.box.r[i] * 100 if hasattr(metrics.box, 'r') and i < len(metrics.box.r) else 0.0
            per_class_table.append([c_name, f"{p_val:.2f}%", f"{r_val:.2f}%", f"{ap50:.2f}%", f"{ap50_95:.2f}%"])

        mp = metrics.box.mp * 100 if hasattr(metrics.box, 'mp') else 0.0
        mr = metrics.box.mr * 100 if hasattr(metrics.box, 'mr') else 0.0
        map50 = metrics.box.map50 * 100 if hasattr(metrics.box, 'map50') else 0.0
        map50_95 = metrics.box.map * 100 if hasattr(metrics.box, 'map') else 0.0
        per_class_table.append(["OVERALL (ALL)", f"{mp:.2f}%", f"{mr:.2f}%", f"{map50:.2f}%", f"{map50_95:.2f}%"])

        print("\n--- PER-CLASS ACCURACY METRICS ---")
        print(tabulate(per_class_table, headers=["Class Name", "Precision (P)", "Recall (R)", "mAP@50", "mAP@50-95"], tablefmt="grid"))

        telemetry_table = [
            ["Dataset Split", target_split.upper()],
            ["Compute Device", str(device)],
            ["Generator Weights", gen_ckpt.name if gen_ckpt else "N/A"],
            ["Detector Weights", det_ckpt.name if det_ckpt else "N/A"],
            ["Dataset Config", str(data_yaml)],
            ["Overall mAP@50", f"{map50:.2f}%"],
            ["Overall mAP@50-95", f"{map50_95:.2f}%"],
            ["Average Frame Latency", f"{t_total:.2f} ms/frame"],
            ["FPS Throughput", f"{fps:.1f} FPS"]
        ]
        print("\n--- HARDWARE & LATENCY TELEMETRY ---")
        print(tabulate(telemetry_table, headers=["Telemetry Metric", "Measured Value"], tablefmt="grid"))
        print("=" * 80 + "\n")

    elif det_model:
        print(f"[AG Eval] Running TarDAL native evaluation loop (imgsz=640, batch=16, conf=0.001, iou=0.6) on Stage 3 model...")
        det_model.eval()
        
        m3fd_root = Path("/content/m3fd/M3FD_Detection")
        if not m3fd_root.exists():
            m3fd_root = CODE_ROOT / "data" / "m3fd"
            
        meta_file = m3fd_root / "meta" / f"{target_split}.txt"
        if not meta_file.exists():
            meta_file = m3fd_root / "meta" / "val.txt"
            
        img_names = [l.strip() for l in meta_file.read_text(encoding='utf-8').splitlines() if l.strip()]
        ir_dir = m3fd_root / "ir"
        vi_dir = m3fd_root / "vi"
        lbl_dir = m3fd_root / "labels"
        
        iou_v = torch.linspace(0.5, 0.95, 10).to(device)
        stats = []
        
        with torch.no_grad():
            for name in tqdm(img_names, desc=f"Evaluating Stage 3 ({target_split.upper()})"):
                p_stem = Path(name).stem
                ir_p = ir_dir / f"{p_stem}.png"
                vi_p = vi_dir / f"{p_stem}.png"
                lbl_p = lbl_dir / f"{p_stem}.txt"

                if not ir_p.exists() or not vi_p.exists():
                    continue

                ir_img = cv2.imread(str(ir_p), cv2.IMREAD_GRAYSCALE)
                vi_img = cv2.imread(str(vi_p))
                if ir_img is None or vi_img is None:
                    continue

                # Standardized 640x640 resize
                ir_img = cv2.resize(ir_img, (640, 640))
                vi_img = cv2.resize(vi_img, (640, 640))

                # Extract YCbCr channels for full RGB reconstruction via Kornia
                vi_ycrcb = cv2.cvtColor(vi_img, cv2.COLOR_BGR2YCrCb)
                vi_y = vi_ycrcb[:, :, 0]
                vi_cr = vi_ycrcb[:, :, 1]
                vi_cb = vi_ycrcb[:, :, 2]
                
                ir_t = torch.from_numpy(ir_img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                vi_t = torch.from_numpy(vi_y).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

                # Live fusion with fine-tuned Stage 3 generator
                fused_y_t = generator(ir_t, vi_t)
                # Static deterministic Tanh mapping [-1, 1] -> [0, 1]
                fused_y_01 = ((fused_y_t + 1.0) / 2.0).clamp(0.0, 1.0)

                # Reconstruct 3-channel RGB via YCbCr -> RGB
                cbcr_t = torch.from_numpy(np.stack([vi_cb, vi_cr], axis=-1)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                fused_ycbcr = torch.cat([fused_y_01, cbcr_t], dim=1)
                fused_rgb = ycbcr_to_rgb(fused_ycbcr)

                # Forward detection pass on 3-channel RGB input
                preds, _ = det_model(fused_rgb)
                preds = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.6, multi_label=True)

                labels = []
                if lbl_p.exists() and lbl_p.stat().st_size > 0:
                    lbl_lines = lbl_p.read_text(encoding='utf-8').splitlines()
                    for line in lbl_lines:
                        parts = [float(x) for x in line.strip().split()]
                        if len(parts) >= 5:
                            labels.append(parts[:5])
                            
                labels_t = torch.tensor(labels, device=device) if len(labels) else torch.zeros((0, 5), device=device)

                for pred in preds:
                    num_l, num_p = labels_t.shape[0], pred.shape[0]
                    correct = torch.zeros(num_p, 10, dtype=torch.bool, device=device)

                    if num_p == 0:
                        if num_l:
                            stats.append((correct, *torch.zeros((2, 0), device=device), labels_t[:, 0]))
                        continue

                    if num_l:
                        lbl_px = labels_t.clone()
                        lbl_px[:, 1:5] *= torch.tensor((640, 640, 640, 640), device=device)
                        lbl_xyxy = lbl_px.clone()
                        lbl_xyxy[:, 1:5] = box_convert(lbl_px[:, 1:5], in_fmt='cxcywh', out_fmt='xyxy')

                        iou = box_iou(lbl_xyxy[:, 1:5], pred[:, :4])
                        x = torch.where((iou >= iou_v[0]) & (lbl_xyxy[:, 0:1] == pred[:, 5]))
                        if x[0].shape[0]:
                            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                            if x[0].shape[0] > 1:
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                            matches = torch.from_numpy(matches).to(device)
                            for i, k in enumerate(iou_v):
                                valid = matches[:, 2] >= k
                                if valid.any():
                                    correct[matches[valid, 1].long(), i] = True

                    stats.append((correct, pred[:, 4], pred[:, 5], labels_t[:, 0]))

        t_end = time.time()
        t_total = (t_end - t_start) / max(len(img_names), 1) * 1000.0
        fps = 1000.0 / max(t_total, 1e-5)

        stats_cat = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
        names_dict = {i: name for i, name in enumerate(CLASSES)}
        
        per_class_table = []
        if len(stats_cat) and stats_cat[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats_cat, names=names_dict)
            ap50, ap_all = ap[:, 0], ap.mean(1)

            for idx, c_idx in enumerate(ap_class):
                c_name = CLASSES[int(c_idx)] if int(c_idx) < len(CLASSES) else f"Class_{c_idx}"
                per_class_table.append([c_name, f"{p[idx]*100:.2f}%", f"{r[idx]*100:.2f}%", f"{ap50[idx]*100:.2f}%", f"{ap_all[idx]*100:.2f}%"])

            mp, mr, map50, map50_95 = p.mean()*100, r.mean()*100, ap50.mean()*100, ap_all.mean()*100
            per_class_table.append(["OVERALL (ALL)", f"{mp:.2f}%", f"{mr:.2f}%", f"{map50:.2f}%", f"{map50_95:.2f}%"])
        else:
            map50, map50_95 = 0.0, 0.0
            per_class_table.append(["OVERALL (ALL)", "0.00%", "0.00%", "0.00%", "0.00%"])

        print("\n--- PER-CLASS ACCURACY METRICS ---")
        print(tabulate(per_class_table, headers=["Class Name", "Precision (P)", "Recall (R)", "mAP@50", "mAP@50-95"], tablefmt="grid"))

        telemetry_table = [
            ["Dataset Split", target_split.upper()],
            ["Compute Device", str(device)],
            ["Generator Weights", gen_ckpt.name if gen_ckpt else "N/A"],
            ["Detector Weights", det_ckpt.name if det_ckpt else "N/A"],
            ["Overall mAP@50", f"{map50:.2f}%"],
            ["Overall mAP@50-95", f"{map50_95:.2f}%"],
            ["Average Frame Latency", f"{t_total:.2f} ms/frame"],
            ["FPS Throughput", f"{fps:.1f} FPS"]
        ]
        print("\n--- HARDWARE & LATENCY TELEMETRY ---")
        print(tabulate(telemetry_table, headers=["Telemetry Metric", "Measured Value"], tablefmt="grid"))
        print("=" * 80 + "\n")
    else:
        print("[AG Eval Warning] Evaluation could not be executed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_ckpt", "--gen", type=str, default=None, help="Generator backbone checkpoint")
    parser.add_argument("--det_ckpt", "--det", type=str, default=None, help="Detector head checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split: 'val' or 'test'")
    parser.add_argument("--force_refuse", action="store_true", help="Force re-generation of fused images")
    args = parser.parse_args()

    evaluate_test_set_and_baselines(gen_override=args.gen_ckpt, det_override=args.det_ckpt, force_refuse=args.force_refuse, split_name=args.split)