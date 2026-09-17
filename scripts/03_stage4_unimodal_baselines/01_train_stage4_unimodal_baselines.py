#!/usr/bin/env python3

import sys
from pathlib import Path

# Add scripts directory to sys.path so path_config is always discoverable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import path_config
from path_config import *

"""
================================================================================
Stage 4: Direct Unimodal Input Training (IR-Only & RGB-Only YOLOv5su Baselines)
================================================================================
Trains YOLOv5su directly on unimodal raw data (Infrared or Visible Optical)
using the exact same official M3FD 60/20/20 train/val/test splits, epochs,
and hyperparameters as Stage 2 and Stage 3 to ensure a controlled comparison.

Standard Class Mapping (Official M3FD Ground Truth):
  0: People | 1: Car | 2: Bus | 3: Lamp | 4: Motorcycle | 5: Truck
================================================================================
"""

import argparse
import os
import sys
import shutil
import time
import math
import json
from datetime import datetime
from pathlib import Path
import yaml
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parent.parent

NAMES_DICT = {0: 'People', 1: 'Car', 2: 'Bus', 3: 'Lamp', 4: 'Motorcycle', 5: 'Truck'}


def resolve_m3fd_root():
    cand_roots = [
        Path('/content/m3fd/M3FD_Detection'),
        Path('/content/m3fd'),
        CODE_ROOT / 'data' / 'm3fd',
        CODE_ROOT / 'data' / 'M3FD_Detection',
        CODE_ROOT / 'm3fd_yolo_official_split',
    ]
    for p in cand_roots:
        if p.exists() and (p / 'meta').exists():
            return p
    return cand_roots[0]


def structure_unimodal_dataset(modality: str, local_ssd_dir: Path, m3fd_root: Path, force_rebuild: bool = False):
    """
    Structures a dedicated YOLO-format dataset directory on local high-speed SSD.
    Modality: 'ir' or 'vi' (rgb)
    """
    if force_rebuild and local_ssd_dir.exists():
        print(f"[Stage 4 Prep] 🔄 Force rebuild requested: removing cached {local_ssd_dir}...")
        shutil.rmtree(local_ssd_dir, ignore_errors=True)

    print(f"[Stage 4 Prep] Structuring unimodal '{modality.upper()}' dataset in {local_ssd_dir}...")
    
    src_img_dir = m3fd_root / ("ir" if modality == "ir" else "vi")
    src_lbl_dir = m3fd_root / "labels"
    meta_dir = m3fd_root / "meta"

    if not src_img_dir.exists() or not src_lbl_dir.exists():
        raise FileNotFoundError(f"Missing source directories in {m3fd_root}: img={src_img_dir}, lbl={src_lbl_dir}")

    for split in ["train", "val", "test"]:
        meta_p = meta_dir / f"{split}.txt"
        if not meta_p.exists():
            continue
        
        split_img_dir = local_ssd_dir / split / "images"
        split_lbl_dir = local_ssd_dir / split / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        stems = [line.strip() for line in meta_p.read_text(encoding='utf-8').splitlines() if line.strip()]
        existing_imgs = len(list(split_img_dir.glob("*.*")))
        
        if existing_imgs >= len(stems) and existing_imgs > 0:
            print(f"  - [{split.upper()}] {existing_imgs} images already structured. Skipping copy.")
            continue

        print(f"  - [{split.upper()}] Copying {len(stems)} {modality.upper()} images and labels...")
        for s in stems:
            stem_name = Path(s).stem
            img_cand = next((src_img_dir / f for f in [f"{stem_name}.png", f"{stem_name}.jpg", f"{stem_name}.jpeg"] if (src_img_dir / f).exists()), None)
            lbl_p = src_lbl_dir / f"{stem_name}.txt"

            if img_cand and img_cand.exists():
                dst_img = split_img_dir / img_cand.name
                if not dst_img.exists():
                    shutil.copy2(str(img_cand), str(dst_img))

            if lbl_p and lbl_p.exists():
                dst_lbl = split_lbl_dir / lbl_p.name
                if not dst_lbl.exists():
                    shutil.copy2(str(lbl_p), str(dst_lbl))

    # Generate YOLO data YAML
    yaml_p = local_ssd_dir / f"data_{modality}.yaml"
    data_dict = {
        'path': str(local_ssd_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': NAMES_DICT
    }
    with open(yaml_p, 'w', encoding='utf-8') as f:
        yaml.dump(data_dict, f, default_flow_style=False)

    print(f"[Stage 4 Prep] ✅ Dataset YAML generated: {yaml_p}")
    return yaml_p


def resolve_checkpoint(weights_str: str, modality: str, drive_ckpt_dir: Path, runs_dir: Path,
                       resume: bool = False, from_best: bool = False) -> Path:
    """
    Intelligently resolves checkpoint file with strict modality isolation:
      - Never loads an RGB checkpoint for IR modality.
      - Never loads an IR checkpoint for RGB modality.
      - Seamlessly resolves 'best' -> yolov5su_<modality>_best.pt.
      - Seamlessly resolves 'last' -> yolov5su_<modality>_last.pt.
      - Seamlessly resolves 'scratch' -> yolov5su.pt.
    """
    if weights_str:
        ws_clean = weights_str.strip().lower()
        if ws_clean in ['scratch', 'none', 'pretrained']:
            weights_str = 'yolov5su.pt'
        elif ws_clean in ['best', 'best.pt']:
            weights_str = f"yolov5su_{modality}_best.pt"
            from_best = True
        elif ws_clean in ['last', 'last.pt']:
            weights_str = f"yolov5su_{modality}_last.pt"
            resume = True

    # Strict cross-modality sanitization
    other_mod = 'rgb' if modality == 'ir' else 'ir'
    if weights_str:
        ws_lower = weights_str.lower()
        if other_mod in ws_lower or (modality == 'ir' and any(x in ws_lower for x in ['_vi', 'vi.', 'vis'])):
            print(f"  [Safety Guard] ⚠️ Checkpoint '{weights_str}' belongs to {other_mod.upper()}, not {modality.upper()}!")
            weights_str = f"yolov5su_{modality}_best.pt" if from_best else (f"yolov5su_{modality}_last.pt" if resume else "yolov5su.pt")
            print(f"  [Safety Guard] 🔄 Auto-switched to {modality.upper()} checkpoint: '{weights_str}'")

    # If scratch / yolov5su.pt requested, resolve directly to pretrained weights
    if weights_str == 'yolov5su.pt' and not from_best and not resume:
        for p in [CODE_ROOT / "weights" / "yolov5su.pt", Path("/content/drive/MyDrive/FYP/code/weights/yolov5su.pt"), Path("yolov5su.pt")]:
            if p.exists() and p.stat().st_size > 0:
                return p
        return Path("yolov5su.pt")

    candidates = []

    # 1. If from_best or weights specifies 'best', prioritize best checkpoint for this modality
    if from_best or (weights_str and 'best' in weights_str.lower()):
        candidates.extend([
            drive_ckpt_dir / f"yolov5su_{modality}_best.pt",
            CODE_ROOT / "checkpoints" / f"yolov5su_{modality}_best.pt",
            runs_dir / f"yolov5su_{modality}" / "weights" / "best.pt",
            Path("/content/drive/MyDrive/FYP/code/runs") / f"stage4_direct_{modality}" / f"yolov5su_{modality}" / "weights" / "best.pt",
        ])

    # 2. Modality-specific checkpoint in checkpoints dir
    if weights_str:
        p = Path(weights_str)
        if p.is_absolute() and p.exists():
            return p
        candidates.extend([
            drive_ckpt_dir / f"yolov5su_{modality}_{weights_str}",
            CODE_ROOT / "checkpoints" / f"yolov5su_{modality}_{weights_str}",
            drive_ckpt_dir / weights_str,
            CODE_ROOT / "checkpoints" / weights_str,
            runs_dir / f"yolov5su_{modality}" / "weights" / weights_str,
            Path("/content/drive/MyDrive/FYP/code/runs") / f"stage4_direct_{modality}" / f"yolov5su_{modality}" / "weights" / weights_str,
            Path("/content/runs") / f"stage4_direct_{modality}" / f"yolov5su_{modality}" / "weights" / weights_str,
            Path(weights_str),
            CODE_ROOT / "weights" / weights_str,
            Path("/content/drive/MyDrive/FYP/code/weights") / weights_str,
        ])

    # 3. If resuming or weights has 'last' in name, look for last checkpoint of this modality
    if resume or (weights_str and 'last' in weights_str.lower()):
        candidates.extend([
            drive_ckpt_dir / f"yolov5su_{modality}_last.pt",
            CODE_ROOT / "checkpoints" / f"yolov5su_{modality}_last.pt",
            runs_dir / f"yolov5su_{modality}" / "weights" / "last.pt",
            Path("/content/drive/MyDrive/FYP/code/runs") / f"stage4_direct_{modality}" / f"yolov5su_{modality}" / "weights" / "last.pt",
            Path("/content/runs") / f"stage4_direct_{modality}" / f"yolov5su_{modality}" / "weights" / "last.pt",
        ])

    # 4. Default pretrained fallback
    candidates.extend([
        CODE_ROOT / "weights" / "yolov5su.pt",
        Path("/content/drive/MyDrive/FYP/code/weights/yolov5su.pt"),
        Path("yolov5su.pt")
    ])

    for cand in candidates:
        if cand and cand.exists() and cand.stat().st_size > 0:
            return cand

    return Path(weights_str) if weights_str else Path("yolov5su.pt")


def is_training_completed(ckpt_path: Path, modality: str, runs_dir: Path, target_epochs: int) -> bool:
    """Checks if checkpoint has already finished all target epochs."""
    # 1. Check results.csv for completed epoch count
    csv_cands = [
        runs_dir / f"yolov5su_{modality}" / "results.csv",
        Path("/content/drive/MyDrive/FYP/code/runs") / f"stage4_direct_{modality}" / f"yolov5su_{modality}" / "results.csv",
        CODE_ROOT / "runs" / f"stage4_direct_{modality}" / f"yolov5su_{modality}" / "results.csv"
    ]
    for csv_p in csv_cands:
        if csv_p.exists():
            try:
                lines = [l for l in csv_p.read_text(encoding='utf-8').splitlines() if l.strip()]
                if len(lines) - 1 >= target_epochs:
                    return True
            except Exception:
                pass

    # 2. Check PyTorch checkpoint state
    try:
        import torch
        st = torch.load(str(ckpt_path), map_location='cpu')
        saved_epoch = st.get('epoch', -1)
        if saved_epoch >= target_epochs - 1:
            return True
        # If optimizer was stripped at end-of-training and best checkpoint exists, training was completed
        best_cand = ckpt_path.parent / f"yolov5su_{modality}_best.pt"
        if not best_cand.exists():
            best_cand = ckpt_path.parent / "best.pt"
        if ('optimizer' not in st or st.get('optimizer') is None) and best_cand.exists() and best_cand.stat().st_size > 0:
            return True
    except Exception:
        pass
    return False


def is_resumable_checkpoint(ckpt_path: Path) -> bool:
    """Checks if checkpoint contains optimizer state required for resume=True."""
    try:
        import torch
        st = torch.load(str(ckpt_path), map_location='cpu')
        return isinstance(st, dict) and st.get('optimizer') is not None
    except Exception:
        return False


def extract_log_summary_and_last_lr(modality: str, drive_ckpt_dir: Path, runs_dir: Path, ckpt_path: Path):
    """
    Inspects previous training logs (results.csv, telemetry JSON, or checkpoint state).
    Returns a comprehensive telemetry dictionary including:
      - total epochs previously completed
      - exact last learning rate before stopping
      - initial vs final loss values
      - peak validation metrics and peak epoch
      - human-readable formatted summary string
    """
    summary = {
        'found': False,
        'completed_epochs': 0,
        'last_lr': None,
        'best_map50': None,
        'best_epoch': None,
        'last_map50': None,
        'last_p': None,
        'last_r': None,
        'last_map50_95': None,
        'init_loss': {},
        'last_loss': {},
        'log_source': None
    }

    # 1. Search for results.csv using direct paths (instantaneous, zero network latency)
    csv_candidates = [
        drive_ckpt_dir / f"yolov5su_{modality}_results.csv",
        runs_dir / f"yolov5su_{modality}" / "results.csv",
        Path(f"/content/drive/MyDrive/FYP/code/runs/stage4_direct_{modality}/yolov5su_{modality}/results.csv"),
        Path(f"/content/drive/MyDrive/FYP/code/checkpoints/yolov5su_{modality}_results.csv"),
        Path(f"/content/runs/stage4_direct_{modality}/yolov5su_{modality}/results.csv"),
        CODE_ROOT / "runs" / f"stage4_direct_{modality}" / f"yolov5su_{modality}" / "results.csv",
        CODE_ROOT / "checkpoints" / f"yolov5su_{modality}_results.csv"
    ]
    shortcut_base = Path("/content/drive/.shortcut-targets-by-id")
    if shortcut_base.exists():
        try:
            for sc in shortcut_base.iterdir():
                cand = sc / f"FYP/code/runs/stage4_direct_{modality}/yolov5su_{modality}/results.csv"
                if cand.exists():
                    csv_candidates.append(cand)
        except Exception:
            pass

    csv_path = next((p for p in csv_candidates if p.exists() and p.stat().st_size > 0), None)

    if csv_path:
        try:
            lines = [l.strip() for l in csv_path.read_text(encoding='utf-8').splitlines() if l.strip()]
            if len(lines) > 1:
                header = [h.strip() for h in lines[0].split(',')]
                rows = []
                for line in lines[1:]:
                    vals = [v.strip() for v in line.split(',')]
                    if len(vals) == len(header):
                        rows.append(dict(zip(header, vals)))

                if rows:
                    summary['found'] = True
                    summary['completed_epochs'] = len(rows)
                    summary['log_source'] = str(csv_path)

                    r0 = rows[0]
                    summary['init_loss'] = {
                        'box': float(r0.get('train/box_loss', 0)),
                        'cls': float(r0.get('train/cls_loss', 0)),
                        'dfl': float(r0.get('train/dfl_loss', 0))
                    }

                    r_last = rows[-1]
                    summary['last_loss'] = {
                        'box': float(r_last.get('train/box_loss', 0)),
                        'cls': float(r_last.get('train/cls_loss', 0)),
                        'dfl': float(r_last.get('train/dfl_loss', 0))
                    }
                    summary['last_p'] = float(r_last.get('metrics/precision(B)', 0))
                    summary['last_r'] = float(r_last.get('metrics/recall(B)', 0))
                    summary['last_map50'] = float(r_last.get('metrics/mAP50(B)', 0))
                    summary['last_map50_95'] = float(r_last.get('metrics/mAP50-95(B)', 0))

                    lr_col = next((c for c in header if c.startswith('lr/pg0') or c == 'lr' or 'lr' in c.lower()), None)
                    if lr_col and r_last.get(lr_col):
                        try:
                            summary['last_lr'] = float(r_last[lr_col])
                        except ValueError:
                            pass

                    best_m = -1.0
                    best_ep = 1
                    for idx, r in enumerate(rows, 1):
                        m = float(r.get('metrics/mAP50(B)', 0))
                        if m > best_m:
                            best_m = m
                            best_ep = idx
                    summary['best_map50'] = best_m
                    summary['best_epoch'] = best_ep
        except Exception:
            pass

    # 2. Check telemetry json if available
    telemetry_path = drive_ckpt_dir / f"yolov5su_{modality}_telemetry.json"
    if telemetry_path.exists():
        try:
            with open(telemetry_path, 'r', encoding='utf-8') as f:
                t_data = json.load(f)
            if t_data.get('last_lr') and not summary['last_lr']:
                summary['last_lr'] = float(t_data['last_lr'])
            if t_data.get('current_epoch') and summary['completed_epochs'] == 0:
                summary['completed_epochs'] = int(t_data['current_epoch'])
                summary['found'] = True
                summary['log_source'] = str(telemetry_path)
        except Exception:
            pass

    # 3. Check PyTorch Checkpoint State
    if ckpt_path and ckpt_path.exists():
        try:
            import torch
            st = torch.load(str(ckpt_path), map_location='cpu')
            if isinstance(st, dict):
                saved_ep = st.get('epoch', -1)
                if saved_ep >= 0 and summary['completed_epochs'] == 0:
                    summary['completed_epochs'] = saved_ep + 1
                    summary['found'] = True
                    summary['log_source'] = f"Checkpoint Header ({ckpt_path.name})"

                opt = st.get('optimizer')
                if isinstance(opt, dict) and 'param_groups' in opt and len(opt['param_groups']) > 0:
                    opt_lr = opt['param_groups'][0].get('lr')
                    if opt_lr is not None and (summary['last_lr'] is None or summary['last_lr'] == 0):
                        summary['last_lr'] = float(opt_lr)
        except Exception:
            pass

    return summary


def print_previous_run_summary(summary: dict, modality: str):
    """Prints a clear, formatted console telemetry summary of the prior training run."""
    if not summary.get('found') or summary.get('completed_epochs', 0) == 0:
        return

    ep_done = summary['completed_epochs']
    best_m = summary.get('best_map50')
    best_ep = summary.get('best_epoch')
    last_lr = summary.get('last_lr')
    last_p = summary.get('last_p')
    last_r = summary.get('last_r')
    last_m50 = summary.get('last_map50')
    last_m95 = summary.get('last_map50_95')
    init_l = summary.get('init_loss', {})
    last_l = summary.get('last_loss', {})

    print("=" * 80)
    print(f" 📜 PREVIOUS TRAINING RUN LOG SUMMARY: {modality.upper()} BASELINE ".center(80))
    print("=" * 80)
    print(f"  Source Log File       : {summary.get('log_source')}")
    print(f"  Completed Epochs      : {ep_done}")
    if best_m is not None and best_m > 0:
        print(f"  Peak mAP@50 Reached   : {best_m * 100:.2f}% (Achieved at Epoch {best_ep})")
    if last_m50 is not None and last_m50 > 0:
        print(f"  Last Validated State  : P = {last_p*100:.1f}%, R = {last_r*100:.1f}%, mAP@50 = {last_m50*100:.2f}%, mAP@50-95 = {last_m95*100:.2f}%")
    if init_l and last_l and init_l.get('box', 0) > 0:
        b_diff = ((last_l['box'] - init_l['box']) / init_l['box']) * 100
        c_diff = ((last_l['cls'] - init_l['cls']) / init_l['cls']) * 100 if init_l.get('cls', 0) > 0 else 0
        print(f"  Loss Progress         : Box {init_l['box']:.3f}->{last_l['box']:.3f} ({b_diff:+.1f}%) | Cls {init_l['cls']:.3f}->{last_l['cls']:.3f} ({c_diff:+.1f}%)")
    if last_lr is not None:
        print(f"  Last Learning Rate    : {last_lr:.6f} (directly before stopping)")
    print("=" * 80 + "\n")


def train_unimodal(modality: str, epochs: int = 30, batch_size: int = 16, lr0: float = 0.001, freeze: int = 0,
                   weights: str = "yolov5su.pt", resume: bool = False, from_best: bool = False,
                   adaptive_lr: bool = True, force_rebuild: bool = False, force_train: bool = False):
    from ultralytics import YOLO

    m3fd_root = resolve_m3fd_root()
    base_dir = Path("/content") if Path("/content").exists() else CODE_ROOT / "runs"
    local_dataset_dir = base_dir / f"direct_{modality}_dataset"
    runs_dir = CODE_ROOT / "runs" / f"stage4_direct_{modality}"
    runs_dir.mkdir(parents=True, exist_ok=True)

    drive_ckpt_dir = Path("/content/drive/MyDrive/FYP/code/checkpoints")
    if not drive_ckpt_dir.exists():
        drive_ckpt_dir = CODE_ROOT / "checkpoints"
    drive_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Determine if resuming is requested with strict modality separation
    is_scratch = str(weights).lower() in ['scratch', 'yolov5su.pt']
    auto_resume = resume or (weights and 'last' in str(weights).lower())
    ckpt_path = resolve_checkpoint(weights, modality, drive_ckpt_dir, runs_dir, resume=auto_resume, from_best=from_best)

    # Inspect previous training logs and checkpoint state for summary & LR continuity
    log_summary = extract_log_summary_and_last_lr(modality, drive_ckpt_dir, runs_dir, ckpt_path)
    if log_summary.get('found') and (auto_resume or from_best or ckpt_path.name != 'yolov5su.pt'):
        print_previous_run_summary(log_summary, modality)

    # If from_best, do NOT treat as resume=True (best.pt has stripped optimizer); treat as fine-tuning with adaptive LR
    is_from_best = from_best or ('best' in ckpt_path.name.lower() and not is_scratch)
    if is_from_best:
        auto_resume = False

    # Check whether checkpoint is actually resumable (has optimizer state)
    has_optimizer = is_resumable_checkpoint(ckpt_path) if ckpt_path.exists() else False
    completed_ep = log_summary.get('completed_epochs', 0)

    # Distinguish between true resume (interrupted run) vs continuation (new epochs on top of checkpoint)
    is_continuation = False
    is_resuming = False
    if auto_resume and not is_from_best:
        if has_optimizer and completed_ep > 0 and epochs > completed_ep:
            # Resuming an interrupted run whose overall target was `epochs` (e.g. 60 target, interrupted at 26)
            is_resuming = True
        elif completed_ep > 0:
            # Checkpoint already completed `completed_ep` epochs. User wants `epochs` new epochs.
            is_continuation = True
            is_resuming = False
            print(f"  [Continuation Mode] 🎯 Checkpoint completed {completed_ep} epochs previously.")
            print(f"  [Continuation Mode] 🚀 Training for exactly {epochs} epochs starting from checkpoint weights.")
            # Archive prior results.csv so previous logs are safely preserved
            prior_csv = drive_ckpt_dir / f"yolov5su_{modality}_results.csv"
            archive_csv = drive_ckpt_dir / f"yolov5su_{modality}_results_prior{completed_ep}ep.csv"
            if prior_csv.exists() and not archive_csv.exists():
                try:
                    shutil.copy2(str(prior_csv), str(archive_csv))
                    print(f"  [Log Archive] 💾 Preserved prior {completed_ep}-epoch log to: {archive_csv.name}")
                except Exception:
                    pass
        elif has_optimizer:
            is_resuming = True

    # Check if this modality has already finished all epochs (only if starting fresh and target already reached)
    if not force_train and not is_from_best and not is_continuation and ckpt_path.exists():
        if completed_ep >= epochs and not auto_resume:
            print("=" * 80)
            print(f" 🚀 STAGE 4: DIRECT {modality.upper()}-ONLY YOLOv5su BASELINE ".center(80))
            print("=" * 80)
            print(f"  Input Modality        : Direct {modality.upper()}")
            print(f"  Checkpoint            : {ckpt_path}")
            print(f"  Status                : ✅ All {epochs} epochs ALREADY COMPLETED ({completed_ep} epochs found)!")
            print(f"  Action                : Skipping training. Pass --force_train or select checkpoint 'last' to train additional epochs.")
            print("=" * 80 + "\n")
            return None
        elif is_resuming and is_training_completed(ckpt_path, modality, runs_dir, epochs):
            print("=" * 80)
            print(f" 🚀 STAGE 4: DIRECT {modality.upper()}-ONLY YOLOv5su BASELINE ".center(80))
            print("=" * 80)
            print(f"  Input Modality        : Direct {modality.upper()}")
            print(f"  Checkpoint            : {ckpt_path}")
            print(f"  Status                : ✅ All {epochs} epochs ALREADY COMPLETED!")
            print(f"  Action                : Skipping training. Baseline ready for benchmarking.")
            print("=" * 80 + "\n")
            return None

    data_yaml = structure_unimodal_dataset(modality, local_dataset_dir, m3fd_root, force_rebuild=force_rebuild)

    # Clean any corrupted 0-byte label cache files
    for cfile in local_dataset_dir.glob("**/*.cache"):
        if cfile.stat().st_size == 0:
            cfile.unlink(missing_ok=True)

    # Adaptive Learning Rate & Continuity Configuration
    last_stop_lr = log_summary.get('last_lr')
    if is_from_best:
        lr0 = min(last_stop_lr or 0.0001, 0.0001)
        cos_lr = True
        warmup_epochs = 0.0
        lrf = 0.01
        sched_desc = f"Adaptive Fine-Tuning from Best (lr0={lr0:.6f}, terminal={lr0*lrf:.2e}, warmup=0.0)"
        print(f"  [LR Tuning] 🎯 Fine-tuning from best weights with gentle LR: lr0 = {lr0:.6f} (no warmup shock)")
    elif auto_resume or is_resuming:
        if last_stop_lr is not None and last_stop_lr > 0:
            # Match the exact learning rate directly before stopping
            lr0 = last_stop_lr
            target_min_lr = 1e-5
            lrf = max(target_min_lr / lr0, 0.001)
            warmup_epochs = 0.0
            cos_lr = True
            sched_desc = f"Direct Resumed Continuity (lr0={lr0:.6f} matching stop LR, terminal={lr0*lrf:.2e}, warmup=0.0)"
            print(f"  [LR Continuity] 🎯 Directly matching learning rate before stop: lr0 = {lr0:.6f} (no warmup shock)")
        elif log_summary.get('completed_epochs', 0) > 0:
            # Reconstruct decayed learning rate at stopped epoch
            prev_ep = log_summary['completed_epochs']
            pct = min(prev_ep / epochs, 1.0)
            factor = ((1 - math.cos(pct * math.pi)) / 2) * (0.01 - 1) + 1
            lr0 = max(0.001 * factor, 1e-5)
            lrf = max(1e-5 / lr0, 0.001)
            warmup_epochs = 0.0
            cos_lr = True
            sched_desc = f"Schedule-Matched Continuity (lr0={lr0:.6f} at ep {prev_ep}, terminal={lr0*lrf:.2e}, warmup=0.0)"
            print(f"  [LR Continuity] 🎯 Reconstructed decayed LR at stop epoch {prev_ep}: lr0 = {lr0:.6f}")
        else:
            cos_lr = True
            warmup_epochs = 0.0
            lrf = 0.01
            sched_desc = f"Standard Resumed Cosine (lr0={lr0}, warmup=0.0)"
    elif adaptive_lr:
        cos_lr = True
        warmup_epochs = 3.0
        lrf = 0.01
        sched_desc = f"Adaptive Cosine Annealing (lr0={lr0}, lrf={lr0*lrf}, warmup={warmup_epochs})"
    else:
        cos_lr = True
        warmup_epochs = 3.0
        lrf = 0.01
        sched_desc = f"Standard Cosine (lr0={lr0}, warmup={warmup_epochs})"

    if is_resuming:
        status_str = '🔄 RESUMING FROM INCOMPLETE CHECKPOINT'
    elif is_from_best:
        status_str = '🎯 FINE-TUNING FROM BEST CHECKPOINT (ADAPTIVE LR)'
    elif is_continuation:
        status_str = f'🎯 CONTINUATION RUN ({epochs} EPOCHS FROM CHECKPOINT)'
    elif auto_resume and not has_optimizer:
        status_str = '🎯 CONTINUING FROM COMPLETED CHECKPOINT'
    elif ckpt_path.exists() and ckpt_path.name != 'yolov5su.pt':
        status_str = f'STARTING FROM CHECKPOINT ({ckpt_path.name})'
    else:
        status_str = 'STARTING NEW FROM PRETRAINED YOLOv5su'

    print("=" * 80)
    print(f" 🚀 STAGE 4: DIRECT {modality.upper()}-ONLY YOLOv5su BASELINE TRAINING ".center(80))
    print("=" * 80)
    print(f"  Input Modality        : Direct {modality.upper()} ({'Thermal Infrared' if modality == 'ir' else 'Visible Optical'})")
    print(f"  Checkpoint / Weights  : {ckpt_path} ({status_str})")
    print(f"  Dataset Config        : {data_yaml}")
    print(f"  Total Epochs          : {epochs}")
    print(f"  Batch Size            : {batch_size}")
    print(f"  Backbone Freeze       : {'None (Full End-to-End Training)' if freeze == 0 else f'layers 0-{freeze-1} (Freeze = {freeze})'}")
    print(f"  Learning Rate (lr0)   : {lr0}")
    print(f"  LR Schedule           : {sched_desc}")
    print(f"  Checkpoint Mirror Dir : {drive_ckpt_dir}")
    print("=" * 80 + "\n")

    model = YOLO(str(ckpt_path))

    # Real-time synchronization and telemetry callback to Google Drive
    def sync_callback(trainer):
        try:
            t_weights = Path(trainer.save_dir) / "weights"
            # 1. Synchronize model weights
            for pt_name in ["best.pt", "last.pt"]:
                src_pt = t_weights / pt_name
                if src_pt.exists() and src_pt.stat().st_size > 0:
                    dest_pt = drive_ckpt_dir / f"yolov5su_{modality}_{pt_name}"
                    shutil.copy2(str(src_pt), str(dest_pt))

            # 2. Synchronize results.csv
            src_csv = Path(trainer.save_dir) / "results.csv"
            if src_csv.exists() and src_csv.stat().st_size > 0:
                dest_csv = drive_ckpt_dir / f"yolov5su_{modality}_results.csv"
                shutil.copy2(str(src_csv), str(dest_csv))

            # 3. Extract real-time telemetry from trainer
            ep_idx = getattr(trainer, 'epoch', -1) + 1
            tot_eps = getattr(trainer, 'epochs', epochs)
            curr_lr = None
            if hasattr(trainer, 'optimizer') and trainer.optimizer and trainer.optimizer.param_groups:
                curr_lr = trainer.optimizer.param_groups[0].get('lr')

            metrics_dict = getattr(trainer, 'metrics', {}) or {}
            val_p = metrics_dict.get('metrics/precision(B)', 0.0)
            val_r = metrics_dict.get('metrics/recall(B)', 0.0)
            val_m50 = metrics_dict.get('metrics/mAP50(B)', 0.0)
            val_m95 = metrics_dict.get('metrics/mAP50-95(B)', 0.0)

            t_loss = getattr(trainer, 'loss_items', None)
            box_l = float(t_loss[0]) if t_loss is not None and len(t_loss) > 0 else 0.0
            cls_l = float(t_loss[1]) if t_loss is not None and len(t_loss) > 1 else 0.0
            dfl_l = float(t_loss[2]) if t_loss is not None and len(t_loss) > 2 else 0.0

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 4. Append to human-readable training log
            log_file = drive_ckpt_dir / f"yolov5su_{modality}_training.log"
            lr_str = f"{curr_lr:.6f}" if curr_lr is not None else "N/A"
            log_line = (f"[{timestamp}] Epoch {ep_idx:02d}/{tot_eps:02d} | "
                        f"LR: {lr_str} | "
                        f"Loss(box={box_l:.3f}, cls={cls_l:.3f}, dfl={dfl_l:.3f}) | "
                        f"Val(P={val_p*100:.1f}%, R={val_r*100:.1f}%, mAP50={val_m50*100:.2f}%, mAP50-95={val_m95*100:.2f}%)\n")
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(log_line)

            # 5. Write structured telemetry JSON
            telemetry_file = drive_ckpt_dir / f"yolov5su_{modality}_telemetry.json"
            telemetry_data = {
                'modality': modality,
                'last_updated': timestamp,
                'current_epoch': ep_idx,
                'total_epochs': tot_eps,
                'last_lr': curr_lr,
                'latest_metrics': {
                    'precision': float(val_p),
                    'recall': float(val_r),
                    'mAP50': float(val_m50),
                    'mAP50-95': float(val_m95)
                },
                'latest_loss': {
                    'box': box_l,
                    'cls': cls_l,
                    'dfl': dfl_l
                }
            }
            with open(telemetry_file, "w", encoding="utf-8") as jf:
                json.dump(telemetry_data, jf, indent=2)
        except Exception:
            pass

    model.add_callback("on_fit_epoch_end", sync_callback)

    train_kwargs = {
        'data': str(data_yaml),
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': 640,
        'project': str(runs_dir),
        'name': f"yolov5su_{modality}",
        'exist_ok': True,
        'lr0': lr0,
        'lrf': lrf,
        'cos_lr': cos_lr,
        'warmup_epochs': warmup_epochs,
        'optimizer': 'AdamW',
        'amp': False,
        'save': True,
        'val': True,
        'verbose': True
    }

    if is_resuming:
        train_kwargs['resume'] = True
    else:
        train_kwargs['freeze'] = freeze

    results = model.train(**train_kwargs)

    # Save final named checkpoints
    final_best = runs_dir / f"yolov5su_{modality}" / "weights" / "best.pt"
    if final_best.exists():
        shutil.copy2(str(final_best), str(drive_ckpt_dir / f"yolov5su_{modality}_best.pt"))
        print(f"[Stage 4 Complete] ✅ Final checkpoint synchronized: {drive_ckpt_dir / f'yolov5su_{modality}_best.pt'}")

    return results


def parse_checkpoint_selection(choice_str: str, modality: str):
    """
    Parses a checkpoint selection string into (effective_weights, from_best, resume).
    Accepts:
      - 'best', 'best.pt', 'yolov5su_ir_best.pt', 'best (yolov5su_ir_best.pt - Adaptive LR)'
      - 'last', 'last.pt', 'yolov5su_ir_last.pt', 'last (yolov5su_ir_last.pt - Resume)'
      - 'scratch', 'yolov5su.pt', 'scratch (yolov5su.pt)'
      - custom file paths
    """
    if not choice_str:
        return None, False, False
    c_lower = choice_str.strip().lower()
    if 'best' in c_lower:
        w = f"yolov5su_{modality}_best.pt" if (c_lower in ['best', 'best.pt'] or 'yolov5su' not in c_lower) else choice_str
        return w, True, False
    elif 'last' in c_lower:
        w = f"yolov5su_{modality}_last.pt" if (c_lower in ['last', 'last.pt'] or 'yolov5su' not in c_lower) else choice_str
        return w, False, True
    elif 'scratch' in c_lower or 'yolo' in c_lower:
        return "yolov5su.pt", False, False
    else:
        return choice_str, False, False


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Direct Unimodal Training (IR or RGB)")
    parser.add_argument('--modality', type=str, required=True, choices=['ir', 'vi', 'rgb', 'both'],
                        help="Modality to train: 'ir', 'rgb' ('vi'), or 'both'")
    parser.add_argument('--epochs', type=int, default=30, help="Total training epochs (default: 30)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument('--freeze', type=int, default=0, help="Layers to freeze in YOLO backbone (default: 0 for full end-to-end training)")
    parser.add_argument('--weights', type=str, default='yolov5su.pt', help="General pretrained weights or checkpoint name (default: yolov5su.pt)")
    
    # Dedicated Per-Model Checkpoint Selection
    parser.add_argument('--ir_checkpoint', type=str, default=None,
                        help="Checkpoint selection for IR: 'best', 'last', 'scratch', or path/filename")
    parser.add_argument('--rgb_checkpoint', type=str, default=None,
                        help="Checkpoint selection for RGB: 'best', 'last', 'scratch', or path/filename")
    parser.add_argument('--ir_weights', type=str, default=None, help="Alias for --ir_checkpoint")
    parser.add_argument('--rgb_weights', type=str, default=None, help="Alias for --rgb_checkpoint")
    
    # Global flags
    parser.add_argument('--from_best', action='store_true', help="Resume/fine-tune from best checkpoints using adaptive learning rate")
    parser.add_argument('--resume', action='store_true', help="Resume training from last saved checkpoint")
    parser.add_argument('--force_train', action='store_true', help="Force training even if epochs were previously reached")
    parser.add_argument('--force_rebuild', action='store_true', help="Force rebuild local SSD dataset cache")
    args = parser.parse_args()

    mod = args.modality.lower()
    if mod == 'vi':
        mod = 'rgb'

    # Resolve IR checkpoint settings
    ir_sel = args.ir_checkpoint or args.ir_weights
    if ir_sel:
        ir_w, ir_from_best, ir_resume = parse_checkpoint_selection(ir_sel, 'ir')
    else:
        ir_w, ir_from_best, ir_resume = args.weights, args.from_best, args.resume

    # Resolve RGB checkpoint settings
    rgb_sel = args.rgb_checkpoint or args.rgb_weights
    if rgb_sel:
        rgb_w, rgb_from_best, rgb_resume = parse_checkpoint_selection(rgb_sel, 'rgb')
    else:
        rgb_w, rgb_from_best, rgb_resume = args.weights, args.from_best, args.resume

    if mod == 'ir':
        train_unimodal('ir', epochs=args.epochs, batch_size=args.batch_size, lr0=args.lr, freeze=args.freeze,
                       weights=ir_w, resume=ir_resume, from_best=ir_from_best,
                       force_rebuild=args.force_rebuild, force_train=args.force_train)
    elif mod == 'rgb':
        train_unimodal('rgb', epochs=args.epochs, batch_size=args.batch_size, lr0=args.lr, freeze=args.freeze,
                       weights=rgb_w, resume=rgb_resume, from_best=rgb_from_best,
                       force_rebuild=args.force_rebuild, force_train=args.force_train)
    elif mod == 'both':
        print("\n" + "=" * 80)
        print(" PHASE 1/2: DIRECT IR BASELINE TRAINING ".center(80, "="))
        print("=" * 80)
        train_unimodal('ir', epochs=args.epochs, batch_size=args.batch_size, lr0=args.lr, freeze=args.freeze,
                       weights=ir_w, resume=ir_resume, from_best=ir_from_best,
                       force_rebuild=args.force_rebuild, force_train=args.force_train)

        print("\n" + "=" * 80)
        print(" PHASE 2/2: DIRECT RGB BASELINE TRAINING ".center(80, "="))
        print("=" * 80)
        train_unimodal('rgb', epochs=args.epochs, batch_size=args.batch_size, lr0=args.lr, freeze=args.freeze,
                       weights=rgb_w, resume=rgb_resume, from_best=rgb_from_best,
                       force_rebuild=args.force_rebuild, force_train=args.force_train)


if __name__ == '__main__':
    main()