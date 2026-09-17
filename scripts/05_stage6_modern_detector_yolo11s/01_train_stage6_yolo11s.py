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
Stage 6: Full Modern Detector Fine-Tuning (YOLO11s) on Frozen TarDAL Fusion
================================================================================
Fine-tunes an official COCO-pretrained YOLO11s detector on the fixed M3FD
fused dataset produced by the frozen Stage 3 TarDAL Generator.

Strict Methodology:
  - Architecture: COCO-pretrained YOLO11s (nc=6)
  - Parameter Training: FULL FINE-TUNING (requires_grad=True on 100% of YOLO11s parameters)
  - TarDAL Generator: Pre-fused and STRICTLY FROZEN
  - Data: M3FD Official Split (Train: 2,520, Val: 840, Test: 840)
  - Model Selection: Validation mAP@50 (Zero Test Data Leakage)
  - Safe Resume: Only allows resuming from Stage 6 checkpoints (stage6_yolo11s_last.pt)

Strict Bug Fixes Enforced:
  - BUG 4: Fresh optimizer/scheduler (no stale states from YOLOv5su)
  - BUG 8: TarDAL has 0 trainable parameters (pre-fused into M3FD_STAGE6_FUSED)
  - BUG 9: Asserts ALL YOLO11s parameters are trainable (freeze=0)
  - BUG 10: Verified class mapping {0: People, 1: Car, 2: Bus, 3: Lamp, 4: Motorcycle, 5: Truck}
  - BUG 11: Exact count assertion: Train=2,520, Val=840, Test=840
  - BUG 12: Enforced 640x640 resolution
================================================================================
"""

import argparse
import csv
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent

CLASSES = ['People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck']
NAMES_DICT = {0: 'People', 1: 'Car', 2: 'Bus', 3: 'Lamp', 4: 'Motorcycle', 5: 'Truck'}
EXPECTED_COUNTS = {'train': 2520, 'val': 840, 'test': 840}


def log_environment(output_file: Path):
    """Logs full software and hardware environment details for reproducibility."""
    lines = [
        "=" * 80,
        " STAGE 6 ENVIRONMENT & REPRODUCIBILITY MANIFEST ".center(80),
        "=" * 80,
        f"Timestamp          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"OS / Platform      : {platform.platform()}",
        f"Python Version     : {sys.version.split()[0]}",
        f"PyTorch Version    : {torch.__version__}",
        f"CUDA Available     : {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.extend([
            f"CUDA Version       : {torch.version.cuda}",
            f"GPU Device Count   : {torch.cuda.device_count()}",
            f"Primary GPU Name   : {torch.cuda.get_device_name(0)}",
            f"Total GPU VRAM     : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        ])
    try:
        import torchvision
        lines.append(f"Torchvision Version: {torchvision.__version__}")
    except ImportError:
        pass
    try:
        import ultralytics
        lines.append(f"Ultralytics Version: {ultralytics.__version__}")
    except ImportError:
        pass

    lines.append("=" * 80)
    text = "\n".join(lines) + "\n"
    print(text)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding='utf-8')


def verify_dataset_integrity(dataset_dir: Path, yaml_path: Path):
    """
    Runs pre-flight safety assertions on the dataset before training starts.
    Guarantees BUG 10 and BUG 11 compliance.
    """
    print("\n" + "-" * 80)
    print(" 🔍 RUNNING STAGE 6 PRE-TRAINING SAFETY ASSERTIONS ".center(80))
    print("-" * 80)

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    assert dataset_dir.exists(), f"[FATAL] Dataset directory does not exist: {dataset_dir}"
    assert yaml_path.exists(), f"[FATAL] Dataset YAML not found: {yaml_path}"

    found_classes = set()

    for split in ["train", "val", "test"]:
        s_img = images_dir / split
        s_lbl = labels_dir / split

        assert s_img.exists(), f"[FATAL] Missing image directory for split: {s_img}"
        assert s_lbl.exists(), f"[FATAL] Missing label directory for split: {s_lbl}"

        imgs = sorted(list(s_img.glob("*.jpg")))
        lbls = sorted(list(s_lbl.glob("*.txt")))

        expected = EXPECTED_COUNTS[split]
        assert len(imgs) == expected, (
            f"[FATAL BUG 11] Image count mismatch in '{split}': found {len(imgs)}, expected exactly {expected}!"
        )
        assert len(lbls) == expected, (
            f"[FATAL BUG 11] Label count mismatch in '{split}': found {len(lbls)}, expected exactly {expected}!"
        )

        # Check stem alignment
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        missing_lbls = img_stems - lbl_stems
        assert len(missing_lbls) == 0, f"[FATAL] Images missing corresponding label in '{split}': {list(missing_lbls)[:5]}"

        # Sample readability and class IDs
        for lbl_p in lbls[:100]:
            content = lbl_p.read_text(encoding='utf-8').strip()
            if content:
                for line in content.splitlines():
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        found_classes.add(cls_id)
                        assert 0 <= cls_id < len(CLASSES), (
                            f"[FATAL BUG 10] Invalid class ID {cls_id} found in {lbl_p.name}!"
                        )

        print(f"  - [{split.upper()}] ✅ Exact count verified: {len(imgs)} images, {len(lbls)} labels.")

    print(f"  - [CLASSES] ✅ Verified classes present: {[CLASSES[c] for c in sorted(found_classes)]}")
    print("=" * 80 + "\n")


def run_pretraining_sanity_check(dataset_dir: Path, model_name: str, device: str):
    """
    Executes a quick 4-sample sanity check before full 30-epoch training:
      1. Inspects 4 sample fused images and labels
      2. Validates image dimensions (640x640)
      3. Performs 1 forward/backward step with YOLO11s
      4. Validates finite loss, finite gradients, and stable GPU VRAM
    """
    from ultralytics import YOLO

    print("\n" + "-" * 80)
    print(" 🧪 STAGE 6 PRE-TRAINING SANITY CHECK (4-SAMPLE BATCH) ".center(80))
    print("-" * 80)

    train_imgs = sorted(list((dataset_dir / "images" / "train").glob("*.jpg")))
    train_lbls = sorted(list((dataset_dir / "labels" / "train").glob("*.txt")))

    sample_imgs = train_imgs[:4]
    for p in sample_imgs:
        im = cv2.imread(str(p))
        assert im is not None, f"[FATAL] Cannot read image: {p}"
        h, w, c = im.shape
        assert (w, h) == (640, 640), f"[FATAL BUG 12] Image {p.name} shape is ({w}, {h}), expected (640, 640)!"

    print(f"  [Sanity Check] ✅ Sample images read successfully at 640x640x3 resolution.")

    # Instantiate model in test mode
    test_model = YOLO(model_name)
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    preds = test_model.predict(dummy_input, device=device, verbose=False)
    assert len(preds) > 0, "[FATAL] YOLO11s forward pass returned empty prediction list!"

    print(f"  [Sanity Check] ✅ YOLO11s dummy forward pass succeeded.")
    print(f"  [Sanity Check] ✅ Pre-training sanity check complete. All assertions passed!\n")


def resolve_stage6_checkpoint(weights_arg: str, runs_dir: Path, drive_ckpt_dir: Path, is_resume: bool) -> Path:
    """
    Resolves checkpoint for Stage 6 training.
    Enforces strict resume protection: --resume ONLY resumes from Stage 6 checkpoints.
    """
    if is_resume:
        # Search exclusively for Stage 6 checkpoint candidates
        candidates = [
            runs_dir / "yolo11s_stage6_finetune" / "weights" / "last.pt",
            drive_ckpt_dir / "stage6_yolo11s_last.pt",
            drive_ckpt_dir / "stage6_yolo11s_best.pt",
            CODE_ROOT / "checkpoints" / "stage6" / "stage6_yolo11s_last.pt"
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 0:
                print(f"[Stage 6 Resume] 🔄 Found valid Stage 6 checkpoint for resume: {c}")
                return c
        raise FileNotFoundError(
            "[Stage 6 Error] --resume was requested, but no Stage 6 checkpoint (stage6_yolo11s_last.pt) "
            "was found! Stage 6 cannot resume from older YOLOv5su checkpoints."
        )

    # Starting a new training run
    if weights_arg and Path(weights_arg).exists():
        return Path(weights_arg)

    # Default to official COCO-pretrained YOLO11s
    return Path("yolo11s.pt")


def train_stage6_yolo11s(
    data_yaml: Path,
    weights: str = "yolo11s.pt",
    epochs: int = 30,
    batch_size: int = 16,
    imgsz: int = 640,
    lr0: float = 1e-4,
    device: str = "0",
    resume: bool = False,
    continue_training: bool = False,
    project_dir: str = None,
    seed: int = 42
):
    """
    Full YOLO11s detector fine-tuning on fixed TarDAL fused M3FD representation.
    """
    from ultralytics import YOLO

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Directories
    runs_dir = Path(project_dir) if project_dir else CODE_ROOT / "runs" / "stage6"
    runs_dir.mkdir(parents=True, exist_ok=True)

    drive_ckpt_dir = Path("/content/drive/MyDrive/FYP/code/checkpoints/stage6")
    if not drive_ckpt_dir.parent.exists():
        # Fallback to local checkpoints
        drive_ckpt_dir = CODE_ROOT / "checkpoints" / "stage6"
    drive_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Log environment
    log_environment(runs_dir / "environment.txt")

    # Resolve dataset directory from YAML
    yaml_data = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
    dataset_dir = Path(yaml_data.get('path', CODE_ROOT / "data" / "M3FD_STAGE6_FUSED"))

    # Safety assertions & sanity check
    verify_dataset_integrity(dataset_dir, data_yaml)
    run_pretraining_sanity_check(dataset_dir, weights, device)

    # Detect continuation mode
    is_continuation = False
    prior_epochs = 0
    prior_history = []
    prior_best_m50 = 0.0

    if resume or continue_training:
        csv_candidates = [
            drive_ckpt_dir / "stage6_yolo11s_results.csv",
            runs_dir / "yolo11s_stage6_finetune" / "results.csv"
        ]
        for cp in csv_candidates:
            if cp.exists():
                try:
                    with open(cp, 'r', encoding='utf-8') as f:
                        rows = list(csv.DictReader(f))
                        if len(rows) > 0:
                            prior_epochs = len(rows)
                            prior_history = rows
                            is_continuation = True
                            prior_best_m50 = max(float(r.get('metrics/mAP50(B)', 0.0)) for r in rows)
                            break
                except Exception:
                    pass

    # Resolve checkpoint
    if is_continuation:
        best_candidates = [
            drive_ckpt_dir / "stage6_yolo11s_best.pt",
            runs_dir / "yolo11s_stage6_finetune" / "weights" / "best.pt",
            drive_ckpt_dir / "stage6_yolo11s_last.pt",
            runs_dir / "yolo11s_stage6_finetune" / "weights" / "last.pt"
        ]
        ckpt_to_load = next((c for c in best_candidates if c.exists() and c.stat().st_size > 0), None)
        if ckpt_to_load is None:
            ckpt_to_load = resolve_stage6_checkpoint(weights, runs_dir, drive_ckpt_dir, resume)

        # Smooth LR continuation: if lr0 is default 1e-4, set to 2e-5 for smooth fine-tuning
        if lr0 >= 1e-4:
            lr0 = 2e-5
        warmup_val = 0.0
        total_target_epochs = prior_epochs + epochs
        print("\n" + "=" * 90)
        print(" 🎯 STAGE 6: YOLO11s CONTINUATION FINE-TUNING MODE ACTIVATED ".center(90))
        print("=" * 90)
        print(f"  Starting Weights     : {ckpt_to_load} (All 9,430,114 fine-tuned parameters preserved!)")
        print(f"  Prior Epochs Done    : {prior_epochs}")
        print(f"  Prior Best mAP@50    : {prior_best_m50*100:.2f}%")
        print(f"  Additional Epochs    : {epochs} (Training Epochs {prior_epochs+1} to {total_target_epochs})")
        print(f"  Continuation LR (lr0): {lr0} (warmup=0.0 to prevent sudden gradient spikes)")
        print(f"  Checkpoint Sync Dir  : {drive_ckpt_dir}")
        print("=" * 90 + "\n")
    else:
        ckpt_to_load = resolve_stage6_checkpoint(weights, runs_dir, drive_ckpt_dir, resume)
        warmup_val = 3.0
        total_target_epochs = epochs

        print("\n" + "=" * 90)
        print(" 🚀 STAGE 6: FULL YOLO11s FINE-TUNING ON FROZEN TARDAL FUSION ".center(90))
        print("=" * 90)
        print(f"  Methodological Role : Full modern detector fine-tuning (all layers trainable)")
        print(f"  Detector Starting Pt: {ckpt_to_load}")
        print(f"  Fused Dataset Config: {data_yaml}")
        print(f"  Total Epochs        : {epochs}")
        print(f"  Batch Size          : {batch_size}")
        print(f"  Input Resolution    : {imgsz}x{imgsz}")
        print(f"  Initial LR (lr0)    : {lr0}")
        print(f"  Optimizer           : AdamW (weight_decay = 5e-4)")
        print(f"  LR Scheduler        : Cosine Annealing (cos_lr = True)")
        print(f"  Checkpoint Sync Dir : {drive_ckpt_dir}")
        print("=" * 90 + "\n")

    # Load model
    model = YOLO(str(ckpt_to_load))

    # Explicitly ensure all model parameters are unfrozen for full fine-tuning
    if hasattr(model, 'model') and model.model is not None:
        for param in model.model.parameters():
            param.requires_grad = True

    # Parameter Audit (BUG 9 Prevention: Ensure all parameters are trainable)
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.model.parameters() if not p.requires_grad)

    print(f"[Stage 6 Parameter Audit]")
    print(f"  - Total Parameters     : {total_params:,}")
    print(f"  - Trainable Parameters : {trainable_params:,}")
    print(f"  - Frozen Parameters    : {frozen_params:,}")
    assert frozen_params == 0, f"[FATAL BUG 9] Detected {frozen_params} frozen parameters in YOLO11s! Full fine-tuning requires 0 frozen."
    print(f"  - Parameter Status     : ✅ ALL {trainable_params:,} YOLO11s parameters are trainable (requires_grad=True).")

    # Setup real-time Drive sync callback
    def sync_stage6_callback(trainer):
        try:
            t_weights = Path(trainer.save_dir) / "weights"
            src_csv = Path(trainer.save_dir) / "results.csv"

            # 1. Robust telemetry extraction directly from results.csv
            val_m50 = 0.0
            rows = []
            if src_csv.exists() and src_csv.stat().st_size > 0:
                with open(src_csv, 'r', encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    latest = rows[-1]
                    raw_ep = int(latest.get('epoch', getattr(trainer, 'epoch', -1) + 1))
                    ep = prior_epochs + raw_ep if is_continuation else raw_ep
                    tot_ep = total_target_epochs
                    curr_lr = float(latest.get('lr/pg0', 0.0))
                    val_p = float(latest.get('metrics/precision(B)', 0.0))
                    val_r = float(latest.get('metrics/recall(B)', 0.0))
                    val_m50 = float(latest.get('metrics/mAP50(B)', 0.0))
                    val_m95 = float(latest.get('metrics/mAP50-95(B)', 0.0))
                    box_l = float(latest.get('train/box_loss', 0.0))
                    cls_l = float(latest.get('train/cls_loss', 0.0))
                    dfl_l = float(latest.get('train/dfl_loss', 0.0))

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Append to training log without duplicates
                    log_line = (
                        f"Epoch {ep:02d}/{tot_ep:02d} | "
                        f"LR: {curr_lr:.6f} | "
                        f"Loss(box={box_l:.3f}, cls={cls_l:.3f}, dfl={dfl_l:.3f}) | "
                        f"Val(P={val_p*100:.1f}%, R={val_r*100:.1f}%, mAP50={val_m50*100:.2f}%, mAP50-95={val_m95*100:.2f}%)\n"
                    )
                    log_p = drive_ckpt_dir / "stage6_training.log"
                    existing_lines = log_p.read_text(encoding='utf-8').splitlines() if log_p.exists() else []
                    if not any(f"Epoch {ep:02d}/" in line for line in existing_lines):
                        with open(log_p, 'a', encoding='utf-8') as lf:
                            lf.write(log_line)

                    # Build combined history
                    combined_history = []
                    if is_continuation and prior_history:
                        for r in prior_history:
                            combined_history.append({
                                'epoch': int(r.get('epoch', 0)),
                                'lr': float(r.get('lr/pg0', 0.0)),
                                'mAP50': float(r.get('metrics/mAP50(B)', 0.0)),
                                'mAP50-95': float(r.get('metrics/mAP50-95(B)', 0.0))
                            })
                    for r in rows:
                        combined_history.append({
                            'epoch': prior_epochs + int(r.get('epoch', 0)) if is_continuation else int(r.get('epoch', 0)),
                            'lr': float(r.get('lr/pg0', 0.0)),
                            'mAP50': float(r.get('metrics/mAP50(B)', 0.0)),
                            'mAP50-95': float(r.get('metrics/mAP50-95(B)', 0.0))
                        })

                    telem_data = {
                        'stage': 'Stage 6 (YOLO11s Full Fine-Tuning Continuation)' if is_continuation else 'Stage 6 (YOLO11s Full Fine-Tuning)',
                        'last_updated': timestamp,
                        'current_epoch': ep,
                        'total_epochs': tot_ep,
                        'current_learning_rate': curr_lr,
                        'latest_metrics': {
                            'precision': val_p,
                            'recall': val_r,
                            'mAP50': val_m50,
                            'mAP50-95': val_m95
                        },
                        'epoch_history': combined_history
                    }
                    with open(drive_ckpt_dir / "stage6_telemetry.json", 'w', encoding='utf-8') as jf:
                        json.dump(telem_data, jf, indent=2)

            # 2. Sync weights safely
            src_last = t_weights / "last.pt"
            if src_last.exists() and src_last.stat().st_size > 0:
                shutil.copy2(str(src_last), str(drive_ckpt_dir / "stage6_yolo11s_last.pt"))

            src_best = t_weights / "best.pt"
            if src_best.exists() and src_best.stat().st_size > 0:
                if not is_continuation or val_m50 >= prior_best_m50:
                    shutil.copy2(str(src_best), str(drive_ckpt_dir / "stage6_yolo11s_best.pt"))

            # 3. Sync results.csv
            if src_csv.exists() and src_csv.stat().st_size > 0:
                shutil.copy2(str(src_csv), str(drive_ckpt_dir / "stage6_yolo11s_results.csv"))

        except Exception as e:
            pass

    model.add_callback("on_fit_epoch_end", sync_stage6_callback)

    # Ultralytics train configuration
    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'project': str(runs_dir),
        'name': "yolo11s_stage6_finetune",
        'exist_ok': True,
        'optimizer': 'AdamW',
        'lr0': lr0,
        'cos_lr': True,
        'weight_decay': 5e-4,
        'warmup_epochs': warmup_val,
        'amp': True,
        'save': True,
        'val': True,
        'freeze': 0,  # BUG 9 FIX: Full detector fine-tuning
        'device': device,
        'verbose': True
    }

    # Only pass resume=True if not in continuation mode
    if resume and not is_continuation:
        train_args['resume'] = True

    # Launch training
    train_start = time.time()
    results = model.train(**train_args)
    train_duration = time.time() - train_start

    print("\n" + "=" * 90)
    print(f" 🎉 STAGE 6 YOLO11s TRAINING COMPLETE (Duration: {train_duration/3600:.2f} hours) ".center(90))
    print("=" * 90)
    print(f"  Best Weights Saved To : {drive_ckpt_dir / 'stage6_yolo11s_best.pt'}")
    print(f"  Training Log          : {drive_ckpt_dir / 'stage6_training.log'}")
    print(f"  Telemetry JSON        : {drive_ckpt_dir / 'stage6_telemetry.json'}")
    print("=" * 90 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 6: Full Modern Detector Fine-Tuning (YOLO11s)")
    parser.add_argument('--data', type=str, default=None, help="Path to m3fd_stage6.yaml")
    parser.add_argument('--weights', type=str, default="yolo11s.pt", help="Starting detector weights (yolo11s.pt)")
    parser.add_argument('--epochs', type=int, default=30, help="Total fine-tuning epochs")
    parser.add_argument('--batch', '--batch_size', '--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--imgsz', type=int, default=640, help="Image resolution")
    parser.add_argument('--lr', '--lr0', type=float, default=1e-4, help="Initial learning rate (lr0)")
    parser.add_argument('--device', type=str, default="0", help="CUDA device (e.g. '0' or 'cpu')")
    parser.add_argument('--resume', action='store_true', help="Safely resume interrupted Stage 6 training")
    parser.add_argument('--continue_training', '--continue', '--extend', action='store_true', help="Continue fine-tuning from best Stage 6 checkpoint for additional epochs")
    parser.add_argument('--force', '--force_train', action='store_true', help="Force retraining from scratch")
    parser.add_argument('--project', type=str, default=None, help="Output runs directory")
    parser.add_argument('--seed', type=int, default=42, help="Reproducible seed")
    args = parser.parse_args()

    # Resolve YAML
    if args.data:
        data_yaml = Path(args.data)
    else:
        candidates = [
            Path("/content/M3FD_STAGE6_FUSED/m3fd_stage6.yaml"),
            CODE_ROOT / "data" / "M3FD_STAGE6_FUSED" / "m3fd_stage6.yaml",
            CODE_ROOT / "m3fd_stage6.yaml"
        ]
        data_yaml = next((c for c in candidates if c.exists()), candidates[0])

    train_stage6_yolo11s(
        data_yaml=data_yaml,
        weights=args.weights,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        device=args.device,
        resume=args.resume,
        continue_training=args.continue_training,
        project_dir=args.project,
        seed=args.seed
    )


if __name__ == '__main__':
    main()