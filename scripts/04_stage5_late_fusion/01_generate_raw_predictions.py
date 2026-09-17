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
Stage 5: RGB + IR Raw Prediction Generator & Cache Engine
================================================================================
Generates and caches raw bounding box detections [xyxy, conf, class_id]
and ground truth annotations for both unimodal YOLOv5su models across
the official M3FD dataset splits ('val' or 'test').

Enables downstream late fusion, grid search optimization, and sensitivity
analysis to run in seconds without repeated neural network inference.
================================================================================
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parent.parent

CLASSES = ['People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck']
NAMES_DICT = {0: 'People', 1: 'Car', 2: 'Bus', 3: 'Lamp', 4: 'Motorcycle', 5: 'Truck'}


def resolve_ckpt(candidates):
    for c in candidates:
        if c and Path(c).exists() and Path(c).stat().st_size > 0:
            return Path(c)
    return None


def resolve_m3fd_root() -> Path:
    candidates = [
        Path("/content/m3fd/M3FD_Detection"),
        Path("/content/m3fd"),
        CODE_ROOT / "data" / "m3fd",
        CODE_ROOT / "data" / "M3FD_Detection"
    ]
    for c in candidates:
        if c.exists() and (c / "meta").exists():
            return c
    return candidates[0]


def load_ground_truth(lbl_path: Path, img_w: int, img_h: int):
    """Loads YOLO format labels (cls, cx, cy, w, h) and converts to pixel xyxy."""
    boxes = []
    classes = []
    if lbl_path and lbl_path.exists():
        try:
            lines = [l.strip() for l in lbl_path.read_text(encoding='utf-8').splitlines() if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx = float(parts[1]) * img_w
                    cy = float(parts[2]) * img_h
                    bw = float(parts[3]) * img_w
                    bh = float(parts[4]) * img_h
                    x1 = max(0.0, cx - bw / 2.0)
                    y1 = max(0.0, cy - bh / 2.0)
                    x2 = min(float(img_w), cx + bw / 2.0)
                    y2 = min(float(img_h), cy + bh / 2.0)
                    boxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
                    classes.append(cls_id)
        except Exception:
            pass
    return boxes, classes


def extract_predictions_for_modality(model, img_paths, batch_size=16, conf=0.001, iou=0.65, device="cuda"):
    """Runs batch inference and extracts [xyxy, conf, class] per image."""
    results_by_path = {}
    
    total = len(img_paths)
    for i in range(0, total, batch_size):
        batch_paths = img_paths[i:i + batch_size]
        str_paths = [str(p) for p in batch_paths]
        preds = model.predict(
            str_paths,
            imgsz=640,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False
        )
        for p, res in zip(batch_paths, preds):
            stem = p.stem
            boxes = []
            scores = []
            classes = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for b, s, c in zip(xyxy, confs, clss):
                    boxes.append([round(float(coord), 2) for coord in b])
                    scores.append(round(float(s), 4))
                    classes.append(int(c))
            
            orig_shape = res.orig_shape # (h, w)
            results_by_path[stem] = {
                'boxes': boxes,
                'scores': scores,
                'classes': classes,
                'orig_shape': [int(orig_shape[0]), int(orig_shape[1])]
            }
    return results_by_path


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Generate Raw Predictions for Late Fusion")
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'both'], help="Dataset split")
    parser.add_argument('--rgb_ckpt', type=str, default=None, help="Path to Direct RGB model checkpoint")
    parser.add_argument('--ir_ckpt', type=str, default=None, help="Path to Direct IR model checkpoint")
    parser.add_argument('--conf', type=float, default=0.001, help="Confidence threshold for raw box extraction")
    parser.add_argument('--iou', type=float, default=0.65, help="NMS IoU threshold for raw box extraction")
    parser.add_argument('--batch_size', type=int, default=16, help="Inference batch size")
    parser.add_argument('--force', action='store_true', help="Force re-generation even if cache exists")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_dir = Path("/content/drive/MyDrive/FYP/code/checkpoints") if Path("/content").exists() else CODE_ROOT / "checkpoints"

    rgb_ckpt = resolve_ckpt([
        args.rgb_ckpt,
        ckpt_dir / "yolov5su_rgb_best.pt",
        CODE_ROOT / "runs" / "stage4_direct_rgb" / "yolov5su_rgb" / "weights" / "best.pt"
    ])

    ir_ckpt = resolve_ckpt([
        args.ir_ckpt,
        ckpt_dir / "yolov5su_ir_best.pt",
        CODE_ROOT / "runs" / "stage4_direct_ir" / "yolov5su_ir" / "weights" / "best.pt"
    ])

    if not rgb_ckpt or not ir_ckpt:
        print(f"[Error] Checkpoints not found! RGB: {rgb_ckpt}, IR: {ir_ckpt}")
        sys.exit(1)

    m3fd_root = resolve_m3fd_root()
    meta_dir = next((m3fd_root / d for d in ["meta", "Meta", "splits", "Splits"] if (m3fd_root / d).exists()), m3fd_root / "meta")
    ir_dir = next((m3fd_root / d for d in ["Ir", "ir", "IR", "thermal", "Thermal"] if (m3fd_root / d).exists()), m3fd_root / "ir")
    vi_dir = next((m3fd_root / d for d in ["Vis", "vis", "VIS", "vi", "Vi", "VI"] if (m3fd_root / d).exists()), m3fd_root / "vi")
    lbl_dir = next((m3fd_root / d for d in ["labels", "Labels", "labels_yolo", "annotations"] if (m3fd_root / d).exists()), m3fd_root / "labels")

    splits_to_process = ['val', 'test'] if args.split == 'both' else [args.split]

    out_dir = CODE_ROOT / "runs" / "stage5_late_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    drive_out_dir = Path("/content/drive/MyDrive/FYP/code/checkpoints")

    from ultralytics import YOLO
    print(f"\n[Stage 5 Prep] Loading YOLOv5su models on {device.upper()}...")
    model_rgb = YOLO(str(rgb_ckpt))
    model_ir = YOLO(str(ir_ckpt))

    def find_file(folders: list, stem: str, exts: list) -> Path:
        for folder in folders:
            if folder and folder.exists():
                for ext in exts:
                    cand = folder / f"{stem}{ext}"
                    if cand.exists():
                        return cand
        return None

    img_exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".bmp"]

    for sp in splits_to_process:
        meta_file = meta_dir / f"{sp}.txt"
        if not meta_file.exists():
            # Check alternative meta locations
            meta_file = next((p / f"{sp}.txt" for p in [m3fd_root / "meta", m3fd_root, CODE_ROOT / "data" / "m3fd" / "meta"] if (p / f"{sp}.txt").exists()), None)
            if not meta_file:
                print(f"[Error] Split file for '{sp}' does not exist under {meta_dir}!")
                continue

        cache_file = out_dir / f"raw_predictions_{sp}.json"
        drive_cache_file = drive_out_dir / f"stage5_raw_predictions_{sp}.json"

        if not args.force:
            found_valid_cache = False
            for cand_f in [cache_file, drive_cache_file]:
                if cand_f.exists() and cand_f.stat().st_size > 1000:
                    try:
                        with open(cand_f, "r", encoding="utf-8") as jf:
                            chk = json.load(jf)
                        if chk.get('num_pairs', 0) > 0 and len(chk.get('predictions', {})) > 0:
                            print(f"  [Cache Hit] Split '{sp}' valid predictions ({chk['num_pairs']} pairs) found at: {cand_f}")
                            if cand_f != cache_file:
                                import shutil
                                shutil.copy2(cand_f, cache_file)
                                print(f"  [Restored] Synced cache from Drive to local SSD: {cache_file}")
                            found_valid_cache = True
                            break
                    except Exception:
                        pass
            if found_valid_cache:
                continue

        raw_stems = [l.strip() for l in meta_file.read_text(encoding='utf-8').splitlines() if l.strip()]
        stems = [Path(s).stem for s in raw_stems]
        print(f"\n================================================================================")
        print(f" 🚀 GENERATING RAW PREDICTIONS: SPLIT = {sp.upper()} ({len(stems)} pairs) ".center(80))
        print(f"================================================================================")
        print(f"  RGB Model       : {rgb_ckpt}")
        print(f"  IR Model        : {ir_ckpt}")
        print(f"  Confidence Min  : {args.conf}")
        print(f"  IoU Threshold   : {args.iou}")
        print(f"  Batch Size      : {args.batch_size}")
        print(f"================================================================================\n")

        # Candidate folders on disk (including Stage 4 structured SSD datasets)
        rgb_folders = [
            Path(f"/content/direct_rgb_dataset/{sp}/images"),
            vi_dir,
            m3fd_root / "Vis",
            m3fd_root / "vis",
            m3fd_root / "vi",
            m3fd_root / "VI"
        ]
        ir_folders = [
            Path(f"/content/direct_ir_dataset/{sp}/images"),
            ir_dir,
            m3fd_root / "Ir",
            m3fd_root / "ir",
            m3fd_root / "IR"
        ]
        lbl_folders = [
            Path(f"/content/direct_rgb_dataset/{sp}/labels"),
            Path(f"/content/direct_ir_dataset/{sp}/labels"),
            lbl_dir,
            m3fd_root / "labels",
            m3fd_root / "Labels"
        ]

        rgb_paths = []
        ir_paths = []
        lbl_paths = {}
        valid_stems = []

        for s in stems:
            r_cand = find_file(rgb_folders, s, img_exts)
            i_cand = find_file(ir_folders, s, img_exts)
            l_cand = find_file(lbl_folders, s, [".txt"])

            if r_cand and i_cand:
                rgb_paths.append(r_cand)
                ir_paths.append(i_cand)
                lbl_paths[s] = l_cand
                valid_stems.append(s)

        if len(valid_stems) == 0:
            print(f"[Error] No matching image pairs found for split '{sp}'!")
            print(f"  Checked RGB folders: {[str(p) for p in rgb_folders if p.exists()]}")
            print(f"  Checked IR folders : {[str(p) for p in ir_folders if p.exists()]}")
            print(f"  Sample stem        : '{stems[0] if stems else 'N/A'}'")
            continue

        print(f"  Extracting Direct RGB predictions ({len(rgb_paths)} images)...")
        t0 = time.time()
        rgb_preds = extract_predictions_for_modality(model_rgb, rgb_paths, batch_size=args.batch_size, conf=args.conf, iou=args.iou, device=device)
        t_rgb = time.time() - t0
        print(f"  ✅ RGB Extraction complete in {t_rgb:.1f}s ({len(rgb_paths)/t_rgb:.1f} FPS)")

        print(f"  Extracting Direct IR predictions ({len(ir_paths)} images)...")
        t0 = time.time()
        ir_preds = extract_predictions_for_modality(model_ir, ir_paths, batch_size=args.batch_size, conf=args.conf, iou=args.iou, device=device)
        t_ir = time.time() - t0
        print(f"  ✅ IR Extraction complete in {t_ir:.1f}s ({len(ir_paths)/t_ir:.1f} FPS)")

        # Compile final dataset JSON
        predictions_map = {}
        ground_truth_map = {}

        for stem in valid_stems:
            r_data = rgb_preds.get(stem, {'boxes': [], 'scores': [], 'classes': [], 'orig_shape': [640, 640]})
            i_data = ir_preds.get(stem, {'boxes': [], 'scores': [], 'classes': [], 'orig_shape': [640, 640]})
            h, w = r_data['orig_shape']

            lbl_f = lbl_paths.get(stem)
            gt_boxes, gt_classes = load_ground_truth(lbl_f, w, h)

            predictions_map[stem] = {
                'orig_shape': [h, w],
                'rgb': {
                    'boxes': r_data['boxes'],
                    'scores': r_data['scores'],
                    'classes': r_data['classes']
                },
                'ir': {
                    'boxes': i_data['boxes'],
                    'scores': i_data['scores'],
                    'classes': i_data['classes']
                }
            }
            ground_truth_map[stem] = {
                'boxes': gt_boxes,
                'classes': gt_classes
            }

        cache_data = {
            'split': sp,
            'num_pairs': len(valid_stems),
            'rgb_checkpoint': str(rgb_ckpt),
            'ir_checkpoint': str(ir_ckpt),
            'extraction_params': {
                'conf': args.conf,
                'iou': args.iou,
                'batch_size': args.batch_size
            },
            'timing': {
                'rgb_time_sec': round(t_rgb, 2),
                'ir_time_sec': round(t_ir, 2)
            },
            'predictions': predictions_map,
            'ground_truth': ground_truth_map
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)
        print(f"  [Cache Saved] 💾 Predictions saved to: {cache_file} ({cache_file.stat().st_size / (1024**2):.1f} MB)")

        if drive_out_dir.exists():
            try:
                with open(drive_cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f)
                print(f"  [Drive Mirrored] ☁️ Mirrored to: {drive_cache_file}")
            except Exception as e:
                print(f"  [Notice] Could not mirror to Drive: {e}")

    print("\n[Stage 5 Prep] ✅ Raw prediction caching completed successfully!\n")


if __name__ == '__main__':
    main()