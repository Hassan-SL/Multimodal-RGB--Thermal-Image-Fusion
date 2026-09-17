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
Stage 6: 6-Panel Qualitative Visualizer & Comparative Case Study Generator
================================================================================
Generates publication-quality 6-panel comparative visualization strips:
  Col 1: Visible Optical (RGB) Sensor Image
  Col 2: Thermal Infrared (IR) Sensor Image
  Col 3: Fixed TarDAL Fused Representation (Stage 3)
  Col 4: Previous Stage 3 Detector Detections (YOLOv5su)
  Col 5: Modern Stage 6 Detector Detections (YOLO11s Full Fine-Tune)
  Col 6: Ground Truth Annotations

Specifically targets the core thesis case studies:
  Case 1: YOLO11s succeeds where YOLOv5su misses (Architectural upgrade gain)
  Case 2: YOLOv5su succeeds where YOLO11s misses (Legacy detector niche)
  Case 3: Both succeed, but YOLO11s provides tighter bounding box localization
  Case 4: False Positive suppression (YOLO11s rejects background clutter)
  Case 5: Small / distant target resolution (<32 px targets)
================================================================================
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parent.parent

CLASSES = ['People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck']
COLOR_PALETTE = [
    (230, 25, 75),   # People (Red)
    (60, 180, 75),   # Car (Green)
    (0, 130, 200),   # Bus (Blue)
    (245, 130, 48),  # Lamp (Orange)
    (145, 30, 180),  # Motorcycle (Purple)
    (70, 240, 240)   # Truck (Cyan)
]


def resolve_image_file(folder: Path, stem: str) -> Path:
    """Finds image with any common extension."""
    if not folder or not folder.exists():
        return None
    for ext in [".jpg", ".png", ".jpeg", ".bmp", ".JPG", ".PNG"]:
        cand = folder / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def draw_boxes_on_image(img_bgr, boxes, scores=None, classes=None, is_gt=False):
    """Renders colored bounding boxes and labels onto an image."""
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]

    for i, box in enumerate(boxes):
        cls_id = int(classes[i]) if classes is not None and i < len(classes) else 0
        cls_name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else f"cls_{cls_id}"
        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]

        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if is_gt:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"GT: {cls_name}"
            bg_col = (0, 180, 0)
        else:
            conf = scores[i] if scores is not None and i < len(scores) else 1.0
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            bg_col = color

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        y_text = max(th + 2, y1 - 3)
        cv2.rectangle(canvas, (x1, y_text - th - 2), (x1 + tw + 2, y_text + 2), bg_col, -1)
        cv2.putText(canvas, label, (x1 + 1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def render_6panel_strip(
    rgb_raw, ir_raw, fused_raw,
    v5_dets, v11_dets, gt_boxes, gt_classes,
    case_title: str, save_path: Path
):
    """Stitches a 6-panel side-by-side comparison strip."""
    # Ensure IR has 3 channels
    if len(ir_raw.shape) == 2:
        ir_vis = cv2.cvtColor(ir_raw, cv2.COLOR_GRAY2BGR)
    else:
        ir_vis = ir_raw.copy()

    # Panel 1: Optical RGB
    p1 = rgb_raw.copy()
    cv2.putText(p1, "1. Optical RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

    # Panel 2: Thermal IR
    p2 = ir_vis.copy()
    cv2.putText(p2, "2. Thermal IR", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

    # Panel 3: TarDAL Fused
    p3 = fused_raw.copy()
    cv2.putText(p3, "3. TarDAL Fused (Fixed)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

    # Panel 4: Stage 3 YOLOv5su Detections
    p4 = draw_boxes_on_image(fused_raw, v5_dets['boxes'], v5_dets['scores'], v5_dets['classes'])
    cv2.putText(p4, "4. Stage 3 (YOLOv5su)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 100, 0), 2)

    # Panel 5: Stage 6 YOLO11s Detections
    p5 = draw_boxes_on_image(fused_raw, v11_dets['boxes'], v11_dets['scores'], v11_dets['classes'])
    cv2.putText(p5, "5. Stage 6 (YOLO11s)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)

    # Panel 6: Ground Truth
    p6 = draw_boxes_on_image(fused_raw, gt_boxes, classes=gt_classes, is_gt=True)
    cv2.putText(p6, "6. Ground Truth Annotations", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)

    # Combine into single row
    row = np.hstack([p1, p2, p3, p4, p5, p6])
    h, w = row.shape[:2]

    # Add header banner
    header = np.zeros((45, w, 3), dtype=np.uint8)
    cv2.putText(header, case_title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    strip = np.vstack([header, row])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), strip)


def load_gt_boxes_and_classes(lbl_path: Path):
    """Loads normalized YOLO boxes and converts to absolute coordinates (640x640)."""
    boxes = []
    classes = []
    if not lbl_path.exists():
        return np.zeros((0, 4)), []

    lines = lbl_path.read_text(encoding='utf-8').strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            xc, yc, bw, bh = [float(v) for v in parts[1:5]]
            x1 = (xc - bw / 2.0) * 640.0
            y1 = (yc - bh / 2.0) * 640.0
            x2 = (xc + bw / 2.0) * 640.0
            y2 = (yc + bh / 2.0) * 640.0
            boxes.append([x1, y1, x2, y2])
            classes.append(cls_id)

    return np.array(boxes, dtype=np.float32).reshape(-1, 4) if boxes else np.zeros((0, 4)), classes


def box_iou_numpy(boxes1, boxes2):
    """Calculates pairwise IoU between two sets of bounding boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1_x1, b1_y1, b1_x2, b1_y2 = np.split(boxes1, 4, axis=1)
    b2_x1, b2_y1, b2_x2, b2_y2 = np.split(boxes2, 4, axis=1)

    inter_x1 = np.maximum(b1_x1, b2_x1.T)
    inter_y1 = np.maximum(b1_y1, b2_y1.T)
    inter_x2 = np.minimum(b1_x2, b2_x2.T)
    inter_y2 = np.minimum(b1_y2, b2_y2.T)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area.T - inter_area

    return np.clip(inter_area / np.maximum(union_area, 1e-6), 0.0, 1.0)


def run_qualitative_visualizer(
    fused_dataset_dir: Path,
    m3fd_root: Path,
    yolo11s_ckpt: Path,
    yolov5su_ckpt: Path,
    output_dir: Path,
    conf_thresh: float = 0.25,
    num_samples: int = 10,
    device: str = "0"
):
    """
    Finds targeted multi-modal case studies and exports 6-panel strips.
    """
    from ultralytics import YOLO

    print("\n" + "=" * 90)
    print(" 🎨 STAGE 6: 6-PANEL QUALITATIVE VISUALIZER & CASE STUDY GENERATOR ".center(90))
    print("=" * 90)
    print(f"  Fused Dataset Dir  : {fused_dataset_dir}")
    print(f"  YOLO11s Checkpoint : {yolo11s_ckpt}")
    print(f"  YOLOv5su Checkpoint: {yolov5su_ckpt}")
    print(f"  Visualization Conf : {conf_thresh}")
    print(f"  Output Directory   : {output_dir}")
    print("=" * 90 + "\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load detectors
    print("[Stage 6 Visualizer] Loading YOLO11s detector...")
    model_v11 = YOLO(str(yolo11s_ckpt))

    print("[Stage 6 Visualizer] Loading Stage 3 YOLOv5su detector...")
    model_v5 = YOLO(str(yolov5su_ckpt)) if yolov5su_ckpt.exists() else None
    if model_v5 is None:
        print("[Stage 6 Visualizer] Warning: YOLOv5su checkpoint not found. Panel 4 will display empty predictions.")

    # Locate test images & labels
    test_img_dir = fused_dataset_dir / "images" / "test"
    test_lbl_dir = fused_dataset_dir / "labels" / "test"

    if not test_img_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {test_img_dir}")

    # Locate raw IR and RGB folders
    ir_dir = None
    for cand in ["ir", "Ir", "IR"]:
        if (m3fd_root / cand).exists():
            ir_dir = m3fd_root / cand
            break

    vi_dir = None
    for cand in ["vi", "Vis", "vis", "VI", "RGB", "rgb"]:
        if (m3fd_root / cand).exists():
            vi_dir = m3fd_root / cand
            break

    test_imgs = sorted(list(test_img_dir.glob("*.jpg")))
    print(f"[Stage 6 Visualizer] Scanning {len(test_imgs)} test images for targeted thesis case studies...")

    cases = {
        'case1_yolo11s_catches_missed': [],
        'case2_yolov5su_catches_missed': [],
        'case3_tighter_localization': [],
        'case4_fp_suppression': [],
        'case5_small_distant_targets': []
    }

    for img_p in test_imgs:
        stem = img_p.stem
        lbl_p = test_lbl_dir / f"{stem}.txt"
        gt_boxes, gt_classes = load_gt_boxes_and_classes(lbl_p)

        if len(gt_boxes) == 0:
            continue

        fused_img = cv2.imread(str(img_p))
        if fused_img is None:
            continue

        # YOLO11s Inference
        res_v11 = model_v11.predict(fused_img, conf=conf_thresh, device=device, verbose=False)[0]
        v11_b = res_v11.boxes.xyxy.cpu().numpy() if len(res_v11.boxes) > 0 else np.zeros((0, 4))
        v11_s = res_v11.boxes.conf.cpu().numpy() if len(res_v11.boxes) > 0 else np.zeros((0,))
        v11_c = res_v11.boxes.cls.cpu().numpy().astype(int) if len(res_v11.boxes) > 0 else np.zeros((0,), dtype=int)

        # YOLOv5su Inference
        if model_v5:
            res_v5 = model_v5.predict(fused_img, conf=conf_thresh, device=device, verbose=False)[0]
            v5_b = res_v5.boxes.xyxy.cpu().numpy() if len(res_v5.boxes) > 0 else np.zeros((0, 4))
            v5_s = res_v5.boxes.conf.cpu().numpy() if len(res_v5.boxes) > 0 else np.zeros((0,))
            v5_c = res_v5.boxes.cls.cpu().numpy().astype(int) if len(res_v5.boxes) > 0 else np.zeros((0,), dtype=int)
        else:
            v5_b, v5_s, v5_c = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

        # Evaluate hits against GT
        ious_v11 = box_iou_numpy(gt_boxes, v11_b) if len(v11_b) > 0 else np.zeros((len(gt_boxes), 0))
        ious_v5 = box_iou_numpy(gt_boxes, v5_b) if len(v5_b) > 0 else np.zeros((len(gt_boxes), 0))

        v11_hits = sum(1 for row in ious_v11 if np.max(row) >= 0.50) if ious_v11.shape[1] > 0 else 0
        v5_hits = sum(1 for row in ious_v5 if np.max(row) >= 0.50) if ious_v5.shape[1] > 0 else 0

        # Categorize
        if v11_hits > v5_hits and len(cases['case1_yolo11s_catches_missed']) < 3:
            cases['case1_yolo11s_catches_missed'].append(stem)
        elif v5_hits > v11_hits and len(cases['case2_yolov5su_catches_missed']) < 2:
            cases['case2_yolov5su_catches_missed'].append(stem)
        elif v11_hits == len(gt_boxes) and v5_hits == len(gt_boxes) and len(gt_boxes) >= 2:
            # Check localization tightness
            if np.mean(np.max(ious_v11, axis=1)) > np.mean(np.max(ious_v5, axis=1)) and len(cases['case3_tighter_localization']) < 3:
                cases['case3_tighter_localization'].append(stem)
        elif len(v5_b) > len(gt_boxes) and len(v11_b) == len(gt_boxes) and len(cases['case4_fp_suppression']) < 2:
            cases['case4_fp_suppression'].append(stem)

        # Check for small targets (<32x32 px)
        small_targets = [b for b in gt_boxes if (b[2] - b[0]) * (b[3] - b[1]) < 1024]
        if len(small_targets) >= 2 and v11_hits > 0 and len(cases['case5_small_distant_targets']) < 2:
            cases['case5_small_distant_targets'].append(stem)

        if sum(len(v) for v in cases.values()) >= num_samples:
            break

    # Render selected strips
    case_titles = {
        'case1_yolo11s_catches_missed': "Case 1: YOLO11s Rescues Missed Target (Modern C3k2/SPPF Feature Extraction)",
        'case2_yolov5su_catches_missed': "Case 2: YOLOv5su Legacy Advantage Case Study",
        'case3_tighter_localization': "Case 3: Decoupled Head Tightened Localization (Higher IoU Bounding Box)",
        'case4_fp_suppression': "Case 4: False Positive Rejection (Clean Background Clutter Suppression)",
        'case5_small_distant_targets': "Case 5: Distant Small Target Resolution (<32 px Target Recognition)"
    }

    rendered_count = 0
    for c_key, stems in cases.items():
        title = case_titles[c_key]
        for idx, stem in enumerate(stems):
            img_p = test_img_dir / f"{stem}.jpg"
            lbl_p = test_lbl_dir / f"{stem}.txt"

            fused_img = cv2.imread(str(img_p))
            gt_boxes, gt_classes = load_gt_boxes_and_classes(lbl_p)

            # Raw IR and RGB
            ir_p = resolve_image_file(ir_dir, stem)
            vi_p = resolve_image_file(vi_dir, stem)

            ir_raw = cv2.imread(str(ir_p), cv2.IMREAD_GRAYSCALE) if ir_p else np.zeros((640, 640), dtype=np.uint8)
            vi_raw = cv2.imread(str(vi_p)) if vi_p else np.zeros((640, 640, 3), dtype=np.uint8)

            ir_raw = cv2.resize(ir_raw, (640, 640))
            vi_raw = cv2.resize(vi_raw, (640, 640))

            # Run models
            r11 = model_v11.predict(fused_img, conf=conf_thresh, device=device, verbose=False)[0]
            v11_dets = {
                'boxes': r11.boxes.xyxy.cpu().numpy() if len(r11.boxes) > 0 else np.zeros((0, 4)),
                'scores': r11.boxes.conf.cpu().numpy() if len(r11.boxes) > 0 else np.zeros((0,)),
                'classes': r11.boxes.cls.cpu().numpy().astype(int) if len(r11.boxes) > 0 else np.zeros((0,), dtype=int)
            }

            if model_v5:
                r5 = model_v5.predict(fused_img, conf=conf_thresh, device=device, verbose=False)[0]
                v5_dets = {
                    'boxes': r5.boxes.xyxy.cpu().numpy() if len(r5.boxes) > 0 else np.zeros((0, 4)),
                    'scores': r5.boxes.conf.cpu().numpy() if len(r5.boxes) > 0 else np.zeros((0,)),
                    'classes': r5.boxes.cls.cpu().numpy().astype(int) if len(r5.boxes) > 0 else np.zeros((0,), dtype=int)
                }
            else:
                v5_dets = {'boxes': np.zeros((0, 4)), 'scores': np.zeros((0,)), 'classes': np.zeros((0,), dtype=int)}

            fig_name = f"{c_key}_ex{idx+1}_{stem}.jpg"
            save_path = output_dir / fig_name
            render_6panel_strip(
                vi_raw, ir_raw, fused_img,
                v5_dets, v11_dets, gt_boxes, gt_classes,
                f"{title} [Sample {stem}]",
                save_path
            )
            print(f"  - [Strip Generated] {fig_name}")
            rendered_count += 1

    print(f"\n[Stage 6 Visualizer] ✅ Generated {rendered_count} 6-panel case study strips in: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 6: 6-Panel Qualitative Visualizer & Case Study Generator")
    parser.add_argument('--fused_dir', type=str, default=None, help="Path to M3FD_STAGE6_FUSED directory")
    parser.add_argument('--m3fd_root', type=str, default=None, help="Path to raw M3FD root (for raw IR/RGB)")
    parser.add_argument('--weights', type=str, default=None, help="Path to stage6_yolo11s_best.pt")
    parser.add_argument('--v5_weights', type=str, default=None, help="Path to Stage 3 best.pt (YOLOv5su)")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory for visual strips")
    parser.add_argument('--conf', type=float, default=0.25, help="Visualization confidence threshold")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of qualitative strips to render")
    parser.add_argument('--device', type=str, default="0", help="CUDA device index or 'cpu'")
    args = parser.parse_args()

    # Resolve paths
    if args.fused_dir:
        fused_dir = Path(args.fused_dir)
    else:
        candidates = [
            Path("/content/M3FD_STAGE6_FUSED"),
            CODE_ROOT / "data" / "M3FD_STAGE6_FUSED"
        ]
        fused_dir = next((c for c in candidates if c.exists()), candidates[0])

    if args.m3fd_root:
        m3fd_root = Path(args.m3fd_root)
    else:
        candidates = [
            Path("/content/m3fd/M3FD_Detection"),
            Path("/content/m3fd"),
            CODE_ROOT / "data" / "m3fd"
        ]
        m3fd_root = next((c for c in candidates if c.exists()), candidates[0])

    yolo11_ckpt = Path(args.weights) if args.weights else Path("/content/drive/MyDrive/FYP/code/checkpoints/stage6/stage6_yolo11s_best.pt")
    if not yolo11_ckpt.exists():
        yolo11_ckpt = CODE_ROOT / "checkpoints" / "stage6" / "stage6_yolo11s_best.pt"

    yolo5_ckpt = Path(args.v5_weights) if args.v5_weights else Path("/content/drive/MyDrive/FYP/code/checkpoints/best.pt")
    if not yolo5_ckpt.exists():
        yolo5_ckpt = CODE_ROOT / "checkpoints" / "best.pt"

    output_dir = Path(args.output_dir) if args.output_dir else CODE_ROOT / "runs" / "stage6" / "qualitative"

    run_qualitative_visualizer(
        fused_dataset_dir=fused_dir,
        m3fd_root=m3fd_root,
        yolo11s_ckpt=yolo11_ckpt,
        yolov5su_ckpt=yolo5_ckpt,
        output_dir=output_dir,
        conf_thresh=args.conf,
        num_samples=args.num_samples,
        device=args.device
    )


if __name__ == '__main__':
    main()