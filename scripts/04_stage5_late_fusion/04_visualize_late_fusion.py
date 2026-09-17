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
Stage 5: Qualitative Late Fusion Visualizer & Case Study Generator
================================================================================
Generates high-resolution 5-panel comparative visualization strips:
  Col 1: Optical RGB Sensor Image
  Col 2: Thermal IR Sensor Image
  Col 3: Direct RGB YOLOv5su Detections
  Col 4: Direct IR YOLOv5su Detections
  Col 5: Late Fusion (WBF) Final Detections + Ground Truth Annotations

Specifically targets the 5 fundamental multi-modal thesis cases:
  Case 1: RGB fails  -> IR succeeds  -> Fusion succeeds (Thermal superiority)
  Case 2: IR fails   -> RGB succeeds -> Fusion succeeds (Photometric superiority)
  Case 3: Both succeed -> Fusion tightens box & consensus confidence
  Case 4: Both fail  -> Challenging extreme occlusion
  Case 5: Cross-modal False Positive / penalty analysis
================================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent

import importlib
eng_17 = importlib.import_module("17_late_fusion_engine")
CLASSES = eng_17.CLASSES
match_and_fuse_single_image = eng_17.match_and_fuse_single_image
box_iou_numpy = eng_17.box_iou_numpy
load_raw_predictions_cache = eng_17.load_raw_predictions_cache

COLOR_PALETTE = [
    (230, 25, 75),   # People (Red)
    (60, 180, 75),   # Car (Green)
    (0, 130, 200),   # Bus (Blue)
    (245, 130, 48),  # Lamp (Orange)
    (145, 30, 180),  # Motorcycle (Purple)
    (70, 240, 240)   # Truck (Cyan)
]


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


def draw_boxes_on_image(img_bgr, boxes, scores=None, classes=None, is_gt=False):
    """Draws colored bounding boxes and labels onto an image."""
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

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_text = max(th + 2, y1 - 3)
        cv2.rectangle(canvas, (x1, y_text - th - 2), (x1 + tw + 2, y_text + 2), bg_col, -1)
        cv2.putText(canvas, label, (x1 + 1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def render_5panel_figure(rgb_raw, ir_raw, rgb_dets, ir_dets, fused_dets, gt_boxes, gt_classes, case_title, save_path):
    """Assembles and saves a 5-panel side-by-side comparison strip."""
    # Ensure IR has 3 channels for visualization
    if len(ir_raw.shape) == 2:
        ir_vis = cv2.cvtColor(ir_raw, cv2.COLOR_GRAY2BGR)
    else:
        ir_vis = ir_raw.copy()

    # Panel 1: Clean RGB
    p1 = rgb_raw.copy()
    cv2.putText(p1, "1. Optical RGB Sensor", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Panel 2: Clean IR
    p2 = ir_vis.copy()
    cv2.putText(p2, "2. Thermal IR Sensor", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Panel 3: RGB Detections
    p3 = draw_boxes_on_image(rgb_raw, rgb_dets['boxes'], rgb_dets['scores'], rgb_dets['classes'])
    cv2.putText(p3, "3. Direct RGB YOLO", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Panel 4: IR Detections
    p4 = draw_boxes_on_image(ir_vis, ir_dets['boxes'], ir_dets['scores'], ir_dets['classes'])
    cv2.putText(p4, "4. Direct IR YOLO", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Panel 5: Late Fusion Detections + GT outline
    p5 = draw_boxes_on_image(rgb_raw, fused_dets['boxes'], fused_dets['scores'], fused_dets['classes'])
    if len(gt_boxes) > 0:
        for gb in gt_boxes:
            gx1, gy1, gx2, gy2 = [int(v) for v in gb]
            cv2.rectangle(p5, (gx1, gy1), (gx2, gy2), (255, 255, 255), 1, cv2.LINE_AA) # White dashed outline for GT
    cv2.putText(p5, "5. Late Fusion (WBF)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Resize all panels to matching height/width
    target_h = 480
    target_w = int(p1.shape[1] * (target_h / p1.shape[0]))
    panels_resized = [cv2.resize(p, (target_w, target_h)) for p in [p1, p2, p3, p4, p5]]

    # Stitch horizontally
    stitched = np.hstack(panels_resized)

    # Add top title banner
    banner_h = 50
    banner = np.zeros((banner_h, stitched.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, f"FYP Phase 5 Qualitative Analysis | {case_title}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    final_img = np.vstack((banner, stitched))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), final_img)
    print(f"  [Saved Figure] 📸 {save_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Qualitative Late Fusion Case Study Visualizer")
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help="Dataset split")
    parser.add_argument('--w_rgb', type=float, default=0.6, help="w_RGB weight")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save visual output")
    args = parser.parse_args()

    test_cache = load_raw_predictions_cache(split=args.split)
    preds = test_cache['predictions']
    gt = test_cache['ground_truth']

    w_rgb = args.w_rgb
    w_ir = round(1.0 - w_rgb, 4)

    m3fd_root = resolve_m3fd_root()
    ir_dir = next((m3fd_root / d for d in ["Ir", "ir", "IR", "thermal", "Thermal"] if (m3fd_root / d).exists()), m3fd_root / "ir")
    vi_dir = next((m3fd_root / d for d in ["Vis", "vis", "VIS", "vi", "Vi", "VI"] if (m3fd_root / d).exists()), m3fd_root / "vi")

    rgb_folders = [
        Path(f"/content/direct_rgb_dataset/{args.split}/images"),
        vi_dir,
        m3fd_root / "Vis",
        m3fd_root / "vis",
        m3fd_root / "vi",
        m3fd_root / "VI"
    ]
    ir_folders = [
        Path(f"/content/direct_ir_dataset/{args.split}/images"),
        ir_dir,
        m3fd_root / "Ir",
        m3fd_root / "ir",
        m3fd_root / "IR"
    ]
    img_exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".bmp"]

    def find_img(folders: list, stem: str) -> Path:
        for folder in folders:
            if folder and folder.exists():
                for ext in img_exts:
                    cand = folder / f"{stem}{ext}"
                    if cand.exists():
                        return cand
        return None

    out_dir = Path(args.output_dir) if args.output_dir else CODE_ROOT / "runs" / "stage5_late_fusion" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(" 🎨 STAGE 5: QUALITATIVE VISUALIZATION & THESIS CASE STUDIES ".center(90))
    print("=" * 90)
    print(f"  Dataset Split  : {args.split.upper()} (840 pairs)")
    print(f"  Fusion Weights : w_RGB={w_rgb:.2f}, w_IR={w_ir:.2f}")
    print(f"  Output Folder  : {out_dir}")
    print("=" * 90 + "\n")

    # Classify candidate image stems into the 5 target cases
    case_candidates = {
        'case1_ir_rescues_rgb': [],
        'case2_rgb_rescues_ir': [],
        'case3_consensus_boost': [],
        'case4_both_fail': [],
        'case5_general_scene': []
    }

    for stem, pdata in preds.items():
        gdata = gt.get(stem, {'boxes': [], 'classes': []})
        gt_boxes = np.array(gdata['boxes'], dtype=np.float32).reshape(-1, 4) if gdata['boxes'] else np.zeros((0, 4))
        gt_classes = np.array(gdata['classes'], dtype=np.int32) if gdata['classes'] else np.zeros((0,))

        if len(gt_boxes) == 0:
            continue

        r_boxes = [b for b, s in zip(pdata['rgb']['boxes'], pdata['rgb']['scores']) if s >= args.conf]
        r_classes = [c for c, s in zip(pdata['rgb']['classes'], pdata['rgb']['scores']) if s >= args.conf]
        r_b_np = np.array(r_boxes, dtype=np.float32).reshape(-1, 4) if r_boxes else np.zeros((0, 4))

        i_boxes = [b for b, s in zip(pdata['ir']['boxes'], pdata['ir']['scores']) if s >= args.conf]
        i_classes = [c for c, s in zip(pdata['ir']['classes'], pdata['ir']['scores']) if s >= args.conf]
        i_b_np = np.array(i_boxes, dtype=np.float32).reshape(-1, 4) if i_boxes else np.zeros((0, 4))

        # Check hits
        r_hits = 0
        i_hits = 0
        for gb, gc in zip(gt_boxes, gt_classes):
            gb_1 = gb.reshape(1, 4)
            if len(r_b_np) > 0 and np.max(box_iou_numpy(gb_1, r_b_np)) >= 0.5:
                r_hits += 1
            if len(i_b_np) > 0 and np.max(box_iou_numpy(gb_1, i_b_np)) >= 0.5:
                i_hits += 1

        if i_hits > r_hits and len(case_candidates['case1_ir_rescues_rgb']) < 3:
            case_candidates['case1_ir_rescues_rgb'].append(stem)
        elif r_hits > i_hits and len(case_candidates['case2_rgb_rescues_ir']) < 3:
            case_candidates['case2_rgb_rescues_ir'].append(stem)
        elif r_hits == len(gt_boxes) and i_hits == len(gt_boxes) and len(gt_boxes) >= 2 and len(case_candidates['case3_consensus_boost']) < 3:
            case_candidates['case3_consensus_boost'].append(stem)
        elif r_hits == 0 and i_hits == 0 and len(case_candidates['case4_both_fail']) < 2:
            case_candidates['case4_both_fail'].append(stem)
        elif len(case_candidates['case5_general_scene']) < 2:
            case_candidates['case5_general_scene'].append(stem)

    case_definitions = [
        ("case1_ir_rescues_rgb", "Case 1: IR Rescues RGB (Thermal Pedestrian Detection)", case_candidates['case1_ir_rescues_rgb']),
        ("case2_rgb_rescues_ir", "Case 2: RGB Rescues IR (Visible High-Frequency Texture & Lamp)", case_candidates['case2_rgb_rescues_ir']),
        ("case3_consensus_boost", "Case 3: Dual-Modal Consensus (Tightened Localization & High Confidence)", case_candidates['case3_consensus_boost']),
        ("case4_both_fail", "Case 4: Extreme Degradation / Distance (Both Models Missed)", case_candidates['case4_both_fail']),
        ("case5_general_scene", "Case 5: Multi-Class Traffic Flow (Car, Bus, Motorcycle Fusion)", case_candidates['case5_general_scene'])
    ]

    for c_key, c_title, stems in case_definitions:
        print(f"\nRendering {c_title} ({len(stems)} examples)...")
        for idx, stem in enumerate(stems):
            r_path = find_img(rgb_folders, stem)
            i_path = find_img(ir_folders, stem)

            if not r_path or not i_path:
                continue

            rgb_img = cv2.imread(str(r_path))
            ir_img = cv2.imread(str(i_path), cv2.IMREAD_GRAYSCALE)
            if rgb_img is None or ir_img is None:
                continue

            pdata = preds[stem]
            gdata = gt.get(stem, {'boxes': [], 'classes': []})

            # Filter single modal detections
            rgb_dets = {
                'boxes': [b for b, s in zip(pdata['rgb']['boxes'], pdata['rgb']['scores']) if s >= args.conf],
                'scores': [s for s in pdata['rgb']['scores'] if s >= args.conf],
                'classes': [c for c, s in zip(pdata['rgb']['classes'], pdata['rgb']['scores']) if s >= args.conf]
            }
            ir_dets = {
                'boxes': [b for b, s in zip(pdata['ir']['boxes'], pdata['ir']['scores']) if s >= args.conf],
                'scores': [s for s in pdata['ir']['scores'] if s >= args.conf],
                'classes': [c for c, s in zip(pdata['ir']['classes'], pdata['ir']['scores']) if s >= args.conf]
            }

            # Run late fusion
            fused_dets = match_and_fuse_single_image(
                pdata['rgb'], pdata['ir'],
                w_rgb=w_rgb, w_ir=w_ir,
                iou_thresh=0.50,
                conf_thresh=args.conf
            )

            fig_name = f"{c_key}_ex{idx+1}_{stem}.jpg"
            save_path = out_dir / fig_name
            render_5panel_figure(
                rgb_img, ir_img, rgb_dets, ir_dets, fused_dets,
                gdata['boxes'], gdata['classes'],
                f"{c_title} [Image {stem}]",
                save_path
            )

    print(f"\n[Stage 5 Visualizer] ✅ All 5 multi-modal case study figures generated in: {out_dir}\n")


if __name__ == '__main__':
    main()