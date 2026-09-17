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
Stage 4: Tri-Modal Side-by-Side Prediction Visualizer
================================================================================
Generates high-resolution comparative visualization grids:
  Col 1: Optical RGB + Direct RGB Predictions
  Col 2: Thermal IR  + Direct IR Predictions
  Col 3: Fused Image + Stage 3 Predictions
  Col 4: Fused Image + Ground Truth Annotations

Enables rapid qualitative failure analysis and edge-case inspection.
================================================================================
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

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


def draw_predictions(img_bgr: np.ndarray, boxes, clss, confs=None, is_gt=False):
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]

    for i, box in enumerate(boxes):
        cls_id = int(clss[i])
        cls_name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else f"cls_{cls_id}"
        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]

        if is_gt:
            # YOLO format: cx, cy, bw, bh
            cx, cy, bw, bh = box
            x1 = int((cx - bw / 2.0) * w)
            y1 = int((cy - bh / 2.0) * h)
            x2 = int((cx + bw / 2.0) * w)
            y2 = int((cy + bh / 2.0) * h)
            label = f"GT: {cls_name}"
        else:
            # xyxy format
            x1, y1, x2, y2 = [int(v) for v in box]
            conf = confs[i] if confs is not None else 1.0
            label = f"{cls_name} {conf:.2f}"

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 1, max(th, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Qualitative Prediction Visualizer")
    parser.add_argument('--num_samples', type=int, default=5, help="Number of test pairs to visualize")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold for bounding boxes")
    parser.add_argument('--rgb_ckpt', type=str, default=None)
    parser.add_argument('--ir_ckpt', type=str, default=None)
    parser.add_argument('--fused_ckpt', type=str, default=None)
    parser.add_argument('--gen_ckpt', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m3fd_root = resolve_m3fd_root()
    test_meta = m3fd_root / "meta" / "test.txt"

    if not test_meta.exists():
        print(f"[Error] Test split file not found: {test_meta}")
        return

    with open(test_meta, 'r') as f:
        test_stems = [line.strip() for line in f if line.strip()]

    # Resolve checkpoints
    ckpt_dir = Path("/content/drive/MyDrive/FYP/code/checkpoints") if Path("/content").exists() else CODE_ROOT / "checkpoints"

    rgb_ckpt = resolve_ckpt([args.rgb_ckpt, ckpt_dir / "yolov5su_rgb_best.pt", CODE_ROOT / "runs" / "stage4_direct_rgb" / "yolov5su_rgb" / "weights" / "best.pt"])
    ir_ckpt = resolve_ckpt([args.ir_ckpt, ckpt_dir / "yolov5su_ir_best.pt", CODE_ROOT / "runs" / "stage4_direct_ir" / "yolov5su_ir" / "weights" / "best.pt"])
    fused_det_ckpt = resolve_ckpt([args.fused_ckpt, ckpt_dir / "stage3_best.pt", ckpt_dir / "best.pt"])
    fused_gen_ckpt = resolve_ckpt([args.gen_ckpt, ckpt_dir / "stage3_gen_best.pt", CODE_ROOT / "runs" / "stage3_joint_end2end" / "stage3_gen_best.pt"])

    # Load models
    rgb_model = YOLO(str(rgb_ckpt)) if rgb_ckpt else None
    ir_model = YOLO(str(ir_ckpt)) if ir_ckpt else None
    fused_model = YOLO(str(fused_det_ckpt)) if fused_det_ckpt else None

    gen_model = None
    if fused_gen_ckpt:
        from module.fuse.generator import Generator
        gen_model = Generator(dim=32, depth=3).to(device)
        st = torch.load(str(fused_gen_ckpt), map_location=device)
        gen_model.load_state_dict(st.get('g', st.get('fuse', st)))
        gen_model.eval()

    save_dir = Path(args.save_dir) if args.save_dir else (CODE_ROOT / "runs" / "stage4_comparison_viz")
    save_dir.mkdir(parents=True, exist_ok=True)

    selected_stems = test_stems[:min(args.num_samples, len(test_stems))]
    print(f"\n[AG Viz] Visualizing {len(selected_stems)} test pairs side-by-side...")

    for idx, stem in enumerate(selected_stems):
        vi_path = m3fd_root / "vi" / f"{stem}.png"
        if not vi_path.exists():
            vi_path = m3fd_root / "vi" / f"{stem}.jpg"
        ir_path = m3fd_root / "ir" / f"{stem}.png"
        if not ir_path.exists():
            ir_path = m3fd_root / "ir" / f"{stem}.jpg"
        lbl_path = m3fd_root / "labels" / f"{stem}.txt"

        if not vi_path.exists() or not ir_path.exists():
            continue

        img_vi = cv2.imread(str(vi_path))
        img_ir_raw = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        img_ir = cv2.cvtColor(img_ir_raw, cv2.COLOR_GRAY2BGR)

        # 1. Ground Truth
        gt_boxes, gt_clss = [], []
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_clss.append(int(parts[0]))
                        gt_boxes.append([float(x) for x in parts[1:5]])

        # 2. RGB Inference
        rgb_canvas = img_vi.copy()
        if rgb_model:
            res = rgb_model.predict(img_vi, conf=args.conf, imgsz=640, verbose=False, device=device)[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            rgb_canvas = draw_predictions(rgb_canvas, boxes, clss, confs)

        # 3. IR Inference
        ir_canvas = img_ir.copy()
        if ir_model:
            res = ir_model.predict(img_ir, conf=args.conf, imgsz=640, verbose=False, device=device)[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            ir_canvas = draw_predictions(ir_canvas, boxes, clss, confs)

        # 4. Fused Generation & Inference
        fused_canvas = img_vi.copy()
        fused_gt_canvas = img_vi.copy()
        if gen_model:
            t_ir = torch.from_numpy(img_ir_raw).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            vi_gray = cv2.cvtColor(img_vi, cv2.COLOR_BGR2GRAY)
            t_vi = torch.from_numpy(vi_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                fused_t = gen_model(t_ir, t_vi)
            fused_np = (fused_t.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            fused_bgr = cv2.cvtColor(fused_np, cv2.COLOR_GRAY2BGR)
            fused_canvas = fused_bgr.copy()
            fused_gt_canvas = fused_bgr.copy()

            if fused_model:
                res = fused_model.predict(fused_bgr, conf=args.conf, imgsz=640, verbose=False, device=device)[0]
                boxes = res.boxes.xyxy.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                fused_canvas = draw_predictions(fused_canvas, boxes, clss, confs)

        fused_gt_canvas = draw_predictions(fused_gt_canvas, gt_boxes, gt_clss, is_gt=True)

        # Plot 1x4 side-by-side grid
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        axes[0].imshow(cv2.cvtColor(rgb_canvas, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Direct RGB Predictions\n({stem})", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(ir_canvas, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Direct IR Predictions\n({stem})", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(fused_canvas, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Stage 3 Fusion Predictions\n({stem})", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        axes[3].imshow(cv2.cvtColor(fused_gt_canvas, cv2.COLOR_BGR2RGB))
        axes[3].set_title(f"Ground Truth Annotations\n({stem})", fontsize=12, fontweight='bold')
        axes[3].axis('off')

        plt.tight_layout()
        out_fig_path = save_dir / f"stage4_comparison_{idx+1:02d}_{stem}.png"
        plt.savefig(out_fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved comparison grid: {out_fig_path.name}")

    print(f"\n[AG Viz] All comparison visualizations saved to: {save_dir}")


if __name__ == "__main__":
    main()