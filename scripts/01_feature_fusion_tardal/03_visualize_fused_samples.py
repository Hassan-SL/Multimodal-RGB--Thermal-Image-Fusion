"""
TarDAL Fused & Detection Results Visualizer (scripts_AG/04_visualize_results.py)
Generates side-by-side comparative grid of Infrared, Visible, and Fused Images with YOLO Predicted Bounding Boxes.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Dynamic path resolver
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
    def get_project_root() -> Path:
        return Path(__file__).resolve().parent.parent

    def get_dataset_root() -> Path:
        cand = [
            Path("/content/m3fd/M3FD_Detection"),
            Path("/content/m3fd"),
            get_project_root() / "data" / "m3fd"
        ]
        return next((c for c in cand if c.exists()), cand[0])

CLASS_NAMES = ['People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck']
CLASS_COLORS = [
    (255, 0, 0),     # 0: Red (People)
    (0, 255, 0),     # 1: Green (Car)
    (0, 0, 255),     # 2: Blue (Bus)
    (255, 255, 0),   # 3: Yellow (Lamp)
    (0, 255, 255),   # 4: Cyan (Motorcycle)
    (255, 0, 255)    # 5: Magenta (Truck)
]

def draw_yolo_boxes(img_bgr: np.ndarray, txt_path: Path, conf_thresh: float = 0.25) -> np.ndarray:
    """Draws YOLO formatted bounding boxes on a BGR image using confidence thresholding (default 0.25)."""
    img_out = img_bgr.copy()
    h_img, w_img = img_out.shape[:2]

    if not txt_path.exists() or txt_path.stat().st_size == 0:
        return img_out

    lines = txt_path.read_text().strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None

            # Skip low confidence noise predictions (standard threshold >= 0.25)
            if conf is not None and conf < conf_thresh:
                continue

            # Convert normalized cxcywh to pixel xyxy
            xmin = int((cx - bw / 2.0) * w_img)
            ymin = int((cy - bh / 2.0) * h_img)
            xmax = int((cx + bw / 2.0) * w_img)
            ymax = int((cy + bh / 2.0) * h_img)

            # Clamp coordinates
            xmin = max(0, min(w_img - 1, xmin))
            ymin = max(0, min(h_img - 1, ymin))
            xmax = max(0, min(w_img - 1, xmax))
            ymax = max(0, min(h_img - 1, ymax))

            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
            label_str = f"{cls_name} {conf:.2f}" if conf is not None else cls_name

            # Draw rectangle
            cv2.rectangle(img_out, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw label banner
            (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_out, (xmin, max(0, ymin - th - 6)), (xmin + tw + 4, ymin), color, -1)
            cv2.putText(img_out, label_str, (xmin + 2, max(th + 2, ymin - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        except Exception:
            continue

    return img_out

def visualize_results(run_dir: Path | None = None, num_samples: int = 4, model: str = 'tardal-tt', split: str = 'val'):
    proj_root = get_project_root()
    dataset_root = get_dataset_root()

    # Search prediction directory
    raw_name = model.lower().replace('.pth', '').replace('.pt', '').strip() if model else 'tardal-tt'
    alias_map = {'dt': 'tardal-dt', 'ct': 'tardal-ct', 'tt': 'tardal-tt'}
    model_name = alias_map.get(raw_name, raw_name)
    if run_dir:
        run_dir = Path(run_dir)
    else:
        cand_runs = [
            Path(f"/content/runs/{model_name}"),
            proj_root / "runs" / f"{model_name}",
            proj_root / "runs" / "inference" / f"{model_name}",
            proj_root / "TarDAL-main" / "runs" / f"{model_name}"
        ]
        run_dir = next((d for d in cand_runs if d.exists() and (d / "images").exists()), cand_runs[0])

    if not run_dir:
        print("[Error] Could not find inference output directory.")
        return

    # Prioritize selected split images (val or test) for evaluation visualization
    split_folder = "val" if split.lower().startswith("val") else "test"
    cand_img_dirs = [
        Path(f"/content/fused_dataset_stage2/{split_folder}/images"),
        proj_root / "runs" / "fused_dataset_stage2" / split_folder / "images",
        Path("/content/fused_dataset_stage2/val/images"),
        run_dir / "images",
        Path("/content/fused_dataset_stage2/images"),
        proj_root / "runs" / "fused_dataset_stage2" / "images"
    ]
    fused_img_dir = next((d for d in cand_img_dirs if d.exists() and len(list(d.glob("*.*"))) > 0), cand_img_dirs[0])
    split_name = f"{split_folder.upper()} (unseen)"

    cand_lbl_dirs = [
        run_dir / "labels",
        run_dir / "val_eval" / "labels",
        run_dir / "val" / "labels",
        run_dir.parent / "labels"
    ]
    pred_label_dir = next((d for d in cand_lbl_dirs if d.exists() and len(list(d.glob("*.txt"))) > 0), cand_lbl_dirs[0])

    # Search dataset ir and vi directories
    ir_dir_cand = [Path("/content/m3fd/M3FD_Detection/Ir"), Path("/content/m3fd/Ir"), dataset_root / "ir", dataset_root / "Ir"]
    vi_dir_cand = [Path("/content/m3fd/M3FD_Detection/Vis"), Path("/content/m3fd/Vis"), dataset_root / "vi", dataset_root / "Vis"]

    ir_dir = next((d for d in ir_dir_cand if d.exists()), None)
    vi_dir = next((d for d in vi_dir_cand if d.exists()), None)

    fused_imgs = sorted(list(fused_img_dir.glob("*.png"))) + sorted(list(fused_img_dir.glob("*.jpg")))
    if not fused_imgs:
        print(f"[Error] No fused images found in candidate directories: {cand_img_dirs}")
        return

    import random
    random.shuffle(fused_imgs)
    selected_imgs = fused_imgs[:num_samples]
    print(f"[AG Vis] 🎲 Randomly selected {len(selected_imgs)} sample image pairs from the {split_name} split ({len(fused_imgs)} total images available).")

    fig, axes = plt.subplots(len(selected_imgs), 4, figsize=(20, 4.5 * len(selected_imgs)))
    if len(selected_imgs) == 1:
        axes = [axes]

    print("==================================================")
    print("      TAR_DAL RESULT VISUALIZATION ENGINE        ")
    print("==================================================")

    for i, fused_p in enumerate(selected_imgs):
        stem = fused_p.stem
        
        # Load Fused image
        fused_bgr = cv2.imread(str(fused_p))
        fused_pure_rgb = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGB)
        
        # Load IR image
        ir_p = next((ir_dir / f for f in [f"{stem}.png", f"{stem}.jpg"] if ir_dir and (ir_dir / f).exists()), None)
        ir_bgr = cv2.imread(str(ir_p)) if ir_p else fused_bgr
        
        # Load VIS image
        vi_p = next((vi_dir / f for f in [f"{stem}.png", f"{stem}.jpg"] if vi_dir and (vi_dir / f).exists()), None)
        vi_bgr = cv2.imread(str(vi_p)) if vi_p else fused_bgr

        # Draw YOLO predictions on Fused image (conf >= 0.25)
        txt_p = pred_label_dir / f"{stem}.txt"
        fused_boxed_bgr = draw_yolo_boxes(fused_bgr, txt_p, conf_thresh=0.25)

        # Fallback to direct model inference with conf=0.25 if txt had no predictions >= 0.25
        cand_best = [
            Path("/content/drive/MyDrive/FYP/code/checkpoints/best.pt"),
            proj_root / "checkpoints" / "best.pt",
            run_dir / "weights" / "best.pt"
        ]
        best_pt = next((w for w in cand_best if w.exists()), None)
        if np.array_equal(fused_boxed_bgr, fused_bgr) and best_pt:
            try:
                from ultralytics import YOLO
                yolo_m = YOLO(str(best_pt))
                pred_res = yolo_m.predict(fused_bgr, conf=0.25, verbose=False)[0]
                fused_boxed_bgr = pred_res.plot() # High quality native rendering
            except Exception:
                pass

        # Convert BGR -> RGB
        ir_rgb = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2RGB)
        vi_rgb = cv2.cvtColor(vi_bgr, cv2.COLOR_BGR2RGB)
        fused_boxed_rgb = cv2.cvtColor(fused_boxed_bgr, cv2.COLOR_BGR2RGB)

        # Plot 1: IR
        axes[i][0].imshow(ir_rgb)
        axes[i][0].set_title(f"Infrared (IR): {stem}", fontsize=11, fontweight='bold')
        axes[i][0].axis('off')

        # Plot 2: VIS
        axes[i][1].imshow(vi_rgb)
        axes[i][1].set_title(f"Visible (VIS): {stem}", fontsize=11, fontweight='bold')
        axes[i][1].axis('off')

        # Plot 3: Pure Fused TarDAL Image
        axes[i][2].imshow(fused_pure_rgb)
        axes[i][2].set_title(f"Pure TarDAL Fused: {stem}", fontsize=11, fontweight='bold', color='navy')
        axes[i][2].axis('off')

        # Plot 4: Fused + YOLO Detections
        axes[i][3].imshow(fused_boxed_rgb)
        axes[i][3].set_title(f"TarDAL Fused + YOLO: {stem}", fontsize=11, fontweight='bold', color='darkgreen')
        axes[i][3].axis('off')

    plt.tight_layout()
    out_save_p = run_dir / "visualization_grid.png"
    plt.savefig(str(out_save_p), dpi=150, bbox_inches='tight')
    print(f"[PASSED] Saved comparative grid visualization -> {out_save_p}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, default=None, help='path to runs directory (e.g. /content/runs/tardal-tt)')
    parser.add_argument('--model', type=str, default='tardal-tt', help='checkpoint name (tardal-tt, tardal-dt, tardal-ct)')
    parser.add_argument('--split', type=str, default='val', help='split to sample from (val or test)')
    parser.add_argument('--num_samples', type=int, default=4, help='number of sample rows to visualize')
    args, _ = parser.parse_known_args()

    visualize_results(
        run_dir=Path(args.run_dir) if args.run_dir else None,
        num_samples=args.num_samples,
        model=args.model,
        split=args.split
    )