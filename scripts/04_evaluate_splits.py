"""
Validation and Test Evaluation Script.
Computes official COCO mAP@50 and mAP@50:95 across specified dataset splits.
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned detector on Validation or Test splits.")
    parser.add_argument("--weights", type=Path, default=Path("weights/yolo11s_stage6_best.pt"), help="Path to detector checkpoint")
    parser.add_argument("--data", type=Path, default=Path("configs/m3fd_data.yaml"), help="Path to dataset YAML configuration")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for benchmark evaluation (default: 0.001)")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold for evaluation")
    args = parser.parse_args()

    print(f"[*] Benchmarking Model [{args.weights.name}] on split [{args.split}]:")
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        verbose=True
    )
    print(f"[+] Results for {args.split}:")
    print(f"    mAP@50:    {metrics.box.map50 * 100:.2f}%")
    print(f"    mAP@50:95: {metrics.box.map * 100:.2f}%")

if __name__ == "__main__":
    main()
