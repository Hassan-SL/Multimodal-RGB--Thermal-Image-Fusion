"""
Detector Training Script (YOLOv5su / YOLO11s).
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune YOLO detector on fused/unimodal data.")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Base model checkpoint (e.g. yolo11s.pt, yolov5su.pt)")
    parser.add_argument("--data", type=Path, default=Path("configs/m3fd_data.yaml"), help="Path to dataset YAML configuration")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    args = parser.parse_args()

    print(f"[*] Starting Detector Training:")
    print(f"  - Architecture: {args.model}")
    print(f"  - Dataset Config: {args.data}")
    print(f"  - Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}")
    
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )

if __name__ == "__main__":
    main()
