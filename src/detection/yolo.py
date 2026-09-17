"""
YOLO Detector Interface for YOLOv5su and YOLO11s.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List

class YOLODetector:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        from ultralytics import YOLO
        self.model = YOLO(str(checkpoint_path))
        self.device = device

    def predict(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> Dict[str, Any]:
        results = self.model.predict(source=image, conf=conf, iou=iou, device=self.device, verbose=False)[0]
        boxes = []
        scores = []
        classes = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy().tolist()
            scores = results.boxes.conf.cpu().numpy().tolist()
            classes = results.boxes.cls.cpu().numpy().astype(int).tolist()
        return {"boxes": boxes, "scores": scores, "classes": classes}
