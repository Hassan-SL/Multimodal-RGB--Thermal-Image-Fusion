"""
Visualization and Bounding Box Rendering Utilities.
"""

import cv2
import numpy as np
from typing import List, Dict, Any

CLASS_NAMES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
CLASS_COLORS = [
    (239, 68, 68),   # People: Red
    (59, 130, 246),  # Car: Blue
    (16, 185, 129),  # Bus: Green
    (245, 158, 11),  # Motorcycle: Orange
    (168, 85, 247),  # Lamp: Purple
    (236, 72, 153)   # Truck: Pink
]

def draw_detections(image: np.ndarray, detections: Dict[str, Any]) -> np.ndarray:
    out = image.copy()
    boxes = detections.get("boxes", [])
    scores = detections.get("scores", [])
    classes = detections.get("classes", [])

    for b, s, c in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, b)
        cls_id = int(c) % len(CLASS_NAMES)
        label = f"{CLASS_NAMES[cls_id]} {s:.2f}"
        color = CLASS_COLORS[cls_id]
        
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out
