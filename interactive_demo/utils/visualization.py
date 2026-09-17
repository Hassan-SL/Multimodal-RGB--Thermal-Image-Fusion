"""
Computer vision visualization utilities for multi-modal object detection.
Renders clean bounding boxes, confidence tags, and sensor comparison panels.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Distinct high-contrast color palette for M3FD classes (BGR format for OpenCV)
CLASS_COLORS = {
    0: (50, 205, 50),     # People: Lime Green
    1: (0, 165, 255),     # Car: Bright Orange
    2: (255, 191, 0),     # Bus: Deep Sky Blue
    3: (0, 255, 255),     # Lamp: Vibrant Yellow
    4: (238, 130, 238),   # Motorcycle: Violet / Magenta
    5: (0, 69, 255)       # Truck: Crimson Red
}

CLASS_NAMES = {
    0: "People",
    1: "Car",
    2: "Bus",
    3: "Lamp",
    4: "Motorcycle",
    5: "Truck"
}


def draw_detections(
    image: np.ndarray,
    boxes: List[List[float]],
    scores: List[float],
    classes: List[int],
    line_thickness: int = 2,
    font_scale: float = 0.55
) -> np.ndarray:
    """
    Renders bounding boxes and label tags onto an image (RGB format).
    Boxes format: [[x1, y1, x2, y2], ...]
    """
    vis = image.copy()
    H, W = vis.shape[:2]

    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        color = CLASS_COLORS.get(int(cls_id), (200, 200, 200))
        # Draw bounding rectangle
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_thickness)

        # Label tag text
        cname = CLASS_NAMES.get(int(cls_id), f"Class {cls_id}")
        label = f"{cname} {score:.2f}"

        # Background text pill
        (txt_w, txt_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        pill_y1 = max(0, y1 - txt_h - 6)
        pill_y2 = y1
        pill_x2 = min(W - 1, x1 + txt_w + 6)

        cv2.rectangle(vis, (x1, pill_y1), (pill_x2, pill_y2), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text for high readability on bright pill
            1,
            cv2.LINE_AA
        )

    return vis


def create_comparison_grid(
    rgb_img: np.ndarray,
    ir_img: np.ndarray,
    fused_img: Optional[np.ndarray],
    detection_img: np.ndarray,
    target_width: int = 640
) -> np.ndarray:
    """
    Constructs a 2x2 or 1x3 research comparative panel for visual inspection.
    """
    def resize_h(img, w):
        h = int(img.shape[0] * (w / img.shape[1]))
        return cv2.resize(img, (w, h))

    r_rgb = resize_h(rgb_img, target_width)
    r_ir = resize_h(ir_img, target_width) if len(ir_img.shape) == 3 else resize_h(cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB), target_width)
    r_det = resize_h(detection_img, target_width)

    if fused_img is not None:
        r_fuse = resize_h(fused_img, target_width)
        top_row = np.hstack([r_rgb, r_ir])
        bot_row = np.hstack([r_fuse, r_det])
        return np.vstack([top_row, bot_row])
    else:
        top_row = np.hstack([r_rgb, r_ir])
        return np.vstack([top_row, np.hstack([r_det, r_det])])
