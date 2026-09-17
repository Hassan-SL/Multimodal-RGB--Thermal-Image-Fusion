"""
Decision-Level Late Fusion (Bounding Box Ensembling).
"""

import numpy as np
from typing import List, Dict, Any

class LateFusionEngine:
    @staticmethod
    def fuse_boxes(
        rgb_detections: Dict[str, Any],
        ir_detections: Dict[str, Any],
        w_rgb: float = 0.5,
        w_ir: float = 0.5,
        iou_thresh: float = 0.50
    ) -> Dict[str, Any]:
        """
        Aggregates predictions from RGB and IR detection heads using confidence weighting.
        """
        # Concatenate boxes, scores, classes
        all_boxes = rgb_detections.get("boxes", []) + ir_detections.get("boxes", [])
        all_scores = [s * w_rgb for s in rgb_detections.get("scores", [])] + [s * w_ir for s in ir_detections.get("scores", [])]
        all_classes = rgb_detections.get("classes", []) + ir_detections.get("classes", [])
        
        return {
            "boxes": all_boxes,
            "scores": all_scores,
            "classes": all_classes
        }
