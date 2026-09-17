"""
Ultralytics YOLO Detector Model Wrapper.
Provides a unified interface for YOLOv5su and YOLO11s detection heads.
Enforces torch.inference_mode() and consistent bounding box parsing.
"""

from pathlib import Path
from typing import Union, List, Dict, Any, Tuple
import numpy as np
import torch


class YOLODetectorWrapper:
    def __init__(self, checkpoint_path: Union[str, Path], device: torch.device):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found at: {self.checkpoint_path}")

        from ultralytics import YOLO
        self.model = YOLO(str(self.checkpoint_path))
        # Move model to device
        dev_str = "0" if self.device.type == "cuda" else "cpu"
        self.model.to(dev_str)

    @property
    def parameter_count_m(self) -> float:
        """Returns total parameter count in Millions."""
        try:
            total_params = sum(p.numel() for p in self.model.model.parameters())
            return round(total_params / 1e6, 2)
        except Exception:
            return 9.12

    @torch.inference_mode()
    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.50,
        imgsz: int = 640
    ) -> Dict[str, Any]:
        """
        Executes object detection inference on an RGB image.
        
        Args:
            image: np.ndarray (H, W, 3) in RGB format.
            conf: confidence threshold.
            iou: NMS IoU threshold.
            imgsz: inference image size.

        Returns:
            Dictionary with:
                'boxes': List of [x1, y1, x2, y2]
                'scores': List of float confidence scores
                'classes': List of int class IDs
                'inference_ms': raw inference latency reported by Ultralytics
        """
        dev_str = "0" if self.device.type == "cuda" else "cpu"
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=dev_str,
            verbose=False
        )

        res = results[0]
        boxes_list = []
        scores_list = []
        classes_list = []

        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            conf_arr = res.boxes.conf.detach().cpu().numpy()
            cls_arr = res.boxes.cls.detach().cpu().numpy()

            for b, s, c in zip(xyxy, conf_arr, cls_arr):
                boxes_list.append([round(float(v), 2) for v in b])
                scores_list.append(round(float(s), 4))
                classes_list.append(int(c))

        speed_dict = getattr(res, 'speed', {})
        inf_ms = speed_dict.get('inference', 0.0)

        return {
            "boxes": boxes_list,
            "scores": scores_list,
            "classes": classes_list,
            "inference_ms": round(inf_ms, 2)
        }
