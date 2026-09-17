"""
Mode B: Direct Thermal (IR) Pipeline.
IR -> 3-Channel Replication -> YOLOv5su -> Detections
"""

import cv2
import numpy as np
from typing import Optional
from .base_pipeline import BasePipeline, DetectionResult
from models.loader import get_model_manager
from utils.timing import PipelineTimer
from utils.visualization import draw_detections, CLASS_NAMES


class IRPipeline(BasePipeline):
    def __init__(self, checkpoint: str = "checkpoints/yolov5su_ir_best.pt"):
        super().__init__("IR — YOLOv5su")
        self.checkpoint = checkpoint
        self.manager = get_model_manager()

    def run(
        self,
        rgb_img: Optional[np.ndarray] = None,
        ir_img: Optional[np.ndarray] = None,
        conf: float = 0.25,
        iou: float = 0.50,
        **kwargs
    ) -> DetectionResult:
        if ir_img is None:
            raise ValueError("IR Pipeline requires a valid IR image input.")

        timer = PipelineTimer()

        # 1. Preprocessing stage (convert grayscale IR to 3-channel replicated format)
        timer.start_stage("preprocess_ms")
        if len(ir_img.shape) == 2:
            proc_img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)
        elif ir_img.shape[2] == 1:
            proc_img = cv2.cvtColor(ir_img.squeeze(), cv2.COLOR_GRAY2RGB)
        else:
            proc_img = ir_img.copy()
        timer.end_stage()

        # 2. Detector inference stage
        timer.start_stage("detector_ms")
        detector = self.manager.get_yolo_detector("yolov5su_ir", self.checkpoint)
        preds = detector.predict(proc_img, conf=conf, iou=iou)
        timer.end_stage()

        # 3. Visualization and postprocessing stage
        timer.start_stage("postprocess_ms")
        boxes = preds['boxes']
        scores = preds['scores']
        classes = preds['classes']
        annotated = draw_detections(proc_img, boxes, scores, classes)

        # Class counts
        class_counts = {name: 0 for name in CLASS_NAMES.values()}
        for c in classes:
            cname = CLASS_NAMES.get(int(c), f"Class {c}")
            class_counts[cname] = class_counts.get(cname, 0) + 1
        timer.end_stage()

        timing_summary = timer.get_summary()

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            classes=classes,
            annotated_image=annotated,
            fused_image=None,
            intermediate_data={"raw_detections": len(boxes)},
            timing=timing_summary,
            class_counts=class_counts,
            model_name=self.name,
            protocol="Operational" if conf >= 0.1 else "Academic"
        )
