"""
Mode E: Decision-Level Late Fusion (Stage 5 WBF with Sensor Dropout Simulation).
RGB -> YOLOv5su ──┐
                  ├── WBF (w_rgb=0.60, w_ir=0.40) -> Final Consensus Detections
IR  -> YOLOv5su ──┘
Supports live simulation of single-sensor failure (Graceful Degradation).
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
from .base_pipeline import BasePipeline, DetectionResult
from models.loader import get_model_manager
from models.late_fusion import apply_weighted_boxes_fusion
from utils.timing import PipelineTimer
from utils.visualization import draw_detections, CLASS_NAMES


class Stage5Pipeline(BasePipeline):
    def __init__(
        self,
        rgb_ckpt: str = "checkpoints/yolov5su_rgb_best.pt",
        ir_ckpt: str = "checkpoints/yolov5su_ir_best.pt",
        default_w_rgb: float = 0.60,
        default_w_ir: float = 0.40
    ):
        super().__init__("RGB + IR Late Fusion (WBF)")
        self.rgb_ckpt = rgb_ckpt
        self.ir_ckpt = ir_ckpt
        self.default_w_rgb = default_w_rgb
        self.default_w_ir = default_w_ir
        self.manager = get_model_manager()

    def run(
        self,
        rgb_img: Optional[np.ndarray],
        ir_img: Optional[np.ndarray],
        conf: float = 0.25,
        iou: float = 0.50,
        w_rgb: Optional[float] = None,
        w_ir: Optional[float] = None,
        rgb_active: bool = True,
        ir_active: bool = True,
        **kwargs
    ) -> DetectionResult:
        w_rgb = self.default_w_rgb if w_rgb is None else w_rgb
        w_ir = self.default_w_ir if w_ir is None else w_ir

        timer = PipelineTimer()
        timer.start_stage("preprocess_ms")

        # Prepare IR 3-channel
        ir_proc = None
        if ir_img is not None:
            if len(ir_img.shape) == 2:
                ir_proc = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)
            elif ir_img.shape[2] == 1:
                ir_proc = cv2.cvtColor(ir_img.squeeze(), cv2.COLOR_GRAY2RGB)
            else:
                ir_proc = ir_img.copy()

        rgb_proc = rgb_img.copy() if rgb_img is not None else None
        timer.end_stage()

        # Handle sensor dropout states
        rgb_preds = {"boxes": [], "scores": [], "classes": []}
        ir_preds = {"boxes": [], "scores": [], "classes": []}

        # 1. Optical Stream (if sensor online)
        if rgb_active and rgb_proc is not None:
            timer.start_stage("detector_rgb_ms")
            rgb_detector = self.manager.get_yolo_detector("yolov5su_rgb", self.rgb_ckpt)
            rgb_preds = rgb_detector.predict(rgb_proc, conf=conf, iou=iou)
            timer.end_stage()

        # 2. Thermal Stream (if sensor online)
        if ir_active and ir_proc is not None:
            timer.start_stage("detector_ir_ms")
            ir_detector = self.manager.get_yolo_detector("yolov5su_ir", self.ir_ckpt)
            ir_preds = ir_detector.predict(ir_proc, conf=conf, iou=iou)
            timer.end_stage()

        # 3. Decision Fusion / Sensor Fallback Stage
        timer.start_stage("wbf_merge_ms")
        status_message = "Normal Dual-Sensor Fusion"

        if rgb_active and ir_active:
            # Both sensors active -> Full WBF Consensus
            fusion_res = apply_weighted_boxes_fusion(
                rgb_preds, ir_preds,
                w_rgb=w_rgb, w_ir=w_ir,
                iou_thresh=iou, conf_thresh=conf
            )
            final_boxes = fusion_res['boxes']
            final_scores = fusion_res['scores']
            final_classes = fusion_res['classes']
            matched_pairs = fusion_res['num_matched']
            status_message = f"Full Multi-Modal Consensus (WBF: {w_rgb:.2f} RGB / {w_ir:.2f} IR)"

        elif rgb_active and not ir_active:
            # IR failure -> Fallback to surviving RGB stream
            final_boxes = rgb_preds['boxes']
            final_scores = rgb_preds['scores']
            final_classes = rgb_preds['classes']
            matched_pairs = 0
            status_message = "IR OFFLINE: Graceful fallback to surviving Visible Optical sensor"

        elif ir_active and not rgb_active:
            # RGB failure -> Fallback to surviving IR stream
            final_boxes = ir_preds['boxes']
            final_scores = ir_preds['scores']
            final_classes = ir_preds['classes']
            matched_pairs = 0
            status_message = "RGB OFFLINE: Graceful fallback to surviving Thermal Infrared sensor"

        else:
            final_boxes = []
            final_scores = []
            final_classes = []
            matched_pairs = 0
            status_message = "ALL SENSORS OFFLINE: Perception unavailable"

        timer.end_stage()

        # 4. Visualization
        timer.start_stage("postprocess_ms")
        base_canvas = rgb_proc if rgb_proc is not None else ir_proc
        if base_canvas is None:
            base_canvas = np.zeros((640, 640, 3), dtype=np.uint8)

        annotated = draw_detections(base_canvas, final_boxes, final_scores, final_classes)

        # Annotated individual stream frames for research inspection
        annotated_rgb = draw_detections(rgb_proc, rgb_preds['boxes'], rgb_preds['scores'], rgb_preds['classes']) if rgb_proc is not None else None
        annotated_ir = draw_detections(ir_proc, ir_preds['boxes'], ir_preds['scores'], ir_preds['classes']) if ir_proc is not None else None

        class_counts = {name: 0 for name in CLASS_NAMES.values()}
        for c in final_classes:
            cname = CLASS_NAMES.get(int(c), f"Class {c}")
            class_counts[cname] = class_counts.get(cname, 0) + 1
        timer.end_stage()

        timing_summary = timer.get_summary()

        return DetectionResult(
            boxes=final_boxes,
            scores=final_scores,
            classes=final_classes,
            annotated_image=annotated,
            fused_image=None,
            intermediate_data={
                "rgb_active": rgb_active,
                "ir_active": ir_active,
                "rgb_detections": len(rgb_preds['boxes']),
                "ir_detections": len(ir_preds['boxes']),
                "fused_detections": len(final_boxes),
                "matched_pairs": matched_pairs,
                "status_message": status_message,
                "annotated_rgb": annotated_rgb,
                "annotated_ir": annotated_ir
            },
            timing=timing_summary,
            class_counts=class_counts,
            model_name=self.name,
            protocol="Operational" if conf >= 0.1 else "Academic"
        )
