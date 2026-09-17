"""
Mode C: TarDAL + YOLOv5su (Stage 3 Task-Driven Feature Fusion).
RGB + IR -> YCrCb Preprocessing -> TarDAL Generator -> Reconstruction -> YOLOv5su -> Detections
"""

import numpy as np
from typing import Optional
from .base_pipeline import BasePipeline, DetectionResult
from models.loader import get_model_manager
from preprocessing.tardal_preprocessor import TarDALPreprocessor
from utils.timing import PipelineTimer
from utils.visualization import draw_detections, CLASS_NAMES


class Stage3Pipeline(BasePipeline):
    def __init__(
        self,
        generator_ckpt: str = "checkpoints/stage3_gen_best.pt",
        detector_ckpt: str = "checkpoints/best.pt"
    ):
        super().__init__("TarDAL + YOLOv5su")
        self.generator_ckpt = generator_ckpt
        self.detector_ckpt = detector_ckpt
        self.manager = get_model_manager()
        self.preprocessor = TarDALPreprocessor(target_size=640)

    def run(
        self,
        rgb_img: Optional[np.ndarray],
        ir_img: Optional[np.ndarray],
        conf: float = 0.25,
        iou: float = 0.50,
        **kwargs
    ) -> DetectionResult:
        if rgb_img is None or ir_img is None:
            raise ValueError("Stage 3 Feature Fusion requires synchronized RGB AND IR inputs.")

        timer = PipelineTimer()

        # 1. Preprocessing stage (YCrCb split & tensor creation)
        timer.start_stage("preprocess_ms")
        ir_tensor, vis_tensor, meta = self.preprocessor.prepare_tensors(
            rgb_img, ir_img, self.manager.device
        )
        timer.end_stage()

        # 2. Generator forward pass (TarDAL feature fusion)
        timer.start_stage("generator_ms")
        generator = self.manager.get_tardal_generator(self.generator_ckpt)
        fused_y = generator.fuse(ir_tensor, vis_tensor)
        timer.end_stage()

        # 3. Chrominance color reconstruction (Tanh remapping -> YCrCb -> RGB)
        timer.start_stage("reconstruction_ms")
        fused_rgb = self.preprocessor.reconstruct_fused_image(fused_y, meta)
        timer.end_stage()

        # 4. YOLOv5su detection inference on fused image
        timer.start_stage("detector_ms")
        detector = self.manager.get_yolo_detector("yolov5su_stage3", self.detector_ckpt)
        preds = detector.predict(fused_rgb, conf=conf, iou=iou)
        timer.end_stage()

        # 5. Visualization and class count summary
        timer.start_stage("postprocess_ms")
        boxes = preds['boxes']
        scores = preds['scores']
        classes = preds['classes']
        annotated = draw_detections(fused_rgb, boxes, scores, classes)

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
            fused_image=fused_rgb,
            intermediate_data={
                "generator_params": "296,577 (Frozen)",
                "detector_params": f"{detector.parameter_count_m}M",
                "raw_detections": len(boxes)
            },
            timing=timing_summary,
            class_counts=class_counts,
            model_name=self.name,
            protocol="Operational" if conf >= 0.1 else "Academic"
        )
