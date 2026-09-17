# Pipelines package
from .base_pipeline import DetectionResult, BasePipeline
from .rgb_pipeline import RGBPipeline
from .ir_pipeline import IRPipeline
from .stage3_pipeline import Stage3Pipeline
from .stage5_pipeline import Stage5Pipeline
from .stage6_pipeline import Stage6Pipeline

__all__ = [
    "DetectionResult", "BasePipeline",
    "RGBPipeline", "IRPipeline",
    "Stage3Pipeline", "Stage5Pipeline", "Stage6Pipeline"
]
