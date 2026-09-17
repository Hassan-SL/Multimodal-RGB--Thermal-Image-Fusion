"""
Standardized Detection Result and Abstract Base Pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class DetectionResult:
    """Standardized output container across all multi-modal detection pipelines."""
    boxes: List[List[float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    classes: List[int] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    fused_image: Optional[np.ndarray] = None
    intermediate_data: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    class_counts: Dict[str, int] = field(default_factory=dict)
    model_name: str = ""
    protocol: str = "Operational"

    @property
    def total_detections(self) -> int:
        return len(self.boxes)

    @property
    def fps(self) -> float:
        return self.timing.get('fps', 0.0)

    @property
    def latency_ms(self) -> float:
        return self.timing.get('total_e2e_ms', 0.0)


class BasePipeline(ABC):
    """Abstract interface for all detection and fusion pipelines."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(
        self,
        rgb_img: Optional[np.ndarray],
        ir_img: Optional[np.ndarray],
        conf: float = 0.25,
        iou: float = 0.50,
        **kwargs
    ) -> DetectionResult:
        """Executes full pipeline and returns standardized DetectionResult."""
        pass
