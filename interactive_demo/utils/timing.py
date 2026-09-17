"""
High-precision latency profiling and timing utilities.
Enforces CUDA synchronization before and after timing blocks to ensure
hardware-accurate millisecond measurements without fabrication.
"""

import time
import torch
from typing import Dict, Optional


class Stopwatch:
    """Context manager and manual timer with hardware CUDA synchronization."""
    def __init__(self, use_cuda_sync: bool = True):
        self.use_cuda_sync = use_cuda_sync and torch.cuda.is_available()
        self.start_time: Optional[float] = None
        self.elapsed_ms: float = 0.0

    def start(self):
        if self.use_cuda_sync:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        if self.start_time is None:
            return 0.0
        if self.use_cuda_sync:
            torch.cuda.synchronize()
        self.elapsed_ms = max(0.01, (time.perf_counter() - self.start_time) * 1000.0)
        return self.elapsed_ms

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class PipelineTimer:
    """Tracks latency breakdown across the complete multimodal pipeline stages."""
    def __init__(self):
        self.stages: Dict[str, float] = {}
        self._active_timer: Optional[Stopwatch] = None
        self._current_stage: Optional[str] = None

    def start_stage(self, stage_name: str):
        if self._active_timer is not None:
            self.end_stage()
        self._current_stage = stage_name
        self._active_timer = Stopwatch().start()

    def end_stage(self):
        if self._active_timer is not None and self._current_stage is not None:
            elapsed = self._active_timer.stop()
            self.stages[self._current_stage] = round(max(0.01, elapsed), 2)
            self._active_timer = None
            self._current_stage = None

    def get_summary(self) -> Dict[str, float]:
        total = sum(self.stages.values())
        summary = dict(self.stages)
        summary['total_e2e_ms'] = round(total, 2)
        summary['fps'] = round(1000.0 / max(total, 0.01), 1)
        return summary
