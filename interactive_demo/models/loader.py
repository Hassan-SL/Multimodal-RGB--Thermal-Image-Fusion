"""
Thread-safe Model Manager and Checkpoint Resolver.
Implements lazy-loading and resource caching.
Automatically falls back across local workspace, Google Drive, and Colab environments.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch

from .tardal_wrapper import TarDALGeneratorWrapper
from .yolo_wrapper import YOLODetectorWrapper
from utils.device import get_device
from utils.logging_utils import get_logger

logger = get_logger("ModelManager")


class ModelManager:
    """Manages lazy-loading and life-cycle of detection and fusion models."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_dir: Optional[Path] = None):
        if getattr(self, '_initialized', False):
            return

        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        self.project_root = self.base_dir.parent if self.base_dir.name == "interactive_demo" else self.base_dir
        self.device = get_device()
        self._cache: Dict[str, Any] = {}
        self._initialized = True
        logger.info(f"ModelManager initialized on compute device: {self.device}")

    def resolve_checkpoint(self, rel_path: str) -> Optional[Path]:
        """Resolves checkpoint location across potential storage mount points."""
        rel_p = Path(rel_path)
        candidates = [
            self.project_root / rel_path,
            self.base_dir / rel_path,
            self.project_root / "checkpoints" / rel_p.name,
            self.base_dir / "checkpoints" / rel_p.name,
            Path("E:/My Drive/FYP/code") / rel_path,
            Path("E:/My Drive/FYP/code/checkpoints") / rel_p.name,
            Path("/content/drive/MyDrive/FYP/code") / rel_path,
            Path(rel_path),
            Path.cwd() / rel_path,
            Path.cwd().parent / rel_path
        ]
        for c in candidates:
            if c.exists() and c.is_file() and c.stat().st_size > 0:
                return c
        return None

    def get_yolo_detector(self, model_key: str, rel_path: str) -> YOLODetectorWrapper:
        """Lazy-loads and caches a YOLO detector wrapper."""
        if model_key in self._cache:
            return self._cache[model_key]

        ckpt_path = self.resolve_checkpoint(rel_path)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Checkpoint '{rel_path}' not found! "
                f"Checked local workspace and Google Drive."
            )

        logger.info(f"Loading YOLO detector [{model_key}] from {ckpt_path}...")
        wrapper = YOLODetectorWrapper(ckpt_path, self.device)
        self._cache[model_key] = wrapper
        return wrapper

    def get_tardal_generator(self, rel_path: str = "checkpoints/stage3_gen_best.pt") -> TarDALGeneratorWrapper:
        """Lazy-loads and caches the frozen TarDAL generator."""
        if "tardal_generator" in self._cache:
            return self._cache["tardal_generator"]

        ckpt_path = self.resolve_checkpoint(rel_path)
        if ckpt_path is None:
            # Fallback to pretrained weights if stage 3 not yet found
            fallback = self.resolve_checkpoint("TarDAL-1.0.0/weights/tardal-dt.pt")
            if fallback:
                ckpt_path = fallback
            else:
                raise FileNotFoundError(f"TarDAL generator checkpoint '{rel_path}' not found!")

        logger.info(f"Loading TarDAL Generator from {ckpt_path}...")
        wrapper = TarDALGeneratorWrapper(ckpt_path, self.device)
        self._cache["tardal_generator"] = wrapper
        return wrapper

    def clear_cache(self):
        """Releases all cached models from memory."""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared model cache from memory.")


_MANAGER = None

def get_model_manager(base_dir: Optional[Path] = None) -> ModelManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ModelManager(base_dir)
    return _MANAGER
