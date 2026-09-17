"""
Thread-safe Model Manager and Checkpoint Resolver.
Implements lazy-loading and resource caching.
Automatically falls back across weights/, checkpoints/, Google Drive, and Colab environments.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import torch

from .tardal_wrapper import TarDALGeneratorWrapper
from .yolo_wrapper import YOLODetectorWrapper
from utils.device import get_device
from utils.logging_utils import get_logger

logger = get_logger("ModelManager")

# Common checkpoint filename aliases across research stages
ALIASES = {
    "best.pt": ["stage3_best.pt", "yolov5su_stage3_best.pt", "best.pt"],
    "stage3_best.pt": ["best.pt", "yolov5su_stage3_best.pt"],
    "yolov5su_stage3_best.pt": ["stage3_best.pt", "best.pt"],
    "stage6_yolo11s_best.pt": ["yolo11s_stage6_best.pt", "stage6_yolo11s_best.pt", "best.pt"],
    "yolo11s_stage6_best.pt": ["stage6_yolo11s_best.pt", "best.pt"],
    "stage3_gen_best.pt": ["tardal_generator.pth", "tardal-tt.pth", "stage3_gen_best.pt"],
    "tardal_generator.pth": ["tardal-tt.pth", "stage3_gen_best.pt"],
    "yolov5su_rgb_best.pt": ["yolov5su_rgb.pt", "rgb_best.pt"],
    "yolov5su_ir_best.pt": ["yolov5su_ir.pt", "ir_best.pt"]
}


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
        filename = rel_p.name

        # Generate list of filename variations
        search_names = [filename] + ALIASES.get(filename, [])

        search_dirs = [
            self.project_root / "weights",
            self.project_root / "checkpoints",
            self.project_root / "checkpoints" / "stage6",
            self.base_dir / "weights",
            self.base_dir / "checkpoints",
            self.project_root.parent / "code" / "checkpoints",
            self.project_root.parent / "code" / "checkpoints" / "stage6",
            self.project_root.parent / "code" / "TarDAL-main" / "weights" / "v1",
            Path("E:/My Drive/FYP/code/checkpoints"),
            Path("/content/drive/MyDrive/FYP/code/checkpoints"),
            self.project_root,
            Path.cwd()
        ]

        # 1. Direct path check
        direct = self.project_root / rel_path
        if direct.exists() and direct.is_file() and direct.stat().st_size > 0:
            return direct

        # 2. Search across dirs and aliases
        for sdir in search_dirs:
            if not sdir.exists():
                continue
            for name in search_names:
                cand = sdir / name
                if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
                    return cand

        return None

    def get_yolo_detector(self, model_key: str, rel_path: str) -> YOLODetectorWrapper:
        """Lazy-loads and caches a YOLO detector wrapper."""
        if model_key in self._cache:
            return self._cache[model_key]

        ckpt_path = self.resolve_checkpoint(rel_path)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Checkpoint '{rel_path}' not found! "
                f"Please download the pre-trained weights into the 'weights/' directory."
            )

        logger.info(f"Loading YOLO detector [{model_key}] from {ckpt_path}...")
        wrapper = YOLODetectorWrapper(ckpt_path, self.device)
        self._cache[model_key] = wrapper
        return wrapper

    def get_tardal_generator(self, rel_path: str = "weights/tardal_generator.pth") -> TarDALGeneratorWrapper:
        """Lazy-loads and caches the frozen TarDAL generator."""
        if "tardal_generator" in self._cache:
            return self._cache["tardal_generator"]

        ckpt_path = self.resolve_checkpoint(rel_path)
        if ckpt_path is None:
            ckpt_path = self.resolve_checkpoint("checkpoints/stage3_gen_best.pt")
        if ckpt_path is None:
            ckpt_path = self.resolve_checkpoint("tardal-tt.pth")

        if ckpt_path is None:
            raise FileNotFoundError(
                f"TarDAL generator checkpoint '{rel_path}' not found! "
                f"Please download 'tardal_generator.pth' into the 'weights/' directory."
            )

        logger.info(f"Loading TarDAL Generator from {ckpt_path}...")
        wrapper = TarDALGeneratorWrapper(ckpt_path, self.device)
        self._cache["tardal_generator"] = wrapper
        return wrapper

    def clear_cache(self):
        """Releases all cached models from memory."""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
