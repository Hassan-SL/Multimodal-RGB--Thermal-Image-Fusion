"""
Image and video file validation and synchronized pair loading.
Ensures rigid multi-modal sensor alignment checks without arbitrary transforms.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def validate_image_file(filename: str) -> bool:
    """Checks if file extension is supported."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def load_image_from_bytes(file_bytes: bytes) -> Optional[np.ndarray]:
    """Decodes raw image bytes to RGB numpy array."""
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def load_and_align_pair(
    rgb_source: Union[str, Path, bytes, np.ndarray],
    ir_source: Union[str, Path, bytes, np.ndarray]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Loads synchronized RGB and IR frames and validates dimensional alignment.
    
    Returns:
        (rgb_img, ir_img, error_message)
    """
    # Load RGB
    if isinstance(rgb_source, (str, Path)):
        p = Path(rgb_source)
        if not p.exists():
            return None, None, f"RGB file not found: {p}"
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            return None, None, f"Could not decode RGB image: {p}"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(rgb_source, bytes):
        rgb = load_image_from_bytes(rgb_source)
        if rgb is None:
            return None, None, "Could not decode uploaded RGB bytes."
    elif isinstance(rgb_source, np.ndarray):
        rgb = rgb_source
    else:
        return None, None, "Invalid RGB source type."

    # Load IR
    if isinstance(ir_source, (str, Path)):
        p = Path(ir_source)
        if not p.exists():
            return None, None, f"IR file not found: {p}"
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            return None, None, f"Could not decode IR image: {p}"
        ir = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(ir_source, bytes):
        ir = load_image_from_bytes(ir_source)
        if ir is None:
            return None, None, "Could not decode uploaded IR bytes."
    elif isinstance(ir_source, np.ndarray):
        ir = ir_source
    else:
        return None, None, "Invalid IR source type."

    # Dimensional Alignment Check (Section 12 requirement)
    if rgb.shape[:2] != ir.shape[:2]:
        return rgb, ir, (
            f"Dimension mismatch between sensors! "
            f"RGB resolution: {rgb.shape[1]}x{rgb.shape[0]}, "
            f"IR resolution: {ir.shape[1]}x{ir.shape[0]}. "
            f"Strict multi-modal alignment requires identical sensor geometry."
        )

    return rgb, ir, None
