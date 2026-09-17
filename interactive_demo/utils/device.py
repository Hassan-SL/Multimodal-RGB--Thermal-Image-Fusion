"""
Hardware device probe and GPU memory utilization utilities.
Never fabricates device metrics: returns accurate measurements or 'N/A'.
"""

import torch
from typing import Dict, Any


def get_device() -> torch.device:
    """Returns optimal torch compute device (cuda if available else cpu)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_device_info() -> Dict[str, Any]:
    """Returns hardware identification and capabilities."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cuda_ver = torch.version.cuda
        return {
            "type": "CUDA",
            "name": gpu_name,
            "total_vram_gb": round(total_vram, 2),
            "cuda_version": cuda_ver,
            "display": f"NVIDIA {gpu_name} ({total_vram:.1f} GB VRAM)"
        }
    return {
        "type": "CPU",
        "name": "Host Processor (CPU)",
        "total_vram_gb": None,
        "cuda_version": None,
        "display": "CPU Execution Mode"
    }


def get_memory_info() -> Dict[str, Any]:
    """Returns current allocated and reserved GPU memory in MB, or 'N/A' on CPU."""
    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
        return {
            "available": True,
            "allocated_mb": round(allocated_mb, 1),
            "reserved_mb": round(reserved_mb, 1),
            "display": f"{allocated_mb:.1f} MB"
        }
    return {
        "available": False,
        "allocated_mb": None,
        "reserved_mb": None,
        "display": "N/A"
    }
