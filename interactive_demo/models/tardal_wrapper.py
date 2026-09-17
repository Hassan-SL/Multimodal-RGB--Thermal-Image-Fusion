"""
TarDAL Dense-Block Generator Model Wrapper.
Implements the learned feature-level fusion network (296,577 parameters).
Strictly enforces eval() mode and requires_grad=False on all parameters.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Optional


class Generator(nn.Module):
    """
    TarDAL Generator Architecture (dim=32, depth=3).
    Fused infrared luminance and optical luminance via dense-block feature extraction.
    Input: ir (B, 1, H, W) in [0, 1], vi (B, 1, H, W) in [0, 1]
    Output: fus (B, 1, H, W) in [-1, +1] (Tanh activation)
    """
    def __init__(self, dim: int = 32, depth: int = 3):
        super(Generator, self).__init__()
        self.depth = depth

        self.encoder = nn.Sequential(
            nn.Conv2d(2, dim, (3, 3), (1, 1), 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.dense = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * (i + 1), dim, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            ) for i in range(depth)
        ])

        self.fuse = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(dim * (depth + 1), dim * 4, (3, 3), (1, 1), 1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim * 4, dim * 2, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(dim * 2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim, 1, (3, 3), (1, 1), 1),
                nn.Tanh()
            ),
        )

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        src = torch.cat([ir, vi], dim=1)
        x = self.encoder(src)
        for i in range(self.depth):
            t = self.dense[i](x)
            x = torch.cat([x, t], dim=1)
        fus = self.fuse(x)
        return fus


class TarDALGeneratorWrapper:
    """Wrapper managing initialization, checkpoint loading, and inference for TarDAL."""
    def __init__(self, checkpoint_path: Union[str, Path], device: torch.device):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.generator = Generator(dim=32, depth=3).to(self.device)
        self._load_and_freeze_weights()

    def _load_and_freeze_weights(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"TarDAL generator checkpoint not found at: {self.checkpoint_path}")

        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("g", checkpoint.get("fuse", checkpoint.get("generator", checkpoint)))
        else:
            state_dict = checkpoint

        cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.generator.load_state_dict(cleaned_state, strict=False)

        # Enforce strict freezing & evaluation state (Section 36 requirement)
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def fuse(self, ir_tensor: torch.Tensor, vis_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes forward pass.
        Args:
            ir_tensor: (1, 1, 640, 640) float32 in [0, 1]
            vis_tensor: (1, 1, 640, 640) float32 in [0, 1]
        Returns:
            fused_tensor: (1, 1, 640, 640) float32 in [-1, +1]
        """
        return self.generator(ir_tensor, vis_tensor)
