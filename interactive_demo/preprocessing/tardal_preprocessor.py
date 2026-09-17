"""
TarDAL Preprocessing and Reconstruction Engine.
Strictly preserves the research mathematical formulations:
- Visible: RGB -> YCrCb color space decomposition
- Infrared: Thermal luminance extraction
- TarDAL input normalization: [0, 1] range at 640x640 resolution
- Tanh remapping: Y_fused = clamp((fus + 1.0) / 2.0, 0.0, 1.0)
- Color reconstruction: Y_fused + Cr_vis + Cb_vis -> YCrCb -> RGB
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Any


class TarDALPreprocessor:
    def __init__(self, target_size: int = 640):
        self.target_size = target_size

    def prepare_tensors(
        self,
        rgb_img: np.ndarray,
        ir_img: np.ndarray,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Preprocesses synchronized RGB and IR images into TarDAL input tensors.
        
        Args:
            rgb_img: np.ndarray (H, W, 3) in RGB format.
            ir_img: np.ndarray (H, W) or (H, W, 3) in IR format.
            device: torch.device.

        Returns:
            ir_tensor: (1, 1, target_size, target_size) float32 in [0, 1]
            vis_tensor: (1, 1, target_size, target_size) float32 in [0, 1]
            meta: dictionary containing original dimensions and (Cr, Cb) channels
        """
        orig_h, orig_w = rgb_img.shape[:2]

        # Resize both to 640x640
        rgb_resized = cv2.resize(rgb_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        if len(ir_img.shape) == 3:
            ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_RGB2GRAY)
        else:
            ir_gray = ir_img
        ir_resized = cv2.resize(ir_gray, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # 1. Visible optical YCrCb decomposition
        ycrcb = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2YCrCb)
        y_vis = ycrcb[:, :, 0].astype(np.float32) / 255.0
        cr_vis = ycrcb[:, :, 1]
        cb_vis = ycrcb[:, :, 2]

        # 2. Infrared thermal luminance extraction
        y_ir = ir_resized.astype(np.float32) / 255.0

        # 3. Create torch tensors
        vis_tensor = torch.from_numpy(y_vis).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
        ir_tensor = torch.from_numpy(y_ir).unsqueeze(0).unsqueeze(0).to(device)    # (1, 1, H, W)

        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "cr_vis": cr_vis,
            "cb_vis": cb_vis,
            "rgb_resized": rgb_resized,
            "ir_resized": ir_resized
        }
        return ir_tensor, vis_tensor, meta

    def reconstruct_fused_image(
        self,
        y_fused_tensor: torch.Tensor,
        meta: Dict[str, Any]
    ) -> np.ndarray:
        """
        Remaps Tanh output from [-1, 1] to [0, 1] and reconstructs RGB using saved Cr/Cb.
        
        Formula:
            Y_fused_norm = clamp((Y_fused + 1.0) / 2.0, 0.0, 1.0)
            Fused_RGB = YCrCb_to_RGB(Y_fused_norm, Cr_vis, Cb_vis)
        """
        # Strict Tanh remapping
        y_norm = torch.clamp((y_fused_tensor + 1.0) / 2.0, 0.0, 1.0)
        y_np = (y_norm.squeeze().detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Recombine with chrominance channels
        cr_vis = meta["cr_vis"]
        cb_vis = meta["cb_vis"]
        fused_ycrcb = np.dstack((y_np, cr_vis, cb_vis))
        fused_rgb = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2RGB)

        # Resize back to original dimensions if needed
        if (meta["orig_h"], meta["orig_w"]) != (self.target_size, self.target_size):
            fused_rgb_orig = cv2.resize(fused_rgb, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_LINEAR)
            return fused_rgb_orig

        return fused_rgb
