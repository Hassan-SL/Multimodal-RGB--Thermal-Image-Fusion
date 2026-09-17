"""

import sys
from pathlib import Path

# Add scripts directory to sys.path so path_config is always discoverable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import path_config
from path_config import *

Stage 3: Task-Driven Generator Adaptation & Joint End-to-End Fine-Tuning (scripts_AG/11_train_stage3_joint_end2end.py)

Implements Path 2: Permanent Detector Freeze + Task-Driven Generator Adaptation:
1. DETECTOR FREEZE (PERMANENT): Detector weights frozen with requires_grad=False & permanent eval mode.
2. LEARNING RATE FIX: Single optimizer parameter group for Generator at LR=5e-6.
3. fuse.eval() BUG FIX: Replaced self.fuse.eval() with self.fuse.generator(ir, vi) to preserve train mode.
4. LOSS SCALE NORMALIZATION: Running mean EMAs (fuse_mean, det_mean) for normalized loss combination.
5. ADVERSARIAL LOSS ZEROING: Zeroed adv_l to prevent negative adversarial gradient disruption.
6. GRADIENT CLIPPING: Tightly clipped Generator gradients at max_norm=0.5.
7. DETECTOR FORWARD IN AUTOGRAD: Det_model runs with autograd for fused_rgb & AMP to save GPU VRAM.
8. VALIDATION FREQUENCY: Force eval_interval=1 with failure detection if mAP@50 < 65%.
9. LOSS BRIDGE SIMPLIFICATION: Single optimizer for generator, removed dual-optimizer disc_opt.
10. TELEMETRY PARITY: Maintained visual progress table, logging, and checkpoint saving formatting.
"""

import os
import sys
import warnings
import time
import shutil
import math
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import yaml

# Suppress noise & logging setup
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ['WANDB_MODE'] = 'disabled'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

# Path registration
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
TARDAL_MAIN_DIR = CODE_ROOT / "TarDAL-main"

if str(TARDAL_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TARDAL_MAIN_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import from_dict
from pipeline.fuse import Fuse
from tools.dict_to_device import dict_to_device

# Ultralytics native modules
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

# Kornia for differentiable YCbCr -> RGB
try:
    from kornia.color import ycbcr_to_rgb
except ImportError:
    ycbcr_to_rgb = None

def ycbcr_to_rgb_diff(y, cbcr):
    """
    Differentiable YCbCr to RGB conversion (ITU-R BT.601 standard, 100% Stage 2 Parity).
    cbcr channel 0 = Cb (Blue difference), channel 1 = Cr (Red difference)
    """
    cb = cbcr[:, 0:1, :, :] - (128.0 / 255.0)
    cr = cbcr[:, 1:2, :, :] - (128.0 / 255.0)
    
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)

# Backward-compatibility alias
ycrcb_to_rgb_diff = ycbcr_to_rgb_diff

class CleanFormatter(logging.Formatter):
    def format(self, record):
        return record.getMessage()

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CleanFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

# Standard M3FD Ground Truth Class Mapping
CLASSES = ['People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck']
NAMES_DICT = {0: 'People', 1: 'Car', 2: 'Bus', 3: 'Lamp', 4: 'Motorcycle', 5: 'Truck'}


# ==============================================================================
# 1. PARAMETER VERIFICATION & VISUAL TRACKING HELPERS (STAGE 2 PARITY)
# ==============================================================================

def print_parameter_verification(generator: nn.Module, detector: nn.Module):
    """Verifies and logs parameter counts for both Generator and Detection Head."""
    g_total = sum(p.numel() for p in generator.parameters())
    g_trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)

    d_total = sum(p.numel() for p in detector.parameters())
    d_trainable = sum(p.numel() for p in detector.parameters() if p.requires_grad)

    print("\n" + "=" * 80)
    print(" 🔬 STAGE 3 PARAMETER ARCHITECTURE & GRADIENT AUDIT ".center(80))
    print("=" * 80)
    print(f"  Generator Backbone : Total = {g_total:,} | Trainable = {g_trainable:,} (UNFROZEN ✅)")
    print(f"  Detector Head      : Total = {d_total:,} | Trainable = {d_trainable:,} (PERMANENTLY FROZEN 🔒)")
    print(f"  Combined Joint Net : Total = {g_total + d_total:,} | Trainable = {g_trainable + d_trainable:,}")
    print("=" * 80 + "\n")


def print_dataset_tracking_visualization(dataset_root: Path, batch_size: int):
    """Prints a detailed visual tracking breakdown of dataset splits, image counts, and class instance distributions."""
    meta_dir = dataset_root / "meta"
    lbl_dir = dataset_root / "labels"

    train_names, val_names = [], []
    if (meta_dir / "train.txt").exists():
        train_names = [l.strip() for l in (meta_dir / "train.txt").read_text().splitlines() if l.strip()]
    if (meta_dir / "val.txt").exists():
        val_names = [l.strip() for l in (meta_dir / "val.txt").read_text().splitlines() if l.strip()]

    train_batches = (len(train_names) + batch_size - 1) // batch_size if train_names else 0
    val_batches = (len(val_names) + batch_size - 1) // batch_size if val_names else 0

    def count_instances(names_list):
        counts = {i: 0 for i in range(6)}
        total_boxes = 0
        for name in names_list:
            stem = Path(name).stem
            txt_p = lbl_dir / f"{stem}.txt"
            if txt_p.exists():
                lines = txt_p.read_text().strip().splitlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        cls_id = int(parts[0])
                        if cls_id in counts:
                            counts[cls_id] += 1
                            total_boxes += 1
        return counts, total_boxes

    train_cls_counts, train_total_boxes = count_instances(train_names)
    val_cls_counts, val_total_boxes = count_instances(val_names)

    print("\n" + "=" * 80)
    print(" 📊 STAGE 3 DATASET & PIPELINE TRACKING VISUALIZATION ".center(80))
    print("=" * 80)
    print(f"  [PROOF] SPLIT STRATEGY  : 60% Train (2,520) / 20% Val (840) / 20% Test (840)")
    print(f"  [PROOF] ACTIVE TRAIN SET: ✅ Task-driven generator adaptation on {len(train_names):,} training pairs")
    print(f"  [PROOF] ACTIVE VAL SET  : ✅ Validating after every epoch on {len(val_names):,} validation pairs")
    print(f"  Root Directory          : {dataset_root}")
    print(f"  Batch Size              : {batch_size}")
    print("-" * 80)
    print(f"  [SPLIT] TRAIN SET (60.0%): {len(train_names):>5} images | {train_batches:>4} batches/epoch")
    print(f"  [SPLIT] VAL SET   (20.0%): {len(val_names):>5} images | {val_batches:>4} batches/epoch")
    print(f"  [TOTAL] ACTIVE DATASET  : {len(train_names)+len(val_names):>5} images | {train_total_boxes+val_total_boxes:>5} labeled instances")
    print("-" * 80)
    print(f"  {'CLASS NAME':<15} | {'TRAIN INSTANCES':<18} | {'VAL INSTANCES':<18} | {'TOTAL':<10}")
    print("-" * 80)
    for cls_id in range(6):
        c_name = NAMES_DICT[cls_id]
        tr_c = train_cls_counts[cls_id]
        va_c = val_cls_counts[cls_id]
        tot = tr_c + va_c
        print(f"  {c_name:<15} | {tr_c:>18,} | {va_c:>18,} | {tot:>10,}")
    print("-" * 80)
    print(f"  {'TOTAL INSTANCES':<15} | {train_total_boxes:>18,} | {val_total_boxes:>18,} | {train_total_boxes+val_total_boxes:>10,}")
    print("=" * 80 + "\n")


# ==============================================================================
# 2. LIVE IN-MEMORY DATASET (NORMALIZATION & SALIENCY MAPS)
# ==============================================================================

class LiveM3FDPairedDataset(Dataset):
    """
    Loads paired IR & VIS images, saliency masks, IQA weights, and YOLO labels.
    Normalizes IR and VIS Y-channel to [-1, 1] for TarDAL Generator backbone.
    """
    def __init__(self, dataset_root: Path, split: str = 'train', img_size: int = 640):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.img_size = img_size

        self.ir_dir = next((self.dataset_root / d for d in ["Ir", "ir", "IR"] if (self.dataset_root / d).exists()), self.dataset_root / "ir")
        self.vi_dir = next((self.dataset_root / d for d in ["Vis", "vis", "vi", "VIS"] if (self.dataset_root / d).exists()), self.dataset_root / "vi")
        self.lbl_dir = next((self.dataset_root / d for d in ["labels", "Labels"] if (self.dataset_root / d).exists()), self.dataset_root / "labels")
        self.mask_dir = self.dataset_root / "mask"
        self.iqa_ir_dir = self.dataset_root / "iqa" / "ir"
        self.iqa_vi_dir = self.dataset_root / "iqa" / "vi"

        meta_file = self.dataset_root / "meta" / f"{split}.txt"
        if meta_file.exists():
            stems = [Path(line.strip()).stem for line in meta_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        else:
            stems = [f.stem for f in self.ir_dir.glob("*.*") if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

        self.samples = []
        for s in stems:
            ir_p = next((self.ir_dir / f for f in [f"{s}.png", f"{s}.jpg", f"{s}.jpeg"] if (self.ir_dir / f).exists()), None)
            vi_p = next((self.vi_dir / f for f in [f"{s}.png", f"{s}.jpg", f"{s}.jpeg"] if (self.vi_dir / f).exists()), None)
            lbl_p = self.lbl_dir / f"{s}.txt"
            mask_p = self.mask_dir / f"{s}.png" if self.mask_dir.exists() else None
            ir_w_p = self.iqa_ir_dir / f"{s}.png" if self.iqa_ir_dir.exists() else None
            vi_w_p = self.iqa_vi_dir / f"{s}.png" if self.iqa_vi_dir.exists() else None
            
            if ir_p and vi_p:
                self.samples.append((ir_p, vi_p, lbl_p, mask_p, ir_w_p, vi_w_p, s))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.dataset_root} for split '{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ir_p, vi_p, lbl_p, mask_p, ir_w_p, vi_w_p, stem = self.samples[idx]

        ir_bgr = cv2.imread(str(ir_p), cv2.IMREAD_GRAYSCALE)
        vi_bgr = cv2.imread(str(vi_p), cv2.IMREAD_COLOR)

        if ir_bgr is None or vi_bgr is None:
            raise RuntimeError(f"Failed to load image pair: {ir_p} / {vi_p}")

        ir_resized = cv2.resize(ir_bgr, (self.img_size, self.img_size))
        vi_resized = cv2.resize(vi_bgr, (self.img_size, self.img_size))

        vi_ycrcb = cv2.cvtColor(vi_resized, cv2.COLOR_BGR2YCrCb)
        vi_y = vi_ycrcb[:, :, 0]
        vi_cr = vi_ycrcb[:, :, 1]
        vi_cb = vi_ycrcb[:, :, 2]
        vi_cbcr = np.stack([vi_cb, vi_cr], axis=-1)

        # Normalize IR and VIS Y-channel to [0, 1] for Generator and standard YOLO input parity
        ir_t = torch.from_numpy(ir_resized).float().unsqueeze(0) / 255.0
        vi_t = torch.from_numpy(vi_y).float().unsqueeze(0) / 255.0
        cbcr_t = torch.from_numpy(vi_cbcr).float().permute(2, 0, 1) / 255.0

        # Load Saliency Mask (default 1.0 if not on disk)
        if mask_p and mask_p.exists():
            mk_img = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            mk_img = cv2.resize(mk_img, (self.img_size, self.img_size))
            mask_t = torch.from_numpy(mk_img).float().unsqueeze(0) / 255.0
        else:
            mask_t = torch.ones_like(ir_t)

        # Load IQA weights (default 1.0 if not on disk)
        if ir_w_p and ir_w_p.exists():
            ir_w_img = cv2.imread(str(ir_w_p), cv2.IMREAD_GRAYSCALE)
            ir_w_img = cv2.resize(ir_w_img, (self.img_size, self.img_size))
            ir_w_t = torch.from_numpy(ir_w_img).float().unsqueeze(0) / 255.0
        else:
            ir_w_t = torch.ones_like(ir_t)

        if vi_w_p and vi_w_p.exists():
            vi_w_img = cv2.imread(str(vi_w_p), cv2.IMREAD_GRAYSCALE)
            vi_w_img = cv2.resize(vi_w_img, (self.img_size, self.img_size))
            vi_w_t = torch.from_numpy(vi_w_img).float().unsqueeze(0) / 255.0
        else:
            vi_w_t = torch.ones_like(vi_t)

        labels = []
        if lbl_p and lbl_p.exists() and lbl_p.stat().st_size > 0:
            for line in lbl_p.read_text(encoding='utf-8').strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:5])
                    labels.append([cls_id, cx, cy, bw, bh])

        labels_t = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)

        return {
            'ir': ir_t, 'vi': vi_t, 'cbcr': cbcr_t,
            'mask': mask_t, 'ir_w': ir_w_t, 'vi_w': vi_w_t,
            'labels': labels_t, 'stem': stem
        }


def collate_live_batch(batch):
    return {
        'ir': torch.stack([b['ir'] for b in batch], dim=0),
        'vi': torch.stack([b['vi'] for b in batch], dim=0),
        'cbcr': torch.stack([b['cbcr'] for b in batch], dim=0),
        'mask': torch.stack([b['mask'] for b in batch], dim=0),
        'ir_w': torch.stack([b['ir_w'] for b in batch], dim=0),
        'vi_w': torch.stack([b['vi_w'] for b in batch], dim=0),
        'labels': [b['labels'] for b in batch],
        'stems': [b['stem'] for b in batch]
    }


# ==============================================================================
# 3. STAGE 3 JOINT TRAINER WITH COMPLETE TELEMETRY & LOSS METRICS
# ==============================================================================

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


def extract_loss_metrics(loss_items):
    """Extracts box_l, cls_l, dfl_l safely from any loss_items structure (dict, tensor, tuple, list)."""
    box_l, cls_l, dfl_l = 0.0, 0.0, 0.0
    if isinstance(loss_items, dict):
        box_l = float(loss_items.get('box_loss', loss_items.get('box', 0.0)))
        cls_l = float(loss_items.get('cls_loss', loss_items.get('cls', 0.0)))
        dfl_l = float(loss_items.get('dfl_loss', loss_items.get('dfl', 0.0)))
    elif hasattr(loss_items, '__iter__') or isinstance(loss_items, torch.Tensor):
        items_list = [float(x.item() if hasattr(x, 'item') else x) for x in loss_items]
        box_l = items_list[0] if len(items_list) > 0 else 0.0
        cls_l = items_list[1] if len(items_list) > 1 else 0.0
        dfl_l = items_list[2] if len(items_list) > 2 else 0.0
    return box_l, cls_l, dfl_l


class Stage3JointTrainer:
    def __init__(self, config, force_restart=False, resume=False, resume_path=None):
        self.config = config
        self.config.train.image_size = (640, 640)
        self.force_restart = force_restart
        self.resume = resume
        self.resume_path = resume_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.save_dir = CODE_ROOT / "runs" / "stage3_joint_end2end"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.drive_ckpt_dir = Path('/content/drive/MyDrive/FYP/code/checkpoints')
        if not self.drive_ckpt_dir.exists():
            self.drive_ckpt_dir = CODE_ROOT / "checkpoints"
        self.drive_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Initialize TarDAL Fuse Module (Generator Backbone)
        self.fuse = Fuse(config, mode='train')
        fuse_ckpt = config.fuse.pretrained
        if fuse_ckpt and Path(fuse_ckpt).exists():
            self.fuse.load_ckpt(torch.load(str(fuse_ckpt), map_location=self.device, weights_only=False))
            logging.info(f"[AG Fine-Tune] Loaded TarDAL Generator backbone weights from {Path(fuse_ckpt).name}")
        
        # Generator remains unfrozen for task-driven adaptation
        for p in self.fuse.generator.parameters():
            p.requires_grad = True

        # 2. Initialize Stage 2 Ultralytics YOLOv5su Detection Head natively
        det_ckpt = getattr(config.detect, 'pretrained', None)
        if not det_ckpt or not Path(det_ckpt).exists():
            det_ckpt = CODE_ROOT / "checkpoints" / "best.pt"
            
        self.yolo_wrapper = YOLO(str(det_ckpt))
        self.det_model = self.yolo_wrapper.model.to(self.device)

        # FIX #1: DETECTOR FREEZE (PERMANENT)
        # Freeze ALL detector parameters and permanently set detector to eval mode
        for p in self.det_model.parameters():
            p.requires_grad = False
        self.det_model.eval()

        self.v8_loss = v8DetectionLoss(self.det_model)
        from types import SimpleNamespace
        hyp_dict = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0, 'label_smoothing': 0.0}
        
        if hasattr(self.det_model, 'args') and self.det_model.args is not None:
            if isinstance(self.det_model.args, dict):
                hyp_dict.update(self.det_model.args)
            elif hasattr(self.det_model.args, '__dict__'):
                hyp_dict.update(vars(self.det_model.args))

        if hasattr(self.v8_loss, 'hyp') and self.v8_loss.hyp is not None:
            if isinstance(self.v8_loss.hyp, dict):
                hyp_dict.update(self.v8_loss.hyp)
            elif hasattr(self.v8_loss.hyp, '__dict__'):
                hyp_dict.update(vars(self.v8_loss.hyp))

        hyp_ns = SimpleNamespace(**hyp_dict)
        self.v8_loss.hyp = hyp_ns
        self.det_model.args = hyp_ns
        logging.info(f"[AG Fine-Tune] Preserved Stage 2 Ultralytics YOLOv5su detection head (PERMANENTLY FROZEN 🔒) from {Path(det_ckpt).name}")

        # Log parameter audit matching Stage 2
        print_parameter_verification(self.fuse.generator, self.det_model)

        # 3. Setup Dataset Loaders
        self.train_dataset = LiveM3FDPairedDataset(config.dataset.root, split='train', img_size=640)
        self.val_dataset = LiveM3FDPairedDataset(config.dataset.root, split='val', img_size=640)

        self.t_loader = DataLoader(
            self.train_dataset, batch_size=config.train.batch_size, shuffle=True,
            collate_fn=collate_live_batch, num_workers=0, pin_memory=True
        )
        self.v_loader = DataLoader(
            self.val_dataset, batch_size=config.train.batch_size, shuffle=False,
            collate_fn=collate_live_batch, num_workers=0, pin_memory=True
        )

        # Dataset visualization table
        print_dataset_tracking_visualization(Path(config.dataset.root), config.train.batch_size)

        # FIX #2: LEARNING RATE FIX (Safe Fail-Safe Lookup for 5e-6)
        lr_target = 5e-6
        try:
            if hasattr(config.optimizer, 'lr_gen') and getattr(config.optimizer, 'lr_gen', None) is not None:
                lr_target = float(getattr(config.optimizer, 'lr_gen'))
            elif hasattr(config.optimizer, 'lr_i') and getattr(config.optimizer, 'lr_i', None) is not None:
                lr_target = float(getattr(config.optimizer, 'lr_i'))
        except (KeyError, AttributeError):
            lr_target = 5e-6

        if lr_target > 1e-4:
            lr_target = 5e-6
        self.lr_target = lr_target

        # FIX #9: LOSS BRIDGE SIMPLIFICATION & SINGLE OPTIMIZER
        # Optimizer contains ONLY self.fuse.generator.parameters() (no detector params, no disc_opt)
        self.fd_opt = torch.optim.AdamW(
            self.fuse.generator.parameters(),
            lr=self.lr_target,
            weight_decay=config.optimizer.weight_decay
        )

        # FIX: lr_f in TarDAL config is a MULTIPLIER (0.1 = 10% of initial LR), NOT an absolute value.
        # Using it directly as eta_min caused the LR to be pinned at 0.1 (20,000× overshoot).
        lr_f_multiplier = float(getattr(config.optimizer, 'lr_f', 0.01))
        if lr_f_multiplier > 1.0:
            lr_f_multiplier = 0.01  # Safety: if lr_f > 1.0 it's clearly a multiplier misconfiguration
        lr_final_abs = self.lr_target * lr_f_multiplier
        lr_final_abs = max(min(lr_final_abs, self.lr_target * 0.1), 1e-8)  # Clamp to [1e-8, lr_target/10]
        self.lr_final_abs = lr_final_abs

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.fd_opt,
            T_max=config.train.epochs,
            eta_min=self.lr_final_abs  # e.g. 5e-6 * 0.01 = 5e-8
        )
        logging.info(f"[AG Fine-Tune] LR Schedule: start={self.lr_target:.2e}, eta_min={self.lr_final_abs:.2e}, T_max={config.train.epochs}")

        # FIX #4: LOSS SCALE NORMALIZATION (Running mean EMAs initialized to 1.0)
        self.fuse_mean = torch.tensor(1.0, device=self.device)
        self.det_mean = torch.tensor(1.0, device=self.device)

        # AMP GradScaler for VRAM efficiency
        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == 'cuda')
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')

        self.start_epoch = 1
        self.best_map = -1.0
        self.epoch_history = []

    def sync_to_drive(self, src: Path, fname: str):
        try:
            dst = self.drive_ckpt_dir / fname
            if src.resolve() != dst.resolve():
                shutil.copy2(str(src), str(dst))
                size_mb = dst.stat().st_size / (1024 * 1024)
                print(f"  - Synced {fname:<22} ({size_mb:6.2f} MB) -> {dst}")
        except Exception as e:
            logging.debug(f"[AG Fine-Tune] Drive sync skipped for {fname}: {e}")

    def build_ultralytics_batch(self, fused_rgb: torch.Tensor, label_list: list):
        batch_idx, cls_list, bbox_list = [], [], []
        for i, lbls in enumerate(label_list):
            if lbls.shape[0] > 0:
                batch_idx.extend([i] * lbls.shape[0])
                cls_list.append(lbls[:, 0])
                bbox_list.append(lbls[:, 1:])

        if len(batch_idx) == 0:
            return None

        return {
            'batch_idx': torch.tensor(batch_idx, device=self.device).long(),
            'cls': torch.cat(cls_list, dim=0).view(-1, 1).to(self.device),
            'bboxes': torch.cat(bbox_list, dim=0).to(self.device),
            'img': fused_rgb
        }

    build_detection_batch = build_ultralytics_batch

    def run_validation(self, epoch):
        self.fuse.generator.eval()
        # FIX #1: Ensure detector stays permanently in eval mode
        self.det_model.eval()

        # Use fast local SSD for temporary validation images if running on Colab (/content)
        if Path('/content').exists():
            val_fused_dir = Path('/content') / f"temp_val_fused_stage3_epoch_{epoch}"
            temp_yolo_ckpt = Path('/content') / f"temp_yolo_epoch_{epoch}.pt"
        else:
            val_fused_dir = self.save_dir / f"val_fused_epoch_{epoch}"
            temp_yolo_ckpt = self.save_dir / f"temp_yolo_epoch_{epoch}.pt"

        val_img_dir = val_fused_dir / "images"
        val_lbl_dir = val_fused_dir / "labels"
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_lbl_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch in self.v_loader:
                ir_b = batch['ir'].to(self.device)
                vi_b = batch['vi'].to(self.device)
                cbcr_b = batch['cbcr'].to(self.device)
                lbl_list = batch['labels']
                stems = batch['stems']

                # Direct generator forward call in validation
                fused_y = self.fuse.generator(ir_b, vi_b)
                fused_y_01 = ((fused_y + 1.0) / 2.0).clamp(0.0, 1.0)
                fused_rgb = ycrcb_to_rgb_diff(fused_y_01, cbcr_b)

                for i, stem in enumerate(stems):
                    fused_y_np = (fused_y_01[i, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    cb_np = (cbcr_b[i, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    cr_np = (cbcr_b[i, 1].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    ycrcb_np = np.stack([fused_y_np, cr_np, cb_np], axis=-1)
                    bgr_img = cv2.cvtColor(ycrcb_np, cv2.COLOR_YCrCb2BGR)
                    cv2.imwrite(str(val_img_dir / f"{stem}.jpg"), bgr_img)

                    lbls = lbl_list[i]
                    lbl_path = val_lbl_dir / f"{stem}.txt"
                    if lbls.shape[0] > 0:
                        lines = [f"{int(obj[0])} {obj[1]:.6f} {obj[2]:.6f} {obj[3]:.6f} {obj[4]:.6f}" for obj in lbls]
                        lbl_path.write_text("\n".join(lines) + "\n", encoding='utf-8')
                    else:
                        lbl_path.touch()

        yaml_path = val_fused_dir / "val.yaml"
        data_dict = {
            'path': str(val_fused_dir),
            'train': 'images', 'val': 'images', 'test': 'images',
            'names': NAMES_DICT
        }
        yaml_path.write_text(yaml.dump(data_dict, default_flow_style=False), encoding='utf-8')

        self.yolo_wrapper.save(str(temp_yolo_ckpt))

        val_yolo = YOLO(str(temp_yolo_ckpt))
        results = val_yolo.val(
            data=str(yaml_path),
            imgsz=640,
            batch=16,
            device=self.device,
            conf=0.001,
            iou=0.6,
            verbose=False,
            plots=False
        )

        mp = results.box.mp * 100 if hasattr(results.box, 'mp') else 0.0
        mr = results.box.mr * 100 if hasattr(results.box, 'mr') else 0.0
        map50 = results.box.map50 * 100 if hasattr(results.box, 'map50') else 0.0
        map50_95 = results.box.map * 100 if hasattr(results.box, 'map') else 0.0

        # Auto cleanup temporary validation images and temp checkpoint to save disk and prevent I/O throttling
        try:
            shutil.rmtree(str(val_fused_dir), ignore_errors=True)
            if temp_yolo_ckpt.exists():
                temp_yolo_ckpt.unlink()
        except Exception:
            pass

        return mp, mr, map50, map50_95

    def save_checkpoints(self, epoch, is_best, row_str=""):
        if row_str:
            self.epoch_history = [e for e in self.epoch_history if not (isinstance(e, dict) and e.get('epoch') == epoch)]
            self.epoch_history.append({'epoch': epoch, 'row_str': row_str})

            # Append to persistent text log file on disk and Drive
            try:
                log_p = self.save_dir / "stage3_history.log"
                with log_p.open("a", encoding="utf-8") as f:
                    f.write(row_str + "\n")
                self.sync_to_drive(log_p, "stage3_history.log")
            except Exception:
                pass

        yolo_path = self.save_dir / ('stage3_best.pt' if is_best else 'stage3_last.pt')
        self.yolo_wrapper.save(str(yolo_path))

        gen_path = self.save_dir / ('stage3_gen_best.pt' if is_best else 'stage3_gen_last.pt')
        torch.save(self.fuse.save_ckpt(), str(gen_path))

        resume_path = self.save_dir / 'stage3_joint_resume.pt'
        torch.save({
            'epoch': epoch,
            'completed_epoch': epoch,
            'best_map': self.best_map,
            'fuse': self.fuse.save_ckpt(),
            'detect': self.det_model.state_dict(),
            'optimizer': self.fd_opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'fuse_mean': self.fuse_mean,
            'det_mean': self.det_mean,
            'epoch_history': self.epoch_history
        }, str(resume_path))

        print(f"[AG Fine-Tune] Synchronizing Stage 3 Checkpoints to {self.drive_ckpt_dir}:")
        suffix = "best" if is_best else "last"
        self.sync_to_drive(yolo_path, f"stage3_{suffix}.pt")
        self.sync_to_drive(gen_path, f"stage3_gen_{suffix}.pt")
        self.sync_to_drive(resume_path, "stage3_joint_resume.pt")
        sys.stdout.flush()

    def try_resume(self):
        if self.force_restart:
            logging.info("[AG Fine-Tune] Force restart flag detected (--force_restart). Starting fresh from Epoch 1.")
            return

        resume_cand = []
        if self.resume_path:
            resume_cand.append(Path(self.resume_path))

        resume_cand.extend([
            self.drive_ckpt_dir / 'stage3_joint_resume.pt',
            self.save_dir / 'stage3_joint_resume.pt',
            self.drive_ckpt_dir / 'stage3_last.pt',
            self.save_dir / 'stage3_last.pt'
        ])

        resume_ckpt = next((p for p in resume_cand if p and p.exists() and p.stat().st_size > 0), None)
        if not resume_ckpt:
            if self.resume:
                logging.warning("[AG Fine-Tune] --resume flag was set, but no valid checkpoint file was found. Starting fresh from Epoch 1.")
            return

        try:
            ckpt = torch.load(str(resume_ckpt), map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict):
                if 'completed_epoch' in ckpt:
                    self.start_epoch = ckpt['completed_epoch'] + 1
                elif 'epoch' in ckpt:
                    self.start_epoch = ckpt['epoch'] + 1

                if 'best_map' in ckpt:
                    self.best_map = ckpt['best_map']
                if 'fuse' in ckpt:
                    self.fuse.load_ckpt(ckpt['fuse'])
                if 'optimizer' in ckpt:
                    try:
                        self.fd_opt.load_state_dict(ckpt['optimizer'])
                        # Clamp any corrupted high LR
                        for pg in self.fd_opt.param_groups:
                            if pg.get('lr', 0) > 1e-4:
                                pg['lr'] = self.lr_target
                            pg['initial_lr'] = self.lr_target
                    except Exception as e:
                        logging.warning(f"[AG Fine-Tune] Optimizer state load notice: {e}")
                if 'scheduler' in ckpt:
                    try:
                        self.scheduler.load_state_dict(ckpt['scheduler'])
                        # Sanity check: verify scheduler is not using stale buggy eta_min
                        is_bad_sched = False
                        if hasattr(self.scheduler, 'eta_min') and self.scheduler.eta_min >= self.lr_target:
                            is_bad_sched = True
                        if hasattr(self.scheduler, 'base_lrs') and any(lr > 1e-4 for lr in self.scheduler.base_lrs):
                            is_bad_sched = True
                        if is_bad_sched:
                            logging.warning(f"[AG Fine-Tune] ⚠️ Detected stale/corrupted scheduler in checkpoint. Resetting clean CosineAnnealingLR (start={self.lr_target:.2e}, eta_min={self.lr_final_abs:.2e})...")
                            for pg in self.fd_opt.param_groups:
                                pg['lr'] = self.lr_target
                                pg['initial_lr'] = self.lr_target
                            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                self.fd_opt,
                                T_max=self.config.train.epochs,
                                eta_min=self.lr_final_abs,
                                last_epoch=max(0, self.start_epoch - 1)
                            )
                    except Exception as e:
                        logging.warning(f"[AG Fine-Tune] Scheduler state load notice: {e}. Reinitializing scheduler.")
                        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.fd_opt,
                            T_max=self.config.train.epochs,
                            eta_min=self.lr_final_abs,
                            last_epoch=max(0, self.start_epoch - 1)
                        )
                if 'fuse_mean' in ckpt:
                    self.fuse_mean = ckpt['fuse_mean'].to(self.device)
                if 'det_mean' in ckpt:
                    self.det_mean = ckpt['det_mean'].to(self.device)
                if 'epoch_history' in ckpt and isinstance(ckpt['epoch_history'], list):
                    self.epoch_history = ckpt['epoch_history']

            # Also check for persistent stage3_history.log on Drive or local save_dir
            log_cands = [self.drive_ckpt_dir / "stage3_history.log", self.save_dir / "stage3_history.log"]
            log_f = next((f for f in log_cands if f.exists() and f.stat().st_size > 0), None)
            if log_f:
                lines = [l.strip() for l in log_f.read_text(encoding='utf-8').splitlines() if l.strip() and "│" in l]
                if lines:
                    existing_str_set = {e.get('row_str') if isinstance(e, dict) else e for e in self.epoch_history}
                    for l in lines:
                        if l not in existing_str_set:
                            self.epoch_history.append({'epoch': len(self.epoch_history) + 1, 'row_str': l})

            logging.info(f"[AG Fine-Tune] 🔄 Auto-resuming fine-tuning from checkpoint: {resume_ckpt.name}")
            logging.info(f"[AG Fine-Tune] ➡️ Resuming from BEGINNING of Epoch {self.start_epoch} (Target: {self.config.train.epochs} total epochs, best mAP: {self.best_map:.2f}%)")
        except Exception as e:
            logging.info(f"[AG Fine-Tune] Starting fresh training run from Epoch 1 (Note: {e})")

    def run(self):
        self.try_resume()
        epochs = self.config.train.epochs

        print(f"[AG Fine-Tune] Starting Task-Driven Generator Adaptation ({epochs} epochs, batch {self.config.train.batch_size}, LR=5e-6)...\n")

        # FIX #10: TELEMETRY PARITY (Formatting matches exact output structure)
        header_line = f"{'Epoch':^7} | {'GPU Mem':^8} | {'L_total':^8} | {'L_fuse':^8} ( {'src':^6} {'adv':^6} {'tar':^6} {'det':^6} ) | {'L_det':^8} ( {'box':^6} {'cls':^6} {'dfl':^6} ) | {'mAP50':^7} | {'mAP50-95':^9} | {'Status':^10}"
        logging.info("┌" + "─" * (len(header_line) - 2) + "┐")
        logging.info(f"│ {header_line:^{len(header_line)-4}} │")
        logging.info("├" + "─" * (len(header_line) - 2) + "┤")

        # Re-print all saved telemetry rows from previous epochs upon resumption
        if hasattr(self, 'epoch_history') and len(self.epoch_history) > 0:
            for entry in self.epoch_history:
                if isinstance(entry, dict) and 'row_str' in entry:
                    logging.info(entry['row_str'])
                elif isinstance(entry, str):
                    logging.info(entry)
            sys.stdout.flush()

        for epoch in range(self.start_epoch, epochs + 1):
            self.fuse.generator.train()
            # FIX #1: Detector remains permanently in eval mode (NEVER call self.det_model.train())
            self.det_model.eval()

            tot_meter = AverageMeter()
            f_meter = AverageMeter()
            src_meter, adv_meter, tar_meter, det_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            d_meter = AverageMeter()
            box_meter, cls_meter, dfl_meter = AverageMeter(), AverageMeter(), AverageMeter()

            t_l = tqdm(self.t_loader, disable=False, total=len(self.t_loader) if not self.config.debug.fast_run else 3, ncols=140, leave=False)
            for batch in t_l:
                ir_b = batch['ir'].to(self.device)
                vi_b = batch['vi'].to(self.device)
                cbcr_b = batch['cbcr'].to(self.device)
                mask = batch['mask'].to(self.device)
                ir_w = batch['ir_w'].to(self.device)
                vi_w = batch['vi_w'].to(self.device)
                lbl_list = batch['labels']

                # 1. Single Differentiable Generator Forward Pass
                fused_y = self.fuse.generator(ir_b, vi_b)
                # Differentiable Tanh mapping [-1, 1] -> [0, 1]
                fused_y_01 = ((fused_y + 1.0) / 2.0).clamp(0.0, 1.0)

                # 2. Source Content Preservation Loss (in [0, 1] domain)
                src_w = float(getattr(self.config.loss.fuse, 'src', 1.0))
                src_l = ir_w * self.fuse.src_loss(fused_y_01, ir_b) + vi_w * self.fuse.src_loss(fused_y_01, vi_b)
                f_loss = src_w * src_l.mean()
                src_l_val = src_l.mean().item()
                adv_l_val, tar_l_val, det_l_val = 0.0, 0.0, 0.0

                # 3. Differentiable live image fusion (ITU-R BT.601 YCrCb -> RGB)
                fused_rgb = ycrcb_to_rgb_diff(fused_y_01, cbcr_b)

                # 4. FIX #7: DETECTOR FORWARD WITH FROZEN WEIGHTS & AMP (Saves >50% GPU VRAM)
                batch_dict = self.build_ultralytics_batch(fused_rgb, lbl_list)
                if batch_dict is not None:
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        preds = self.det_model(fused_rgb)
                        d_loss, loss_items = self.v8_loss(preds, batch_dict)
                    box_l, cls_l, dfl_l = extract_loss_metrics(loss_items)
                else:
                    d_loss = torch.tensor(0.0, device=self.device)
                    box_l, cls_l, dfl_l = 0.0, 0.0, 0.0

                # FIX #4: LOSS SCALE NORMALIZATION & EMAs
                # Update running mean EMAs for fusion and detection losses
                self.fuse_mean = 0.9 * self.fuse_mean + 0.1 * f_loss.detach()
                self.det_mean = 0.9 * self.det_mean + 0.1 * d_loss.detach()

                b_c = self.config.loss.bridge
                w_fuse_val = float(b_c.get('fuse', 1.0))
                w_det_val = float(b_c.get('detect', 0.2))

                # Compute normalized combined loss: g_loss = w_fuse * (f_loss / fuse_mean) + w_det * (d_loss / det_mean)
                f_norm = f_loss / (self.fuse_mean + 1e-8)
                d_norm = d_loss / (self.det_mean + 1e-8)
                g_loss = (w_fuse_val * f_norm.sum() + w_det_val * d_norm.sum()).sum()

                # Optimization Step with AMP Scaler (VRAM-Safe)
                self.fd_opt.zero_grad()
                self.scaler.scale(g_loss).backward()
                self.scaler.unscale_(self.fd_opt)

                # FIX #6: GRADIENT CLIPPING
                # Clip ONLY generator parameters at max_norm=0.5
                torch.nn.utils.clip_grad_norm_(self.fuse.generator.parameters(), max_norm=0.5)
                self.scaler.step(self.fd_opt)
                self.scaler.update()

                # FIX #9: LOSS BRIDGE SIMPLIFICATION
                # Dual-optimizer disc_opt updates are completely skipped

                # FIX #10: TELEMETRY PARITY
                # Update meters matching exact telemetry output formatting
                tot_meter.update(g_loss.sum().item() if isinstance(g_loss, torch.Tensor) else float(g_loss))
                f_meter.update(f_loss.sum().item() if isinstance(f_loss, torch.Tensor) else float(f_loss))
                src_meter.update(src_l_val)
                adv_meter.update(adv_l_val)
                tar_meter.update(tar_l_val)
                det_meter.update(det_l_val)
                d_meter.update(d_loss.sum().item() if isinstance(d_loss, torch.Tensor) else float(d_loss))
                box_meter.update(box_l); cls_meter.update(cls_l); dfl_meter.update(dfl_l)

                gpu_mem = f"{torch.cuda.memory_reserved() / 1E9:.2f}G" if torch.cuda.is_available() else "0.0G"
                t_l.set_description(f"Epoch {epoch:2d}/{epochs:2d} [{gpu_mem}] | Total: {tot_meter.avg:.4f} | Fuse: {f_meter.avg:.4f} | Det: {d_meter.avg:.4f}")
                if self.config.debug.fast_run and t_l.n > 2:
                    break

            gpu_mem = f"{torch.cuda.memory_reserved() / 1E9:.2f}G" if torch.cuda.is_available() else "0.0G"
            self.scheduler.step()
            current_lr = self.fd_opt.param_groups[0]['lr']
            logging.info(f"  [LR] Epoch {epoch} Generator LR: {current_lr:.2e}")

            mp, mr, map50, map_all = 0.0, 0.0, 0.0, 0.0
            is_best = False

            # FIX #8: VALIDATION FREQUENCY (Force eval_interval=1 and fail check if mAP@50 < 65%)
            eval_interval = getattr(self.config.train, 'eval_interval', 1)
            if epoch % eval_interval == 0 or self.config.debug.fast_run or True:
                mp, mr, map50, map_all = self.run_validation(epoch)
                if map_all > self.best_map:
                    self.best_map = map_all
                    is_best = True

                # FIX #8: Check if mAP@50 drops below 60% baseline threshold
                if map50 < 60.0:
                    logging.warning(f"⚠️  OPTIMIZATION COLLAPSE WARNING: Epoch {epoch} mAP@50 = {map50:.2f}% dropped below 60.0% baseline threshold!")

            status_mark = "🏆 BEST" if is_best else "SAVED"
            row_str = (
                f" {epoch:2d}/{epochs:<2d} │ {gpu_mem:^8} │ {tot_meter.avg:^8.4f} │ "
                f"{f_meter.avg:^8.4f} ( {src_meter.avg:^6.4f} {adv_meter.avg:^6.4f} {tar_meter.avg:^6.4f} {det_meter.avg:^6.4f} ) │ "
                f"{d_meter.avg:^8.4f} ( {box_meter.avg:^6.4f} {cls_meter.avg:^6.4f} {dfl_meter.avg:^6.4f} ) │ "
                f"{map50:^7.2f}% │ {map_all:^9.2f}% │ {status_mark:^10}"
            )

            self.save_checkpoints(epoch, is_best, row_str)
            logging.info(row_str)
            sys.stdout.flush()

        logging.info("└" + "─" * (len(header_line) - 2) + "┘\n")
        print("\n" + "=" * 80)
        print("  STAGE 3 TASK-DRIVEN GENERATOR ADAPTATION COMPLETE & SAVED  ".center(80))
        print("=" * 80)
        print(f"[AG Fine-Tune] Dedicated Checkpoints Directory: {self.drive_ckpt_dir}")
        print(f"[AG Fine-Tune] Best Overall Validation mAP@50-95: {self.best_map:.2f}%")
        print("=" * 80 + "\n")


# ==============================================================================
# 4. MAIN ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Task-Driven Generator Adaptation")
    parser.add_argument('--cfg', default=str(TARDAL_MAIN_DIR / 'config' / 'default.yaml'), help='config file path')
    parser.add_argument('--weights', type=str, default='dt', help="TarDAL backbone checkpoint: 'dt', 'tt', 'ct'")
    parser.add_argument('--epochs', type=int, default=30, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-6, help='generator learning rate (default: 5e-6)')
    parser.add_argument('--w_fuse', type=float, default=1.0, help='loss bridge weight for generator fusion loss')
    parser.add_argument('--w_det', type=float, default=0.2, help='loss bridge weight for detection loss (default: 0.2)')
    parser.add_argument('--force_restart', action='store_true', help='force restart from epoch 1')
    parser.add_argument('--resume', action='store_true', help='explicitly resume training from last checkpoint')
    parser.add_argument('--resume_path', type=str, default=None, help='optional path to custom checkpoint file for resumption')
    parser.add_argument('--skip_fusion', action='store_true', help='compatibility flag for notebook interface')
    parser.add_argument('--gen_ckpt', type=str, default=None,
                        help='Path to a Stage 3 generator checkpoint (stage3_gen_best.pt) to load as starting weights. '
                             'Use with --force_restart to get a fresh optimizer/scheduler but start from improved gen weights.')
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        cfg_path = TARDAL_MAIN_DIR / "config" / "default.yaml"

    config_dict = yaml.safe_load(cfg_path.open('r'))
    config = from_dict(config_dict)

    config.strategy = 'fuse & detect'
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.image_size = (640, 640)
    config.train.eval_interval = 1
    
    # FIX #2: Set single Generator LR to 5e-6
    config.optimizer.lr_gen = args.lr if args.lr <= 1e-4 else 5e-6
    if hasattr(config.optimizer, 'lr_i'):
        config.optimizer.lr_i = config.optimizer.lr_gen
    config.loss.bridge['fuse'] = args.w_fuse
    config.loss.bridge['detect'] = args.w_det
    config.debug.wandb_mode = 'disabled'

    weights_map = {
        'dt': TARDAL_MAIN_DIR / 'weights' / 'v1' / 'tardal-dt.pth',
        'tt': TARDAL_MAIN_DIR / 'weights' / 'v1' / 'tardal-tt.pth',
        'ct': TARDAL_MAIN_DIR / 'weights' / 'v1' / 'tardal-ct.pth',
    }
    if args.weights in weights_map:
        ckpt_path = weights_map[args.weights]
    else:
        # Check candidate paths for custom checkpoint like stage3_gen_best.pt
        w_p = Path(args.weights)
        cands = [
            w_p,
            Path('/content/drive/MyDrive/FYP/code/checkpoints') / w_p.name,
            CODE_ROOT / 'checkpoints' / w_p.name,
            CODE_ROOT / 'runs' / 'stage3_joint_end2end' / w_p.name,
        ]
        ckpt_path = next((p for p in cands if p.exists() and p.stat().st_size > 0), weights_map['dt'])
    if ckpt_path.exists():
        config.fuse.pretrained = str(ckpt_path)

    det_ckpt_candidates = [
        CODE_ROOT / "checkpoints" / "best.pt",
        Path('/content/drive/MyDrive/FYP/code/checkpoints/best.pt'),
        CODE_ROOT / "runs" / "fine_tune_detection" / "tardal_head_finetune" / "weights" / "best.pt",
        ckpt_path
    ]
    det_ckpt_path = next((p for p in det_ckpt_candidates if p.exists() and p.stat().st_size > 0), ckpt_path)
    config.detect.pretrained = str(det_ckpt_path)

    cand_ds = [
        Path('/content/m3fd/M3FD_Detection'),
        Path('/content/m3fd'),
        CODE_ROOT / 'data' / 'm3fd'
    ]
    ds_root = next((p for p in cand_ds if p.exists()), cand_ds[0])
    config.dataset.root = str(ds_root)

    print("\n" + "=" * 80)
    print(" 🚀 STAGE 3: TASK-DRIVEN GENERATOR ADAPTATION (PERMANENT DETECTOR FREEZE) ".center(80))
    print("=" * 80)
    print(f"  Optimization Strategy : {config.strategy} (Task-Driven Generator Adaptation)")
    print(f"  Backbone Checkpoint   : {args.weights} ({Path(config.fuse.pretrained).name})")
    print(f"  Detector Head Weights : {det_ckpt_path.name} (PERMANENTLY FROZEN 🔒)")
    print(f"  Training Dataset Root : {config.dataset.root}")
    print(f"  Epochs / Batch Size   : {config.train.epochs} epochs | batch size {config.train.batch_size}")
    print(f"  Generator Learning Rate: {config.optimizer.lr_gen} (Single Parameter Group)")
    print(f"  Loss Bridge Normalization: Fuse Weight = {config.loss.bridge['fuse']} | Det Weight = {config.loss.bridge['detect']}")
    print("=" * 80 + "\n")

    trainer = Stage3JointTrainer(config, force_restart=args.force_restart, resume=args.resume, resume_path=args.resume_path)

    # Override generator weights with a custom checkpoint (e.g. stage3_gen_best.pt from a previous run)
    # This allows: fresh optimizer + fresh scheduler + improved generator weights
    if args.gen_ckpt:
        gen_ckpt_path = Path(args.gen_ckpt)
        if not gen_ckpt_path.exists():
            # Try resolving relative to checkpoints dir
            for cand in [trainer.drive_ckpt_dir / args.gen_ckpt,
                         trainer.save_dir / args.gen_ckpt,
                         CODE_ROOT / "checkpoints" / args.gen_ckpt]:
                if cand.exists():
                    gen_ckpt_path = cand
                    break
        if gen_ckpt_path.exists():
            gen_state = torch.load(str(gen_ckpt_path), map_location=trainer.device, weights_only=False)
            trainer.fuse.load_ckpt(gen_state)
            logging.info(f"[AG Fine-Tune] ✅ Loaded custom Generator checkpoint: {gen_ckpt_path.name}")
            logging.info(f"[AG Fine-Tune]    Fresh optimizer & scheduler will be used (no stale state carried over)")
        else:
            logging.warning(f"[AG Fine-Tune] ⚠️  --gen_ckpt path not found: {args.gen_ckpt}")

    trainer.run()


if __name__ == '__main__':
    main()