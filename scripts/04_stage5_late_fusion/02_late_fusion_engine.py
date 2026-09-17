#!/usr/bin/env python3

import sys
from pathlib import Path

# Add scripts directory to sys.path so path_config is always discoverable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import path_config
from path_config import *

"""
================================================================================
Stage 5: RGB + IR Late Fusion Engine & Optimization Dashboard
================================================================================
Implements prediction-level decision fusion:
  - Class-constrained IoU matching (RGB class == IR class, IoU >= threshold)
  - Score Fusion: S_fusion = w_rgb * S_rgb + w_ir * S_ir
  - Bounding Box Fusion: Weighted Box Fusion (WBF)
  - Unmatched Box Preservation (maintains single-modality detections)
  - Validation Split Weight Sweep & Optimization (LF-1 to LF-5, w_rgb in [0.1, 0.9])
  - Test Split Benchmark with Frozen Optimal Weights
================================================================================
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from tabulate import tabulate

CODE_ROOT = Path(__file__).resolve().parent.parent

CLASSES = ['People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck']
NAMES_DICT = {0: 'People', 1: 'Car', 2: 'Bus', 3: 'Lamp', 4: 'Motorcycle', 5: 'Truck'}
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


def box_iou_numpy(box1, box2):
    """Pairwise IoU between box1 (N, 4) and box2 (M, 4) in [x1, y1, x2, y2]."""
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area[:, None] + b2_area[None, :] - inter_area

    return inter_area / np.maximum(union_area, 1e-10)


def match_and_fuse_single_image(rgb_data, ir_data, w_rgb=0.6, w_ir=0.4, iou_thresh=0.5, conf_thresh=0.25, fusion_mode='wbf'):
    """Fuses detections from RGB and IR for a single image."""
    r_boxes = np.array(rgb_data['boxes'], dtype=np.float32).reshape(-1, 4) if rgb_data['boxes'] else np.zeros((0, 4), dtype=np.float32)
    r_scores = np.array(rgb_data['scores'], dtype=np.float32) if rgb_data['scores'] else np.zeros((0,), dtype=np.float32)
    r_classes = np.array(rgb_data['classes'], dtype=np.int32) if rgb_data['classes'] else np.zeros((0,), dtype=int)

    i_boxes = np.array(ir_data['boxes'], dtype=np.float32).reshape(-1, 4) if ir_data['boxes'] else np.zeros((0, 4), dtype=np.float32)
    i_scores = np.array(ir_data['scores'], dtype=np.float32) if ir_data['scores'] else np.zeros((0,), dtype=np.float32)
    i_classes = np.array(ir_data['classes'], dtype=np.int32) if ir_data['classes'] else np.zeros((0,), dtype=int)

    fused_boxes = []
    fused_scores = []
    fused_classes = []

    unique_classes = np.unique(np.concatenate([r_classes, i_classes])) if (len(r_classes) or len(i_classes)) else []

    for cls_id in unique_classes:
        r_idx = np.where(r_classes == cls_id)[0]
        i_idx = np.where(i_classes == cls_id)[0]

        if len(r_idx) == 0 and len(i_idx) == 0:
            continue

        if len(r_idx) == 0:
            for idx in i_idx:
                s = float(i_scores[idx]) * w_ir
                if s >= conf_thresh:
                    fused_boxes.append(i_boxes[idx].tolist())
                    fused_scores.append(round(s, 4))
                    fused_classes.append(int(cls_id))
            continue

        if len(i_idx) == 0:
            for idx in r_idx:
                s = float(r_scores[idx]) * w_rgb
                if s >= conf_thresh:
                    fused_boxes.append(r_boxes[idx].tolist())
                    fused_scores.append(round(s, 4))
                    fused_classes.append(int(cls_id))
            continue

        sub_r_boxes = r_boxes[r_idx]
        sub_i_boxes = i_boxes[i_idx]
        iou_mat = box_iou_numpy(sub_r_boxes, sub_i_boxes)

        matched_r = set()
        matched_i = set()

        while True:
            if iou_mat.size == 0 or np.max(iou_mat) < iou_thresh:
                break
            best_r, best_i = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            if iou_mat[best_r, best_i] < iou_thresh:
                break

            orig_r_idx = r_idx[best_r]
            orig_i_idx = i_idx[best_i]

            sr = float(r_scores[orig_r_idx])
            si = float(i_scores[orig_i_idx])
            br = sub_r_boxes[best_r]
            bi = sub_i_boxes[best_i]

            sf = w_rgb * sr + w_ir * si

            if fusion_mode == 'wbf':
                denom = (w_rgb * sr + w_ir * si)
                bf = (w_rgb * sr * br + w_ir * si * bi) / max(denom, 1e-6)
            else:
                bf = w_rgb * br + w_ir * bi

            if sf >= conf_thresh:
                fused_boxes.append([round(float(v), 2) for v in bf])
                fused_scores.append(round(float(sf), 4))
                fused_classes.append(int(cls_id))

            matched_r.add(best_r)
            matched_i.add(best_i)

            iou_mat[best_r, :] = -1.0
            iou_mat[:, best_i] = -1.0

        for r_pos, orig_r in enumerate(r_idx):
            if r_pos not in matched_r:
                s = float(r_scores[orig_r]) * w_rgb
                if s >= conf_thresh:
                    fused_boxes.append([round(float(v), 2) for v in sub_r_boxes[r_pos]])
                    fused_scores.append(round(s, 4))
                    fused_classes.append(int(cls_id))

        for i_pos, orig_i in enumerate(i_idx):
            if i_pos not in matched_i:
                s = float(i_scores[orig_i]) * w_ir
                if s >= conf_thresh:
                    fused_boxes.append([round(float(v), 2) for v in sub_i_boxes[i_pos]])
                    fused_scores.append(round(s, 4))
                    fused_classes.append(int(cls_id))

    return {
        'boxes': fused_boxes,
        'scores': fused_scores,
        'classes': fused_classes
    }


def compute_ap_single_class(tp_mat, conf_arr, n_gt):
    """Computes AP across 10 IoU thresholds for one class."""
    if n_gt == 0:
        return np.zeros(len(IOU_THRESHOLDS)), 0.0, 0.0
    if len(conf_arr) == 0:
        return np.zeros(len(IOU_THRESHOLDS)), 0.0, 0.0

    sort_idx = np.argsort(-conf_arr)
    tp = tp_mat[sort_idx]

    tp_cum = np.cumsum(tp, axis=0)
    fp_cum = np.cumsum(~tp, axis=0)

    rec = tp_cum / max(n_gt, 1e-10)
    prec = tp_cum / (tp_cum + fp_cum + 1e-10)

    mrec = np.linspace(0.0, 1.0, 101)
    ap = np.zeros(tp.shape[1])

    for j in range(tp.shape[1]):
        r = rec[:, j]
        p = prec[:, j]
        p_interp = np.zeros(101)
        for i, r_val in enumerate(mrec):
            idx = np.where(r >= r_val)[0]
            if len(idx) > 0:
                p_interp[i] = np.max(p[idx])
        ap[j] = np.mean(p_interp)

    final_p = prec[-1, 0] if len(prec) > 0 else 0.0
    final_r = rec[-1, 0] if len(rec) > 0 else 0.0
    return ap, final_p, final_r


def evaluate_dataset_predictions(preds_dict, gt_dict):
    """
    Computes overall & per-class mAP@50 and mAP@50-95 using official 101-point COCO protocol.
    """
    per_class_tp = {c: [] for c in range(len(CLASSES))}
    per_class_conf = {c: [] for c in range(len(CLASSES))}
    per_class_n_gt = {c: 0 for c in range(len(CLASSES))}

    for stem, pdata in preds_dict.items():
        gdata = gt_dict.get(stem, {'boxes': [], 'classes': []})
        gt_boxes = np.array(gdata['boxes'], dtype=np.float32).reshape(-1, 4) if gdata['boxes'] else np.zeros((0, 4), dtype=np.float32)
        gt_classes = np.array(gdata['classes'], dtype=np.int32) if gdata['classes'] else np.zeros((0,), dtype=int)

        pred_boxes = np.array(pdata['boxes'], dtype=np.float32).reshape(-1, 4) if pdata['boxes'] else np.zeros((0, 4), dtype=np.float32)
        pred_scores = np.array(pdata['scores'], dtype=np.float32) if pdata['scores'] else np.zeros((0,), dtype=np.float32)
        pred_classes = np.array(pdata['classes'], dtype=np.int32) if pdata['classes'] else np.zeros((0,), dtype=int)

        for c in range(len(CLASSES)):
            c_gt_idx = np.where(gt_classes == c)[0]
            n_gt_c = len(c_gt_idx)
            per_class_n_gt[c] += n_gt_c

            c_pred_idx = np.where(pred_classes == c)[0]
            if len(c_pred_idx) == 0:
                continue

            c_p_boxes = pred_boxes[c_pred_idx]
            c_p_scores = pred_scores[c_pred_idx]

            if n_gt_c == 0:
                tp_img = np.zeros((len(c_pred_idx), len(IOU_THRESHOLDS)), dtype=bool)
                per_class_tp[c].append(tp_img)
                per_class_conf[c].append(c_p_scores)
                continue

            c_g_boxes = gt_boxes[c_gt_idx]
            ious = box_iou_numpy(c_p_boxes, c_g_boxes) # (N_pred, N_gt)

            tp_img = np.zeros((len(c_pred_idx), len(IOU_THRESHOLDS)), dtype=bool)

            for t_idx, iou_t in enumerate(IOU_THRESHOLDS):
                matched_gt = set()
                # Sort predictions by score descending for matching
                order = np.argsort(-c_p_scores)
                for p_i in order:
                    best_gt = -1
                    best_iou = iou_t
                    for g_i in range(n_gt_c):
                        if g_i not in matched_gt and ious[p_i, g_i] >= best_iou:
                            best_iou = ious[p_i, g_i]
                            best_gt = g_i
                    if best_gt >= 0:
                        tp_img[p_i, t_idx] = True
                        matched_gt.add(best_gt)

            per_class_tp[c].append(tp_img)
            per_class_conf[c].append(c_p_scores)

    per_class_results = {}
    all_ap50 = []
    all_ap95 = []
    all_p = []
    all_r = []

    for c in range(len(CLASSES)):
        c_name = CLASSES[c]
        if len(per_class_tp[c]) > 0:
            tp_all = np.vstack(per_class_tp[c])
            conf_all = np.concatenate(per_class_conf[c])
        else:
            tp_all = np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
            conf_all = np.zeros((0,), dtype=np.float32)

        ap, p, r = compute_ap_single_class(tp_all, conf_all, per_class_n_gt[c])
        ap50 = float(ap[0] * 100.0)
        ap95 = float(np.mean(ap) * 100.0)

        per_class_results[c_name] = {
            'P': round(float(p * 100.0), 2),
            'R': round(float(r * 100.0), 2),
            'mAP50': round(ap50, 2),
            'mAP50-95': round(ap95, 2),
            'n_gt': per_class_n_gt[c]
        }
        all_ap50.append(ap50)
        all_ap95.append(ap95)
        all_p.append(float(p * 100.0))
        all_r.append(float(r * 100.0))

    return {
        'mp': round(float(np.mean(all_p)), 2),
        'mr': round(float(np.mean(all_r)), 2),
        'map50': round(float(np.mean(all_ap50)), 2),
        'map50_95': round(float(np.mean(all_ap95)), 2),
        'per_class': per_class_results
    }


def load_raw_predictions_cache(split="val"):
    """Loads cached raw predictions and ground truth annotations."""
    cands = [
        CODE_ROOT / "runs" / "stage5_late_fusion" / f"raw_predictions_{split}.json",
        Path("/content/drive/MyDrive/FYP/code/checkpoints") / f"stage5_raw_predictions_{split}.json",
        Path("/content/drive/MyDrive/FYP/code/checkpoints") / f"raw_predictions_{split}.json"
    ]
    target_f = next((p for p in cands if p.exists() and p.stat().st_size > 1000), None)
    if not target_f:
        print(f"[Notice] Cache not found for split '{split}'. Calling 16_generate_raw_predictions.py...")
        import subprocess
        gen_script = CODE_ROOT / "scripts_AG" / "16_generate_raw_predictions.py"
        subprocess.run([sys.executable, str(gen_script), "--split", split], check=True)
        target_f = next((p for p in cands if p.exists()), None)
    if not target_f or not target_f.exists():
        raise FileNotFoundError(f"Could not load or generate raw predictions for split '{split}'.")

    with open(target_f, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def optimize_weights_on_validation(val_cache, conf_thresh=0.25, iou_match=0.5, fusion_mode='wbf'):
    """
    Performs grid search on validation split only.
    Tests LF-1 to LF-5 and systematic sweep w_rgb in [0.1 .. 0.9].
    """
    preds = val_cache['predictions']
    gt = val_cache['ground_truth']

    weight_configs = [
        ("LF-1", 0.5, 0.5),
        ("LF-2", 0.6, 0.4),
        ("LF-3", 0.7, 0.3),
        ("LF-4", 0.8, 0.2),
        ("LF-5", 0.4, 0.6),
        ("Sweep-1", 0.1, 0.9),
        ("Sweep-2", 0.2, 0.8),
        ("Sweep-3", 0.3, 0.7),
        ("Sweep-4", 0.9, 0.1),
    ]

    print("\n" + "=" * 90)
    print(" 🧪 STEP 5: LATE FUSION WEIGHT OPTIMIZATION ON VALIDATION SET ".center(90))
    print("=" * 90)
    print(f"  Validation Pairs : {len(preds)}")
    print(f"  IoU Match Thresh : {iou_match}")
    print(f"  Conf Threshold   : {conf_thresh}")
    print(f"  Fusion Mode      : {fusion_mode.upper()}")
    print("=" * 90 + "\n")

    val_table = []
    headers = ["Exp ID", "w_RGB", "w_IR", "Precision", "Recall", "mAP@50", "mAP@50-95", "Delta vs 50/50"]
    best_config = None
    best_m50 = -1.0
    base_5050_m50 = None

    all_results = []

    for exp_id, wr, wi in weight_configs:
        t0 = time.time()
        fused_preds = {}
        for stem, pdata in preds.items():
            fused_preds[stem] = match_and_fuse_single_image(
                pdata['rgb'], pdata['ir'],
                w_rgb=wr, w_ir=wi,
                iou_thresh=iou_match,
                conf_thresh=conf_thresh,
                fusion_mode=fusion_mode
            )
        metrics = evaluate_dataset_predictions(fused_preds, gt)
        dur = time.time() - t0

        if exp_id == "LF-1":
            base_5050_m50 = metrics['map50']

        delta_str = f"{metrics['map50'] - base_5050_m50:+.2f}%" if base_5050_m50 is not None else "0.00%"

        val_table.append([
            exp_id, f"{wr:.2f}", f"{wi:.2f}",
            f"{metrics['mp']:.2f}%", f"{metrics['mr']:.2f}%",
            f"{metrics['map50']:.2f}%", f"{metrics['map50_95']:.2f}%",
            delta_str
        ])

        all_results.append({
            'exp_id': exp_id, 'w_rgb': wr, 'w_ir': wi,
            'metrics': metrics
        })

        if metrics['map50'] > best_m50:
            best_m50 = metrics['map50']
            best_config = (exp_id, wr, wi, metrics)

    # Sort table by mAP@50 descending
    val_table_sorted = sorted(val_table, key=lambda row: float(row[5].replace('%', '')), reverse=True)
    print(tabulate(val_table_sorted, headers=headers, tablefmt="grid"))

    print("\n" + "-" * 90)
    print(f"  🏆 OPTIMAL CONFIGURATION SELECTED: {best_config[0]} (w_RGB = {best_config[1]:.2f}, w_IR = {best_config[2]:.2f})")
    print(f"  🎯 Validation mAP@50: {best_config[3]['map50']:.2f}% | mAP@50-95: {best_config[3]['map50_95']:.2f}%")
    print("-" * 90 + "\n")

    optimal_data = {
        'optimal_exp_id': best_config[0],
        'w_rgb': best_config[1],
        'w_ir': best_config[2],
        'conf_thresh': conf_thresh,
        'iou_match': iou_match,
        'fusion_mode': fusion_mode,
        'val_metrics': best_config[3],
        'all_experiments': all_results
    }

    out_file = CODE_ROOT / "runs" / "stage5_late_fusion" / "optimal_weights.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(optimal_data, f, indent=2)

    drive_out = Path("/content/drive/MyDrive/FYP/code/checkpoints/stage5_optimal_weights.json")
    try:
        with open(drive_out, "w", encoding="utf-8") as f:
            json.dump(optimal_data, f, indent=2)
    except Exception:
        pass

    return best_config[1], best_config[2]


def benchmark_on_test_set(test_cache, w_rgb, w_ir, conf_thresh=0.25, iou_match=0.5, fusion_mode='wbf'):
    """Evaluates frozen optimal weights on official 840-image TEST set."""
    preds = test_cache['predictions']
    gt = test_cache['ground_truth']

    print("\n" + "=" * 90)
    print(" 🔬 STEP 7: BENCHMARKING FROZEN LATE FUSION ON TEST SPLIT (840 pairs) ".center(90))
    print("=" * 90)
    print(f"  Frozen Weights   : w_RGB = {w_rgb:.2f}, w_IR = {w_ir:.2f}")
    print(f"  IoU Match Thresh : {iou_match}")
    print(f"  Conf Threshold   : {conf_thresh}")
    print(f"  Fusion Mode      : {fusion_mode.upper()}")
    print("=" * 90 + "\n")

    t_start = time.perf_counter()
    fused_preds = {}
    for stem, pdata in preds.items():
        fused_preds[stem] = match_and_fuse_single_image(
            pdata['rgb'], pdata['ir'],
            w_rgb=w_rgb, w_ir=w_ir,
            iou_thresh=iou_match,
            conf_thresh=conf_thresh,
            fusion_mode=fusion_mode
        )
    t_end = time.perf_counter()

    fusion_lat_ms = ((t_end - t_start) / max(len(preds), 1)) * 1000.0

    # Timing from raw prediction cache
    rgb_timing = test_cache.get('timing', {}).get('rgb_time_sec', 15.0)
    ir_timing = test_cache.get('timing', {}).get('ir_time_sec', 15.0)
    yolo_rgb_lat = (rgb_timing / max(len(preds), 1)) * 1000.0
    yolo_ir_lat = (ir_timing / max(len(preds), 1)) * 1000.0

    # In parallel or sequential execution
    total_seq_lat = yolo_rgb_lat + yolo_ir_lat + fusion_lat_ms
    total_par_lat = max(yolo_rgb_lat, yolo_ir_lat) + fusion_lat_ms
    fps_seq = 1000.0 / max(total_seq_lat, 1e-5)
    fps_par = 1000.0 / max(total_par_lat, 1e-5)

    # Evaluate accuracy
    metrics = evaluate_dataset_predictions(fused_preds, gt)

    # Also evaluate pure unimodal RGB and IR from same cached test predictions for head-to-head comparison
    rgb_only_preds = {}
    ir_only_preds = {}
    for stem, pdata in preds.items():
        rgb_only_preds[stem] = {
            'boxes': [b for b, s in zip(pdata['rgb']['boxes'], pdata['rgb']['scores']) if s >= conf_thresh],
            'scores': [s for s in pdata['rgb']['scores'] if s >= conf_thresh],
            'classes': [c for c, s in zip(pdata['rgb']['classes'], pdata['rgb']['scores']) if s >= conf_thresh]
        }
        ir_only_preds[stem] = {
            'boxes': [b for b, s in zip(pdata['ir']['boxes'], pdata['ir']['scores']) if s >= conf_thresh],
            'scores': [s for s in pdata['ir']['scores'] if s >= conf_thresh],
            'classes': [c for c, s in zip(pdata['ir']['classes'], pdata['ir']['scores']) if s >= conf_thresh]
        }

    rgb_metrics = evaluate_dataset_predictions(rgb_only_preds, gt)
    ir_metrics = evaluate_dataset_predictions(ir_only_preds, gt)

    # Head-to-Head Comparison Table
    cmp_headers = [
        "System / Approach", "Precision", "Recall", "mAP@50", "mAP@50-95",
        "Pure Inf Latency", "Pipeline Latency (Seq)", "Throughput (FPS)"
    ]
    cmp_table = [
        [
            "Direct RGB (YOLOv5su)",
            f"{rgb_metrics['mp']:.2f}%", f"{rgb_metrics['mr']:.2f}%",
            f"{rgb_metrics['map50']:.2f}%", f"{rgb_metrics['map50_95']:.2f}%",
            f"{yolo_rgb_lat:.2f} ms", f"{yolo_rgb_lat:.2f} ms", f"{1000.0/yolo_rgb_lat:.1f} FPS"
        ],
        [
            "Direct IR (YOLOv5su)",
            f"{ir_metrics['mp']:.2f}%", f"{ir_metrics['mr']:.2f}%",
            f"{ir_metrics['map50']:.2f}%", f"{ir_metrics['map50_95']:.2f}%",
            f"{yolo_ir_lat:.2f} ms", f"{yolo_ir_lat:.2f} ms", f"{1000.0/yolo_ir_lat:.1f} FPS"
        ],
        [
            f"Late Fusion (w_RGB={w_rgb:.1f}, w_IR={w_ir:.1f})",
            f"{metrics['mp']:.2f}%", f"{metrics['mr']:.2f}%",
            f"{metrics['map50']:.2f}%", f"{metrics['map50_95']:.2f}%",
            f"{fusion_lat_ms:.2f} ms (fuse)", f"{total_seq_lat:.2f} ms", f"{fps_seq:.1f} FPS"
        ]
    ]

    print("=" * 105)
    print(" 📊 HEAD-TO-HEAD COMPARISON ON M3FD TEST SPLIT (840 pairs) ".center(105))
    print("=" * 105)
    print(tabulate(cmp_table, headers=cmp_headers, tablefmt="grid"))

    # Per-Class Comparison Table
    pc_headers = ["Class", "Direct RGB mAP@50", "Direct IR mAP@50", "Late Fusion mAP@50", "Delta vs Best Unimodal"]
    pc_table = []
    for c in CLASSES:
        r_m50 = rgb_metrics['per_class'][c]['mAP50']
        i_m50 = ir_metrics['per_class'][c]['mAP50']
        f_m50 = metrics['per_class'][c]['mAP50']
        max_uni = max(r_m50, i_m50)
        delta = f"{f_m50 - max_uni:+.2f}%"
        pc_table.append([c, f"{r_m50:.2f}%", f"{i_m50:.2f}%", f"{f_m50:.2f}%", delta])

    print("\n" + "=" * 90)
    print(" 🎯 PER-CLASS mAP@50 BREAKDOWN ".center(90))
    print("=" * 90)
    print(tabulate(pc_table, headers=pc_headers, tablefmt="grid"))

    # Save benchmark telemetry to JSON
    benchmark_data = {
        'split': 'test',
        'weights': {'w_rgb': w_rgb, 'w_ir': w_ir},
        'conf_thresh': conf_thresh,
        'iou_match': iou_match,
        'late_fusion_metrics': metrics,
        'direct_rgb_metrics': rgb_metrics,
        'direct_ir_metrics': ir_metrics,
        'timing': {
            'yolo_rgb_lat_ms': round(yolo_rgb_lat, 2),
            'yolo_ir_lat_ms': round(yolo_ir_lat, 2),
            'fusion_lat_ms': round(fusion_lat_ms, 2),
            'total_seq_lat_ms': round(total_seq_lat, 2),
            'fps_seq': round(fps_seq, 1),
            'fps_par': round(fps_par, 1)
        }
    }

    out_file = CODE_ROOT / "runs" / "stage5_late_fusion" / "late_fusion_test_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    drive_out = Path("/content/drive/MyDrive/FYP/code/checkpoints/stage5_late_fusion_test_benchmark.json")
    try:
        with open(drive_out, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2)
        print(f"\n  [Artifact] 💾 Benchmark telemetry saved to: {drive_out}")
    except Exception:
        pass

    return benchmark_data


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Late Fusion Engine & Optimization")
    parser.add_argument('--mode', type=str, default='full_pipeline', choices=['optimize_val', 'eval_test', 'full_pipeline'], help="Workflow mode")
    parser.add_argument('--w_rgb', type=float, default=None, help="Manual w_RGB weight (if specified, skips val optimization)")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold for fused detections")
    parser.add_argument('--iou_match', type=float, default=0.50, help="IoU matching threshold between RGB and IR boxes")
    parser.add_argument('--fusion_mode', type=str, default='wbf', choices=['wbf', 'linear'], help="Bounding box fusion mode")
    args = parser.parse_args()

    w_rgb = args.w_rgb
    w_ir = round(1.0 - w_rgb, 4) if w_rgb is not None else None

    if args.mode in ['optimize_val', 'full_pipeline'] and w_rgb is None:
        val_cache = load_raw_predictions_cache(split='val')
        w_rgb, w_ir = optimize_weights_on_validation(val_cache, conf_thresh=args.conf, iou_match=args.iou_match, fusion_mode=args.fusion_mode)

    if args.mode in ['eval_test', 'full_pipeline']:
        if w_rgb is None:
            w_rgb = 0.6
            w_ir = 0.4
        test_cache = load_raw_predictions_cache(split='test')
        benchmark_on_test_set(test_cache, w_rgb=w_rgb, w_ir=w_ir, conf_thresh=args.conf, iou_match=args.iou_match, fusion_mode=args.fusion_mode)


if __name__ == '__main__':
    main()