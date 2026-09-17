"""
Class-Constrained Weighted Boxes Fusion (WBF) Engine.
Implements the decision-level late fusion algorithm for Stage 5.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


def box_iou_numpy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise Intersection-over-Union between two box arrays (N, 4) and (M, 4)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


def apply_weighted_boxes_fusion(
    rgb_data: Dict[str, Any],
    ir_data: Dict[str, Any],
    w_rgb: float = 0.60,
    w_ir: float = 0.40,
    iou_thresh: float = 0.50,
    conf_thresh: float = 0.25
) -> Dict[str, Any]:
    """
    Executes class-constrained Weighted Boxes Fusion with single-sensor penalty scaling.
    
    Mathematical Formulation:
    - Matched pairs (IoU >= iou_thresh, identical class):
        Box_fused = (w_rgb * s_r * b_r + w_ir * s_i * b_i) / (w_rgb * s_r + w_ir * s_i)
        Score_fused = w_rgb * s_r + w_ir * s_i
    - Unmatched proposals:
        Score_unmatched = s_sensor * w_sensor
    """
    r_boxes = np.array(rgb_data['boxes'], dtype=np.float32).reshape(-1, 4) if rgb_data.get('boxes') else np.zeros((0, 4), dtype=np.float32)
    r_scores = np.array(rgb_data['scores'], dtype=np.float32) if rgb_data.get('scores') else np.zeros((0,), dtype=np.float32)
    r_classes = np.array(rgb_data['classes'], dtype=np.int32) if rgb_data.get('classes') else np.zeros((0,), dtype=int)

    i_boxes = np.array(ir_data['boxes'], dtype=np.float32).reshape(-1, 4) if ir_data.get('boxes') else np.zeros((0, 4), dtype=np.float32)
    i_scores = np.array(ir_data['scores'], dtype=np.float32) if ir_data.get('scores') else np.zeros((0,), dtype=np.float32)
    i_classes = np.array(ir_data['classes'], dtype=np.int32) if ir_data.get('classes') else np.zeros((0,), dtype=int)

    fused_boxes = []
    fused_scores = []
    fused_classes = []
    matched_pairs_count = 0

    all_classes = np.unique(np.concatenate([r_classes, i_classes])) if (len(r_classes) or len(i_classes)) else []

    for cls_id in all_classes:
        r_idx = np.where(r_classes == cls_id)[0]
        i_idx = np.where(i_classes == cls_id)[0]

        if len(r_idx) == 0 and len(i_idx) == 0:
            continue

        # Single-sensor IR only for this class
        if len(r_idx) == 0:
            for idx in i_idx:
                s = float(i_scores[idx]) * w_ir
                if s >= conf_thresh:
                    fused_boxes.append([round(float(v), 2) for v in i_boxes[idx]])
                    fused_scores.append(round(s, 4))
                    fused_classes.append(int(cls_id))
            continue

        # Single-sensor RGB only for this class
        if len(i_idx) == 0:
            for idx in r_idx:
                s = float(r_scores[idx]) * w_rgb
                if s >= conf_thresh:
                    fused_boxes.append([round(float(v), 2) for v in r_boxes[idx]])
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
            denom = (w_rgb * sr + w_ir * si)
            bf = (w_rgb * sr * br + w_ir * si * bi) / max(denom, 1e-6)

            if sf >= conf_thresh:
                fused_boxes.append([round(float(v), 2) for v in bf])
                fused_scores.append(round(float(sf), 4))
                fused_classes.append(int(cls_id))
                matched_pairs_count += 1

            matched_r.add(best_r)
            matched_i.add(best_i)

            iou_mat[best_r, :] = -1.0
            iou_mat[:, best_i] = -1.0

        # Preserve unmatched RGB boxes with sensor weight penalty
        for r_pos, orig_r in enumerate(r_idx):
            if r_pos not in matched_r:
                s = float(r_scores[orig_r]) * w_rgb
                if s >= conf_thresh:
                    fused_boxes.append([round(float(v), 2) for v in r_boxes[orig_r]])
                    fused_scores.append(round(s, 4))
                    fused_classes.append(int(cls_id))

        # Preserve unmatched IR boxes with sensor weight penalty
        for i_pos, orig_i in enumerate(i_idx):
            if i_pos not in matched_i:
                s = float(i_scores[orig_i]) * w_ir
                if s >= conf_thresh:
                    fused_boxes.append([round(float(v), 2) for v in i_boxes[orig_i]])
                    fused_scores.append(round(s, 4))
                    fused_classes.append(int(cls_id))

    return {
        "boxes": fused_boxes,
        "scores": fused_scores,
        "classes": fused_classes,
        "num_rgb": len(r_boxes),
        "num_ir": len(i_boxes),
        "num_fused": len(fused_boxes),
        "num_matched": matched_pairs_count
    }
