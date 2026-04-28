"""Single-class mAP@0.5 implementation.

Algorithm follows the pycocotools / PASCAL VOC 2010 convention:
greedy IoU matching of predictions to ground-truth at IoU >= 0.5,
then 11-point precision-recall integration. Source reference:
- pycocotools.cocoeval.COCOeval (https://github.com/cocodataset/cocoapi)
- VOC 2010 mAP integration (https://github.com/Cartucho/mAP)

Single-class only: every prediction and GT box is treated as the same
class (cat, class_id=0). The harness drops class id altogether.

Bounding boxes are normalized [x1, y1, x2, y2] with origin top-left.
"""
from __future__ import annotations

from typing import Sequence


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU on two normalized [x1,y1,x2,y2] boxes. Returns 0.0 if degenerate."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def compute_map50(
    predictions_per_image: list[list[dict]],
    gt_per_image: list[list[list[float]]],
    iou_threshold: float = 0.5,
) -> float:
    """Single-class mAP@0.5.

    Args:
        predictions_per_image: list[list[{"bbox": [x1,y1,x2,y2], "confidence": float}]]
        gt_per_image: list[list[[x1,y1,x2,y2]]] aligned with predictions_per_image
        iou_threshold: IoU above which a prediction matches a GT box.

    Returns:
        AP@iou_threshold averaged with the VOC 2010 11-point method.
    """
    assert len(predictions_per_image) == len(gt_per_image), \
        "predictions and gt must align by image"

    flat_preds: list[tuple[int, float, list[float]]] = []
    for img_idx, preds in enumerate(predictions_per_image):
        for p in preds:
            flat_preds.append((img_idx, float(p["confidence"]), list(p["bbox"])))

    flat_preds.sort(key=lambda x: x[1], reverse=True)

    total_gt = sum(len(g) for g in gt_per_image)
    if total_gt == 0:
        return 0.0

    matched = [[False] * len(g) for g in gt_per_image]
    tp = [0] * len(flat_preds)
    fp = [0] * len(flat_preds)

    for i, (img_idx, _conf, box) in enumerate(flat_preds):
        gts = gt_per_image[img_idx]
        best_iou = 0.0
        best_j = -1
        for j, gt_box in enumerate(gts):
            if matched[img_idx][j]:
                continue
            iou = iou_xyxy(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            tp[i] = 1
            matched[img_idx][best_j] = True
        else:
            fp[i] = 1

    cum_tp = 0
    cum_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for i in range(len(flat_preds)):
        cum_tp += tp[i]
        cum_fp += fp[i]
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / total_gt)

    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        p_at_t = 0.0
        for r, p in zip(recalls, precisions):
            if r >= t and p > p_at_t:
                p_at_t = p
        ap += p_at_t / 11.0
    return ap
