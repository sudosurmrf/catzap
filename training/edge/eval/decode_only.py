"""iter-G: standalone YOLO output decode + NMS, off-graph reference for the
firmware C port.

Why this module exists
----------------------
The .tflite candidates produced by US-004 / US-006 / iter-A / iter-B / iter-C
/ iter-D / iter-E ship with NO in-graph NMS — onnx2tf and ultralytics' ONNX
exporter strip post-processing. So the eval harness's
``training/edge/eval/adapters/onnx_adapter.py:_decode_yolo_onnx`` and
``training/edge/eval/adapters/tflite_adapter.py:TfliteAdapter.predict`` both
implement the missing decode in Python.

For the ESP32-P4 firmware the same decode + NMS has to live OUTSIDE the
.tflite (that's why this iteration's edge-bench run reports inference-only
latency — what TFLM actually executes on chip). This module is the canonical
reference: a single ``decode_and_nms()`` that the firmware C port can mirror
op-for-op, plus a verifier that flags any candidate .tflite whose op_breakdown
contains an in-graph NMS / Detect post-process op (which would be a regression
against the assumption that decode is host-side).

Math reference
--------------
The bbox decode mirrors ``onnx_adapter._decode_yolo_onnx`` exactly:

    xywh_norm = xywh_pixels / imgsz
    confidence = max(class_scores) over the cls channels
    keep = confidence >= conf_threshold
    bbox = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
    bbox = clip(bbox, 0, 1)

NMS is the new addition vs ``_decode_yolo_onnx`` (which does NOT NMS).
We greedy-pick the highest-confidence box, drop any remaining box whose
IoU >= ``nms_iou_threshold``, repeat. Class-agnostic single-class NMS
matches the eval harness's "cat is class 0, single output channel" shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# Known in-graph NMS / Detect post-process op names emitted by various TFLite
# exporters. If any of these appear in a candidate's op_breakdown, decode is
# happening inside the .tflite — which means the firmware would be paying
# decode/NMS as part of "inference" rather than as a separate post-process
# pass. iter-G's contract requires the candidates to be off-graph.
IN_GRAPH_NMS_OPS: frozenset[str] = frozenset(
    {
        "TFLite_Detection_PostProcess",
        "NON_MAX_SUPPRESSION_V4",
        "NON_MAX_SUPPRESSION_V5",
        "CombinedNonMaxSuppression",
    }
)


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of scanning an op_breakdown for in-graph NMS."""

    off_graph: bool
    offending_ops: tuple[str, ...]

    @property
    def status(self) -> str:
        return "off_graph_confirmed" if self.off_graph else "blocked"


def _normalize_raw_to_anchors_first(out: np.ndarray) -> np.ndarray | None:
    """Coerce the raw YOLO output to shape (channels, num_anchors).

    YOLOv8 ONNX/TFLite exports come out as one of:
      (1, 4 + nc, num_anchors)  — channels-first  (most common)
      (1, num_anchors, 4 + nc)  — channels-last   (some exporter variants)
      (4 + nc, num_anchors)     — already 2D, batch stripped
      (num_anchors, 4 + nc)     — 2D, anchors first

    Returns shape (4 + nc, num_anchors), or None if the input shape is
    unrecognized. Mirrors ``onnx_adapter._decode_yolo_onnx``'s leading
    branches so a future C port has one canonical normalization step.
    """
    arr = np.asarray(out)
    if arr.ndim == 3 and arr.shape[1] >= 5 and arr.shape[1] < arr.shape[2]:
        return arr[0]
    if arr.ndim == 3:
        return arr[0].T
    if arr.ndim == 2:
        if arr.shape[0] >= 5 and arr.shape[0] < arr.shape[1]:
            return arr
        return arr.T
    return None


def decode_yolo_raw(
    out: np.ndarray, imgsz: int, conf_threshold: float
) -> list[dict]:
    """Decode raw YOLOv8 output to a list of {bbox, confidence}.

    No NMS — this is the literal port of
    ``onnx_adapter._decode_yolo_onnx`` so parity tests can compare
    bit-for-bit. Use ``decode_and_nms`` for the full firmware-equivalent
    pipeline.
    """
    arr = _normalize_raw_to_anchors_first(out)
    if arr is None or arr.shape[0] < 5:
        return []

    xywh = arr[:4, :].astype(np.float32) / float(imgsz)
    cls_scores = arr[4:, :]
    conf = cls_scores.max(axis=0)

    keep = conf >= conf_threshold
    xywh = xywh[:, keep]
    conf = conf[keep]

    out_list: list[dict] = []
    for i in range(xywh.shape[1]):
        cx, cy, w, h = xywh[:, i]
        x1 = max(0.0, min(1.0, float(cx - w / 2.0)))
        y1 = max(0.0, min(1.0, float(cy - h / 2.0)))
        x2 = max(0.0, min(1.0, float(cx + w / 2.0)))
        y2 = max(0.0, min(1.0, float(cy + h / 2.0)))
        out_list.append({"bbox": [x1, y1, x2, y2], "confidence": float(conf[i])})
    return out_list


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Standard xyxy IoU. Returns 0.0 for degenerate boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def nms_single_class(
    boxes: list[dict], iou_threshold: float = 0.45
) -> list[dict]:
    """Greedy IoU-based NMS for a single class.

    Sort by confidence descending, keep the top box, drop every remaining
    box whose IoU with the kept box exceeds ``iou_threshold``, repeat.
    Returns a new list (the input is not mutated). Class-agnostic — the
    cat-only models we benchmark have a single confidence channel so we
    don't need per-class buckets.
    """
    if not boxes:
        return []
    candidates = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    kept: list[dict] = []
    while candidates:
        head = candidates.pop(0)
        kept.append(head)
        candidates = [
            b
            for b in candidates
            if _iou_xyxy(head["bbox"], b["bbox"]) < iou_threshold
        ]
    return kept


def decode_and_nms(
    out: np.ndarray,
    imgsz: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict]:
    """Full off-graph post-process: raw tensor → boxes after NMS.

    Reference implementation for the firmware C port. Op-for-op:
      1. Reshape raw tensor to (4 + nc, num_anchors).
      2. Divide xywh by imgsz to normalize.
      3. Confidence = max class score across cls channels.
      4. Filter by ``conf_threshold``.
      5. Convert xywh → xyxy and clip to [0, 1].
      6. Greedy NMS at ``iou_threshold``.
    """
    decoded = decode_yolo_raw(out, imgsz=imgsz, conf_threshold=conf_threshold)
    return nms_single_class(decoded, iou_threshold=iou_threshold)


def verify_off_graph(op_breakdown: Iterable[dict]) -> VerifyResult:
    """Scan an edge-bench op_breakdown for in-graph NMS / Detect post-process.

    ``op_breakdown`` is the list of ``{op_name, count, total_us, ...}`` rows
    produced by ``firmware/edge-bench/run_bench.py``. Returns ``off_graph=True``
    if no NMS-class op appears, else lists offenders.
    """
    offenders: list[str] = []
    for row in op_breakdown:
        name = str(row.get("op_name", ""))
        if name in IN_GRAPH_NMS_OPS:
            offenders.append(name)
    if offenders:
        return VerifyResult(off_graph=False, offending_ops=tuple(sorted(set(offenders))))
    return VerifyResult(off_graph=True, offending_ops=())


__all__ = [
    "IN_GRAPH_NMS_OPS",
    "VerifyResult",
    "decode_and_nms",
    "decode_yolo_raw",
    "nms_single_class",
    "verify_off_graph",
]
