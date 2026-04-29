"""iter-G: tests for the off-graph YOLO decode + NMS reference.

Mirrors the existing test_eval_harness style — no heavy ML deps, all logic
is pure numpy. Covers the three load-bearing assertions from iter-G's AC:

  (a) decode_yolo_raw matches onnx_adapter._decode_yolo_onnx bit-for-bit
      on a fixed raw tensor (the firmware C port's parity reference).
  (b) nms_single_class drops overlapping duplicates.
  (c) verify_off_graph flags in-graph NMS ops when given a fixture
      op_breakdown that includes one.
"""
from __future__ import annotations

import numpy as np

from training.edge.eval.adapters.onnx_adapter import _decode_yolo_onnx
from training.edge.eval.decode_only import (
    IN_GRAPH_NMS_OPS,
    decode_and_nms,
    decode_yolo_raw,
    nms_single_class,
    verify_off_graph,
)


def _fixed_raw_tensor() -> np.ndarray:
    """Hand-crafted (1, 5, 8) raw tensor — channels: cx, cy, w, h, cls0.

    Three anchors are above the default 0.25 confidence threshold, the
    rest sit below. Two of the three above-threshold boxes overlap
    heavily so NMS will collapse them; the third sits in a distinct
    region. Channels-first shape (5, 8) matches the
    ``_decode_yolo_onnx`` "channels first when channels < anchors"
    branch.
    """
    cx = [56.0, 60.0, 168.0, 100.0, 0.0, 0.0, 0.0, 0.0]
    cy = [56.0, 58.0, 168.0, 100.0, 0.0, 0.0, 0.0, 0.0]
    w = [40.0, 42.0, 60.0, 30.0, 1.0, 1.0, 1.0, 1.0]
    h = [40.0, 42.0, 60.0, 30.0, 1.0, 1.0, 1.0, 1.0]
    cls = [0.95, 0.92, 0.80, 0.10, 0.05, 0.05, 0.05, 0.05]
    raw = np.array([cx, cy, w, h, cls], dtype=np.float32)[None, ...]
    assert raw.shape == (1, 5, 8)
    return raw


def test_decode_only_matches_onnx_adapter_on_fixed_tensor() -> None:
    """Parity with onnx_adapter._decode_yolo_onnx — the C port's contract."""
    raw = _fixed_raw_tensor()
    imgsz = 224
    conf_thr = 0.25

    ours = decode_yolo_raw(raw, imgsz=imgsz, conf_threshold=conf_thr)
    theirs = _decode_yolo_onnx(raw, conf_thr=conf_thr, imgsz=imgsz)

    assert len(ours) == len(theirs) == 3
    for a, b in zip(ours, theirs):
        assert a["confidence"] == b["confidence"]
        assert a["bbox"] == b["bbox"], (a, b)


def test_nms_deduplicates_overlapping_boxes() -> None:
    """Greedy NMS keeps the highest-confidence box and drops near-duplicates."""
    raw = _fixed_raw_tensor()
    decoded = decode_yolo_raw(raw, imgsz=224, conf_threshold=0.25)

    # Pre-NMS we have 3 boxes (cx=56 conf=0.95, cx=60 conf=0.92, cx=168 conf=0.80).
    # The first two overlap heavily (centers 4 px apart, both ~40 px wide),
    # the third sits in a different region.
    assert len(decoded) == 3

    kept = nms_single_class(decoded, iou_threshold=0.45)
    assert len(kept) == 2

    # The 0.95 box wins the overlap; the 0.80 isolated box also survives.
    confs = sorted(b["confidence"] for b in kept)
    assert confs == [pytest_approx(0.80), pytest_approx(0.95)]


def test_decode_and_nms_round_trip_via_full_pipeline() -> None:
    """decode_and_nms() is decode_yolo_raw + nms_single_class glued together."""
    raw = _fixed_raw_tensor()
    direct = decode_and_nms(raw, imgsz=224, conf_threshold=0.25, iou_threshold=0.45)
    indirect = nms_single_class(
        decode_yolo_raw(raw, imgsz=224, conf_threshold=0.25),
        iou_threshold=0.45,
    )
    assert direct == indirect


def test_decode_handles_unrecognized_shape_gracefully() -> None:
    """A wrong-shape tensor returns [], not a crash — important for the C port."""
    bad = np.zeros((4,), dtype=np.float32)
    assert decode_yolo_raw(bad, imgsz=224, conf_threshold=0.25) == []
    assert decode_and_nms(bad, imgsz=224) == []


def test_verify_off_graph_flags_in_graph_nms_op() -> None:
    """The verifier MUST surface an in-graph NMS op as a blocker."""
    op_breakdown = [
        {"op_name": "CONV_2D", "count": 64, "total_us": 1_000_000},
        {"op_name": "TFLite_Detection_PostProcess", "count": 1, "total_us": 12_345},
        {"op_name": "RESHAPE", "count": 8, "total_us": 50},
    ]
    result = verify_off_graph(op_breakdown)
    assert result.off_graph is False
    assert "TFLite_Detection_PostProcess" in result.offending_ops
    assert result.status == "blocked"


def test_verify_off_graph_passes_for_iter_a_op_set() -> None:
    """A real iter-A op_breakdown contains no NMS — the verifier should pass."""
    op_breakdown = [
        {"op_name": "ADD", "count": 400, "total_us": 38943},
        {"op_name": "CONV_2D", "count": 3200, "total_us": 54012488},
        {"op_name": "LOGISTIC", "count": 2900, "total_us": 3853839},
        {"op_name": "DEQUANTIZE", "count": 50, "total_us": 126},
        {"op_name": "RESHAPE", "count": 250, "total_us": 264},
        {"op_name": "TRANSPOSE", "count": 200, "total_us": 2953},
    ]
    result = verify_off_graph(op_breakdown)
    assert result.off_graph is True
    assert result.offending_ops == ()
    assert result.status == "off_graph_confirmed"


def test_verify_off_graph_lists_multiple_offenders_deduplicated() -> None:
    """Duplicate offending op_names appear once, sorted, in offending_ops."""
    op_breakdown = [
        {"op_name": "NON_MAX_SUPPRESSION_V5", "count": 1, "total_us": 100},
        {"op_name": "TFLite_Detection_PostProcess", "count": 1, "total_us": 200},
        {"op_name": "NON_MAX_SUPPRESSION_V5", "count": 1, "total_us": 50},
    ]
    result = verify_off_graph(op_breakdown)
    assert result.off_graph is False
    assert result.offending_ops == (
        "NON_MAX_SUPPRESSION_V5",
        "TFLite_Detection_PostProcess",
    )


def test_in_graph_nms_ops_set_covers_known_exporter_outputs() -> None:
    """Catch regressions if someone narrows the recognized NMS-op set."""
    for required in (
        "TFLite_Detection_PostProcess",
        "NON_MAX_SUPPRESSION_V4",
        "NON_MAX_SUPPRESSION_V5",
        "CombinedNonMaxSuppression",
    ):
        assert required in IN_GRAPH_NMS_OPS


# Tiny local helper so the test module doesn't depend on pytest.approx
# transitively (mirrors the metrics-test convention).
def pytest_approx(value: float, tol: float = 1e-6):
    class _Approx:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, (int, float)) and abs(other - value) < tol

        def __repr__(self) -> str:
            return f"approx({value})"

    return _Approx()
