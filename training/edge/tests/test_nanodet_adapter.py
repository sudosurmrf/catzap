"""Tests for the NanoDet eval-harness adapter.

Mirrors server/tests/test_detector.py style: pytest + unittest.mock.@patch,
no real model weights. Covers:

* The class-remap helper drops non-cat detections AND remaps id 15 -> 0.
* The ONNX-output decoder produces the right detection list for a synthetic
  output blob with a dominant cat-channel score.
* :class:`NanodetAdapter` (onnx backend) dispatches the decoder to a mocked
  ONNX session and returns the eval-harness contract shape.
* run_eval._load_adapter dispatches NanoDet paths through nanodet_adapter
  regardless of --format.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from training.edge.eval.adapters.nanodet_adapter import (
    COCO_CAT_CLASS,
    DEFAULT_NUM_CLASSES,
    DEFAULT_REG_MAX,
    DEFAULT_STRIDES,
    NanodetAdapter,
    _generate_center_priors,
    decode_nanodet_output,
    remap_nanodet_detections_to_cat_only,
)


def test_remap_drops_non_cat_classes_and_remaps_to_id_zero() -> None:
    """The auto-labeler-style class remap: keep only class 15, re-emit as 0."""
    raw = [
        # [class_id, x1, y1, x2, y2, conf]
        [0, 10, 20, 30, 40, 0.92],   # person -> drop
        [15, 50, 60, 100, 120, 0.81],  # cat -> keep, normalized by imgsz=200
        [16, 100, 100, 200, 200, 0.77],  # dog -> drop
        [15, 0, 0, 200, 200, 0.10],  # cat below threshold -> drop
    ]

    out = remap_nanodet_detections_to_cat_only(
        raw, imgsz=200, confidence_threshold=0.25
    )

    assert len(out) == 1
    only = out[0]
    assert only["confidence"] == pytest.approx(0.81)
    # id 0 is implicit in the contract: every entry IS a cat detection.
    assert only["bbox"] == pytest.approx([0.25, 0.30, 0.50, 0.60])


def test_remap_skips_short_rows_and_low_confidence() -> None:
    raw = [
        [15, 0, 0, 10, 10],  # too short, no conf
        [15, 0, 0, 10, 10, 0.05],  # below threshold
    ]
    out = remap_nanodet_detections_to_cat_only(raw, imgsz=100, confidence_threshold=0.25)
    assert out == []


def test_center_priors_match_upstream_layout() -> None:
    """priors are emitted per-stride in concat order; per-row stride col == stride."""
    priors = _generate_center_priors(imgsz=64, strides=(8, 16))
    # 8x8 + 4x4 = 80 priors
    assert priors.shape == (80, 4)
    # First 64 rows have stride 8.
    assert (priors[:64, 2] == 8).all()
    assert (priors[64:, 2] == 16).all()
    # First two priors at stride 8: (0, 0), (8, 0)
    assert priors[0].tolist() == [0.0, 0.0, 8.0, 8.0]
    assert priors[1].tolist() == [8.0, 0.0, 8.0, 8.0]


def _make_synthetic_nanodet_output(
    imgsz: int,
    cat_prior_idx: int,
    cat_score: float,
    num_classes: int = DEFAULT_NUM_CLASSES,
    reg_max: int = DEFAULT_REG_MAX,
    strides=DEFAULT_STRIDES,
    distance_bin: int = 2,
    other_class_idx: int = 0,
    other_score: float = 0.95,
) -> np.ndarray:
    """Forge a NanoDet-Plus ``_forward_onnx`` output with one strong cat prior.

    All other priors get a strong score in a non-cat channel, so the cat
    filter must drop them. The DFL distribution is one-hot at ``distance_bin``
    so the projected per-side distance is exactly that integer.
    """
    priors = _generate_center_priors(imgsz, strides)
    n = priors.shape[0]
    bins = reg_max + 1
    out = np.zeros((1, n, num_classes + 4 * bins), dtype=np.float32)
    # Default cls scores: one hot at `other_class_idx` so cat is silent everywhere.
    out[0, :, other_class_idx] = other_score
    # Make the chosen prior a strong cat detection.
    out[0, cat_prior_idx, other_class_idx] = 0.0
    out[0, cat_prior_idx, COCO_CAT_CLASS] = cat_score
    # DFL distribution: one-hot at `distance_bin` for every side.
    one_hot = np.zeros(bins, dtype=np.float32)
    one_hot[distance_bin] = 50.0  # large, so softmax collapses to ~1
    for side in range(4):
        out[0, :, num_classes + side * bins:num_classes + (side + 1) * bins] = one_hot
    return out


def test_decode_keeps_only_cat_channel_and_normalizes_box() -> None:
    """Decoder thresholds on cat-channel score and ignores other classes."""
    imgsz = 64
    priors = _generate_center_priors(imgsz, DEFAULT_STRIDES)
    # Pick a prior at stride 8 so we can predict the decoded box.
    target_prior = 9  # row 1 col 1 in the 8x8 grid -> (8, 8)
    cx, cy, stride, _ = priors[target_prior]
    distance_bin = 2  # one-hot at 2 -> projected distance 2 -> pixels = 2 * stride

    raw = _make_synthetic_nanodet_output(
        imgsz=imgsz,
        cat_prior_idx=target_prior,
        cat_score=0.88,
        distance_bin=distance_bin,
    )

    dets = decode_nanodet_output(
        raw, imgsz=imgsz, confidence_threshold=0.5,
    )

    assert len(dets) == 1
    d = dets[0]
    assert d["confidence"] == pytest.approx(0.88, abs=1e-5)
    # box = distance2bbox(center, [stride*bin]*4) / imgsz
    pixel_dist = distance_bin * stride
    expected = [
        max(0.0, (cx - pixel_dist) / imgsz),
        max(0.0, (cy - pixel_dist) / imgsz),
        min(1.0, (cx + pixel_dist) / imgsz),
        min(1.0, (cy + pixel_dist) / imgsz),
    ]
    assert d["bbox"] == pytest.approx(expected, abs=1e-5)


def test_decode_drops_low_confidence_cat_priors() -> None:
    raw = _make_synthetic_nanodet_output(
        imgsz=64, cat_prior_idx=0, cat_score=0.10
    )
    assert decode_nanodet_output(raw, imgsz=64, confidence_threshold=0.25) == []


def test_decode_returns_empty_on_other_class_dominated_output() -> None:
    """When no priors have cat-channel score above threshold, output is empty.

    This is the explicit "other-classes-only" case the PRD calls out: the
    adapter must DROP non-cat-class detections, not silently return them.
    """
    raw = _make_synthetic_nanodet_output(
        imgsz=64, cat_prior_idx=0, cat_score=0.0, other_score=0.99,
    )
    assert decode_nanodet_output(raw, imgsz=64, confidence_threshold=0.25) == []


def test_decode_handles_channels_first_export_via_transpose() -> None:
    """Some exports may emit (channels, num_priors); decoder transposes."""
    imgsz = 64
    raw = _make_synthetic_nanodet_output(imgsz=imgsz, cat_prior_idx=0, cat_score=0.9)
    swapped = raw.transpose(0, 2, 1)
    dets = decode_nanodet_output(swapped, imgsz=imgsz, confidence_threshold=0.25)
    assert len(dets) == 1


def test_nanodet_adapter_onnx_backend_uses_session_factory() -> None:
    """End-to-end via mocked ONNXRuntime session factory."""
    imgsz = 64
    raw = _make_synthetic_nanodet_output(imgsz=imgsz, cat_prior_idx=5, cat_score=0.93)

    fake_session = MagicMock()
    fake_session.get_inputs.return_value = [MagicMock(name="data")]
    fake_session.get_inputs.return_value[0].name = "data"
    fake_session.run.return_value = [raw]

    adapter = NanodetAdapter(
        model_path="/fake/path/nanodet_plus_m.onnx",
        imgsz=imgsz,
        confidence_threshold=0.5,
        backend="onnx",
        session_factory=lambda _path: fake_session,
    )

    frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    out = adapter.predict(frame)
    assert len(out) == 1
    assert 0.0 <= out[0]["confidence"] <= 1.0
    assert out[0]["confidence"] == pytest.approx(0.93, abs=1e-5)
    # Session was actually invoked with the preprocessed input.
    fake_session.run.assert_called_once()
    args, _kwargs = fake_session.run.call_args
    assert args[0] is None
    fed = args[1]
    assert "data" in fed
    assert fed["data"].shape == (1, 3, imgsz, imgsz)
    assert fed["data"].dtype == np.float32


def test_nanodet_adapter_pytorch_backend_uses_torch_factory() -> None:
    """The .pth backend lazy-imports nanodet; in tests we inject a torch_factory."""
    fake_model = MagicMock()
    # Upstream's `inference()` returns {img_id: {class_id: [[x1,y1,x2,y2,conf], ...]}}
    fake_model.inference.return_value = {
        0: {
            COCO_CAT_CLASS: [[10.0, 20.0, 30.0, 40.0, 0.88]],
            0: [[0.0, 0.0, 50.0, 50.0, 0.99]],  # person — must drop
        }
    }
    fake_model.parameters.return_value = [MagicMock(numel=lambda: 100), MagicMock(numel=lambda: 200)]

    with patch.dict("sys.modules", {"torch": MagicMock()}):
        import sys
        torch_mod = sys.modules["torch"]
        torch_mod.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        torch_mod.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        torch_mod.from_numpy.side_effect = lambda x: x

        adapter = NanodetAdapter(
            model_path="/fake/path/nanodet_plus_m_0.5x_pretrained.pth",
            imgsz=100,
            confidence_threshold=0.5,
            backend="pytorch",
            torch_factory=lambda _path: fake_model,
        )
        out = adapter.predict(np.zeros((100, 100, 3), dtype=np.uint8))

    assert adapter.num_params() == 300
    assert len(out) == 1
    assert out[0]["confidence"] == pytest.approx(0.88, abs=1e-5)
    # bbox normalized by imgsz=100
    assert out[0]["bbox"] == pytest.approx([0.10, 0.20, 0.30, 0.40], abs=1e-5)


def test_pytorch_backend_raises_when_nanodet_missing(tmp_path: Path) -> None:
    """When neither torch_factory nor nanodet pip is available, raise informative."""
    # Hide the nanodet module name from sys.modules to force the real import path.
    import importlib
    import sys

    saved = sys.modules.pop("nanodet", None)
    try:
        with patch.object(importlib, "import_module", side_effect=ImportError("nope")):
            with pytest.raises(ImportError, match="setup_venv.sh|isolated venv"):
                NanodetAdapter(
                    model_path=str(tmp_path / "missing.pth"),
                    imgsz=416,
                    backend="pytorch",
                )
    finally:
        if saved is not None:
            sys.modules["nanodet"] = saved


def test_run_eval_dispatches_nanodet_path_regardless_of_format() -> None:
    """`_load_adapter` routes nanodet/ paths through nanodet_adapter even with --format pytorch."""
    from training.edge.eval import run_eval as run_eval_mod

    with patch(
        "training.edge.eval.adapters.nanodet_adapter.load",
        return_value=MagicMock(input_hw=(416, 416)),
    ) as mocked_load:
        run_eval_mod._load_adapter(
            "training/edge/nanodet/checkpoints/nanodet_plus_m_0.5x_pretrained.pth",
            "pytorch",
            416,
        )
    mocked_load.assert_called_once()
    _args, kwargs = mocked_load.call_args
    assert kwargs.get("backend") == "pytorch"
    assert kwargs.get("imgsz") == 416


def test_run_eval_dispatches_nanodet_onnx_path() -> None:
    from training.edge.eval import run_eval as run_eval_mod

    with patch(
        "training.edge.eval.adapters.nanodet_adapter.load",
        return_value=MagicMock(input_hw=(416, 416)),
    ) as mocked_load:
        run_eval_mod._load_adapter(
            "training/edge/nanodet/checkpoints/nanodet_plus_m_416.onnx",
            "onnx",
            416,
        )
    mocked_load.assert_called_once()
    _args, kwargs = mocked_load.call_args
    assert kwargs.get("backend") == "onnx"


def test_run_eval_does_not_dispatch_yolo_paths_to_nanodet() -> None:
    from training.edge.eval import run_eval as run_eval_mod

    with patch(
        "training.edge.eval.adapters.pytorch_adapter.load",
        return_value=MagicMock(input_hw=(224, 224)),
    ) as mocked_yolo_load, patch(
        "training.edge.eval.adapters.nanodet_adapter.load"
    ) as mocked_nanodet_load:
        run_eval_mod._load_adapter("training/edge/models/yolov8n_cat.pt", "pytorch", 224)
    mocked_yolo_load.assert_called_once()
    mocked_nanodet_load.assert_not_called()
