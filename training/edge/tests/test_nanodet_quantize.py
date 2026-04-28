"""Tests for training/edge/nanodet/export_quantize.py.

The PRD-required acceptance test is the first one: ``calibration_loader`` must
be the SAME callable imported from ``training.edge.yolo.export_quantize`` (not
a separately-defined one) so the YOLO and NanoDet INT8 paths share their
calibration set byte-for-byte. Everything else is DI-based mock testing in
the same shape as ``test_quantize.py`` so the suite stays fast and runs
without TF / nanodet installed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from training.edge.nanodet import export_quantize as nq
from training.edge.yolo import export_quantize as yq


# --- shared calibration_loader --------------------------------------------


def test_calibration_loader_is_imported_from_yolo_export_quantize() -> None:
    """The PRD acceptance criterion: calibration_loader is the SAME callable
    imported from training.edge.yolo.export_quantize, not a separately-defined
    function. This guarantees US-004 (YOLOv8n INT8) and US-009 (NanoDet INT8)
    quantize against the IDENTICAL calibration set, which makes the apples-to-
    apples mAP comparison required by US-012 well-defined.
    """
    assert nq.calibration_loader is yq.calibration_loader


def test_calibration_loader_yields_same_frames_at_two_imgsz(
    tmp_path: Path,
) -> None:
    """Sanity: the same shared loader is parametrized purely by imgsz; the
    image set it draws from is whatever ``calib_dir`` points at. So pointing
    both YOLO (imgsz=224) and NanoDet (imgsz=224 or 416) at the same dir gives
    deterministic, identical sample paths in the same order.
    """
    import cv2

    calib = tmp_path / "calib"
    calib.mkdir()
    for i in range(5):
        cv2.imwrite(
            str(calib / f"frame_{i:04d}.jpg"),
            np.full((32, 32, 3), i * 30, dtype=np.uint8),
        )

    loader_a = nq.calibration_loader(calib_dir=calib, imgsz=224, max_frames=5)
    loader_b = yq.calibration_loader(calib_dir=calib, imgsz=224, max_frames=5)

    items_a = list(loader_a())
    items_b = list(loader_b())
    assert len(items_a) == len(items_b) == 5
    for a, b in zip(items_a, items_b):
        assert np.array_equal(a[0], b[0])


# --- export_pth_to_onnx ----------------------------------------------------


def test_export_pth_to_onnx_invokes_nanodet_main_with_correct_kwargs(
    tmp_path: Path,
) -> None:
    """export_pth_to_onnx must wrap nanodet.tools.export_onnx.main with the
    config / model / output / input_shape kwargs, lazy-importing nanodet."""
    pth_path = tmp_path / "weights.pth"
    pth_path.write_bytes(b"fake-pth")
    config_path = tmp_path / "cat.yml"
    config_path.write_text("model: { arch: { name: NanoDetPlus }}\n")
    out_path = tmp_path / "out.onnx"

    fake_main = MagicMock()
    result = nq.export_pth_to_onnx(
        pth_path,
        config_path,
        out_path,
        imgsz=224,
        nanodet_export_main=fake_main,
    )

    fake_main.assert_called_once_with(
        config_path=str(config_path),
        model_path=str(pth_path),
        output_path=str(out_path),
        input_shape=(224, 224),
    )
    assert result == out_path


# --- export_and_quantize orchestrator -------------------------------------


def test_export_and_quantize_chains_three_stages_with_di(tmp_path: Path) -> None:
    """The orchestrator must thread pth -> onnx -> savedmodel -> tflite via
    the DI hooks. Verify intermediate paths flow through correctly."""
    pth_path = tmp_path / "model.pth"
    pth_path.write_bytes(b"fake-pth")
    config_path = tmp_path / "cat.yml"
    config_path.write_text("model: { arch: { name: NanoDetPlus }}\n")

    # Seed a calibration dir so calibration_loader doesn't error.
    import cv2

    calib = tmp_path / "calib"
    calib.mkdir()
    for i in range(3):
        cv2.imwrite(
            str(calib / f"f_{i:04d}.jpg"), np.zeros((32, 32, 3), dtype=np.uint8)
        )

    work = tmp_path / "work"
    out = tmp_path / "model_int8.tflite"

    fake_onnx = work / "model.onnx"
    fake_sm = work / "saved_model"

    onnx_exporter = MagicMock(return_value=fake_onnx)
    savedmodel_converter = MagicMock(return_value=fake_sm)

    def _quantizer(saved_model_dir: Path, rep_ds, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake-tflite")
        return out_path

    quantizer = MagicMock(side_effect=_quantizer)

    final = nq.export_and_quantize(
        pth_path=pth_path,
        config_path=config_path,
        calib_dir=calib,
        out_path=out,
        imgsz=224,
        max_calib_frames=3,
        work_dir=work,
        onnx_exporter=onnx_exporter,
        savedmodel_converter=savedmodel_converter,
        quantizer=quantizer,
    )

    assert final == out
    assert final.exists()
    onnx_exporter.assert_called_once()
    savedmodel_converter.assert_called_once_with(fake_onnx, work / "saved_model")
    qargs = quantizer.call_args
    assert qargs.args[0] == fake_sm
    assert callable(qargs.args[1])
    assert qargs.args[2] == out


def test_export_and_quantize_skips_pth_export_when_onnx_supplied(
    tmp_path: Path,
) -> None:
    """If --onnx is passed, the .pth -> ONNX exporter is not called: the
    pre-existing ONNX is used directly. This is how the pretrained nanodet
    ONNX gets quantized as a smoke test without needing the isolated venv."""
    config_path = tmp_path / "cat.yml"
    config_path.write_text("# placeholder\n")

    import cv2

    calib = tmp_path / "calib"
    calib.mkdir()
    for i in range(3):
        cv2.imwrite(
            str(calib / f"f_{i:04d}.jpg"), np.zeros((32, 32, 3), dtype=np.uint8)
        )

    work = tmp_path / "work"
    out = tmp_path / "model_int8.tflite"

    pre_onnx = tmp_path / "pre_existing.onnx"
    pre_onnx.write_bytes(b"fake-pre-onnx")

    onnx_exporter = MagicMock()
    savedmodel_converter = MagicMock(return_value=work / "saved_model")

    def _quantizer(_sm: Path, _rep, op: Path) -> Path:
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_bytes(b"fake-tflite")
        return op

    quantizer = MagicMock(side_effect=_quantizer)

    final = nq.export_and_quantize(
        pth_path=tmp_path / "ignored.pth",
        config_path=config_path,
        onnx_path=pre_onnx,
        calib_dir=calib,
        out_path=out,
        imgsz=224,
        max_calib_frames=3,
        work_dir=work,
        onnx_exporter=onnx_exporter,
        savedmodel_converter=savedmodel_converter,
        quantizer=quantizer,
    )

    assert final == out
    onnx_exporter.assert_not_called()
    # The savedmodel_converter is invoked with the ONNX *copied into* work_dir,
    # not the original location, so subsequent stages see consistent paths.
    sm_args, _ = savedmodel_converter.call_args
    assert sm_args[0] == work / "pre_existing.onnx"
    assert sm_args[0].exists()


# --- shared converter behavior ---------------------------------------------


def test_quantize_uses_same_converter_settings_as_yolo() -> None:
    """quantize_to_int8_tflite is also re-imported from the YOLO module so
    every converter knob (Optimize.DEFAULT, BUILTINS_INT8, int8 in/out dtype,
    experimental_new_quantizer) is identical between the two paths.
    """
    assert nq.quantize_to_int8_tflite is yq.quantize_to_int8_tflite
    assert nq.onnx_to_savedmodel is yq.onnx_to_savedmodel


# --- representative dataset >= 100 ------------------------------------------


def test_representative_dataset_yields_at_least_100_frames(tmp_path: Path) -> None:
    """The shared calibration_loader, when given >=100 frames, yields >=100
    samples for the converter — same threshold as US-004 and US-006."""
    import cv2

    calib = tmp_path / "calib"
    calib.mkdir()
    for i in range(150):
        cv2.imwrite(
            str(calib / f"f_{i:04d}.jpg"), np.full((32, 32, 3), i % 200, dtype=np.uint8)
        )
    rep_ds = nq.calibration_loader(calib_dir=calib, imgsz=64, max_frames=150)
    assert sum(1 for _ in rep_ds()) >= 100


# --- tflite backend dispatch ------------------------------------------------


def test_nanodet_adapter_dispatches_tflite_path_to_tflite_backend() -> None:
    """The adapter must auto-detect a .tflite path and use the tflite backend
    (rather than trying to load it as ONNX or pytorch)."""
    from training.edge.eval.adapters.nanodet_adapter import NanodetAdapter

    fake_interp = MagicMock()
    fake_interp.get_input_details.return_value = [
        {"index": 0, "shape": [1, 224, 224, 3], "dtype": np.float32, "quantization": (0.0, 0)}
    ]
    fake_interp.get_output_details.return_value = [
        {"index": 1, "shape": [1, 700, 112], "dtype": np.float32, "quantization": (0.0, 0)}
    ]
    fake_interp.get_tensor_details.return_value = []

    factory = MagicMock(return_value=fake_interp)
    adapter = NanodetAdapter(
        "training/edge/models/nanodet_cat_0.5x_224_int8.tflite",
        imgsz=224,
        interpreter_factory=factory,
    )
    assert adapter.backend == "tflite"
    factory.assert_called_once()
    fake_interp.allocate_tensors.assert_called_once()


def test_nanodet_adapter_tflite_predict_runs_decoder(tmp_path: Path) -> None:
    """The TFLite backend must dequantize int8 output, normalize cls if needed,
    and feed decode_nanodet_output. We verify by injecting a synthetic high-
    confidence output and asserting at least one detection comes back."""
    from training.edge.eval.adapters.nanodet_adapter import (
        DEFAULT_REG_MAX,
        NanodetAdapter,
    )

    imgsz = 224
    num_classes = 80
    reg_max = DEFAULT_REG_MAX
    # Match strides (8,16,32,64) at imgsz=224 -> ceil(224/s) per stride
    import math

    num_priors = sum(math.ceil(imgsz / s) ** 2 for s in (8, 16, 32, 64))
    channels = num_classes + 4 * (reg_max + 1)
    fake_out = np.zeros((1, num_priors, channels), dtype=np.float32)
    # crank the cat channel (idx 15) above threshold on prior 0
    fake_out[0, 0, 15] = 5.0  # logit; will be sigmoid'd by post_quant normalize
    # set DFL so distances are non-degenerate
    fake_out[0, 0, num_classes : num_classes + (reg_max + 1)] = np.linspace(0, 1, reg_max + 1)

    fake_interp = MagicMock()
    fake_interp.get_input_details.return_value = [
        {"index": 0, "shape": [1, imgsz, imgsz, 3], "dtype": np.float32, "quantization": (0.0, 0)}
    ]
    fake_interp.get_output_details.return_value = [
        {"index": 1, "shape": list(fake_out.shape), "dtype": np.float32, "quantization": (0.0, 0)}
    ]
    fake_interp.get_tensor_details.return_value = []
    fake_interp.get_tensor.return_value = fake_out

    adapter = NanodetAdapter(
        "training/edge/models/nanodet_cat_0.5x_224_int8.tflite",
        imgsz=imgsz,
        confidence_threshold=0.25,
        interpreter_factory=MagicMock(return_value=fake_interp),
    )
    frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    detections = adapter.predict(frame)
    fake_interp.invoke.assert_called_once()
    assert isinstance(detections, list)
    assert len(detections) >= 1
    assert all("bbox" in d and "confidence" in d for d in detections)


# --- run_eval routing -------------------------------------------------------


def test_run_eval_routes_nanodet_tflite_path_to_nanodet_adapter() -> None:
    """run_eval._is_nanodet_path matches the canonical US-009 output filename
    so the eval harness picks up the NanoDet TFLite via the NanoDet adapter
    (with its DFL decoder), not the YOLO TFLite adapter."""
    from training.edge.eval.run_eval import _is_nanodet_path

    assert _is_nanodet_path(
        "training/edge/models/nanodet_cat_0.5x_224_int8.tflite"
    )
    assert _is_nanodet_path(
        "training/edge/nanodet/checkpoints/nanodet_cat_0.5x_224.pth"
    )
    # Unrelated YOLO models still go to the YOLO adapter.
    assert not _is_nanodet_path("training/edge/models/yolov8n_cat_int8.tflite")


def test_post_quant_score_normalize_sigmoids_when_out_of_range() -> None:
    """The dequantized cls scores can land outside [0,1] due to per-tensor
    output quant; _post_quant_score_normalize must apply sigmoid to push them
    back. Pass-through when already in [0,1]."""
    from training.edge.eval.adapters.nanodet_adapter import (
        DEFAULT_REG_MAX,
        DEFAULT_NUM_CLASSES,
        _post_quant_score_normalize,
    )

    bins = DEFAULT_REG_MAX + 1
    channels = DEFAULT_NUM_CLASSES + 4 * bins
    arr_logits = np.zeros((1, 4, channels), dtype=np.float32)
    arr_logits[0, 0, 0] = 5.0  # large logit
    arr_logits[0, 1, 0] = -5.0
    out = _post_quant_score_normalize(arr_logits)
    assert pytest.approx(float(out[0, 0, 0]), abs=1e-3) == 1.0 / (1.0 + np.exp(-5.0))
    assert pytest.approx(float(out[0, 1, 0]), abs=1e-3) == 1.0 / (1.0 + np.exp(5.0))

    # Already in [0,1] -> pass-through unchanged
    arr_already = np.zeros((1, 4, channels), dtype=np.float32)
    arr_already[0, 0, 0] = 0.7
    out2 = _post_quant_score_normalize(arr_already)
    assert pytest.approx(float(out2[0, 0, 0]), abs=1e-6) == 0.7
