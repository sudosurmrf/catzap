"""Tests for training/edge/yolo/export_quantize.py — mocked end-to-end.

Mirrors the test pattern in test_yolo_train.py: every heavy dependency
(ultralytics.YOLO, onnx2tf, tf.lite.TFLiteConverter) is swapped via DI hooks
on the public functions so the suite runs in <1 s without TF/onnx installed.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from training.edge.yolo import export_quantize as eq


# --- calibration_loader -----------------------------------------------------


def _seed_calib_dir(tmp_path: Path, n: int, imgsz: int = 32) -> Path:
    """Write n tiny BGR JPEGs into tmp_path/calib and return that dir."""
    import cv2

    calib = tmp_path / "calib"
    calib.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        # deterministic gradient — content doesn't matter, just needs to decode
        img = np.full((imgsz, imgsz, 3), i % 200, dtype=np.uint8)
        cv2.imwrite(str(calib / f"frame_{i:04d}.jpg"), img)
    return calib


def test_calibration_loader_yields_nhwc_float32_zero_one(tmp_path: Path) -> None:
    """representative_dataset must yield (1, imgsz, imgsz, 3) float32 in [0,1]."""
    calib = _seed_calib_dir(tmp_path, n=5)
    rep_ds = eq.calibration_loader(calib_dir=calib, imgsz=224, max_frames=5)

    items = list(rep_ds())
    assert len(items) == 5
    for item in items:
        assert isinstance(item, list) and len(item) == 1
        x = item[0]
        assert x.dtype == np.float32
        assert x.shape == (1, 224, 224, 3)  # NHWC
        assert x.min() >= 0.0 and x.max() <= 1.0


def test_calibration_loader_caps_at_max_frames(tmp_path: Path) -> None:
    """max_frames truncates the calibration set even when more files exist."""
    calib = _seed_calib_dir(tmp_path, n=10)
    rep_ds = eq.calibration_loader(calib_dir=calib, imgsz=64, max_frames=3)
    assert len(list(rep_ds())) == 3


def test_calibration_loader_missing_dir_raises(tmp_path: Path) -> None:
    """Pointing the loader at a non-existent dir is a hard error."""
    with pytest.raises(FileNotFoundError):
        eq.calibration_loader(calib_dir=tmp_path / "missing", imgsz=224)


def test_calibration_loader_yields_at_least_100_frames_for_real_calib_set(
    tmp_path: Path,
) -> None:
    """The PTQ pipeline requires a minimum of 100 calibration steps; verify the
    loader can supply that many. Mirrors US-004 acceptance criterion that
    representative_dataset is called >= 100 times."""
    calib = _seed_calib_dir(tmp_path, n=150)
    rep_ds = eq.calibration_loader(calib_dir=calib, imgsz=64, max_frames=150)
    n_yielded = sum(1 for _ in rep_ds())
    assert n_yielded >= 100


# --- quantize_to_int8_tflite ------------------------------------------------


def _make_fake_tf_module() -> SimpleNamespace:
    """Build a stub tensorflow module shaped like the real one for our calls."""
    fake_int8 = object()
    fake_default = object()
    fake_int8_op = object()

    class _ConverterCls:
        from_saved_model = MagicMock()

    converter = MagicMock()
    # Configure target_spec and other attrs so the production code's setattr
    # calls succeed without errors.
    converter.target_spec = MagicMock()
    converter.convert.return_value = b"\x00fake-tflite-bytes"
    _ConverterCls.from_saved_model.return_value = converter

    fake_tf = SimpleNamespace(
        int8=fake_int8,
        lite=SimpleNamespace(
            Optimize=SimpleNamespace(DEFAULT=fake_default),
            OpsSet=SimpleNamespace(TFLITE_BUILTINS_INT8=fake_int8_op),
            TFLiteConverter=_ConverterCls,
        ),
    )
    return fake_tf


def test_quantize_to_int8_tflite_sets_int8_dtypes_and_calls_converter(
    tmp_path: Path,
) -> None:
    """The converter MUST be configured with INT8 input/output dtypes per US-004."""
    fake_tf = _make_fake_tf_module()
    # Use a short representative dataset for this single check
    rep_ds = lambda: ([np.zeros((1, 32, 32, 3), dtype=np.float32)] for _ in range(5))

    out = tmp_path / "model_int8.tflite"
    eq.quantize_to_int8_tflite(
        saved_model_dir=tmp_path / "saved_model",
        representative_dataset=rep_ds,
        out_path=out,
        tf_module=fake_tf,
    )

    # The converter was created via from_saved_model
    fake_tf.lite.TFLiteConverter.from_saved_model.assert_called_once()
    converter = fake_tf.lite.TFLiteConverter.from_saved_model.return_value

    assert converter.optimizations == [fake_tf.lite.Optimize.DEFAULT]
    assert converter.representative_dataset is rep_ds
    assert converter.target_spec.supported_ops == [
        fake_tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    assert converter.inference_input_type is fake_tf.int8
    assert converter.inference_output_type is fake_tf.int8

    # Output bytes are written to disk
    assert out.exists()
    assert out.read_bytes() == b"\x00fake-tflite-bytes"


def test_quantize_representative_dataset_yields_at_least_100_frames(
    tmp_path: Path,
) -> None:
    """When wired up to a real calibration_loader, the representative_dataset
    handed to the converter yields >= 100 calibration frames (US-004 criterion).
    """
    calib = _seed_calib_dir(tmp_path, n=150)
    rep_ds = eq.calibration_loader(calib_dir=calib, imgsz=64, max_frames=150)

    fake_tf = _make_fake_tf_module()
    eq.quantize_to_int8_tflite(
        saved_model_dir=tmp_path / "saved_model",
        representative_dataset=rep_ds,
        out_path=tmp_path / "out.tflite",
        tf_module=fake_tf,
    )

    converter = fake_tf.lite.TFLiteConverter.from_saved_model.return_value
    captured = converter.representative_dataset
    n_calls = sum(1 for _ in captured())
    assert n_calls >= 100


# --- export_and_quantize orchestrator ---------------------------------------


def test_export_and_quantize_chains_three_stages_with_di(tmp_path: Path) -> None:
    """The orchestrator threads pt -> onnx -> savedmodel -> tflite via the
    DI hooks. We verify the right intermediate paths are passed downstream."""
    pt_path = tmp_path / "model.pt"
    pt_path.write_bytes(b"fake-pt")
    calib = _seed_calib_dir(tmp_path, n=3)
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

    final = eq.export_and_quantize(
        pt_path=pt_path,
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
    # quantizer received our saved_model dir, a callable rep_ds, and the out path
    qargs = quantizer.call_args
    assert qargs.args[0] == fake_sm
    assert callable(qargs.args[1])
    assert qargs.args[2] == out


def test_export_to_onnx_invokes_ultralytics_with_correct_kwargs(
    tmp_path: Path,
) -> None:
    """export_to_onnx must call YOLO(.pt).export(format='onnx', dynamic=False, simplify=True)."""
    pt_path = tmp_path / "weights.pt"
    pt_path.write_bytes(b"fake-pt")
    fake_model = MagicMock()
    fake_onnx_src = tmp_path / "weights.onnx"
    fake_onnx_src.write_bytes(b"fake-onnx")
    fake_model.export.return_value = str(fake_onnx_src)
    fake_factory = MagicMock(return_value=fake_model)

    out_dir = tmp_path / "onnx_out"
    result = eq.export_to_onnx(pt_path, out_dir, imgsz=224, yolo_factory=fake_factory)

    fake_factory.assert_called_once_with(str(pt_path))
    fake_model.export.assert_called_once()
    kwargs = fake_model.export.call_args.kwargs
    assert kwargs["format"] == "onnx"
    assert kwargs["imgsz"] == 224
    assert kwargs["dynamic"] is False
    assert kwargs["simplify"] is True
    assert result.exists()
    assert result.suffix == ".onnx"


def test_onnx_to_savedmodel_invokes_onnx2tf_with_nhwc_signaturedefs(
    tmp_path: Path,
) -> None:
    """onnx_to_savedmodel must request signaturedefs for the NHWC SavedModel."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    out_dir = tmp_path / "savedmodel"
    fake_converter = MagicMock()

    eq.onnx_to_savedmodel(onnx_path, out_dir, converter=fake_converter)

    fake_converter.assert_called_once()
    kwargs = fake_converter.call_args.kwargs
    assert kwargs["input_onnx_file_path"] == str(onnx_path)
    assert kwargs["output_folder_path"] == str(out_dir)
    assert kwargs["output_signaturedefs"] is True
