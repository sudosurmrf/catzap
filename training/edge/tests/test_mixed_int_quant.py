"""iter-D: tests for the INT8-weights / INT16-activations PTQ pipeline.

Mirrors test_per_channel_quant.py from iter-A. All TF interactions go
through DI seams (``converter_factory`` / ``tf_module`` / ``onnx_exporter``
/ ``savedmodel_converter`` / ``quantizer``) so this file does NOT need TF
or onnx2tf installed at import time. Each test runs in <100 ms.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from training.edge.yolo import export_quantize as yq
from training.edge.yolo import mixed_int_quant as miq


# ---------------------------------------------------------------------------
# Test 1: converter is configured with the INT16x8 supported_ops set, INT16
# input, FLOAT32 output, and the MLIR new quantizer enabled. This is the
# load-bearing iter-D delta vs iter-A's per-channel converter step.
# ---------------------------------------------------------------------------
def test_quantize_uses_int16x8_supported_ops_float_io(
    tmp_path: Path,
) -> None:
    converter = MagicMock()
    converter.convert.return_value = b"FAKE_TFLITE_BYTES_INT16x8"
    converter_factory = MagicMock(return_value=converter)
    tf_module = SimpleNamespace(
        lite=SimpleNamespace(
            Optimize=SimpleNamespace(DEFAULT="OPTIMIZE_DEFAULT"),
            OpsSet=SimpleNamespace(
                EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8=(
                    "INT16x8_OPSET"
                ),
            ),
        ),
        float32="DTYPE_FLOAT32",
    )

    saved_model_dir = tmp_path / "saved_model"
    saved_model_dir.mkdir()
    out_path = tmp_path / "out.tflite"

    miq.quantize_to_int16x8_tflite(
        saved_model_dir=saved_model_dir,
        representative_dataset=lambda: iter([]),
        out_path=out_path,
        converter_factory=converter_factory,
        tf_module=tf_module,
    )

    # Supported ops set is the INT16x8 hybrid — load-bearing iter-D delta.
    assert converter.target_spec.supported_ops == ["INT16x8_OPSET"]
    # Float IO — keeps the existing tflite_adapter compatible (no new int16
    # branch needed); the converter prepends QUANTIZE / appends DEQUANTIZE.
    assert converter.inference_input_type == "DTYPE_FLOAT32"
    assert converter.inference_output_type == "DTYPE_FLOAT32"
    # MLIR quantizer remains on for per-channel weight scales.
    assert converter.experimental_new_quantizer is True
    # Optimization preset and rep_ds wiring are unchanged from iter-A.
    assert converter.optimizations == ["OPTIMIZE_DEFAULT"]
    assert callable(converter.representative_dataset)
    # And the file got written.
    assert out_path.read_bytes() == b"FAKE_TFLITE_BYTES_INT16x8"
    converter_factory.assert_called_once_with(str(saved_model_dir))


# ---------------------------------------------------------------------------
# Test 2: calibration_loader is the SAME callable as in export_quantize and
# also identical to the iter-A module's binding. The codebase pattern
# "shared calibration_loader is the contract" covers iter-D too.
# ---------------------------------------------------------------------------
def test_calibration_loader_is_shared_with_export_quantize() -> None:
    from training.edge.yolo import per_channel_quant as pcq

    assert miq.calibration_loader is yq.calibration_loader
    assert miq.calibration_loader is pcq.calibration_loader


# ---------------------------------------------------------------------------
# Test 3: end-to-end DI — export_mixed_int_quant chains exporter, converter,
# quantizer and feeds the (calib_dir, imgsz, max_calib_frames) into the
# representative_dataset that lands in the quantizer call.
# ---------------------------------------------------------------------------
def test_export_mixed_int_quant_dependency_injection_full_chain(tmp_path: Path) -> None:
    pt = tmp_path / "fake.pt"
    pt.write_bytes(b"")

    work = tmp_path / "work"
    calib = tmp_path / "calib"
    calib.mkdir()
    (calib / "frame_0001.jpg").write_bytes(b"")  # path-existence only; gen lazy

    onnx_exporter = MagicMock(return_value=work / "fake.onnx")
    saved_model_dir = work / "saved_model"
    savedmodel_converter = MagicMock(return_value=saved_model_dir)

    captured: dict = {}

    def quantizer_stub(saved_dir, rep_ds, out_path):
        captured["saved_dir"] = saved_dir
        captured["rep_ds"] = rep_ds
        captured["out_path"] = out_path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"OUT")
        return out_path

    out = miq.export_mixed_int_quant(
        pt_path=pt,
        calib_dir=calib,
        out_path=tmp_path / "out.tflite",
        imgsz=192,
        max_calib_frames=8,
        work_dir=work,
        onnx_exporter=onnx_exporter,
        savedmodel_converter=savedmodel_converter,
        quantizer=quantizer_stub,
    )

    onnx_exporter.assert_called_once()
    onnx_args = onnx_exporter.call_args
    assert onnx_args.kwargs.get("imgsz") == 192 or 192 in onnx_args.args
    savedmodel_converter.assert_called_once()
    sm_args = savedmodel_converter.call_args
    assert sm_args.args[0] == work / "fake.onnx"
    assert sm_args.args[1] == work / "saved_model"
    assert captured["saved_dir"] == saved_model_dir
    assert captured["out_path"] == tmp_path / "out.tflite"
    assert callable(captured["rep_ds"])
    assert out == tmp_path / "out.tflite"


# ---------------------------------------------------------------------------
# Test 4: the CLI's blocked-fallback path emits a well-formed .export.json
# sidecar even when the heavy quantize step raises (e.g. TFLM-incompatible
# op or onnx2tf shape error). The PRD permits "blocked" iter-D outcomes —
# but the JSON MUST still land so the iter-H aggregator can render the row.
# ---------------------------------------------------------------------------
def test_main_writes_blocked_export_sidecar_on_quantize_failure(
    tmp_path: Path, monkeypatch
) -> None:
    out_path = tmp_path / "out.tflite"

    def boom(*_a, **_kw):
        raise RuntimeError("simulated onnx2tf NHWC layout failure")

    monkeypatch.setattr(miq, "export_mixed_int_quant", boom)

    rc = miq.main(
        [
            "--model",
            str(tmp_path / "fake.pt"),
            "--calib-dir",
            str(tmp_path / "calib"),
            "--out",
            str(out_path),
            "--imgsz",
            "224",
        ]
    )
    # Even on failure the CLI returns 0 and writes the sidecar so callers
    # (aggregate_iter_d.py) can encode the blocked reason in the rollup.
    assert rc == 0
    sidecar = out_path.with_suffix(out_path.suffix + ".export.json")
    assert sidecar.exists()
    import json as _json

    payload = _json.loads(sidecar.read_text())
    assert payload["story_id"] == "iter-D"
    assert payload["blocked_reason"] is not None
    assert "NHWC" in payload["blocked_reason"]
    assert payload["size_bytes"] == 0
    assert (
        payload["supported_ops"]
        == "EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8"
    )
