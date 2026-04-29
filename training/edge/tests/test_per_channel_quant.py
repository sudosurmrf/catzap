"""iter-A: tests for the per-channel-friendly INT8 PTQ pipeline.

Mirrors test_quantize.py from US-004. All TF interactions go through DI seams
(``converter_factory`` / ``tf_module`` / ``onnx_exporter`` / ``savedmodel_converter``
/ ``quantizer``) so this file does NOT need TF or onnx2tf installed at import
time. Each test runs in <100 ms.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from training.edge.yolo import export_quantize as yq
from training.edge.yolo import per_channel_quant as pcq


# ---------------------------------------------------------------------------
# Test 1: converter is configured with int8 INPUT and float32 OUTPUT (the
# load-bearing iter-A delta vs US-004's converter step).
# ---------------------------------------------------------------------------
def test_quantize_uses_int8_input_and_float32_output(tmp_path: Path) -> None:
    converter = MagicMock()
    converter.convert.return_value = b"FAKE_TFLITE_BYTES"
    converter_factory = MagicMock(return_value=converter)
    tf_module = SimpleNamespace(
        lite=SimpleNamespace(
            Optimize=SimpleNamespace(DEFAULT="OPTIMIZE_DEFAULT"),
            OpsSet=SimpleNamespace(TFLITE_BUILTINS_INT8="BUILTINS_INT8"),
        ),
        int8="DTYPE_INT8",
        float32="DTYPE_FLOAT32",
    )

    saved_model_dir = tmp_path / "saved_model"
    saved_model_dir.mkdir()
    out_path = tmp_path / "out.tflite"

    pcq.quantize_to_int8_float_output_tflite(
        saved_model_dir=saved_model_dir,
        representative_dataset=lambda: iter([]),
        out_path=out_path,
        converter_factory=converter_factory,
        tf_module=tf_module,
    )

    # Input dtype is int8 (kept from v1 — TFLM accepts int8 input on ESP32-P4).
    assert converter.inference_input_type == "DTYPE_INT8"
    # Output dtype is FLOAT32 — this is the iter-A fix that prevents per-tensor
    # cls collapse on the YOLOv8 Detect head.
    assert converter.inference_output_type == "DTYPE_FLOAT32"
    # MLIR quantizer also stays on for per-channel weight scales.
    assert converter.experimental_new_quantizer is True
    # And the file got written.
    assert out_path.read_bytes() == b"FAKE_TFLITE_BYTES"
    converter_factory.assert_called_once_with(str(saved_model_dir))


# ---------------------------------------------------------------------------
# Test 2: calibration_loader is the SAME callable as in export_quantize. This
# is load-bearing — progress.txt's "shared calibration_loader is the contract"
# pattern says future stories that pull in calibration must `is`-identity it.
# ---------------------------------------------------------------------------
def test_calibration_loader_is_shared_with_export_quantize() -> None:
    assert pcq.calibration_loader is yq.calibration_loader


# ---------------------------------------------------------------------------
# Test 3: representative_dataset is exhausted >=100 times. Mirror the v1
# US-004 acceptance ("representative_dataset is called >= 100 times").
# ---------------------------------------------------------------------------
def test_representative_dataset_yields_at_least_100_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Build 120 dummy "calibration images" — only file existence matters; the
    # generator inside calibration_loader actually opens them via cv2 lazily.
    # Use real cv2.imwrite so we don't have to monkey-patch sys.modules['cv2']
    # (doing so leaks into other test files via the import cache).
    import cv2  # real cv2

    import numpy as np

    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    fake_images = []
    for i in range(120):
        p = calib_dir / f"frame_{i:04d}.jpg"
        # 224x224x3 deterministic-content frame; cv2.imread will load it back.
        cv2.imwrite(str(p), np.full((224, 224, 3), (i % 250) + 1, dtype=np.uint8))
        fake_images.append(p)

    # Monkeypatch _list_calib_images to skip the disk-scan filtering and force
    # exactly the 120 frames we wrote.
    monkeypatch.setattr(yq, "_list_calib_images", lambda d, mf: fake_images[:mf])

    gen_factory = pcq.calibration_loader(
        calib_dir=calib_dir, imgsz=224, max_frames=120
    )
    # Drain the generator like tf.lite would.
    n = sum(1 for _ in gen_factory())
    assert n == 120, f"expected 120 calibration tensors, got {n}"
    assert n >= 100, "PRD requires >=100 representative_dataset calls"


# ---------------------------------------------------------------------------
# Test 4: end-to-end DI — export_per_channel_int8 chains exporter, converter,
# quantizer and feeds the (calib_dir, imgsz, max_calib_frames) into the
# representative_dataset that lands in the quantizer call.
# ---------------------------------------------------------------------------
def test_export_per_channel_int8_dependency_injection_full_chain(tmp_path: Path) -> None:
    pt = tmp_path / "fake.pt"
    pt.write_bytes(b"")  # path needs to exist for hypothetical stat checks

    work = tmp_path / "work"

    # calibration_loader needs the dir to exist with >=1 image listing.
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

    out = pcq.export_per_channel_int8(
        pt_path=pt,
        calib_dir=calib,  # we never enter calibration_loader's generator inside the test
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
    assert callable(captured["rep_ds"])  # representative_dataset is a callable
    assert out == tmp_path / "out.tflite"
    assert (tmp_path / "out.tflite").read_bytes() == b"OUT"


# ---------------------------------------------------------------------------
# Test 5: build_pareto_verdict — dominates branch (size shrinks past threshold).
# ---------------------------------------------------------------------------
def test_pareto_dominates_when_one_axis_passes_threshold() -> None:
    v = pcq.build_pareto_verdict(
        # Candidate: same mAP, 30% smaller (>20% threshold).
        candidate_map50=0.205,
        candidate_size_bytes=int(3_220_840 * 0.7),
        candidate_arena_used_bytes=692_816,
        candidate_predicted_p4_latency_ms_p50=5_883.36,
        # Baseline: US-006-int8 numbers from v1.
        baseline_story="US-006-int8",
        baseline_map50=0.205,
        baseline_size_bytes=3_220_840,
        baseline_arena_used_bytes=692_816,
        baseline_predicted_p4_latency_ms_p50=5_883.36,
    )
    assert v["verdict"] == "dominates"
    assert v["deltas"]["size_pct"] < -pcq.SIZE_IMPROVE_PCT
    assert v["baseline_story"] == "US-006-int8"


# ---------------------------------------------------------------------------
# Test 6: build_pareto_verdict — equal branch (no axis past threshold).
# ---------------------------------------------------------------------------
def test_pareto_equal_when_no_axis_past_threshold() -> None:
    v = pcq.build_pareto_verdict(
        # Candidate: same mAP, only 5% smaller (well under 20%).
        candidate_map50=0.205,
        candidate_size_bytes=int(3_220_840 * 0.95),
        candidate_arena_used_bytes=int(692_816 * 0.95),
        candidate_predicted_p4_latency_ms_p50=5_883.36 * 0.95,
        baseline_story="US-006-int8",
        baseline_map50=0.205,
        baseline_size_bytes=3_220_840,
        baseline_arena_used_bytes=692_816,
        baseline_predicted_p4_latency_ms_p50=5_883.36,
    )
    assert v["verdict"] == "equal"


# ---------------------------------------------------------------------------
# Test 7: build_pareto_verdict — regress branch (mAP dropped).
# ---------------------------------------------------------------------------
def test_pareto_regress_when_map_drops() -> None:
    v = pcq.build_pareto_verdict(
        candidate_map50=0.10,
        candidate_size_bytes=100_000,  # tiny; would be dominate-worthy
        candidate_arena_used_bytes=100_000,
        candidate_predicted_p4_latency_ms_p50=100.0,
        baseline_story="US-006-int8",
        baseline_map50=0.205,
        baseline_size_bytes=3_220_840,
        baseline_arena_used_bytes=692_816,
        baseline_predicted_p4_latency_ms_p50=5_883.36,
    )
    assert v["verdict"] == "regress"
    assert v["deltas"]["map50"] < 0


# ---------------------------------------------------------------------------
# Test 8: build_pareto_verdict — blocked branch.
# ---------------------------------------------------------------------------
def test_pareto_blocked_short_circuits_with_reason() -> None:
    v = pcq.build_pareto_verdict(
        candidate_map50=0.0,
        candidate_size_bytes=0,
        candidate_arena_used_bytes=0,
        candidate_predicted_p4_latency_ms_p50=0.0,
        baseline_story="US-006-int8",
        baseline_map50=0.205,
        baseline_size_bytes=3_220_840,
        baseline_arena_used_bytes=692_816,
        baseline_predicted_p4_latency_ms_p50=5_883.36,
        blocked_reason="onnx2tf raised NHWC layout error on Detect head",
    )
    assert v["verdict"] == "blocked"
    assert "NHWC" in v["blocked_reason"]
    assert v["deltas"] is None


# ---------------------------------------------------------------------------
# Test 9: representative_dataset reaches the converter when going through the
# full DI chain — important guard so a future refactor doesn't accidentally
# decouple calibration from quantization.
# ---------------------------------------------------------------------------
def test_quantize_receives_callable_representative_dataset(tmp_path: Path) -> None:
    received: dict = {}

    def quantizer_stub(saved_dir, rep_ds, out_path):
        received["rep_ds_callable"] = callable(rep_ds)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"OUT")
        return out_path

    pt = tmp_path / "fake.pt"
    pt.write_bytes(b"")
    work = tmp_path / "work"
    calib = tmp_path / "calib"
    calib.mkdir()
    (calib / "frame_0001.jpg").write_bytes(b"")

    pcq.export_per_channel_int8(
        pt_path=pt,
        calib_dir=calib,
        out_path=tmp_path / "out.tflite",
        imgsz=224,
        max_calib_frames=200,
        work_dir=work,
        onnx_exporter=MagicMock(return_value=work / "fake.onnx"),
        savedmodel_converter=MagicMock(return_value=work / "saved_model"),
        quantizer=quantizer_stub,
    )
    assert received["rep_ds_callable"] is True
