"""Tests for training/edge/yolo/qat_finetune.py — both decision branches.

Mirrors the DI pattern used by test_quantize.py: heavy steps (ultralytics
training, TF/onnx2tf export, eval harness) are swapped via kwargs on the
public ``orchestrate`` so the suite runs in <1 s without TF/torch installed.

The parametrized tests assert the conditional contract from US-005:
  - mAP_drop <= 3.0 points (in fraction terms 0.02 == 2 mAP-points) -> SKIP
  - mAP_drop >  3.0 points (0.05 == 5 mAP-points)                   -> QAT
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from training.edge.yolo import qat_finetune as qf


# --- helpers ---------------------------------------------------------------


def _write_eval_json(path: Path, story_id: str, map50: float) -> None:
    """Write a minimal EvalResult-shaped JSON for decide()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "story_id": story_id,
        "model_path": f"training/edge/models/{story_id}.pt",
        "model_format": "pytorch" if story_id == "US-003" else "tflite_int8",
        "map50": map50,
        "size_bytes": 1234,
        "params": 1000,
        "flops": 0,
        "input_hw": [224, 224],
        "latency_ms_p50": 1.0,
        "latency_ms_p95": 2.0,
        "val_images": 100,
        "notes": "fixture",
    }
    path.write_text(json.dumps(payload))


# --- decide() pure logic ---------------------------------------------------


@pytest.mark.parametrize(
    "us003_map50,us004_map50,expect_run_qat,expect_drop_points",
    [
        # mAP_drop = 2 points -> skip (<=3.0 threshold)
        (0.92, 0.90, False, 2.0),
        # mAP_drop = 2.5 points -> skip (well under threshold)
        (0.925, 0.90, False, 2.5),
        # mAP_drop = 5 points -> QAT
        (0.95, 0.90, True, 5.0),
        # mAP_drop = 90.9 points (the actual US-004 result) -> QAT
        (0.909, 0.000, True, 90.9),
    ],
)
def test_decide_branch_matches_threshold(
    tmp_path: Path,
    us003_map50: float,
    us004_map50: float,
    expect_run_qat: bool,
    expect_drop_points: float,
) -> None:
    """decide() reads both JSONs, computes mAP_drop in points, picks branch."""
    us003 = tmp_path / "US-003.json"
    us004 = tmp_path / "US-004.json"
    _write_eval_json(us003, "US-003", us003_map50)
    _write_eval_json(us004, "US-004", us004_map50)

    decision = qf.decide(us003, us004, threshold_points=3.0)

    assert decision.us003_map50 == pytest.approx(us003_map50)
    assert decision.us004_map50 == pytest.approx(us004_map50)
    assert decision.map_drop_points == pytest.approx(expect_drop_points, abs=1e-6)
    assert decision.run_qat is expect_run_qat
    assert decision.threshold_points == 3.0


def test_decide_missing_us004_raises(tmp_path: Path) -> None:
    """decide() must hard-error if a prior result file is missing — the
    pipeline cannot infer a default and silent-falling-through to skip would
    mask a real US-004 failure."""
    us003 = tmp_path / "US-003.json"
    _write_eval_json(us003, "US-003", 0.9)
    with pytest.raises(FileNotFoundError):
        qf.decide(us003, tmp_path / "missing-us004.json")


# --- write_skipped_result() schema ----------------------------------------


def test_write_skipped_result_has_status_skipped(tmp_path: Path) -> None:
    """The skipped JSON must include status='skipped' so US-012's aggregator
    can branch on it; carry forward the PTQ map50 so SUMMARY rows remain
    meaningful even when QAT was bypassed."""
    decision = qf.Decision(
        map_drop_points=2.0,
        threshold_points=3.0,
        run_qat=False,
        us003_map50=0.92,
        us004_map50=0.90,
    )
    out = tmp_path / "US-005.json"
    qf.write_skipped_result(decision, out)
    payload = json.loads(out.read_text())

    assert payload["status"] == "skipped"
    assert payload["story_id"] == "US-005"
    assert payload["map50"] == pytest.approx(0.90)  # carries forward US-004
    assert payload["map_drop_points"] == pytest.approx(2.0)
    assert payload["threshold_points"] == 3.0
    assert "skipped" in payload["notes"].lower()


# --- orchestrate() branches via DI hooks -----------------------------------


def test_orchestrate_skip_branch_writes_skipped_json_and_no_train(
    tmp_path: Path,
) -> None:
    """Given mAP_drop = 2.0 points the orchestrator MUST NOT call qat_fn /
    export_fn / eval_fn — only write the skipped JSON."""
    us003 = tmp_path / "US-003.json"
    us004 = tmp_path / "US-004.json"
    _write_eval_json(us003, "US-003", 0.92)
    _write_eval_json(us004, "US-004", 0.90)

    qat_fn = MagicMock()
    export_fn = MagicMock()
    eval_fn = MagicMock()

    summary = qf.orchestrate(
        us003_path=us003,
        us004_path=us004,
        results_dir=tmp_path / "results",
        qat_fn=qat_fn,
        export_fn=export_fn,
        eval_fn=eval_fn,
    )

    qat_fn.assert_not_called()
    export_fn.assert_not_called()
    eval_fn.assert_not_called()

    assert summary["branch"] == "skipped"
    out_json = tmp_path / "results" / "US-005.json"
    assert out_json.exists()
    payload = json.loads(out_json.read_text())
    assert payload["status"] == "skipped"


def test_orchestrate_qat_branch_invokes_train_export_and_eval(
    tmp_path: Path,
) -> None:
    """Given mAP_drop = 5.0 points the orchestrator MUST invoke the train,
    export, and eval steps — verifies the conditional path in full."""
    us003 = tmp_path / "US-003.json"
    us004 = tmp_path / "US-004.json"
    _write_eval_json(us003, "US-003", 0.95)
    _write_eval_json(us004, "US-004", 0.90)

    # Mock training: pretend a best.pt was produced at runs_dir/<name>/weights/best.pt
    runs_dir = tmp_path / "runs"
    pretend_best = runs_dir / "yolov8n_cat_qat" / "weights" / "best.pt"
    pretend_best.parent.mkdir(parents=True, exist_ok=True)
    pretend_best.write_bytes(b"fake-qat-weights")

    qat_fn = MagicMock(
        return_value=(
            pretend_best,
            {"fake_quant_inserted": True, "fallback_reason": ""},
        )
    )

    out_tflite = tmp_path / "yolov8n_cat_qat_int8.tflite"

    def _export(pt_path, calib_dir, out_path, imgsz, max_calib_frames=200, work_dir=None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake-qat-int8-tflite")
        return out_path

    export_fn = MagicMock(side_effect=_export)

    # Mock eval to return an object with a write(path) method
    fake_eval_result = MagicMock()

    def _write(p):
        Path(p).write_text(json.dumps({"story_id": "US-005", "map50": 0.93}))

    fake_eval_result.write = MagicMock(side_effect=_write)
    eval_fn = MagicMock(return_value=fake_eval_result)

    # Use a calib dir that exists so the orchestrator doesn't trip on Path checks
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()

    summary = qf.orchestrate(
        us003_path=us003,
        us004_path=us004,
        base_pt=tmp_path / "fake_base.pt",
        data_yaml=tmp_path / "data.yaml",
        runs_dir=runs_dir,
        out_tflite=out_tflite,
        results_dir=tmp_path / "results",
        val_dir=tmp_path / "val",
        calib_dir=calib_dir,
        qat_fn=qat_fn,
        export_fn=export_fn,
        eval_fn=eval_fn,
    )

    qat_fn.assert_called_once()
    export_fn.assert_called_once()
    eval_fn.assert_called_once()

    # Eval was called with the QAT-branch story id and tflite_int8 format
    eval_kwargs = eval_fn.call_args
    args = eval_kwargs.args
    assert args[1] == "tflite_int8"
    assert args[2] == "US-005"

    assert summary["branch"] == "qat"
    assert summary["qat_info"]["fake_quant_inserted"] is True

    # The eval result was written to results_dir/US-005.json
    out_json = tmp_path / "results" / "US-005.json"
    assert out_json.exists()
    payload = json.loads(out_json.read_text())
    assert payload["story_id"] == "US-005"


def test_orchestrate_qat_branch_with_failed_fake_quant_falls_through(
    tmp_path: Path,
) -> None:
    """If torch.ao.quantization.prepare_qat refused to insert observers
    (custom op the tracer can't handle), the QAT branch must still complete
    with fake_quant_inserted=False — the failure is recorded, not fatal."""
    us003 = tmp_path / "US-003.json"
    us004 = tmp_path / "US-004.json"
    _write_eval_json(us003, "US-003", 0.95)
    _write_eval_json(us004, "US-004", 0.90)

    runs_dir = tmp_path / "runs"
    pretend_best = runs_dir / "yolov8n_cat_qat" / "weights" / "best.pt"
    pretend_best.parent.mkdir(parents=True, exist_ok=True)
    pretend_best.write_bytes(b"fake-qat-weights")

    qat_fn = MagicMock(
        return_value=(
            pretend_best,
            {
                "fake_quant_inserted": False,
                "fallback_reason": "RuntimeError: tracer cannot handle DetectHead",
            },
        )
    )

    def _export(pt_path, calib_dir, out_path, imgsz, max_calib_frames=200, work_dir=None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake-tflite")
        return out_path

    export_fn = MagicMock(side_effect=_export)

    fake_eval_result = MagicMock()
    fake_eval_result.write = MagicMock(side_effect=lambda p: Path(p).write_text("{}"))
    eval_fn = MagicMock(return_value=fake_eval_result)

    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()

    summary = qf.orchestrate(
        us003_path=us003,
        us004_path=us004,
        base_pt=tmp_path / "fake_base.pt",
        data_yaml=tmp_path / "data.yaml",
        runs_dir=runs_dir,
        out_tflite=tmp_path / "out.tflite",
        results_dir=tmp_path / "results",
        val_dir=tmp_path / "val",
        calib_dir=calib_dir,
        qat_fn=qat_fn,
        export_fn=export_fn,
        eval_fn=eval_fn,
    )

    assert summary["branch"] == "qat"
    assert summary["qat_info"]["fake_quant_inserted"] is False
    assert "tracer" in summary["qat_info"]["fallback_reason"]


# --- run_eval_on_qat() writes the canonical US-005.json -------------------


def test_run_eval_on_qat_writes_us005_json(tmp_path: Path) -> None:
    """run_eval_on_qat must end in results_dir/US-005.json regardless of
    where the underlying eval_fn put its first artifact."""
    fake_result = MagicMock()
    fake_result.write = MagicMock(
        side_effect=lambda p: Path(p).write_text(
            json.dumps({"story_id": "US-005", "map50": 0.45})
        )
    )
    fake_eval_fn = MagicMock(return_value=fake_result)

    out = qf.run_eval_on_qat(
        tflite_path=tmp_path / "qat.tflite",
        val_dir=tmp_path / "val",
        results_dir=tmp_path / "results",
        notes="test",
        eval_fn=fake_eval_fn,
    )

    fake_eval_fn.assert_called_once()
    args = fake_eval_fn.call_args.args
    assert args[1] == "tflite_int8"
    assert args[2] == "US-005"
    assert out == tmp_path / "results" / "US-005.json"
    assert out.exists()
