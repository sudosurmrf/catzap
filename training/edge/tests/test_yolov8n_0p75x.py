"""iter-E: tests for the YOLOv8n width=0.75x retrain + KD distill + PTQ pipeline.

All heavy steps (yolo_factory / distill_train / quantize / eval / bench) go
through DI seams so this file does NOT need ultralytics, torch, TF, cv2, or
the edgebench binary. Each test runs in <100 ms.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from training.edge.yolo import train_yolov8n_0p75x_cat as iter_e


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _baseline_aggregate_dict() -> dict:
    """Mimic the iter-A_per_channel_quant.json shape that iter-E ingests."""
    return {
        "story_id": "iter-A",
        "candidate_metrics": {
            "map50": 0.2155,
            "size_bytes": 3_221_000,
            "arena_used_bytes": 692_912,
            "predicted_p4_latency_ms_p50": 5_820.51,
            "predicted_p4_fps": 0.1718,
        },
    }


def _make_eval_dict(*, story_id: str, model_path: Path, map50: float, imgsz: int) -> dict:
    return {
        "story_id": story_id,
        "model_path": str(model_path),
        "model_format": "tflite_int8",
        "map50": map50,
        "size_bytes": int(Path(model_path).stat().st_size),
        "params": 1_700_000,
        "flops": 0,
        "input_hw": [imgsz, imgsz],
        "latency_ms_p50": 9.0,
        "latency_ms_p95": 11.5,
        "val_images": 120,
        "notes": "",
    }


def _make_tflm_dict(
    *,
    story_id: str,
    model_path: Path,
    raw_x86_us_p50: float,
    arena_used_bytes: int,
) -> dict:
    return {
        "story_id": story_id,
        "model_path": str(model_path),
        "model_size_bytes": int(Path(model_path).stat().st_size),
        "runs": 50,
        "arena_size_bytes": 8_388_608,
        "arena_used_bytes": arena_used_bytes,
        "tflm_compatible": True,
        "op_breakdown": [],
        "raw_x86_us_p50": raw_x86_us_p50,
        "raw_x86_us_p95": raw_x86_us_p50 * 1.05,
    }


def _baseline_yaml_text() -> str:
    return (
        "nc: 1\n"
        "depth_multiple: 0.33\n"
        "width_multiple: 0.1875\n"
        "backbone: []\n"
        "head: []\n"
    )


# ---------------------------------------------------------------------------
# REQUIRED test 1 (per AC): width_multiple=0.75 reaches the trainer kwargs.
# ---------------------------------------------------------------------------


def test_pipeline_passes_width_multiple_0p75_to_trainer(tmp_path: Path) -> None:
    """The user-facing knob ``width_multiple=0.75`` must arrive intact at
    the inner ``trainer_fn`` kwargs. This is the load-bearing assertion the
    iter-E acceptance criteria call out: "asserting width_multiple=0.75
    reaches the trainer kwargs"."""
    config_yaml = tmp_path / "yolov8n_0p75x.yaml"
    config_yaml.write_text(_baseline_yaml_text())
    fp32_out = tmp_path / "out.pt"
    int8_out = tmp_path / "out.tflite"
    results_dir = tmp_path / "results"
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()

    captured_kwargs: dict = {}

    def fake_trainer(**kwargs):
        captured_kwargs.update(kwargs)
        # Honor the (best_pt, init_summary) return contract.
        best_pt = tmp_path / "best.pt"
        best_pt.write_bytes(b"FAKE_BEST")
        init_summary = {
            "base_pt": "yolov8n.pt",
            "matched_keys_total": 200,
            "loaded_keys": 200,
            "skipped_missing_in_student": 0,
            "skipped_mismatched_shape": 0,
            "fallback_random_init": False,
        }
        return best_pt, init_summary

    def fake_quantize(*, pt_path, calib_dir, out_path, imgsz):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"X" * 2_000_000)
        return out_path

    def fake_eval(*, model_path, story_id, val_dir, imgsz, results_dir):
        return _make_eval_dict(
            story_id=story_id, model_path=Path(model_path), map50=0.20, imgsz=imgsz
        )

    def fake_bench(*, model_path, story_id, results_dir, runs):
        return _make_tflm_dict(
            story_id=story_id,
            model_path=Path(model_path),
            raw_x86_us_p50=900_000.0,
            arena_used_bytes=520_000,
        )

    doc = iter_e.run_pipeline(
        teacher_pt=tmp_path / "teacher.pt",
        base_pt=tmp_path / "base.pt",
        config_yaml=config_yaml,
        data_yaml=tmp_path / "data.yaml",
        fp32_out=fp32_out,
        int8_out=int8_out,
        calib_dir=calib_dir,
        val_dir=val_dir,
        results_dir=results_dir,
        runs_dir=tmp_path / "runs",
        width_multiple=0.75,
        depth_multiple=1.0,
        imgsz=224,
        trainer_fn=fake_trainer,
        quantize_fn=fake_quantize,
        eval_fn=fake_eval,
        bench_fn=fake_bench,
        baseline_loader=_baseline_aggregate_dict,
    )

    # The user-facing 0.75 reaches trainer_fn unchanged.
    assert captured_kwargs.get("width_multiple") == 0.75
    assert captured_kwargs.get("depth_multiple") == 1.0
    # Other AC kwargs threaded through as well.
    assert captured_kwargs.get("imgsz") == 224
    assert captured_kwargs.get("batch") == iter_e.DEFAULT_BATCH
    assert captured_kwargs.get("lr") == iter_e.DEFAULT_LR
    assert captured_kwargs.get("alpha") == iter_e.DEFAULT_ALPHA
    # The rollup also surfaces the user-facing value (audit trail).
    assert doc["width_multiple"] == 0.75
    assert doc["depth_multiple"] == 1.0


# ---------------------------------------------------------------------------
# REQUIRED test 2 (per AC): the distill+quant pipeline IS called end-to-end.
# ---------------------------------------------------------------------------


def test_pipeline_calls_distill_then_quantize_then_eval_then_bench(tmp_path: Path) -> None:
    """run_pipeline must orchestrate trainer -> quantize -> eval -> bench in
    that order. Trainer wraps the KD distill step so this exercise
    confirms the full chain reaches every stage when each succeeds."""
    config_yaml = tmp_path / "y.yaml"
    config_yaml.write_text(_baseline_yaml_text())

    call_order: list[str] = []

    def fake_trainer(**kwargs):
        call_order.append("trainer")
        best_pt = tmp_path / "best.pt"
        best_pt.write_bytes(b"P")
        return best_pt, {
            "matched_keys_total": 200,
            "loaded_keys": 195,
            "skipped_missing_in_student": 0,
            "skipped_mismatched_shape": 5,
            "fallback_random_init": False,
        }

    def fake_quantize(*, pt_path, calib_dir, out_path, imgsz):
        call_order.append("quantize")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"X" * 1_800_000)
        # The .pt MUST already exist when quantize is called — load-bearing.
        assert Path(pt_path).exists(), "quantize ran before trainer wrote .pt"
        return out_path

    def fake_eval(*, model_path, story_id, val_dir, imgsz, results_dir):
        call_order.append("eval")
        return _make_eval_dict(
            story_id=story_id, model_path=Path(model_path), map50=0.22, imgsz=imgsz
        )

    def fake_bench(*, model_path, story_id, results_dir, runs):
        call_order.append("bench")
        return _make_tflm_dict(
            story_id=story_id,
            model_path=Path(model_path),
            raw_x86_us_p50=850_000.0,
            arena_used_bytes=500_000,
        )

    doc = iter_e.run_pipeline(
        teacher_pt=tmp_path / "t.pt",
        base_pt=tmp_path / "b.pt",
        config_yaml=config_yaml,
        data_yaml=tmp_path / "data.yaml",
        fp32_out=tmp_path / "out.pt",
        int8_out=tmp_path / "out.tflite",
        calib_dir=tmp_path / "calib",
        val_dir=tmp_path / "val",
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        trainer_fn=fake_trainer,
        quantize_fn=fake_quantize,
        eval_fn=fake_eval,
        bench_fn=fake_bench,
        baseline_loader=_baseline_aggregate_dict,
    )

    # AC: distill+quant pipeline is called.
    assert call_order == ["trainer", "quantize", "eval", "bench"], call_order
    # The rollup file was written and is readable.
    out_json = tmp_path / "results" / "iter-E_yolov8n_0p75x.json"
    assert out_json.exists()
    on_disk = json.loads(out_json.read_text())
    assert on_disk["status"] == "passed"
    assert on_disk["pareto"]["verdict"] in {"dominates", "equal", "regress"}


# ---------------------------------------------------------------------------
# Test 3: regenerate_config composes user-facing 0.75 with yolov8n's 0.25 base.
# ---------------------------------------------------------------------------


def test_regenerate_config_composes_with_yolov8n_base(tmp_path: Path) -> None:
    """User-facing ``width_multiple=0.75`` (= "0.75x of yolov8n's width") is
    composed with yolov8n's 0.25 base to stamp the yaml's effective
    ``width_multiple = 0.1875``. depth follows the same composition rule."""
    cfg_path = tmp_path / "iter_e.yaml"
    cfg_path.write_text(_baseline_yaml_text())

    cfg = iter_e.regenerate_config(
        cfg_path,
        width_multiple=0.75,
        depth_multiple=1.0,
    )

    # Composed values.
    assert cfg["width_multiple"] == pytest.approx(0.25 * 0.75)
    assert cfg["depth_multiple"] == pytest.approx(0.33 * 1.0)
    assert cfg["nc"] == 1

    # The on-disk yaml round-trips so ultralytics will see the same numbers.
    import yaml

    on_disk = yaml.safe_load(cfg_path.read_text())
    assert on_disk["width_multiple"] == pytest.approx(0.1875)
    assert on_disk["depth_multiple"] == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# Test 4: a quantize failure becomes status="blocked" and JSON IS still written.
# ---------------------------------------------------------------------------


def test_pipeline_writes_blocked_json_when_quantize_fails(tmp_path: Path) -> None:
    config_yaml = tmp_path / "y.yaml"
    config_yaml.write_text(_baseline_yaml_text())

    def fake_trainer(**kwargs):
        best_pt = tmp_path / "best.pt"
        best_pt.write_bytes(b"P")
        return best_pt, {
            "matched_keys_total": 200,
            "loaded_keys": 200,
            "skipped_missing_in_student": 0,
            "skipped_mismatched_shape": 0,
            "fallback_random_init": False,
        }

    def flaky_quantize(*, pt_path, calib_dir, out_path, imgsz):
        raise RuntimeError("simulated onnx2tf NHWC layout error in iter-E")

    eval_mock = MagicMock()
    bench_mock = MagicMock()

    doc = iter_e.run_pipeline(
        teacher_pt=tmp_path / "t.pt",
        base_pt=tmp_path / "b.pt",
        config_yaml=config_yaml,
        data_yaml=tmp_path / "data.yaml",
        fp32_out=tmp_path / "out.pt",
        int8_out=tmp_path / "out.tflite",
        calib_dir=tmp_path / "calib",
        val_dir=tmp_path / "val",
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        trainer_fn=fake_trainer,
        quantize_fn=flaky_quantize,
        eval_fn=eval_mock,
        bench_fn=bench_mock,
        baseline_loader=_baseline_aggregate_dict,
    )

    # Per AC ("If retrain fails ... record status='blocked' with concrete
    # reason; emit JSON anyway"). Same isolation contract for quantize.
    out_json = tmp_path / "results" / "iter-E_yolov8n_0p75x.json"
    assert out_json.exists()
    assert doc["status"] == "blocked"
    assert doc["pareto"]["verdict"] == "blocked"
    assert "onnx2tf" in doc["blocked_reason"]
    eval_mock.assert_not_called()
    bench_mock.assert_not_called()
    # baseline_metrics still populated so iter-H can render the row.
    assert doc["baseline_metrics"]["map50"] == pytest.approx(0.2155)


# ---------------------------------------------------------------------------
# Test 5: a retrain failure ALSO writes JSON with status="blocked" (per AC).
# ---------------------------------------------------------------------------


def test_pipeline_writes_blocked_json_when_retrain_fails(tmp_path: Path) -> None:
    config_yaml = tmp_path / "y.yaml"
    config_yaml.write_text(_baseline_yaml_text())

    def flaky_trainer(**kwargs):
        raise RuntimeError("CUDA OOM at batch=32, imgsz=224, width=0.1875")

    quantize_mock = MagicMock()
    eval_mock = MagicMock()
    bench_mock = MagicMock()

    doc = iter_e.run_pipeline(
        teacher_pt=tmp_path / "t.pt",
        base_pt=tmp_path / "b.pt",
        config_yaml=config_yaml,
        data_yaml=tmp_path / "data.yaml",
        fp32_out=tmp_path / "out.pt",
        int8_out=tmp_path / "out.tflite",
        calib_dir=tmp_path / "calib",
        val_dir=tmp_path / "val",
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        trainer_fn=flaky_trainer,
        quantize_fn=quantize_mock,
        eval_fn=eval_mock,
        bench_fn=bench_mock,
        baseline_loader=_baseline_aggregate_dict,
    )

    quantize_mock.assert_not_called()
    eval_mock.assert_not_called()
    bench_mock.assert_not_called()
    assert doc["status"] == "blocked"
    assert "CUDA OOM" in doc["blocked_reason"]
    out_json = tmp_path / "results" / "iter-E_yolov8n_0p75x.json"
    assert out_json.exists()


# ---------------------------------------------------------------------------
# Test 6: the rollup JSON has the iter-H ingestion schema (mirrors iter-A/B/C).
# ---------------------------------------------------------------------------


def test_rollup_json_has_iter_h_ingestion_schema(tmp_path: Path) -> None:
    config_yaml = tmp_path / "y.yaml"
    config_yaml.write_text(_baseline_yaml_text())

    def fake_trainer(**kwargs):
        best_pt = tmp_path / "best.pt"
        best_pt.write_bytes(b"P")
        return best_pt, {
            "matched_keys_total": 200,
            "loaded_keys": 195,
            "skipped_missing_in_student": 0,
            "skipped_mismatched_shape": 5,
            "fallback_random_init": False,
        }

    def fake_quantize(*, pt_path, calib_dir, out_path, imgsz):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # 50% smaller than iter-A's 3,221,000 — fixture engineered to dominate.
        Path(out_path).write_bytes(b"X" * 1_600_000)
        return out_path

    def fake_eval(*, model_path, story_id, val_dir, imgsz, results_dir):
        return _make_eval_dict(
            story_id=story_id, model_path=Path(model_path), map50=0.2155, imgsz=imgsz
        )

    def fake_bench(*, model_path, story_id, results_dir, runs):
        # ~30% lower latency, ~20% smaller arena — should trip dominates.
        return _make_tflm_dict(
            story_id=story_id,
            model_path=Path(model_path),
            raw_x86_us_p50=820_000.0,
            arena_used_bytes=540_000,
        )

    doc = iter_e.run_pipeline(
        teacher_pt=tmp_path / "t.pt",
        base_pt=tmp_path / "b.pt",
        config_yaml=config_yaml,
        data_yaml=tmp_path / "data.yaml",
        fp32_out=tmp_path / "out.pt",
        int8_out=tmp_path / "out.tflite",
        calib_dir=tmp_path / "calib",
        val_dir=tmp_path / "val",
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        trainer_fn=fake_trainer,
        quantize_fn=fake_quantize,
        eval_fn=fake_eval,
        bench_fn=fake_bench,
        baseline_loader=_baseline_aggregate_dict,
    )

    # Required keys (mirrors iter-A_per_channel_quant.json schema).
    for required in (
        "story_id",
        "title",
        "status",
        "blocked_reason",
        "model_path",
        "p4_multiplier",
        "p4_multiplier_source",
        "width_multiple",  # iter-E-specific audit field
        "depth_multiple",
        "init_summary",
        "eval",
        "tflm",
        "pareto",
        "candidate_metrics",
        "baseline_metrics",
    ):
        assert required in doc, f"missing key {required}: {list(doc)}"

    # Sanity: this fixture dominates on size + latency + arena, mAP held.
    assert doc["pareto"]["verdict"] == "dominates"
    assert doc["candidate_metrics"]["predicted_p4_fps"] > 0.0
    # init_summary surfaces partial-load info so iter-H can flag random-init runs.
    assert doc["init_summary"]["matched_keys_total"] == 200


# ---------------------------------------------------------------------------
# Test 7: partial_load_yolov8n_weights falls back to random-init when the
# donor .pt is missing (per AC: "from-scratch if the partial load fails").
# ---------------------------------------------------------------------------


def test_partial_load_falls_back_to_random_init_when_donor_missing(tmp_path: Path) -> None:
    fake_student = MagicMock()
    yolo_factory = MagicMock()
    torch_module = MagicMock()

    summary = iter_e.partial_load_yolov8n_weights(
        fake_student,
        base_pt=tmp_path / "does_not_exist.pt",
        yolo_factory=yolo_factory,
        torch_module=torch_module,
    )

    assert summary["fallback_random_init"] is True
    assert "missing" in summary["error"]
    yolo_factory.assert_not_called()  # no donor load attempted


# ---------------------------------------------------------------------------
# Test 8: the Pareto verdict uses iter-A's candidate_metrics as the baseline.
# Confirms the same convention iter-B / iter-C use — iter-A is the v2 floor.
# ---------------------------------------------------------------------------


def test_pareto_verdict_uses_iter_a_candidate_metrics_as_baseline(tmp_path: Path) -> None:
    config_yaml = tmp_path / "y.yaml"
    config_yaml.write_text(_baseline_yaml_text())

    def fake_trainer(**kwargs):
        best_pt = tmp_path / "best.pt"
        best_pt.write_bytes(b"P")
        return best_pt, {
            "matched_keys_total": 100,
            "loaded_keys": 100,
            "skipped_missing_in_student": 0,
            "skipped_mismatched_shape": 0,
            "fallback_random_init": False,
        }

    def fake_quantize(*, pt_path, calib_dir, out_path, imgsz):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # Match iter-A's exact size — equal verdict expected (mAP held).
        Path(out_path).write_bytes(b"X" * 3_221_000)
        return out_path

    def fake_eval(*, model_path, story_id, val_dir, imgsz, results_dir):
        return _make_eval_dict(
            story_id=story_id, model_path=Path(model_path), map50=0.2155, imgsz=imgsz
        )

    def fake_bench(*, model_path, story_id, results_dir, runs):
        # Match iter-A's x86 latency (1_164_102 us) — should land verdict='equal'.
        return _make_tflm_dict(
            story_id=story_id,
            model_path=Path(model_path),
            raw_x86_us_p50=1_164_102.0,
            arena_used_bytes=692_912,
        )

    doc = iter_e.run_pipeline(
        teacher_pt=tmp_path / "t.pt",
        base_pt=tmp_path / "b.pt",
        config_yaml=config_yaml,
        data_yaml=tmp_path / "data.yaml",
        fp32_out=tmp_path / "out.pt",
        int8_out=tmp_path / "out.tflite",
        calib_dir=tmp_path / "calib",
        val_dir=tmp_path / "val",
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        trainer_fn=fake_trainer,
        quantize_fn=fake_quantize,
        eval_fn=fake_eval,
        bench_fn=fake_bench,
        baseline_loader=_baseline_aggregate_dict,
    )

    p = doc["pareto"]
    assert p["verdict"] == "equal"
    assert p["baseline_story"] == iter_e.BASELINE_STORY
    assert "iter-A" in p["baseline_story"]
    # baseline_metrics matches iter-A's candidate_metrics — NOT US-006-int8.
    assert doc["baseline_metrics"]["map50"] == pytest.approx(0.2155)
    assert doc["baseline_metrics"]["size_bytes"] == 3_221_000
