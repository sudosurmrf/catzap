"""iter-B: tests for the imgsz sweep orchestrator.

All inner steps (quantize / eval / bench / baseline_loader) go through DI
seams so this test runs without TF, ultralytics, cv2, or the edgebench
binary. Each test runs in <100 ms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from training.edge.yolo import imgsz_sweep as ibs

# aggregate_iter_b.py lives in training/edge/results/ which isn't a package
# (parallel to aggregate_summary.py — see test_summary_aggregate.py for the
# same pattern). Put the dir on sys.path before importing.
_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
if str(_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(_RESULTS_DIR))

import aggregate_iter_b  # noqa: E402


def _baseline_aggregate_dict() -> dict:
    """Mimic the iter-A_per_channel_quant.json shape the sweep ingests."""
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


def _make_eval_dict(*, story_id: str, imgsz: int, map50: float, size_bytes: int) -> dict:
    """Mimic the EvalResult JSON shape the sweep stamps into the rollup."""
    return {
        "story_id": story_id,
        "model_path": f"training/edge/models/sweep_imgsz/yolov8n_cat_distilled_int8_pc_{imgsz}.tflite",
        "model_format": "tflite_int8",
        "map50": map50,
        "size_bytes": size_bytes,
        "params": 7_881_671,
        "flops": 0,
        "input_hw": [imgsz, imgsz],
        "latency_ms_p50": 12.0,
        "latency_ms_p95": 15.0,
        "val_images": 120,
        "notes": "",
    }


def _make_tflm_dict(
    *, story_id: str, raw_x86_us_p50: float, arena_used_bytes: int, model_size_bytes: int
) -> dict:
    """Mimic firmware/edge-bench/run_bench.py:report_to_result output."""
    return {
        "story_id": story_id,
        "model_size_bytes": model_size_bytes,
        "runs": 50,
        "arena_size_bytes": 8_388_608,
        "arena_used_bytes": arena_used_bytes,
        "input_bytes": 150_528,
        "output_bytes": 20_580,
        "schema_status": "ok",
        "allocate_tensors_status": "ok",
        "timed_invoke_status": "ok",
        "profiler_overflowed": False,
        "tflm_compatible": True,
        "op_breakdown": [
            {"op_name": "CONV_2D", "count": 64, "total_us": int(raw_x86_us_p50 * 0.9), "percent": 90.0},
            {"op_name": "LOGISTIC", "count": 58, "total_us": int(raw_x86_us_p50 * 0.07), "percent": 7.0},
        ],
        "raw_x86_us_p50": raw_x86_us_p50,
        "raw_x86_us_p95": raw_x86_us_p50 * 1.05,
        "raw_x86_us_min": raw_x86_us_p50 * 0.95,
        "raw_x86_us_max": raw_x86_us_p50 * 1.10,
        "raw_x86_us_mean": raw_x86_us_p50,
        "x86_to_s3_multiplier": 8.0,
        "predicted_s3_us_p50": raw_x86_us_p50 * 8.0,
        "predicted_s3_us_p95": raw_x86_us_p50 * 8.0 * 1.05,
        "predicted_s3_fps": 1_000_000.0 / (raw_x86_us_p50 * 8.0),
        "binary": "fake/edgebench",
        "tflm_commit": "deadbeef",
    }


# ---------------------------------------------------------------------------
# Test 1: the sweep iterates the full imgsz list, calls inner export once per
# imgsz with the right kwargs, and writes one rollup JSON per imgsz with the
# expected file names.
# ---------------------------------------------------------------------------
def test_sweep_iterates_imgsz_list_and_writes_one_json_per_value(tmp_path: Path) -> None:
    pt_path = tmp_path / "fake.pt"
    pt_path.write_bytes(b"")
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    out_dir = tmp_path / "models"
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    results_dir = tmp_path / "results"

    quantize_calls: list[dict] = []
    eval_calls: list[dict] = []
    bench_calls: list[dict] = []

    def fake_quantize(*, pt_path, calib_dir, out_path, imgsz):
        quantize_calls.append({"pt_path": pt_path, "out_path": out_path, "imgsz": imgsz})
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"FAKE_TFLITE")
        return out_path

    def fake_eval(*, model_path, story_id, val_dir, imgsz, results_dir):
        eval_calls.append({"model_path": model_path, "imgsz": imgsz, "story_id": story_id})
        # Same mAP across all imgsz to keep verdict at "equal".
        return _make_eval_dict(
            story_id=story_id,
            imgsz=imgsz,
            map50=0.2155,
            size_bytes=Path(model_path).stat().st_size,
        )

    def fake_bench(*, model_path, story_id, results_dir, runs):
        bench_calls.append({"model_path": model_path, "story_id": story_id, "runs": runs})
        # Latency scales roughly with imgsz^2 — pick numbers that stay within
        # 15% so verdict stays "equal" across the sweep in this fixture.
        # imgsz 224 = 1_164_102 us per iter-A; pretend 192/224/256/288/320
        # all clock 1_164_102 us so we exercise the equal branch.
        return _make_tflm_dict(
            story_id=story_id,
            raw_x86_us_p50=1_164_102.0,
            arena_used_bytes=692_912,
            model_size_bytes=Path(model_path).stat().st_size,
        )

    aggregates = ibs.sweep(
        pt_path=pt_path,
        calib_dir=calib_dir,
        out_dir=out_dir,
        val_dir=val_dir,
        results_dir=results_dir,
        imgsz_list=[192, 224, 256, 288, 320],
        runs=50,
        quantize_fn=fake_quantize,
        eval_fn=fake_eval,
        bench_fn=fake_bench,
        baseline_loader=_baseline_aggregate_dict,
    )

    # Quantize / eval / bench each called exactly 5 times in order.
    assert [c["imgsz"] for c in quantize_calls] == [192, 224, 256, 288, 320]
    assert len(eval_calls) == 5
    assert len(bench_calls) == 5

    # One rollup JSON per imgsz was written.
    expected_files = [
        results_dir / f"iter-B_imgsz_{n}.json" for n in (192, 224, 256, 288, 320)
    ]
    for f in expected_files:
        assert f.exists(), f"missing rollup at {f}"

    # All rollups passed (no blocked).
    assert all(a["status"] == "passed" for a in aggregates)
    # Each rollup has the iter-A_per_channel_quant.json schema keys.
    for agg in aggregates:
        for required in (
            "story_id",
            "imgsz",
            "title",
            "status",
            "blocked_reason",
            "model_path",
            "p4_multiplier",
            "p4_multiplier_source",
            "eval",
            "tflm",
            "pareto",
            "candidate_metrics",
            "baseline_metrics",
        ):
            assert required in agg, f"rollup missing key {required}: {list(agg)}"


# ---------------------------------------------------------------------------
# Test 2: a quantize failure at one imgsz is recorded as status="blocked"
# with a concrete reason, and the sweep still produces JSONs for the rest.
# This is the per-imgsz failure-isolation contract from the AC.
# ---------------------------------------------------------------------------
def test_sweep_continues_past_a_single_quantize_failure(tmp_path: Path) -> None:
    pt_path = tmp_path / "fake.pt"
    pt_path.write_bytes(b"")
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    out_dir = tmp_path / "models"
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    results_dir = tmp_path / "results"

    def flaky_quantize(*, pt_path, calib_dir, out_path, imgsz):
        if imgsz == 256:
            raise RuntimeError("simulated onnx2tf NHWC layout error at imgsz=256")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"FAKE")
        return out_path

    def fake_eval(*, model_path, story_id, val_dir, imgsz, results_dir):
        return _make_eval_dict(
            story_id=story_id, imgsz=imgsz, map50=0.2155, size_bytes=4
        )

    def fake_bench(*, model_path, story_id, results_dir, runs):
        return _make_tflm_dict(
            story_id=story_id,
            raw_x86_us_p50=1_000_000.0,
            arena_used_bytes=600_000,
            model_size_bytes=4,
        )

    aggregates = ibs.sweep(
        pt_path=pt_path,
        calib_dir=calib_dir,
        out_dir=out_dir,
        val_dir=val_dir,
        results_dir=results_dir,
        imgsz_list=[192, 224, 256, 288, 320],
        quantize_fn=flaky_quantize,
        eval_fn=fake_eval,
        bench_fn=fake_bench,
        baseline_loader=_baseline_aggregate_dict,
    )

    # One per imgsz, all written to disk.
    by_imgsz = {a["imgsz"]: a for a in aggregates}
    assert set(by_imgsz) == {192, 224, 256, 288, 320}
    for imgsz in (192, 224, 256, 288, 320):
        assert (results_dir / f"iter-B_imgsz_{imgsz}.json").exists()

    # The 256 row is blocked; others passed.
    assert by_imgsz[256]["status"] == "blocked"
    assert by_imgsz[256]["pareto"]["verdict"] == "blocked"
    assert "256" in by_imgsz[256]["blocked_reason"]
    assert "onnx2tf" in by_imgsz[256]["blocked_reason"]
    for other in (192, 224, 288, 320):
        assert by_imgsz[other]["status"] == "passed"
        assert by_imgsz[other]["pareto"]["verdict"] in {"dominates", "equal"}


# ---------------------------------------------------------------------------
# Test 3: the Pareto verdict is computed against iter-A's candidate_metrics
# (NOT against US-006-int8). A candidate that dominates iter-A on size by
# >=20% must come back as verdict="dominates".
# ---------------------------------------------------------------------------
def test_sweep_uses_iter_a_candidate_metrics_as_baseline(tmp_path: Path) -> None:
    pt_path = tmp_path / "fake.pt"
    pt_path.write_bytes(b"")
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    out_dir = tmp_path / "models"
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    results_dir = tmp_path / "results"

    iter_a_size = 3_221_000
    iter_a_arena = 692_912
    iter_a_x86_p50 = 1_164_102.0  # gives baseline P4 ms = 5820.51 at mult=5.0

    def quant(*, pt_path, calib_dir, out_path, imgsz):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"X")
        return out_path

    def fake_eval(*, model_path, story_id, val_dir, imgsz, results_dir):
        # Hold mAP at iter-A's number.
        # 30%-smaller "size" — should trip the dominates threshold.
        return _make_eval_dict(
            story_id=story_id,
            imgsz=imgsz,
            map50=0.2155,
            size_bytes=int(iter_a_size * 0.7),
        )

    def fake_bench(*, model_path, story_id, results_dir, runs):
        return _make_tflm_dict(
            story_id=story_id,
            raw_x86_us_p50=iter_a_x86_p50,
            arena_used_bytes=iter_a_arena,
            model_size_bytes=int(iter_a_size * 0.7),
        )

    aggregates = ibs.sweep(
        pt_path=pt_path,
        calib_dir=calib_dir,
        out_dir=out_dir,
        val_dir=val_dir,
        results_dir=results_dir,
        imgsz_list=[224],  # one row is enough for this assertion
        quantize_fn=quant,
        eval_fn=fake_eval,
        bench_fn=fake_bench,
        baseline_loader=_baseline_aggregate_dict,
    )

    [agg] = aggregates
    p = agg["pareto"]
    assert p["verdict"] == "dominates", agg
    # The baseline_story field carries the iter-A name so the verdict is
    # auditable downstream (per build_pareto_verdict's contract).
    assert p["baseline_story"] == ibs.BASELINE_STORY
    assert "iter-A" in p["baseline_story"]
    # Baseline numbers came from the aggregate's candidate_metrics, NOT from
    # iter-A's baseline_metrics field (which would point at US-006-int8).
    assert agg["baseline_metrics"]["map50"] == pytest.approx(0.2155)
    assert agg["baseline_metrics"]["size_bytes"] == iter_a_size
    assert p["deltas"]["size_pct"] < -20.0  # past the 20% size threshold


# ---------------------------------------------------------------------------
# Test 4: the rollup JSON's schema lines up with iter-A_per_channel_quant.json.
# Important so iter-H's SUMMARY_v2 aggregator can ingest both with one parser.
# ---------------------------------------------------------------------------
def test_rollup_schema_matches_iter_a_aggregate(tmp_path: Path) -> None:
    pt_path = tmp_path / "fake.pt"
    pt_path.write_bytes(b"")
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    out_dir = tmp_path / "models"
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    results_dir = tmp_path / "results"

    def quant(*, pt_path, calib_dir, out_path, imgsz):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"X")
        return out_path

    def fake_eval(*, model_path, story_id, val_dir, imgsz, results_dir):
        return _make_eval_dict(
            story_id=story_id, imgsz=imgsz, map50=0.2155, size_bytes=3_221_000
        )

    def fake_bench(*, model_path, story_id, results_dir, runs):
        return _make_tflm_dict(
            story_id=story_id,
            raw_x86_us_p50=1_164_102.0,
            arena_used_bytes=692_912,
            model_size_bytes=3_221_000,
        )

    aggregates = ibs.sweep(
        pt_path=pt_path,
        calib_dir=calib_dir,
        out_dir=out_dir,
        val_dir=val_dir,
        results_dir=results_dir,
        imgsz_list=[224],
        quantize_fn=quant,
        eval_fn=fake_eval,
        bench_fn=fake_bench,
        baseline_loader=_baseline_aggregate_dict,
    )

    [agg] = aggregates
    # Round-trip through JSON disk to catch any non-serializable values.
    on_disk = json.loads((results_dir / "iter-B_imgsz_224.json").read_text())
    assert on_disk == agg
    # Fields iter-H's aggregate_summary_v2 will read.
    assert isinstance(agg["candidate_metrics"]["map50"], float)
    assert isinstance(agg["candidate_metrics"]["size_bytes"], int)
    assert isinstance(agg["candidate_metrics"]["arena_used_bytes"], int)
    assert isinstance(agg["candidate_metrics"]["predicted_p4_latency_ms_p50"], float)
    # P4 multiplier is recorded so iter-H can re-do the math with new
    # boards-in-hand multipliers without re-running the bench.
    assert agg["p4_multiplier"] == ibs.DEFAULT_P4_MULTIPLIER
    assert "ESP32-S3" in agg["p4_multiplier_source"] or "P4" in agg["p4_multiplier_source"]


# ---------------------------------------------------------------------------
# Test 5: the CLI parses --imgsz-list as a comma-separated list.
# ---------------------------------------------------------------------------
def test_cli_parses_comma_separated_imgsz_list() -> None:
    parsed = ibs._parse_imgsz_list("192,224,256, 288 ,320")
    assert parsed == [192, 224, 256, 288, 320]
    assert ibs._parse_imgsz_list("224") == [224]
    assert ibs._parse_imgsz_list("") == []


# ---------------------------------------------------------------------------
# Test 6: a regress on mAP at one imgsz is recorded as verdict="regress",
# the row stays in the JSON (not silently dropped) so iter-H can flag it.
# ---------------------------------------------------------------------------
def test_sweep_records_regress_when_map_drops(tmp_path: Path) -> None:
    pt_path = tmp_path / "fake.pt"
    pt_path.write_bytes(b"")
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    out_dir = tmp_path / "models"
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    results_dir = tmp_path / "results"

    def quant(*, pt_path, calib_dir, out_path, imgsz):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"X")
        return out_path

    def fake_eval(*, model_path, story_id, val_dir, imgsz, results_dir):
        # imgsz=192 gets a big mAP drop — common when the model was trained at 224.
        map50 = 0.05 if imgsz == 192 else 0.2155
        return _make_eval_dict(
            story_id=story_id, imgsz=imgsz, map50=map50, size_bytes=3_221_000
        )

    def fake_bench(*, model_path, story_id, results_dir, runs):
        # Smaller imgsz is faster — but verdict should still be "regress" because
        # mAP dropped (regress takes precedence over efficiency wins per
        # build_pareto_verdict's contract).
        return _make_tflm_dict(
            story_id=story_id,
            raw_x86_us_p50=500_000.0,
            arena_used_bytes=400_000,
            model_size_bytes=3_221_000,
        )

    aggregates = ibs.sweep(
        pt_path=pt_path,
        calib_dir=calib_dir,
        out_dir=out_dir,
        val_dir=val_dir,
        results_dir=results_dir,
        imgsz_list=[192, 224],
        quantize_fn=quant,
        eval_fn=fake_eval,
        bench_fn=fake_bench,
        baseline_loader=_baseline_aggregate_dict,
    )

    by_imgsz = {a["imgsz"]: a for a in aggregates}
    assert by_imgsz[192]["pareto"]["verdict"] == "regress"
    assert by_imgsz[192]["pareto"]["deltas"]["map50"] < 0
    # Even on regress, status stays "passed" — only quantize/eval/bench errors
    # become status="blocked". This matches build_pareto_verdict's contract.
    assert by_imgsz[192]["status"] == "passed"


# ---------------------------------------------------------------------------
# Test 7: aggregate_iter_b.pick_winner — picks the smallest imgsz that holds
# mAP within 1 mAP-point AND wins >=15% on latency. Also verifies the
# eval/bench sidecar JSONs are filtered out of the rollup loader.
# ---------------------------------------------------------------------------
def test_aggregate_iter_b_picks_smallest_qualifying_imgsz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The aggregator reads from training/edge/results/, so monkeypatch its
    # RESULTS path to a temp dir.
    agg = aggregate_iter_b
    monkeypatch.setattr(agg, "RESULTS", tmp_path)

    def write(name: str, doc: dict) -> None:
        (tmp_path / name).write_text(json.dumps(doc))

    baseline = {
        "map50": 0.2155,
        "size_bytes": 3_221_000,
        "arena_used_bytes": 692_912,
        "predicted_p4_latency_ms_p50": 5_820.51,
    }

    # Three rollups: 192 regress (mAP drop > 1pt), 256 holds mAP + 25% latency win
    # (qualifies), 288 also holds mAP + 30% latency win (also qualifies but
    # bigger). Winner = 256 (smallest qualifying).
    for imgsz, map50, latency_pct, verdict in (
        (192, 0.05, -27.07, "regress"),
        (256, 0.2155, -25.0, "dominates"),
        (288, 0.2160, -30.0, "dominates"),
    ):
        rollup = {
            "story_id": f"iter-B_imgsz_{imgsz}",
            "imgsz": imgsz,
            "status": "passed",
            "candidate_metrics": {
                "map50": map50,
                "size_bytes": 3_221_000,
                "arena_used_bytes": 692_912,
                "predicted_p4_latency_ms_p50": 5_820.51 * (1.0 + latency_pct / 100.0),
                "predicted_p4_fps": 0.2,
            },
            "baseline_metrics": baseline,
            "pareto": {
                "verdict": verdict,
                "deltas": {
                    "map50": map50 - baseline["map50"],
                    "size_pct": 0.0,
                    "arena_pct": 0.0,
                    "latency_pct": latency_pct,
                },
            },
            "model_path": f"training/edge/models/sweep_imgsz/foo_{imgsz}.tflite",
        }
        write(f"iter-B_imgsz_{imgsz}.json", rollup)
        # Also drop a sidecar (eval) to verify it's filtered out.
        write(f"iter-B_imgsz_{imgsz}_eval.json", {"map50": 0.0, "model_path": ""})
        write(
            f"iter-B_imgsz_{imgsz}_bench-tflm.json",
            {"arena_used_bytes": 100, "raw_x86_us_p50": 1.0},
        )

    rollups = agg.load_rollups()
    # Sidecars must NOT be loaded as rollups — only 3 entries.
    assert [r["imgsz"] for r in rollups] == [192, 256, 288]

    winner = agg.pick_winner(rollups)
    assert winner is not None
    assert winner["imgsz"] == 256, "smallest qualifying imgsz should win"

    md = agg.format_summary(rollups)
    assert "imgsz=256" in md
    assert "iter-B" in md
    assert "Pareto" in md or "verdict" in md
    # Regress row stays in the table — failure is a finding, never silently drop.
    assert "192" in md and "regress" in md.lower()
