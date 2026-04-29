"""Tests for iter-H's aggregate_summary_v2.py.

Per iter-H acceptance: >=4 tests covering
  (a) frontier picker excludes blocked + zero-mAP rows
  (b) frontier output sorted by predicted_p4_fps desc
  (c) deprecation list includes v1 INT8 candidates the v2 frontier dominates
  (d) recommendation addresses MIPI-CSI vs HTTP-stream

Mirrors `test_summary_aggregate.py`'s import + fixture-builder pattern.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# aggregate_summary_v2 lives in training/edge/results/ which isn't a package.
_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
if str(_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(_RESULTS_DIR))

import aggregate_summary_v2 as agg  # noqa: E402


# ---------- fixture builders ----------


def _iter_payload(
    *,
    story_id: str,
    map50: float,
    size_bytes: int = 3_221_000,
    arena_used_bytes: int = 692_912,
    predicted_p4_latency_ms_p50: float = 5820.51,
    predicted_p4_fps: float = 0.172,
    verdict: str = "equal",
    status: str = "passed",
    blocked_reason: str | None = None,
    tflm_compatible: bool = True,
    pareto_key: str = "pareto",
) -> dict:
    """Match iter-A/B/C/D/E rollup shape (rich)."""
    payload = {
        "story_id": story_id,
        "title": f"{story_id} fixture",
        "status": status,
        "blocked_reason": blocked_reason,
        "model_path": f"training/edge/models/{story_id}.tflite",
        "p4_multiplier": 5.0,
        "p4_multiplier_source": "fixture",
        "eval": {
            "story_id": story_id,
            "model_path": f"training/edge/models/{story_id}.tflite",
            "model_format": "tflite_int8",
            "map50": map50,
            "size_bytes": size_bytes,
            "params": 3_011_043,
            "flops": 0,
            "input_hw": [224, 224],
            "latency_ms_p50": 3.6,
            "latency_ms_p95": 4.0,
            "val_images": 120,
            "notes": "",
        },
        "tflm": {
            "story_id": story_id,
            "model_path": f"training/edge/models/{story_id}.tflite",
            "model_size_bytes": size_bytes,
            "runs": 50,
            "arena_size_bytes": 8_388_608,
            "arena_used_bytes": arena_used_bytes,
            "tflm_compatible": tflm_compatible,
            "schema_status": "ok",
            "allocate_tensors_status": "ok",
            "timed_invoke_status": "ok",
        },
        "candidate_metrics": {
            "map50": map50,
            "size_bytes": size_bytes,
            "arena_used_bytes": arena_used_bytes,
            "predicted_p4_latency_ms_p50": predicted_p4_latency_ms_p50,
            "predicted_p4_fps": predicted_p4_fps,
        },
        pareto_key: {
            "verdict": verdict,
            "baseline_story": "fixture-baseline",
            "baseline_map50": 0.0,
            "baseline_size_bytes": 0,
            "baseline_arena_used_bytes": 0,
            "baseline_predicted_p4_latency_ms_p50": 0.0,
            "deltas": {},
            "thresholds": {},
        },
    }
    return payload


def _v1_eval_payload(*, story_id: str, map50: float, size_bytes: int) -> dict:
    return {
        "story_id": story_id,
        "model_path": f"training/edge/models/{story_id}.tflite",
        "model_format": "tflite_int8",
        "map50": map50,
        "size_bytes": size_bytes,
        "params": 3_011_043,
        "flops": 0,
        "input_hw": [224, 224],
        "latency_ms_p50": 3.6,
        "latency_ms_p95": 4.0,
        "val_images": 120,
        "notes": "",
    }


def _seed_full_fixture(tmp_path: Path) -> Path:
    """Write a complete iter-A..iter-G + v1 INT8 fixture under ``tmp_path``.

    The fixture deliberately picks distinct fps + mAP values so the frontier
    sort is deterministically observable.
    """
    results = tmp_path / "results"
    results.mkdir()

    # iter-A: deployable, baseline
    (results / "iter-A_per_channel_quant.json").write_text(
        json.dumps(_iter_payload(
            story_id="iter-A",
            map50=0.21,
            predicted_p4_latency_ms_p50=5800.0,
            predicted_p4_fps=0.172,
            verdict="equal",
        ))
    )
    # iter-B (single representative): deployable, lower fps
    (results / "iter-B_imgsz_224.json").write_text(
        json.dumps(_iter_payload(
            story_id="iter-B-224",
            map50=0.215,
            predicted_p4_latency_ms_p50=5790.0,
            predicted_p4_fps=0.173,
            verdict="equal",
        ))
    )
    # iter-C: zero-mAP (cls collapse) — must be excluded from frontier
    (results / "iter-C_pruned.json").write_text(
        json.dumps(_iter_payload(
            story_id="iter-C",
            map50=0.0,
            predicted_p4_latency_ms_p50=5855.0,
            predicted_p4_fps=0.171,
            verdict="regress",
        ))
    )
    # iter-D: highest mAP — should be the winner
    (results / "iter-D_mixed_int.json").write_text(
        json.dumps(_iter_payload(
            story_id="iter-D",
            map50=0.488,
            size_bytes=3_298_712,
            arena_used_bytes=1_307_552,
            predicted_p4_latency_ms_p50=5480.0,
            predicted_p4_fps=0.182,
            verdict="equal",
        ))
    )
    # iter-E: zero-mAP, narrowest student — excluded
    (results / "iter-E_yolov8n_0p75x.json").write_text(
        json.dumps(_iter_payload(
            story_id="iter-E",
            map50=0.0,
            size_bytes=1_977_944,
            arena_used_bytes=683_312,
            predicted_p4_latency_ms_p50=3804.0,
            predicted_p4_fps=0.263,
            verdict="regress",
        ))
    )
    # iter-F: dual-baseline shape (pareto_vs_yolo_v2_frontier)
    (results / "iter-F_nanodet_per_channel.json").write_text(
        json.dumps(_iter_payload(
            story_id="iter-F",
            map50=0.0,
            size_bytes=1_613_232,
            arena_used_bytes=2_253_888,
            predicted_p4_latency_ms_p50=8940.0,
            predicted_p4_fps=0.112,
            verdict="regress",
            pareto_key="pareto_vs_yolo_v2_frontier",
        ))
    )
    # iter-G: NMS off-graph — uses iter_a_baseline_metrics + inference_only_latency
    iter_g = {
        "story_id": "iter-G",
        "title": "iter-G fixture",
        "status": "passed",
        "blocked_reason": None,
        "model_path": "training/edge/models/x.tflite",
        "p4_multiplier": 5.0,
        "p4_multiplier_source": "fixture",
        "iter_a_baseline_metrics": {
            "map50": 0.21,
            "size_bytes": 3_221_000,
            "arena_used_bytes": 692_912,
            "predicted_p4_latency_ms_p50": 5820.0,
            "predicted_p4_fps": 0.171,
        },
        "inference_only_latency": {
            "predicted_p4_ms_p50": 5784.49,
            "predicted_p4_fps": 0.173,
        },
        "tflm_runs200": {
            "tflm_compatible": True,
        },
    }
    (results / "iter-G_off_graph_nms.json").write_text(json.dumps(iter_g))

    # iter-B siblings — include the rest so the matrix renders fully.
    # imgsz=192 collapses to zero-mAP in real data; the others retain mAP.
    iter_b_extras = (
        ("192", 0.0, 0.236),
        ("256", 0.20, 0.131),
        ("288", 0.198, 0.102),
        ("320", 0.199, 0.083),
    )
    for size, map50, fps in iter_b_extras:
        (results / f"iter-B_imgsz_{size}.json").write_text(
            json.dumps(_iter_payload(
                story_id=f"iter-B-{size}",
                map50=map50,
                predicted_p4_latency_ms_p50=4000.0 + int(size),
                predicted_p4_fps=fps,
                verdict="regress",
            ))
        )

    # v1 INT8 candidates — US-004 / US-005 / US-006-int8 / US-009
    (results / "US-004.json").write_text(
        json.dumps(_v1_eval_payload(story_id="US-004", map50=0.0, size_bytes=3_220_840))
    )
    (results / "US-005.json").write_text(
        json.dumps(_v1_eval_payload(story_id="US-005", map50=0.0, size_bytes=3_220_840))
    )
    (results / "US-006-int8.json").write_text(
        json.dumps(_v1_eval_payload(story_id="US-006-int8", map50=0.205, size_bytes=3_220_840))
    )
    (results / "US-009.json").write_text(
        json.dumps(_v1_eval_payload(story_id="US-009", map50=0.0, size_bytes=1_613_072))
    )
    return results


# ---------- tests ----------


def test_frontier_picker_excludes_blocked_and_zero_mAP(tmp_path: Path) -> None:
    """iter-H AC (a): frontier excludes blocked + zero-mAP rows."""
    results = _seed_full_fixture(tmp_path)

    # Overwrite iter-D as blocked to also cover the blocked exclusion.
    blocked = _iter_payload(
        story_id="iter-D",
        map50=0.488,
        size_bytes=3_298_712,
        predicted_p4_latency_ms_p50=5480.0,
        predicted_p4_fps=0.182,
        verdict="blocked",
        status="blocked",
        blocked_reason="CUDA OOM during fixture run",
    )
    (results / "iter-D_mixed_int.json").write_text(json.dumps(blocked))

    aggregate = agg.aggregate(results)
    frontier = agg.pick_frontier(aggregate.rows)
    labels = [r.label for r in frontier]

    # Zero-mAP rows excluded
    assert "iter-C: pruned 25%" not in labels
    assert "iter-E: width=0.75x" not in labels
    assert "iter-F: NanoDet per-channel" not in labels
    assert "iter-B: imgsz=192" not in labels
    # Blocked row excluded
    assert "iter-D: INT8w/INT16a" not in labels
    # Survivors stay
    assert "iter-A: per-channel quant (v2 floor)" in labels
    assert "iter-B: imgsz=224" in labels


def test_frontier_sorted_by_predicted_p4_fps_desc(tmp_path: Path) -> None:
    """iter-H AC (b): frontier rows are sorted by predicted_p4_fps desc."""
    results = _seed_full_fixture(tmp_path)
    aggregate = agg.aggregate(results)
    frontier = agg.pick_frontier(aggregate.rows)

    # Must have at least two rows so the sort is observable
    assert len(frontier) >= 2

    fps_seq = [r.predicted_p4_fps or 0.0 for r in frontier]
    assert fps_seq == sorted(fps_seq, reverse=True), (
        f"frontier fps sequence not descending: {fps_seq}"
    )

    # The fixture's iter-D fps (0.182) is the highest among nonzero-mAP rows;
    # verify it ranks first.
    assert frontier[0].label == "iter-D: INT8w/INT16a"


def test_deprecation_list_includes_v1_INT8_candidates_dominated(tmp_path: Path) -> None:
    """iter-H AC (c): deprecation list lists v1 INT8 candidates the v2 frontier dominates.

    Fixture: v2 winner has mAP 0.488; v1 candidates have mAPs 0.0, 0.0,
    0.205, 0.0 — all strictly lower, so all four should be in the list.
    """
    results = _seed_full_fixture(tmp_path)
    aggregate = agg.aggregate(results)
    frontier = agg.pick_frontier(aggregate.rows)
    deprecated = agg.deprecated_v1_rows(aggregate.v1_rows, frontier)

    deprecated_labels = {v.label for v in deprecated}
    assert "YOLOv8n-cat INT8 (US-004)" in deprecated_labels
    assert "YOLOv8n-cat-QAT INT8 (US-005)" in deprecated_labels
    assert "YOLOv8n-cat-distilled INT8 (US-006)" in deprecated_labels
    assert "NanoDet-cat INT8 (US-009)" in deprecated_labels


def test_recommendation_addresses_mipi_csi_vs_http_stream(tmp_path: Path) -> None:
    """iter-H AC (d): recommendation paragraph addresses MIPI-CSI vs HTTP-stream."""
    results = _seed_full_fixture(tmp_path)
    aggregate = agg.aggregate(results)
    frontier = agg.pick_frontier(aggregate.rows)
    winner = agg.pick_winner(frontier)
    text = agg.format_recommendation(winner, frontier)

    lower = text.lower()
    assert "mipi-csi" in lower
    assert "http" in lower and "stream" in lower
    # Also addresses PSRAM placement and classifier handoff per AC
    assert "psram" in lower
    assert "server/vision/classifier.py" in text


def test_summary_v2_md_has_required_sections(tmp_path: Path) -> None:
    """Smoke-test that format_summary renders all four PRD-required sections."""
    results = _seed_full_fixture(tmp_path)
    aggregate = agg.aggregate(results)
    text = agg.format_summary(aggregate)

    assert "## v2 Pareto frontier" in text
    assert "## Recommendation" in text
    assert "## Deprecation list" in text
    assert "Predicted fps" in text or "predicted ESP32-P4" in text
    # The aggregator must not generate the SUMMARY.md path — only SUMMARY_v2.md
    assert "SUMMARY_v2.md" in text


def test_write_outputs_emits_both_files(tmp_path: Path) -> None:
    """write_outputs emits both SUMMARY_v2.md and v2_frontier.json with valid shape."""
    results = _seed_full_fixture(tmp_path)
    summary_path = tmp_path / "out" / "SUMMARY_v2.md"
    frontier_path = tmp_path / "out" / "v2_frontier.json"
    summary_path.parent.mkdir()

    result = agg.write_outputs(results, summary_path, frontier_path)

    assert summary_path.exists()
    assert frontier_path.exists()
    payload = json.loads(frontier_path.read_text())
    assert "frontier" in payload
    assert "winner" in payload
    assert "deprecated_v1" in payload
    # Aggregator should report at least the iter-A row.
    assert any(r.label.startswith("iter-A") for r in result.rows)


def test_iter_g_extraction_uses_inference_only_latency(tmp_path: Path) -> None:
    """iter-G has no candidate_metrics; aggregator must read inference_only_latency."""
    results = _seed_full_fixture(tmp_path)
    aggregate = agg.aggregate(results)
    iter_g = next(r for r in aggregate.rows if r.label.startswith("iter-G"))

    # Per fixture: iter_a_baseline_metrics.predicted_p4_fps = 0.171
    # but inference_only_latency.predicted_p4_fps = 0.173 -> the latter wins.
    assert iter_g.predicted_p4_fps == pytest.approx(0.173)
    assert iter_g.predicted_p4_latency_ms_p50 == pytest.approx(5784.49)
    # mAP carries over from iter_a_baseline_metrics since iter-G uses iter-A's tflite.
    assert iter_g.map50 == pytest.approx(0.21)


def test_winner_falls_back_to_none_when_frontier_empty(tmp_path: Path) -> None:
    """When every row is blocked or zero-mAP, pick_winner returns None and the
    recommendation surfaces a 'no deployable candidate' message.
    """
    results = tmp_path / "results"
    results.mkdir()
    # Only zero-mAP rows
    (results / "iter-A_per_channel_quant.json").write_text(
        json.dumps(_iter_payload(
            story_id="iter-A",
            map50=0.0,
            verdict="regress",
        ))
    )
    aggregate = agg.aggregate(results)
    frontier = agg.pick_frontier(aggregate.rows)
    winner = agg.pick_winner(frontier)
    assert winner is None
    text = agg.format_recommendation(winner, frontier)
    assert "no deployable" in text.lower()


def test_aggregator_reports_missing_required_iter_jsons(tmp_path: Path) -> None:
    """Missing required iter rollups land in result.missing for the SUMMARY to flag."""
    results = tmp_path / "results"
    results.mkdir()
    # Only iter-A present; the others should appear in `missing`.
    (results / "iter-A_per_channel_quant.json").write_text(
        json.dumps(_iter_payload(story_id="iter-A", map50=0.21))
    )
    aggregate = agg.aggregate(results)
    assert "iter-D_mixed_int.json" in aggregate.missing
    assert "iter-G_off_graph_nms.json" in aggregate.missing
    # iter-A is present so should not be in missing.
    assert "iter-A_per_channel_quant.json" not in aggregate.missing
