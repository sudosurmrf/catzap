"""Tests for US-012's aggregate_summary.py.

Per US-012 acceptance: given fixture training/edge/results/*.json files,
aggregate_summary.py produces a SUMMARY.md whose decision matrix has the
expected rows/columns. Mirrors test_us011_aggregate.py's import pattern
(extra dir on sys.path).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make aggregate_summary importable. It lives in training/edge/results/
# which isn't a package, so we put the dir on sys.path.
_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
if str(_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(_RESULTS_DIR))

import aggregate_summary  # noqa: E402


# ---------- fixture builders ----------

def _eval_payload(
    *,
    story_id: str,
    model_path: str = "training/edge/models/x.tflite",
    model_format: str = "tflite_int8",
    map50: float = 0.5,
    size_bytes: int = 3_220_840,
    latency_ms_p50: float = 3.6,
    latency_ms_p95: float = 4.0,
    notes: str = "",
) -> dict:
    """Match training.edge.eval.types.EvalResult on-disk shape."""
    return {
        "story_id": story_id,
        "model_path": model_path,
        "model_format": model_format,
        "map50": map50,
        "size_bytes": size_bytes,
        "params": 3_011_043,
        "flops": 0,
        "input_hw": [224, 224],
        "latency_ms_p50": latency_ms_p50,
        "latency_ms_p95": latency_ms_p95,
        "val_images": 120,
        "notes": notes,
    }


def _tflm_payload(
    *,
    story_id: str,
    arena_used: int = 692_816,
    p50_us: float = 1_181_854.0,
    p95_us: float = 1_194_148.0,
    multiplier: float = 8.0,
    tflm_compatible: bool = True,
    op_breakdown: list[dict] | None = None,
) -> dict:
    """Match firmware/edge-bench/run_bench.py:report_to_result on-disk shape."""
    if op_breakdown is None:
        op_breakdown = [
            {"op_name": "CONV_2D", "count": 3000, "total_us": 480_000, "percent": 90.0},
            {"op_name": "LOGISTIC", "count": 50, "total_us": 40_000, "percent": 7.5},
            {"op_name": "MUL", "count": 750, "total_us": 13_000, "percent": 2.5},
        ]
    return {
        "story_id": story_id,
        "model_path": "training/edge/models/x.tflite",
        "model_size_bytes": 3_220_840,
        "runs": 50,
        "arena_size_bytes": 8_388_608,
        "arena_used_bytes": arena_used,
        "input_bytes": 150_528,
        "output_bytes": 5_145,
        "schema_status": "ok",
        "allocate_tensors_status": "ok",
        "timed_invoke_status": "ok",
        "profiler_overflowed": False,
        "tflm_compatible": tflm_compatible,
        "op_breakdown": op_breakdown,
        "raw_x86_us_p50": p50_us,
        "raw_x86_us_p95": p95_us,
        "raw_x86_us_min": p50_us - 5000,
        "raw_x86_us_max": p95_us + 5000,
        "raw_x86_us_mean": (p50_us + p95_us) / 2.0,
        "x86_to_s3_multiplier": multiplier,
        "predicted_s3_us_p50": p50_us * multiplier,
        "predicted_s3_us_p95": p95_us * multiplier,
        "predicted_s3_fps": 1_000_000.0 / (p50_us * multiplier),
        "binary": "/abs/edgebench",
        "tflm_commit": "deadbeef",
    }


def _write(dir_: Path, name: str, payload: dict) -> Path:
    p = dir_ / name
    p.write_text(json.dumps(payload, indent=2))
    return p


def _seed_full_fixture(dir_: Path) -> None:
    """Write a minimal but PRD-complete set of result JSONs into dir_.

    All six matrix rows present (including optional QAT). Numbers are made up
    but consistent so per-row assertions are unambiguous.
    """
    # Eval JSONs
    _write(dir_, "US-003.json", _eval_payload(
        story_id="US-003", model_format="pytorch", map50=0.91,
        size_bytes=6_201_881, latency_ms_p50=5.9))
    _write(dir_, "US-004.json", _eval_payload(
        story_id="US-004", model_format="tflite_int8", map50=0.0,
        size_bytes=3_220_840, latency_ms_p50=3.6))
    _write(dir_, "US-005.json", _eval_payload(
        story_id="US-005", model_format="tflite_int8", map50=0.0,
        size_bytes=3_220_840, latency_ms_p50=3.8))
    _write(dir_, "US-006-int8.json", _eval_payload(
        story_id="US-006-int8", model_format="tflite_int8", map50=0.205,
        size_bytes=3_220_840, latency_ms_p50=3.7))
    _write(dir_, "US-008.json", _eval_payload(
        story_id="US-008", model_format="pytorch", map50=0.0,
        size_bytes=0, latency_ms_p50=0.0,
        notes="DEVIATION: cat .pth not produced"))
    _write(dir_, "US-009.json", _eval_payload(
        story_id="US-009", model_format="tflite_int8", map50=0.0,
        size_bytes=1_613_072, latency_ms_p50=56.2))
    # TFLM bench JSONs
    _write(dir_, "US-011-yolov8n-tflm.json", _tflm_payload(
        story_id="US-011-yolov8n", arena_used=692_816, p50_us=1_181_854.0))
    _write(dir_, "US-011-yolov8n-qat-tflm.json", _tflm_payload(
        story_id="US-011-yolov8n-qat", arena_used=692_816, p50_us=1_182_374.0))
    _write(dir_, "US-011-yolov8n-distilled-tflm.json", _tflm_payload(
        story_id="US-011-yolov8n-distilled",
        arena_used=692_816, p50_us=1_176_673.0))
    _write(dir_, "US-011-nanodet-tflm.json", _tflm_payload(
        story_id="US-011-nanodet",
        arena_used=2_253_792, p50_us=1_834_548.0))


# ---------- tests ----------

def test_aggregate_full_fixture_produces_expected_six_rows(tmp_path: Path) -> None:
    """All six PRD-spec'd rows appear, in the spec'd order, with the right labels."""
    _seed_full_fixture(tmp_path)
    result = aggregate_summary.aggregate(tmp_path)
    assert result.missing == []
    labels = [r.label for r in result.rows]
    assert labels == [
        "YOLOv8n-cat fp32",
        "YOLOv8n-cat INT8",
        "YOLOv8n-cat-QAT INT8",
        "YOLOv8n-cat-distilled INT8",
        "NanoDet-cat fp32",
        "NanoDet-cat INT8",
    ]


def test_aggregate_pulls_size_and_map_from_eval_json(tmp_path: Path) -> None:
    _seed_full_fixture(tmp_path)
    rows = aggregate_summary.aggregate(tmp_path).rows
    distilled = next(r for r in rows if r.label == "YOLOv8n-cat-distilled INT8")
    assert distilled.size_bytes == 3_220_840
    assert distilled.map50 == pytest.approx(0.205)
    assert distilled.host_latency_ms_p50 == pytest.approx(3.7)
    assert distilled.eval_source == "US-006-int8.json"


def test_aggregate_pulls_arena_and_fps_from_tflm_json(tmp_path: Path) -> None:
    _seed_full_fixture(tmp_path)
    rows = aggregate_summary.aggregate(tmp_path).rows
    yolov8n_int8 = next(r for r in rows if r.label == "YOLOv8n-cat INT8")
    assert yolov8n_int8.tflm_arena_bytes == 692_816
    assert yolov8n_int8.tflm_compatible == "yes"
    assert yolov8n_int8.predicted_s3_fps == pytest.approx(
        1_000_000.0 / (1_181_854.0 * 8.0)
    )
    assert yolov8n_int8.tflm_source == "US-011-yolov8n-tflm.json"


def test_fp32_rows_have_no_tflm_columns(tmp_path: Path) -> None:
    _seed_full_fixture(tmp_path)
    rows = aggregate_summary.aggregate(tmp_path).rows
    fp32 = next(r for r in rows if r.label == "YOLOv8n-cat fp32")
    assert fp32.tflm_arena_bytes is None
    assert fp32.predicted_s3_fps is None
    assert fp32.tflm_compatible == "n/a"
    assert fp32.tflm_source is None


def test_optional_qat_row_skipped_when_us005_missing(tmp_path: Path) -> None:
    """If US-005.json is absent (PTQ acceptable, QAT skipped), the QAT row is silently dropped."""
    _seed_full_fixture(tmp_path)
    (tmp_path / "US-005.json").unlink()
    (tmp_path / "US-011-yolov8n-qat-tflm.json").unlink()
    result = aggregate_summary.aggregate(tmp_path)
    labels = [r.label for r in result.rows]
    assert "YOLOv8n-cat-QAT INT8" not in labels
    # Optional row absence does NOT pollute `missing`
    assert "US-005.json" not in result.missing


def test_required_row_missing_is_reported(tmp_path: Path) -> None:
    _seed_full_fixture(tmp_path)
    (tmp_path / "US-009.json").unlink()
    result = aggregate_summary.aggregate(tmp_path)
    assert "US-009.json" in result.missing
    assert all(r.label != "NanoDet-cat INT8" for r in result.rows)


def test_tflm_incompatible_row_keeps_row_and_tags_compat(tmp_path: Path) -> None:
    """Per US-011 acceptance: failure is itself a finding — row stays in matrix."""
    _seed_full_fixture(tmp_path)
    bad = _tflm_payload(
        story_id="US-011-nanodet",
        arena_used=2_253_792,
        p50_us=1_834_548.0,
        tflm_compatible=False,
        op_breakdown=[
            {
                "op_name": "CUSTOM_NMS",
                "count": 1,
                "total_us": 0,
                "percent": 0.0,
                "unsupported": True,
            }
        ],
    )
    _write(tmp_path, "US-011-nanodet-tflm.json", bad)
    rows = aggregate_summary.aggregate(tmp_path).rows
    nanodet = next(r for r in rows if r.label == "NanoDet-cat INT8")
    assert nanodet.tflm_compatible.startswith("no")
    assert "CUSTOM_NMS" in nanodet.tflm_compatible


def test_format_decision_matrix_has_all_seven_columns(tmp_path: Path) -> None:
    _seed_full_fixture(tmp_path)
    result = aggregate_summary.aggregate(tmp_path)
    md = aggregate_summary.format_decision_matrix(result.rows)
    # PRD-specified columns
    for col in [
        "model",
        "size_bytes",
        "mAP@0.5",
        "host_int8_latency_ms_p50",
        "predicted_s3_fps",
        "tflm_arena_bytes",
        "tflm_compatible",
    ]:
        assert col in md, f"column {col!r} missing from rendered matrix"
    # Six body rows + 1 header + 1 separator = 8 lines
    assert md.count("\n") == 7


def test_recommendation_addresses_all_prd_items(tmp_path: Path) -> None:
    """PRD recommendation must address (a) winner, (b) fps, (c) PSRAM headroom,
    (d) <5 fps caveat, (e) >5pt fp32-vs-INT8 follow-up trigger."""
    _seed_full_fixture(tmp_path)
    result = aggregate_summary.aggregate(tmp_path)
    md = aggregate_summary.format_recommendation(result.rows)
    # (a) winner — distilled INT8 is the only nonzero-mAP INT8 candidate
    assert "YOLOv8n-cat-distilled INT8" in md
    # (b) fps — three-decimal predicted_s3_fps
    assert "fps" in md
    # (c) PSRAM headroom in MB
    assert "PSRAM" in md and "headroom" in md
    # (d) sub-5 fps caveat
    assert "5-fps" in md or "5 fps" in md
    # (e) >5-point fp32 vs INT8 trigger; distilled INT8 is 0.205 vs fp32 0.91 ~70 pt drop
    assert "Follow-up" in md or "follow-up" in md
    assert "QAT" in md or "wider" in md


def test_recommendation_handles_no_int8_winner(tmp_path: Path) -> None:
    """If every INT8 row is TFLM-incompatible, the recommendation flags it instead of crashing."""
    _seed_full_fixture(tmp_path)
    # Mark every TFLM bench as incompatible.
    for name in [
        "US-011-yolov8n-tflm.json",
        "US-011-yolov8n-qat-tflm.json",
        "US-011-yolov8n-distilled-tflm.json",
        "US-011-nanodet-tflm.json",
    ]:
        bad = _tflm_payload(
            story_id=name.replace("-tflm.json", "").lstrip("US-011-"),
            tflm_compatible=False,
            op_breakdown=[
                {"op_name": "BAD_OP", "count": 1, "total_us": 0,
                 "percent": 0.0, "unsupported": True}
            ],
        )
        _write(tmp_path, name, bad)
    result = aggregate_summary.aggregate(tmp_path)
    md = aggregate_summary.format_recommendation(result.rows)
    assert "No deployable INT8 candidate" in md


def test_write_summary_creates_file_with_decision_and_recommendation(
    tmp_path: Path,
) -> None:
    _seed_full_fixture(tmp_path)
    out = tmp_path / "SUMMARY.md"
    result = aggregate_summary.write_summary(tmp_path, out)
    assert out.exists()
    body = out.read_text()
    assert "# Edge Model Reduction PoC — Decision Matrix" in body
    assert "## Decision matrix" in body
    assert "## Recommendation" in body
    assert "## How to regenerate" in body
    # All six rows rendered
    assert "YOLOv8n-cat fp32" in body
    assert "YOLOv8n-cat INT8" in body
    assert "YOLOv8n-cat-QAT INT8" in body
    assert "YOLOv8n-cat-distilled INT8" in body
    assert "NanoDet-cat fp32" in body
    assert "NanoDet-cat INT8" in body
    assert result.missing == []


def test_main_cli_round_trip(tmp_path: Path) -> None:
    """`python aggregate_summary.py --results-dir <tmp> --output <tmp>/SUMMARY.md` works."""
    _seed_full_fixture(tmp_path)
    out = tmp_path / "SUMMARY.md"
    rc = aggregate_summary.main(
        ["--results-dir", str(tmp_path), "--output", str(out)]
    )
    assert rc == 0
    assert out.exists()
    body = out.read_text()
    assert "Decision matrix" in body
