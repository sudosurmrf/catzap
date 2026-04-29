"""Tests for the US-011 aggregator.

Per US-011 acceptance: given two bench-JSON fixtures, the aggregator must
produce the comparison-table format US-012 will ingest. Mirrors
test_tflm_bench_parser.py's import pattern (firmware/edge-bench/ on sys.path).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BENCH_DIR = Path(__file__).resolve().parents[3] / "firmware" / "edge-bench"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import aggregate  # noqa: E402


def _make_report(
    *,
    story_id: str,
    model_path: str = "training/edge/models/x_int8.tflite",
    size: int = 3_220_840,
    arena: int = 692_816,
    p50_us: float = 1_181_854.0,
    p95_us: float = 1_194_148.0,
    multiplier: float = 8.0,
    op_breakdown: list[dict] | None = None,
    tflm_compatible: bool = True,
    allocate_status: str = "ok",
    timed_status: str = "ok",
    schema_status: str = "ok",
) -> dict:
    """Build a bench-JSON dict matching run_bench.report_to_result's schema."""
    if op_breakdown is None:
        op_breakdown = [
            {"op_name": "CONV_2D", "count": 3000, "total_us": 480_000, "percent": 90.0},
            {"op_name": "LOGISTIC", "count": 50, "total_us": 40_000, "percent": 7.5},
            {"op_name": "MUL", "count": 750, "total_us": 13_000, "percent": 2.4},
            {"op_name": "ADD", "count": 750, "total_us": 500, "percent": 0.1},
        ]
    return {
        "story_id": story_id,
        "model_path": model_path,
        "model_size_bytes": size,
        "runs": 50,
        "arena_size_bytes": 8_388_608,
        "arena_used_bytes": arena,
        "input_bytes": 150_528,
        "output_bytes": 5_145,
        "schema_status": schema_status,
        "allocate_tensors_status": allocate_status,
        "timed_invoke_status": timed_status,
        "profiler_overflowed": False,
        "tflm_compatible": tflm_compatible,
        "op_breakdown": op_breakdown,
        "raw_x86_us_p50": p50_us,
        "raw_x86_us_p95": p95_us,
        "raw_x86_us_min": p50_us - 5_000,
        "raw_x86_us_max": p95_us + 5_000,
        "raw_x86_us_mean": (p50_us + p95_us) / 2.0,
        "x86_to_s3_multiplier": multiplier,
        "predicted_s3_us_p50": p50_us * multiplier,
        "predicted_s3_us_p95": p95_us * multiplier,
        "predicted_s3_fps": 1_000_000.0 / (p50_us * multiplier),
        "binary": "/abs/firmware/edge-bench/build/edgebench",
        "tflm_commit": "51bee03bed4776f1de88dd87226ff8c260f88e3c",
    }


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload, indent=2))
    return p


def test_aggregate_two_fixtures_preserves_order(tmp_path: Path) -> None:
    yolo = _write(
        tmp_path,
        "US-011-yolov8n-tflm.json",
        _make_report(
            story_id="US-011-yolov8n",
            model_path="training/edge/models/yolov8n_cat_int8.tflite",
            size=3_220_840,
            arena=692_816,
            p50_us=1_181_854.0,
        ),
    )
    nanodet = _write(
        tmp_path,
        "US-011-nanodet-tflm.json",
        _make_report(
            story_id="US-011-nanodet",
            model_path="training/edge/models/nanodet_cat_0.5x_224_int8.tflite",
            size=1_613_072,
            arena=2_253_792,
            p50_us=1_834_548.0,
            op_breakdown=[
                {"op_name": "CONV_2D", "count": 4500, "total_us": 808_000, "percent": 88.1},
                {"op_name": "DEPTHWISE_CONV_2D", "count": 1700, "total_us": 82_870, "percent": 9.0},
                {"op_name": "LEAKY_RELU", "count": 800, "total_us": 10_969, "percent": 1.2},
            ],
        ),
    )
    rows = aggregate.aggregate_us011_jsons([yolo, nanodet])
    assert [r["model"] for r in rows] == ["yolov8n", "nanodet"]
    assert rows[0]["size_bytes"] == 3_220_840
    assert rows[1]["arena_used_bytes"] == 2_253_792


@pytest.mark.parametrize(
    "filename,expected_model",
    [
        ("US-011-yolov8n-tflm.json", "yolov8n"),
        ("US-011-yolov8n-qat-tflm.json", "yolov8n-qat"),
        ("US-011-yolov8n-distilled-tflm.json", "yolov8n-distilled"),
        ("US-011-nanodet-tflm.json", "nanodet"),
    ],
)
def test_model_name_derived_from_filename(
    tmp_path: Path, filename: str, expected_model: str
) -> None:
    p = _write(tmp_path, filename, _make_report(story_id="x"))
    rows = aggregate.aggregate_us011_jsons([p])
    assert rows[0]["model"] == expected_model


def test_aggregate_row_keeps_top3_ops_only_in_descending_order(tmp_path: Path) -> None:
    rep = _make_report(
        story_id="US-011-yolov8n",
        op_breakdown=[
            {"op_name": "ADD", "count": 50, "total_us": 100, "percent": 0.1},
            {"op_name": "CONV_2D", "count": 3000, "total_us": 500_000, "percent": 90.0},
            {"op_name": "MUL", "count": 750, "total_us": 13_000, "percent": 2.4},
            {"op_name": "LOGISTIC", "count": 50, "total_us": 40_000, "percent": 7.4},
            {"op_name": "RESHAPE", "count": 5, "total_us": 5, "percent": 0.0},
        ],
    )
    p = _write(tmp_path, "US-011-yolov8n-tflm.json", rep)
    rows = aggregate.aggregate_us011_jsons([p])
    names = [op["op_name"] for op in rows[0]["top3_ops"]]
    assert names == ["CONV_2D", "LOGISTIC", "MUL"]


def test_predicted_s3_fps_consistent_with_predicted_us(tmp_path: Path) -> None:
    rep = _make_report(story_id="x", p50_us=1_000_000.0, multiplier=10.0)
    p = _write(tmp_path, "US-011-x-tflm.json", rep)
    rows = aggregate.aggregate_us011_jsons([p])
    row = rows[0]
    assert row["predicted_s3_us_p50"] == pytest.approx(1_000_000.0 * 10.0)
    assert row["predicted_s3_fps"] == pytest.approx(
        1_000_000.0 / (1_000_000.0 * 10.0)
    )


def test_tflm_incompatible_row_is_kept_with_offending_ops_and_tag(tmp_path: Path) -> None:
    """Per PRD: TFLM-incompatible models MUST appear in the table — not dropped."""
    rep = _make_report(
        story_id="US-011-broken",
        tflm_compatible=False,
        allocate_status="failed",
        op_breakdown=[
            {
                "op_name": "CUSTOM_NMS",
                "count": 0,
                "total_us": 0,
                "percent": 0.0,
                "unsupported": True,
            },
            {
                "op_name": "CONV_2D",
                "count": 100,
                "total_us": 1000,
                "percent": 100.0,
            },
        ],
    )
    p = _write(tmp_path, "US-011-broken-tflm.json", rep)
    rows = aggregate.aggregate_us011_jsons([p])
    assert len(rows) == 1
    assert rows[0]["tflm_compatible"] is False
    assert rows[0]["offending_ops"] == ["CUSTOM_NMS"]
    md = aggregate.format_comparison_table(rows)
    assert "no (CUSTOM_NMS)" in md


def test_format_comparison_table_emits_markdown_with_required_columns(tmp_path: Path) -> None:
    rep = _make_report(story_id="US-011-yolov8n")
    p = _write(tmp_path, "US-011-yolov8n-tflm.json", rep)
    rows = aggregate.aggregate_us011_jsons([p])
    md = aggregate.format_comparison_table(rows)
    for col in [
        "model",
        "size_bytes",
        "arena_used_bytes",
        "top3_ops_by_time",
        "raw_x86_us_p50",
        "predicted_s3_us_p50",
        "predicted_s3_fps",
        "tflm_compatible",
    ]:
        assert col in md
    assert "yolov8n" in md
    assert "yes" in md  # tflm_compatible column for the happy path


def test_aggregate_empty_op_breakdown_does_not_crash(tmp_path: Path) -> None:
    rep = _make_report(story_id="US-011-x", op_breakdown=[])
    p = _write(tmp_path, "US-011-x-tflm.json", rep)
    rows = aggregate.aggregate_us011_jsons([p])
    assert rows[0]["top3_ops"] == []
    md = aggregate.format_comparison_table(rows)
    assert " - " in md  # the "-" placeholder for the empty top3 cell
