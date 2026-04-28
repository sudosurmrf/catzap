"""Aggregate US-011 TFLM benchmark JSONs into a comparison-table format.

Consumes JSONs produced by ``firmware/edge-bench/run_bench.py``
(``training/edge/results/<story-id>-tflm.json``) and emits a list of dicts that
US-012's ``aggregate_summary.py`` can ingest without re-parsing the raw shape.

Each row carries the side-by-side fields the PRD asks for:

  model | size_bytes | arena_used_bytes | top3_ops |
  raw_x86_us_p50 | predicted_s3_us_p50 | predicted_s3_fps |
  tflm_compatible | offending_ops

`offending_ops` is the list of TFLM-incompatible op names surfaced by edgebench
(empty when the model loaded cleanly). Per US-011 acceptance, models that fail
to load MUST appear in the table tagged TFLM-incompatible — not silently
dropped.

The model_name is derived from the JSON filename: ``US-011-yolov8n-tflm.json``
-> ``yolov8n``. Pass an explicit ``model_name`` in the JSON if a different
display label is desired.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import run_bench

DEFAULT_TOP_K = 3


def _model_name_from_path(path: Path) -> str:
    """``training/edge/results/US-011-yolov8n-tflm.json`` -> ``yolov8n``."""
    stem = path.stem
    if stem.endswith("-tflm"):
        stem = stem[: -len("-tflm")]
    if stem.startswith("US-011-"):
        stem = stem[len("US-011-") :]
    return stem


def aggregate_row(report: dict, *, source_path: Path | None = None) -> dict:
    """Convert one bench JSON into a comparison-table row.

    The input is the dict shape written by ``run_bench.report_to_result``.
    """
    op_breakdown = report.get("op_breakdown", []) or []
    top3 = run_bench.top_k_ops(op_breakdown, k=DEFAULT_TOP_K)
    offending = [op["op_name"] for op in op_breakdown if op.get("unsupported")]
    if source_path is not None:
        default_name = _model_name_from_path(source_path)
    else:
        default_name = report.get("story_id", "unknown")
    return {
        "model": report.get("model_name", default_name),
        "story_id": report.get("story_id", ""),
        "model_path": report.get("model_path", ""),
        "size_bytes": int(report.get("model_size_bytes", 0)),
        "arena_used_bytes": int(report.get("arena_used_bytes", 0)),
        "top3_ops": [
            {
                "op_name": op["op_name"],
                "total_us": int(op.get("total_us", 0)),
                "percent": float(op.get("percent", 0.0)),
            }
            for op in top3
        ],
        "raw_x86_us_p50": float(report.get("raw_x86_us_p50", 0.0)),
        "predicted_s3_us_p50": float(report.get("predicted_s3_us_p50", 0.0)),
        "predicted_s3_fps": float(report.get("predicted_s3_fps", 0.0)),
        "x86_to_s3_multiplier": float(report.get("x86_to_s3_multiplier", 0.0)),
        "tflm_compatible": bool(report.get("tflm_compatible", False)),
        "offending_ops": offending,
        "tflm_commit": report.get("tflm_commit", "unknown"),
    }


def aggregate_us011_jsons(paths: Iterable[Path | str]) -> list[dict]:
    """Aggregate a sequence of US-011 bench JSONs into comparison-table rows.

    Order is preserved from the input iterable so callers control row order.
    """
    rows: list[dict] = []
    for raw in paths:
        path = Path(raw)
        report = json.loads(path.read_text())
        rows.append(aggregate_row(report, source_path=path))
    return rows


def _fmt_top3(top3: list[dict]) -> str:
    if not top3:
        return "-"
    return ", ".join(
        f"{op['op_name']} ({op['percent']:.1f}%)" for op in top3
    )


def _fmt_compat(row: dict) -> str:
    if row["tflm_compatible"]:
        return "yes"
    if row["offending_ops"]:
        ops = ", ".join(row["offending_ops"])
        return f"no ({ops})"
    return "no"


def format_comparison_table(rows: list[dict]) -> str:
    """Render the rows as a GitHub-flavored markdown table.

    Columns match the PRD's US-011 spec:
        model | size | arena_bytes | top-3-ops-by-time |
        x86_us_p50 | predicted_s3_us_p50 | predicted_s3_fps | tflm_compatible
    """
    header = (
        "| model | size_bytes | arena_used_bytes | top3_ops_by_time | "
        "raw_x86_us_p50 | predicted_s3_us_p50 | predicted_s3_fps | "
        "tflm_compatible |"
    )
    sep = "|" + "|".join([" --- "] * 8) + "|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {model} | {size} | {arena} | {ops} | {p50:.0f} | "
            "{s3:.0f} | {fps:.3f} | {compat} |".format(
                model=row["model"],
                size=row["size_bytes"],
                arena=row["arena_used_bytes"],
                ops=_fmt_top3(row["top3_ops"]),
                p50=row["raw_x86_us_p50"],
                s3=row["predicted_s3_us_p50"],
                fps=row["predicted_s3_fps"],
                compat=_fmt_compat(row),
            )
        )
    return "\n".join(lines)
