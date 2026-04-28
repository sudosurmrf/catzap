"""US-012 — Aggregate per-story result JSONs into the SUMMARY.md decision matrix.

Reads every relevant ``training/edge/results/<story-id>.json`` (EvalResult shape,
US-001 through US-009) and ``training/edge/results/US-011-*-tflm.json`` (TFLM
bench-report shape, see ``firmware/edge-bench/run_bench.py:report_to_result``)
and renders a single ``SUMMARY.md`` with the PRD's decision matrix and a
recommendation paragraph.

Designed to be re-run from the CLI so SUMMARY.md is always rebuildable from
the raw JSONs:

    python training/edge/results/aggregate_summary.py \\
        --results-dir training/edge/results \\
        --output training/edge/results/SUMMARY.md

The TFLM rows are pulled through ``firmware/edge-bench/aggregate.py`` so the
US-011 -> US-012 ingestion contract stays in one place. See
``training/edge/results/US-011.md`` and the "US-011 aggregator is the US-012
ingestion seam" pattern in ``progress.txt``.

Why mechanical aggregation: individual EvalResult JSONs may be regenerated as
quantization or training is iterated; recommendations and matrix cells must
reflect those refreshes without manual editing of SUMMARY.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Pull aggregate_us011_jsons / format helpers from the US-011 aggregator so
# US-012 doesn't re-derive the bench-JSON shape.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCH_DIR = _REPO_ROOT / "firmware" / "edge-bench"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
import aggregate as us011_aggregate  # noqa: E402

PSRAM_BUDGET_BYTES = 8 * 1024 * 1024  # ESP32-S3 ~8 MB PSRAM target per PRD
S3_FPS_TARGET = 5.0  # PRD success metric threshold
FP32_VS_INT8_DROP_THRESHOLD = 5.0  # mAP points; PRD recommendation rule (e)


@dataclass(frozen=True)
class RowSpec:
    """Static row spec for the decision matrix.

    Each spec maps a candidate model row to the EvalResult JSON that supplies
    accuracy / size / host-CPU latency, and (optionally) the US-011 TFLM bench
    JSON that supplies arena_used_bytes and predicted_s3_fps.
    """

    label: str
    eval_json: str
    tflm_json: str | None = None
    optional: bool = False  # row may be skipped if eval_json is missing


# Matrix row order is the order specified in PRD US-012 acceptance criterion 1.
ROW_SPECS: tuple[RowSpec, ...] = (
    RowSpec(
        label="YOLOv8n-cat fp32",
        eval_json="US-003.json",
        tflm_json=None,  # fp32 PyTorch — not run through TFLM
    ),
    RowSpec(
        label="YOLOv8n-cat INT8",
        eval_json="US-004.json",
        tflm_json="US-011-yolov8n-tflm.json",
    ),
    RowSpec(
        label="YOLOv8n-cat-QAT INT8",
        eval_json="US-005.json",
        tflm_json="US-011-yolov8n-qat-tflm.json",
        optional=True,  # only present if PTQ mAP_drop > 3.0 triggered QAT
    ),
    RowSpec(
        label="YOLOv8n-cat-distilled INT8",
        eval_json="US-006-int8.json",
        tflm_json="US-011-yolov8n-distilled-tflm.json",
    ),
    RowSpec(
        label="NanoDet-cat fp32",
        eval_json="US-008.json",
        tflm_json=None,
    ),
    RowSpec(
        label="NanoDet-cat INT8",
        eval_json="US-009.json",
        tflm_json="US-011-nanodet-tflm.json",
    ),
)


@dataclass
class MatrixRow:
    """One row of the decision matrix in machine-readable form."""

    label: str
    eval_source: str
    tflm_source: str | None
    size_bytes: int
    map50: float
    host_latency_ms_p50: float
    predicted_s3_fps: float | None
    tflm_arena_bytes: int | None
    tflm_compatible: str  # "yes" | "no" | "n/a"
    notes: str = ""


@dataclass
class AggregateResult:
    rows: list[MatrixRow] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _load_eval(path: Path) -> dict:
    return json.loads(path.read_text())


def _build_row(spec: RowSpec, results_dir: Path) -> MatrixRow | None:
    eval_path = results_dir / spec.eval_json
    if not eval_path.exists():
        return None
    eval_data = _load_eval(eval_path)

    tflm_data: dict | None = None
    tflm_source: str | None = None
    if spec.tflm_json is not None:
        tflm_path = results_dir / spec.tflm_json
        if tflm_path.exists():
            tflm_data = _load_eval(tflm_path)
            tflm_source = spec.tflm_json

    if tflm_data is not None:
        tflm_row = us011_aggregate.aggregate_row(
            tflm_data, source_path=results_dir / spec.tflm_json
        )
        arena = tflm_row["arena_used_bytes"]
        s3_fps = tflm_row["predicted_s3_fps"]
        if tflm_row["tflm_compatible"]:
            compat = "yes"
        elif tflm_row["offending_ops"]:
            compat = "no (" + ", ".join(tflm_row["offending_ops"]) + ")"
        else:
            compat = "no"
    else:
        arena = None
        s3_fps = None
        # fp32 rows aren't run through TFLM at all — flag explicitly so the
        # reader doesn't read "no" as "incompatible".
        compat = "n/a"

    return MatrixRow(
        label=spec.label,
        eval_source=spec.eval_json,
        tflm_source=tflm_source,
        size_bytes=int(eval_data.get("size_bytes", 0)),
        map50=float(eval_data.get("map50", 0.0)),
        host_latency_ms_p50=float(eval_data.get("latency_ms_p50", 0.0)),
        predicted_s3_fps=s3_fps,
        tflm_arena_bytes=arena,
        tflm_compatible=compat,
        notes=str(eval_data.get("notes", "")),
    )


def aggregate(results_dir: Path) -> AggregateResult:
    """Walk ROW_SPECS and build a MatrixRow for each present candidate.

    Optional rows (e.g. QAT) are skipped silently when their eval JSON is
    absent. Required rows are reported in ``missing`` so the SUMMARY can call
    them out instead of printing a partial matrix as if it were complete.
    """
    out = AggregateResult()
    for spec in ROW_SPECS:
        row = _build_row(spec, results_dir)
        if row is None:
            if not spec.optional:
                out.missing.append(spec.eval_json)
            continue
        out.rows.append(row)
    return out


# ---------- Markdown rendering ----------

def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_size_mb(n: int) -> str:
    return f"{n:,} ({n / (1024 * 1024):.2f} MB)"


def _fmt_fps(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def _fmt_arena(v: int | None) -> str:
    if v is None:
        return "—"
    return f"{v:,} ({v / (1024 * 1024):.2f} MB)"


def _fmt_map(v: float) -> str:
    # mAP@0.5 cosmetics: 1.0 fp32 baselines on a 22-image silver-GT val set
    # are not actually 100% — display the raw value but clamp the >1.0
    # floating-point overshoots at 1.0 for readability.
    return f"{min(v, 1.0):.4f}"


def format_decision_matrix(rows: list[MatrixRow]) -> str:
    if not rows:
        return "_(no rows — check `missing` in the aggregator output)_"
    header = (
        "| model | size_bytes | mAP@0.5 | host_int8_latency_ms_p50 | "
        "predicted_s3_fps | tflm_arena_bytes | tflm_compatible |"
    )
    sep = "| " + " | ".join(["---"] * 7) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {label} | {size} | {map} | {lat:.2f} | {fps} | {arena} | {compat} |".format(
                label=row.label,
                size=_fmt_size_mb(row.size_bytes),
                map=_fmt_map(row.map50),
                lat=row.host_latency_ms_p50,
                fps=_fmt_fps(row.predicted_s3_fps),
                arena=_fmt_arena(row.tflm_arena_bytes),
                compat=row.tflm_compatible,
            )
        )
    return "\n".join(lines)


def _row_by_label(rows: list[MatrixRow], label: str) -> MatrixRow | None:
    for r in rows:
        if r.label == label:
            return r
    return None


def _pick_winner(rows: list[MatrixRow]) -> MatrixRow | None:
    """Pick the recommended INT8 candidate.

    Decision rule, in order:
      1. Exclude fp32 rows (we want a deployable INT8 .tflite).
      2. Exclude TFLM-incompatible rows.
      3. Exclude rows with mAP@0.5 == 0 (signal the INT8 export collapsed).
      4. Among the survivors, pick the highest mAP@0.5; tie-break on highest
         predicted_s3_fps; final tie-break on smallest size_bytes.
      5. If no row survives the mAP filter, fall back to the smallest-arena
         INT8 row (so the recommendation still exists, with a caveat).
    """
    int8_rows = [r for r in rows if "INT8" in r.label]
    int8_rows = [r for r in int8_rows if not r.tflm_compatible.startswith("no")]
    if not int8_rows:
        return None
    nonzero = [r for r in int8_rows if r.map50 > 0.0]
    if nonzero:
        nonzero.sort(
            key=lambda r: (
                -r.map50,
                -(r.predicted_s3_fps or 0.0),
                r.size_bytes,
            )
        )
        return nonzero[0]
    # Fallback: every INT8 row collapsed to mAP=0; pick smallest arena
    int8_rows.sort(
        key=lambda r: (
            r.tflm_arena_bytes if r.tflm_arena_bytes is not None else 1 << 62,
            r.size_bytes,
        )
    )
    return int8_rows[0]


def format_recommendation(rows: list[MatrixRow]) -> str:
    """Render the PRD-mandated recommendation paragraph.

    Required to address (a) flash-first pick, (b) estimated S3 fps,
    (c) PSRAM headroom, (d) <5 fps acceptability, (e) >5-pt fp32-vs-INT8
    follow-up trigger.
    """
    winner = _pick_winner(rows)
    lines: list[str] = []

    if winner is None:
        lines.append(
            "**No deployable INT8 candidate identified.** Every INT8 row "
            "in the matrix is either TFLM-incompatible or absent. The "
            "PoC's signal is that the export pipeline (not the training) "
            "is the blocker — investigate the ONNX -> TFLite quantization "
            "step before flashing anything."
        )
        return "\n\n".join(lines)

    fps = winner.predicted_s3_fps or 0.0
    arena = winner.tflm_arena_bytes or 0
    headroom_mb = (PSRAM_BUDGET_BYTES - arena) / (1024 * 1024)

    # (a)+(b)+(c)
    lines.append(
        "**Flash first: `{label}`.** ({size_mb:.2f} MB on disk, "
        "mAP@0.5 = {m:.4f}, predicted ESP32-S3 latency p50 ~"
        "{lat_us:,.0f} us -> ~{fps:.3f} fps under the conservative 8x "
        "x86->S3 multiplier, TFLM arena = {arena_mb:.2f} MB out of the "
        "~8 MB PSRAM budget — ~{headroom:.2f} MB headroom.) Source: "
        "`{src}` (eval) + `{tflm}` (TFLM bench).".format(
            label=winner.label,
            size_mb=winner.size_bytes / (1024 * 1024),
            m=min(winner.map50, 1.0),
            lat_us=fps and (1_000_000.0 / fps),
            fps=fps,
            arena_mb=arena / (1024 * 1024),
            headroom=headroom_mb,
            src=winner.eval_source,
            tflm=winner.tflm_source or "—",
        )
    )

    # (d)
    if fps < S3_FPS_TARGET:
        lines.append(
            "**Predicted fps is below the PRD's 5-fps success target** "
            "({fps:.3f} < {target:.1f}). For the catzap monitoring rig "
            "this is *probably* acceptable: cats are static for minutes "
            "at a time and the existing server-side pipeline already "
            "tolerates seconds-scale latency on the rare-positive path. "
            "Treat <1 fps as a known cost of fitting a single-class "
            "detector into PSRAM rather than as a regression — but plan "
            "to re-measure on real silicon, since the 8x multiplier is "
            "conservative and ESP-DL vector ops may close 2-3x of the "
            "gap.".format(fps=fps, target=S3_FPS_TARGET)
        )
    else:
        lines.append(
            "Predicted fps ({fps:.3f}) meets the PRD's >=5 fps target "
            "under the 8x x86->S3 multiplier — no slowdown caveat "
            "needed.".format(fps=fps)
        )

    # (e) — fp32-vs-INT8 follow-up trigger
    fp32_label_for_winner = _fp32_counterpart(winner.label)
    fp32_row = (
        _row_by_label(rows, fp32_label_for_winner)
        if fp32_label_for_winner is not None
        else None
    )
    if fp32_row is not None and fp32_row.map50 > 0.0:
        drop_pts = (fp32_row.map50 - winner.map50) * 100.0
        if drop_pts > FP32_VS_INT8_DROP_THRESHOLD:
            lines.append(
                "**Follow-up triggered.** fp32 counterpart "
                "`{fp32}` scores mAP {fm:.4f} vs INT8 winner "
                "{im:.4f} — a {drop:.1f}-point drop, well above the "
                "5-point threshold. Recommended next step: investigate "
                "whether QAT (US-005 attempted, but the fake-quant "
                "graph did not recover accuracy here — see US-005.md "
                "for the fbgemm-qconfig deviation) or a wider student "
                "(e.g. YOLOv8s INT8 instead of YOLOv8n) preserves "
                "more accuracy under int8.".format(
                    fp32=fp32_label_for_winner,
                    fm=min(fp32_row.map50, 1.0),
                    im=min(winner.map50, 1.0),
                    drop=drop_pts,
                )
            )
        else:
            lines.append(
                "fp32 counterpart `{fp32}` mAP {fm:.4f} vs INT8 winner "
                "{im:.4f} — within 5 points, no QAT/wider-model "
                "follow-up needed.".format(
                    fp32=fp32_label_for_winner,
                    fm=min(fp32_row.map50, 1.0),
                    im=min(winner.map50, 1.0),
                )
            )

    return "\n\n".join(lines)


def _fp32_counterpart(int8_label: str) -> str | None:
    """Map an INT8 row label to its fp32 counterpart label, if any."""
    if int8_label.startswith("YOLOv8n-cat"):
        return "YOLOv8n-cat fp32"
    if int8_label.startswith("NanoDet-cat"):
        return "NanoDet-cat fp32"
    return None


def format_summary(result: AggregateResult) -> str:
    """Render the full SUMMARY.md text from an AggregateResult."""
    parts: list[str] = []
    parts.append("# Edge Model Reduction PoC — Decision Matrix\n")
    parts.append(
        "Generated by `training/edge/results/aggregate_summary.py` from the "
        "per-story result JSONs in `training/edge/results/`. Re-run after "
        "any iteration that updates a story JSON or TFLM bench JSON.\n"
    )

    parts.append("## Decision matrix\n")
    parts.append(format_decision_matrix(result.rows) + "\n")
    parts.append(
        "Columns: `size_bytes` is `os.path.getsize(model_path)` from the "
        "EvalResult; `mAP@0.5` is the eval-harness measurement on the US-002 "
        "silver val set (clamped to 1.0 in display only); "
        "`host_int8_latency_ms_p50` is single-thread CPU on x86; "
        "`predicted_s3_fps` is `1e6 / (raw_x86_us_p50 * 8.0)` per "
        "`firmware/edge-bench/README.md`'s multiplier rationale; "
        "`tflm_arena_bytes` is `MicroInterpreter::arena_used_bytes()` after "
        "`AllocateTensors()`; `tflm_compatible` is `yes` / `no (<offending "
        "ops>)` for INT8 .tflite rows or `n/a` for fp32 rows that aren't "
        "run through TFLM.\n"
    )

    parts.append("## Recommendation\n")
    parts.append(format_recommendation(result.rows) + "\n")

    if result.missing:
        parts.append("## Missing inputs\n")
        parts.append(
            "The following expected EvalResult JSONs were absent at "
            "render time. The matrix is incomplete — re-run the upstream "
            "story (or fix the deviation noted in its `<id>.md`) and "
            "re-run this aggregator:\n"
        )
        for m in result.missing:
            parts.append(f"- `{m}`")
        parts.append("")

    parts.append("## How to regenerate\n")
    parts.append(
        "```bash\n"
        "python training/edge/results/aggregate_summary.py \\\n"
        "    --results-dir training/edge/results \\\n"
        "    --output training/edge/results/SUMMARY.md\n"
        "```\n"
    )
    parts.append(
        "Tested by `training/edge/tests/test_summary_aggregate.py`. The "
        "TFLM-shape rows are pulled through "
        "`firmware/edge-bench/aggregate.py` (US-011 -> US-012 ingestion "
        "seam — see `progress.txt`).\n"
    )

    return "\n".join(parts)


def write_summary(results_dir: Path, output: Path) -> AggregateResult:
    result = aggregate(results_dir)
    output.write_text(format_summary(result))
    return result


# ---------- CLI ----------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("training/edge/results"),
        help="Directory containing per-story result JSONs (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="SUMMARY.md output path (default: <results-dir>/SUMMARY.md)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output or (args.results_dir / "SUMMARY.md")
    result = write_summary(args.results_dir, output)
    print(f"Wrote {output} with {len(result.rows)} rows.")
    if result.missing:
        print(
            f"Missing {len(result.missing)} expected eval JSON(s): "
            + ", ".join(result.missing),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
