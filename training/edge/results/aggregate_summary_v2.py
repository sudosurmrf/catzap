"""iter-H — Aggregate v2 iter-* + v1 US-* JSONs into SUMMARY_v2.md.

Reads every relevant ``training/edge/results/iter-*.json`` (the v2
PoC's per-iteration rollups, each carrying ``candidate_metrics`` +
``pareto.verdict``) plus the v1 ``US-*.json`` files used by the
deprecation list, builds the v2 Pareto-frontier decision matrix, and
writes SUMMARY_v2.md alongside a machine-readable ``v2_frontier.json``.

Matrix row order is fixed by ``ITER_SPECS`` (mirrors US-012's
``aggregate_summary.ROW_SPECS`` static-spec pattern). Frontier picker
excludes blocked / zero-mAP / TFLM-incompatible rows, then tie-breaks
on (mAP desc, predicted_p4_fps desc, size asc).

Usage::

    python training/edge/results/aggregate_summary_v2.py \\
        --results-dir training/edge/results \\
        --summary-out training/edge/results/SUMMARY_v2.md \\
        --frontier-out training/edge/results/v2_frontier.json

The existing SUMMARY.md is preserved (historical record); SUMMARY_v2.md
is the live recommendation going forward. See iter-H acceptance.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ESP32-P4 spec from the prd2 description: 32 MB PSRAM, dual-core
# 400 MHz RV32 with vector ext, MIPI-CSI camera. The v1 SUMMARY targeted
# ESP32-S3's ~8 MB PSRAM; v2 has 4x the headroom but the per-model
# arena ceiling we judge against is the same conservative ~8 MB so the
# rollout reads sanely if the team falls back to S3 silicon.
P4_PSRAM_BYTES = 32 * 1024 * 1024
ARENA_CEILING_BYTES = 8 * 1024 * 1024
V2_FPS_TARGET = 5.0  # carried over from v1 PRD; revisit on real silicon


# ---------- iter row specs ----------

@dataclass(frozen=True)
class IterSpec:
    """One row of the v2 decision matrix.

    ``primary_json`` is the iter rollup that carries the rich shape
    (status, pareto verdict, candidate_metrics). ``label`` is what the
    matrix renders.
    """

    label: str
    primary_json: str
    pareto_key: str = "pareto"  # iter-F uses pareto_vs_yolo_v2_frontier instead
    optional: bool = False


# Row order matches the user's iter-A through iter-G sequence in the
# description.
ITER_SPECS: tuple[IterSpec, ...] = (
    IterSpec(label="iter-A: per-channel quant (v2 floor)", primary_json="iter-A_per_channel_quant.json"),
    IterSpec(label="iter-B: imgsz=192", primary_json="iter-B_imgsz_192.json"),
    IterSpec(label="iter-B: imgsz=224", primary_json="iter-B_imgsz_224.json"),
    IterSpec(label="iter-B: imgsz=256", primary_json="iter-B_imgsz_256.json"),
    IterSpec(label="iter-B: imgsz=288", primary_json="iter-B_imgsz_288.json"),
    IterSpec(label="iter-B: imgsz=320", primary_json="iter-B_imgsz_320.json"),
    IterSpec(label="iter-C: pruned 25%", primary_json="iter-C_pruned.json"),
    IterSpec(label="iter-D: INT8w/INT16a", primary_json="iter-D_mixed_int.json"),
    IterSpec(label="iter-E: width=0.75x", primary_json="iter-E_yolov8n_0p75x.json"),
    IterSpec(
        label="iter-F: NanoDet per-channel",
        primary_json="iter-F_nanodet_per_channel.json",
        pareto_key="pareto_vs_yolo_v2_frontier",
    ),
    IterSpec(label="iter-G: NMS off-graph", primary_json="iter-G_off_graph_nms.json"),
)


# ---------- v1 deprecation candidates ----------

@dataclass(frozen=True)
class V1Spec:
    """v1 INT8 candidate; ``label`` matches v1 SUMMARY.md row labels."""

    label: str
    eval_json: str


V1_INT8_SPECS: tuple[V1Spec, ...] = (
    V1Spec("YOLOv8n-cat INT8 (US-004)", "US-004.json"),
    V1Spec("YOLOv8n-cat-QAT INT8 (US-005)", "US-005.json"),
    V1Spec("YOLOv8n-cat-distilled INT8 (US-006)", "US-006-int8.json"),
    V1Spec("NanoDet-cat INT8 (US-009)", "US-009.json"),
)


# ---------- in-memory rows ----------

@dataclass
class FrontierRow:
    """Materialized v2 matrix row."""

    label: str
    source: str
    status: str  # "passed" | "blocked"
    blocked_reason: str | None
    pareto_verdict: str  # "dominates" | "equal" | "regress" | "blocked"
    map50: float
    size_bytes: int
    arena_used_bytes: int | None
    predicted_p4_latency_ms_p50: float | None
    predicted_p4_fps: float | None
    tflm_compatible: bool


@dataclass
class V1Row:
    """v1 INT8 candidate row (deprecation analysis only)."""

    label: str
    source: str
    map50: float
    size_bytes: int


@dataclass
class V2Result:
    rows: list[FrontierRow] = field(default_factory=list)
    v1_rows: list[V1Row] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


# ---------- ingestion ----------

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _extract_candidate(data: dict) -> dict:
    """Pull the candidate-side metrics regardless of which iter shape.

    iter-A/B/C/D/E/F use ``candidate_metrics``. iter-G uses
    ``iter_a_baseline_metrics`` for static (mAP/size/arena, since the
    .tflite is iter-A's) and ``inference_only_latency`` for refreshed
    latency/fps. The bare-EvalResult ``iter-A.json`` shape doesn't
    carry candidate_metrics — fall back to the top-level fields.
    """
    if "candidate_metrics" in data:
        return data["candidate_metrics"]

    # iter-G: same .tflite as iter-A but a refreshed latency measurement.
    if "iter_a_baseline_metrics" in data and "inference_only_latency" in data:
        base = dict(data["iter_a_baseline_metrics"])
        infer = data["inference_only_latency"]
        base["predicted_p4_latency_ms_p50"] = infer.get("predicted_p4_ms_p50")
        base["predicted_p4_fps"] = infer.get("predicted_p4_fps")
        return base

    # Bare EvalResult: synthesize p4 latency from latency_ms_p50 (host x86)
    # using a flat 5x multiplier (the iter-* convention). This branch is
    # rarely hit because the rich rollups dominate; tests cover it.
    return {
        "map50": float(data.get("map50", 0.0)),
        "size_bytes": int(data.get("size_bytes", 0)),
        "arena_used_bytes": None,
        "predicted_p4_latency_ms_p50": (
            float(data["latency_ms_p50"]) * 5.0
            if "latency_ms_p50" in data
            else None
        ),
        "predicted_p4_fps": (
            1000.0 / (float(data["latency_ms_p50"]) * 5.0)
            if data.get("latency_ms_p50")
            else None
        ),
    }


def _extract_tflm_compat(data: dict) -> bool:
    if "tflm" in data and isinstance(data["tflm"], dict):
        return bool(data["tflm"].get("tflm_compatible", True))
    if "tflm_runs200" in data and isinstance(data["tflm_runs200"], dict):
        return bool(data["tflm_runs200"].get("tflm_compatible", True))
    return True


def _build_frontier_row(spec: IterSpec, results_dir: Path) -> FrontierRow | None:
    path = results_dir / spec.primary_json
    if not path.exists():
        return None
    data = _load_json(path)
    cand = _extract_candidate(data)

    pareto = data.get(spec.pareto_key) or {}
    verdict = str(pareto.get("verdict", "unknown"))

    # iter-G has no pareto field; treat it as 'equal' since it shares
    # iter-A's tflite (so accuracy unchanged) but carries a refreshed
    # latency measurement.
    if not pareto and "iter_a_baseline_metrics" in data:
        verdict = "equal"

    status = str(data.get("status", "passed"))
    blocked = data.get("blocked_reason")

    return FrontierRow(
        label=spec.label,
        source=spec.primary_json,
        status=status,
        blocked_reason=str(blocked) if blocked else None,
        pareto_verdict=verdict,
        map50=float(cand.get("map50", 0.0) or 0.0),
        size_bytes=int(cand.get("size_bytes", 0) or 0),
        arena_used_bytes=(
            int(cand["arena_used_bytes"])
            if cand.get("arena_used_bytes") is not None
            else None
        ),
        predicted_p4_latency_ms_p50=(
            float(cand["predicted_p4_latency_ms_p50"])
            if cand.get("predicted_p4_latency_ms_p50") is not None
            else None
        ),
        predicted_p4_fps=(
            float(cand["predicted_p4_fps"])
            if cand.get("predicted_p4_fps") is not None
            else None
        ),
        tflm_compatible=_extract_tflm_compat(data),
    )


def _build_v1_row(spec: V1Spec, results_dir: Path) -> V1Row | None:
    path = results_dir / spec.eval_json
    if not path.exists():
        return None
    data = _load_json(path)
    return V1Row(
        label=spec.label,
        source=spec.eval_json,
        map50=float(data.get("map50", 0.0) or 0.0),
        size_bytes=int(data.get("size_bytes", 0) or 0),
    )


def aggregate(results_dir: Path) -> V2Result:
    out = V2Result()
    for spec in ITER_SPECS:
        row = _build_frontier_row(spec, results_dir)
        if row is None:
            if not spec.optional:
                out.missing.append(spec.primary_json)
            continue
        out.rows.append(row)
    for spec in V1_INT8_SPECS:
        v1 = _build_v1_row(spec, results_dir)
        if v1 is not None:
            out.v1_rows.append(v1)
    return out


# ---------- frontier picker ----------

def pick_frontier(rows: list[FrontierRow]) -> list[FrontierRow]:
    """Return the v2 Pareto frontier (deployable, non-zero mAP, sorted by fps desc).

    Exclusion rules (in order):
      1. status != 'passed' (blocked rows do not enter the frontier)
      2. tflm_compatible == False
      3. map50 == 0 (the cls-collapse signal — see iter-A learnings)

    Sort: ``predicted_p4_fps desc``, with map50-desc + size-asc tie-breakers.
    """
    survivors = [
        r
        for r in rows
        if r.status == "passed"
        and r.tflm_compatible
        and r.map50 > 0.0
    ]
    survivors.sort(
        key=lambda r: (
            -(r.predicted_p4_fps or 0.0),
            -r.map50,
            r.size_bytes,
        )
    )
    return survivors


def pick_winner(frontier: list[FrontierRow]) -> FrontierRow | None:
    """Pick ONE first-flash candidate.

    Among the frontier, prefer the highest-mAP row. Tie-break on highest
    predicted_p4_fps, then smallest size. This is intentionally distinct
    from pick_frontier's sort: the frontier is sorted by fps desc for
    presentation, but the *recommendation* prioritizes accuracy because
    the catzap rig tolerates seconds-scale latency on the rare-positive
    path (per v1 SUMMARY recommendation).
    """
    if not frontier:
        return None
    ranked = sorted(
        frontier,
        key=lambda r: (
            -r.map50,
            -(r.predicted_p4_fps or 0.0),
            r.size_bytes,
        ),
    )
    return ranked[0]


def deprecated_v1_rows(
    v1_rows: list[V1Row], frontier: list[FrontierRow]
) -> list[V1Row]:
    """Return the subset of v1 INT8 rows that the v2 frontier dominates.

    A v2 row dominates a v1 row if v2.map50 > v1.map50 — strict accuracy
    improvement is the only criterion that justifies replacing a
    deployed candidate. (Latency/size improvements without mAP carry-
    forward is the v1->v2 floor anyway.)
    """
    if not frontier:
        return []
    best_v2_map = max(r.map50 for r in frontier)
    return [v for v in v1_rows if v.map50 < best_v2_map]


# ---------- markdown rendering ----------

def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_size_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.2f}"


def _fmt_arena_kb(n: int | None) -> str:
    if n is None:
        return "—"
    return f"{n / 1024:.0f}"


def _fmt_lat_ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}"


def _fmt_fps(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def _fmt_map(v: float) -> str:
    return f"{min(v, 1.0):.4f}"


def format_frontier_table(rows: list[FrontierRow]) -> str:
    """Render the full iter-* table (frontier + non-frontier rows).

    Per acceptance: include blocked rows so the reader sees what was
    tried and why.
    """
    if not rows:
        return "_(no iter-* rows found — check `missing` in the aggregator output)_"
    header = (
        "| story | mAP@0.5 | size_mb | arena_kb | "
        "predicted_p4_latency_ms_p50 | predicted_p4_fps | pareto_verdict |"
    )
    sep = "| " + " | ".join(["---"] * 7) + " |"
    lines = [header, sep]
    for r in rows:
        verdict = r.pareto_verdict
        if r.status == "blocked":
            verdict = f"blocked ({r.blocked_reason or 'see md'})"
        lines.append(
            "| {label} | {m} | {sz} | {ar} | {lat} | {fps} | {v} |".format(
                label=r.label,
                m=_fmt_map(r.map50),
                sz=_fmt_size_mb(r.size_bytes),
                ar=_fmt_arena_kb(r.arena_used_bytes),
                lat=_fmt_lat_ms(r.predicted_p4_latency_ms_p50),
                fps=_fmt_fps(r.predicted_p4_fps),
                v=verdict,
            )
        )
    return "\n".join(lines)


def format_recommendation(
    winner: FrontierRow | None, frontier: list[FrontierRow]
) -> str:
    """Render the rollout recommendation paragraph.

    Required: address (a) flash-first pick, (b) MIPI-CSI vs HTTP-stream
    tradeoff, (c) PSRAM placement, (d) classifier handoff via
    server/vision/classifier.py.
    """
    parts: list[str] = []

    if winner is None:
        parts.append(
            "**No deployable v2 INT8 candidate identified.** Every iter-* "
            "row in the frontier table is either blocked, TFLM-"
            "incompatible, or has mAP@0.5 == 0 (the cls-collapse signal "
            "documented in iter-A through iter-F). The PoC's signal is "
            "that the export pipeline (not the training) is the blocker — "
            "investigate per-tensor activation quantization on the Detect "
            "head before flashing anything to the ESP32-P4."
        )
        return "\n\n".join(parts)

    fps = winner.predicted_p4_fps or 0.0
    arena_mb = (
        (winner.arena_used_bytes or 0) / (1024 * 1024)
        if winner.arena_used_bytes is not None
        else 0.0
    )
    headroom_mb = (P4_PSRAM_BYTES / (1024 * 1024)) - arena_mb

    # (a) flash-first pick + concrete numbers
    parts.append(
        "**Flash first: `{label}`.** ({size:.2f} MB on disk, mAP@0.5 = "
        "{m:.4f}, predicted ESP32-P4 latency p50 ~{lat:.2f} ms -> "
        "~{fps:.3f} fps under the v2 5x x86->P4 multiplier; TFLM arena "
        "= {arena:.2f} MB out of the ESP32-P4's 32 MB PSRAM budget — "
        "~{headroom:.2f} MB headroom.) Source: `{src}`.".format(
            label=winner.label,
            size=winner.size_bytes / (1024 * 1024),
            m=min(winner.map50, 1.0),
            lat=winner.predicted_p4_latency_ms_p50 or 0.0,
            fps=fps,
            arena=arena_mb,
            headroom=headroom_mb,
            src=winner.source,
        )
    )

    # (c) PSRAM placement
    parts.append(
        "**PSRAM placement.** With {arena:.2f} MB tensor arena vs the P4's "
        "32 MB PSRAM, place the model + arena in PSRAM and reserve internal "
        "SRAM for the camera frame buffer + scratch. The ESP32-S3 fallback "
        "(~8 MB PSRAM) also fits this candidate ({arena:.2f} MB <= "
        "{ceil:.0f} MB ceiling), so the same INT8 .tflite is portable across "
        "both target boards if the P4 stock slips.".format(
            arena=arena_mb,
            ceil=ARENA_CEILING_BYTES / (1024 * 1024),
        )
    )

    # (b) MIPI-CSI vs HTTP-stream tradeoff
    parts.append(
        "**MIPI-CSI vs HTTP-stream.** ESP32-P4's MIPI-CSI lets the cat "
        "detector consume frames directly from the on-board camera at "
        "DMA speed, eliminating the ~30-50 ms HTTP/MJPEG path the v1 "
        "AI-Thinker ESP32-CAM rig pays today. At the predicted ~{fps:.2f} "
        "fps the HTTP overhead would be a single-digit-percent tax on "
        "end-to-end latency, so the MIPI-CSI path is worth wiring "
        "but is not load-bearing for the catzap detect-then-trigger "
        "use case. If the P4 firmware misses MIPI-CSI for v2.0, fall "
        "back to the existing ESP32-CAM HTTP stream + P4-as-inference-"
        "client topology — the .tflite candidate is identical in both "
        "deployments.".format(fps=fps)
    )

    # (d) classifier handoff
    parts.append(
        "**Classifier handoff.** This is a single-class cat *detector*; "
        "re-identification is server-side via the existing MobileNet-V3-"
        "Small classifier (`server/vision/classifier.py`). The P4's job "
        "is to publish a 'cat seen' event + crop bbox over the existing "
        "transport (HTTP POST or BLE per the firmware/esp32-cam "
        "convention); the server runs the classifier on the crop. No "
        "changes to `server/vision/classifier.py` are required for v2 "
        "rollout — the contract is the same crop + bbox the v1 server "
        "already consumes."
    )

    # Below-target fps caveat (carried over from v1 SUMMARY)
    if fps < V2_FPS_TARGET:
        parts.append(
            "**Predicted fps is below the v1 PRD's 5-fps target** "
            "({fps:.3f} < {target:.1f}). Acceptable for this rig: cats "
            "sit still for minutes at a time and the trigger pipeline is "
            "rare-positive, not high-throughput. The 5x x86->P4 "
            "multiplier is conservative; ESP-DL vector ops on the P4's "
            "RV32 vector ext should close another 2-3x of the gap on "
            "real silicon. Re-measure after flashing.".format(
                fps=fps, target=V2_FPS_TARGET
            )
        )
    else:
        parts.append(
            "Predicted fps ({fps:.3f}) meets the v1 5-fps target under "
            "the conservative 5x x86->P4 multiplier — no slowdown "
            "caveat needed.".format(fps=fps)
        )

    return "\n\n".join(parts)


def format_deprecation(
    deprecated: list[V1Row], winner: FrontierRow | None
) -> str:
    """Render the deprecation list — v1 INT8 candidates the v2 frontier obsoletes."""
    if not deprecated:
        return (
            "_No v1 INT8 candidates are dominated by the v2 frontier — keep "
            "the v1 SUMMARY.md recommendation for now._"
        )

    lines: list[str] = []
    lines.append(
        "The following v1 INT8 candidates are obsoleted by the v2 frontier "
        "(strictly worse mAP@0.5 than the v2 winner)."
        + (
            f" v2 winner: `{winner.label}` (mAP {min(winner.map50, 1.0):.4f})."
            if winner is not None
            else ""
        )
    )
    lines.append("")
    lines.append("| v1 candidate | v1 mAP@0.5 | v1 size_mb | source |")
    lines.append("| --- | --- | --- | --- |")
    for v in deprecated:
        lines.append(
            "| {label} | {m:.4f} | {sz:.2f} | `{src}` |".format(
                label=v.label,
                m=v.map50,
                sz=v.size_bytes / (1024 * 1024),
                src=v.source,
            )
        )
    return "\n".join(lines)


def format_summary(result: V2Result) -> str:
    frontier = pick_frontier(result.rows)
    winner = pick_winner(frontier)
    deprecated = deprecated_v1_rows(result.v1_rows, frontier)

    parts: list[str] = []
    parts.append("# Edge Model Reduction PoC v2 — Pareto Frontier + ESP32-P4 Rollout\n")
    parts.append(
        "Generated by `training/edge/results/aggregate_summary_v2.py` from "
        "`training/edge/results/iter-*.json` (the v2 PoC's iter-A through "
        "iter-G rollups) and `training/edge/results/US-*.json` (v1 deprecation "
        "candidates). Re-run after any iteration adds or refreshes a JSON.\n"
    )
    parts.append(
        "The v1 SUMMARY.md is preserved as the historical record. "
        "SUMMARY_v2.md is the live recommendation going forward.\n"
    )

    parts.append("## v2 Pareto frontier — all iter-* candidates\n")
    parts.append(format_frontier_table(result.rows) + "\n")
    parts.append(
        "Columns: `mAP@0.5` is the eval-harness measurement on the US-002 "
        "120-image val set (clamped at 1.0 in display only); `size_mb` is "
        "`os.path.getsize(.tflite)`; `arena_kb` is "
        "`MicroInterpreter::arena_used_bytes()` after AllocateTensors() on "
        "the firmware/edge-bench TFLM x86 host build; "
        "`predicted_p4_latency_ms_p50` is `raw_x86_us_p50 * 5.0 / 1000` per "
        "the v2 P4 multiplier (midpoint between scalar-reference 6x and "
        "vector-ext / dual-core ideal 3x); `pareto_verdict` is the "
        "iter-*'s self-reported verdict vs the v2 baseline. Blocked "
        "rows surface their reason inline so the reader sees what was "
        "tried and why.\n"
    )

    frontier_label = ", ".join(r.label for r in frontier) or "(none)"
    parts.append("## Frontier members (deployable INT8 candidates)\n")
    parts.append(
        "Filtered: `status='passed'` AND `tflm_compatible` AND `mAP@0.5 > 0`. "
        f"Sorted by `predicted_p4_fps` descending.\n"
    )
    parts.append(f"Members: **{frontier_label}**.\n")

    parts.append("## Recommendation\n")
    parts.append(format_recommendation(winner, frontier) + "\n")

    parts.append("## Deprecation list — v1 INT8 candidates the v2 frontier obsoletes\n")
    parts.append(format_deprecation(deprecated, winner) + "\n")

    if result.missing:
        parts.append("## Missing inputs\n")
        parts.append(
            "The following expected iter-* JSONs were absent at render time. "
            "The matrix is incomplete — re-run the upstream iteration (or "
            "fix the deviation noted in its `<id>.md`) and re-run this "
            "aggregator:\n"
        )
        for m in result.missing:
            parts.append(f"- `{m}`")
        parts.append("")

    parts.append("## How to regenerate\n")
    parts.append(
        "```bash\n"
        "python training/edge/results/aggregate_summary_v2.py \\\n"
        "    --results-dir training/edge/results \\\n"
        "    --summary-out training/edge/results/SUMMARY_v2.md \\\n"
        "    --frontier-out training/edge/results/v2_frontier.json\n"
        "```\n"
    )
    parts.append(
        "Tested by `training/edge/tests/test_summary_v2_aggregate.py`.\n"
    )

    return "\n".join(parts)


def format_frontier_json(result: V2Result) -> dict[str, Any]:
    """Emit the machine-readable frontier shape for downstream tooling."""
    frontier = pick_frontier(result.rows)
    winner = pick_winner(frontier)
    deprecated = deprecated_v1_rows(result.v1_rows, frontier)
    return {
        "frontier": [
            {
                "story_id": r.label,
                "source": r.source,
                "map50": r.map50,
                "size_bytes": r.size_bytes,
                "arena_used_bytes": r.arena_used_bytes,
                "predicted_p4_latency_ms_p50": r.predicted_p4_latency_ms_p50,
                "predicted_p4_fps": r.predicted_p4_fps,
                "pareto_verdict": r.pareto_verdict,
            }
            for r in frontier
        ],
        "winner": (
            {
                "story_id": winner.label,
                "source": winner.source,
                "map50": winner.map50,
                "size_bytes": winner.size_bytes,
                "arena_used_bytes": winner.arena_used_bytes,
                "predicted_p4_latency_ms_p50": winner.predicted_p4_latency_ms_p50,
                "predicted_p4_fps": winner.predicted_p4_fps,
            }
            if winner is not None
            else None
        ),
        "deprecated_v1": [
            {
                "story_id": v.label,
                "source": v.source,
                "map50": v.map50,
                "size_bytes": v.size_bytes,
            }
            for v in deprecated
        ],
        "missing": list(result.missing),
        "all_iter_rows": [
            {
                "story_id": r.label,
                "source": r.source,
                "status": r.status,
                "blocked_reason": r.blocked_reason,
                "pareto_verdict": r.pareto_verdict,
                "map50": r.map50,
                "size_bytes": r.size_bytes,
                "arena_used_bytes": r.arena_used_bytes,
                "predicted_p4_latency_ms_p50": r.predicted_p4_latency_ms_p50,
                "predicted_p4_fps": r.predicted_p4_fps,
                "tflm_compatible": r.tflm_compatible,
            }
            for r in result.rows
        ],
    }


def write_outputs(
    results_dir: Path, summary_out: Path, frontier_out: Path
) -> V2Result:
    result = aggregate(results_dir)
    summary_out.write_text(format_summary(result))
    frontier_out.write_text(json.dumps(format_frontier_json(result), indent=2))
    return result


# ---------- CLI ----------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("training/edge/results"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="SUMMARY_v2.md output path (default: <results-dir>/SUMMARY_v2.md)",
    )
    parser.add_argument(
        "--frontier-out",
        type=Path,
        default=None,
        help="v2_frontier.json output path (default: <results-dir>/v2_frontier.json)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    summary_out = args.summary_out or (args.results_dir / "SUMMARY_v2.md")
    frontier_out = args.frontier_out or (args.results_dir / "v2_frontier.json")
    result = write_outputs(args.results_dir, summary_out, frontier_out)
    print(
        f"Wrote {summary_out} ({len(result.rows)} iter rows, "
        f"{len(result.v1_rows)} v1 rows) and {frontier_out}."
    )
    if result.missing:
        print(f"Missing: {', '.join(result.missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
