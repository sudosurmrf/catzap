"""iter-B: pick the best Pareto point from the imgsz sweep.

Reads the per-imgsz rollups written by ``training/edge/yolo/imgsz_sweep.py``
(``training/edge/results/iter-B_imgsz_<N>.json``), prints a comparison
table, and writes ``training/edge/results/iter-B_summary.md`` picking the
"smallest imgsz that holds mAP within 1 point of iter-A AND improves
predicted-P4 latency by >= 15%" — the rule from the iter-B acceptance
criteria.

Usage::

    python training/edge/results/aggregate_iter_b.py

Pure helper — no DI seams; the sweep module is the unit-tested orchestrator.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("training/edge/results")

# 1 mAP-point on the [0,1] mAP scale.
MAP_TOLERANCE = 0.01
LATENCY_IMPROVE_PCT = 15.0


def load_rollups() -> list[dict]:
    """Load only the rollup JSONs (the ones with `imgsz` set as a top-level key).

    The sweep also writes per-imgsz eval and bench JSONs whose paths look
    similar (e.g. ``iter-B_imgsz_192_eval.json``,
    ``iter-B_imgsz_192_bench-tflm.json``) — those don't carry the rollup
    schema and must be filtered out.
    """
    paths = sorted(RESULTS.glob("iter-B_imgsz_*.json"))
    docs = []
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARN: skipping {p}: {e}")
            continue
        if "imgsz" not in d or "candidate_metrics" not in d:
            continue  # skip eval/bench sidecars
        docs.append(d)
    docs.sort(key=lambda d: int(d.get("imgsz", 0)))
    return docs


def pick_winner(rollups: list[dict]) -> dict | None:
    """Smallest imgsz that holds mAP within 1 point AND wins >=15% on latency.

    Returns the rollup dict, or None if no candidate qualifies.
    """
    qualifying: list[dict] = []
    for r in rollups:
        if r.get("status") != "passed":
            continue
        deltas = (r.get("pareto") or {}).get("deltas") or {}
        map_delta = deltas.get("map50", 0.0)
        latency_pct = deltas.get("latency_pct", 0.0)
        if map_delta < -MAP_TOLERANCE:
            continue
        if latency_pct > -LATENCY_IMPROVE_PCT:  # not faster by >=15%
            continue
        qualifying.append(r)
    qualifying.sort(key=lambda r: int(r.get("imgsz", 0)))
    return qualifying[0] if qualifying else None


def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(p) -> str:
    if p is None:
        return "-"
    return f"{float(p):+.2f}%"


def _fmt_map(m) -> str:
    if m is None:
        return "-"
    return f"{float(m):.4f}"


def format_table(rollups: list[dict]) -> str:
    """Markdown comparison table: one row per imgsz."""
    header = (
        "| imgsz | status | mAP@0.5 | size | arena | predicted P4 ms p50 | "
        "predicted P4 fps | mAP delta | size delta | arena delta | latency delta | verdict |\n"
        "|-------|--------|---------|------|-------|---------------------|"
        "------------------|-----------|------------|-------------|---------------|---------|\n"
    )
    rows = []
    for r in rollups:
        cm = r.get("candidate_metrics") or {}
        deltas = (r.get("pareto") or {}).get("deltas") or {}
        rows.append(
            "| {imgsz} | {status} | {map} | {size} | {arena} | "
            "{p4_ms:.2f} | {p4_fps:.3f} | "
            "{map_d} | {size_d} | {arena_d} | {lat_d} | {verdict} |".format(
                imgsz=r.get("imgsz", "?"),
                status=r.get("status", "?"),
                map=_fmt_map(cm.get("map50")),
                size=_fmt_int(cm.get("size_bytes")),
                arena=_fmt_int(cm.get("arena_used_bytes")),
                p4_ms=cm.get("predicted_p4_latency_ms_p50") or 0.0,
                p4_fps=cm.get("predicted_p4_fps") or 0.0,
                map_d=f"{deltas.get('map50', 0.0):+.4f}" if deltas else "-",
                size_d=_fmt_pct(deltas.get("size_pct")),
                arena_d=_fmt_pct(deltas.get("arena_pct")),
                lat_d=_fmt_pct(deltas.get("latency_pct")),
                verdict=(r.get("pareto") or {}).get("verdict", "-"),
            )
        )
    return header + "\n".join(rows) + "\n"


def format_summary(rollups: list[dict]) -> str:
    """Build the full iter-B_summary.md content."""
    if not rollups:
        return (
            "# iter-B summary\n\n"
            "**Status**: blocked — no per-imgsz rollups found at "
            "`training/edge/results/iter-B_imgsz_*.json`. Run the sweep:\n\n"
            "```sh\npython -m training.edge.yolo.imgsz_sweep\n```\n"
        )

    # Baseline numbers come from the first rollup's baseline_metrics — they
    # are identical across the sweep (every imgsz compares to the same iter-A).
    baseline = rollups[0].get("baseline_metrics") or {}
    winner = pick_winner(rollups)

    lines: list[str] = []
    lines.append("# iter-B — input resolution sweep on the iter-A frontier\n")
    lines.append(
        "**Baseline**: iter-A (per-channel-friendly INT8 PTQ on "
        "`yolov8n_cat_distilled.pt` at imgsz=224)\n"
    )
    lines.append(
        f"- iter-A mAP@0.5 = {_fmt_map(baseline.get('map50'))}\n"
        f"- iter-A size_bytes = {_fmt_int(baseline.get('size_bytes'))}\n"
        f"- iter-A arena_used_bytes = {_fmt_int(baseline.get('arena_used_bytes'))}\n"
        f"- iter-A predicted P4 latency ms p50 = "
        f"{(baseline.get('predicted_p4_latency_ms_p50') or 0.0):.2f}\n"
    )
    lines.append("\n## Per-imgsz comparison\n")
    lines.append(format_table(rollups))
    lines.append(
        "\n*Pareto thresholds (from `build_pareto_verdict`): mAP must hold; "
        "verdict=`dominates` requires latency >=15% OR size >=20% OR "
        "arena >=15% better than iter-A. The iter-B winner rule (below) is "
        "stricter than the dominates threshold — it requires the mAP "
        "to be within 1 mAP-point of iter-A AND latency >=15% better.*\n"
    )

    lines.append("\n## Winner — smallest imgsz with mAP held + latency >=15% better\n")
    if winner is None:
        # Find any rollup that holds mAP at all (regardless of latency). If
        # one exists but no latency win, recommend keeping iter-A's imgsz.
        held = [r for r in rollups if (r.get("pareto") or {}).get("deltas", {}).get("map50", 0.0) >= -MAP_TOLERANCE and r.get("status") == "passed"]
        if held:
            best_lat = min(held, key=lambda r: r["candidate_metrics"]["predicted_p4_latency_ms_p50"])
            lat_pct = (best_lat.get("pareto") or {}).get("deltas", {}).get("latency_pct", 0.0)
            lines.append(
                f"No imgsz beat iter-A by the required >=15% latency margin. "
                f"Best mAP-holding candidate is imgsz={best_lat['imgsz']} with "
                f"latency_delta={lat_pct:+.2f}% (mAP held within tolerance). "
                f"**Recommendation**: stay at iter-A's imgsz=224 — input "
                f"resolution is not the right efficiency lever for this "
                f"checkpoint without retraining at the smaller crop "
                f"(future iter-E or a dedicated retrain story).\n"
            )
        else:
            lines.append(
                "No imgsz qualifies — every smaller crop dropped mAP by more "
                "than 1 point. The .pt was trained at 224 and never saw "
                "smaller crops, so this is consistent with the "
                "imgsz_sweep.py docstring's 'expect mAP regression at "
                "imgsz<<224 because the model was never trained on smaller "
                "crops' note. **Recommendation**: stay at iter-A's "
                "imgsz=224 and revisit as a retraining story (iter-E or a "
                "future dedicated retrain at the imgsz target).\n"
            )
    else:
        wm = winner["candidate_metrics"]
        wd = (winner.get("pareto") or {}).get("deltas") or {}
        lines.append(
            f"**imgsz={winner['imgsz']}** is the smallest input resolution "
            f"that holds mAP within 1 mAP-point of iter-A (delta="
            f"{wd.get('map50', 0.0):+.4f}) and improves predicted-P4 "
            f"latency by {wd.get('latency_pct', 0.0):+.2f}% "
            f"(>= -{LATENCY_IMPROVE_PCT}% threshold). "
            f"Concrete numbers: mAP={_fmt_map(wm.get('map50'))}, "
            f"size={_fmt_int(wm.get('size_bytes'))} bytes, "
            f"arena={_fmt_int(wm.get('arena_used_bytes'))} bytes, "
            f"predicted_p4_latency_ms_p50={(wm.get('predicted_p4_latency_ms_p50') or 0.0):.2f}, "
            f"predicted_p4_fps={(wm.get('predicted_p4_fps') or 0.0):.3f}.\n"
        )
        lines.append(
            f"\n**Pareto verdict for the winner**: "
            f"`{(winner.get('pareto') or {}).get('verdict')}` "
            f"(size_pct={_fmt_pct(wd.get('size_pct'))}, "
            f"arena_pct={_fmt_pct(wd.get('arena_pct'))}, "
            f"latency_pct={_fmt_pct(wd.get('latency_pct'))}).\n"
        )
        lines.append(
            f"\n**Downstream effect**: iter-E (YOLOv8n width=0.75x retrain + KD "
            f"distill) should train at imgsz={winner['imgsz']} so the imgsz "
            f"and width reductions compound rather than re-litigating the "
            f"resolution decision.\n"
        )

    lines.append("\n## Files\n\n")
    for r in rollups:
        path = RESULTS / f"iter-B_imgsz_{r['imgsz']}.json"
        model = r.get("model_path", "?")
        lines.append(f"- `{path}` -> `{model}`\n")
    lines.append("\n## Reproduce\n\n")
    lines.append(
        "```sh\n"
        "# 1. Bootstrap labeled JPEGs + repair val manifest (per iter-A pattern):\n"
        "python -m training.edge.auto_label bootstrap --src-dir data/cat_photos --out-dir training/edge/data/labeled --target-frames 600 --augments-per-image 25 --seed 42\n"
        "python -m training.edge.make_dataset_manifest --dataset-dir training/edge/data/labeled\n"
        "python training/edge/data/regenerate_val_manifest.py\n"
        "\n"
        "# 2. Run the sweep (default arguments cover the AC list 192,224,256,288,320):\n"
        "python -m training.edge.yolo.imgsz_sweep\n"
        "\n"
        "# 3. Re-aggregate the markdown summary:\n"
        "python training/edge/results/aggregate_iter_b.py\n"
        "```\n"
    )
    return "".join(lines)


def main() -> int:
    rollups = load_rollups()
    md = format_summary(rollups)
    out = RESULTS / "iter-B_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out}")
    if rollups:
        print(format_table(rollups))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
