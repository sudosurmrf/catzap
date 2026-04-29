"""iter-A aggregator: combine eval JSON + TFLM bench JSON into the final
training/edge/results/iter-A_per_channel_quant.json with explicit Pareto
verdict vs the v1 INT8 frontier (US-006-int8 + US-011-yolov8n-distilled).

Usage:
    python training/edge/results/aggregate_iter_a.py

Reads:
    training/edge/results/iter-A.json                    (this iter's eval)
    training/edge/results/iter-A-tflm.json               (this iter's TFLM bench)
    training/edge/results/US-006-int8.json               (baseline eval)
    training/edge/results/US-011-yolov8n-distilled-tflm.json (baseline TFLM)

Writes:
    training/edge/results/iter-A_per_channel_quant.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make repo root imports work when run as a plain script.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.edge.yolo.per_channel_quant import (  # noqa: E402
    DEFAULT_P4_MULTIPLIER,
    DEFAULT_P4_MULTIPLIER_SOURCE,
    build_pareto_verdict,
)

RESULTS = Path("training/edge/results")
OUT_PATH = RESULTS / "iter-A_per_channel_quant.json"
EVAL_PATH = RESULTS / "iter-A.json"
TFLM_PATH = RESULTS / "iter-A-tflm.json"

# Baseline = v1 INT8 frontier (KD-distilled YOLOv8n).
BASELINE_EVAL_PATH = RESULTS / "US-006-int8.json"
BASELINE_TFLM_PATH = RESULTS / "US-011-yolov8n-distilled-tflm.json"
BASELINE_STORY = "US-006-int8 (v1 KD distilled INT8 frontier)"


def _safe_load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARN: failed to load {p}: {e}", file=sys.stderr)
        return None


def _x86_us_to_p4_ms(us: float, multiplier: float) -> float:
    """Convert raw x86 microseconds to predicted ESP32-P4 milliseconds."""
    return float(us) * float(multiplier) / 1000.0


def aggregate(
    *,
    multiplier: float = DEFAULT_P4_MULTIPLIER,
    multiplier_source: str = DEFAULT_P4_MULTIPLIER_SOURCE,
) -> dict:
    eval_result = _safe_load(EVAL_PATH)
    tflm_result = _safe_load(TFLM_PATH)
    base_eval = _safe_load(BASELINE_EVAL_PATH)
    base_tflm = _safe_load(BASELINE_TFLM_PATH)

    blocked_reason: str | None = None
    if eval_result is None:
        blocked_reason = (
            f"eval JSON missing: run `python -m training.edge.eval.run_eval "
            f"--model training/edge/models/yolov8n_cat_distilled_int8_pc.tflite "
            f"--format tflite_int8 --story-id iter-A --val-dir "
            f"training/edge/data/labeled/val/`"
        )
    elif tflm_result is None:
        blocked_reason = (
            "TFLM bench JSON missing: run "
            "`python firmware/edge-bench/run_bench.py --model "
            "training/edge/models/yolov8n_cat_distilled_int8_pc.tflite "
            "--story-id iter-A --runs 50` (build edgebench first if absent)"
        )
    elif base_eval is None or base_tflm is None:
        blocked_reason = (
            "baseline JSONs (US-006-int8 or US-011-yolov8n-distilled-tflm) missing"
        )

    if blocked_reason is not None:
        # Best-effort partial aggregate so the file IS committed even when blocked.
        verdict = build_pareto_verdict(
            candidate_map50=(eval_result or {}).get("map50", 0.0),
            candidate_size_bytes=(eval_result or {}).get("size_bytes", 0)
            or (tflm_result or {}).get("model_size_bytes", 0),
            candidate_arena_used_bytes=(tflm_result or {}).get("arena_used_bytes", 0),
            candidate_predicted_p4_latency_ms_p50=_x86_us_to_p4_ms(
                (tflm_result or {}).get("raw_x86_us_p50", 0.0), multiplier
            ),
            baseline_story=BASELINE_STORY,
            baseline_map50=(base_eval or {}).get("map50", 0.0),
            baseline_size_bytes=(base_eval or {}).get("size_bytes", 0),
            baseline_arena_used_bytes=(base_tflm or {}).get("arena_used_bytes", 0),
            baseline_predicted_p4_latency_ms_p50=_x86_us_to_p4_ms(
                (base_tflm or {}).get("raw_x86_us_p50", 0.0), multiplier
            ),
            blocked_reason=blocked_reason,
        )
    else:
        verdict = build_pareto_verdict(
            candidate_map50=eval_result["map50"],
            candidate_size_bytes=eval_result["size_bytes"],
            candidate_arena_used_bytes=tflm_result["arena_used_bytes"],
            candidate_predicted_p4_latency_ms_p50=_x86_us_to_p4_ms(
                tflm_result["raw_x86_us_p50"], multiplier
            ),
            baseline_story=BASELINE_STORY,
            baseline_map50=base_eval["map50"],
            baseline_size_bytes=base_eval["size_bytes"],
            baseline_arena_used_bytes=base_tflm["arena_used_bytes"],
            baseline_predicted_p4_latency_ms_p50=_x86_us_to_p4_ms(
                base_tflm["raw_x86_us_p50"], multiplier
            ),
        )

    aggregate_doc = {
        "story_id": "iter-A",
        "title": (
            "Per-channel output quantization fix: float32 OUTPUT dequant "
            "preserves cls precision through INT8 PTQ"
        ),
        "status": "blocked" if blocked_reason else "passed",
        "blocked_reason": blocked_reason,
        "model_path": (eval_result or tflm_result or {}).get(
            "model_path", "training/edge/models/yolov8n_cat_distilled_int8_pc.tflite"
        ),
        "p4_multiplier": multiplier,
        "p4_multiplier_source": multiplier_source,
        # Eval metrics (mAP, host-CPU INT8 latency).
        "eval": eval_result,
        # TFLM bench metrics (arena, predicted ESP32-P4 latency).
        "tflm": tflm_result,
        # Pareto verdict vs baseline.
        "pareto": verdict,
        # Convenience fields the SUMMARY_v2 aggregator (iter-H) reads.
        "candidate_metrics": {
            "map50": (eval_result or {}).get("map50", 0.0),
            "size_bytes": (eval_result or {}).get("size_bytes", 0),
            "arena_used_bytes": (tflm_result or {}).get("arena_used_bytes", 0),
            "predicted_p4_latency_ms_p50": _x86_us_to_p4_ms(
                (tflm_result or {}).get("raw_x86_us_p50", 0.0), multiplier
            ),
            "predicted_p4_fps": (
                1000.0
                / _x86_us_to_p4_ms(tflm_result["raw_x86_us_p50"], multiplier)
                if tflm_result and tflm_result.get("raw_x86_us_p50", 0) > 0
                else 0.0
            ),
        },
        "baseline_metrics": {
            "map50": (base_eval or {}).get("map50", 0.0),
            "size_bytes": (base_eval or {}).get("size_bytes", 0),
            "arena_used_bytes": (base_tflm or {}).get("arena_used_bytes", 0),
            "predicted_p4_latency_ms_p50": _x86_us_to_p4_ms(
                (base_tflm or {}).get("raw_x86_us_p50", 0.0), multiplier
            ),
        },
    }
    return aggregate_doc


def main() -> int:
    doc = aggregate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    print(f"verdict: {doc['pareto']['verdict']}")
    if doc["pareto"].get("deltas"):
        d = doc["pareto"]["deltas"]
        print(
            f"  mAP delta: {d['map50']:+.4f}, "
            f"size: {d['size_pct']:+.2f}%, "
            f"arena: {d['arena_pct']:+.2f}%, "
            f"latency: {d['latency_pct']:+.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
