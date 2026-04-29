"""iter-D aggregator: combine eval JSON + TFLM bench JSON into the final
training/edge/results/iter-D_mixed_int.json with explicit Pareto verdict
vs the v2 INT8 frontier (iter-A).

Usage:
    python training/edge/results/aggregate_iter_d.py

Reads:
    training/edge/results/iter-D.json                    (this iter's eval)
    training/edge/results/iter-D-tflm.json               (this iter's TFLM bench)
    training/edge/results/iter-A_per_channel_quant.json  (frontier baseline)

Writes:
    training/edge/results/iter-D_mixed_int.json
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
OUT_PATH = RESULTS / "iter-D_mixed_int.json"
EVAL_PATH = RESULTS / "iter-D.json"
TFLM_PATH = RESULTS / "iter-D-tflm.json"

# Baseline = iter-A frontier (per-channel + float-output INT8 KD distilled).
ITER_A_AGG_PATH = RESULTS / "iter-A_per_channel_quant.json"
BASELINE_STORY = "iter-A (per-channel float-output INT8 KD distilled)"


def _safe_load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARN: failed to load {p}: {e}", file=sys.stderr)
        return None


def _x86_us_to_p4_ms(us: float, multiplier: float) -> float:
    return float(us) * float(multiplier) / 1000.0


def _baseline_metrics_from_iter_a(iter_a_agg: dict) -> dict:
    """Pull iter-A's already-computed candidate_metrics — that's the contract.

    Per progress.txt pattern "baseline_metrics uses iter-A's candidate_metrics,
    NOT US-006-int8". iter-A is the v2 accuracy floor; iter-D / iter-E / etc
    compare against it.
    """
    cm = iter_a_agg.get("candidate_metrics", {})
    return {
        "map50": cm.get("map50", 0.0),
        "size_bytes": cm.get("size_bytes", 0),
        "arena_used_bytes": cm.get("arena_used_bytes", 0),
        "predicted_p4_latency_ms_p50": cm.get("predicted_p4_latency_ms_p50", 0.0),
    }


def aggregate(
    *,
    multiplier: float = DEFAULT_P4_MULTIPLIER,
    multiplier_source: str = DEFAULT_P4_MULTIPLIER_SOURCE,
) -> dict:
    eval_result = _safe_load(EVAL_PATH)
    tflm_result = _safe_load(TFLM_PATH)
    iter_a_agg = _safe_load(ITER_A_AGG_PATH)

    blocked_reason: str | None = None
    if eval_result is None:
        blocked_reason = (
            "eval JSON missing: run `python -m training.edge.eval.run_eval "
            "--model training/edge/models/yolov8n_cat_distilled_int8w_int16a.tflite "
            "--format tflite_int8 --story-id iter-D --val-dir "
            "training/edge/data/labeled/val/`"
        )
    elif tflm_result is None:
        blocked_reason = (
            "TFLM bench JSON missing: run "
            "`python firmware/edge-bench/run_bench.py --model "
            "training/edge/models/yolov8n_cat_distilled_int8w_int16a.tflite "
            "--story-id iter-D --runs 50` (build edgebench first if absent)"
        )
    elif iter_a_agg is None:
        blocked_reason = (
            "iter-A aggregate JSON missing: run "
            "`python training/edge/results/aggregate_iter_a.py` first"
        )

    base = _baseline_metrics_from_iter_a(iter_a_agg or {})

    if blocked_reason is not None:
        verdict = build_pareto_verdict(
            candidate_map50=(eval_result or {}).get("map50", 0.0),
            candidate_size_bytes=(eval_result or {}).get("size_bytes", 0)
            or (tflm_result or {}).get("model_size_bytes", 0),
            candidate_arena_used_bytes=(tflm_result or {}).get("arena_used_bytes", 0),
            candidate_predicted_p4_latency_ms_p50=_x86_us_to_p4_ms(
                (tflm_result or {}).get("raw_x86_us_p50", 0.0), multiplier
            ),
            baseline_story=BASELINE_STORY,
            baseline_map50=base["map50"],
            baseline_size_bytes=base["size_bytes"],
            baseline_arena_used_bytes=base["arena_used_bytes"],
            baseline_predicted_p4_latency_ms_p50=base["predicted_p4_latency_ms_p50"],
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
            baseline_map50=base["map50"],
            baseline_size_bytes=base["size_bytes"],
            baseline_arena_used_bytes=base["arena_used_bytes"],
            baseline_predicted_p4_latency_ms_p50=base["predicted_p4_latency_ms_p50"],
        )

    candidate_p4_ms = _x86_us_to_p4_ms(
        (tflm_result or {}).get("raw_x86_us_p50", 0.0), multiplier
    )
    aggregate_doc = {
        "story_id": "iter-D",
        "title": (
            "Mixed INT8/INT16 quantization: INT8 weights + INT16 activations "
            "via TFLite EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8"
        ),
        "status": "blocked" if blocked_reason else "passed",
        "blocked_reason": blocked_reason,
        "model_path": (eval_result or tflm_result or {}).get(
            "model_path",
            "training/edge/models/yolov8n_cat_distilled_int8w_int16a.tflite",
        ),
        "p4_multiplier": multiplier,
        "p4_multiplier_source": multiplier_source,
        "chosen_path": (
            "Path (1) — full INT16 activations + INT8 weights via TF builtin "
            "EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8. "
            "Path (2) (head-only INT16 via two-binary firmware loader) was "
            "deferred per PRD technical-notes guidance."
        ),
        "eval": eval_result,
        "tflm": tflm_result,
        "pareto": verdict,
        "candidate_metrics": {
            "map50": (eval_result or {}).get("map50", 0.0),
            "size_bytes": (eval_result or {}).get("size_bytes", 0),
            "arena_used_bytes": (tflm_result or {}).get("arena_used_bytes", 0),
            "predicted_p4_latency_ms_p50": candidate_p4_ms,
            "predicted_p4_fps": (
                1000.0 / candidate_p4_ms if candidate_p4_ms > 0 else 0.0
            ),
        },
        "baseline_metrics": base,
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
