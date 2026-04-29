"""iter-F aggregator: combine eval JSON + TFLM bench JSON into the final
training/edge/results/iter-F_nanodet_per_channel.json with TWO Pareto verdicts:

  1. vs US-009 (prior NanoDet INT8) — same-architecture comparison.
  2. vs iter-A (v2 YOLOv8n frontier) — cross-architecture comparison.

Mirrors the shape of aggregate_iter_a.py from iter-A.

Usage:
    python training/edge/results/aggregate_iter_f.py

Reads:
    training/edge/results/iter-F.json                       (this iter's eval)
    training/edge/results/iter-F-tflm.json                  (this iter's TFLM bench)
    training/edge/results/US-009.json                       (NanoDet INT8 baseline eval)
    training/edge/results/US-011-nanodet-tflm.json          (NanoDet INT8 baseline TFLM)
    training/edge/results/iter-A.json                       (YOLOv8n v2 frontier eval)
    training/edge/results/iter-A-tflm.json                  (YOLOv8n v2 frontier TFLM)

Writes:
    training/edge/results/iter-F_nanodet_per_channel.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.edge.yolo.per_channel_quant import (  # noqa: E402
    DEFAULT_P4_MULTIPLIER,
    DEFAULT_P4_MULTIPLIER_SOURCE,
    build_pareto_verdict,
)

RESULTS = Path("training/edge/results")
OUT_PATH = RESULTS / "iter-F_nanodet_per_channel.json"
EVAL_PATH = RESULTS / "iter-F.json"
TFLM_PATH = RESULTS / "iter-F-tflm.json"

# Same-architecture baseline = US-009 (NanoDet INT8 with int8-output cls collapse).
NANODET_EVAL_PATH = RESULTS / "US-009.json"
NANODET_TFLM_PATH = RESULTS / "US-011-nanodet-tflm.json"
NANODET_BASELINE = "US-009 (v1 NanoDet INT8 with int8-output cls collapse)"

# Cross-architecture baseline = iter-A (v2 YOLOv8n frontier).
YOLO_EVAL_PATH = RESULTS / "iter-A.json"
YOLO_TFLM_PATH = RESULTS / "iter-A-tflm.json"
YOLO_BASELINE = "iter-A (v2 YOLOv8n frontier — KD distilled INT8 with float-output dequant)"


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


def _verdict_against(
    *,
    eval_result: dict | None,
    tflm_result: dict | None,
    base_eval: dict | None,
    base_tflm: dict | None,
    baseline_label: str,
    multiplier: float,
    blocked_reason: str | None,
) -> dict:
    return build_pareto_verdict(
        candidate_map50=(eval_result or {}).get("map50", 0.0),
        candidate_size_bytes=(eval_result or {}).get("size_bytes", 0)
        or (tflm_result or {}).get("model_size_bytes", 0),
        candidate_arena_used_bytes=(tflm_result or {}).get("arena_used_bytes", 0),
        candidate_predicted_p4_latency_ms_p50=_x86_us_to_p4_ms(
            (tflm_result or {}).get("raw_x86_us_p50", 0.0), multiplier
        ),
        baseline_story=baseline_label,
        baseline_map50=(base_eval or {}).get("map50", 0.0),
        baseline_size_bytes=(base_eval or {}).get("size_bytes", 0),
        baseline_arena_used_bytes=(base_tflm or {}).get("arena_used_bytes", 0),
        baseline_predicted_p4_latency_ms_p50=_x86_us_to_p4_ms(
            (base_tflm or {}).get("raw_x86_us_p50", 0.0), multiplier
        ),
        blocked_reason=blocked_reason,
    )


def aggregate(
    *,
    multiplier: float = DEFAULT_P4_MULTIPLIER,
    multiplier_source: str = DEFAULT_P4_MULTIPLIER_SOURCE,
) -> dict:
    eval_result = _safe_load(EVAL_PATH)
    tflm_result = _safe_load(TFLM_PATH)
    nanodet_eval = _safe_load(NANODET_EVAL_PATH)
    nanodet_tflm = _safe_load(NANODET_TFLM_PATH)
    yolo_eval = _safe_load(YOLO_EVAL_PATH)
    yolo_tflm = _safe_load(YOLO_TFLM_PATH)

    blocked_reason: str | None = None
    if eval_result is None:
        blocked_reason = (
            "eval JSON missing: run "
            "`python -m training.edge.eval.run_eval --model "
            "training/edge/models/nanodet_cat_0.5x_224_int8_pc.tflite "
            "--format tflite_int8 --story-id iter-F --val-dir "
            "training/edge/data/labeled/val/ --imgsz 416`"
        )
    elif tflm_result is None:
        blocked_reason = (
            "TFLM bench JSON missing: run "
            "`python firmware/edge-bench/run_bench.py --model "
            "training/edge/models/nanodet_cat_0.5x_224_int8_pc.tflite "
            "--story-id iter-F --runs 50`"
        )

    pareto_vs_nanodet = _verdict_against(
        eval_result=eval_result,
        tflm_result=tflm_result,
        base_eval=nanodet_eval,
        base_tflm=nanodet_tflm,
        baseline_label=NANODET_BASELINE,
        multiplier=multiplier,
        blocked_reason=blocked_reason,
    )
    pareto_vs_yolo = _verdict_against(
        eval_result=eval_result,
        tflm_result=tflm_result,
        base_eval=yolo_eval,
        base_tflm=yolo_tflm,
        baseline_label=YOLO_BASELINE,
        multiplier=multiplier,
        blocked_reason=blocked_reason,
    )

    candidate_p4_ms = _x86_us_to_p4_ms(
        (tflm_result or {}).get("raw_x86_us_p50", 0.0), multiplier
    )

    aggregate_doc = {
        "story_id": "iter-F",
        "title": (
            "NanoDet revisit with per-channel quant: float32 OUTPUT dequant on "
            "the NanoDet-Plus head — same iter-A converter delta applied to the "
            "v1 US-009 PTQ pipeline"
        ),
        "status": "blocked" if blocked_reason else "passed",
        "blocked_reason": blocked_reason,
        "model_path": (eval_result or tflm_result or {}).get(
            "model_path", "training/edge/models/nanodet_cat_0.5x_224_int8_pc.tflite"
        ),
        "p4_multiplier": multiplier,
        "p4_multiplier_source": multiplier_source,
        "deviation_note": (
            "imgsz=416 not 224: only the upstream nanodet-plus-m_416.onnx is "
            "published (NO 0.5x .pth; v1 US-007/US-008 deviation). Canonical "
            "filename nanodet_cat_0.5x_224_int8_pc.tflite is preserved for "
            "spec consistency; the actual model is the 1.0x backbone at 416. "
            "Re-run after US-008's venv-gated cat fine-tune produces a true "
            "0.5x .pth at 224."
        ),
        "eval": eval_result,
        "tflm": tflm_result,
        # Two Pareto verdicts per the iter-F PRD acceptance: same-arch + cross-arch.
        "pareto_vs_nanodet": pareto_vs_nanodet,
        "pareto_vs_yolo_v2_frontier": pareto_vs_yolo,
        "candidate_metrics": {
            "map50": (eval_result or {}).get("map50", 0.0),
            "size_bytes": (eval_result or {}).get("size_bytes", 0),
            "arena_used_bytes": (tflm_result or {}).get("arena_used_bytes", 0),
            "predicted_p4_latency_ms_p50": candidate_p4_ms,
            "predicted_p4_fps": (
                1000.0 / candidate_p4_ms if candidate_p4_ms > 0 else 0.0
            ),
        },
        "nanodet_baseline_metrics": {
            "story": NANODET_BASELINE,
            "map50": (nanodet_eval or {}).get("map50", 0.0),
            "size_bytes": (nanodet_eval or {}).get("size_bytes", 0),
            "arena_used_bytes": (nanodet_tflm or {}).get("arena_used_bytes", 0),
            "predicted_p4_latency_ms_p50": _x86_us_to_p4_ms(
                (nanodet_tflm or {}).get("raw_x86_us_p50", 0.0), multiplier
            ),
        },
        "yolo_v2_frontier_metrics": {
            "story": YOLO_BASELINE,
            "map50": (yolo_eval or {}).get("map50", 0.0),
            "size_bytes": (yolo_eval or {}).get("size_bytes", 0),
            "arena_used_bytes": (yolo_tflm or {}).get("arena_used_bytes", 0),
            "predicted_p4_latency_ms_p50": _x86_us_to_p4_ms(
                (yolo_tflm or {}).get("raw_x86_us_p50", 0.0), multiplier
            ),
        },
    }
    return aggregate_doc


def main() -> int:
    doc = aggregate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    print(f"verdict vs US-009:  {doc['pareto_vs_nanodet']['verdict']}")
    print(f"verdict vs iter-A:  {doc['pareto_vs_yolo_v2_frontier']['verdict']}")
    for label, key in (
        ("nanodet", "pareto_vs_nanodet"),
        ("yolo-v2", "pareto_vs_yolo_v2_frontier"),
    ):
        d = doc[key].get("deltas")
        if d:
            print(
                f"  vs {label}: mAP {d['map50']:+.4f}, "
                f"size {d['size_pct']:+.2f}%, "
                f"arena {d['arena_pct']:+.2f}%, "
                f"latency {d['latency_pct']:+.2f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
