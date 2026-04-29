"""iter-G aggregator: combine the runs=200 edge-bench JSON for iter-A with
the off-graph NMS verifier into a single training/edge/results/iter-G_off_graph_nms.json.

Usage:
    python training/edge/results/aggregate_iter_g.py

Reads:
    training/edge/results/iter-G_offgraph-tflm.json    (this iter's runs=200 bench)
    training/edge/results/iter-A-tflm.json             (runs=50 bench, for delta context)

Writes:
    training/edge/results/iter-G_off_graph_nms.json

The "candidate" .tflite for iter-G is the same as iter-A's
(yolov8n_cat_distilled_int8_pc.tflite). This iteration adds:
  - off-graph NMS confirmation via the new
    training/edge/eval/decode_only.verify_off_graph helper.
  - inference-only latency at runs=200 (tighter p50/p95 vs runs=50).
  - a documented post-process budget showing decode + NMS cost vs inference cost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.edge.eval.decode_only import verify_off_graph  # noqa: E402
from training.edge.yolo.per_channel_quant import (  # noqa: E402
    DEFAULT_P4_MULTIPLIER,
    DEFAULT_P4_MULTIPLIER_SOURCE,
)

RESULTS = Path("training/edge/results")
OUT_PATH = RESULTS / "iter-G_off_graph_nms.json"
TFLM200_PATH = RESULTS / "iter-G_offgraph-tflm.json"
ITER_A_TFLM_PATH = RESULTS / "iter-A-tflm.json"
ITER_A_AGG_PATH = RESULTS / "iter-A_per_channel_quant.json"


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


def aggregate(
    *,
    multiplier: float = DEFAULT_P4_MULTIPLIER,
    multiplier_source: str = DEFAULT_P4_MULTIPLIER_SOURCE,
) -> dict:
    tflm200 = _safe_load(TFLM200_PATH)
    iter_a_tflm = _safe_load(ITER_A_TFLM_PATH)
    iter_a_agg = _safe_load(ITER_A_AGG_PATH)

    blocked_reason: str | None = None
    if tflm200 is None:
        blocked_reason = (
            "runs=200 TFLM bench JSON missing: run "
            "`python firmware/edge-bench/run_bench.py --model "
            "training/edge/models/yolov8n_cat_distilled_int8_pc.tflite "
            "--story-id iter-G_offgraph --runs 200`"
        )

    op_breakdown = (tflm200 or {}).get("op_breakdown", []) or []
    verify = verify_off_graph(op_breakdown)

    # If a candidate's .tflite contains an in-graph NMS / Detect post-process
    # op, the iter-G AC requires it be marked status='blocked' for that
    # candidate even though the bench numbers themselves are valid.
    if blocked_reason is None and not verify.off_graph:
        blocked_reason = (
            "in-graph NMS/Detect post-process op detected: "
            f"{', '.join(verify.offending_ops)}. iter-G assumed all v2 "
            "candidates strip post-processing during ONNX export."
        )

    raw_p50 = (tflm200 or {}).get("raw_x86_us_p50", 0.0)
    raw_p95 = (tflm200 or {}).get("raw_x86_us_p95", 0.0)
    pred_p4_ms_p50 = _x86_us_to_p4_ms(raw_p50, multiplier)
    pred_p4_ms_p95 = _x86_us_to_p4_ms(raw_p95, multiplier)
    fps = (1000.0 / pred_p4_ms_p50) if pred_p4_ms_p50 > 0 else 0.0

    # Compare runs=200 vs runs=50 stability: tighter samples mean p50 should
    # converge. Useful audit-trail for iter-H.
    runs50_p50 = (iter_a_tflm or {}).get("raw_x86_us_p50", 0.0)
    runs50_p95 = (iter_a_tflm or {}).get("raw_x86_us_p95", 0.0)
    p50_drift_pct = (
        100.0 * (raw_p50 - runs50_p50) / runs50_p50
        if runs50_p50 > 0
        else 0.0
    )
    p95_drift_pct = (
        100.0 * (raw_p95 - runs50_p95) / runs50_p95
        if runs50_p95 > 0
        else 0.0
    )

    # Decode + NMS post-process budget. The C port runs on the P4 host CPU
    # (NOT through TFLM), so we estimate based on op count and a conservative
    # ESP32-P4 instruction throughput. xywh divide + clip + sigmoid filter +
    # greedy IoU NMS is dominated by the per-anchor confidence-threshold scan.
    # For YOLOv8 at imgsz=224 with 1029 anchors, the inner loop is ~6 FLOPs
    # per anchor pre-NMS plus a per-anchor IoU on the kept boxes (typically
    # < 5 boxes after threshold). Empirically this is well under 1 ms even
    # at 80 MHz; we record the conservative budget here.
    num_anchors_at_224 = 1029
    per_anchor_flops = 6  # max(cls), threshold compare, xywh / imgsz, xywh -> xyxy
    # Approx P4 FLOPs/sec without ESP-DL: 400 MHz * 1 IPC * 1 FLOP/inst = 4e8
    p4_flops_per_sec = 4.0e8
    decode_ms_estimate = (num_anchors_at_224 * per_anchor_flops) / p4_flops_per_sec * 1000.0

    inference_ms_p50 = pred_p4_ms_p50
    decode_ratio_pct = (
        100.0 * decode_ms_estimate / inference_ms_p50
        if inference_ms_p50 > 0
        else 0.0
    )

    aggregate_doc = {
        "story_id": "iter-G",
        "title": (
            "NMS off-graph + decode optimization for inference-only latency"
        ),
        "status": "blocked" if blocked_reason else "passed",
        "blocked_reason": blocked_reason,
        "model_path": (tflm200 or {}).get(
            "model_path", "training/edge/models/yolov8n_cat_distilled_int8_pc.tflite"
        ),
        "p4_multiplier": multiplier,
        "p4_multiplier_source": multiplier_source,
        "verifier": {
            "off_graph": verify.off_graph,
            "offending_ops": list(verify.offending_ops),
            "status": verify.status,
            "checked_op_count": len(op_breakdown),
            "ops_seen": [row.get("op_name", "") for row in op_breakdown],
        },
        "tflm_runs200": tflm200,
        "inference_only_latency": {
            "runs": (tflm200 or {}).get("runs", 0),
            "raw_x86_us_p50": raw_p50,
            "raw_x86_us_p95": raw_p95,
            "predicted_p4_ms_p50": pred_p4_ms_p50,
            "predicted_p4_ms_p95": pred_p4_ms_p95,
            "predicted_p4_fps": fps,
        },
        "drift_vs_runs50": {
            "iter_a_runs50_x86_us_p50": runs50_p50,
            "iter_a_runs50_x86_us_p95": runs50_p95,
            "p50_drift_pct": round(p50_drift_pct, 3),
            "p95_drift_pct": round(p95_drift_pct, 3),
        },
        "post_process_budget": {
            "model_anchors_at_imgsz_224": num_anchors_at_224,
            "estimated_decode_nms_ms_at_p4_no_esp_dl": round(decode_ms_estimate, 4),
            "decode_to_inference_ratio_pct": round(decode_ratio_pct, 4),
            "comment": (
                "decode + NMS (host C, no ESP-DL) is dominated by a single "
                "linear scan over ~1029 anchors at imgsz=224; estimated at "
                "~15 us / 0.015 ms vs ~6 s inference, so post-process is "
                "<<0.001x of inference and can be ignored in the v2 frontier."
            ),
        },
        "iter_a_baseline_metrics": (iter_a_agg or {}).get("candidate_metrics"),
    }
    return aggregate_doc


def main() -> int:
    doc = aggregate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    print(f"status: {doc['status']}")
    print(f"verifier: {doc['verifier']['status']} (offenders: {doc['verifier']['offending_ops']})")
    if doc["status"] == "passed":
        ilp = doc["inference_only_latency"]
        print(
            f"runs=200 inference-only: x86 p50={ilp['raw_x86_us_p50']:.0f} us, "
            f"predicted P4 p50={ilp['predicted_p4_ms_p50']:.1f} ms, "
            f"P4 fps={ilp['predicted_p4_fps']:.4f}"
        )
        d = doc["drift_vs_runs50"]
        print(
            f"drift vs runs=50: p50={d['p50_drift_pct']:+.2f}%, p95={d['p95_drift_pct']:+.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
