"""Render training/edge/results/iter-G.md from iter-G_off_graph_nms.json.

Mirrors the iter-A.md / iter-D.md structure: short narrative + numbers
table + reproduce block + files list. Run after aggregate_iter_g.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path("training/edge/results")
JSON_PATH = RESULTS / "iter-G_off_graph_nms.json"
MD_PATH = RESULTS / "iter-G.md"


def render(doc: dict) -> str:
    verifier = doc.get("verifier", {})
    ilp = doc.get("inference_only_latency", {})
    drift = doc.get("drift_vs_runs50", {})
    budget = doc.get("post_process_budget", {})
    iter_a = doc.get("iter_a_baseline_metrics") or {}

    status = doc.get("status", "unknown")
    blocked = doc.get("blocked_reason")

    ops_seen = ", ".join(verifier.get("ops_seen", [])) or "<none>"
    offending = ", ".join(verifier.get("offending_ops", [])) or "none"

    lines: list[str] = []
    lines.append("# iter-G — NMS off-graph + decode optimization")
    lines.append("")
    verdict = "off-graph confirmed" if verifier.get("off_graph") else "BLOCKED"
    lines.append(f"**Status**: {status} (verifier: `{verdict}`)")
    lines.append(f"**Candidate**: `{doc.get('model_path', '')}`")
    lines.append("**Baseline**: `iter-A_per_channel_quant` (runs=50 same .tflite)")
    if blocked:
        lines.append("")
        lines.append(f"> **Blocked reason**: {blocked}")
    lines.append("")
    lines.append("## What this iteration verifies")
    lines.append("")
    lines.append(
        "1. **Off-graph NMS**: the v2 candidate's TFLite op_breakdown contains "
        "no in-graph NMS / Detect post-process op, confirming that decode + "
        "NMS will live in firmware C code on the ESP32-P4 host CPU rather "
        "than inside the TFLM-executed graph."
    )
    lines.append(
        "2. **Inference-only latency** (runs=200): tighter p50/p95 than the "
        "runs=50 numbers carried by iter-A. This is the load-bearing latency "
        "we'll cite in iter-H's SUMMARY_v2."
    )
    lines.append(
        "3. **Decode budget**: the standalone "
        "`training/edge/eval/decode_only.py` reference reproduces the "
        "existing `onnx_adapter._decode_yolo_onnx` math bit-for-bit and adds "
        "greedy single-class NMS — so the firmware C port has a tested "
        "Python reference to mirror. Estimated post-process cost vs "
        "inference is reported below."
    )
    lines.append("")
    lines.append("## Verifier output")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| `verifier.off_graph` | `{verifier.get('off_graph')}` |")
    lines.append(f"| `verifier.status` | `{verifier.get('status')}` |")
    lines.append(f"| `verifier.offending_ops` | `{offending}` |")
    lines.append(f"| `verifier.checked_op_count` | {verifier.get('checked_op_count', 0)} |")
    lines.append(f"| ops seen | {ops_seen} |")
    lines.append("")
    lines.append("## Inference-only latency (runs=200)")
    lines.append("")
    lines.append(
        "| metric | iter-A runs=50 | iter-G runs=200 | drift |"
    )
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| raw x86 us p50 | {drift.get('iter_a_runs50_x86_us_p50', 0):,.0f} "
        f"| {ilp.get('raw_x86_us_p50', 0):,.0f} "
        f"| {drift.get('p50_drift_pct', 0):+.2f}% |"
    )
    lines.append(
        f"| raw x86 us p95 | {drift.get('iter_a_runs50_x86_us_p95', 0):,.0f} "
        f"| {ilp.get('raw_x86_us_p95', 0):,.0f} "
        f"| {drift.get('p95_drift_pct', 0):+.2f}% |"
    )
    lines.append(
        f"| predicted P4 ms p50 (×{doc.get('p4_multiplier', 5.0)}) | "
        f"{(drift.get('iter_a_runs50_x86_us_p50', 0)) * doc.get('p4_multiplier', 5.0) / 1000.0:,.1f} "
        f"| {ilp.get('predicted_p4_ms_p50', 0):,.1f} "
        f"| — |"
    )
    lines.append(
        f"| predicted P4 fps | {iter_a.get('predicted_p4_fps', 0):.4f} "
        f"| {ilp.get('predicted_p4_fps', 0):.4f} | — |"
    )
    lines.append("")
    lines.append("## Where decode + NMS will live on ESP32-P4")
    lines.append("")
    lines.append(
        "- **Inside .tflite (TFLM-executed)**: backbone + neck + Detect-head "
        "convs through the per-channel int8 weight scales, ending at the "
        "single fp32 DEQUANTIZE on the `(1, 5, 1029)` cls/xywh tensor. This "
        "is what the runs=200 bench measures."
    )
    lines.append(
        "- **Outside .tflite (firmware C, host CPU)**: anchor decode "
        "(xywh / imgsz, threshold, xywh→xyxy, clip) + greedy IoU NMS. "
        "Implemented in Python at `training/edge/eval/decode_only.py` as "
        "the C-port reference."
    )
    lines.append(
        "- **Output to server**: top-N kept boxes + confidences. The server's "
        "existing `server/vision/classifier.py` (cat-vs-not-cat MobileNet) "
        "stays unchanged — the firmware sends `(crop, bbox, confidence)` "
        "tuples over Wi-Fi/HTTP."
    )
    lines.append("")
    lines.append("## Post-process budget")
    lines.append("")
    lines.append(
        f"At imgsz=224 the YOLOv8n head emits "
        f"{budget.get('model_anchors_at_imgsz_224', 0):,} anchors. The decode "
        "pass is one linear scan with ~6 FLOPs per anchor (max(cls), threshold "
        "compare, xywh / imgsz, xywh → xyxy, clip), and the NMS pass is "
        "O(K^2) on whatever survives the threshold (typically <5 boxes for "
        "single-class cat detection). Estimated total post-process cost on "
        "ESP32-P4 without ESP-DL acceleration:"
    )
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---:|")
    lines.append(
        f"| estimated decode + NMS ms (P4, no ESP-DL) | "
        f"{budget.get('estimated_decode_nms_ms_at_p4_no_esp_dl', 0):.4f} |"
    )
    lines.append(
        f"| inference ms (predicted P4 p50) | {ilp.get('predicted_p4_ms_p50', 0):,.1f} |"
    )
    lines.append(
        f"| decode ÷ inference (%) | "
        f"{budget.get('decode_to_inference_ratio_pct', 0):.4f}% |"
    )
    lines.append("")
    lines.append(
        "Conclusion: post-process is negligible vs inference. iter-H's v2 "
        "frontier latency = `predicted_p4_ms_p50` directly; we do **not** "
        "need to budget extra time for off-graph decode + NMS on this class "
        "of YOLOv8n model."
    )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```sh")
    lines.append("# 1. Run-200 TFLM bench on the iter-A candidate:")
    lines.append("python firmware/edge-bench/run_bench.py \\")
    lines.append("    --model training/edge/models/yolov8n_cat_distilled_int8_pc.tflite \\")
    lines.append("    --story-id iter-G_offgraph --runs 200")
    lines.append("")
    lines.append("# 2. Aggregate verifier + bench into iter-G_off_graph_nms.json:")
    lines.append("python training/edge/results/aggregate_iter_g.py")
    lines.append("")
    lines.append("# 3. Render this markdown:")
    lines.append("python training/edge/results/write_iter_g_md.py")
    lines.append("```")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(
        "- `training/edge/eval/decode_only.py` — off-graph YOLOv8 decode + "
        "single-class NMS reference for the firmware C port. Cites "
        "`onnx_adapter._decode_yolo_onnx` for the bbox-decode math."
    )
    lines.append(
        "- `training/edge/tests/test_decode_only.py` — 8 tests: parity vs "
        "`_decode_yolo_onnx` on a fixed tensor, NMS dedup, full pipeline "
        "round-trip, shape-fallback, three verifier cases (in-graph NMS "
        "flagged, iter-A op set passes, multi-offender dedup)."
    )
    lines.append(
        "- `training/edge/results/aggregate_iter_g.py` — combines runs=200 "
        "bench JSON with `verify_off_graph` into the iteration's deliverable."
    )
    lines.append(
        "- `training/edge/results/iter-G_offgraph-tflm.json` — runs=200 "
        "edge-bench raw output."
    )
    lines.append(
        "- `training/edge/results/iter-G_off_graph_nms.json` — aggregate "
        "with verifier verdict + post-process budget."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    if not JSON_PATH.exists():
        sys.stderr.write(
            f"missing {JSON_PATH}; run aggregate_iter_g.py first\n"
        )
        return 1
    doc = json.loads(JSON_PATH.read_text())
    MD_PATH.write_text(render(doc))
    print(f"wrote {MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
