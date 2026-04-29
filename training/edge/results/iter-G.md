# iter-G — NMS off-graph + decode optimization

**Status**: passed (verifier: `off-graph confirmed`)
**Candidate**: `training/edge/models/yolov8n_cat_distilled_int8_pc.tflite`
**Baseline**: `iter-A_per_channel_quant` (runs=50 same .tflite)

## What this iteration verifies

1. **Off-graph NMS**: the v2 candidate's TFLite op_breakdown contains no in-graph NMS / Detect post-process op, confirming that decode + NMS will live in firmware C code on the ESP32-P4 host CPU rather than inside the TFLM-executed graph.
2. **Inference-only latency** (runs=200): tighter p50/p95 than the runs=50 numbers carried by iter-A. This is the load-bearing latency we'll cite in iter-H's SUMMARY_v2.
3. **Decode budget**: the standalone `training/edge/eval/decode_only.py` reference reproduces the existing `onnx_adapter._decode_yolo_onnx` math bit-for-bit and adds greedy single-class NMS — so the firmware C port has a tested Python reference to mirror. Estimated post-process cost vs inference is reported below.

## Verifier output

| field | value |
|---|---|
| `verifier.off_graph` | `True` |
| `verifier.status` | `off_graph_confirmed` |
| `verifier.offending_ops` | `none` |
| `verifier.checked_op_count` | 16 |
| ops seen | ADD, CONCATENATION, CONV_2D, DEQUANTIZE, LOGISTIC, MAX_POOL_2D, MUL, PAD, QUANTIZE, RESHAPE, RESIZE_NEAREST_NEIGHBOR, SOFTMAX, SPLIT, STRIDED_SLICE, SUB, TRANSPOSE |

## Inference-only latency (runs=200)

| metric | iter-A runs=50 | iter-G runs=200 | drift |
|---|---:|---:|---:|
| raw x86 us p50 | 1,164,102 | 1,156,898 | -0.62% |
| raw x86 us p95 | 1,190,556 | 1,190,409 | -0.01% |
| predicted P4 ms p50 (×5.0) | 5,820.5 | 5,784.5 | — |
| predicted P4 fps | 0.1718 | 0.1729 | — |

## Where decode + NMS will live on ESP32-P4

- **Inside .tflite (TFLM-executed)**: backbone + neck + Detect-head convs through the per-channel int8 weight scales, ending at the single fp32 DEQUANTIZE on the `(1, 5, 1029)` cls/xywh tensor. This is what the runs=200 bench measures.
- **Outside .tflite (firmware C, host CPU)**: anchor decode (xywh / imgsz, threshold, xywh→xyxy, clip) + greedy IoU NMS. Implemented in Python at `training/edge/eval/decode_only.py` as the C-port reference.
- **Output to server**: top-N kept boxes + confidences. The server's existing `server/vision/classifier.py` (cat-vs-not-cat MobileNet) stays unchanged — the firmware sends `(crop, bbox, confidence)` tuples over Wi-Fi/HTTP.

## Post-process budget

At imgsz=224 the YOLOv8n head emits 1,029 anchors. The decode pass is one linear scan with ~6 FLOPs per anchor (max(cls), threshold compare, xywh / imgsz, xywh → xyxy, clip), and the NMS pass is O(K^2) on whatever survives the threshold (typically <5 boxes for single-class cat detection). Estimated total post-process cost on ESP32-P4 without ESP-DL acceleration:

| field | value |
|---|---:|
| estimated decode + NMS ms (P4, no ESP-DL) | 0.0154 |
| inference ms (predicted P4 p50) | 5,784.5 |
| decode ÷ inference (%) | 0.0003% |

Conclusion: post-process is negligible vs inference. iter-H's v2 frontier latency = `predicted_p4_ms_p50` directly; we do **not** need to budget extra time for off-graph decode + NMS on this class of YOLOv8n model.

## Reproduce

```sh
# 1. Run-200 TFLM bench on the iter-A candidate:
python firmware/edge-bench/run_bench.py \
    --model training/edge/models/yolov8n_cat_distilled_int8_pc.tflite \
    --story-id iter-G_offgraph --runs 200

# 2. Aggregate verifier + bench into iter-G_off_graph_nms.json:
python training/edge/results/aggregate_iter_g.py

# 3. Render this markdown:
python training/edge/results/write_iter_g_md.py
```

## Files

- `training/edge/eval/decode_only.py` — off-graph YOLOv8 decode + single-class NMS reference for the firmware C port. Cites `onnx_adapter._decode_yolo_onnx` for the bbox-decode math.
- `training/edge/tests/test_decode_only.py` — 8 tests: parity vs `_decode_yolo_onnx` on a fixed tensor, NMS dedup, full pipeline round-trip, shape-fallback, three verifier cases (in-graph NMS flagged, iter-A op set passes, multi-offender dedup).
- `training/edge/results/aggregate_iter_g.py` — combines runs=200 bench JSON with `verify_off_graph` into the iteration's deliverable.
- `training/edge/results/iter-G_offgraph-tflm.json` — runs=200 edge-bench raw output.
- `training/edge/results/iter-G_off_graph_nms.json` — aggregate with verifier verdict + post-process budget.
