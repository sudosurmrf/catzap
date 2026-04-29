# iter-F — NanoDet revisit with per-channel quant

**Status**: passed (Pareto verdicts: vs US-009 = `equal`, vs iter-A = `regress`)
**Candidate**: `training/edge/models/nanodet_cat_0.5x_224_int8_pc.tflite`
**Baselines**:
- `US-009` — v1 NanoDet INT8 with int8-output cls collapse (same architecture).
- `iter-A` — v2 YOLOv8n frontier (cross-architecture, KD-distilled INT8 + float-output dequant).

## What changed

iter-A flipped one converter knob — `inference_output_type=tf.float32` — and
recovered YOLOv8n's PTQ cls collapse from mAP=0.205 (US-006) to mAP=0.215. The
same fix applied to the v1 US-009 NanoDet pipeline yields...

| metric                          | baseline `US-009` | iter-F | delta              |
|---------------------------------|-------------------|--------|--------------------|
| mAP@0.5                         | 0.0000            | 0.0000 | +0.0000            |
| size_bytes                      | 1,613,072         | 1,613,232 | +160 (+0.01%)   |
| arena_used_bytes (TFLM)         | 2,253,792         | 2,253,888 | +96 (+0.00%)    |
| raw x86 p50 (us, 50 runs)       | 1,834,548         | 1,788,134 | -46,414 (-2.53%) |
| predicted P4 p50 (ms, mult=5.0) | 9,172.74          | 8,940.67 | -232.07 (-2.53%) |
| predicted P4 fps                | 0.109             | 0.112    | +0.003           |
| output dtype                    | int8              | float32  | (the iter-F delta) |
| tflm_compatible                 | true              | true     | —                |

## Pareto verdict — `equal` vs US-009

The float-output fix that worked on YOLOv8 does NOT recover NanoDet's mAP. The
v1 progress.txt finding ("Per-tensor INT8 output scale collapses NanoDet-Plus
cls scores") accurately predicted this would not be a free win. Diagnostic
probe at confidence_threshold=0.05 confirms: the model produces ~1 detection
per frame, but the predicted bbox is constant across all 120 val frames
(`[0.015, 0.000, 1.000, 0.996]`) — meaning the reg-DFL distance bins also
collapsed under per-tensor activation quant. The float dequant on the way out
preserves the (tiny) cls precision in the head's last tensor, but the upstream
INT8 conv layers feeding that head have already lost the per-prior bbox
discriminability.

Per the v2 promotion thresholds (mAP held AND >=15% latency OR >=20% size OR
>=15% arena), iter-F sits at `equal` to US-009: a wash on accuracy
(both 0.000), a microscopic latency improvement (-2.5%, well under threshold).

## Pareto verdict — `regress` vs iter-A (cross-architecture)

| metric                  | iter-A (YOLOv8n KD) | iter-F (NanoDet) | delta            |
|-------------------------|---------------------|------------------|------------------|
| mAP@0.5                 | 0.2155              | 0.0000           | -0.2155          |
| size_bytes              | 3,221,000           | 1,613,232        | -49.92%          |
| arena_used_bytes        | 692,912             | 2,253,888        | +225.28%         |
| predicted P4 ms p50     | 5,820.51            | 8,940.67         | +53.61%          |
| predicted P4 fps        | 0.172               | 0.112            | -0.060           |

NanoDet at 416 is 50% smaller on disk but uses 3.25× the arena (the larger
input feature maps dominate) and is 1.54× slower at p50 — and contributes
zero accuracy after PTQ. The v2 frontier remains iter-A.

## Why YOLOv8 recovered and NanoDet did not

iter-A worked because YOLOv8's cls precision was destroyed by the converter's
output-tensor scale, not by the head's INT8 conv path. The KD-distilled student
already had calibrated cls magnitudes such that the head's last-int8 tensor
preserved enough cls signal — the fix was purely "stop the int8-output scale
from crushing it on the way to float."

NanoDet-Plus' failure mode is structurally deeper:

* NanoDet exports cls already-sigmoided (`NanoDetPlusHead._forward_onnx` does
  `cls.sigmoid()` before the concat), so the head's last tensor mixes [0,1]
  cls values with unbounded reg-DFL logits. **A single per-tensor activation
  scale on the head's INT8 conv outputs (UPSTREAM of the float-output dequant)
  picks a scale fitted to the larger reg-DFL range and quantizes cls into a
  near-zero bucket BEFORE we get to dequantize it back to float.** The
  float-output dequant then preserves whatever survived — which isn't enough.
* The reg-DFL bins themselves also collapsed: every prior predicts the same
  distances post-quant, so all 3,598 priors after NMS converge on a single
  ~full-image bbox. Confirmed by the constant prediction.

## What would actually fix NanoDet

Listed for iter-G / iter-H consideration, NOT in scope for iter-F:

1. **Output-head split via ONNX-graph surgery on `NanoDetPlusHead._forward_onnx`**:
   emit cls and reg-DFL as separate output tensors so they get separate
   activation scales. PRD-acknowledged as the structurally pure fix.
2. **fp16 fallback** (PRD-authorized in US-009 technical notes): keep the
   model fp16 instead of int8. Drops the size advantage but recovers fp32 mAP
   at ~2× size of INT8.
3. **Genuine 0.5x cat fine-tune** via the isolated venv (US-008 deferred) —
   would shrink params from 1.19 M to ~0.4 M and thereby reduce the per-tensor
   activation range. May render the head-split surgery unnecessary.

## Pareto thresholds (constants from `build_pareto_verdict`)

* `map_regress_tolerance = 0.0` (mAP must not drop)
* `latency_improve_pct = 15.0` (>=15% faster on predicted P4)
* `size_improve_pct = 20.0`
* `arena_improve_pct = 15.0`

## Deviation acknowledgement

Per the iter-F PRD acceptance criterion, this deviation is documented:

> "If the 0.5x checkpoint isn't truly 0.5x (per the v1 deviation: only the
> 1.0x .pth was upstream-published) acknowledge the deviation in the markdown."

The model evaluated and benchmarked here is the upstream pretrained
`nanodet-plus-m_416.onnx` (1.0x backbone at 416×416), not a true 0.5x cat-only
fine-tune. v1 already documented that no `nanodet-plus-m_0.5x` .pth exists in
upstream releases (only `nanodet-plus-m` 1.0x at 320/416 and `nanodet-plus-m-1.5x`).
The canonical filename `nanodet_cat_0.5x_224_int8_pc.tflite` is preserved for
spec consistency, but the input is 416×416 — not 224 — and the architecture
is 1.0x — not 0.5x. The iter-F finding (float-output dequant alone does not
fix NanoDet's PTQ collapse) is independent of which width / imgsz the model
runs at, because the failure mode is in the head's per-tensor activation
quant upstream of the output dtype.

## Reproduce

```sh
# 1. Stage the upstream NanoDet-Plus ONNX (gitignored, regenerable):
mkdir -p training/edge/nanodet/checkpoints
curl -L -o training/edge/nanodet/checkpoints/nanodet-plus-m_416.onnx \
    https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx
ln -sfn nanodet-plus-m_416.onnx \
    training/edge/nanodet/checkpoints/nanodet_plus_m_0.5x_pretrained.onnx

# 2. Per-channel-friendly INT8 export (float32 output dtype):
python -m training.edge.nanodet.per_channel_quant \
    --onnx training/edge/nanodet/checkpoints/nanodet_plus_m_0.5x_pretrained.onnx \
    --calib-dir training/edge/data/calibration_frames \
    --out training/edge/models/nanodet_cat_0.5x_224_int8_pc.tflite \
    --imgsz 416

# 3. Eval harness:
python -m training.edge.eval.run_eval \
    --model training/edge/models/nanodet_cat_0.5x_224_int8_pc.tflite \
    --format tflite_int8 --story-id iter-F \
    --val-dir training/edge/data/labeled/val/ --imgsz 416

# 4. TFLM x86 bench:
python firmware/edge-bench/run_bench.py \
    --model training/edge/models/nanodet_cat_0.5x_224_int8_pc.tflite \
    --story-id iter-F --runs 50

# 5. Aggregate eval+bench into Pareto-verdict JSON:
python training/edge/results/aggregate_iter_f.py
```

## Files

- `training/edge/nanodet/per_channel_quant.py` — module + CLI (re-uses
  `quantize_to_int8_float_output_tflite` from iter-A's
  `training.edge.yolo.per_channel_quant`).
- `training/edge/tests/test_nanodet_per_channel.py` — 5 unit tests
  (DI-mocked TF + nanodet, no real conversion).
- `training/edge/results/aggregate_iter_f.py` — combines eval + TFLM JSONs
  with two Pareto verdicts (vs US-009, vs iter-A).
- `training/edge/results/iter-F.json` — eval (mAP / size / latency).
- `training/edge/results/iter-F-tflm.json` — TFLM x86 bench output.
- `training/edge/results/iter-F_nanodet_per_channel.json` — final aggregate
  with both `pareto_vs_*` verdicts (this story's deliverable).
