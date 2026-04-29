# iter-D — Mixed INT8/INT16 quantization (INT8 weights + INT16 activations)

**Status**: passed (Pareto verdict: `equal` under v2 thresholds; **+0.2731 absolute mAP gain** vs the iter-A frontier — the largest accuracy lift in the v2 loop so far)
**Candidate**: `training/edge/models/yolov8n_cat_distilled_int8w_int16a.tflite`
**Baseline**: iter-A (per-channel float-output INT8 KD distilled — the v2 frontier)

## Chosen path

**Path (1)** — full INT16 activations + INT8 weights via the TF builtin
`tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8`.

The PRD's technical notes named two viable paths:

  1. promote ALL activations to INT16 (single converter knob, pure TF) —
     not the surgical head-only ask but a real INT8/INT16 hybrid;
  2. split-export and link two .tflites (backbone INT8, head INT16) —
     firmware territory, requires ONNX surgery on
     `ultralytics.nn.modules.head.Detect` and a two-binary loader.

We took path (1). It's a single supported_ops swap on top of the iter-A
pipeline (same calibration_loader, same converter shape, same DI seams
for tests). Path (2) is left for a later iteration if the size/arena
cost of full-model INT16 turns out to matter on the real ESP32-P4.

## Why this works

iter-A fixed the **output**-side cls collapse by appending a DEQUANTIZE
op after the head's int8 output tensor. iter-D attacks the same problem
from the **internal** side:

  * INT16 activations have 256× the dynamic range of INT8 (32,768 vs
    128 distinguishable steps per scale).
  * That means the per-tensor activation scale on every internal tensor
    — including the Detect head's pre-output `(1, 5, 1029)` — can carry
    BOTH xywh (pixel-scale 0..imgsz) AND cls (sigmoid 0..1) without
    crushing cls into one bucket.
  * Weights stay INT8 (per-channel scales) so the on-disk size is close
    to iter-A; only the activation tensors held in arena grow 2×.

iter-A and iter-D therefore stack: float output dequant + INT16
internal activations recover **+0.2731 mAP** vs iter-A alone.

## Numbers

| metric                          | baseline iter-A | iter-D INT16x8 | delta            |
|---------------------------------|-----------------|----------------|------------------|
| mAP@0.5                         | 0.2155          | **0.4885**     | **+0.2731**      |
| size_bytes                      | 3,221,000       | 3,298,712      | +77,712 (+2.41%) |
| arena_used_bytes (TFLM)         | 692,912         | 1,307,552      | +614,640 (+88.70%)|
| raw x86 p50 (us, 50 runs)       | 1,164,102       | 1,096,152      | -67,950 (-5.84%) |
| host-CPU latency p50 (ms)       | 13.31           | 112.69         | +99.38 (XNNPACK off for INT16) |
| predicted P4 p50 (ms, mult=5.0) | 5,820.51        | 5,480.76       | -339.75 (-5.84%) |
| predicted P4 fps                | 0.172           | 0.183          | +0.011           |
| input_bytes (per inference)     | 150,528 (int8)  | 602,112 (f32)  | +451,584 (4×)    |
| output_bytes (per inference)    | 20,580 (f32)    | 20,580 (f32)   | 0                |
| tflm_compatible                 | true            | **true**       | —                |

(Predicted P4 numbers use multiplier 5.0 — see
`training/edge/yolo/per_channel_quant.DEFAULT_P4_MULTIPLIER_SOURCE` for
justification. Update once real boards arrive.)

The **host-CPU latency** of 112.69 ms p50 vs iter-A's 13.31 ms is
XNNPACK-related: TFLite's XNNPACK delegate accelerates INT8 paths but
doesn't currently lower INT16-activation CONV_2D, so the eval-harness
path falls back to reference ops. The TFLM x86 bench (which never uses
XNNPACK) shows a different story: iter-D is actually marginally
**faster** (-5.84%) than iter-A on the reference path, suggesting the
real-board P4 latency will track the TFLM number, not the host-XNNPACK
number.

## Pareto verdict — `equal` (mAP-dominant non-promotion)

Under the v2 promotion thresholds (latency >=15% OR size >=20% OR arena
>=15%), iter-D does **not** promote to a new frontier:

  * mAP **gained** +0.2731 — well past any "no regress" floor;
  * size grew +2.41% (within margin);
  * arena grew **+88.70%** (a real cost — INT16 activation tensors are
    2× int8);
  * P4 latency improved -5.84% (real but well under the 15% threshold).

`build_pareto_verdict` returns `equal` because the verdict logic only
recognizes "regress" when accuracy drops, and "dominates" when at least
one efficiency axis improves past threshold. iter-D has neither — it's
an **accuracy-only frontier** sitting on top of iter-A.

This is exactly the case the user's promotion rule excludes: "the
iteration is only kept (i.e., only promoted as the new frontier) if it
dominates or equals on accuracy AND improves at least one efficiency
axis". iter-D equals/dominates on accuracy (massively) but improves no
efficiency axis past threshold, AND regresses on arena. So the v2
**latency frontier** stays at iter-A. iter-D becomes the v2 **accuracy
frontier** — a separate Pareto point the iter-H synthesis must show
honestly.

### Decision implication for iter-H

If the ESP32-P4 has the headroom (arena 1.31 MB vs 8 MB PSRAM ceiling),
iter-D is the right pick when accuracy matters more than the ~89% arena
overhead. iter-A remains the right pick when arena pressure dominates
(e.g. running multiple models concurrently, or sharing PSRAM with an
image buffer). iter-H's recommendation paragraph must surface this
trade-off explicitly.

## TFLM compatibility

`tflm_compatible: true`. Every INT16-activation op produced by the
converter (CONV_2D, MUL, LOGISTIC, ADD, PAD, SOFTMAX, SPLIT, etc.) has
a TFLM implementation in the registered op resolver. CONV_2D dominates
98.83 % of inference time (vs iter-A's 92.5 %) — the relative shift
matches what we'd expect: INT16 raises CONV_2D's compute cost more than
proportionally vs the lighter ops.

## Files

- `training/edge/yolo/mixed_int_quant.py` — module + CLI; same DI shape
  as `per_channel_quant.py` (only the converter step swapped).
- `training/edge/tests/test_mixed_int_quant.py` — 4 unit tests (DI-mocked
  TF, no real conversion); covers supported_ops choice, IO dtypes, the
  shared calibration_loader contract, and the blocked-fallback CLI path.
- `training/edge/results/aggregate_iter_d.py` — combines eval + TFLM
  JSONs with the iter-A Pareto-verdict helper; baseline pulled from
  `iter-A_per_channel_quant.candidate_metrics`.
- `training/edge/results/iter-D.json` — eval (mAP / size / latency).
- `training/edge/results/iter-D-tflm.json` — TFLM x86 bench output.
- `training/edge/results/iter-D_mixed_int.json` — final aggregate
  (this story's deliverable).
- `training/edge/eval/adapters/tflite_adapter.py` — small extension:
  the input-dtype branch now keys off `input_details[0]["dtype"]` only
  rather than the OR-of-flag-and-dtype condition. iter-D's float-IO
  model needed this; the change preserves identical behavior for every
  v1/iter-A int8-input model (their `in_dt` is already int8).

## Reproduce

```sh
# 1-2: same dataset bootstrap as iter-A — see training/edge/results/iter-A.md.

# 3. Mixed INT8/INT16 export:
python -m training.edge.yolo.mixed_int_quant \
    --model training/edge/models/yolov8n_cat_distilled.pt \
    --calib-dir training/edge/data/calibration_frames \
    --out training/edge/models/yolov8n_cat_distilled_int8w_int16a.tflite \
    --imgsz 224

# 4. Eval harness (uses --format tflite_int8 even though IO is float — the
#    adapter respects the model's actual input dtype):
python -m training.edge.eval.run_eval \
    --model training/edge/models/yolov8n_cat_distilled_int8w_int16a.tflite \
    --format tflite_int8 --story-id iter-D \
    --val-dir training/edge/data/labeled/val/ --imgsz 224

# 5. TFLM x86 bench (build edgebench first if absent):
bash firmware/edge-bench/build.sh   # one-time
python firmware/edge-bench/run_bench.py \
    --model training/edge/models/yolov8n_cat_distilled_int8w_int16a.tflite \
    --story-id iter-D --runs 50

# 6. Aggregate eval+bench into Pareto-verdict JSON:
python training/edge/results/aggregate_iter_d.py
```
