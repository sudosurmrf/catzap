# iter-A — Per-channel output quantization fix

**Status**: passed (Pareto verdict: `equal`)
**Candidate**: `training/edge/models/yolov8n_cat_distilled_int8_pc.tflite`
**Baseline**: `US-006-int8` (KD-distilled YOLOv8n INT8, the v1 frontier)

## What changed

The v1 PTQ pipeline (`training/edge/yolo/export_quantize.py`) ran the
`tf.lite.TFLiteConverter` with both `inference_input_type=tf.int8` AND
`inference_output_type=tf.int8`. This packs YOLOv8's monolithic Detect
output `(1, 5, 1029)` — which mixes xywh (pixel scale `0..imgsz`) with
cls (sigmoid `0..1`) — into a single per-tensor int8 scale dominated by
xywh. The cls channel collapses into one int8 bucket, every confidence
becomes 0, mAP=0 (US-004) or 0.205 after KD partial-recovery (US-006).

iter-A flips one knob: `inference_output_type=tf.float32`. Internally the
model is still INT8 (per-channel weight scales via the MLIR quantizer);
externally the converter appends a DEQUANTIZE op so the cls head reads
out at full float resolution. Cost is a single dequant on a `(1,5,1029)`
tensor (~5,145 floats; ~5 µs on x86 reference TFLM, smaller on ESP32-P4
vector ops).

## Numbers

| metric                          | baseline `US-006-int8` | iter-A `pc` | delta            |
|---------------------------------|------------------------|-------------|------------------|
| mAP@0.5                         | 0.2047                 | 0.2155      | **+0.0108**      |
| size_bytes                      | 3,220,840              | 3,221,000   | +160 (+0.005%)   |
| arena_used_bytes (TFLM)         | 692,816                | 692,912     | +96   (+0.014%)  |
| raw x86 p50 (us, 50 runs)       | 1,176,673              | 1,164,102   | -12,571 (-1.07%) |
| predicted P4 p50 (ms, mult=5.0) | 5,883.36               | 5,820.51    | -62.85 (-1.07%)  |
| predicted P4 fps                | 0.170                  | 0.172       | +0.002           |
| output_bytes (float32 vs int8)  | 5,145 (int8)           | 20,580 (f32)| +15,435 (4×)     |
| tflm_compatible                 | true                   | true        | —                |

(Predicted P4 numbers use multiplier 5.0 — conservative midpoint of v1's
8.0× ESP32-S3 multiplier and the ~3× lower bound assuming PIE/vector ext
on P4. See `training/edge/yolo/per_channel_quant.DEFAULT_P4_MULTIPLIER_SOURCE`
for the full justification. Update once real boards arrive.)

## Pareto verdict — `equal`

iter-A holds accuracy (in fact gains +1.08 mAP-points, ~5% relative) at
near-zero efficiency cost. Under the v2 promotion thresholds
(`latency >=15% OR size >=20% OR arena >=15%`) it does NOT promote to
new-frontier status. But it is the strict accuracy floor for everything
downstream:

  - iter-B (imgsz sweep) and iter-C (channel pruning) MUST measure their
    candidates against iter-A's mAP=0.2155, not US-006-int8's 0.2047.
  - iter-D (mixed INT8/INT16) is allowed to drop mAP back to 0.2047 only
    if it gains >=15% latency or >=15% arena vs iter-A.

## Why this works (codified in v1 progress.txt)

The v1 progress patterns ("YOLOv8 PTQ + per-tensor output quant
collapses cls head", "Detect head channel layout: [4*reg_max box-DFL,
nc cls]") describe the failure mode. The v1 KD recovery (US-006)
addressed the **student-side** activation distribution: by distilling
YOLOv8s teacher cls magnitudes into the student, the post-PTQ
per-tensor scale on the head's output no longer crushes cls into one
bucket entirely. iter-A addresses the **converter-side** issue
orthogonally: by keeping the output as float32, no int8 output scale
exists to fight in the first place. The two fixes compound — KD shapes
the activations the calibrator sees, and the float-output dequant
preserves whatever precision survived.

The architecturally pure fix (split Detect into separate cls and xywh
output tensors via ONNX-graph surgery) is still on the table. It would
let us keep INT8 output at smaller output_bytes — only ~5,145 bytes vs
20,580. We defer that to a later iteration if the float-output cost
becomes meaningful on the P4 (current TFLM bench shows the dequant op
at ~5 µs out of ~1.16 s total — negligible).

## Reproduce

```sh
# 1. Bootstrap labeled JPEGs (gitignored, regenerable). Note that
#    auto_label.bootstrap uses uuid4() filenames so each run produces a
#    DIFFERENT 600-frame set with byte-equivalent CONTENT (seed=42 covers
#    augmentation, not naming). v1's tracked .txt labels won't pair with
#    the freshly-bootstrapped jpgs — that's expected.
python -m training.edge.auto_label bootstrap \
    --src-dir data/cat_photos \
    --out-dir training/edge/data/labeled \
    --target-frames 600 --augments-per-image 25 --seed 42
python -m training.edge.make_dataset_manifest --dataset-dir training/edge/data/labeled

# 1b. Repair val/manifest.txt for the freshly-bootstrapped UUIDs:
python training/edge/data/regenerate_val_manifest.py

# 2. Build the calibration subset (200 frames):
python -c "
import shutil; from pathlib import Path
src = Path('training/edge/data/labeled')
dst = Path('training/edge/data/calibration_frames'); dst.mkdir(exist_ok=True)
for j in sorted(src.glob('*.jpg'))[:200]: shutil.copy2(j, dst/j.name)
"

# 3. Per-channel-friendly INT8 export:
python -m training.edge.yolo.per_channel_quant \
    --model training/edge/models/yolov8n_cat_distilled.pt \
    --calib-dir training/edge/data/calibration_frames \
    --out training/edge/models/yolov8n_cat_distilled_int8_pc.tflite \
    --imgsz 224

# 4. Eval harness:
python -m training.edge.eval.run_eval \
    --model training/edge/models/yolov8n_cat_distilled_int8_pc.tflite \
    --format tflite_int8 --story-id iter-A \
    --val-dir training/edge/data/labeled/val/ --imgsz 224

# 5. TFLM x86 bench (build edgebench first if needed):
bash firmware/edge-bench/build.sh   # one-time
python firmware/edge-bench/run_bench.py \
    --model training/edge/models/yolov8n_cat_distilled_int8_pc.tflite \
    --story-id iter-A --runs 50

# 6. Aggregate eval+bench into Pareto-verdict JSON:
python training/edge/results/aggregate_iter_a.py
```

## Files

- `training/edge/yolo/per_channel_quant.py` — module + CLI, plus
  `build_pareto_verdict` helper used by every downstream iter-* JSON.
- `training/edge/tests/test_per_channel_quant.py` — 9 unit tests
  (DI-mocked TF, no real conversion).
- `training/edge/results/aggregate_iter_a.py` — combines eval + TFLM
  JSONs with the verdict helper.
- `training/edge/results/iter-A.json` — eval (mAP / size / latency).
- `training/edge/results/iter-A-tflm.json` — TFLM x86 bench output.
- `training/edge/results/iter-A_per_channel_quant.json` — final
  aggregate with `pareto.verdict` + deltas (this story's deliverable).
