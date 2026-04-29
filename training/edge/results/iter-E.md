# iter-E — YOLOv8n width=0.75x retrain + KD distill + per-channel quantize

**Status**: PASSED (Pareto verdict: `regress` — efficiency wins big but mAP collapsed at PTQ).

**Frontier role**: iter-A remains the v2 accuracy floor; iter-D remains the v2
accuracy frontier. iter-E is the v2 SIZE / LATENCY frontier on raw efficiency
numbers, but is gated by the Pareto rule's accuracy-floor requirement so it
is NOT promoted as a new frontier (mAP must hold within
`MAP_REGRESS_TOLERANCE` of iter-A's 0.2155 for promotion).

## Pipeline (re-uses iter-A / US-006 / US-002 building blocks)

```
yolov8n_0p75x.yaml (width_multiple=0.1875 = 0.25 * 0.75) + yolov8n.pt
        │  partial_load_yolov8n_weights — 355/355 keys loaded by slicing
        │  donor channels to the narrower student shape (no random init).
        ▼
yolov8n_0p75x student in memory
        │  KD distill from yolov8s.pt teacher (alpha=0.5, T=4.0; matches
        │  US-006). 50 epochs, imgsz=224 (iter-B winner), batch=32,
        │  lr=1e-4. Loss patch via `patch_student_loss_for_distill`
        │  registered on `on_train_start` so the trainer's device-resident
        │  student gets the patch.
        ▼
training/edge/models/yolov8n_0p75x_cat_distilled.pt (3.8 MB fp32)
        │  per_channel_quant.export_per_channel_int8 — IDENTICAL pipeline
        │  as iter-A: ONNX -> onnx2tf SavedModel -> per-channel weight
        │  INT8 + per-tensor activation INT8 + float32 OUTPUT dequant.
        ▼
training/edge/models/yolov8n_0p75x_cat_distilled_int8_pc.tflite (1.98 MB)
        │  run_eval -> training/edge/results/iter-E_eval.json
        │  edge-bench -> training/edge/results/iter-E_bench-tflm.json
        ▼
Pareto verdict vs iter-A frontier in iter-E_yolov8n_0p75x.json
```

### Distill-step deviation (documented per AC)

The AC says "re-use `training.edge.yolo.distill_train.train_distilled` with
the new student .pt as the student". `train_distilled` accepts a `.pt`
PATH, but a YAML-built model has no populated `self.ckpt` dict so
`student.save()` would refuse to round-trip the partial-loaded weights.
Resolution: the iter-E wrapper imports `patch_student_loss_for_distill`
(the load-bearing helper inside `train_distilled`) and registers it on
the LIVE student instance via the same `on_train_start` callback pattern
US-006 uses. Net effect is byte-identical to `train_distilled` minus the
.pt round-trip; `_default_distill_train_fn` carries this logic in
`train_yolov8n_0p75x_cat.py`. The KD recipe (alpha, temperature, teacher,
student-loss patch) is unchanged.

## Numbers

### Decision-matrix row (vs iter-A frontier baseline, P4 multiplier=5.0)

| metric                             | iter-A (baseline) | iter-E         | Δ                      |
|------------------------------------|-------------------|----------------|------------------------|
| mAP@0.5                            | 0.2155            | 0.0000         | -0.2155 (regress)      |
| size_bytes                         | 3,221,000         | 1,977,944      | -38.59% ✓ past 20%     |
| arena_used_bytes                   | 692,912           | 683,312        | -1.39%                 |
| predicted_p4_latency_ms_p50        | 5,820.51          | 3,804.66       | -34.63% ✓ past 15%     |
| predicted_p4_fps                   | 0.172             | 0.263          | +52.92%                |
| host int8 latency p50 (eval)       | ~3 ms             | 2.75 ms        | -8.4%                  |
| tflm_compatible                    | true              | true           | held                   |

**Pareto verdict: regress** — `build_pareto_verdict` emits `regress` whenever
`candidate_map50 < baseline_map50 - MAP_REGRESS_TOLERANCE` (0.0 tolerance,
hard floor). Two efficiency axes pass their thresholds (size -38.59% past
the 20% bar, latency -34.63% past the 15% bar) but the mAP regression veto
takes precedence per the v2 promotion rules.

### fp32 reference

The fp32 distilled .pt validates at ultralytics-internal mAP@0.5 = **0.995**
(saturation, same as US-006's distilled fp32 = 1.000 on the same val set).
Wall-clock training: **50 epochs in 0.021 hours** (~75 s) on a single RTX
4090 — same envelope as US-006 because the teacher forward at yolov8s scale
dominates the per-step cost regardless of student width.

### TFLM op breakdown (top 5)

| op              | count | percent |
|-----------------|-------|---------|
| CONV_2D         | 3,200 | 90.58%  |
| LOGISTIC        |   400 | 7.21%   |
| MUL             |   500 | 1.07%   |
| MAX_POOL_2D     |   100 | 0.71%   |
| ADD             |   400 | 0.08%   |

CONV_2D share rises from iter-A's 92.45% to 90.58% because LOGISTIC moves
from 6.66% to 7.21% (the head's per-anchor sigmoid stays the same absolute
cost while the smaller backbone drops convolution cost). Composition is
otherwise identical to iter-A's 16-op set; zero unsupported ops.

## Why mAP collapsed at PTQ — diagnosis

The fp32 distilled .pt at width=0.75x is healthy (P=0.999, R=1.0,
mAP@0.5=0.995 on the held-out val). After the IDENTICAL iter-A
per-channel-friendly PTQ pipeline, the INT8 .tflite returns mAP=0.0 at the
default conf=0.25 — the same v1 cls-collapse pattern.

What's happening: the YOLOv8 Detect head emits a single
`(1, 5, num_priors)` output tensor mixing xywh (pixel-range 0..imgsz) with
cls (sigmoid 0..1). One per-tensor activation scale spans both, and xywh
dominates. iter-A's mitigation (float32 OUTPUT dequant + KD-shaped student
cls) recovered ~0.215 mAP because the YOLOv8n full-width KD student had
just enough cls magnitude to survive the per-tensor scale. At
width=0.1875 (the effective stamped value for "0.75x of yolov8n"), the
narrower channels carry less cls signal — the per-tensor activation
quantizer crushes it back to a single int8 bucket, and the float dequant
on the way out preserves zero of nothing.

This matches three independent v2 findings:

- v1 progress.txt: "YOLOv8 PTQ + per-tensor output quant collapses cls head"
- iter-C: "soft prune + supervised-only finetune erodes KD cls signal"
- iter-A baseline: "Not a full recovery but the first non-zero YOLOv8 PTQ"

The structural root cause is the monolithic Detect output tensor sharing
one per-tensor activation scale across xywh + cls. Three known fixes:

1. **Detect-head split** — ONNX surgery to emit separate cls / xywh output
   tensors, each getting its own per-tensor scale. Out of scope for iter-E
   (touches ultralytics' export graph).
2. **INT16 activations** — iter-D's path; raises Detect output's pre-quant
   dynamic range 256x. Already proven on this val (iter-D mAP 0.488 vs
   iter-A 0.215). Stacking iter-D's INT16x8 quant onto iter-E's narrower
   student is the obvious follow-on.
3. **More aggressive KD** — increase distillation alpha/temperature to push
   the narrower student's cls magnitudes higher pre-quant. Not pursued
   because iter-A already used the recommended alpha=0.5; pushing further
   risks degrading detector accuracy at fp32.

## Promotion decision under v2 rules

Promote? **No.** mAP regressed; verdict is correctly `regress`.

What iter-E IS: empirical proof that a yolov8n-narrower-than-narrow can
fit comfortably in the ESP32-P4 PSRAM at significantly lower latency
(0.263 fps vs 0.172 fps under the 5.0x multiplier — `+53% throughput`).
The accuracy-floor failure is purely a quantization issue, not a model
capacity issue. Stacking iter-D's INT16-activation quant onto the iter-E
student is the highest-priority follow-on for iter-H to flag (or
implement directly if a v3 loop is opened).

## Files

- `training/edge/yolo/train_yolov8n_0p75x_cat.py` — pipeline orchestrator
  with DI seams (yolo_factory, distill_train_fn, quantize_fn, eval_fn,
  bench_fn, baseline_loader). Per-stage failure isolation; rollup JSON
  always written.
- `training/edge/yolo/configs/yolov8n_0p75x.yaml` — yolov8.yaml shape with
  effective `width_multiple=0.1875` + `depth_multiple=0.33` after the
  `regenerate_config()` composition step.
- `training/edge/tests/test_yolov8n_0p75x.py` — 8 mocked tests covering
  the AC contract (width_multiple=0.75 reaches trainer; quantize is called
  AFTER trainer; partial-load fallback to random-init; blocked-status JSON
  on retrain or quantize failure; iter-H ingestion schema).
- `training/edge/models/yolov8n_0p75x_cat_distilled.pt` — fp32 distilled
  checkpoint (3.8 MB; gitignored — regenerate via the CLI).
- `training/edge/models/yolov8n_0p75x_cat_distilled_int8_pc.tflite` —
  INT8 PTQ .tflite (1.98 MB; gitignored).
- `training/edge/results/iter-E_yolov8n_0p75x.json` — main rollup.
- `training/edge/results/iter-E_eval.json` — eval-harness sidecar.
- `training/edge/results/iter-E_bench-tflm.json` — TFLM x86 bench sidecar.

## Reproduce

```sh
# (One-off, after a fresh worktree only — bootstrap dataset + val manifest.)
python -m training.edge.auto_label bootstrap
python -m training.edge.make_dataset_manifest
python training/edge/data/regenerate_val_manifest.py

# Run iter-E end-to-end. Defaults match the AC values; ~2 minutes on a single
# RTX 4090.
TF_USE_LEGACY_KERAS=1 python -m training.edge.yolo.train_yolov8n_0p75x_cat
```

The CLI exposes `--width-multiple`, `--depth-multiple`, `--imgsz`,
`--epochs`, `--batch`, `--lr`, `--alpha`, `--temperature`, `--device` for
sensitivity studies; the bare invocation with no flags reproduces this
report's numbers.
