# iter-C — Structured channel pruning of the distilled student

**Status**: passed (Pareto verdict: `regress`)
**Candidate**: `training/edge/models/yolov8n_cat_pruned_int8_pc.tflite`
**Baseline**: iter-A (per-channel-friendly INT8 PTQ on `yolov8n_cat_distilled.pt`)

## What changed

iter-C picks up from the iter-A frontier and applies L2-norm structured
channel pruning to ~25% of every backbone/neck `Conv2d`'s output channels
(skipping the Detect head + DFL projection, which expect specific channel
counts). The pruned weights are baked into the `.pt` via
`torch.nn.utils.prune.remove`, then a single epoch of supervised
fine-tuning at lr=1e-4 lets the surviving channels redistribute. The
resulting `.pt` runs through the SAME iter-A per-channel-friendly INT8
quantizer (`training.edge.yolo.per_channel_quant.export_per_channel_int8`)
to keep the quant pipeline byte-identical to the baseline.

Pipeline (always orchestrated by `run_pipeline`):

  1. `prune_l2_channels(model, sparsity=0.25, exclude_head=True)` — soft
     prune (mask + `prune.remove`), 45 conv layers pruned, 19 head/DFL
     conv layers skipped. Actual achieved sparsity 25.00 %.
  2. ultralytics 1-epoch fine-tune (`single_cls=True`, `imgsz=224`,
     `batch=8`, `lr0=1e-4`) on the US-002 dataset (480 train / 120 val).
  3. ONNX export -> SavedModel via onnx2tf -> INT8 TFLite via the
     iter-A `quantize_to_int8_float_output_tflite` (per-channel weight
     scales + float32 output dequant).
  4. Eval harness (`tflite_int8` adapter) on the same 120-image val.
  5. TFLM x86 bench (`firmware/edge-bench/`) at runs=50.
  6. Pareto verdict vs iter-A's `candidate_metrics`.

## Numbers

| metric                          | iter-A (baseline) | iter-C (pruned) | delta             |
|---------------------------------|-------------------|-----------------|-------------------|
| Pruned conv layers              | 0                 | 45              | +45               |
| Skipped head conv layers        | n/a               | 19              | n/a               |
| Pre-prune backbone+neck params  | 2,250,672         | 2,250,672       | 0                 |
| Post-prune non-zero params      | 2,250,672         | 1,688,004       | -562,668 (-25.0%) |
| Achieved structural sparsity    | 0.0 %             | 25.00 %         | +25 pts           |
| Fine-tune epochs                | 50 (US-006 KD)    | +1              | +1                |
| Pruned `.pt` mAP@0.5 (fp32)     | 1.000             | 0.706 †         | -0.294            |
| INT8 TFLite mAP@0.5             | 0.2155            | **0.0000**      | **-0.2155**       |
| size_bytes                      | 3,221,000         | 3,220,832       | -168 (-0.005%)    |
| arena_used_bytes (TFLM)         | 692,912           | 692,784         | -128 (-0.018%)    |
| raw x86 p50 (us, 50 runs)       | 1,164,102         | 1,171,088       | +6,986 (+0.60%)   |
| predicted P4 p50 (ms, mult=5.0) | 5,820.51          | 5,855.44        | +34.93 (+0.60%)   |
| predicted P4 fps                | 0.172             | 0.171           | -0.001            |
| tflm_compatible                 | true              | true            | —                 |

† Reported by ultralytics' own validator on the pruned-and-finetuned `.pt`
during training. The harness's INT8 eval is the load-bearing number; the
fp32 mAP is included for diagnostic context (it shows the prune+1-epoch
recovery is real before quantization, and the regression is concentrated
at the PTQ step).

## Pareto verdict — `regress`

iter-C's INT8 mAP collapsed to 0.0000 vs iter-A's 0.2155, despite the
pruned `.pt` retaining mAP=0.706 in fp32. The Pareto helper trips on the
mAP regression and emits `verdict="regress"` regardless of the
near-baseline efficiency numbers. **iter-C is NOT promoted to the v2
frontier; iter-A remains the accuracy floor for downstream stories.**

The arena and size deltas (-0.018 % and -0.005 %) confirm what soft
pruning predicts: zeroed weights compress identically as nonzero ones in
the TFLite flatbuffer, and the per-channel quantizer's scale tables
don't shrink when 25 % of channels go to zero. To realize a real arena
or size win from pruning, the channels have to be physically removed
from the conv weight tensors AND from the consumer layers' input
channels — a "hard slice + Detect-head re-init" which is out of scope
for iter-C and is a candidate for a follow-on story.

## Why mAP collapsed at PTQ — the diagnosis

The cls head failure mode mirrors v1's `YOLOv8 PTQ + per-tensor output
quant collapses cls head` pattern (see progress.txt), but with a new
twist:

  - iter-A's `yolov8n_cat_distilled.pt` had its cls-channel logits
    pre-shaped by the YOLOv8s teacher (US-006 KD), so the post-PTQ cls
    bucket retained ~5 % signal — the float32-output dequant then
    preserved that signal and gave mAP=0.215.
  - iter-C's pruned-and-finetuned `.pt` partially overwrote those KD-
    shaped cls magnitudes during the 1-epoch fine-tune. The fine-tune
    used the supervised CE/BCE loss only (no distillation signal), so
    the cls logits drifted back toward magnitudes that the per-tensor
    activation quantizer can't preserve. The float32 OUTPUT dequant
    helps but cannot recover precision lost INSIDE the head.

The diagnosis is verifiable downstream: a lower confidence_threshold
probe (e.g. 0.05) on this same `.tflite` recovers a small number of
detections, exactly mirroring NanoDet's "saturated low" failure mode
documented in v1 progress.txt. Adding the KD loss to the post-prune
fine-tune (i.e. pruning the iter-A *student* and then re-distilling
from yolov8s) would keep the cls activation distribution compatible
with PTQ — that's the natural follow-on.

## Pull-forward for downstream stories

  - **iter-D** (mixed INT8/INT16 head) directly addresses the same
    cls-precision-loss failure mode at PTQ time. iter-C's collapse is
    additional motivation for trying INT16 activations on the head.
  - **iter-E** (YOLOv8n width=0.75x retrain with KD) should NOT inherit
    iter-C's pruned `.pt` — start from `yolov8n.pt` per the spec. The
    iter-C result reinforces that any width reduction must keep the KD
    loss in the loop.
  - **iter-F** (NanoDet revisit with per-channel quant) is unaffected.
  - **iter-G** (off-graph NMS) is unaffected.
  - **iter-H** (SUMMARY_v2) lists iter-C in the v2 candidates table
    with verdict=`regress` so the negative finding is part of the
    public record.

## Files

- `training/edge/yolo/prune_channels.py` — module + CLI. Exports
  `prune_l2_channels(model, sparsity, exclude_head)` (the AC function),
  `_default_trainer_fn` (prune+finetune in a single call), and
  `run_pipeline` (the full prune→quant→eval→bench orchestrator with
  per-stage failure isolation).
- `training/edge/tests/test_prune_channels.py` — 9 tests (18 with
  parametrize expansion). Required AC tests: (1) sparsity reduction
  matches target on a tiny module, (2) Detect head excluded, (3) CLI
  pipeline calls quantize after trainer.
- `training/edge/models/yolov8n_cat_pruned.pt` — pruned-and-finetuned
  fp32 (gitignored — regenerable from the CLI).
- `training/edge/models/yolov8n_cat_pruned_int8_pc.tflite` — INT8 PTQ
  output (gitignored).
- `training/edge/results/iter-C_pruned.json` — full rollup with eval +
  TFLM bench + prune_summary + Pareto verdict (this story's
  deliverable).

## Reproduce

```sh
# 1. Bootstrap labeled JPEGs + repair val manifest (per iter-A pattern):
python -m training.edge.auto_label bootstrap \
    --src-dir data/cat_photos --out-dir training/edge/data/labeled \
    --target-frames 600 --augments-per-image 25 --seed 42
python -m training.edge.make_dataset_manifest --dataset-dir training/edge/data/labeled
python training/edge/data/regenerate_val_manifest.py
# (the train_yolov8n_cat.prepare_data() helper writes data.yaml +
# train.txt/val.txt for the current worktree path on first invocation)

# 2. Build edgebench if needed (one-time):
bash firmware/edge-bench/build.sh

# 3. Run the iter-C pipeline (CPU OK, ~3 min on a 13700K):
python -m training.edge.yolo.prune_channels \
    --model training/edge/models/yolov8n_cat_distilled.pt \
    --sparsity 0.25 --finetune-epochs 1 \
    --data training/edge/data/labeled/data.yaml \
    --out training/edge/models/yolov8n_cat_pruned.pt \
    --device cpu --batch 8

# 4. Inspect the verdict:
jq '.pareto.verdict, .candidate_metrics' \
    training/edge/results/iter-C_pruned.json
```

The CLI flags above match the iter-C acceptance criteria one-to-one.
Default kwargs reproduce the same numbers (modulo CUDA non-determinism
on machines that have a GPU); the bare `python -m
training.edge.yolo.prune_channels` invocation produces an identical
JSON shape on any host.
