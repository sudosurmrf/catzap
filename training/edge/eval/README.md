# Edge eval harness — `training/edge/eval/`

Uniform mAP / size / params / FLOPs / single-thread CPU latency for any
TFLite, ONNX, or PyTorch cat detector. Every later story commits one
`training/edge/results/<story-id>.json` row through this harness so the
final SUMMARY (US-012) is mechanical.

## Quick reference

```bash
python -m training.edge.eval.run_eval \
    --model yolov8n.pt \
    --format pytorch \
    --story-id US-001-yolov8n-baseline-224 \
    --val-dir training/edge/data/val \
    --imgsz 224
```

`--format` is one of `pytorch | onnx | tflite_fp32 | tflite_int8`.

## EvalResult schema

See `training/edge/eval/types.py`. JSON keys mirror the dataclass fields.

## mAP@0.5

Single-class only (cat = id 0). Implementation in
`training/edge/eval/metrics.py` follows the pycocotools / VOC 2010
greedy-IoU + 11-point precision-recall convention:

- pycocotools: <https://github.com/cocodataset/cocoapi>
- VOC 2010 mAP: <https://github.com/Cartucho/mAP>

## Latency

`measure_latency` (in `latency.py`) does **10 warmup runs**, then **100
timed single-thread runs** with `time.perf_counter`. Each adapter sets
its underlying engine to single thread:

- PyTorch: `torch.set_num_threads(1)`
- ONNX Runtime: `intra_op_num_threads = 1`, `inter_op_num_threads = 1`
- TFLite: `Interpreter(num_threads=1)`

## Val set — silver GT bootstrap

The held-out val set under `training/edge/data/val/` is bootstrapped by
running the **existing** `server.vision.detector.CatDetector` (yolov8s)
over every JPEG under `data/cat_photos/`:

```bash
python -m training.edge.eval.bootstrap_val \
    --src data/cat_photos \
    --out training/edge/data/val \
    --teacher yolov8s.pt
```

The output layout is the standard ultralytics convention:

```
training/edge/data/val/
    images/<stem>.jpg
    labels/<stem>.txt   # YOLO format: class cx cy w h, all normalized
```

**This is silver GT** — predictions are inherited from the yolov8s
teacher and carry whatever errors that model has. As a direct
consequence, evaluating yolov8s itself against this val set will show a
near-1.0 mAP by construction. US-002 replaces this with a manually
reviewed dataset.

## Adding a new model format

1. Implement an adapter under `training/edge/eval/adapters/<name>_adapter.py`
   exposing `Predictor`-shaped `predict()`, `num_params()`, `num_flops()`,
   and an `input_hw: tuple[int, int]` attribute.
2. Wire it into `run_eval._load_adapter`.

## Optional dependencies

`requirements_edge.txt` pins everything the harness can use, but only
the format you select needs to be installed at runtime. Specifically:

- `pytorch` format — needs ultralytics + torch (already in `server/requirements.txt`)
- `onnx` format — needs `onnxruntime` (and `onnx` for param count)
- `tflite_*` formats — needs `tensorflow-cpu` (>=2.15)
- FLOPs — `thop` is best-effort; if unavailable, FLOPs is recorded as 0.
