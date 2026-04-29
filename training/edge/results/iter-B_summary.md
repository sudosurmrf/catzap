# iter-B — input resolution sweep on the iter-A frontier
**Baseline**: iter-A (per-channel-friendly INT8 PTQ on `yolov8n_cat_distilled.pt` at imgsz=224)
- iter-A mAP@0.5 = 0.2155
- iter-A size_bytes = 3,221,000
- iter-A arena_used_bytes = 692,912
- iter-A predicted P4 latency ms p50 = 5820.51

## Per-imgsz comparison
| imgsz | status | mAP@0.5 | size | arena | predicted P4 ms p50 | predicted P4 fps | mAP delta | size delta | arena delta | latency delta | verdict |
|-------|--------|---------|------|-------|---------------------|------------------|-----------|------------|-------------|---------------|---------|
| 192 | passed | 0.0000 | 3,220,672 | 532,144 | 4244.81 | 0.236 | -0.2155 | -0.01% | -23.20% | -27.07% | regress |
| 224 | passed | 0.2155 | 3,221,672 | 692,912 | 5796.17 | 0.173 | +0.0000 | +0.02% | +0.00% | -0.42% | equal |
| 256 | passed | 0.2040 | 3,222,664 | 878,256 | 7651.81 | 0.131 | -0.0115 | +0.05% | +26.75% | +31.46% | regress |
| 288 | passed | 0.1983 | 3,224,432 | 1,088,176 | 9772.54 | 0.102 | -0.0172 | +0.11% | +57.04% | +67.90% | regress |
| 320 | passed | 0.1995 | 3,225,624 | 1,322,672 | 12062.99 | 0.083 | -0.0160 | +0.14% | +90.89% | +107.25% | regress |

*Pareto thresholds (from `build_pareto_verdict`): mAP must hold; verdict=`dominates` requires latency >=15% OR size >=20% OR arena >=15% better than iter-A. The iter-B winner rule (below) is stricter than the dominates threshold — it requires the mAP to be within 1 mAP-point of iter-A AND latency >=15% better.*

## Winner — smallest imgsz with mAP held + latency >=15% better
No imgsz beat iter-A by the required >=15% latency margin. Best mAP-holding candidate is imgsz=224 with latency_delta=-0.42% (mAP held within tolerance). **Recommendation**: stay at iter-A's imgsz=224 — input resolution is not the right efficiency lever for this checkpoint without retraining at the smaller crop (future iter-E or a dedicated retrain story).

## Files

- `training/edge/results/iter-B_imgsz_192.json` -> `training/edge/models/sweep_imgsz/yolov8n_cat_distilled_int8_pc_192.tflite`
- `training/edge/results/iter-B_imgsz_224.json` -> `training/edge/models/sweep_imgsz/yolov8n_cat_distilled_int8_pc_224.tflite`
- `training/edge/results/iter-B_imgsz_256.json` -> `training/edge/models/sweep_imgsz/yolov8n_cat_distilled_int8_pc_256.tflite`
- `training/edge/results/iter-B_imgsz_288.json` -> `training/edge/models/sweep_imgsz/yolov8n_cat_distilled_int8_pc_288.tflite`
- `training/edge/results/iter-B_imgsz_320.json` -> `training/edge/models/sweep_imgsz/yolov8n_cat_distilled_int8_pc_320.tflite`

## Reproduce

```sh
# 1. Bootstrap labeled JPEGs + repair val manifest (per iter-A pattern):
python -m training.edge.auto_label bootstrap --src-dir data/cat_photos --out-dir training/edge/data/labeled --target-frames 600 --augments-per-image 25 --seed 42
python -m training.edge.make_dataset_manifest --dataset-dir training/edge/data/labeled
python training/edge/data/regenerate_val_manifest.py

# 2. Run the sweep (default arguments cover the AC list 192,224,256,288,320):
python -m training.edge.yolo.imgsz_sweep

# 3. Re-aggregate the markdown summary:
python training/edge/results/aggregate_iter_b.py
```
