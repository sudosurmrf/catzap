# Edge Model Reduction PoC — Product Requirements

## Overview

**Problem**: The catzap on-device cat detector currently runs a 22 MB YOLOv8s model (`server/vision/detector.py:8`) on the server with a 1280-pixel input. This is far too large to fit on an ESP32-S3 / ESP32-P4 class board (typically 8 MB PSRAM, ~512 KB SRAM, no GPU, INT8-only inference via TFLite Micro). New ESP32-S3/P4 hardware arrives in a few days, and we have nothing ready to flash.

**Solution**: A self-contained, parallel proof-of-concept under `training/edge/` that (a) shrinks YOLOv8n cat-only via fine-tune + INT8 quantization (and optional QAT / KD), (b) evaluates NanoDet-Plus-m 0.5x as an alternate architecture, and (c) emulates TFLite Micro execution on the host so we can predict ESP32-S3 fps/RAM before the boards arrive. Everything runs on existing developer hardware (CPU/GPU server + AI-Thinker ESP32-CAM streaming MJPEG VGA) and produces a final decision matrix recommending which model to flash first.

**Branch**: `ralph/edge-model-poc`

---

## Goals & Success

### Primary Goal

Produce a quantized cat-only detector (and a host-side TFLM benchmark for it) that the team can flash to the ESP32-S3 immediately on arrival, with measured evidence that it will fit in flash + PSRAM and run at a usable frame rate.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|--------------|
| Final model size on disk | <= 2.0 MB INT8 .tflite | `os.path.getsize` recorded by eval harness |
| mAP@0.5 vs YOLOv8s baseline | within 5 points absolute | `training/edge/eval/` harness on held-out cat val set |
| Host-CPU INT8 latency (224x224) | <= 250 ms / frame on a single core | TFLite Python interpreter, single-thread, median of 100 runs |
| Predicted ESP32-S3 tensor arena | <= 4 MB PSRAM | TFLM x86 host build reports peak arena |
| Models compared end-to-end | 2 (quantized YOLOv8n-cat, quantized NanoDet-Plus-m-cat) | `training/edge/results/SUMMARY.md` decision matrix |
| Auto-labeled cat training set | >= 500 frames | Count of YOLO-format `.txt` labels in dataset dir |
| Existing server detector unchanged | yes | `git diff main -- server/vision/detector.py server/vision/classifier.py` is empty at PR time |

### Non-Goals (Out of Scope)

- **Modifying the live server detector** — `server/vision/detector.py` and `server/vision/classifier.py` stay untouched. Reason: we cannot afford to destabilize the running rig while the PoC is in flight.
- **Real ESP32-S3 / P4 firmware** — boards arrive after this PRD lands. We only emulate via TFLite Micro x86 host build. A separate PRD will own real-board integration.
- **Re-identification / classifier shrinking** — the MobileNet-V3-Small classifier (`server/vision/classifier.py`) stays server-side. Edge work is detection-only.
- **Multi-class support** — the edge model is single-class "cat". Other COCO classes are dropped.
- **Replacing the auto-label pipeline with manual labeling at scale** — we use the existing YOLOv8s detector as a teacher labeler with a small manual-cleanup pass. Reason: <500 frames is enough for fine-tuning a single-class detector and we don't have human labelers.
- **Quantization-aware training as a default** — only run QAT if PTQ mAP drop > 3 points (Story US-005). Reason: QAT is expensive and PTQ is usually sufficient for single-class.
- **Training infrastructure refactor** — we don't extend `server/vision/trainer.py`. The new training scripts live in `training/edge/` and use ultralytics + nanodet upstream training loops directly.

---

## User & Context

### Target User

- **Who**: Catzap project maintainer (Ari) and a small dev team prepping the rig for off-the-shelf ESP32-S3 / P4 deployment.
- **Role**: Developer running training jobs on a workstation, then flashing firmware to dev boards.
- **Current Pain**: No model exists that's small enough for the new boards. YOLOv8s is ~22 MB, doesn't quantize cleanly, doesn't fit in PSRAM. The team wants two candidate models with empirical evidence so the first flash isn't a guess.

### User Journey

1. **Trigger**: Boards arrive in ~3-5 days. The maintainer needs a quantized .tflite + a benchmark report ready to flash on day 1.
2. **Action**: Maintainer runs `training/edge/` scripts in order (Stories US-001 -> US-012), each story producing a JSON metric file + short markdown report. They review `training/edge/results/SUMMARY.md` to pick a winner.
3. **Outcome**: Two candidate INT8 .tflite models, a host TFLM benchmark predicting ESP32-S3 fps and arena size for each, and a written recommendation. PR merged before boards arrive.

---

## UX Requirements

### Interaction Model

**CLI-only.** This is a research / training PoC, not an end-user feature. Each story produces:

- A Python CLI entry point under `training/edge/` (e.g. `python -m training.edge.eval.run_eval --model path/to/model.tflite`)
- A machine-readable result file at `training/edge/results/<story-id>.json`
- A short human-readable markdown report at `training/edge/results/<story-id>.md`

No HTTP endpoints, no UI, no integration with the live server.

### States to Handle

| State | Description | Behavior |
|-------|-------------|----------|
| Empty | No labeled dataset yet (before US-002) | Eval harness uses synthetic / COCO-cat val frames as a placeholder; logs warning |
| Loading | Model file too large to load on host | Eval harness fails fast with a clear error citing model size |
| Error | Model output shape mismatch (NanoDet vs YOLO heads) | Eval harness has pluggable post-processors keyed off model name; raises if no matching post-processor |
| Success | All metrics computed | Writes `<story-id>.json` with mAP, size_bytes, params, flops, latency_ms_p50, latency_ms_p95 |

### Edge cases

- Auto-labeler producing zero detections on a frame -> frame is skipped (not labeled "no cat"), logged, count tracked.
- Calibration set smaller than 200 frames -> fall back to whatever's available, log warning.
- TFLM x86 host build missing -> US-010 acceptance fails; do not silently skip.
- mAP delta after PTQ is implausibly small (< 0.1) -> log warning that quantization may not have actually executed.

---

## Technical Context

### Patterns to Follow

- **Detector class shape**: `server/vision/detector.py:7-40` — single class (`CatDetector`), `__init__(model_path, ...)`, `detect(frame) -> list[dict]` with `{"bbox": [x1,y1,x2,y2 normalized], "confidence": float}`. Mirror this signature for any new edge inference wrappers so the eval harness stays consistent.
- **COCO cat class constant**: `server/vision/detector.py:4` defines `COCO_CAT_CLASS = 15`. Reuse this in the auto-labeler when filtering teacher detections.
- **Existing training loop**: `server/vision/trainer.py:70-174` — pattern of `train_<thing>(input, output_path, epochs, batch_size, lr) -> status_dict`, with a global `training_status` for progress polling. Mirror the function signature and JSON status shape for new training scripts even though we're not exposing them via FastAPI.
- **Test pattern**: `server/tests/test_detector.py` — pytest + `unittest.mock.@patch`, mock the `ultralytics.YOLO` class and assert on returned dicts. New `training/edge/tests/` mirrors this.
- **Existing model files at repo root**: `yolov8s.pt` (22 MB) and `yolov8n.pt` (6.5 MB) already sit at repo root. Use these as the starting checkpoints for stories US-001 and US-003.
- **ESP32-CAM stream**: `firmware/esp32-cam/src/main.cpp:33-58, 121` exposes `GET /stream` as MJPEG multipart (boundary `frame`). VGA 640x480, JPEG quality 12. Pull frames via `httpx` or `cv2.VideoCapture("http://<ip>/stream")`.

### Types & Interfaces

```python
# Eval harness common types — define under training/edge/eval/types.py
from dataclasses import dataclass
from typing import Literal

ModelFormat = Literal["pytorch", "onnx", "tflite_fp32", "tflite_int8"]

@dataclass
class EvalResult:
    story_id: str           # e.g. "US-001-yolov8s-baseline"
    model_path: str
    model_format: ModelFormat
    map50: float            # mAP@0.5 on val set
    size_bytes: int         # os.path.getsize(model_path)
    params: int             # parameter count
    flops: int              # FLOPs at the eval input size
    input_hw: tuple[int, int]  # (height, width) e.g. (224, 224) for edge runs
    latency_ms_p50: float   # single-thread host CPU INT8 latency
    latency_ms_p95: float
    val_images: int         # number of val frames used
    notes: str              # free-form
```

```python
# Auto-labeler output — YOLO format on disk, per-frame .txt next to .jpg
# Each line: "<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>"
# class_id is 0 (single-class "cat")
```

### Architecture Notes

- **Strict isolation**: All new code lives under `training/edge/` and `firmware/edge-bench/`. The only "read-from" of existing code is the `CatDetector` class itself, used by the auto-labeler. No edits to `server/vision/detector.py`, `server/vision/classifier.py`, `server/vision/trainer.py`, or `server/main.py`.
- **Separate Python deps**: `training/edge/requirements_edge.txt` adds `onnx`, `onnxruntime`, `tensorflow` (CPU, for tf.lite converter and TFLM tooling), and `nanodet-plus`. Keep `server/requirements.txt` unchanged. NanoDet may need its own venv (`training/edge/nanodet/.venv/`) — that's allowed by the input spec.
- **Eval harness is the integration spine**: Every story past US-001 must end by writing a result row through the same `EvalResult` dataclass to `training/edge/results/<story-id>.json`. This is what makes the final SUMMARY.md decision matrix mechanical.
- **Calibration set**: Stored at `training/edge/data/calibration_frames/` (~200 representative ESP32-CAM frames captured during US-002). Reused across PTQ for both YOLOv8n (US-004) and NanoDet (US-009).
- **TFLM x86 host build (US-010)**: Use upstream `https://github.com/tensorflow/tflite-micro` `tools/make/Makefile` with `TARGET=linux` to produce `genericbenchmark` or a custom binary that loads a .tflite, runs N inferences, and prints op-by-op timings + `MicroInterpreter::arena_used_bytes()`. Wrapped by a Python harness in `firmware/edge-bench/`.
- **Single PR at the end**: All 12 stories ship together in one PR off `ralph/edge-model-poc`.

### Dependencies on existing files (read-only)

- `server/vision/detector.py` — imported by US-002 auto-labeler to use the existing YOLOv8s as teacher. **Read only.**
- `server/vision/trainer.py` — pattern reference only, not imported.
- `firmware/esp32-cam/src/main.cpp:121` — `/stream` endpoint, frame source for US-002. Read only.
- `yolov8s.pt`, `yolov8n.pt` (repo root) — used as starting checkpoints for baseline + fine-tune.

---

## Implementation Summary

### Story Overview

| ID | Title | Priority | Dependencies |
|----|-------|----------|--------------|
| US-001 | Build eval harness + record YOLOv8s/n baselines | 1 | — |
| US-002 | Auto-label ESP32-CAM frames into YOLO-format dataset | 2 | US-001 |
| US-003 | Fine-tune YOLOv8n single-class cat | 3 | US-001, US-002 |
| US-004 | PTQ: ONNX -> TFLite INT8 with calibration | 4 | US-001, US-003 |
| US-005 | Conditional QAT if PTQ mAP drop > 3 points | 5 | US-003, US-004 |
| US-006 | Knowledge distillation: YOLOv8s -> YOLOv8n cat-only | 6 | US-002, US-003, US-004 |
| US-007 | Set up NanoDet-Plus-m 0.5x environment + reproduce baseline | 7 | US-001 |
| US-008 | Fine-tune NanoDet-Plus-m 0.5x cat-only at 224x224 | 8 | US-002, US-007 |
| US-009 | NanoDet ONNX -> TFLite INT8 with calibration | 9 | US-001, US-008 |
| US-010 | TFLite Micro x86 emulator/benchmark harness | 10 | US-001 |
| US-011 | Run TFLM emulator on quantized YOLOv8n + NanoDet | 11 | US-004, US-009, US-010 |
| US-012 | Final SUMMARY.md decision matrix + recommendation | 12 | US-001..US-011 |

### Dependency Graph

```
US-001 (eval harness + baselines)
   |
   +--> US-002 (auto-label dataset)
   |       |
   |       +--> US-003 (fine-tune YOLOv8n)
   |       |       |
   |       |       +--> US-004 (PTQ -> TFLite INT8) ----+
   |       |       |       |                            |
   |       |       |       +--> US-005 (QAT if needed)  |
   |       |       |                                    |
   |       |       +--> US-006 (KD YOLOv8s -> n)        |
   |       |                                            |
   |       +--> US-008 (fine-tune NanoDet) -> US-009 (NanoDet INT8) -+
   |                                                                 |
   +--> US-007 (NanoDet env + baseline)                              |
   |                                                                 |
   +--> US-010 (TFLM x86 harness) -> US-011 (run on US-004 + US-009)-+
                                                                     |
                                                                     v
                                                          US-012 (SUMMARY + decision)
```

---

## Validation Requirements

Every story must pass:

- [ ] `pytest training/edge/tests/` — story-specific tests pass (mirror `server/tests/test_detector.py` patterns)
- [ ] `python -m training.edge.eval.run_eval --story <story-id>` produces a `training/edge/results/<story-id>.json` matching the `EvalResult` schema
- [ ] No diff on `server/vision/detector.py`, `server/vision/classifier.py`, `server/vision/trainer.py`, `server/main.py` (`git diff main -- server/` should be empty for those files)
- [ ] `python -c "from server.vision.detector import CatDetector; CatDetector('yolov8s.pt')"` still works (import smoke test for the live server detector)
- [ ] Story-specific markdown report written at `training/edge/results/<story-id>.md` summarizing what changed and the metric delta vs the previous step

Final-PR validation:

- [ ] `training/edge/results/SUMMARY.md` exists and contains the decision matrix
- [ ] `git diff main -- server/vision/ server/main.py` is empty
- [ ] `pytest server/tests/` still passes (existing server tests untouched)
- [ ] Two `.tflite` INT8 artifacts exist under `training/edge/models/` (one YOLOv8n-cat, one NanoDet-cat)
- [ ] TFLM x86 benchmark binary builds and produces non-zero arena reports for both

---

*Generated: 2026-04-28*
