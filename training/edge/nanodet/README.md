# NanoDet-Plus integration (US-007 / US-008 / US-009)

Upstream: <https://github.com/RangiLyu/nanodet> (RangiLyu/nanodet, Apache 2.0).

The NanoDet-Plus model family is a candidate edge detector evaluated alongside
the YOLOv8n-cat track. This directory holds the integration scaffolding —
configs, wrapped training entry-points, and a canonical checkpoint path —
without copying the upstream code into the repo. The eval-harness adapter at
[`training/edge/eval/adapters/nanodet_adapter.py`](../eval/adapters/nanodet_adapter.py)
is the only piece in the main env; it lazy-imports either `nanodet` (pth/ckpt
backend) or `onnxruntime` (ONNX backend).

## Why an isolated venv

NanoDet's pinned dependencies in
[`upstream/requirements.txt`](https://github.com/RangiLyu/nanodet/blob/main/requirements.txt)
include `torch>=1.10,<2.0` and `pytorch-lightning>=1.9.0,<2.0.0`. The catzap
main env runs `torch 2.3.0+cu121` plus `ultralytics 8.2.0` (which itself
requires torch ≥2). A shared install would break ultralytics, so the train /
eval-from-pth path runs from `training/edge/nanodet/.venv/` (created by
[`setup_venv.sh`](setup_venv.sh)). The ONNX-backend eval path is main-env-safe.

## Setup

```bash
cd training/edge/nanodet
./setup_venv.sh                          # ~3 min, ~3 GB on disk
source .venv/bin/activate
python -c "from nanodet.util import cfg, load_config"  # smoke check
```

`setup_venv.sh` clones upstream at a pinned commit (recorded in
[`requirements_nanodet.txt`](requirements_nanodet.txt)), installs the pinned
deps, and `pip install -e`'s the upstream package so the module
`nanodet` resolves.

## Pretrained checkpoint (US-007)

Upstream's release page does NOT publish a `nanodet-plus-m_0.5x` pretrained
checkpoint at any input size — only `nanodet-plus-m` (1.0x) at 320 / 416 and
`nanodet-plus-m-1.5x` at 320 / 416. The 0.5x variant exists in the config
tree (legacy v0.x) and is what US-008 fine-tunes from scratch with the
`nanodet-plus-m_416.yml` config scaled to 0.5x.

For US-007's "reproduce upstream baseline" step we therefore use the closest
available published baseline: `nanodet-plus-m_416` (1.0x at 416×416). The
eval through our harness validates the integration end-to-end — the absolute
mAP is informational only because the model is general-purpose COCO and not
cat-finetuned. Documented as a deviation in
[`../results/US-007.md`](../results/US-007.md).

| Asset                                          | URL                                                                                                                  | SHA256                                                             | Size  |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|-------|
| `nanodet-plus-m_416_checkpoint.ckpt` (PyTorch) | <https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt>             | recorded in [`checkpoints/SHA256SUMS`](checkpoints/SHA256SUMS)     | ~22 MB |
| `nanodet-plus-m_416.onnx` (ONNX export)        | <https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx>                        | recorded in [`checkpoints/SHA256SUMS`](checkpoints/SHA256SUMS)     | ~10 MB |

The `.ckpt` is canonically copied to
`training/edge/nanodet/checkpoints/nanodet_plus_m_0.5x_pretrained.pth` so the
PRD-shaped command line (`--model
training/edge/nanodet/checkpoints/nanodet_plus_m_0.5x_pretrained.pth --format
pytorch`) resolves; the actual checkpoint file is the 1.0x baseline per the
deviation above. Filename is preserved for spec consistency; the deviation is
encoded in `EvalResult.notes` and `US-007.md`.

Both files are gitignored (regenerable from the pinned URLs).

## Reproducing the eval

ONNX backend (main env, no venv needed):

```bash
python -m training.edge.eval.run_eval \
    --model training/edge/nanodet/checkpoints/nanodet_plus_m_416.onnx \
    --format onnx \
    --story-id US-007 \
    --val-dir training/edge/data/labeled/val/ \
    --imgsz 416
```

PyTorch / ckpt backend (requires the isolated venv):

```bash
source training/edge/nanodet/.venv/bin/activate
python -m training.edge.eval.run_eval \
    --model training/edge/nanodet/checkpoints/nanodet_plus_m_0.5x_pretrained.pth \
    --format pytorch \
    --story-id US-007 \
    --val-dir training/edge/data/labeled/val/ \
    --imgsz 416
```

Both routes run through `nanodet_adapter.NanodetAdapter`, which dispatches on
file extension. Class outputs are filtered to COCO `class_id=15` (cat) and
re-emitted as our single-class `id=0`, mirroring the auto-labeler's class
remap (US-002).

## Outputs landed in the repo

* `configs/nanodet_plus_m_0.5x_cat.yml` — placeholder, populated by US-008
* `checkpoints/SHA256SUMS` — SHA-256 of the downloaded artifacts
* `setup_venv.sh` — venv bootstrap
* `requirements_nanodet.txt` — pinned upstream deps
