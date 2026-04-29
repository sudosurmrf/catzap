"""iter-F: per-channel-friendly INT8 PTQ for NanoDet-Plus.

Reuses the iter-A YOLOv8 quantizer (``inference_output_type=tf.float32`` plus
``experimental_new_quantizer=True``) on the NanoDet pipeline so the cls
collapse documented in v1's progress.txt — "Per-tensor INT8 output scale
collapses NanoDet-Plus cls scores" — gets the same converter-side fix that
worked for YOLOv8 in iter-A.

Background
----------
v1's US-009 ran NanoDet through ``training.edge.nanodet.export_quantize`` with
``inference_output_type=tf.int8``. NanoDet-Plus' single output tensor
``(1, num_priors, num_classes + 4*(reg_max+1))`` packs cls (sigmoid in [0,1])
alongside reg-DFL bins (unbounded logits). One per-tensor int8 output scale
spans both, reg dominates, cls saturates near zero post-dequant. mAP@0.5
collapsed from 0.704 fp32 → 0.000 int8 at default conf threshold.

iter-F flips one knob: ``inference_output_type=tf.float32``. The internal
weights stay INT8 with per-channel scales (MLIR new quantizer); externally a
DEQUANTIZE op gets appended after the head's int8 output tensor so cls reads
out at full float resolution. Same fix as iter-A; the question is whether it
recovers NanoDet's fp32 mAP the way it did for the (KD-shaped) YOLOv8 head.

The pipeline is otherwise byte-identical to v1's US-009: ONNX -> SavedModel
via onnx2tf NHWC, calibration_loader imported (``is``-identity) from
``training.edge.yolo.export_quantize`` so YOLO and NanoDet draw from the same
representative dataset.

Usage
-----
    python -m training.edge.nanodet.per_channel_quant \\
        --onnx training/edge/nanodet/checkpoints/nanodet_plus_m_0.5x_pretrained.onnx \\
        --calib-dir training/edge/data/calibration_frames \\
        --out training/edge/models/nanodet_cat_0.5x_224_int8_pc.tflite \\
        --imgsz 224

The CLI also writes ``<out>.export.json`` with the pipeline parameters so the
sibling iter-F aggregator can locate the .tflite and its provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

# TF_USE_LEGACY_KERAS=1 must be set BEFORE any keras/tf import, including the
# lazy ones inside onnx_to_savedmodel — see v1 progress.txt pattern
# "TF_USE_LEGACY_KERAS=1 is mandatory before onnx2tf".
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Re-export the YOLO calibration_loader so iter-F's representative_dataset is
# the SAME callable that drove iter-A INT8 quantization. The
# test_nanodet_per_channel.py acceptance test asserts identity (`is`).
from training.edge.nanodet.export_quantize import (  # noqa: E402, F401
    DEFAULT_CONFIG,
    DEFAULT_PTH,
    export_pth_to_onnx,
)
from training.edge.yolo.export_quantize import (  # noqa: E402
    DEFAULT_CALIB_DIR,
    DEFAULT_MAX_CALIB_FRAMES,
    calibration_loader,
    onnx_to_savedmodel,
)
from training.edge.yolo.per_channel_quant import (  # noqa: E402
    quantize_to_int8_float_output_tflite,
)

DEFAULT_ONNX = Path(
    "training/edge/nanodet/checkpoints/nanodet_plus_m_0.5x_pretrained.onnx"
)
DEFAULT_OUT = Path("training/edge/models/nanodet_cat_0.5x_224_int8_pc.tflite")
DEFAULT_IMGSZ = 224


def export_per_channel_int8(
    onnx_path: Path | None = None,
    pth_path: Path = DEFAULT_PTH,
    config_path: Path = DEFAULT_CONFIG,
    calib_dir: Path = DEFAULT_CALIB_DIR,
    out_path: Path = DEFAULT_OUT,
    imgsz: int = DEFAULT_IMGSZ,
    max_calib_frames: int = DEFAULT_MAX_CALIB_FRAMES,
    work_dir: Path | None = None,
    onnx_exporter: Callable[..., Path] | None = None,
    savedmodel_converter: Callable[..., Path] | None = None,
    quantizer: Callable[..., Path] | None = None,
) -> Path:
    """End-to-end NanoDet -> int8-with-float-output .tflite. DI seams on every stage.

    If ``onnx_path`` is provided, the .pth -> ONNX step is skipped and the
    pipeline starts from the existing ONNX (the typical iter-F path: re-quant
    the pretrained ONNX from US-007 outside the nanodet venv).

    The default ``quantizer`` is :func:`training.edge.yolo.per_channel_quant.quantize_to_int8_float_output_tflite`
    — the same callable iter-A uses, so the converter settings are identical
    across YOLO and NanoDet INT8 paths.
    """
    if onnx_exporter is None:
        onnx_exporter = export_pth_to_onnx
    if savedmodel_converter is None:
        savedmodel_converter = onnx_to_savedmodel
    if quantizer is None:
        quantizer = quantize_to_int8_float_output_tflite

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="iter_f_nanodet_pcquant_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    if onnx_path is None:
        onnx_path = onnx_exporter(
            pth_path, config_path, work_dir / f"{pth_path.stem}.onnx", imgsz=imgsz
        )
    else:
        copied = work_dir / Path(onnx_path).name
        if Path(onnx_path).resolve() != copied.resolve():
            shutil.copy2(onnx_path, copied)
        onnx_path = copied

    saved_model_dir = savedmodel_converter(onnx_path, work_dir / "saved_model")
    rep_ds = calibration_loader(
        calib_dir=calib_dir, imgsz=imgsz, max_frames=max_calib_frames
    )
    return quantizer(saved_model_dir, rep_ds, out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nanodet_per_channel_quant")
    ap.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX,
        help="path to NanoDet ONNX (skips .pth->ONNX step)",
    )
    ap.add_argument(
        "--model", type=Path, default=DEFAULT_PTH, help="path to .pth/.ckpt"
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument(
        "--max-calib-frames", type=int, default=DEFAULT_MAX_CALIB_FRAMES
    )
    ap.add_argument(
        "--work-dir", type=Path, default=None,
        help="optional; defaults to a tempdir under /tmp",
    )
    args = ap.parse_args(argv)

    out = export_per_channel_int8(
        onnx_path=args.onnx,
        pth_path=args.model,
        config_path=args.config,
        calib_dir=args.calib_dir,
        out_path=args.out,
        imgsz=args.imgsz,
        max_calib_frames=args.max_calib_frames,
        work_dir=args.work_dir,
    )
    side = out.with_suffix(out.suffix + ".export.json")
    side.write_text(
        json.dumps(
            {
                "onnx_path": str(args.onnx) if args.onnx else None,
                "pth_path": str(args.model),
                "config_path": str(args.config),
                "calib_dir": str(args.calib_dir),
                "tflite_path": str(out),
                "size_bytes": out.stat().st_size if out.exists() else 0,
                "imgsz": args.imgsz,
                "max_calib_frames": args.max_calib_frames,
                "inference_input_type": "int8",
                "inference_output_type": "float32",
                "experimental_new_quantizer": True,
                "story_id": "iter-F",
            },
            indent=2,
        )
    )
    sz = out.stat().st_size if out.exists() else 0
    print(f"int8-pc nanodet tflite -> {out} ({sz} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
