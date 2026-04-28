"""US-009: NanoDet-Plus-m 0.5x cat .pth -> ONNX -> INT8 TFLite.

Pipeline:

    pth/ckpt --[wraps upstream tools/export_onnx.py]--> ONNX
    ONNX     --[onnx2tf NHWC]-->  SavedModel
    SavedModel --[tf.lite.TFLiteConverter int8]--> .tflite

The ONNX -> SavedModel and SavedModel -> TFLite stages reuse the YOLO US-004
helpers verbatim by importing them from :mod:`training.edge.yolo.export_quantize`
so the converter settings (``Optimize.DEFAULT``, ``BUILTINS_INT8``,
``inference_input_type=tf.int8``, ``inference_output_type=tf.int8``,
``experimental_new_quantizer=True``) are guaranteed identical between the YOLO
and NanoDet INT8 paths. **The calibration_loader is imported directly** from
``training.edge.yolo.export_quantize`` rather than redefined locally — this is
required by US-009's spec so the apples-to-apples comparison required by US-012
is well-defined.

The ``.pth -> ONNX`` step wraps :mod:`nanodet.tools.export_onnx` (the upstream
script). NanoDet's package and pytorch-lightning pin ``torch<2.0``, so the
default exporter lazy-imports nanodet and is intended to run inside the
isolated venv from ``training/edge/nanodet/setup_venv.sh``. Tests drive a
``onnx_exporter`` DI seam to keep the suite import-safe in the main env.

Usage (inside the isolated venv, after US-008 has produced the cat .pth):

    cd training/edge/nanodet && source .venv/bin/activate && cd ../../..
    python -m training.edge.nanodet.export_quantize \\
        --model training/edge/nanodet/checkpoints/nanodet_cat_0.5x_224.pth \\
        --config training/edge/nanodet/configs/nanodet_plus_m_0.5x_cat.yml \\
        --calib-dir training/edge/data/calibration_frames \\
        --out training/edge/models/nanodet_cat_0.5x_224_int8.tflite \\
        --imgsz 224

Outside the venv, the orchestrator still works against an existing ONNX file
(``--onnx <path>``) so the pipeline can be smoke-tested on the US-007
pretrained ONNX without nanodet installed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Re-export the YOLO calibration_loader so US-009's representative_dataset is
# the SAME callable that drove US-004 / US-006 INT8 quantization. The
# test_nanodet_quantize.py acceptance test asserts identity (`is`).
from training.edge.yolo.export_quantize import (  # noqa: F401 (re-export)
    DEFAULT_CALIB_DIR,
    DEFAULT_MAX_CALIB_FRAMES,
    calibration_loader,
    onnx_to_savedmodel,
    quantize_to_int8_tflite,
)

DEFAULT_PTH = Path(
    "training/edge/nanodet/checkpoints/nanodet_cat_0.5x_224.pth"
)
DEFAULT_CONFIG = Path(
    "training/edge/nanodet/configs/nanodet_plus_m_0.5x_cat.yml"
)
DEFAULT_OUT = Path("training/edge/models/nanodet_cat_0.5x_224_int8.tflite")
DEFAULT_IMGSZ = 224


def export_pth_to_onnx(
    pth_path: Path,
    config_path: Path,
    out_path: Path,
    imgsz: int = DEFAULT_IMGSZ,
    nanodet_export_main: Callable[..., None] | None = None,
) -> Path:
    """Wrap upstream ``nanodet.tools.export_onnx.main`` to produce an ONNX file.

    The upstream entry point has signature
    ``main(config, model_path, output_path, input_shape=(H, W))``. We lazy-import
    it so this module stays importable in the catzap main env (where nanodet
    isn't installed). For tests, pass ``nanodet_export_main`` directly.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if nanodet_export_main is None:
        try:
            from nanodet.util import cfg as nano_cfg, load_config  # type: ignore
            from nanodet.tools import export_onnx as _eo  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "NanoDet ONNX export requires the isolated venv "
                "(see training/edge/nanodet/setup_venv.sh). "
                "For main-env testing pass an ONNX directly via --onnx."
            ) from exc

        load_config(nano_cfg, str(config_path))
        _eo.main(
            nano_cfg, str(pth_path), str(out_path), input_shape=(int(imgsz), int(imgsz))
        )
    else:
        nanodet_export_main(
            config_path=str(config_path),
            model_path=str(pth_path),
            output_path=str(out_path),
            input_shape=(int(imgsz), int(imgsz)),
        )
    return out_path


def export_and_quantize(
    pth_path: Path = DEFAULT_PTH,
    config_path: Path = DEFAULT_CONFIG,
    onnx_path: Path | None = None,
    calib_dir: Path = DEFAULT_CALIB_DIR,
    out_path: Path = DEFAULT_OUT,
    imgsz: int = DEFAULT_IMGSZ,
    max_calib_frames: int = DEFAULT_MAX_CALIB_FRAMES,
    work_dir: Path | None = None,
    onnx_exporter: Callable[..., Path] | None = None,
    savedmodel_converter: Callable[..., Path] | None = None,
    quantizer: Callable[..., Path] | None = None,
) -> Path:
    """End-to-end pth -> int8 .tflite. Each stage is DI-overridable.

    If ``onnx_path`` is provided, the .pth -> ONNX step is skipped and the
    pipeline starts from the existing ONNX. This is how the US-007 pretrained
    ONNX gets quantized as a smoke-test of the rest of the pipeline without
    needing the isolated nanodet venv.
    """
    if onnx_exporter is None:
        onnx_exporter = export_pth_to_onnx
    if savedmodel_converter is None:
        savedmodel_converter = onnx_to_savedmodel
    if quantizer is None:
        quantizer = quantize_to_int8_tflite

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="us009_nanodet_quant_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    if onnx_path is None:
        onnx_path = onnx_exporter(
            pth_path, config_path, work_dir / f"{pth_path.stem}.onnx", imgsz=imgsz
        )
    else:
        # If a pre-existing ONNX is supplied, copy it into the work dir so the
        # downstream SavedModel sits next to it — mirrors export_pth_to_onnx.
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
    ap = argparse.ArgumentParser(prog="nanodet_export_quantize")
    ap.add_argument("--model", type=Path, default=DEFAULT_PTH, help="path to .pth/.ckpt")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="skip .pth->ONNX and start from this ONNX (used outside the nanodet venv)",
    )
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument(
        "--max-calib-frames", type=int, default=DEFAULT_MAX_CALIB_FRAMES
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="optional; defaults to a tempdir under /tmp",
    )
    args = ap.parse_args(argv)

    out = export_and_quantize(
        pth_path=args.model,
        config_path=args.config,
        onnx_path=args.onnx,
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
                "pth_path": str(args.model),
                "config_path": str(args.config),
                "onnx_path": str(args.onnx) if args.onnx else None,
                "calib_dir": str(args.calib_dir),
                "tflite_path": str(out),
                "size_bytes": out.stat().st_size if out.exists() else 0,
                "imgsz": args.imgsz,
                "max_calib_frames": args.max_calib_frames,
            },
            indent=2,
        )
    )
    print(f"int8 tflite -> {out} ({out.stat().st_size if out.exists() else 0} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
