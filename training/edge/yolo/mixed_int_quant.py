"""iter-D: Mixed INT8/INT16 PTQ — INT8 weights with INT16 activations.

Background
----------
v1 + iter-A documented the YOLOv8 PTQ failure mode: the Detect head's
monolithic output ``(1, 5, 1029)`` packs xywh (pixel range 0..imgsz) with
cls (sigmoid 0..1). Per-tensor INT8 activation quant picks one scale
dominated by xywh, crushing cls into a single int8 bucket.

iter-A's float32-OUTPUT dequant fixed the cls collapse on the *output*
tensor by sidestepping per-tensor quant entirely on the way out. iter-D
attacks the same problem from the *internal* side: keep weights INT8 (so
the size cost stays close to iter-A) but promote *activations* to INT16
across the model. INT16 activations have 256× the dynamic range of INT8,
so cls + xywh can co-exist on one per-tensor scale without the cls
saturation that bit iter-A's predecessors.

Path chosen — TF builtin ``EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8``
-----------------------------------------------------------------------------------
The PRD's technical notes call out two viable paths:

    (1) full INT16 activations + INT8 weights via
        ``supported_ops=[EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]``
        — promotes ALL activations to INT16, not just the head, so the
        size/latency cost is higher than the surgical mixed approach asks
        for, but it is a real INT8/INT16 hybrid;
    (2) split-export and link two .tflites — backbone INT8 + head INT16.
        Path (2) is firmware territory.

We take path (1). It is achievable in pure TF 2.17 with a one-line
``supported_ops`` change and remains compatible with the iter-A pipeline
(same calibration loader, same converter shape). Path (2) requires ONNX
graph surgery on ``ultralytics.nn.modules.head.Detect`` plus a
two-binary firmware loader and is left to a future iteration.

TFLM compatibility caveat
-------------------------
TFLM has incomplete INT16 op coverage. The edge-bench harness is
expected to flag the resulting .tflite as ``tflm_compatible=false`` if
any op (e.g. ``CONV_2D`` int16-activation, ``LOGISTIC`` int16) is
missing from the registered op resolver. This is an honest finding the
PRD permits — we still emit the JSON and let the iter-H aggregator show
the row with its blocked reason.

Usage
-----
    python -m training.edge.yolo.mixed_int_quant \\
        --model training/edge/models/yolov8n_cat_distilled.pt \\
        --calib-dir training/edge/data/calibration_frames \\
        --out training/edge/models/yolov8n_cat_distilled_int8w_int16a.tflite \\
        --imgsz 224
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

# Same env opt-in as US-004 / US-006 / iter-A — set BEFORE any keras / tf
# import in the call chain (including the lazy ones inside onnx_to_savedmodel).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Re-use iter-A's calibration loader + onnx export + savedmodel converter.
# calibration_loader is the load-bearing identity contract from progress.txt.
from training.edge.yolo.export_quantize import (  # noqa: E402
    DEFAULT_IMGSZ,
    DEFAULT_MAX_CALIB_FRAMES,
    calibration_loader,
    export_to_onnx,
    onnx_to_savedmodel,
)

DEFAULT_PT = Path("training/edge/models/yolov8n_cat_distilled.pt")
DEFAULT_CALIB_DIR = Path("training/edge/data/calibration_frames")
DEFAULT_OUT = Path("training/edge/models/yolov8n_cat_distilled_int8w_int16a.tflite")


def quantize_to_int16x8_tflite(
    saved_model_dir: Path,
    representative_dataset: Callable[[], Iterable[list[np.ndarray]]],
    out_path: Path,
    converter_factory: Callable[[str], Any] | None = None,
    tf_module: Any = None,
) -> Path:
    """INT16-activations + INT8-weights PTQ converter run.

    Settings (the iter-D delta vs iter-A):

      - optimizations=[tf.lite.Optimize.DEFAULT]
      - representative_dataset=<provided callable>
      - target_spec.supported_ops=[
            EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]                                              (CHANGED from BUILTINS_INT8)
      - inference_input_type=tf.float32                (CHANGED from int8 — keeps the
        existing tflite_adapter compatible without dtype-branch extension; the
        converter prepends a QUANTIZE op that matches what the adapter would
        have done by hand)
      - inference_output_type=tf.float32               (kept from iter-A — preserves cls)
      - experimental_new_quantizer=True                (per-channel weight quant)

    DI hooks (``converter_factory``, ``tf_module``) match iter-A so tests
    in ``test_mixed_int_quant.py`` swap in MagicMocks without TF.
    """
    if tf_module is None:
        import tensorflow as tf  # lazy

        tf_module = tf
    if converter_factory is None:
        converter_factory = tf_module.lite.TFLiteConverter.from_saved_model

    converter = converter_factory(str(saved_model_dir))
    converter.optimizations = [tf_module.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf_module.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    # Use float32 IO so the existing tflite_adapter (built for int8 or float)
    # works without a new int16 dtype branch. The converter inserts a
    # QUANTIZE op at the input and a DEQUANTIZE at the output; both are
    # cheap relative to the body's INT16 conv ops.
    converter.inference_input_type = tf_module.float32
    converter.inference_output_type = tf_module.float32
    if hasattr(converter, "experimental_new_quantizer"):
        converter.experimental_new_quantizer = True

    tflite_bytes = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(tflite_bytes))
    return out_path


def export_mixed_int_quant(
    pt_path: Path = DEFAULT_PT,
    calib_dir: Path = DEFAULT_CALIB_DIR,
    out_path: Path = DEFAULT_OUT,
    imgsz: int = DEFAULT_IMGSZ,
    max_calib_frames: int = DEFAULT_MAX_CALIB_FRAMES,
    work_dir: Path | None = None,
    onnx_exporter: Callable[..., Path] | None = None,
    savedmodel_converter: Callable[..., Path] | None = None,
    quantizer: Callable[..., Path] | None = None,
) -> Path:
    """End-to-end pt -> int8w/int16a .tflite. DI seams on every stage.

    Mirrors training.edge.yolo.per_channel_quant.export_per_channel_int8 —
    same pipeline, only the converter step swapped for the INT16x8 variant.
    """
    if onnx_exporter is None:
        onnx_exporter = export_to_onnx
    if savedmodel_converter is None:
        savedmodel_converter = onnx_to_savedmodel
    if quantizer is None:
        quantizer = quantize_to_int16x8_tflite

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="iter_d_mixed_int_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = onnx_exporter(pt_path, work_dir, imgsz=imgsz)
    saved_model_dir = savedmodel_converter(onnx_path, work_dir / "saved_model")
    rep_ds = calibration_loader(
        calib_dir=calib_dir, imgsz=imgsz, max_frames=max_calib_frames
    )
    return quantizer(saved_model_dir, rep_ds, out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mixed_int_quant")
    ap.add_argument("--model", type=Path, default=DEFAULT_PT)
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--max-calib-frames", type=int, default=DEFAULT_MAX_CALIB_FRAMES)
    ap.add_argument("--work-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    blocked_reason: str | None = None
    out: Path | None = None
    try:
        out = export_mixed_int_quant(
            pt_path=args.model,
            calib_dir=args.calib_dir,
            out_path=args.out,
            imgsz=args.imgsz,
            max_calib_frames=args.max_calib_frames,
            work_dir=args.work_dir,
        )
    except Exception as exc:  # noqa: BLE001 — capture for blocked-reason JSON
        blocked_reason = f"{type(exc).__name__}: {exc}"

    side = (out or args.out).with_suffix((out or args.out).suffix + ".export.json")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(
        json.dumps(
            {
                "pt_path": str(args.model),
                "calib_dir": str(args.calib_dir),
                "tflite_path": str(out or args.out),
                "size_bytes": (
                    out.stat().st_size if (out is not None and out.exists()) else 0
                ),
                "imgsz": args.imgsz,
                "max_calib_frames": args.max_calib_frames,
                "supported_ops": (
                    "EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8"
                ),
                "inference_input_type": "float32",
                "inference_output_type": "float32",
                "experimental_new_quantizer": True,
                "story_id": "iter-D",
                "blocked_reason": blocked_reason,
            },
            indent=2,
        )
    )

    if blocked_reason is not None:
        print(f"BLOCKED: {blocked_reason}")
        return 0  # caller still gets the .export.json sidecar
    sz = out.stat().st_size if out and out.exists() else 0
    print(f"int8w/int16a tflite -> {out} ({sz} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
