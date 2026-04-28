"""US-004: PTQ pipeline pt -> ONNX -> TF SavedModel -> INT8 TFLite.

Pipeline path chosen (criterion: "loads the resulting ONNX in tf.lite via
onnx-tf or via the tf2onnx reverse path; document chosen path"):

    pt --[ultralytics .export(format='onnx', imgsz=224, dynamic=False, simplify=True)]-->
    onnx --[onnx2tf]--> SavedModel (NHWC) --[tf.lite.TFLiteConverter]--> int8 .tflite

We use ``onnx2tf`` rather than ``onnx-tf`` because onnx2tf produces a TF graph
with native NHWC ordering (the layout TFLite Micro on the ESP32-S3 expects),
while onnx-tf preserves the ONNX NCHW ordering and would force a transpose
wrapper. The calibration loader therefore yields (1, 224, 224, 3) float32 [0,1]
tensors directly and feeds them into the converter as ``representative_dataset``
without any intermediate reshape — see ``calibration_loader`` for details.

Usage:
    python -m training.edge.yolo.export_quantize \\
        --model training/edge/models/yolov8n_cat.pt \\
        --calib-dir training/edge/data/calibration_frames \\
        --out training/edge/models/yolov8n_cat_int8.tflite \\
        --imgsz 224

The CLI also writes a side-car JSON ``<out>.export.json`` with the ONNX path,
SavedModel dir, and final size_bytes — convenient for downstream stories
(US-006 distillation re-uses this same pipeline; US-009 NanoDet quantization
re-uses ``calibration_loader`` from this module).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

# onnx2tf builds a tf.keras model under the hood. Newer TF ships Keras 3 by
# default, but onnx2tf was written for Keras 2 — it crashes with "A KerasTensor
# cannot be used as input to a TensorFlow function" without this opt-in. Must
# be set before any keras/tensorflow import inside the call chain.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

DEFAULT_PT = Path("training/edge/models/yolov8n_cat.pt")
DEFAULT_CALIB_DIR = Path("training/edge/data/calibration_frames")
DEFAULT_OUT = Path("training/edge/models/yolov8n_cat_int8.tflite")
DEFAULT_IMGSZ = 224
DEFAULT_MAX_CALIB_FRAMES = 200


def _list_calib_images(calib_dir: Path, max_frames: int) -> list[Path]:
    """Sorted list of <= max_frames jpg/png files under calib_dir."""
    if not calib_dir.exists():
        raise FileNotFoundError(f"calibration dir not found: {calib_dir}")
    images = sorted(
        p
        for p in calib_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise FileNotFoundError(f"no calibration images in {calib_dir}")
    return images[:max_frames]


def calibration_loader(
    calib_dir: Path = DEFAULT_CALIB_DIR,
    imgsz: int = DEFAULT_IMGSZ,
    max_frames: int = DEFAULT_MAX_CALIB_FRAMES,
) -> Callable[[], Iterable[list[np.ndarray]]]:
    """Build a representative_dataset callable for tf.lite.TFLiteConverter.

    Yields ``[ndarray]`` (singleton list of input tensors) per call, where
    each ndarray has shape ``(1, imgsz, imgsz, 3)`` (NHWC), dtype float32,
    values in ``[0, 1]``. This is what tf.lite expects for an INT8 PTQ run.

    Re-used by US-009 (NanoDet quantization) — keep the implementation pure
    so the calibration set is identical across model families.
    """
    image_paths = _list_calib_images(calib_dir, max_frames)

    def _gen() -> Iterable[list[np.ndarray]]:
        import cv2  # lazy import; tests don't need cv2

        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if frame.shape[:2] != (imgsz, imgsz):
                frame = cv2.resize(frame, (imgsz, imgsz))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = rgb.astype(np.float32) / 255.0
            x = x[None, ...]  # NHWC: (1, imgsz, imgsz, 3)
            yield [x]

    return _gen


def export_to_onnx(
    pt_path: Path,
    out_dir: Path,
    imgsz: int = DEFAULT_IMGSZ,
    yolo_factory: Any = None,
) -> Path:
    """Export an ultralytics .pt checkpoint to ONNX (dynamic=False, simplify=True).

    Returns the path to the .onnx file. The ONNX is placed inside ``out_dir``
    using the .pt's stem so the SavedModel and TFLite siblings can be located
    from the same prefix.
    """
    if yolo_factory is None:
        from ultralytics import YOLO  # lazy

        yolo_factory = YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    model = yolo_factory(str(pt_path))
    onnx_str = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=True,
    )
    src = Path(onnx_str)
    dest = out_dir / f"{pt_path.stem}.onnx"
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def onnx_to_savedmodel(
    onnx_path: Path,
    out_dir: Path,
    converter: Callable[..., Any] | None = None,
) -> Path:
    """Convert an ONNX file to a TF SavedModel directory (NHWC) via onnx2tf.

    The onnx2tf default layout transformer takes care of NCHW->NHWC. We pass
    --output_signaturedefs so the resulting SavedModel can be consumed by
    ``tf.lite.TFLiteConverter.from_saved_model`` without further surgery.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if converter is None:
        import onnx2tf  # lazy

        converter = onnx2tf.convert
    converter(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(out_dir),
        output_signaturedefs=True,
        copy_onnx_input_output_names_to_tflite=False,
        non_verbose=True,
    )
    return out_dir


def quantize_to_int8_tflite(
    saved_model_dir: Path,
    representative_dataset: Callable[[], Iterable[list[np.ndarray]]],
    out_path: Path,
    converter_factory: Callable[[str], Any] | None = None,
    tf_module: Any = None,
) -> Path:
    """Run the explicit tf.lite.TFLiteConverter -> INT8 TFLite step.

    Settings (per US-004 acceptance criteria):
      - optimizations=[tf.lite.Optimize.DEFAULT]
      - representative_dataset=<provided callable>
      - target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
      - inference_input_type=tf.int8
      - inference_output_type=tf.int8

    Both ``converter_factory`` and ``tf_module`` are dependency-injected so
    the test in test_quantize.py can swap in a MagicMock without TF installed.
    """
    if tf_module is None:
        import tensorflow as tf  # lazy

        tf_module = tf
    if converter_factory is None:
        converter_factory = tf_module.lite.TFLiteConverter.from_saved_model

    converter = converter_factory(str(saved_model_dir))
    converter.optimizations = [tf_module.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf_module.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf_module.int8
    converter.inference_output_type = tf_module.int8
    # MLIR-based per-channel quantizer reduces accuracy loss on YOLO-style
    # heads where xywh (pixel range) and cls (sigmoid 0-1) share an output
    # tensor. The legacy TOCO quantizer collapses the cls channel because
    # it picks a single tensor scale dominated by xywh.
    if hasattr(converter, "experimental_new_quantizer"):
        converter.experimental_new_quantizer = True

    tflite_bytes = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(tflite_bytes))
    return out_path


def export_and_quantize(
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
    """End-to-end pt -> int8 .tflite. Each stage is DI-overridable for tests.

    Steps:
      1. ``onnx_exporter(pt_path, work_dir, imgsz)`` -> ONNX file
      2. ``savedmodel_converter(onnx_path, work_dir / 'saved_model')`` -> SavedModel dir
      3. ``quantizer(saved_model_dir, representative_dataset, out_path)`` -> .tflite
    """
    if onnx_exporter is None:
        onnx_exporter = export_to_onnx
    if savedmodel_converter is None:
        savedmodel_converter = onnx_to_savedmodel
    if quantizer is None:
        quantizer = quantize_to_int8_tflite

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="us004_quant_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = onnx_exporter(pt_path, work_dir, imgsz=imgsz)
    saved_model_dir = savedmodel_converter(onnx_path, work_dir / "saved_model")
    rep_ds = calibration_loader(
        calib_dir=calib_dir, imgsz=imgsz, max_frames=max_calib_frames
    )
    return quantizer(saved_model_dir, rep_ds, out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="export_quantize")
    ap.add_argument("--model", type=Path, default=DEFAULT_PT)
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
        pt_path=args.model,
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
                "pt_path": str(args.model),
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
