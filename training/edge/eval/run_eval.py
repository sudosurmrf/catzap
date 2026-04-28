"""Eval harness CLI — uniform mAP/size/params/FLOPs/latency for any model format.

Usage:
    python -m training.edge.eval.run_eval \\
        --model yolov8s.pt \\
        --format pytorch \\
        --story-id US-001-yolov8s-baseline \\
        --val-dir training/edge/data/val \\
        --imgsz 1280

The output is a single training/edge/results/<story-id>.json file matching
the EvalResult dataclass. A short markdown summary is also printed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .latency import measure_latency
from .metrics import compute_map50
from .types import EvalResult
from .val_loader import load_val_set


def _load_adapter(model_path: str, fmt: str, imgsz: int):
    if fmt == "pytorch":
        from .adapters.pytorch_adapter import load

        return load(model_path, imgsz=imgsz)
    if fmt == "onnx":
        from .adapters.onnx_adapter import load

        return load(model_path, imgsz=imgsz)
    if fmt == "tflite_fp32":
        from .adapters.tflite_adapter import load

        return load(model_path, imgsz=imgsz, is_int8=False)
    if fmt == "tflite_int8":
        from .adapters.tflite_adapter import load

        return load(model_path, imgsz=imgsz, is_int8=True)
    raise ValueError(f"unsupported format: {fmt}")


def evaluate(
    model_path: str,
    fmt: str,
    story_id: str,
    val_dir: Path,
    imgsz: int = 224,
    notes: str = "",
) -> EvalResult:
    adapter = _load_adapter(model_path, fmt, imgsz)
    images, gts, _paths = load_val_set(val_dir)

    predictions: list[list[dict]] = []
    for frame in images:
        predictions.append(adapter.predict(frame))

    map50 = compute_map50(predictions, gts) if images else 0.0

    if images:
        sample = images[0]
    else:
        sample = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    p50, p95 = measure_latency(adapter.predict, sample)

    return EvalResult(
        story_id=story_id,
        model_path=str(model_path),
        model_format=fmt,
        map50=float(map50),
        size_bytes=int(os.path.getsize(model_path)) if Path(model_path).exists() else 0,
        params=int(adapter.num_params()),
        flops=int(adapter.num_flops()),
        input_hw=(imgsz, imgsz),
        latency_ms_p50=float(p50),
        latency_ms_p95=float(p95),
        val_images=len(images),
        notes=notes,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_eval")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--format",
        required=True,
        choices=["pytorch", "onnx", "tflite_fp32", "tflite_int8"],
    )
    ap.add_argument("--story-id", required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--notes", default="")
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("training/edge/results"),
    )
    args = ap.parse_args(argv)

    result = evaluate(
        args.model, args.format, args.story_id, args.val_dir, args.imgsz, args.notes
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"{args.story_id}.json"
    result.write(out_path)
    print(f"wrote {out_path}")
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
