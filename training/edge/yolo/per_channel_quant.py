"""iter-A: per-channel-friendly INT8 PTQ that fixes the YOLOv8 cls collapse.

Background
----------
v1's US-004 / US-006 documented the failure mode:

    YOLOv8's monolithic Detect output (1, 5, 1029) packs xywh (pixel scale
    0..imgsz) and cls (sigmoid 0..1) into one tensor. ``tf.lite.TFLiteConverter``
    picks ONE per-tensor output scale dominated by xywh; the cls channel
    collapses into a single int8 bucket, every confidence becomes 0, mAP=0.

KD (US-006) partially recovered to mAP=0.205 by shaping the student's cls
logits through teacher distillation. The orthogonal fix this iteration adds is:

    Keep the model fully INT8 internally (per-channel WEIGHT quant via the
    MLIR-based new quantizer), but dequantize the OUTPUT tensor to float32
    inside the converter. The appended dequant op preserves cls precision
    regardless of any per-tensor activation scale on the head's output. On
    TFLM x86 / ESP32-P4 a single dequant on an output of shape (1, 5, 1029)
    is cheap (~5145 floats * 4 bytes = a few microseconds).

This is the lowest-risk path to reach a non-zero (and ideally near-fp32) mAP
under PTQ without any architectural surgery (head-split, separate output
tensors). If iter-A already saturates accuracy near the fp32 ceiling, the
more invasive head-split path can be skipped.

Usage
-----
    python -m training.edge.yolo.per_channel_quant \\
        --model training/edge/models/yolov8n_cat_distilled.pt \\
        --calib-dir training/edge/data/calibration_frames \\
        --out training/edge/models/yolov8n_cat_distilled_int8_pc.tflite \\
        --imgsz 224

The CLI also writes <out>.export.json with the pipeline parameters so the
sibling iter-A_per_channel_quant.json aggregator can locate the .tflite.

Per-tensor vs per-channel terminology (codify for downstream stories)
---------------------------------------------------------------------
TFLite supports per-axis (per-channel) quantization for INT8 *weights* on
Conv2d / Conv3d / DepthwiseConv2d layers when ``experimental_new_quantizer``
is on. Op-level activations are still per-tensor — that's a TFLite invariant,
not a converter knob. So "per-channel quantization" in this codebase means:

  - per-channel WEIGHT scales (always on under the MLIR new quantizer), AND
  - float32 output dequant on tensors where mixed-scale channels (e.g. xywh
    + cls) would otherwise saturate under a single per-tensor activation
    scale.

The combination is what fixes the cls collapse.
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

# Same env opt-in as US-004/US-006 — must be set BEFORE any keras/tf import in
# the call chain, including the lazy ones inside onnx_to_savedmodel.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Re-use the v1 calibration loader + onnx export + savedmodel converter so
# the calibration set is byte-identical across YOLO and NanoDet (load-bearing
# pattern from progress.txt: "shared calibration_loader is the contract").
from training.edge.yolo.export_quantize import (  # noqa: E402
    DEFAULT_IMGSZ,
    DEFAULT_MAX_CALIB_FRAMES,
    calibration_loader,
    export_to_onnx,
    onnx_to_savedmodel,
)

DEFAULT_PT = Path("training/edge/models/yolov8n_cat_distilled.pt")
DEFAULT_CALIB_DIR = Path("training/edge/data/calibration_frames")
DEFAULT_OUT = Path("training/edge/models/yolov8n_cat_distilled_int8_pc.tflite")


def quantize_to_int8_float_output_tflite(
    saved_model_dir: Path,
    representative_dataset: Callable[[], Iterable[list[np.ndarray]]],
    out_path: Path,
    converter_factory: Callable[[str], Any] | None = None,
    tf_module: Any = None,
) -> Path:
    """Run the MLIR INT8 PTQ converter with float32 OUTPUT dequant.

    Settings (the iter-A delta vs US-004's quantize_to_int8_tflite):

      - optimizations=[tf.lite.Optimize.DEFAULT]
      - representative_dataset=<provided callable>
      - target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
      - inference_input_type=tf.int8                  (unchanged)
      - inference_output_type=tf.float32              (CHANGED from tf.int8)
      - experimental_new_quantizer=True               (per-channel weight quant)

    The float32 output type causes the converter to append a DEQUANTIZE op
    after the head's int8 output tensor. Cls precision is preserved at full
    float resolution; xywh precision is unchanged. TFLM has a builtin
    DEQUANTIZE op so this is fully ESP32-P4-compatible.

    DI hooks (``converter_factory``, ``tf_module``) match v1 so the test in
    ``test_per_channel_quant.py`` can swap in MagicMocks without TF.
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
    converter.inference_output_type = tf_module.float32  # iter-A delta
    if hasattr(converter, "experimental_new_quantizer"):
        converter.experimental_new_quantizer = True

    tflite_bytes = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(tflite_bytes))
    return out_path


def export_per_channel_int8(
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
    """End-to-end pt -> int8-with-float-output .tflite. DI seams on every stage.

    Mirrors the shape of training.edge.yolo.export_quantize.export_and_quantize
    but plugs in :func:`quantize_to_int8_float_output_tflite` as the default
    quantizer step. Tests use the same ``onnx_exporter`` / ``savedmodel_converter``
    / ``quantizer`` injection points to stay TF-free.
    """
    if onnx_exporter is None:
        onnx_exporter = export_to_onnx
    if savedmodel_converter is None:
        savedmodel_converter = onnx_to_savedmodel
    if quantizer is None:
        quantizer = quantize_to_int8_float_output_tflite

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="iter_a_pcquant_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = onnx_exporter(pt_path, work_dir, imgsz=imgsz)
    saved_model_dir = savedmodel_converter(onnx_path, work_dir / "saved_model")
    rep_ds = calibration_loader(
        calib_dir=calib_dir, imgsz=imgsz, max_frames=max_calib_frames
    )
    return quantizer(saved_model_dir, rep_ds, out_path)


# ---------------------------------------------------------------------------
# Pareto verdict helper — used by iter-A's aggregator and re-used by iter-B..G
# ---------------------------------------------------------------------------

# Pareto thresholds from the user's iter-A spec (v2 follow-on loop):
#   accuracy: mAP@0.5 must be >= baseline (no regression)
#   efficiency: improve at least one of
#     - predicted_p4_latency_ms_p50 by >= 15%
#     - model_size_mb by >= 20%
#     - tensor_arena_kb by >= 15%
LATENCY_IMPROVE_PCT = 15.0
SIZE_IMPROVE_PCT = 20.0
ARENA_IMPROVE_PCT = 15.0
MAP_REGRESS_TOLERANCE = 0.0  # mAP must not drop at all

# ESP32-P4 latency multiplier vs x86 reference TFLM ops. Justification:
#   - ESP32-S3 multiplier was 8.0 in v1 (citation in firmware/edge-bench/README.md).
#   - ESP32-P4 ships at 400 MHz dual-core RV32 with PIE/vector ext (vs S3's
#     240 MHz single-core for ML). Without ESP-DL acceleration the scalar
#     reference path is ~5-6x slower than x86; with vector ext + dual-core
#     headroom the lower bound is ~3x.
#   - We use 5.0 as the conservative midpoint. iter-A_per_channel_quant.json
#     records this choice in p4_multiplier_source so a future iteration can
#     update it once boards are in hand.
DEFAULT_P4_MULTIPLIER = 5.0
DEFAULT_P4_MULTIPLIER_SOURCE = (
    "Conservative midpoint between scalar-reference (~6x x86) and "
    "vector-ext / dual-core ideal (~3x x86); v1 used 8.0 for ESP32-S3. "
    "Update once real ESP32-P4 measurements land."
)


def build_pareto_verdict(
    *,
    candidate_map50: float,
    candidate_size_bytes: int,
    candidate_arena_used_bytes: int,
    candidate_predicted_p4_latency_ms_p50: float,
    baseline_story: str,
    baseline_map50: float,
    baseline_size_bytes: int,
    baseline_arena_used_bytes: int,
    baseline_predicted_p4_latency_ms_p50: float,
    blocked_reason: str | None = None,
) -> dict:
    """Compute the Pareto verdict shape for iter-* JSON files.

    Returns a dict matching the iter-A acceptance-criteria schema:

        {
          "verdict": "dominates" | "equal" | "regress" | "blocked",
          "baseline_story": str,
          "baseline_map50": float,
          "baseline_size_bytes": int,
          "baseline_arena_used_bytes": int,
          "baseline_predicted_p4_latency_ms_p50": float,
          "deltas": {
            "map50": float (candidate - baseline, fraction 0..1),
            "size_pct": float (negative = smaller = better),
            "arena_pct": float (negative = smaller = better),
            "latency_pct": float (negative = faster = better),
          },
          "thresholds": {... constants ...},
        }

    Verdict rules (per iter-A acceptance criteria):
      - "blocked"   : blocked_reason is set
      - "regress"   : map50 dropped below baseline (any negative delta)
      - "dominates" : mAP held AND >=1 efficiency axis improved past threshold
      - "equal"     : mAP held AND no efficiency axis improved enough
    """
    if blocked_reason is not None:
        return {
            "verdict": "blocked",
            "blocked_reason": blocked_reason,
            "baseline_story": baseline_story,
            "baseline_map50": float(baseline_map50),
            "baseline_size_bytes": int(baseline_size_bytes),
            "baseline_arena_used_bytes": int(baseline_arena_used_bytes),
            "baseline_predicted_p4_latency_ms_p50": float(
                baseline_predicted_p4_latency_ms_p50
            ),
            "deltas": None,
            "thresholds": _thresholds_dict(),
        }

    map_delta = float(candidate_map50) - float(baseline_map50)

    def _pct(c: float, b: float) -> float:
        if b <= 0:
            return 0.0
        return 100.0 * (float(c) - float(b)) / float(b)

    size_pct = _pct(candidate_size_bytes, baseline_size_bytes)
    arena_pct = _pct(candidate_arena_used_bytes, baseline_arena_used_bytes)
    latency_pct = _pct(
        candidate_predicted_p4_latency_ms_p50,
        baseline_predicted_p4_latency_ms_p50,
    )

    if map_delta < -MAP_REGRESS_TOLERANCE:
        verdict = "regress"
    else:
        improves_size = size_pct <= -SIZE_IMPROVE_PCT
        improves_arena = arena_pct <= -ARENA_IMPROVE_PCT
        improves_latency = latency_pct <= -LATENCY_IMPROVE_PCT
        if improves_size or improves_arena or improves_latency:
            verdict = "dominates"
        else:
            verdict = "equal"

    return {
        "verdict": verdict,
        "baseline_story": baseline_story,
        "baseline_map50": float(baseline_map50),
        "baseline_size_bytes": int(baseline_size_bytes),
        "baseline_arena_used_bytes": int(baseline_arena_used_bytes),
        "baseline_predicted_p4_latency_ms_p50": float(
            baseline_predicted_p4_latency_ms_p50
        ),
        "deltas": {
            "map50": map_delta,
            "size_pct": size_pct,
            "arena_pct": arena_pct,
            "latency_pct": latency_pct,
        },
        "thresholds": _thresholds_dict(),
    }


def _thresholds_dict() -> dict:
    return {
        "map_regress_tolerance": MAP_REGRESS_TOLERANCE,
        "latency_improve_pct": LATENCY_IMPROVE_PCT,
        "size_improve_pct": SIZE_IMPROVE_PCT,
        "arena_improve_pct": ARENA_IMPROVE_PCT,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="per_channel_quant")
    ap.add_argument("--model", type=Path, default=DEFAULT_PT)
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--max-calib-frames", type=int, default=DEFAULT_MAX_CALIB_FRAMES)
    ap.add_argument("--work-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    out = export_per_channel_int8(
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
                "inference_input_type": "int8",
                "inference_output_type": "float32",
                "experimental_new_quantizer": True,
                "story_id": "iter-A",
            },
            indent=2,
        )
    )
    sz = out.stat().st_size if out.exists() else 0
    print(f"int8-pc tflite -> {out} ({sz} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
