"""iter-B: input-resolution sweep on the iter-A frontier checkpoint.

For each ``imgsz`` in ``--imgsz-list``, this script:

  1. Re-exports ``training/edge/models/yolov8n_cat_distilled.pt`` (the iter-A
     frontier checkpoint) to INT8 TFLite at the target ``imgsz`` via
     :func:`training.edge.yolo.per_channel_quant.export_per_channel_int8` —
     reusing iter-A's per-channel weights + float32-output-dequant fix to
     keep cls precision.
  2. Runs the existing eval harness
     (:func:`training.edge.eval.run_eval.evaluate`) on the freshly-quantized
     ``.tflite`` to measure mAP@0.5, host-CPU INT8 latency, and ``size_bytes``
     at the target imgsz.
  3. Runs the existing TFLM x86 host bench (``firmware/edge-bench/``) to
     measure ``arena_used_bytes`` and per-op timings.
  4. Folds eval + TFLM bench into one
     ``training/edge/results/iter-B_imgsz_<N>.json`` rollup matching the
     iter-A_per_channel_quant.json schema (``status / model_path /
     p4_multiplier / eval / tflm / pareto / candidate_metrics /
     baseline_metrics``).
  5. Compares against iter-A's INT8 candidate metrics (NOT the v1
     US-006-int8 baseline — iter-A is the new accuracy floor for v2).

Failure isolation
-----------------
If any single imgsz fails to quantize / eval / bench (for example onnx2tf's
NHWC layout transformer occasionally trips on imgsz values where the
backbone strides don't divide cleanly), the script writes
``status="blocked"`` for THAT imgsz only with a concrete ``blocked_reason``
and continues with the rest. This is the iter-A "always commit the JSON
even when blocked" pattern, applied per imgsz.

Usage::

    python -m training.edge.yolo.imgsz_sweep \\
        --model training/edge/models/yolov8n_cat_distilled.pt \\
        --imgsz-list 192,224,256,288,320 \\
        --calib-dir training/edge/data/calibration_frames \\
        --out-dir training/edge/models/sweep_imgsz/

The default arguments wire to the iter-A frontier inputs so the bare
``python -m training.edge.yolo.imgsz_sweep`` invocation reproduces the
sweep iter-B is responsible for.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Make firmware/edge-bench/ importable for run_bench. The directory has a
# hyphen in its name so it's not a python package; we mimic the pattern
# already used by training/edge/tests/test_tflm_bench_parser.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCH_DIR = _REPO_ROOT / "firmware" / "edge-bench"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from training.edge.yolo.per_channel_quant import (  # noqa: E402
    DEFAULT_P4_MULTIPLIER,
    DEFAULT_P4_MULTIPLIER_SOURCE,
    build_pareto_verdict,
    export_per_channel_int8,
)

DEFAULT_PT = Path("training/edge/models/yolov8n_cat_distilled.pt")
DEFAULT_CALIB_DIR = Path("training/edge/data/calibration_frames")
DEFAULT_OUT_DIR = Path("training/edge/models/sweep_imgsz")
DEFAULT_VAL_DIR = Path("training/edge/data/labeled/val")
DEFAULT_RESULTS_DIR = Path("training/edge/results")
DEFAULT_IMGSZ_LIST: tuple[int, ...] = (192, 224, 256, 288, 320)
DEFAULT_RUNS = 50
BASELINE_AGGREGATE_PATH = DEFAULT_RESULTS_DIR / "iter-A_per_channel_quant.json"
BASELINE_STORY = "iter-A (per-channel-friendly INT8 PTQ; v2 accuracy floor)"

# Quantize / eval / bench Callables — all DI-overridable so tests can swap
# in MagicMocks without TF, ultralytics, or the edgebench binary installed.
QuantizeFn = Callable[..., Path]
EvalFn = Callable[..., dict]
BenchFn = Callable[..., dict]
BaselineLoaderFn = Callable[[], dict]


def _x86_us_to_p4_ms(us: float, multiplier: float) -> float:
    return float(us) * float(multiplier) / 1000.0


def _default_quantize_fn(
    *, pt_path: Path, calib_dir: Path, out_path: Path, imgsz: int
) -> Path:
    return export_per_channel_int8(
        pt_path=pt_path,
        calib_dir=calib_dir,
        out_path=out_path,
        imgsz=imgsz,
    )


def _default_eval_fn(
    *, model_path: Path, story_id: str, val_dir: Path, imgsz: int, results_dir: Path
) -> dict:
    """Run the standard eval harness and return the on-disk JSON dict."""
    # Lazy import keeps test files free of cv2/torch/ultralytics requirements.
    from training.edge.eval.run_eval import evaluate

    result = evaluate(
        str(model_path), "tflite_int8", story_id, val_dir, imgsz=imgsz
    )
    out_path = results_dir / f"{story_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write(out_path)
    return json.loads(out_path.read_text())


def _default_bench_fn(
    *, model_path: Path, story_id: str, results_dir: Path, runs: int
) -> dict:
    """Run firmware/edge-bench/run_bench.py programmatically; return TFLM dict."""
    import run_bench  # type: ignore[import-not-found]  # from _BENCH_DIR

    binary = run_bench._resolve_binary(None)
    stdout = run_bench.run(
        binary=binary, model=Path(model_path), runs=runs, arena_kb=8192
    )
    parsed = run_bench.parse_edgebench_stdout(stdout)
    result = run_bench.report_to_result(
        parsed,
        story_id=story_id,
        binary_path=str(binary),
        tflm_commit=run_bench._read_tflm_commit(),
    )
    out_path = results_dir / f"{story_id}-tflm.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def _default_baseline_loader() -> dict:
    if not BASELINE_AGGREGATE_PATH.exists():
        raise FileNotFoundError(
            f"baseline aggregate not found at {BASELINE_AGGREGATE_PATH}; "
            "run `python training/edge/results/aggregate_iter_a.py` first"
        )
    return json.loads(BASELINE_AGGREGATE_PATH.read_text())


def _baseline_metrics_from_iter_a(baseline_aggregate: dict) -> dict:
    """Extract the four-axis baseline numbers iter-B compares against.

    iter-A's aggregate JSON exposes ``candidate_metrics`` (its own measured
    numbers) — those become iter-B's ``baseline_metrics``.
    """
    cm = baseline_aggregate.get("candidate_metrics") or {}
    return {
        "map50": float(cm.get("map50", 0.0)),
        "size_bytes": int(cm.get("size_bytes", 0)),
        "arena_used_bytes": int(cm.get("arena_used_bytes", 0)),
        "predicted_p4_latency_ms_p50": float(
            cm.get("predicted_p4_latency_ms_p50", 0.0)
        ),
    }


def _build_aggregate_doc(
    *,
    story_id: str,
    imgsz: int,
    model_path: Path,
    eval_dict: dict | None,
    tflm_dict: dict | None,
    multiplier: float,
    multiplier_source: str,
    baseline_metrics: dict,
    blocked_reason: str | None,
) -> dict:
    """Assemble the iter-B_imgsz_<N>.json rollup; mirrors aggregate_iter_a.py shape."""
    candidate_predicted_p4_ms = _x86_us_to_p4_ms(
        (tflm_dict or {}).get("raw_x86_us_p50", 0.0), multiplier
    )
    candidate_map = float((eval_dict or {}).get("map50", 0.0))
    candidate_size = int(
        (eval_dict or {}).get("size_bytes")
        or (tflm_dict or {}).get("model_size_bytes", 0)
    )
    candidate_arena = int((tflm_dict or {}).get("arena_used_bytes", 0))

    verdict = build_pareto_verdict(
        candidate_map50=candidate_map,
        candidate_size_bytes=candidate_size,
        candidate_arena_used_bytes=candidate_arena,
        candidate_predicted_p4_latency_ms_p50=candidate_predicted_p4_ms,
        baseline_story=BASELINE_STORY,
        baseline_map50=baseline_metrics["map50"],
        baseline_size_bytes=baseline_metrics["size_bytes"],
        baseline_arena_used_bytes=baseline_metrics["arena_used_bytes"],
        baseline_predicted_p4_latency_ms_p50=baseline_metrics[
            "predicted_p4_latency_ms_p50"
        ],
        blocked_reason=blocked_reason,
    )

    candidate_fps = (
        1000.0 / candidate_predicted_p4_ms if candidate_predicted_p4_ms > 0 else 0.0
    )

    return {
        "story_id": story_id,
        "imgsz": int(imgsz),
        "title": f"iter-B input-resolution sweep at imgsz={imgsz}",
        "status": "blocked" if blocked_reason else "passed",
        "blocked_reason": blocked_reason,
        "model_path": str(model_path),
        "p4_multiplier": multiplier,
        "p4_multiplier_source": multiplier_source,
        "eval": eval_dict,
        "tflm": tflm_dict,
        "pareto": verdict,
        "candidate_metrics": {
            "map50": candidate_map,
            "size_bytes": candidate_size,
            "arena_used_bytes": candidate_arena,
            "predicted_p4_latency_ms_p50": candidate_predicted_p4_ms,
            "predicted_p4_fps": candidate_fps,
        },
        "baseline_metrics": dict(baseline_metrics),
    }


def _write_aggregate(doc: dict, results_dir: Path, imgsz: int) -> Path:
    out_path = results_dir / f"iter-B_imgsz_{imgsz}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    return out_path


def _tflite_path_for(out_dir: Path, imgsz: int) -> Path:
    return out_dir / f"yolov8n_cat_distilled_int8_pc_{imgsz}.tflite"


def sweep(
    *,
    pt_path: Path = DEFAULT_PT,
    calib_dir: Path = DEFAULT_CALIB_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    val_dir: Path = DEFAULT_VAL_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    imgsz_list: list[int] | tuple[int, ...] = DEFAULT_IMGSZ_LIST,
    runs: int = DEFAULT_RUNS,
    multiplier: float = DEFAULT_P4_MULTIPLIER,
    multiplier_source: str = DEFAULT_P4_MULTIPLIER_SOURCE,
    quantize_fn: QuantizeFn | None = None,
    eval_fn: EvalFn | None = None,
    bench_fn: BenchFn | None = None,
    baseline_loader: BaselineLoaderFn | None = None,
) -> list[dict]:
    """Run the full imgsz sweep. Returns one rollup dict per imgsz.

    Each iteration writes (a) the eval JSON via ``eval_fn``, (b) the TFLM
    bench JSON via ``bench_fn``, and (c) the iter-B rollup at
    ``results_dir / iter-B_imgsz_<N>.json``. Quantize / eval / bench failures
    are recorded as ``status="blocked"`` for that single imgsz; the loop
    keeps going.
    """
    quantize_fn = quantize_fn or _default_quantize_fn
    eval_fn = eval_fn or _default_eval_fn
    bench_fn = bench_fn or _default_bench_fn
    baseline_loader = baseline_loader or _default_baseline_loader

    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_aggregate = baseline_loader()
    baseline_metrics = _baseline_metrics_from_iter_a(baseline_aggregate)

    aggregates: list[dict] = []
    for imgsz in imgsz_list:
        story_id = f"iter-B_imgsz_{imgsz}"
        eval_story = f"{story_id}_eval"
        bench_story = f"{story_id}_bench"
        tflite_path = _tflite_path_for(out_dir, imgsz)

        eval_dict: dict | None = None
        tflm_dict: dict | None = None
        blocked_reason: str | None = None

        # 1. Quantize
        try:
            quantize_fn(
                pt_path=pt_path,
                calib_dir=calib_dir,
                out_path=tflite_path,
                imgsz=imgsz,
            )
        except Exception as exc:  # noqa: BLE001 - defensive sweep boundary
            blocked_reason = (
                f"quantize step failed at imgsz={imgsz}: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

        # 2. Eval
        if blocked_reason is None:
            try:
                eval_dict = eval_fn(
                    model_path=tflite_path,
                    story_id=eval_story,
                    val_dir=val_dir,
                    imgsz=imgsz,
                    results_dir=results_dir,
                )
            except Exception as exc:  # noqa: BLE001
                blocked_reason = (
                    f"eval step failed at imgsz={imgsz}: "
                    f"{type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

        # 3. Bench
        if blocked_reason is None:
            try:
                tflm_dict = bench_fn(
                    model_path=tflite_path,
                    story_id=bench_story,
                    results_dir=results_dir,
                    runs=runs,
                )
            except Exception as exc:  # noqa: BLE001
                blocked_reason = (
                    f"bench step failed at imgsz={imgsz}: "
                    f"{type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

        agg = _build_aggregate_doc(
            story_id=story_id,
            imgsz=imgsz,
            model_path=tflite_path,
            eval_dict=eval_dict,
            tflm_dict=tflm_dict,
            multiplier=multiplier,
            multiplier_source=multiplier_source,
            baseline_metrics=baseline_metrics,
            blocked_reason=blocked_reason,
        )
        out_path = _write_aggregate(agg, results_dir, imgsz)
        aggregates.append(agg)
        verdict = agg["pareto"]["verdict"]
        print(f"imgsz={imgsz} -> {out_path.name}  status={agg['status']} verdict={verdict}")

    return aggregates


def _parse_imgsz_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="imgsz_sweep")
    ap.add_argument("--model", type=Path, default=DEFAULT_PT)
    ap.add_argument(
        "--imgsz-list",
        type=_parse_imgsz_list,
        default=list(DEFAULT_IMGSZ_LIST),
        help="comma-separated imgsz values (default: 192,224,256,288,320)",
    )
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    args = ap.parse_args(argv)

    sweep(
        pt_path=args.model,
        calib_dir=args.calib_dir,
        out_dir=args.out_dir,
        val_dir=args.val_dir,
        results_dir=args.results_dir,
        imgsz_list=args.imgsz_list,
        runs=args.runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
