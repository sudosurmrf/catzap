"""iter-C: structured channel pruning of the distilled student.

Pipeline:
    pt -> L2-norm structured channel prune (~25%, head excluded)
       -> 1-epoch finetune on training/edge/data/labeled/
       -> per-channel-friendly INT8 quantize (re-uses iter-A pipeline)
       -> eval + edge-bench
       -> Pareto verdict vs iter-A frontier

Notes
-----
- ``torch.nn.utils.prune.LnStructured(amount=0.25, n=2, dim=0)`` zeros out
  the lowest-L2 25% of OUTPUT channels per ``Conv2d`` layer. This is
  "soft" pruning — the masked weights stay in the parameter tensor (now
  set to zero) so the architecture is unchanged. The Detect head expects
  specific channel counts so a hard slice would break it; the soft prune
  + flatbuffer compression in the int8 .tflite still shrinks the artifact
  because zero-weight channels compress well.
- Detect head (final ``Detect`` module in ultralytics' YOLOv8) is excluded
  via class-name match — both ``Detect`` and ``DFL`` are skipped.
- Wall-clock target: ~2 min on RTX 4090 for the 1-epoch finetune. Whole
  pipeline (prune + finetune + quant + eval + bench) typically <5 min.
- Failure isolation: if any step (prune / finetune / quant / eval / bench)
  raises, the script records ``status="blocked"`` with a concrete reason
  in iter-C_pruned.json and stops — but the JSON IS still written.

Usage::

    python -m training.edge.yolo.prune_channels \\
        --model training/edge/models/yolov8n_cat_distilled.pt \\
        --sparsity 0.25 \\
        --finetune-epochs 1 \\
        --data training/edge/data/labeled/data.yaml \\
        --out training/edge/models/yolov8n_cat_pruned.pt

CLI defaults wire to the iter-C acceptance-criteria values; the bare
invocation reproduces the iter-C result.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

# Make firmware/edge-bench/ importable for run_bench. Same pattern as
# imgsz_sweep.py (the dir has a hyphen so it's not a python package).
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
DEFAULT_OUT_PT = Path("training/edge/models/yolov8n_cat_pruned.pt")
DEFAULT_OUT_TFLITE = Path("training/edge/models/yolov8n_cat_pruned_int8_pc.tflite")
DEFAULT_DATA_YAML = Path("training/edge/data/labeled/data.yaml")
DEFAULT_CALIB_DIR = Path("training/edge/data/calibration_frames")
DEFAULT_VAL_DIR = Path("training/edge/data/labeled/val")
DEFAULT_RESULTS_DIR = Path("training/edge/results")
DEFAULT_RUNS_DIR = Path("training/edge/yolo/runs")
DEFAULT_SPARSITY = 0.25
DEFAULT_FINETUNE_EPOCHS = 1
DEFAULT_IMGSZ = 224
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-4
DEFAULT_BENCH_RUNS = 50
DEFAULT_DEVICE = ""  # empty string lets ultralytics auto-detect CUDA/MPS/CPU
BASELINE_AGGREGATE_PATH = DEFAULT_RESULTS_DIR / "iter-A_per_channel_quant.json"
BASELINE_STORY = "iter-A (per-channel-friendly INT8 PTQ; v2 accuracy floor)"

# Class names the L2 channel pruner skips when ``exclude_head=True``. These
# match ultralytics' YOLOv8 head module (``Detect``) and the distribution
# focal loss helper (``DFL``) it contains. Any Conv2d nested under one of
# these is left untouched.
HEAD_MODULE_NAMES: frozenset[str] = frozenset({"Detect", "DFL"})


# ---------------------------------------------------------------------------
# Pruning core
# ---------------------------------------------------------------------------


def _is_head_path(name: str, head_indices: Iterable[int]) -> bool:
    """Return True if ``name`` (a dotted module path) lives under the head.

    ultralytics' YOLOv8 stores the backbone + neck + head in a single
    ``nn.Sequential`` at ``model.model``, indexed numerically. The Detect
    head is conventionally the last index. We walk this Sequential via
    ``inner.named_modules()`` where ``inner = root.model``, so names look
    like ``"22.cv2.0.0.conv"`` (no ``model.`` prefix). For tests/callers
    that DO pass full ``"model.22..."`` paths we also accept that shape.
    """
    if not name:
        return False
    parts = name.split(".")
    # Form 1: leading numeric index — names from inner.named_modules().
    if parts[0].isdigit() and int(parts[0]) in head_indices:
        return True
    # Form 2: "model.<int>..." prefix — names from root.named_modules().
    for i in range(len(parts) - 1):
        if parts[i] == "model" and parts[i + 1].isdigit():
            if int(parts[i + 1]) in head_indices:
                return True
    return False


def _find_head_indices(root: Any) -> set[int]:
    """Inspect ``root.model`` (the inner nn.Sequential) for Detect/DFL blocks.

    Returns the set of integer indices in the Sequential that are head
    modules. For YOLOv8 this is typically ``{22}`` (single Detect block).
    On a non-ultralytics module (e.g. test fakes) the function returns the
    empty set — pruning then runs on every Conv2d and the test asserts the
    head-exclusion path via the ``exclude_head=False`` branch instead.
    """
    inner = getattr(root, "model", None)
    if inner is None:
        return set()
    indices: set[int] = set()
    try:
        for idx, child in enumerate(inner):
            cname = type(child).__name__
            if cname in HEAD_MODULE_NAMES:
                indices.add(idx)
    except TypeError:
        return set()
    return indices


def _count_nonzero(weight: Any) -> int:
    """Count non-zero scalar entries in a weight tensor without importing torch.

    Falls back to ``int(weight.ne(0).sum().item())`` when torch is available
    (the standard idiom). The signature accepts ``Any`` so tests can pass
    a ``MagicMock`` whose ``ne(0).sum().item()`` returns a fixed number.
    """
    return int(weight.ne(0).sum().item())


def _count_total(weight: Any) -> int:
    return int(weight.numel())


def prune_l2_channels(
    model: Any,
    *,
    sparsity: float = DEFAULT_SPARSITY,
    exclude_head: bool = True,
    prune_module: Any = None,
    nn_module: Any = None,
) -> dict:
    """Apply L2-norm structured channel prune to all backbone/neck Conv2d.

    Parameters
    ----------
    model
        An nn.Module (or ultralytics ``YOLO`` wrapper). The actual
        nn.Module to walk is ``getattr(model, 'model', model)`` — matches
        ultralytics' convention without breaking on plain modules.
    sparsity
        Fraction (0..1) of OUTPUT channels to zero out per Conv2d layer.
        Default 0.25 per the iter-C acceptance criteria.
    exclude_head
        If True, skip Conv2d under the Detect head (and any DFL sub-block).
        Default True — Detect expects specific channel counts so structured
        pruning on the head would break the forward.
    prune_module, nn_module
        DI seams. Default to ``torch.nn.utils.prune`` and ``torch.nn``.
        Tests pass mocks so this function is import-safe without torch.

    Returns
    -------
    A summary dict::

        {
          "sparsity": 0.25,
          "exclude_head": True,
          "head_indices": [22],
          "pruned_modules": ["model.0.conv", "model.1.conv", ...],
          "skipped_head_modules": ["model.22.cv2.0.0.conv", ...],
          "params_before": 3_011_043,
          "params_after_nonzero": 2_283_287,
          "params_zeroed": 727_756,
          "actual_sparsity": 0.2418,
        }

    The "params_after_nonzero" count reflects the post-prune live (non-zero)
    weight count. The architecture (parameter shapes) is unchanged — this
    is a soft prune; ``prune.remove`` is called so the masks bake in but
    the zeroed-out weights remain in the tensor and quantize down well in
    flatbuffer compression. A future iter could add a hard slice + Detect
    head re-init, but that's out of scope for iter-C.
    """
    if prune_module is None:
        from torch.nn.utils import prune as _prune  # lazy

        prune_module = _prune
    if nn_module is None:
        from torch import nn as _nn  # lazy

        nn_module = _nn

    inner = getattr(model, "model", model)
    head_indices = _find_head_indices(model) if exclude_head else set()

    pruned: list[str] = []
    skipped_head: list[str] = []
    params_before = 0
    params_after_nonzero = 0

    for name, module in inner.named_modules():
        if not isinstance(module, nn_module.Conv2d):
            continue
        if exclude_head and _is_head_path(name, head_indices):
            skipped_head.append(name)
            continue

        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        params_before += _count_total(weight)

        try:
            prune_module.ln_structured(
                module, name="weight", amount=float(sparsity), n=2, dim=0
            )
            # Bake the mask into the weight tensor so the saved .pt has the
            # zeros directly (ultralytics' .save() / state_dict round-trip
            # otherwise strips ``.weight_orig`` / ``.weight_mask`` buffers).
            prune_module.remove(module, "weight")
            pruned.append(name)
        except Exception as exc:  # noqa: BLE001
            # Not fatal — record and keep going so a single problematic
            # layer doesn't abort the prune. Caller surfaces the count.
            skipped_head.append(f"{name} (error: {type(exc).__name__})")
            continue

        # Re-read weight (post-remove the prune buffers are gone).
        weight = getattr(module, "weight", None)
        if weight is not None:
            params_after_nonzero += _count_nonzero(weight)

    actual_sparsity = (
        1.0 - (params_after_nonzero / params_before) if params_before else 0.0
    )
    return {
        "sparsity_target": float(sparsity),
        "exclude_head": bool(exclude_head),
        "head_indices": sorted(head_indices),
        "pruned_modules": pruned,
        "skipped_head_modules": skipped_head,
        "params_before": int(params_before),
        "params_after_nonzero": int(params_after_nonzero),
        "params_zeroed": int(params_before - params_after_nonzero),
        "actual_sparsity": float(actual_sparsity),
    }


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


PruneFn = Callable[..., dict]
TrainerFn = Callable[..., Path]
QuantizeFn = Callable[..., Path]
EvalFn = Callable[..., dict]
BenchFn = Callable[..., dict]
BaselineLoaderFn = Callable[[], dict]


def _x86_us_to_p4_ms(us: float, multiplier: float) -> float:
    return float(us) * float(multiplier) / 1000.0


def _default_yolo_factory(pt_path: Path) -> Any:
    """Lazy ultralytics import — only triggered when the real pipeline runs."""
    from ultralytics import YOLO  # lazy

    return YOLO(str(pt_path))


def _default_trainer_fn(
    *,
    pt_path: Path,
    pruned_pt_path: Path,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    lr: float,
    runs_dir: Path,
    name: str,
    yolo_factory: Callable[[Path], Any],
    pruner_fn: PruneFn,
    sparsity: float,
    exclude_head: bool,
    device: str = DEFAULT_DEVICE,
) -> tuple[Path, dict]:
    """Load pt -> prune -> finetune -> save canonical pt. Returns (path, prune_summary)."""
    model = yolo_factory(pt_path)
    inner = getattr(model, "model", model)
    prune_summary = pruner_fn(
        inner, sparsity=sparsity, exclude_head=exclude_head
    )

    runs_dir.mkdir(parents=True, exist_ok=True)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        single_cls=True,
        device=device,
        project=str(runs_dir),
        name=name,
        exist_ok=True,
        verbose=True,
    )
    best = runs_dir / name / "weights" / "best.pt"
    if not best.exists():
        last = runs_dir / name / "weights" / "last.pt"
        if last.exists():
            best = last
    pruned_pt_path.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        import shutil  # stdlib

        shutil.copy2(best, pruned_pt_path)
    return pruned_pt_path, prune_summary


def _default_quantize_fn(*, pt_path: Path, calib_dir: Path, out_path: Path, imgsz: int) -> Path:
    return export_per_channel_int8(
        pt_path=pt_path, calib_dir=calib_dir, out_path=out_path, imgsz=imgsz
    )


def _default_eval_fn(
    *, model_path: Path, story_id: str, val_dir: Path, imgsz: int, results_dir: Path
) -> dict:
    from training.edge.eval.run_eval import evaluate  # lazy

    result = evaluate(str(model_path), "tflite_int8", story_id, val_dir, imgsz=imgsz)
    out_path = results_dir / f"{story_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write(out_path)
    return json.loads(out_path.read_text())


def _default_bench_fn(
    *, model_path: Path, story_id: str, results_dir: Path, runs: int
) -> dict:
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
    model_path: Path,
    eval_dict: dict | None,
    tflm_dict: dict | None,
    prune_summary: dict | None,
    multiplier: float,
    multiplier_source: str,
    baseline_metrics: dict,
    blocked_reason: str | None,
) -> dict:
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
        "title": (
            "Structured channel pruning of the iter-A frontier student "
            "(~25% L2-norm output channels, head excluded) + 1-epoch finetune "
            "+ per-channel-friendly INT8 quant"
        ),
        "status": "blocked" if blocked_reason else "passed",
        "blocked_reason": blocked_reason,
        "model_path": str(model_path),
        "p4_multiplier": multiplier,
        "p4_multiplier_source": multiplier_source,
        "prune_summary": prune_summary,
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


def _write_aggregate(doc: dict, results_dir: Path) -> Path:
    out_path = results_dir / "iter-C_pruned.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    return out_path


def run_pipeline(
    *,
    pt_path: Path = DEFAULT_PT,
    pruned_pt_path: Path = DEFAULT_OUT_PT,
    out_tflite: Path = DEFAULT_OUT_TFLITE,
    data_yaml: Path = DEFAULT_DATA_YAML,
    calib_dir: Path = DEFAULT_CALIB_DIR,
    val_dir: Path = DEFAULT_VAL_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    sparsity: float = DEFAULT_SPARSITY,
    exclude_head: bool = True,
    finetune_epochs: int = DEFAULT_FINETUNE_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    bench_runs: int = DEFAULT_BENCH_RUNS,
    device: str = DEFAULT_DEVICE,
    multiplier: float = DEFAULT_P4_MULTIPLIER,
    multiplier_source: str = DEFAULT_P4_MULTIPLIER_SOURCE,
    yolo_factory: Callable[[Path], Any] | None = None,
    pruner_fn: PruneFn | None = None,
    trainer_fn: TrainerFn | None = None,
    quantize_fn: QuantizeFn | None = None,
    eval_fn: EvalFn | None = None,
    bench_fn: BenchFn | None = None,
    baseline_loader: BaselineLoaderFn | None = None,
) -> dict:
    """Run prune -> finetune -> quantize -> eval -> bench. Always writes JSON.

    Each stage is DI-overridable so tests can substitute in mocks. The
    happy-path defaults wire to ultralytics + per_channel_quant + eval +
    edgebench. On any stage failure the pipeline records
    ``status="blocked"`` with a concrete reason and returns; later stages
    are skipped but the iter-C_pruned.json file IS still written.
    """
    yolo_factory = yolo_factory or _default_yolo_factory
    pruner_fn = pruner_fn or prune_l2_channels
    trainer_fn = trainer_fn or _default_trainer_fn
    quantize_fn = quantize_fn or _default_quantize_fn
    eval_fn = eval_fn or _default_eval_fn
    bench_fn = bench_fn or _default_bench_fn
    baseline_loader = baseline_loader or _default_baseline_loader

    results_dir.mkdir(parents=True, exist_ok=True)

    # Baseline (iter-A frontier) — must load before everything else so the
    # blocked-path JSON still has ``baseline_metrics`` filled in.
    try:
        baseline_aggregate = baseline_loader()
        baseline_metrics = _baseline_metrics_from_iter_a(baseline_aggregate)
    except Exception as exc:  # noqa: BLE001
        baseline_metrics = {
            "map50": 0.0,
            "size_bytes": 0,
            "arena_used_bytes": 0,
            "predicted_p4_latency_ms_p50": 0.0,
        }
        doc = _build_aggregate_doc(
            story_id="iter-C",
            model_path=pruned_pt_path,
            eval_dict=None,
            tflm_dict=None,
            prune_summary=None,
            multiplier=multiplier,
            multiplier_source=multiplier_source,
            baseline_metrics=baseline_metrics,
            blocked_reason=(
                f"baseline loader failed: {type(exc).__name__}: {exc}"
            ),
        )
        _write_aggregate(doc, results_dir)
        return doc

    eval_dict: dict | None = None
    tflm_dict: dict | None = None
    prune_summary: dict | None = None
    blocked_reason: str | None = None

    # 1. Prune + 2. Finetune (combined in trainer_fn since prune happens on
    # the live model handed to ultralytics).
    try:
        result = trainer_fn(
            pt_path=pt_path,
            pruned_pt_path=pruned_pt_path,
            data_yaml=data_yaml,
            epochs=finetune_epochs,
            imgsz=imgsz,
            batch=batch,
            lr=lr,
            runs_dir=runs_dir,
            name="yolov8n_cat_pruned",
            yolo_factory=yolo_factory,
            pruner_fn=pruner_fn,
            sparsity=sparsity,
            exclude_head=exclude_head,
            device=device,
        )
        # trainer_fn may return either Path or (Path, prune_summary).
        if isinstance(result, tuple):
            pruned_pt_path, prune_summary = result
        else:
            pruned_pt_path = Path(result)
    except Exception as exc:  # noqa: BLE001
        blocked_reason = (
            f"prune+finetune step failed: {type(exc).__name__}: {exc}"
        )
        traceback.print_exc()

    # 3. Quantize via the iter-A per-channel-friendly converter.
    if blocked_reason is None:
        try:
            quantize_fn(
                pt_path=pruned_pt_path,
                calib_dir=calib_dir,
                out_path=out_tflite,
                imgsz=imgsz,
            )
        except Exception as exc:  # noqa: BLE001
            blocked_reason = f"quantize step failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()

    # 4. Eval
    if blocked_reason is None:
        try:
            eval_dict = eval_fn(
                model_path=out_tflite,
                story_id="iter-C_eval",
                val_dir=val_dir,
                imgsz=imgsz,
                results_dir=results_dir,
            )
        except Exception as exc:  # noqa: BLE001
            blocked_reason = f"eval step failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()

    # 5. Bench
    if blocked_reason is None:
        try:
            tflm_dict = bench_fn(
                model_path=out_tflite,
                story_id="iter-C_bench",
                results_dir=results_dir,
                runs=bench_runs,
            )
        except Exception as exc:  # noqa: BLE001
            blocked_reason = f"bench step failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()

    doc = _build_aggregate_doc(
        story_id="iter-C",
        model_path=out_tflite if (blocked_reason is None or eval_dict) else pruned_pt_path,
        eval_dict=eval_dict,
        tflm_dict=tflm_dict,
        prune_summary=prune_summary,
        multiplier=multiplier,
        multiplier_source=multiplier_source,
        baseline_metrics=baseline_metrics,
        blocked_reason=blocked_reason,
    )
    out_path = _write_aggregate(doc, results_dir)
    print(
        f"iter-C -> {out_path.name}  status={doc['status']} "
        f"verdict={doc['pareto']['verdict']}"
    )
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prune_channels")
    ap.add_argument("--model", type=Path, default=DEFAULT_PT)
    ap.add_argument("--sparsity", type=float, default=DEFAULT_SPARSITY)
    ap.add_argument("--finetune-epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PT)
    ap.add_argument("--out-tflite", type=Path, default=DEFAULT_OUT_TFLITE)
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    ap.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--bench-runs", type=int, default=DEFAULT_BENCH_RUNS)
    ap.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Device for ultralytics .train() (empty string = auto-detect, 'cpu', or '0')",
    )
    ap.add_argument(
        "--include-head",
        action="store_true",
        help="Prune the Detect head too (default: skip the head)",
    )
    args = ap.parse_args(argv)

    run_pipeline(
        pt_path=args.model,
        pruned_pt_path=args.out,
        out_tflite=args.out_tflite,
        data_yaml=args.data,
        calib_dir=args.calib_dir,
        val_dir=args.val_dir,
        results_dir=args.results_dir,
        runs_dir=args.runs_dir,
        sparsity=args.sparsity,
        exclude_head=not args.include_head,
        finetune_epochs=args.finetune_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr=args.lr,
        bench_runs=args.bench_runs,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
