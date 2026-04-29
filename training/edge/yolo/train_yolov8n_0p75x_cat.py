"""iter-E: YOLOv8n width=0.75x retrain + KD distill + per-channel-friendly INT8 PTQ.

Pipeline
--------
    yolov8n_0p75x.yaml (depth=0.33, width=0.1875)  +  yolov8n.pt initial weights
                    │  partial-load: copy overlapping channel slices into the
                    │  narrower student. Strict load_state_dict will fail
                    │  because every Conv2d has 25 % fewer output channels;
                    │  we accept the partial load and let the distill+
                    │  finetune step recover.
                    ▼
    width=0.75x student .pt
                    │  re-use training.edge.yolo.distill_train.train_distilled
                    │  with yolov8s.pt as teacher and alpha=0.5 (matches US-006).
                    ▼
    yolov8n_0p75x_cat_distilled.pt (canonical fp32)
                    │  re-use
                    │  training.edge.yolo.per_channel_quant.export_per_channel_int8
                    │  (the iter-A pipeline) so the calibration set + PTQ
                    │  recipe is byte-identical to iter-A / iter-B / iter-C.
                    ▼
    yolov8n_0p75x_cat_distilled_int8_pc.tflite
                    │  run_eval -> training/edge/results/iter-E_eval.json
                    │  edge-bench -> training/edge/results/iter-E_bench-tflm.json
                    ▼
    Pareto verdict vs iter-A frontier, written to
    training/edge/results/iter-E_yolov8n_0p75x.json (rollup).

Failure isolation
-----------------
Every stage (config-prep / weight-init / distill / quantize / eval / bench)
is wrapped in its own ``try/except`` and on failure the rollup is written
with ``status="blocked"`` plus a concrete reason. Later stages are skipped
but the JSON IS still committed — same isolation contract used by
``training/edge/yolo/imgsz_sweep.py`` and ``training/edge/yolo/prune_channels.py``.

User-facing knob: ``width_multiple``
------------------------------------
The CLI exposes ``--width-multiple`` (default ``0.75``) with the SEMANTIC
of "fraction of yolov8n's already-narrow channels". The yaml stamped at
``training/edge/yolo/configs/yolov8n_0p75x.yaml`` carries the COMPOSED
value ``width_multiple = 0.25 * <user-knob>`` (i.e. ``0.1875`` for the
default), because ultralytics' ``parse_model`` interprets ``width_multiple``
as a fraction of the YOLOv8 base channel widths — not yolov8n's. The two
spaces are clearly separated in the wrapper, and tests assert the
user-facing ``0.75`` is what reaches the trainer kwargs.

Usage
-----

::

    python -m training.edge.yolo.train_yolov8n_0p75x_cat \\
        --teacher yolov8s.pt \\
        --base yolov8n.pt \\
        --config training/edge/yolo/configs/yolov8n_0p75x.yaml \\
        --width-multiple 0.75 \\
        --imgsz 224 \\
        --epochs 50 --batch 32 --lr 1e-4

The default arguments wire to the iter-E acceptance-criteria values; the
bare invocation reproduces the iter-E result.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Same env opt-in as US-004/US-006/iter-A — must be set BEFORE TF/keras
# imports the quantize step triggers.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Make firmware/edge-bench/ importable for run_bench. Same pattern as
# imgsz_sweep.py and prune_channels.py (the dir has a hyphen so it's not a
# python package — we manually inject it into sys.path).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCH_DIR = _REPO_ROOT / "firmware" / "edge-bench"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import yaml as _yaml  # noqa: E402

from training.edge.yolo.per_channel_quant import (  # noqa: E402
    DEFAULT_P4_MULTIPLIER,
    DEFAULT_P4_MULTIPLIER_SOURCE,
    build_pareto_verdict,
    export_per_channel_int8,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TEACHER_PT = Path("yolov8s.pt")
DEFAULT_BASE_PT = Path("yolov8n.pt")
DEFAULT_CONFIG_YAML = Path("training/edge/yolo/configs/yolov8n_0p75x.yaml")
DEFAULT_DATA_YAML = Path("training/edge/data/labeled/data.yaml")
DEFAULT_FP32_OUT = Path("training/edge/models/yolov8n_0p75x_cat_distilled.pt")
DEFAULT_INT8_OUT = Path(
    "training/edge/models/yolov8n_0p75x_cat_distilled_int8_pc.tflite"
)
DEFAULT_CALIB_DIR = Path("training/edge/data/calibration_frames")
DEFAULT_VAL_DIR = Path("training/edge/data/labeled/val")
DEFAULT_RESULTS_DIR = Path("training/edge/results")
DEFAULT_RUNS_DIR = Path("training/edge/yolo/runs")

DEFAULT_WIDTH_MULTIPLE = 0.75       # vs yolov8n's already-narrow 0.25 base
DEFAULT_DEPTH_MULTIPLE = 1.0        # vs yolov8n base depth (0.33) — kept as-is
DEFAULT_IMGSZ = 224                 # iter-B winner
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 50
DEFAULT_ALPHA = 0.5                 # KD weight; matches US-006
DEFAULT_TEMPERATURE = 4.0           # KD temperature; matches US-006
DEFAULT_BENCH_RUNS = 50
DEFAULT_DEVICE = ""                 # empty -> ultralytics auto-detects

YOLOV8N_BASE_WIDTH = 0.25           # yolov8n's effective width (vs YOLOv8 base)
YOLOV8N_BASE_DEPTH = 0.33           # yolov8n's effective depth (vs YOLOv8 base)

BASELINE_AGGREGATE_PATH = DEFAULT_RESULTS_DIR / "iter-A_per_channel_quant.json"
BASELINE_STORY = "iter-A (per-channel-friendly INT8 PTQ; v2 accuracy floor)"


# ---------------------------------------------------------------------------
# Stage 1 — config + initial-weights handling
# ---------------------------------------------------------------------------


def regenerate_config(
    config_path: Path = DEFAULT_CONFIG_YAML,
    *,
    width_multiple: float = DEFAULT_WIDTH_MULTIPLE,
    depth_multiple: float = DEFAULT_DEPTH_MULTIPLE,
    yolov8n_base_width: float = YOLOV8N_BASE_WIDTH,
    yolov8n_base_depth: float = YOLOV8N_BASE_DEPTH,
) -> dict:
    """Stamp the *effective* multipliers into ``config_path`` and return the dict.

    User-facing knobs are **vs yolov8n** (e.g. ``width_multiple=0.75`` =
    "75 % of yolov8n's narrow width"). ultralytics' ``parse_model`` reads
    the yaml's ``width_multiple`` / ``depth_multiple`` as fractions of the
    YOLOv8 BASE channel/repeat counts (per ``ultralytics/nn/tasks.py:837``),
    so we compose with the yolov8n base values before stamping.

    Composed values stamped into the yaml:

      effective_width = yolov8n_base_width * width_multiple
      effective_depth = yolov8n_base_depth * depth_multiple

    Returns the in-memory dict for tests to assert against without re-reading
    the file.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"missing iter-E base config at {config_path}; "
            "run from repo root or pass --config explicitly"
        )
    cfg = _yaml.safe_load(config_path.read_text()) or {}
    effective_width = float(yolov8n_base_width) * float(width_multiple)
    effective_depth = float(yolov8n_base_depth) * float(depth_multiple)
    cfg["width_multiple"] = effective_width
    cfg["depth_multiple"] = effective_depth
    cfg["nc"] = 1
    config_path.write_text(_yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def partial_load_yolov8n_weights(
    student_model: Any,
    base_pt: Path,
    *,
    yolo_factory: Callable[[str], Any] | None = None,
    torch_module: Any = None,
) -> dict:
    """Best-effort copy of overlapping channel slices from yolov8n.pt -> student.

    Strict ``load_state_dict`` fails when channel counts differ (every
    Conv2d in the 0.75x student has fewer out_channels than yolov8n). We
    instead walk the parameter dicts, match keys, and slice each donor
    tensor to the student's smaller shape along the leading channel axis.
    Any key whose shape can't be aligned (after slicing) is skipped and
    counted in ``skipped_mismatched_shape`` — they get random-init weights.

    Returns a summary dict::

        {
          "base_pt": "yolov8n.pt",
          "matched_keys_total": int,
          "loaded_keys": int,
          "skipped_missing_in_student": int,
          "skipped_mismatched_shape": int,
          "fallback_random_init": bool,
        }

    Falls back to a fresh random-init student if base_pt cannot be loaded
    at all (per the AC: "or from-scratch if the partial load fails").
    """
    if yolo_factory is None:
        from ultralytics import YOLO  # lazy

        yolo_factory = YOLO
    if torch_module is None:
        import torch  # lazy

        torch_module = torch

    summary = {
        "base_pt": str(base_pt),
        "matched_keys_total": 0,
        "loaded_keys": 0,
        "skipped_missing_in_student": 0,
        "skipped_mismatched_shape": 0,
        "fallback_random_init": False,
    }
    if not base_pt.exists():
        summary["fallback_random_init"] = True
        summary["error"] = f"base_pt missing at {base_pt}"
        return summary

    try:
        donor = yolo_factory(str(base_pt))
        # Donor is the YOLO wrapper; `.model` is the DetectionModel whose
        # state_dict has keys like `model.0.conv.weight`. We want to read at
        # the DetectionModel level (one wrap deep) — same level the student
        # is at when the caller passes ``student.model``.
        donor_state = donor.model.state_dict()
    except Exception as exc:  # noqa: BLE001
        summary["fallback_random_init"] = True
        summary["error"] = f"donor load failed: {type(exc).__name__}: {exc}"
        return summary

    # Caller passes the student's DetectionModel directly (e.g. ``student.model``
    # where ``student = YOLO(yaml)``); read state at the same level so the
    # ``model.<idx>.<...>`` key names match the donor's exactly.
    student_state = student_model.state_dict()

    for key, donor_tensor in donor_state.items():
        if key not in student_state:
            summary["skipped_missing_in_student"] += 1
            continue
        summary["matched_keys_total"] += 1
        student_tensor = student_state[key]
        if tuple(donor_tensor.shape) == tuple(student_tensor.shape):
            student_tensor.copy_(donor_tensor)
            summary["loaded_keys"] += 1
            continue
        # Slice donor to student shape along each axis where donor is bigger.
        try:
            slices = tuple(slice(0, s) for s in student_tensor.shape)
            sliced = donor_tensor[slices]
            if tuple(sliced.shape) != tuple(student_tensor.shape):
                summary["skipped_mismatched_shape"] += 1
                continue
            student_tensor.copy_(sliced)
            summary["loaded_keys"] += 1
        except Exception:  # noqa: BLE001 - per-key best-effort copy
            summary["skipped_mismatched_shape"] += 1

    student_model.load_state_dict(student_state)
    return summary


# ---------------------------------------------------------------------------
# Stage 2 — train (KD distill)
# ---------------------------------------------------------------------------


def _default_distill_train_fn(
    *,
    student: Any,
    teacher_pt: Path,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    lr: float,
    alpha: float,
    temperature: float,
    runs_dir: Path,
    name: str,
    device: str,
) -> Path:
    """Run KD distillation directly on the live student YOLO instance.

    Mirrors ``training.edge.yolo.distill_train.train_distilled`` but skips the
    ``student.save()`` round-trip — that path requires a populated
    ``self.ckpt`` dict which YAML-built models don't have. We instead register
    the same ``on_train_start`` callback to patch the student's loss with the
    distill term, then call ``student.train(...)``.

    Returns the path to ``best.pt`` (or ``last.pt`` fallback).
    """
    from ultralytics import YOLO  # lazy
    from training.edge.yolo.distill_train import patch_student_loss_for_distill

    teacher = YOLO(str(teacher_pt))

    def _on_train_start(trainer):
        device_ = next(trainer.model.parameters()).device
        teacher.model.to(device_)
        patch_student_loss_for_distill(
            trainer.model,
            teacher.model,
            alpha=alpha,
            temperature=temperature,
        )

    student.add_callback("on_train_start", _on_train_start)
    runs_dir.mkdir(parents=True, exist_ok=True)
    student.train(
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
    return best


def train_with_distill(
    *,
    config_yaml: Path,
    teacher_pt: Path,
    base_pt: Path,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    lr: float,
    alpha: float,
    temperature: float,
    runs_dir: Path,
    name: str,
    width_multiple: float,
    depth_multiple: float,
    device: str,
    yolo_factory: Callable[[str], Any] | None = None,
    distill_train_fn: Callable[..., Path] | None = None,
    config_writer: Callable[..., dict] | None = None,
    weight_loader: Callable[..., dict] | None = None,
) -> tuple[Path, dict]:
    """Compose config + initial weights, then run KD distillation.

    Returns ``(best_pt_path, init_summary)`` where ``init_summary`` is the
    output of ``partial_load_yolov8n_weights``. ``distill_train_fn`` is a
    DI seam: defaults to ``_default_distill_train_fn`` which runs KD on the
    live student instance (no .pt round-trip required).
    """
    if yolo_factory is None:
        from ultralytics import YOLO  # lazy

        yolo_factory = YOLO
    if distill_train_fn is None:
        distill_train_fn = _default_distill_train_fn
    if config_writer is None:
        config_writer = regenerate_config
    if weight_loader is None:
        weight_loader = partial_load_yolov8n_weights

    # 1. Stamp effective multipliers into the yaml.
    config_writer(
        config_yaml,
        width_multiple=width_multiple,
        depth_multiple=depth_multiple,
    )

    # 2. Build the student from the (now updated) yaml. Then partial-load
    #    yolov8n.pt into it. This is the "coarse copy of overlapping
    #    channels" step the AC describes.
    student = yolo_factory(str(config_yaml))
    init_summary = weight_loader(student.model, base_pt)

    # 3. Run KD distillation against the yolov8s teacher on the LIVE student.
    best_pt = distill_train_fn(
        student=student,
        teacher_pt=teacher_pt,
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr=lr,
        alpha=alpha,
        temperature=temperature,
        runs_dir=runs_dir,
        name=name,
        device=device,
    )
    return Path(best_pt), init_summary


def copy_to_canonical(src: Path, dest: Path = DEFAULT_FP32_OUT) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


# ---------------------------------------------------------------------------
# Stages 3-5 — quantize / eval / bench (DI defaults match prune_channels.py)
# ---------------------------------------------------------------------------


def _default_quantize_fn(
    *, pt_path: Path, calib_dir: Path, out_path: Path, imgsz: int
) -> Path:
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


def _x86_us_to_p4_ms(us: float, multiplier: float) -> float:
    return float(us) * float(multiplier) / 1000.0


def _build_aggregate_doc(
    *,
    story_id: str,
    width_multiple: float,
    depth_multiple: float,
    model_path: Path,
    eval_dict: dict | None,
    tflm_dict: dict | None,
    init_summary: dict | None,
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
            f"YOLOv8n width={width_multiple}x retrain (depth={depth_multiple}) "
            "+ KD distill from yolov8s + per-channel-friendly INT8 PTQ"
        ),
        "status": "blocked" if blocked_reason else "passed",
        "blocked_reason": blocked_reason,
        "model_path": str(model_path),
        "p4_multiplier": multiplier,
        "p4_multiplier_source": multiplier_source,
        "width_multiple": float(width_multiple),  # user-facing (vs yolov8n)
        "depth_multiple": float(depth_multiple),  # user-facing (vs yolov8n)
        "init_summary": init_summary,
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
    out_path = results_dir / "iter-E_yolov8n_0p75x.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


TrainerFn = Callable[..., tuple[Path, dict]]
QuantizeFn = Callable[..., Path]
EvalFn = Callable[..., dict]
BenchFn = Callable[..., dict]
BaselineLoaderFn = Callable[[], dict]


def run_pipeline(
    *,
    teacher_pt: Path = DEFAULT_TEACHER_PT,
    base_pt: Path = DEFAULT_BASE_PT,
    config_yaml: Path = DEFAULT_CONFIG_YAML,
    data_yaml: Path = DEFAULT_DATA_YAML,
    fp32_out: Path = DEFAULT_FP32_OUT,
    int8_out: Path = DEFAULT_INT8_OUT,
    calib_dir: Path = DEFAULT_CALIB_DIR,
    val_dir: Path = DEFAULT_VAL_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    width_multiple: float = DEFAULT_WIDTH_MULTIPLE,
    depth_multiple: float = DEFAULT_DEPTH_MULTIPLE,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    alpha: float = DEFAULT_ALPHA,
    temperature: float = DEFAULT_TEMPERATURE,
    bench_runs: int = DEFAULT_BENCH_RUNS,
    device: str = DEFAULT_DEVICE,
    multiplier: float = DEFAULT_P4_MULTIPLIER,
    multiplier_source: str = DEFAULT_P4_MULTIPLIER_SOURCE,
    name: str = "yolov8n_0p75x_cat_distilled",
    trainer_fn: TrainerFn | None = None,
    quantize_fn: QuantizeFn | None = None,
    eval_fn: EvalFn | None = None,
    bench_fn: BenchFn | None = None,
    baseline_loader: BaselineLoaderFn | None = None,
) -> dict:
    """Run config-prep -> distill -> quant -> eval -> bench. Always writes JSON.

    Each stage is DI-overridable so tests can substitute mocks. Defaults
    wire to ultralytics + distill_train + per_channel_quant + eval +
    edgebench. On any stage failure the pipeline records
    ``status="blocked"`` with a concrete reason and returns; later stages
    are skipped but iter-E_yolov8n_0p75x.json IS still committed (per the
    "emit JSON anyway" AC).
    """
    trainer_fn = trainer_fn or train_with_distill
    quantize_fn = quantize_fn or _default_quantize_fn
    eval_fn = eval_fn or _default_eval_fn
    bench_fn = bench_fn or _default_bench_fn
    baseline_loader = baseline_loader or _default_baseline_loader

    results_dir.mkdir(parents=True, exist_ok=True)

    # Baseline (iter-A frontier). Load this first so a blocked rollup still
    # has baseline_metrics filled in for iter-H ingestion.
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
            story_id="iter-E",
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            model_path=fp32_out,
            eval_dict=None,
            tflm_dict=None,
            init_summary=None,
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
    init_summary: dict | None = None
    blocked_reason: str | None = None

    # 1. Train (config prep + partial-load + KD distill)
    try:
        best_pt, init_summary = trainer_fn(
            config_yaml=config_yaml,
            teacher_pt=teacher_pt,
            base_pt=base_pt,
            data_yaml=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr=lr,
            alpha=alpha,
            temperature=temperature,
            runs_dir=runs_dir,
            name=name,
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            device=device,
        )
        copy_to_canonical(best_pt, fp32_out)
    except Exception as exc:  # noqa: BLE001
        blocked_reason = (
            f"train+distill step failed: {type(exc).__name__}: {exc}"
        )
        traceback.print_exc()

    # 2. Quantize via the iter-A per-channel-friendly converter.
    if blocked_reason is None:
        try:
            quantize_fn(
                pt_path=fp32_out,
                calib_dir=calib_dir,
                out_path=int8_out,
                imgsz=imgsz,
            )
        except Exception as exc:  # noqa: BLE001
            blocked_reason = f"quantize step failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()

    # 3. Eval
    if blocked_reason is None:
        try:
            eval_dict = eval_fn(
                model_path=int8_out,
                story_id="iter-E_eval",
                val_dir=val_dir,
                imgsz=imgsz,
                results_dir=results_dir,
            )
        except Exception as exc:  # noqa: BLE001
            blocked_reason = f"eval step failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()

    # 4. Bench
    if blocked_reason is None:
        try:
            tflm_dict = bench_fn(
                model_path=int8_out,
                story_id="iter-E_bench",
                results_dir=results_dir,
                runs=bench_runs,
            )
        except Exception as exc:  # noqa: BLE001
            blocked_reason = f"bench step failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()

    doc = _build_aggregate_doc(
        story_id="iter-E",
        width_multiple=width_multiple,
        depth_multiple=depth_multiple,
        model_path=int8_out if (blocked_reason is None or eval_dict) else fp32_out,
        eval_dict=eval_dict,
        tflm_dict=tflm_dict,
        init_summary=init_summary,
        multiplier=multiplier,
        multiplier_source=multiplier_source,
        baseline_metrics=baseline_metrics,
        blocked_reason=blocked_reason,
    )
    out_path = _write_aggregate(doc, results_dir)
    print(
        f"iter-E -> {out_path.name}  status={doc['status']} "
        f"verdict={doc['pareto']['verdict']}"
    )
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train_yolov8n_0p75x_cat")
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER_PT)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE_PT)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_YAML)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    ap.add_argument("--fp32-out", type=Path, default=DEFAULT_FP32_OUT)
    ap.add_argument("--int8-out", type=Path, default=DEFAULT_INT8_OUT)
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    ap.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument(
        "--width-multiple",
        type=float,
        default=DEFAULT_WIDTH_MULTIPLE,
        help="fraction of yolov8n's width to retain (default 0.75 = iter-E spec)",
    )
    ap.add_argument(
        "--depth-multiple",
        type=float,
        default=DEFAULT_DEPTH_MULTIPLE,
        help="fraction of yolov8n's depth to retain (default 1.0 = unchanged)",
    )
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--bench-runs", type=int, default=DEFAULT_BENCH_RUNS)
    ap.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="ultralytics device flag ('' auto, 'cpu', '0' for cuda:0)",
    )
    args = ap.parse_args(argv)

    run_pipeline(
        teacher_pt=args.teacher,
        base_pt=args.base,
        config_yaml=args.config,
        data_yaml=args.data,
        fp32_out=args.fp32_out,
        int8_out=args.int8_out,
        calib_dir=args.calib_dir,
        val_dir=args.val_dir,
        results_dir=args.results_dir,
        runs_dir=args.runs_dir,
        width_multiple=args.width_multiple,
        depth_multiple=args.depth_multiple,
        imgsz=args.imgsz,
        batch=args.batch,
        lr=args.lr,
        epochs=args.epochs,
        alpha=args.alpha,
        temperature=args.temperature,
        bench_runs=args.bench_runs,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
