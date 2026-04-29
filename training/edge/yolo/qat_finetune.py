"""US-005: Conditional QAT recovery if PTQ mAP dropped > 3 points.

This script reads ``training/edge/results/US-003.json`` (fp32 PyTorch baseline)
and ``training/edge/results/US-004.json`` (PTQ INT8) and decides whether to
run a one-epoch quantization-aware fine-tune:

    mAP_drop = US-003.map50 - US-004.map50

    if mAP_drop <= 3.0  -> SKIP: write US-005.json with status='skipped'
    if mAP_drop >  3.0  -> RUN:  one epoch of QAT @ lr=1e-4, then re-export
                                  to INT8 TFLite via the US-004 PTQ pipeline.

The actual QAT step is deliberately small (one epoch) per the input spec.
We attempt to insert fake-quant observers via ``torch.ao.quantization.prepare_qat``
on the model's ``model.model`` torch.nn.Module so quantization noise is
visible during the fine-tune. If torch.ao.quantization rejects the YOLOv8
graph (custom ops, dynamic shapes), we fall back to a plain fp32 fine-tune
at lr=1e-4 with a clear note in the result — the conditional design says
to run "one epoch of QAT" as a recovery attempt; failing to insert observers
is itself a finding worth recording.

Important: the PRD's mAP_drop threshold is in **mAP points**. The
``EvalResult.map50`` field is a float in [0,1] where 0.909 means 90.9 mAP
points. The script multiplies by 100 before comparing so the "3 points"
threshold matches its plain-English meaning.

Usage:
    python -m training.edge.yolo.qat_finetune \\
        --us003 training/edge/results/US-003.json \\
        --us004 training/edge/results/US-004.json \\
        --base training/edge/models/yolov8n_cat.pt \\
        --epochs 1 --lr 1e-4 --imgsz 224 \\
        --out training/edge/models/yolov8n_cat_qat_int8.tflite \\
        --results-dir training/edge/results

Test contract:
    ``decide(us003_path, us004_path)`` returns a dict that test_qat_decision.py
    can assert on without invoking the orchestrator.
    ``orchestrate(...)`` runs the chosen branch with DI hooks for the train
    step, the export step, and the eval step so the parametrized test mocks
    the heavy work.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# onnx2tf needs Keras-2-style APIs; this env var must be set before any
# tensorflow/keras import that the export step triggers. Mirrors
# training/edge/yolo/export_quantize.py — repeat here because qat_finetune
# is the entry point for the QAT branch and import order is not guaranteed.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

DEFAULT_US003 = Path("training/edge/results/US-003.json")
DEFAULT_US004 = Path("training/edge/results/US-004.json")
DEFAULT_BASE = Path("training/edge/models/yolov8n_cat.pt")
DEFAULT_DATA_YAML = Path("training/edge/data/labeled/data.yaml")
DEFAULT_RUNS_DIR = Path("training/edge/yolo/runs")
DEFAULT_OUT = Path("training/edge/models/yolov8n_cat_qat_int8.tflite")
DEFAULT_RESULTS_DIR = Path("training/edge/results")
DEFAULT_VAL_DIR = Path("training/edge/data/labeled/val")
DEFAULT_THRESHOLD_POINTS = 3.0


@dataclass
class Decision:
    map_drop_points: float          # mAP_drop in absolute mAP-points (0..100)
    threshold_points: float         # the cutoff (default 3.0)
    run_qat: bool                   # True iff map_drop_points > threshold
    us003_map50: float              # fp32 baseline (fraction 0..1)
    us004_map50: float              # PTQ INT8 (fraction 0..1)


def decide(
    us003_path: Path = DEFAULT_US003,
    us004_path: Path = DEFAULT_US004,
    threshold_points: float = DEFAULT_THRESHOLD_POINTS,
) -> Decision:
    """Read prior eval JSONs and decide whether QAT is required.

    Both paths must exist; missing files surface as FileNotFoundError so the
    pipeline fails loud rather than silently picking a default.
    """
    us003 = json.loads(Path(us003_path).read_text())
    us004 = json.loads(Path(us004_path).read_text())
    m3 = float(us003["map50"])
    m4 = float(us004["map50"])
    drop_points = (m3 - m4) * 100.0
    return Decision(
        map_drop_points=drop_points,
        threshold_points=threshold_points,
        run_qat=drop_points > threshold_points,
        us003_map50=m3,
        us004_map50=m4,
    )


def write_skipped_result(
    decision: Decision,
    out_path: Path,
    notes: str | None = None,
) -> Path:
    """Emit the canonical 'skipped' US-005.json payload.

    The JSON is shaped so it round-trips through the same field names as an
    EvalResult plus a leading ``status`` flag — US-012's aggregator can pick
    out the skipped branch via ``status == 'skipped'``.
    """
    explanation = (
        f"PTQ mAP drop = {decision.map_drop_points:.3f} points "
        f"<= threshold {decision.threshold_points} points; QAT skipped."
    )
    payload = {
        "story_id": "US-005",
        "status": "skipped",
        "map_drop_points": decision.map_drop_points,
        "threshold_points": decision.threshold_points,
        "us003_map50": decision.us003_map50,
        "us004_map50": decision.us004_map50,
        "model_path": "",
        "model_format": "tflite_int8",
        "map50": decision.us004_map50,  # carry forward PTQ score for SUMMARY
        "size_bytes": 0,
        "params": 0,
        "flops": 0,
        "input_hw": [224, 224],
        "latency_ms_p50": 0.0,
        "latency_ms_p95": 0.0,
        "val_images": 0,
        "notes": notes or explanation,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def qat_one_epoch(
    base_pt: Path,
    data_yaml: Path,
    runs_dir: Path,
    name: str = "yolov8n_cat_qat",
    epochs: int = 1,
    lr: float = 1e-4,
    imgsz: int = 224,
    batch: int = 32,
    device: str = "auto",
    yolo_factory: Any = None,
) -> tuple[Path, dict[str, Any]]:
    """Run one epoch of QAT-flavored fine-tuning, return (best.pt, info_dict).

    Strategy:
      1. Load the US-003 fine-tuned checkpoint (yolov8n_cat.pt).
      2. Try to swap the underlying nn.Module into QAT mode via
         ``torch.ao.quantization.prepare_qat``. If torch raises (the YOLOv8
         graph mixes ops the prepare_qat tracer can't handle, e.g. the
         Detect head's anchor reshape), fall back to plain fine-tune and
         record ``info['fake_quant_inserted'] = False`` so the markdown
         report reflects what actually happened.
      3. Hand control to ``ultralytics.YOLO.train(...)`` for one epoch with
         lr0=lr (1e-4 default). Ultralytics handles the data loop, val,
         and best.pt selection.

    The two-step (insert fake-quant, then ultralytics fine-tune) keeps us
    aligned with both the input spec ("insert fake-quant nodes via
    torch.ao.quantization") and the dependency surface (ultralytics owns
    the optimizer + LR schedule + best.pt rewrite).
    """
    if yolo_factory is None:
        from ultralytics import YOLO  # lazy

        yolo_factory = YOLO

    runs_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        "fake_quant_inserted": False,
        "fallback_reason": "",
        "epochs": epochs,
        "lr": lr,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
    }

    model = yolo_factory(str(base_pt))

    # Attempt fake-quant insertion. We do this BEFORE train() so the
    # subsequent forward passes see fake-quant ops on every conv. If the
    # tracer can't handle the graph we silently fall back; the failure is
    # recorded in info_dict and surfaced in US-005.md.
    try:
        import torch.ao.quantization as taq

        inner = model.model  # ultralytics' wrapped nn.Module
        # prepare_qat asserts training mode; ultralytics loads checkpoints in
        # eval mode by default, so flip it before the fake-quant insertion.
        inner.train()
        # qconfig must be set before prepare_qat; per-channel weight + per-tensor
        # activation is the default symmetric int8 setup most edge runtimes use.
        qconfig = taq.get_default_qat_qconfig("fbgemm")
        inner.qconfig = qconfig
        # prepare_qat is in-place. Wrap in a try so a failure is recoverable.
        taq.prepare_qat(inner, inplace=True)
        info["fake_quant_inserted"] = True
    except Exception as exc:  # noqa: BLE001 — broad on purpose; document the cause
        info["fake_quant_inserted"] = False
        info["fallback_reason"] = f"{type(exc).__name__}: {exc}"

    # ultralytics' train() accepts lr0 (initial LR) and lrf (final LR factor).
    # For a one-epoch fine-tune we want lr to stay near the requested value;
    # set lrf=1.0 so cosine decay is effectively flat.
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        single_cls=True,
        device=device,
        project=str(runs_dir),
        name=name,
        exist_ok=True,
        lr0=lr,
        lrf=1.0,
        verbose=True,
    )

    best = runs_dir / name / "weights" / "best.pt"
    if not best.exists():
        last = runs_dir / name / "weights" / "last.pt"
        if last.exists():
            best = last
    return best, info


def export_qat_to_int8_tflite(
    qat_pt: Path,
    out_tflite: Path,
    calib_dir: Path,
    imgsz: int = 224,
    max_calib_frames: int = 200,
    work_dir: Path | None = None,
    export_fn: Callable[..., Path] | None = None,
) -> Path:
    """Re-export a QAT-fine-tuned .pt through the US-004 PTQ pipeline.

    Reuses ``training.edge.yolo.export_quantize.export_and_quantize`` so the
    quantization recipe stays identical to PTQ — the QAT contribution shows
    up purely as different weights, not a different export path.
    """
    if export_fn is None:
        from training.edge.yolo.export_quantize import export_and_quantize

        export_fn = export_and_quantize
    return export_fn(
        pt_path=qat_pt,
        calib_dir=calib_dir,
        out_path=out_tflite,
        imgsz=imgsz,
        max_calib_frames=max_calib_frames,
        work_dir=work_dir,
    )


def run_eval_on_qat(
    tflite_path: Path,
    val_dir: Path,
    results_dir: Path,
    imgsz: int = 224,
    notes: str = "",
    eval_fn: Callable[..., Any] | None = None,
) -> Path:
    """Score the QAT INT8 .tflite through the eval harness; write US-005.json."""
    if eval_fn is None:
        from training.edge.eval.run_eval import evaluate

        eval_fn = evaluate
    result = eval_fn(
        str(tflite_path),
        "tflite_int8",
        "US-005",
        val_dir,
        imgsz=imgsz,
        notes=notes,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "US-005.json"
    # EvalResult instances expose a .write(path) helper; mocked eval_fn is
    # free to return any object with that same interface.
    if hasattr(result, "write"):
        result.write(out_json)
    else:
        out_json.write_text(json.dumps(result, indent=2))
    return out_json


def orchestrate(
    us003_path: Path = DEFAULT_US003,
    us004_path: Path = DEFAULT_US004,
    base_pt: Path = DEFAULT_BASE,
    data_yaml: Path = DEFAULT_DATA_YAML,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    out_tflite: Path = DEFAULT_OUT,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    val_dir: Path = DEFAULT_VAL_DIR,
    calib_dir: Path = Path("training/edge/data/calibration_frames"),
    epochs: int = 1,
    lr: float = 1e-4,
    imgsz: int = 224,
    batch: int = 32,
    device: str = "auto",
    threshold_points: float = DEFAULT_THRESHOLD_POINTS,
    qat_fn: Callable[..., tuple[Path, dict[str, Any]]] | None = None,
    export_fn: Callable[..., Path] | None = None,
    eval_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Top-level orchestrator: decide -> (skip | qat -> export -> eval).

    Returns a small summary dict useful for both the CLI ``main`` and the
    pytest suite, so the tests can assert on which branch ran without
    re-parsing the on-disk JSON.
    """
    decision = decide(us003_path, us004_path, threshold_points=threshold_points)
    summary: dict[str, Any] = {
        "decision": decision,
        "branch": "skipped" if not decision.run_qat else "qat",
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "US-005.json"

    if not decision.run_qat:
        write_skipped_result(decision, out_json)
        summary["json_path"] = out_json
        return summary

    # QAT branch
    if qat_fn is None:
        qat_fn = qat_one_epoch
    best_pt, qat_info = qat_fn(
        base_pt=base_pt,
        data_yaml=data_yaml,
        runs_dir=runs_dir,
        epochs=epochs,
        lr=lr,
        imgsz=imgsz,
        batch=batch,
        device=device,
    )

    # Copy best.pt to a canonical location next to other model artifacts so
    # downstream stories (US-011 TFLM bench) can rely on a stable path.
    canonical_pt = Path("training/edge/models/yolov8n_cat_qat.pt")
    canonical_pt.parent.mkdir(parents=True, exist_ok=True)
    if best_pt.exists() and best_pt.resolve() != canonical_pt.resolve():
        canonical_pt.write_bytes(best_pt.read_bytes())

    tflite_path = export_qat_to_int8_tflite(
        qat_pt=canonical_pt if canonical_pt.exists() else best_pt,
        out_tflite=out_tflite,
        calib_dir=calib_dir,
        imgsz=imgsz,
        export_fn=export_fn,
    )

    notes = (
        f"QAT: 1 epoch @ lr={lr}, fake_quant_inserted={qat_info['fake_quant_inserted']}"
    )
    if qat_info.get("fallback_reason"):
        notes += f"; fallback_reason={qat_info['fallback_reason']}"
    json_path = run_eval_on_qat(
        tflite_path=tflite_path,
        val_dir=val_dir,
        results_dir=results_dir,
        imgsz=imgsz,
        notes=notes,
        eval_fn=eval_fn,
    )
    summary["qat_info"] = qat_info
    summary["pt_path"] = canonical_pt
    summary["tflite_path"] = tflite_path
    summary["json_path"] = json_path
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qat_finetune")
    ap.add_argument("--us003", type=Path, default=DEFAULT_US003)
    ap.add_argument("--us004", type=Path, default=DEFAULT_US004)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    ap.add_argument(
        "--calib-dir",
        type=Path,
        default=Path("training/edge/data/calibration_frames"),
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--threshold-points",
        type=float,
        default=DEFAULT_THRESHOLD_POINTS,
        help="mAP-point cutoff for the QAT decision (default 3.0)",
    )
    args = ap.parse_args(argv)

    summary = orchestrate(
        us003_path=args.us003,
        us004_path=args.us004,
        base_pt=args.base,
        data_yaml=args.data_yaml,
        runs_dir=args.runs_dir,
        out_tflite=args.out,
        results_dir=args.results_dir,
        val_dir=args.val_dir,
        calib_dir=args.calib_dir,
        epochs=args.epochs,
        lr=args.lr,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        threshold_points=args.threshold_points,
    )
    decision = summary["decision"]
    if not decision.run_qat:
        print(
            f"PTQ acceptable; QAT skipped. (mAP_drop={decision.map_drop_points:.3f} pts"
            f" <= threshold {decision.threshold_points} pts)"
        )
    else:
        print(
            f"QAT ran. branch={summary['branch']} "
            f"json={summary.get('json_path')} tflite={summary.get('tflite_path')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
