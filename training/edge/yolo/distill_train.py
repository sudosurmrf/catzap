"""US-006: Knowledge distillation YOLOv8s teacher -> YOLOv8n student (cat-only).

Pipeline:

    yolov8s.pt (teacher, frozen, eval, nc=80)
                    │  forward(img)
                    ▼
            t_preds: list[(B, 144, H, W)]   # 64 box-DFL + 80 cls
                    │
                    │  KL_distill on **cat channel** (teacher idx 64+15)
                    │  vs student class-0 channel (idx 64) — both single-channel
                    │  binary head; we use BCE-with-soft-targets at temperature T
                    │  (T**2 scaling per Hinton-style KD).
                    ▼
    yolov8n.pt (student, fresh, single_cls -> nc=1)
                    │  forward(img)  -> s_preds: list[(B, 65, H, W)]
                    │  + ultralytics' v8DetectionLoss(s_preds, batch)  → sup loss
                    ▼
            total = sup + alpha * distill
                    │  backward through student only (teacher.requires_grad_=False)
                    ▼
    yolov8n_cat_distilled.pt (canonical)
                    │  reuses training.edge.yolo.export_quantize.export_and_quantize
                    ▼
    yolov8n_cat_distilled_int8.tflite

Design notes:
  * Hook surface for the loss: ultralytics' BaseTrainer calls
    ``self.model(batch_dict)`` which, for a dict input, dispatches to
    ``BaseModel.loss(batch)``. We monkey-patch ``student.model.loss`` after
    instantiation to interleave the teacher forward + distill term. Same
    pattern qat_finetune.py uses — the trainer is left untouched.
  * Channel alignment: both teacher and student use ``Detect`` heads whose
    training output concatenates ``[4*reg_max box-DFL, nc cls]`` along channel
    1. With reg_max=16 fixed and yolov8n/yolov8s sharing the same strides
    (8/16/32), the spatial grids match for any common imgsz — no resampling.
  * Why cat-channel only (not multi-class softmax KL): the student is
    nc=1 (cat-only), so a class-axis softmax is degenerate. The teacher's
    cat channel is a binary sigmoid head; matching it via temperature-scaled
    BCE is the standard practice for binary-head KD (T**2 retains the same
    gradient magnitude as plain BCE).

Usage:
    python -m training.edge.yolo.distill_train \\
        --teacher yolov8s.pt \\
        --student yolov8n.pt \\
        --data training/edge/data/labeled/data.yaml \\
        --epochs 50 --imgsz 224 --batch 32 --device auto \\
        --alpha 0.5 --temperature 4.0 \\
        --fp32-out training/edge/models/yolov8n_cat_distilled.pt \\
        --int8-out training/edge/models/yolov8n_cat_distilled_int8.tflite

Test contract:
    ``compute_distill_loss(s_preds, t_preds, T, cat_idx)`` is pure.
    ``combined_loss(...)`` returns (total, sup, distill) — the test mocks
    sup_loss_fn / teacher / student preds and asserts:
        - total = sup + alpha * distill
        - teacher_preds.requires_grad is False (gradient detached)
        - total.backward() updates student parameters (gradient flows)
"""
from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

# Mirror export_quantize: TF_USE_LEGACY_KERAS=1 must be set BEFORE any
# tensorflow/keras import the quantize step triggers.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import torch
import torch.nn.functional as F  # noqa: E402

DEFAULT_TEACHER_PT = Path("yolov8s.pt")
DEFAULT_STUDENT_PT = Path("yolov8n.pt")
DEFAULT_DATA_YAML = Path("training/edge/data/labeled/data.yaml")
DEFAULT_RUNS_DIR = Path("training/edge/yolo/runs")
DEFAULT_FP32_OUT = Path("training/edge/models/yolov8n_cat_distilled.pt")
DEFAULT_INT8_OUT = Path("training/edge/models/yolov8n_cat_distilled_int8.tflite")
DEFAULT_CALIB_DIR = Path("training/edge/data/calibration_frames")
DEFAULT_NAME = "yolov8n_cat_distilled"

DEFAULT_ALPHA = 0.5  # weight on the distillation term in the combined loss
DEFAULT_TEMPERATURE = 4.0  # KD softening temperature; standard Hinton choice
DEFAULT_REG_MAX = 16  # ultralytics' default; first 4*reg_max channels are box-DFL
COCO_CAT_CLASS = 15  # mirrors server/vision/detector.py:4 (read-only reference)


# ----------------------------------------------------------------------------
# Distillation loss (pure, easy to test)
# ----------------------------------------------------------------------------


def _slice_cat_logits(
    preds: torch.Tensor,
    cat_channel_idx: int,
    reg_max: int = DEFAULT_REG_MAX,
) -> torch.Tensor:
    """Extract the single cat-channel logit from a Detect-head feature map.

    ``preds`` shape: (B, 4*reg_max + nc, H, W). The cls block starts at
    ``4*reg_max`` and ``cat_channel_idx`` indexes within the cls block.
    Returns (B, 1, H, W) preserving the channel dim.
    """
    cls_start = 4 * reg_max
    return preds[:, cls_start + cat_channel_idx : cls_start + cat_channel_idx + 1, :, :]


def compute_distill_loss(
    student_preds: Iterable[torch.Tensor],
    teacher_preds: Iterable[torch.Tensor],
    temperature: float = DEFAULT_TEMPERATURE,
    student_cat_idx: int = 0,           # student is single_cls (nc=1) -> idx 0
    teacher_cat_idx: int = COCO_CAT_CLASS,
    reg_max: int = DEFAULT_REG_MAX,
) -> torch.Tensor:
    """Temperature-scaled BCE distillation on the cat channel of each scale.

    Per-scale loss = BCE(student_logit / T, sigmoid(teacher_logit / T)) * T**2.
    Averaged over the three feature scales (P3/P4/P5) so the magnitude is
    invariant to the number of detection layers.

    Mirrors the binary-head KD recipe from Hinton 2015 §4 — the T**2 scaling
    keeps the gradient w.r.t. student logits comparable to a plain BCE.
    """
    s_list = list(student_preds)
    t_list = list(teacher_preds)
    if len(s_list) != len(t_list):
        raise ValueError(
            f"student/teacher emit different scale counts: {len(s_list)} vs {len(t_list)}"
        )
    if len(s_list) == 0:
        raise ValueError("empty preds — cannot compute distill loss")

    losses: list[torch.Tensor] = []
    T = float(temperature)
    for sp, tp in zip(s_list, t_list):
        if sp.shape[2:] != tp.shape[2:]:
            raise ValueError(
                f"scale shape mismatch: student {sp.shape} vs teacher {tp.shape}"
            )
        s_logit = _slice_cat_logits(sp, student_cat_idx, reg_max=reg_max)
        # Teacher must not contribute gradient — the .detach() is structural,
        # not just hygienic. Without it, calling backward through the combined
        # loss would walk into teacher.parameters() and mutate them.
        t_logit = _slice_cat_logits(tp, teacher_cat_idx, reg_max=reg_max).detach()
        soft_target = torch.sigmoid(t_logit / T)
        bce = F.binary_cross_entropy_with_logits(s_logit / T, soft_target)
        losses.append(bce * (T * T))
    return torch.stack(losses).mean()


def combined_loss(
    student_preds: Iterable[torch.Tensor],
    teacher_preds: Iterable[torch.Tensor],
    sup_loss: torch.Tensor,
    alpha: float = DEFAULT_ALPHA,
    temperature: float = DEFAULT_TEMPERATURE,
    student_cat_idx: int = 0,
    teacher_cat_idx: int = COCO_CAT_CLASS,
    reg_max: int = DEFAULT_REG_MAX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(total, sup_loss, distill_loss)``.

    ``total = sup_loss + alpha * distill_loss``. Caller handles ``.backward()``.
    The supervised loss is provided externally so this function stays unit-
    testable without ultralytics' v8DetectionLoss (which needs a fully
    initialized model + batch dict).
    """
    distill = compute_distill_loss(
        student_preds,
        teacher_preds,
        temperature=temperature,
        student_cat_idx=student_cat_idx,
        teacher_cat_idx=teacher_cat_idx,
        reg_max=reg_max,
    )
    total = sup_loss + alpha * distill
    return total, sup_loss, distill


# ----------------------------------------------------------------------------
# Loss patching — wraps student model.loss(batch) with the distill term
# ----------------------------------------------------------------------------


def _extract_detect_features(out: Any) -> list[torch.Tensor]:
    """Normalize a Detect head output to the list of training-mode features.

    ``Detect.forward`` returns either:
      * a plain ``list[Tensor]`` of shape (B, 4*reg_max + nc, H, W) — training mode
      * a ``(y, x)`` tuple — eval / inference mode, where ``x`` is the same
        list above and ``y`` is the post-decoded inference output.

    We always want the list (so we can slice the cls block by channel index).
    """
    if isinstance(out, list):
        return out
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], list):
        return out[1]
    raise ValueError(f"unexpected Detect head output type: {type(out)}")


def patch_student_loss_for_distill(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    alpha: float = DEFAULT_ALPHA,
    temperature: float = DEFAULT_TEMPERATURE,
    student_cat_idx: int = 0,
    teacher_cat_idx: int = COCO_CAT_CLASS,
    reg_max: int = DEFAULT_REG_MAX,
) -> torch.nn.Module:
    """Mutate ``student_model`` so its ``loss(batch)`` includes distillation.

    Side effects:
      - teacher_model.eval() and parameters frozen (requires_grad=False)
      - student_model.loss replaced with a closure that:
          1. Runs ONE student forward, captures the raw Detect feature maps
          2. Runs teacher forward inside torch.no_grad()
          3. Computes sup_loss via the original criterion using the SAME
             student preds (no double-forward)
          4. Adds alpha * distill_loss to the supervised total
          5. Returns ``(total, loss_items)`` matching ultralytics' contract

    Distill loss is recorded on ``student_model._last_distill`` for inspection
    but not added to loss_items (its length must stay 3 for label_loss_items).
    """
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    original_loss = student_model.loss  # bound method, captured before patch

    def distill_loss(batch: dict, preds: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        img = batch["img"]
        # Run student forward once and reuse for both sup loss + distill.
        s_out = preds if preds is not None else student_model.forward(img)
        s_features = _extract_detect_features(s_out)
        sup_total, sup_items = original_loss(batch, preds=s_features)

        with torch.no_grad():
            t_out = teacher_model(img)
        t_features = _extract_detect_features(t_out)

        distill = compute_distill_loss(
            s_features,
            t_features,
            temperature=temperature,
            student_cat_idx=student_cat_idx,
            teacher_cat_idx=teacher_cat_idx,
            reg_max=reg_max,
        )
        student_model._last_distill = distill.detach().clone()
        return sup_total + alpha * distill, sup_items

    student_model.loss = distill_loss  # type: ignore[assignment]
    student_model._distill_alpha = alpha
    student_model._distill_temperature = temperature
    return student_model


# ----------------------------------------------------------------------------
# Train + export pipeline
# ----------------------------------------------------------------------------


def train_distilled(
    teacher_pt: Path = DEFAULT_TEACHER_PT,
    student_pt: Path = DEFAULT_STUDENT_PT,
    data_yaml: Path = DEFAULT_DATA_YAML,
    epochs: int = 50,
    imgsz: int = 224,
    batch: int = 32,
    device: str = "auto",
    alpha: float = DEFAULT_ALPHA,
    temperature: float = DEFAULT_TEMPERATURE,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    name: str = DEFAULT_NAME,
    yolo_factory: Any = None,
    patch_fn: Callable[..., torch.nn.Module] | None = None,
) -> Path:
    """Run ultralytics fine-tune with KL distillation patched into the loss.

    Returns the path to the produced best.pt (or last.pt if best is missing).
    Heavy deps are DI-overridable for tests:
      * ``yolo_factory`` — defaults to ``ultralytics.YOLO``
      * ``patch_fn`` — defaults to ``patch_student_loss_for_distill``
    """
    if yolo_factory is None:
        from ultralytics import YOLO  # lazy

        yolo_factory = YOLO
    if patch_fn is None:
        patch_fn = patch_student_loss_for_distill

    runs_dir.mkdir(parents=True, exist_ok=True)
    teacher = yolo_factory(str(teacher_pt))
    student = yolo_factory(str(student_pt))

    # Move teacher to the student's eventual device once training starts. The
    # trainer moves the student model lazily; we mirror by deferring teacher
    # placement to a callback so they end up co-resident.
    def _on_train_start(trainer):
        device_ = next(trainer.model.parameters()).device
        teacher.model.to(device_)
        patch_fn(
            trainer.model,
            teacher.model,
            alpha=alpha,
            temperature=temperature,
        )

    student.add_callback("on_train_start", _on_train_start)

    student.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
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


def copy_to_canonical(src: Path, dest: Path = DEFAULT_FP32_OUT) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def quantize_distilled(
    fp32_pt: Path = DEFAULT_FP32_OUT,
    out_tflite: Path = DEFAULT_INT8_OUT,
    calib_dir: Path = DEFAULT_CALIB_DIR,
    imgsz: int = 224,
    max_calib_frames: int = 200,
    work_dir: Path | None = None,
    export_fn: Callable[..., Path] | None = None,
) -> Path:
    """Re-use the US-004 PTQ pipeline so the quantization recipe is identical."""
    if export_fn is None:
        from training.edge.yolo.export_quantize import export_and_quantize

        export_fn = export_and_quantize
    return export_fn(
        pt_path=fp32_pt,
        calib_dir=calib_dir,
        out_path=out_tflite,
        imgsz=imgsz,
        max_calib_frames=max_calib_frames,
        work_dir=work_dir,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="distill_train")
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER_PT)
    ap.add_argument("--student", type=Path, default=DEFAULT_STUDENT_PT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--fp32-out", type=Path, default=DEFAULT_FP32_OUT)
    ap.add_argument("--int8-out", type=Path, default=DEFAULT_INT8_OUT)
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    ap.add_argument("--max-calib-frames", type=int, default=200)
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="skip the train step and just quantize an existing fp32-out (re-runs)",
    )
    ap.add_argument(
        "--skip-quantize",
        action="store_true",
        help="skip the int8 quantize step (e.g. when running eval-only)",
    )
    args = ap.parse_args(argv)

    if not args.skip_train:
        best = train_distilled(
            teacher_pt=args.teacher,
            student_pt=args.student,
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            alpha=args.alpha,
            temperature=args.temperature,
            runs_dir=args.runs_dir,
            name=args.name,
        )
        copy_to_canonical(best, args.fp32_out)
        print(f"distilled best.pt={best} -> {args.fp32_out}")

    if not args.skip_quantize:
        quantize_distilled(
            fp32_pt=args.fp32_out,
            out_tflite=args.int8_out,
            calib_dir=args.calib_dir,
            imgsz=args.imgsz,
            max_calib_frames=args.max_calib_frames,
        )
        size = args.int8_out.stat().st_size if args.int8_out.exists() else 0
        print(f"int8 tflite -> {args.int8_out} ({size} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
