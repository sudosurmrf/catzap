"""Tests for training/edge/yolo/distill_train.py.

Mirrors the DI pattern in test_quantize.py / test_qat_decision.py: the
ultralytics + TF heavy deps are avoided entirely. Distillation math is
exercised on tiny tensors; the trainer integration is covered through
patch_fn / yolo_factory mocks.

Acceptance focus (US-006):
    "at least one test that mocks teacher/student and asserts the combined
     loss is computed (teacher gradient is detached, student gradient flows)."

That contract is the first test below; the rest cover edge cases and the
public API surface.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from training.edge.yolo import distill_train as dt


# --- core: combined loss with detached teacher + flowing student gradient --


def _make_feature_maps(
    nc_student: int,
    nc_teacher: int,
    *,
    student_grad: bool,
    teacher_grad: bool,
    reg_max: int = 16,
    scales: tuple[tuple[int, int], ...] = ((28, 28), (14, 14), (7, 7)),
    batch: int = 2,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build (student_preds, teacher_preds) shaped like ultralytics Detect outputs.

    Each scale has shape (B, 4*reg_max + nc, H, W). Student requires_grad lets
    us check that backward populates gradients on the leaf tensor; teacher
    requires_grad=True (initially) lets us prove the .detach() inside
    compute_distill_loss really does block the gradient path.
    """
    s_no = 4 * reg_max + nc_student
    t_no = 4 * reg_max + nc_teacher
    s_list = [
        torch.randn(batch, s_no, h, w, requires_grad=student_grad)
        for (h, w) in scales
    ]
    t_list = [
        torch.randn(batch, t_no, h, w, requires_grad=teacher_grad)
        for (h, w) in scales
    ]
    return s_list, t_list


def test_combined_loss_total_equals_sup_plus_alpha_distill() -> None:
    """``total = sup + alpha * distill`` — the canonical KD recipe.

    Asserting the exact equality catches sign / weight bugs that would still
    let the model train but track the wrong objective.
    """
    s_preds, t_preds = _make_feature_maps(
        nc_student=1,
        nc_teacher=80,
        student_grad=True,
        teacher_grad=False,
    )
    sup_loss = torch.tensor(2.5, requires_grad=True)
    alpha = 0.5

    total, sup, distill = dt.combined_loss(
        s_preds,
        t_preds,
        sup_loss,
        alpha=alpha,
        temperature=4.0,
    )

    assert torch.isclose(total, sup + alpha * distill)
    assert sup.item() == pytest.approx(2.5)


def test_combined_loss_teacher_gradient_is_detached() -> None:
    """Backward through the combined loss must NOT update teacher tensors.

    The .detach() on the teacher slice in compute_distill_loss is the
    structural barrier here. We give the teacher tensors requires_grad=True
    on purpose and assert their .grad stays None after backward.
    """
    s_preds, t_preds = _make_feature_maps(
        nc_student=1,
        nc_teacher=80,
        student_grad=True,
        teacher_grad=True,  # would propagate if .detach() were missing
    )
    sup_loss = torch.tensor(0.0, requires_grad=True)

    total, _, _ = dt.combined_loss(s_preds, t_preds, sup_loss, alpha=0.5)
    total.backward()

    for tp in t_preds:
        assert tp.grad is None, "teacher gradient leaked through distill loss"


def test_combined_loss_student_gradient_flows() -> None:
    """Backward must populate gradients on every student feature-map scale.

    If a scale's gradient is None or all-zero, the distill term isn't actually
    affecting that head — i.e., the patch is a no-op. Both failure modes get
    caught here.
    """
    s_preds, t_preds = _make_feature_maps(
        nc_student=1,
        nc_teacher=80,
        student_grad=True,
        teacher_grad=False,
    )
    # Use sup=0 so the entire gradient signal must come from distill — this
    # makes the test sensitive to the distill term specifically.
    sup_loss = torch.zeros((), requires_grad=True)

    total, _, distill = dt.combined_loss(s_preds, t_preds, sup_loss, alpha=1.0)
    assert distill.requires_grad
    total.backward()

    for sp in s_preds:
        assert sp.grad is not None, "student gradient missing from a scale"
        assert sp.grad.abs().sum() > 0, "student gradient is identically zero"


# --- compute_distill_loss edge cases ---------------------------------------


def test_compute_distill_loss_scale_count_mismatch_raises() -> None:
    s = [torch.randn(1, 65, 7, 7)]
    t = [torch.randn(1, 144, 7, 7), torch.randn(1, 144, 14, 14)]
    with pytest.raises(ValueError, match="scale counts"):
        dt.compute_distill_loss(s, t)


def test_compute_distill_loss_scale_shape_mismatch_raises() -> None:
    s = [torch.randn(1, 65, 7, 7)]
    t = [torch.randn(1, 144, 14, 14)]
    with pytest.raises(ValueError, match="shape mismatch"):
        dt.compute_distill_loss(s, t)


def test_compute_distill_loss_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        dt.compute_distill_loss([], [])


def test_compute_distill_loss_scales_with_temperature_squared() -> None:
    """T**2 scaling: doubling T at fixed logits multiplies loss by ~4.

    Hinton's KD recipe scales BCE/CE by T**2 so the gradient magnitude is
    invariant to T near zero logits. Verifying the ratio catches T**1 bugs.
    """
    torch.manual_seed(0)
    s_preds, t_preds = _make_feature_maps(
        nc_student=1,
        nc_teacher=80,
        student_grad=False,
        teacher_grad=False,
        scales=((7, 7),),
        batch=4,
    )
    # Make logits non-zero so the BCE term itself is non-trivial.
    s_preds = [sp.detach() + 1.0 for sp in s_preds]
    t_preds = [tp.detach() - 0.5 for tp in t_preds]

    loss_t1 = dt.compute_distill_loss(s_preds, t_preds, temperature=1.0)
    loss_t2 = dt.compute_distill_loss(s_preds, t_preds, temperature=2.0)
    # The exact ratio depends on logit values; for our seeded data the T**2
    # scaling dominates so the ratio is in (3.0, 5.0). Loose bound on purpose
    # — the strict invariant we want is "T**2 prefactor is present".
    assert 3.0 < (loss_t2 / loss_t1) < 5.0


# --- patch_student_loss_for_distill ----------------------------------------


def _make_fake_yolo_module(
    nc: int,
    *,
    grad: bool = True,
    reg_max: int = 16,
) -> torch.nn.Module:
    """Build a minimal nn.Module shaped like an ultralytics DetectionModel.

    Forward returns the training-mode list-of-features layout. ``loss`` is
    set to a dummy that returns (sup_total, loss_items) so the patch can
    wrap it.
    """
    no = 4 * reg_max + nc

    class _Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # one trainable parameter so requires_grad bookkeeping works
            self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=grad)

        def forward(self, x):  # type: ignore[override]
            B = x.shape[0]
            scales = (28, 14, 7)
            outs = []
            for s in scales:
                # Make output depend on input + scale param so autograd has a path
                out = torch.randn(B, no, s, s) * self.scale
                outs.append(out)
            return outs

        def loss(self, batch, preds=None):
            if preds is None:
                preds = self.forward(batch["img"])
            # Simple sup loss: mean-square of all preds, scaled
            sup = sum(p.pow(2).mean() for p in preds) / len(preds)
            sup_items = torch.tensor([sup.item(), 0.0, 0.0])
            return sup, sup_items

    return _Fake()


def test_patch_freezes_teacher_and_sets_eval_mode() -> None:
    student = _make_fake_yolo_module(nc=1)
    teacher = _make_fake_yolo_module(nc=80)
    teacher.train()  # confirm patch flips back to eval

    dt.patch_student_loss_for_distill(student, teacher)

    assert teacher.training is False
    for p in teacher.parameters():
        assert p.requires_grad is False


def test_patched_loss_returns_total_with_distill_added() -> None:
    """After patching, ``student.loss(batch)`` returns sup + alpha*distill."""
    torch.manual_seed(0)
    student = _make_fake_yolo_module(nc=1)
    teacher = _make_fake_yolo_module(nc=80)
    dt.patch_student_loss_for_distill(student, teacher, alpha=0.5, temperature=2.0)

    batch = {"img": torch.randn(2, 3, 224, 224)}
    total, items = student.loss(batch)

    assert isinstance(total, torch.Tensor)
    assert items.numel() == 3
    # _last_distill is recorded on the student for inspection
    assert hasattr(student, "_last_distill")
    assert torch.isfinite(student._last_distill)


def test_patched_loss_uses_provided_preds_without_double_forward() -> None:
    """When the trainer passes preds, the patched loss must reuse them."""
    student = _make_fake_yolo_module(nc=1)
    teacher = _make_fake_yolo_module(nc=80)
    # Track student.forward call count
    call_count = {"n": 0}
    real_forward = student.forward

    def counting_forward(x):
        call_count["n"] += 1
        return real_forward(x)

    student.forward = counting_forward  # type: ignore[assignment]
    dt.patch_student_loss_for_distill(student, teacher)

    img = torch.randn(2, 3, 224, 224)
    s_preds = student.forward(img)
    call_count["n"] = 0  # reset after the explicit pre-forward
    student.loss({"img": img}, preds=s_preds)
    # Inside distill_loss we should NOT invoke student.forward when preds
    # was passed in — we still call teacher() once (counted on teacher, not here).
    assert call_count["n"] == 0


# --- _extract_detect_features ---------------------------------------------


def test_extract_features_handles_training_list() -> None:
    feats = [torch.zeros(1, 65, 7, 7)]
    assert dt._extract_detect_features(feats) is feats


def test_extract_features_handles_eval_tuple() -> None:
    y = torch.zeros(1, 5, 49)
    x = [torch.zeros(1, 144, 7, 7)]
    assert dt._extract_detect_features((y, x)) is x


def test_extract_features_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="unexpected Detect head output"):
        dt._extract_detect_features(torch.zeros(1, 1))


# --- train_distilled DI surface -------------------------------------------


def test_train_distilled_passes_single_cls_imgsz_alpha_to_yolo(
    tmp_path: Path,
) -> None:
    """``train_distilled`` must invoke ultralytics with single_cls=True and
    imgsz=224. Mocks the YOLO factory + patch_fn so no real training runs.
    """
    fake_teacher = MagicMock()
    fake_teacher.model = MagicMock()
    fake_student = MagicMock()
    fake_student.model = MagicMock()

    # Pre-create a pretend best.pt so train_distilled finds it
    runs_dir = tmp_path / "runs"
    pretend_best = runs_dir / "yolov8n_cat_distilled" / "weights" / "best.pt"
    pretend_best.parent.mkdir(parents=True, exist_ok=True)
    pretend_best.write_bytes(b"fake-distilled-weights")

    # YOLO factory returns teacher first, then student
    factory = MagicMock(side_effect=[fake_teacher, fake_student])
    patch_fn = MagicMock()

    out = dt.train_distilled(
        teacher_pt=tmp_path / "teacher.pt",
        student_pt=tmp_path / "student.pt",
        data_yaml=tmp_path / "data.yaml",
        epochs=1,
        imgsz=224,
        batch=8,
        alpha=0.7,
        temperature=3.0,
        runs_dir=runs_dir,
        yolo_factory=factory,
        patch_fn=patch_fn,
    )

    # YOLO was called for both teacher and student
    assert factory.call_count == 2
    factory.assert_any_call(str(tmp_path / "teacher.pt"))
    factory.assert_any_call(str(tmp_path / "student.pt"))

    # train() called with the required kwargs
    fake_student.train.assert_called_once()
    train_kwargs = fake_student.train.call_args.kwargs
    assert train_kwargs["single_cls"] is True
    assert train_kwargs["imgsz"] == 224
    assert train_kwargs["epochs"] == 1
    assert train_kwargs["batch"] == 8
    assert train_kwargs["data"] == str(tmp_path / "data.yaml")

    # The on_train_start callback was registered (it's what calls patch_fn)
    fake_student.add_callback.assert_called_once()
    cb_args = fake_student.add_callback.call_args.args
    assert cb_args[0] == "on_train_start"

    # Returned path resolves to the pretend best.pt
    assert out == pretend_best


def test_quantize_distilled_delegates_to_export_and_quantize(
    tmp_path: Path,
) -> None:
    fp32 = tmp_path / "distilled.pt"
    fp32.write_bytes(b"fake-pt")
    out = tmp_path / "distilled_int8.tflite"

    captured: dict = {}

    def fake_export(**kwargs):
        captured.update(kwargs)
        kwargs["out_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["out_path"].write_bytes(b"fake-tflite")
        return kwargs["out_path"]

    result = dt.quantize_distilled(
        fp32_pt=fp32,
        out_tflite=out,
        calib_dir=tmp_path / "calib",
        imgsz=224,
        max_calib_frames=200,
        export_fn=fake_export,
    )

    assert result == out
    assert captured["pt_path"] == fp32
    assert captured["out_path"] == out
    assert captured["imgsz"] == 224
    assert captured["max_calib_frames"] == 200
