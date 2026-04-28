"""Fine-tune NanoDet-Plus-m 0.5x cat-only at 224x224 on the US-002 dataset.

This wraps the upstream nanodet trainer (``nanodet.trainer.task.TrainingTask``
+ pytorch-lightning ``Trainer``) so the ESP32-edge candidate has a single CLI
analogous to ``training/edge/yolo/train_yolov8n_cat.py``. It:

1. Re-emits the canonical 0.5x cat config (``configs/nanodet_plus_m_0.5x_cat.yml``)
   from a small set of CLI overrides (``--num-classes``, ``--input-size``,
   ``--epochs``, ``--lr``). The hand-checked-in YAML is the source of truth
   for hyperparameters; the regenerator produces the same shape and is what
   the tests assert against.
2. Converts the YOLO-format US-002 labels to COCO instances JSON via
   :mod:`training.edge.nanodet.yolo_to_coco`.
3. Invokes the upstream lightning ``Trainer`` (factory-injectable for tests).
4. Copies the best ``.pth`` (or ``last.ckpt`` fallback) to the canonical
   ``training/edge/nanodet/checkpoints/nanodet_cat_0.5x_224.pth``.

Usage:
    python -m training.edge.nanodet.train_nanodet_cat \\
        --config training/edge/nanodet/configs/nanodet_plus_m_0.5x_cat.yml \\
        --epochs 50 \\
        --num-classes 1 \\
        --input-size 224 \\
        --lr 1e-4

Requires the isolated venv from ``training/edge/nanodet/setup_venv.sh`` because
upstream nanodet pins ``torch<2.0``. The script is import-safe in the main env
(no top-level torch / lightning imports) so the tests can mock the trainer.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Callable

import yaml

from . import yolo_to_coco

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "nanodet_plus_m_0.5x_cat.yml"
)
DEFAULT_DATASET_DIR = Path("training/edge/data/labeled")
DEFAULT_COCO_DIR = DEFAULT_DATASET_DIR / "coco"
DEFAULT_CHECKPOINT_OUT = Path(
    "training/edge/nanodet/checkpoints/nanodet_cat_0.5x_224.pth"
)
DEFAULT_RUNS_DIR = Path("training/edge/nanodet/runs/nanodet_cat_0.5x_224")


def regenerate_config(
    config_path: Path = DEFAULT_CONFIG,
    num_classes: int = 1,
    input_size: int = 224,
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    train_ann: Path = DEFAULT_COCO_DIR / "instances_train.json",
    val_ann: Path = DEFAULT_COCO_DIR / "instances_val.json",
    img_dir: Path = DEFAULT_DATASET_DIR,
    save_dir: Path = DEFAULT_RUNS_DIR,
) -> dict[str, Any]:
    """Read the cat config, apply CLI overrides, write back, return the dict.

    The on-disk YAML is the source of truth for everything we don't expose as
    a flag (loss weights, augmentations, head channels). This function
    surfaces only the four PRD-critical knobs.
    """
    cfg = yaml.safe_load(config_path.read_text())
    cfg["model"]["arch"]["head"]["num_classes"] = int(num_classes)
    if "aux_head" in cfg["model"]["arch"]:
        cfg["model"]["arch"]["aux_head"]["num_classes"] = int(num_classes)
    cfg["data"]["train"]["input_size"] = [int(input_size), int(input_size)]
    cfg["data"]["val"]["input_size"] = [int(input_size), int(input_size)]
    cfg["data"]["train"]["ann_path"] = str(train_ann)
    cfg["data"]["val"]["ann_path"] = str(val_ann)
    cfg["data"]["train"]["img_path"] = str(img_dir)
    cfg["data"]["val"]["img_path"] = str(img_dir)
    cfg["schedule"]["optimizer"]["lr"] = float(lr)
    cfg["schedule"]["total_epochs"] = int(epochs)
    cfg["schedule"]["lr_schedule"]["T_max"] = int(epochs)
    cfg["device"]["batchsize_per_gpu"] = int(batch_size)
    cfg["save_dir"] = str(save_dir)
    if int(num_classes) == 1:
        cfg["class_names"] = ["cat"]
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def prepare_data(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    out_dir: Path = DEFAULT_COCO_DIR,
) -> tuple[Path, Path]:
    """Convert YOLO labels under ``dataset_dir`` to COCO instances JSON."""
    return yolo_to_coco.convert(dataset_dir=dataset_dir, out_dir=out_dir)


def train(
    config_path: Path = DEFAULT_CONFIG,
    epochs: int | None = None,
    num_classes: int = 1,
    input_size: int = 224,
    lr: float = 1e-4,
    batch_size: int = 32,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    skip_data_prep: bool = False,
    trainer_factory: Callable[[Path, dict[str, Any]], Path] | None = None,
) -> Path:
    """Run upstream nanodet training; return path to the best checkpoint.

    ``trainer_factory(config_path, cfg) -> best_ckpt_path`` is the seam tests
    inject into. The default factory imports nanodet/pytorch-lightning lazily
    and runs the upstream training loop end-to-end.
    """
    if not skip_data_prep:
        prepare_data()
    cfg = regenerate_config(
        config_path,
        num_classes=num_classes,
        input_size=input_size,
        epochs=epochs if epochs is not None else 50,
        lr=lr,
        batch_size=batch_size,
        save_dir=runs_dir,
    )
    runs_dir.mkdir(parents=True, exist_ok=True)
    factory = trainer_factory or _default_trainer_factory
    return factory(config_path, cfg)


def _default_trainer_factory(config_path: Path, cfg: dict[str, Any]) -> Path:
    """Run upstream nanodet ``TrainingTask`` end-to-end.

    Lazy imports so the module is import-safe in the main env (where neither
    nanodet nor pytorch-lightning are installed). When called outside the
    isolated venv this raises a clear ``ImportError`` directing the user to
    ``setup_venv.sh``.
    """
    try:
        import pytorch_lightning as pl  # type: ignore
        import torch  # type: ignore
        from nanodet.data.collate import naive_collate  # type: ignore
        from nanodet.data.dataset import build_dataset  # type: ignore
        from nanodet.evaluator import build_evaluator  # type: ignore
        from nanodet.trainer.task import TrainingTask  # type: ignore
        from nanodet.util import (  # type: ignore
            NanoDetLightningLogger,
            cfg as nano_cfg,
            load_config,
            mkdir,
        )
    except ImportError as exc:
        raise ImportError(
            "NanoDet upstream training requires the isolated venv "
            "(see training/edge/nanodet/setup_venv.sh). "
            "Install it with: cd training/edge/nanodet && ./setup_venv.sh && "
            "source .venv/bin/activate, then re-run."
        ) from exc

    load_config(nano_cfg, str(config_path))
    save_dir = Path(nano_cfg.save_dir)
    mkdir(0, str(save_dir))
    logger = NanoDetLightningLogger(str(save_dir))

    train_dataset = build_dataset(nano_cfg.data.train, "train")
    val_dataset = build_dataset(nano_cfg.data.val, "val")
    evaluator = build_evaluator(nano_cfg.evaluator, val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=nano_cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=nano_cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=nano_cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=nano_cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )

    task = TrainingTask(nano_cfg, evaluator)
    trainer = pl.Trainer(
        default_root_dir=str(save_dir),
        max_epochs=nano_cfg.schedule.total_epochs,
        gradient_clip_val=getattr(nano_cfg, "grad_clip", 35.0),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=nano_cfg.log.interval,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=nano_cfg.schedule.val_intervals,
        logger=logger,
    )
    trainer.fit(task, train_loader, val_loader)

    # Upstream task writes a 'model_best.ckpt' under save_dir; fall back to
    # last.ckpt if the run didn't pass a val_interval.
    best = save_dir / "model_best" / "model_best.ckpt"
    if not best.exists():
        best = save_dir / "model_best.ckpt"
    if not best.exists():
        last = save_dir / "last.ckpt"
        if last.exists():
            best = last
    return best


def copy_to_canonical(
    src: Path, dest: Path = DEFAULT_CHECKPOINT_OUT
) -> Path:
    """Copy the upstream best checkpoint to the canonical path the eval and
    quantization pipelines (US-009) consume."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train_nanodet_cat")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--num-classes", type=int, default=1)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT_OUT)
    ap.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip the YOLO->COCO conversion step (use when COCO json already exists)",
    )
    args = ap.parse_args(argv)

    best = train(
        config_path=args.config,
        epochs=args.epochs,
        num_classes=args.num_classes,
        input_size=args.input_size,
        lr=args.lr,
        batch_size=args.batch_size,
        runs_dir=args.runs_dir,
        skip_data_prep=args.skip_data_prep,
    )
    final = copy_to_canonical(best, args.output)
    print(f"best ckpt={best} -> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
