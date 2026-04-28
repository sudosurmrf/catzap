"""Tests for training/edge/nanodet/train_nanodet_cat.py and yolo_to_coco.py.

Mirrors training/edge/tests/test_yolo_train.py: pytest + unittest.mock.MagicMock,
no real model weights, no nanodet/lightning import. The trainer factory is
swapped out via DI so US-008's smoke test verifies the config-overrides path
without actually training.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from training.edge.nanodet import train_nanodet_cat as t
from training.edge.nanodet import yolo_to_coco as y


# ---- yolo_to_coco --------------------------------------------------------- #


def _seed_yolo_dataset(dataset_dir: Path, n_train: int = 3, n_val: int = 2) -> Path:
    """Lay down a minimal US-002-style dataset (jpg sentinels + yolo labels)."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train: list[dict] = []
    val: list[dict] = []
    for i in range(n_train):
        stem = f"train_{i:03d}"
        (dataset_dir / f"{stem}.jpg").write_bytes(b"fakejpeg")
        (dataset_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        train.append({"stem": stem, "image": f"{stem}.jpg", "label": f"{stem}.txt"})
    for i in range(n_val):
        stem = f"val_{i:03d}"
        (dataset_dir / f"{stem}.jpg").write_bytes(b"fakejpeg")
        (dataset_dir / f"{stem}.txt").write_text("0 0.6 0.6 0.3 0.3\n0 0.2 0.2 0.1 0.1\n")
        val.append({"stem": stem, "image": f"{stem}.jpg", "label": f"{stem}.txt"})
    manifest = {
        "dataset_dir": str(dataset_dir),
        "count": n_train + n_val,
        "train_count": n_train,
        "val_count": n_val,
        "val_fraction": n_val / (n_train + n_val),
        "unpaired_files": [],
        "train": train,
        "val": val,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))
    return dataset_dir


def test_yolo_to_coco_emits_single_class_with_pixel_bboxes(tmp_path: Path) -> None:
    """Train + val COCO files have category cat (id 0) and pixel bboxes."""
    dataset = _seed_yolo_dataset(tmp_path / "labeled", n_train=2, n_val=1)
    out_dir = tmp_path / "labeled" / "coco"

    fake_size = MagicMock(return_value=(320, 240))
    train_path, val_path = y.convert(
        dataset_dir=dataset, out_dir=out_dir, image_size_reader=fake_size
    )

    train_coco = json.loads(train_path.read_text())
    val_coco = json.loads(val_path.read_text())
    assert train_coco["categories"] == [
        {"id": 0, "name": "cat", "supercategory": "animal"}
    ]
    assert val_coco["categories"] == [
        {"id": 0, "name": "cat", "supercategory": "animal"}
    ]
    assert len(train_coco["images"]) == 2
    assert len(val_coco["images"]) == 1
    # 0.5 cx 0.5 cy 0.4 w 0.4 h on 320x240 -> [96, 72, 128, 96]
    train_ann = train_coco["annotations"][0]
    assert train_ann["category_id"] == 0
    assert train_ann["iscrowd"] == 0
    assert train_ann["bbox"] == [96.0, 72.0, 128.0, 96.0]
    # Two GT boxes per val image -> two annotations per image.
    assert len(val_coco["annotations"]) == 2
    assert {a["image_id"] for a in val_coco["annotations"]} == {1}


def test_yolo_to_coco_skips_missing_images(tmp_path: Path) -> None:
    """A missing JPEG should not crash conversion — image is silently dropped."""
    dataset = _seed_yolo_dataset(tmp_path / "labeled", n_train=1, n_val=1)
    (dataset / "val_000.jpg").unlink()  # simulate missing image

    def reader(p: Path) -> tuple[int, int]:
        if not p.exists():
            raise FileNotFoundError(p)
        return 320, 240

    train_path, val_path = y.convert(
        dataset_dir=dataset, out_dir=tmp_path / "out", image_size_reader=reader
    )
    val_coco = json.loads(val_path.read_text())
    assert len(val_coco["images"]) == 0
    assert len(val_coco["annotations"]) == 0
    train_coco = json.loads(train_path.read_text())
    assert len(train_coco["images"]) == 1


# ---- train_nanodet_cat: config overrides --------------------------------- #


def test_regenerate_config_writes_num_classes_one_and_input_size_224(
    tmp_path: Path,
) -> None:
    """Smoke test: --num-classes=1 and --input-size=224 land in the YAML.

    Both are required by US-008.
    """
    cfg_path = tmp_path / "cat.yml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "save_dir": "x",
                "model": {
                    "arch": {
                        "head": {"num_classes": 80},
                        "aux_head": {"num_classes": 80},
                    }
                },
                "data": {
                    "train": {"input_size": [416, 416], "ann_path": "a", "img_path": "b"},
                    "val": {"input_size": [416, 416], "ann_path": "c", "img_path": "d"},
                },
                "schedule": {
                    "optimizer": {"lr": 0.001},
                    "total_epochs": 300,
                    "lr_schedule": {"T_max": 300},
                },
                "device": {"batchsize_per_gpu": 96},
                "class_names": ["foo"] * 80,
            }
        )
    )

    cfg = t.regenerate_config(
        config_path=cfg_path,
        num_classes=1,
        input_size=224,
        epochs=50,
        lr=1e-4,
    )

    assert cfg["model"]["arch"]["head"]["num_classes"] == 1
    assert cfg["model"]["arch"]["aux_head"]["num_classes"] == 1
    assert cfg["data"]["train"]["input_size"] == [224, 224]
    assert cfg["data"]["val"]["input_size"] == [224, 224]
    assert cfg["schedule"]["optimizer"]["lr"] == 1e-4
    assert cfg["schedule"]["total_epochs"] == 50
    assert cfg["schedule"]["lr_schedule"]["T_max"] == 50
    assert cfg["class_names"] == ["cat"]

    # Round-tripped on disk too.
    rt = yaml.safe_load(cfg_path.read_text())
    assert rt["model"]["arch"]["head"]["num_classes"] == 1
    assert rt["data"]["train"]["input_size"] == [224, 224]


def test_train_passes_num_classes_and_input_size_to_factory(
    tmp_path: Path,
) -> None:
    """End-to-end: train(...) regenerates the config, calls trainer_factory
    with that config containing num_classes=1 and input_size=224."""
    cfg_path = tmp_path / "cat.yml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "save_dir": "x",
                "model": {
                    "arch": {
                        "head": {"num_classes": 80},
                        "aux_head": {"num_classes": 80},
                    }
                },
                "data": {
                    "train": {"input_size": [416, 416], "ann_path": "a", "img_path": "b"},
                    "val": {"input_size": [416, 416], "ann_path": "c", "img_path": "d"},
                },
                "schedule": {
                    "optimizer": {"lr": 0.001},
                    "total_epochs": 300,
                    "lr_schedule": {"T_max": 300},
                },
                "device": {"batchsize_per_gpu": 96},
                "class_names": ["foo"] * 80,
            }
        )
    )

    fake_best = tmp_path / "fake_best.ckpt"
    fake_best.write_bytes(b"weights")
    fake_factory = MagicMock(return_value=fake_best)

    runs_dir = tmp_path / "runs"
    out = t.train(
        config_path=cfg_path,
        epochs=50,
        num_classes=1,
        input_size=224,
        lr=1e-4,
        batch_size=32,
        runs_dir=runs_dir,
        skip_data_prep=True,
        trainer_factory=fake_factory,
    )

    assert out == fake_best
    fake_factory.assert_called_once()
    passed_path, passed_cfg = fake_factory.call_args.args
    assert passed_path == cfg_path
    assert passed_cfg["model"]["arch"]["head"]["num_classes"] == 1
    assert passed_cfg["data"]["train"]["input_size"] == [224, 224]
    assert passed_cfg["data"]["val"]["input_size"] == [224, 224]
    assert passed_cfg["schedule"]["optimizer"]["lr"] == 1e-4
    assert passed_cfg["schedule"]["total_epochs"] == 50


def test_copy_to_canonical_creates_parent_and_copies(tmp_path: Path) -> None:
    src = tmp_path / "best.ckpt"
    src.write_bytes(b"weights")
    dest = tmp_path / "checkpoints" / "nanodet_cat_0.5x_224.pth"
    out = t.copy_to_canonical(src, dest)
    assert out == dest
    assert dest.read_bytes() == b"weights"


def test_default_checkpoint_path_is_canonical_us008_path() -> None:
    """The PRD pins the canonical filename — guard against silent renames."""
    assert t.DEFAULT_CHECKPOINT_OUT == Path(
        "training/edge/nanodet/checkpoints/nanodet_cat_0.5x_224.pth"
    )


# ---- shipped config sanity check ----------------------------------------- #


def test_shipped_config_has_num_classes_one_and_input_size_224() -> None:
    """The hand-checked-in cat config must satisfy the PRD invariants."""
    cfg = yaml.safe_load(t.DEFAULT_CONFIG.read_text())
    assert cfg["model"]["arch"]["head"]["num_classes"] == 1
    assert cfg["model"]["arch"]["aux_head"]["num_classes"] == 1
    assert cfg["data"]["train"]["input_size"] == [224, 224]
    assert cfg["data"]["val"]["input_size"] == [224, 224]
    assert cfg["schedule"]["total_epochs"] == 50
    assert cfg["schedule"]["optimizer"]["lr"] == 1e-4
    assert cfg["class_names"] == ["cat"]
    assert cfg["model"]["arch"]["backbone"]["model_size"] == "0.5x"
