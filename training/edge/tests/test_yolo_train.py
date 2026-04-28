"""Tests for training/edge/yolo/train_yolov8n_cat.py — mirrors test_detector.py.

The ultralytics YOLO class is mocked end-to-end so the suite has no heavy
ML dependencies and no actual training happens.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from training.edge.yolo import train_yolov8n_cat as t


def _seed_dataset(dataset_dir: Path, n_train: int = 4, n_val: int = 2) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train: list[dict] = []
    val: list[dict] = []
    for i in range(n_train):
        stem = f"train_{i:03d}"
        (dataset_dir / f"{stem}.jpg").write_bytes(b"fakejpeg")
        (dataset_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        train.append({"stem": stem, "image": f"{stem}.jpg", "label": f"{stem}.txt", "sha256": "x"})
    for i in range(n_val):
        stem = f"val_{i:03d}"
        (dataset_dir / f"{stem}.jpg").write_bytes(b"fakejpeg")
        (dataset_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
        val.append({"stem": stem, "image": f"{stem}.jpg", "label": f"{stem}.txt", "sha256": "x"})
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


def test_prepare_data_writes_yaml_and_split_files(tmp_path: Path) -> None:
    """data.yaml is single-class with names=['cat']; train/val.txt list image paths."""
    dataset = _seed_dataset(tmp_path / "labeled", n_train=3, n_val=2)
    yaml_path = t.prepare_data(dataset)

    assert yaml_path == dataset / "data.yaml"
    cfg = yaml.safe_load(yaml_path.read_text())
    assert cfg["nc"] == 1
    assert cfg["names"] == ["cat"]

    train_txt = (dataset / "train.txt").read_text().strip().splitlines()
    val_txt = (dataset / "val.txt").read_text().strip().splitlines()
    assert len(train_txt) == 3
    assert len(val_txt) == 2
    for line in train_txt + val_txt:
        assert line.endswith(".jpg")
        assert Path(line).exists()


def test_prepare_data_emits_eval_val_manifest(tmp_path: Path) -> None:
    """An eval-harness-compatible val/manifest.txt is generated next to the data."""
    dataset = _seed_dataset(tmp_path / "labeled", n_train=2, n_val=2)
    t.prepare_data(dataset)

    eval_manifest = dataset / "val" / "manifest.txt"
    assert eval_manifest.exists()
    lines = eval_manifest.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        img_rel, lbl_rel = line.split("\t")
        # Eval harness resolves these relative to the manifest's directory
        # (training/edge/data/labeled/val/), so `../` must point back at the
        # actual jpg/txt pair in training/edge/data/labeled/.
        assert img_rel.startswith("../")
        assert lbl_rel.startswith("../")
        assert (eval_manifest.parent / img_rel).resolve().exists()
        assert (eval_manifest.parent / lbl_rel).resolve().exists()


def test_prepare_data_missing_manifest_raises(tmp_path: Path) -> None:
    """If US-002 hasn't run yet, prepare_data should fail loudly."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        t.prepare_data(empty)


def test_train_passes_single_cls_and_imgsz_224(tmp_path: Path) -> None:
    """Smoke test: ultralytics.YOLO(yolov8n.pt).train(...) is called with
    single_cls=True AND imgsz=224. Both are required by US-003."""
    dataset = _seed_dataset(tmp_path / "labeled")
    data_yaml = t.prepare_data(dataset)

    fake_model = MagicMock()
    fake_factory = MagicMock(return_value=fake_model)

    runs_dir = tmp_path / "runs"
    weights_dir = runs_dir / "yolov8n_cat" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "best.pt").write_bytes(b"fake-checkpoint")

    best = t.train(
        base=Path("yolov8n.pt"),
        data_yaml=data_yaml,
        epochs=1,
        imgsz=224,
        batch=32,
        device="cpu",
        runs_dir=runs_dir,
        yolo_factory=fake_factory,
    )

    fake_factory.assert_called_once_with("yolov8n.pt")
    fake_model.train.assert_called_once()
    kwargs = fake_model.train.call_args.kwargs
    assert kwargs["single_cls"] is True
    assert kwargs["imgsz"] == 224
    assert kwargs["data"] == str(data_yaml)
    assert kwargs["batch"] == 32
    assert kwargs["device"] == "cpu"
    assert best == weights_dir / "best.pt"


def test_train_falls_back_to_last_when_best_missing(tmp_path: Path) -> None:
    """Some ultralytics runs only emit last.pt when training is short."""
    dataset = _seed_dataset(tmp_path / "labeled")
    data_yaml = t.prepare_data(dataset)
    runs_dir = tmp_path / "runs"
    weights_dir = runs_dir / "yolov8n_cat" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "last.pt").write_bytes(b"fake")

    fake_factory = MagicMock(return_value=MagicMock())
    best = t.train(
        base=Path("yolov8n.pt"),
        data_yaml=data_yaml,
        runs_dir=runs_dir,
        epochs=1,
        yolo_factory=fake_factory,
    )
    assert best == weights_dir / "last.pt"


def test_copy_to_canonical_creates_parent_and_copies(tmp_path: Path) -> None:
    src = tmp_path / "best.pt"
    src.write_bytes(b"weights")
    dest = tmp_path / "models" / "yolov8n_cat.pt"
    out = t.copy_to_canonical(src, dest)
    assert out == dest
    assert dest.read_bytes() == b"weights"
