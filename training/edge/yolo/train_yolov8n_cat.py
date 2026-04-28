"""Fine-tune YOLOv8n single-class (cat-only) on the US-002 auto-labeled set.

Usage:
    python -m training.edge.yolo.train_yolov8n_cat \\
        --base yolov8n.pt \\
        --epochs 50 \\
        --imgsz 224 \\
        --batch 32 \\
        --device auto \\
        --output training/edge/models/yolov8n_cat.pt

The training data comes from training/edge/data/labeled/manifest.json (US-002).
A data.yaml + train.txt + val.txt are generated alongside the dataset so
ultralytics can ingest them. The best.pt produced by ultralytics is copied
to the canonical path at training/edge/models/yolov8n_cat.pt.

A small training/edge/data/labeled/val/manifest.txt is also written so the
eval harness in US-001 can score this checkpoint against the same val split.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

DEFAULT_DATASET_DIR = Path("training/edge/data/labeled")
DEFAULT_BASE = Path("yolov8n.pt")
DEFAULT_OUTPUT = Path("training/edge/models/yolov8n_cat.pt")
DEFAULT_RUNS_DIR = Path("training/edge/yolo/runs")


def write_split_files(
    dataset_dir: Path,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    """Write absolute-path train.txt and val.txt for ultralytics."""
    abs_dir = dataset_dir.resolve()
    train_lines = [str(abs_dir / e["image"]) for e in manifest["train"]]
    val_lines = [str(abs_dir / e["image"]) for e in manifest["val"]]
    train_txt = dataset_dir / "train.txt"
    val_txt = dataset_dir / "val.txt"
    train_txt.write_text("\n".join(train_lines) + "\n")
    val_txt.write_text("\n".join(val_lines) + "\n")
    return train_txt, val_txt


def write_eval_val_manifest(dataset_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write training/edge/data/labeled/val/manifest.txt for the eval harness.

    The eval harness's load_val_set() reads tab-separated `<image>\\t<label>`
    pairs whose paths are resolved relative to the manifest's directory.
    """
    val_dir = dataset_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for e in manifest["val"]:
        img_rel = f"../{e['image']}"
        lbl_rel = f"../{e['label']}"
        lines.append(f"{img_rel}\t{lbl_rel}")
    manifest_path = val_dir / "manifest.txt"
    manifest_path.write_text("\n".join(lines) + "\n")
    return manifest_path


def write_data_yaml(
    dataset_dir: Path,
    train_txt: Path,
    val_txt: Path,
) -> Path:
    """Write the ultralytics data.yaml at training/edge/data/labeled/data.yaml."""
    data = {
        "path": str(dataset_dir.resolve()),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "nc": 1,
        "names": ["cat"],
    }
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return yaml_path


def prepare_data(dataset_dir: Path = DEFAULT_DATASET_DIR) -> Path:
    """Read US-002 manifest and emit data.yaml + split files + eval manifest."""
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing US-002 manifest at {manifest_path}; "
            "run `python -m training.edge.make_dataset_manifest` first"
        )
    manifest = json.loads(manifest_path.read_text())
    train_txt, val_txt = write_split_files(dataset_dir, manifest)
    write_eval_val_manifest(dataset_dir, manifest)
    return write_data_yaml(dataset_dir, train_txt, val_txt)


def train(
    base: Path = DEFAULT_BASE,
    data_yaml: Path | None = None,
    epochs: int = 50,
    imgsz: int = 224,
    batch: int = 32,
    device: str = "auto",
    runs_dir: Path = DEFAULT_RUNS_DIR,
    name: str = "yolov8n_cat",
    yolo_factory: Any = YOLO,
) -> Path:
    """Run ultralytics fine-tune; return path to best.pt.

    `device='auto'` defers to ultralytics' own auto-detection so the script
    runs on whatever the workstation has (CUDA / MPS / CPU).
    """
    if data_yaml is None:
        data_yaml = prepare_data()
    runs_dir.mkdir(parents=True, exist_ok=True)
    model = yolo_factory(str(base))
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
        verbose=True,
    )
    best = runs_dir / name / "weights" / "best.pt"
    if not best.exists():
        last = runs_dir / name / "weights" / "last.pt"
        if last.exists():
            best = last
    return best


def copy_to_canonical(src: Path, dest: Path = DEFAULT_OUTPUT) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train_yolov8n_cat")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--name", default="yolov8n_cat")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args(argv)

    data_yaml = prepare_data(args.dataset_dir)
    best = train(
        base=args.base,
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        runs_dir=args.runs_dir,
        name=args.name,
    )
    final = copy_to_canonical(best, args.output)
    print(f"best.pt={best} -> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
