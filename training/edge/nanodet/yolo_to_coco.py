"""Convert the US-002 YOLO-format dataset into COCO instances JSON.

NanoDet-Plus' upstream training loop expects a COCO-format dataset (per
``data.train.name: CocoDataset`` in the YAMLs); the catzap auto-labeled set
under ``training/edge/data/labeled/`` is YOLO-format (one ``<stem>.txt`` per
image, with ``<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>`` lines).

This module emits two JSON files matching COCO ``instances_*.json`` shape
(``images`` + ``annotations`` + ``categories``) using the train / val split
recorded in ``manifest.json``. The image dimensions are read from the JPEGs
themselves (Pillow), so every annotation's pixel-space ``bbox`` is accurate
regardless of the original capture resolution.

Output (default):
    training/edge/data/labeled/coco/instances_train.json
    training/edge/data/labeled/coco/instances_val.json

Categories: a single class ``cat`` with ``id=0`` (mirrors the auto-labeler's
class remap, US-002).

Usage:
    python -m training.edge.nanodet.yolo_to_coco \\
        --dataset-dir training/edge/data/labeled \\
        --out-dir training/edge/data/labeled/coco

Run this before ``train_nanodet_cat.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DATASET_DIR = Path("training/edge/data/labeled")
DEFAULT_OUT_DIR = DEFAULT_DATASET_DIR / "coco"
DEFAULT_CATEGORY_ID = 0
DEFAULT_CATEGORY_NAME = "cat"


def _read_image_size(jpg_path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` for ``jpg_path``."""
    from PIL import Image  # lazy import — keeps test paths import-safe

    with Image.open(jpg_path) as im:
        return int(im.width), int(im.height)


def _yolo_lines_to_pixel_boxes(
    label_path: Path, width: int, height: int
) -> list[tuple[int, list[float], float]]:
    """Parse a YOLO label file -> ``[(class_id, [x, y, w, h], area), ...]``.

    Output bboxes are in COCO ``[x_min, y_min, w, h]`` pixel format. Lines
    that do not parse to five floats are skipped silently (matches the
    auto-labeler's tolerance for trailing whitespace).
    """
    out: list[tuple[int, list[float], float]] = []
    if not label_path.exists():
        return out
    for raw in label_path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx = float(parts[1]) * width
            cy = float(parts[2]) * height
            w = float(parts[3]) * width
            h = float(parts[4]) * height
        except ValueError:
            continue
        x = max(0.0, cx - w / 2.0)
        y = max(0.0, cy - h / 2.0)
        # Clamp to image bounds — ultralytics-style augmentations don't apply
        # at val time and our labels are already in [0, 1], but defensive.
        w = max(0.0, min(w, float(width) - x))
        h = max(0.0, min(h, float(height) - y))
        out.append((cls, [x, y, w, h], w * h))
    return out


def build_coco_split(
    entries: Iterable[dict[str, Any]],
    dataset_dir: Path,
    category_id: int = DEFAULT_CATEGORY_ID,
    category_name: str = DEFAULT_CATEGORY_NAME,
    image_size_reader: Any = _read_image_size,
) -> dict[str, Any]:
    """Build a COCO ``instances_*.json``-shaped dict from manifest entries.

    ``entries`` are the manifest's ``train`` / ``val`` rows. Annotation IDs
    start at 1 (COCO requires positive IDs). Image IDs use the manifest stem
    when it parses as an int, otherwise sequentially from 1.

    ``image_size_reader`` is injected for testability — pass a stub that
    returns ``(w, h)`` to skip the Pillow open in tests.
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_ann_id = 1

    for image_idx, entry in enumerate(entries, start=1):
        image_name = entry["image"]
        label_name = entry["label"]
        jpg_path = dataset_dir / image_name
        label_path = dataset_dir / label_name

        try:
            width, height = image_size_reader(jpg_path)
        except FileNotFoundError:
            continue

        image_id = image_idx
        images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": int(width),
                "height": int(height),
            }
        )

        for _src_cls, bbox, area in _yolo_lines_to_pixel_boxes(label_path, width, height):
            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": [float(v) for v in bbox],
                    "area": float(area),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            next_ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": int(category_id), "name": category_name, "supercategory": "animal"}
        ],
    }


def convert(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    image_size_reader: Any = _read_image_size,
) -> tuple[Path, Path]:
    """Read manifest.json, write instances_train.json + instances_val.json."""
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing US-002 manifest at {manifest_path}; "
            "run `python -m training.edge.make_dataset_manifest` first"
        )
    manifest = json.loads(manifest_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    train_coco = build_coco_split(
        manifest["train"], dataset_dir, image_size_reader=image_size_reader
    )
    val_coco = build_coco_split(
        manifest["val"], dataset_dir, image_size_reader=image_size_reader
    )
    train_path = out_dir / "instances_train.json"
    val_path = out_dir / "instances_val.json"
    train_path.write_text(json.dumps(train_coco))
    val_path.write_text(json.dumps(val_coco))
    return train_path, val_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="yolo_to_coco")
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)
    train_path, val_path = convert(args.dataset_dir, args.out_dir)
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
