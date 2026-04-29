"""Load a val set.

Two supported layouts:

(A) **Manifest layout** (default for US-001 silver GT — keeps repo small).
    <val_dir>/manifest.txt — each non-empty, non-# line is
    `<image_path>\\t<label_path>` where paths are relative to repo root.
    Labels are YOLO format (class cx cy w h, all normalized).

(B) **Self-contained YOLO layout** (used from US-002 onward).
    <val_dir>/images/<stem>.jpg
    <val_dir>/labels/<stem>.txt
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def yolo_line_to_xyxy(line: str) -> list[float]:
    parts = line.strip().split()
    if len(parts) < 5:
        return []
    cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _read_yolo_labels(label_file: Path) -> list[list[float]]:
    if not label_file.exists():
        return []
    out: list[list[float]] = []
    for line in label_file.read_text().splitlines():
        b = yolo_line_to_xyxy(line)
        if b:
            out.append(b)
    return out


def _iter_pairs_from_manifest(manifest: Path) -> Iterator[tuple[Path, Path]]:
    base = manifest.parent
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        img_p = Path(parts[0])
        lbl_p = Path(parts[1])
        if not img_p.is_absolute():
            img_p = (base / img_p).resolve()
        if not lbl_p.is_absolute():
            lbl_p = (base / lbl_p).resolve()
        yield img_p, lbl_p


def _iter_pairs_from_dir(val_dir: Path) -> Iterator[tuple[Path, Path]]:
    images_dir = val_dir / "images"
    labels_dir = val_dir / "labels"
    root = images_dir if images_dir.exists() else val_dir
    for ext in (".jpg", ".jpeg", ".png"):
        for img in sorted(root.rglob(f"*{ext}")):
            if labels_dir.exists():
                lbl = labels_dir / f"{img.stem}.txt"
            else:
                lbl = img.with_suffix(".txt")
            yield img, lbl


def load_val_set(
    val_dir: Path,
) -> tuple[list[np.ndarray], list[list[list[float]]], list[Path]]:
    manifest = val_dir / "manifest.txt"
    pairs = list(
        _iter_pairs_from_manifest(manifest)
        if manifest.exists()
        else _iter_pairs_from_dir(val_dir)
    )

    images: list[np.ndarray] = []
    gts: list[list[list[float]]] = []
    paths: list[Path] = []
    for image_path, label_path in pairs:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        images.append(frame)
        gts.append(_read_yolo_labels(label_path))
        paths.append(image_path)
    return images, gts, paths
