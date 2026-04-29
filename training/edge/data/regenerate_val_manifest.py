"""Regenerate ``training/edge/data/labeled/val/manifest.txt`` from the current
80/20 split in ``training/edge/data/labeled/manifest.json``.

Why this exists
---------------
The v1 auto-labeler bootstrap (``training/edge/auto_label.py:bootstrap``) writes
files under ``uuid4().hex.jpg`` + ``.txt`` filenames. The progress.txt
"deterministic seed=42" claim covers the IMAGE CONTENT (augmentation,
brightness, etc.) but NOT the filenames — uuid4() is not seeded. As a
consequence, ``val/manifest.txt`` (which maps val images to label files by
filename) goes stale every time the bootstrap is re-run on a fresh worktree.

This helper rebuilds ``val/manifest.txt`` from the current paired files in
``training/edge/data/labeled/`` so the eval harness can locate the val set
without manual editing. Run it after ``auto_label bootstrap`` +
``make_dataset_manifest``.

Usage
-----
    python training/edge/data/regenerate_val_manifest.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATASET_DIR = Path("training/edge/data/labeled")


def regenerate(dataset_dir: Path = DATASET_DIR) -> Path:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"manifest.json not found at {manifest_path}; run "
            "`python -m training.edge.make_dataset_manifest` first"
        )
    manifest = json.loads(manifest_path.read_text())
    val_pairs = manifest["val"]

    val_dir = dataset_dir / "val"
    val_dir.mkdir(exist_ok=True)
    out = val_dir / "manifest.txt"
    # Tab-separated <image>\t<label>, paths relative to manifest.txt's dir.
    lines = [f"../{e['image']}\t../{e['label']}" for e in val_pairs]
    out.write_text("\n".join(lines) + "\n")

    # Also rebuild train.txt / val.txt for ultralytics consumers.
    abs_dir = dataset_dir.resolve()
    (dataset_dir / "train.txt").write_text(
        "\n".join(str(abs_dir / e["image"]) for e in manifest["train"]) + "\n"
    )
    (dataset_dir / "val.txt").write_text(
        "\n".join(str(abs_dir / e["image"]) for e in manifest["val"]) + "\n"
    )
    return out


def main() -> int:
    out = regenerate()
    print(f"wrote {out} from manifest.json's val split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
