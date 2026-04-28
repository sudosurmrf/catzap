"""Write training/edge/data/labeled/manifest.json for the labeled dataset.

The manifest records:
  - count: total image+label pairs
  - train / val: file lists for an 80/20 deterministic split (sorted by stem)
  - sha256 per image (so future stories can detect dataset drift)

CLI:
    python -m training.edge.make_dataset_manifest \\
        --dataset-dir training/edge/data/labeled \\
        --val-fraction 0.2

The split is deterministic: files are sorted by stem and every 5th file
goes to val. This avoids needing a global random seed and keeps the
split stable across runs as long as filenames are stable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def build_manifest(dataset_dir: Path, val_fraction: float = 0.2) -> dict:
    """Walk dataset_dir for <stem>.jpg + <stem>.txt pairs and build a manifest.

    Pairs missing either side are skipped and counted under
    `unpaired_files`. The 80/20 split is computed on stem-sorted order:
    1 in every `1/val_fraction` pairs is val.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")

    images = sorted(p for p in dataset_dir.glob("*.jpg"))
    labels = sorted(p for p in dataset_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    paired_stems = sorted(image_stems & label_stems)
    unpaired = sorted(image_stems ^ label_stems)

    if not paired_stems:
        raise RuntimeError(
            f"no <stem>.jpg + <stem>.txt pairs found in {dataset_dir}"
        )

    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    val_every = max(2, round(1.0 / val_fraction))

    train: list[dict] = []
    val: list[dict] = []
    for i, stem in enumerate(paired_stems):
        img = dataset_dir / f"{stem}.jpg"
        lbl = dataset_dir / f"{stem}.txt"
        entry = {
            "stem": stem,
            "image": img.name,
            "label": lbl.name,
            "sha256": _sha256(img),
        }
        if i % val_every == 0:
            val.append(entry)
        else:
            train.append(entry)

    return {
        "dataset_dir": str(dataset_dir),
        "count": len(paired_stems),
        "train_count": len(train),
        "val_count": len(val),
        "val_fraction": val_fraction,
        "unpaired_files": unpaired,
        "train": train,
        "val": val,
    }


def write_manifest(dataset_dir: Path, val_fraction: float = 0.2) -> Path:
    manifest = build_manifest(dataset_dir, val_fraction)
    out = dataset_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="make_dataset_manifest")
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("training/edge/data/labeled"),
    )
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args(argv)

    out = write_manifest(args.dataset_dir, args.val_fraction)
    summary = json.loads(out.read_text())
    print(
        f"wrote {out} (count={summary['count']}, "
        f"train={summary['train_count']}, val={summary['val_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
