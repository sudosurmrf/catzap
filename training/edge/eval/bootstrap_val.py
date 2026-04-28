"""Bootstrap a 'silver' val set from data/cat_photos/.

Runs the existing CatDetector (server/vision/detector.py — read-only import)
over every JPEG under data/cat_photos/ and writes:
    training/edge/data/val/labels/<image_stem>.txt   (YOLO format, class 0)
    training/edge/data/val/manifest.txt              (pairs source-image <-> label)

Images are **NOT** duplicated under training/edge/data/val/ — the
manifest points back at the originals under data/cat_photos/. This keeps
the repo small (the labels themselves are < 100 KB).

This is silver GT — it inherits any biases or false negatives of the
yolov8s teacher. US-002 manual cleanup replaces it.

CLI:
    python -m training.edge.eval.bootstrap_val \\
        --src data/cat_photos \\
        --out training/edge/data/val \\
        --teacher yolov8s.pt \\
        --confidence 0.35
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _yolo_line(bbox: list[float]) -> str:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def bootstrap(src: Path, out: Path, teacher_path: str, confidence: float = 0.35) -> dict:
    import cv2

    from server.vision.detector import CatDetector

    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.txt"

    teacher = CatDetector(teacher_path, confidence_threshold=confidence)

    written = 0
    skipped_no_detection = 0
    total = 0
    manifest_lines: list[str] = ["# image_path\tlabel_path (both relative to this manifest's directory)"]
    out_resolved = out.resolve()
    for image in sorted(src.rglob("*.jpg")):
        total += 1
        frame = cv2.imread(str(image))
        if frame is None:
            continue
        detections = teacher.detect(frame)
        if not detections:
            skipped_no_detection += 1
            continue
        out_label = labels_dir / f"{image.stem}.txt"
        out_label.write_text(
            "\n".join(_yolo_line(d["bbox"]) for d in detections) + "\n"
        )
        import os as _os

        img_rel = _os.path.relpath(image.resolve(), out_resolved)
        lbl_rel = _os.path.relpath(out_label.resolve(), out_resolved)
        manifest_lines.append(f"{img_rel}\t{lbl_rel}")
        written += 1
    manifest.write_text("\n".join(manifest_lines) + "\n")
    return {
        "total_source_images": total,
        "labeled": written,
        "skipped_no_detection": skipped_no_detection,
        "manifest": str(manifest),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bootstrap_val")
    ap.add_argument("--src", type=Path, default=Path("data/cat_photos"))
    ap.add_argument("--out", type=Path, default=Path("training/edge/data/val"))
    ap.add_argument("--teacher", default="yolov8s.pt")
    ap.add_argument("--confidence", type=float, default=0.35)
    args = ap.parse_args(argv)

    summary = bootstrap(args.src, args.out, args.teacher, args.confidence)
    print(f"bootstrap: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
