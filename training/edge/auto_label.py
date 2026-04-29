"""Auto-labeler — pull frames, run CatDetector teacher, emit YOLO labels.

Three subcommands:

  (default capture)
    python -m training.edge.auto_label \\
        --stream-url http://<esp32cam-ip>/stream \\
        --out-dir training/edge/data/raw/ \\
        --target-frames 600 \\
        --min-confidence 0.5

  (manual cleanup, opens cv2.imshow window for each frame)
    python -m training.edge.auto_label review \\
        --in-dir training/edge/data/raw/ \\
        --accept-dir training/edge/data/labeled/ \\
        --reject-dir training/edge/data/rejected/

  (offline augment — bootstraps a labeled set from a local image dir
   when no ESP32-CAM is available, e.g. from data/cat_photos/)
    python -m training.edge.auto_label bootstrap \\
        --src-dir data/cat_photos \\
        --out-dir training/edge/data/labeled \\
        --target-frames 600 \\
        --augments-per-image 20

YOLO label format (one line per detection):
    <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
class_id is always 0 (single-class "cat") regardless of teacher's COCO id.

Deviation from input spec: the input spec asked for server/vision/auto_label.py
but the PRD non-goal forbids touching server/. This module lives under
training/edge/ instead. The teacher detector is imported READ-ONLY from
server.vision.detector.CatDetector — no edits to server/ at all.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import cv2
import numpy as np

# Single-class id for our cat-only dataset. The teacher's COCO_CAT_CLASS=15
# (server/vision/detector.py:4) is filtered there; we re-emit as id 0.
CAT_CLASS_ID = 0


def xyxy_to_yolo(bbox: list[float]) -> tuple[float, float, float, float]:
    """Convert normalized [x1,y1,x2,y2] to YOLO (cx, cy, w, h), all in [0,1]."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return cx, cy, w, h


def yolo_label_lines(detections: list[dict]) -> list[str]:
    """detections: [{"bbox":[x1,y1,x2,y2 normalized], "confidence":float}, ...]"""
    lines: list[str] = []
    for det in detections:
        cx, cy, w, h = xyxy_to_yolo(det["bbox"])
        lines.append(f"{CAT_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def write_pair(image: np.ndarray, detections: list[dict], stem: str, out_dir: Path) -> tuple[Path, Path]:
    """Write <stem>.jpg + <stem>.txt under out_dir/ (creates dir).

    Returns (image_path, label_path). Image saved with quality=85.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{stem}.jpg"
    lbl_path = out_dir / f"{stem}.txt"
    if not cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise RuntimeError(f"cv2.imwrite failed for {img_path}")
    lbl_path.write_text("\n".join(yolo_label_lines(detections)) + "\n")
    return img_path, lbl_path


# ----------------------------------------------------------------------
# capture (live ESP32-CAM stream)
# ----------------------------------------------------------------------


def iter_mjpeg_frames(stream_url: str) -> Iterator[np.ndarray]:
    """Yield decoded BGR frames from an MJPEG stream.

    cv2.VideoCapture handles the multipart/x-mixed-replace body produced
    by the ESP32-CAM /stream endpoint (see firmware/esp32-cam/src/main.cpp:33-58).
    """
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open stream: {stream_url}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield frame
    finally:
        cap.release()


def capture(
    stream_url: str,
    out_dir: Path,
    target_frames: int,
    min_confidence: float,
    teacher_path: str = "yolov8s.pt",
    sleep_between_frames: float = 0.0,
) -> dict:
    """Pull frames from MJPEG stream, run teacher, write YOLO label pairs.

    Returns a summary dict suitable for printing/JSON serialization.
    """
    from server.vision.detector import CatDetector

    teacher = CatDetector(teacher_path, confidence_threshold=min_confidence)

    written = 0
    skipped_no_detection = 0
    skipped_low_confidence = 0
    seen = 0

    for frame in iter_mjpeg_frames(stream_url):
        seen += 1
        detections = teacher.detect(frame)
        if not detections:
            skipped_no_detection += 1
            continue
        kept = [d for d in detections if d["confidence"] >= min_confidence]
        if not kept:
            skipped_low_confidence += 1
            continue
        write_pair(frame, kept, uuid4().hex, out_dir)
        written += 1
        if written >= target_frames:
            break
        if sleep_between_frames > 0:
            time.sleep(sleep_between_frames)

    return {
        "frames_seen": seen,
        "written": written,
        "skipped_no_detection": skipped_no_detection,
        "skipped_low_confidence": skipped_low_confidence,
        "out_dir": str(out_dir),
    }


# ----------------------------------------------------------------------
# review (manual cleanup — cv2.imshow window)
# ----------------------------------------------------------------------


def _draw_yolo_overlays(frame: np.ndarray, label_path: Path) -> np.ndarray:
    """Return a copy of frame with YOLO label boxes drawn."""
    overlay = frame.copy()
    if not label_path.exists():
        return overlay
    h, w = frame.shape[:2]
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        x1 = int((cx - bw / 2.0) * w)
        y1 = int((cy - bh / 2.0) * h)
        x2 = int((cx + bw / 2.0) * w)
        y2 = int((cy + bh / 2.0) * h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return overlay


def review(
    in_dir: Path,
    accept_dir: Path,
    reject_dir: Path,
    *,
    show_fn=None,
    waitkey_fn=None,
) -> dict:
    """Walk every <stem>.jpg in in_dir; show with overlay; act on key.

    Keys:
      [a] accept  -> move pair to accept_dir/
      [r] reject  -> move pair to reject_dir/
      [d] delete-box -> rewrite label file with all boxes removed, keep in in_dir
      [n] next    -> leave pair untouched in in_dir

    `show_fn` and `waitkey_fn` are dependency-injection hooks for unit
    testing (cv2.imshow / cv2.waitKey are non-deterministic in CI).
    """
    if show_fn is None:
        show_fn = lambda title, img: cv2.imshow(title, img)  # noqa: E731
    if waitkey_fn is None:
        waitkey_fn = lambda: cv2.waitKey(0) & 0xFF  # noqa: E731

    accept_dir.mkdir(parents=True, exist_ok=True)
    reject_dir.mkdir(parents=True, exist_ok=True)

    counts = {"accepted": 0, "rejected": 0, "boxes_deleted": 0, "skipped": 0, "missing_image": 0}

    for jpg in sorted(in_dir.glob("*.jpg")):
        lbl = jpg.with_suffix(".txt")
        frame = cv2.imread(str(jpg))
        if frame is None:
            counts["missing_image"] += 1
            continue
        overlay = _draw_yolo_overlays(frame, lbl)
        show_fn(f"review: {jpg.name}", overlay)
        key = waitkey_fn()
        if key == ord("a"):
            jpg.rename(accept_dir / jpg.name)
            if lbl.exists():
                lbl.rename(accept_dir / lbl.name)
            counts["accepted"] += 1
        elif key == ord("r"):
            jpg.rename(reject_dir / jpg.name)
            if lbl.exists():
                lbl.rename(reject_dir / lbl.name)
            counts["rejected"] += 1
        elif key == ord("d"):
            if lbl.exists():
                lbl.write_text("")
            counts["boxes_deleted"] += 1
        else:  # 'n' or anything else
            counts["skipped"] += 1

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    return counts


# ----------------------------------------------------------------------
# bootstrap (offline augmentation when no live stream is available)
# ----------------------------------------------------------------------


def _augment_frame(frame: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Apply a small bag of bbox-preserving (or hflip) augmentations.

    Returns (augmented_frame, was_hflipped). hflip is the only spatial
    augmentation; callers must mirror cx in YOLO labels when True.
    """
    out = frame.copy()
    flipped = False
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
        flipped = True
    # brightness / contrast jitter (alpha=contrast, beta=brightness)
    alpha = float(rng.uniform(0.85, 1.15))
    beta = float(rng.uniform(-15.0, 15.0))
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    # hue shift in HSV
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-10, 10))) % 180
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # mild gaussian noise
    if rng.random() < 0.3:
        noise = rng.normal(0, 4, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out, flipped


def _hflip_yolo(detections: list[dict]) -> list[dict]:
    flipped = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        flipped.append({"bbox": [1.0 - x2, y1, 1.0 - x1, y2], "confidence": det["confidence"]})
    return flipped


def bootstrap(
    src_dir: Path,
    out_dir: Path,
    target_frames: int,
    augments_per_image: int = 20,
    min_confidence: float = 0.35,
    teacher_path: str = "yolov8s.pt",
    resize_to: tuple[int, int] = (320, 240),
    seed: int = 42,
) -> dict:
    """Offline augmentation bootstrap.

    Reads images from src_dir, runs the teacher to get a single-class
    cat label per image, then synthesizes `augments_per_image` augmented
    copies per source image until target_frames is reached. Writes
    <stem>.jpg + <stem>.txt pairs under out_dir.

    Used when no ESP32-CAM stream is available — see US-002.md for the
    deviation rationale.
    """
    from server.vision.detector import CatDetector

    rng = np.random.default_rng(seed)
    teacher = CatDetector(teacher_path, confidence_threshold=min_confidence)

    sources = sorted(src_dir.rglob("*.jpg"))
    if not sources:
        return {"written": 0, "sources": 0, "skipped_no_detection": 0, "out_dir": str(out_dir)}

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_no_detection = 0
    used_sources = 0
    src_idx = 0
    while written < target_frames and src_idx < len(sources) * augments_per_image:
        src_path = sources[src_idx % len(sources)]
        src_idx += 1
        frame = cv2.imread(str(src_path))
        if frame is None:
            continue
        if resize_to is not None:
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
        detections = teacher.detect(frame)
        if not detections:
            skipped_no_detection += 1
            continue
        if src_idx <= len(sources):
            used_sources += 1
        aug, flipped = _augment_frame(frame, rng)
        labels = _hflip_yolo(detections) if flipped else detections
        write_pair(aug, labels, uuid4().hex, out_dir)
        written += 1

    return {
        "written": written,
        "sources": len(sources),
        "used_sources": used_sources,
        "skipped_no_detection": skipped_no_detection,
        "out_dir": str(out_dir),
    }


# ----------------------------------------------------------------------
# CLI dispatch
# ----------------------------------------------------------------------


def _capture_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="auto_label")
    ap.add_argument("--stream-url", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-frames", type=int, default=600)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--teacher", default="yolov8s.pt")
    args = ap.parse_args(argv)
    summary = capture(
        args.stream_url, args.out_dir, args.target_frames, args.min_confidence, args.teacher
    )
    print(f"capture: {summary}")
    return 0


def _review_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="auto_label review")
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--accept-dir", type=Path, default=Path("training/edge/data/labeled"))
    ap.add_argument("--reject-dir", type=Path, default=Path("training/edge/data/rejected"))
    args = ap.parse_args(argv)
    counts = review(args.in_dir, args.accept_dir, args.reject_dir)
    print(f"review: {counts}")
    return 0


def _bootstrap_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="auto_label bootstrap")
    ap.add_argument("--src-dir", type=Path, default=Path("data/cat_photos"))
    ap.add_argument("--out-dir", type=Path, default=Path("training/edge/data/labeled"))
    ap.add_argument("--target-frames", type=int, default=600)
    ap.add_argument("--augments-per-image", type=int, default=25)
    ap.add_argument("--min-confidence", type=float, default=0.35)
    ap.add_argument("--teacher", default="yolov8s.pt")
    ap.add_argument("--resize", type=int, nargs=2, default=[320, 240], metavar=("W", "H"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    summary = bootstrap(
        args.src_dir,
        args.out_dir,
        args.target_frames,
        augments_per_image=args.augments_per_image,
        min_confidence=args.min_confidence,
        teacher_path=args.teacher,
        resize_to=(args.resize[0], args.resize[1]),
        seed=args.seed,
    )
    print(f"bootstrap: {summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "review":
        return _review_main(argv[1:])
    if argv and argv[0] == "bootstrap":
        return _bootstrap_main(argv[1:])
    return _capture_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
