"""Tests for training/edge/auto_label.py — mirrors server/tests/test_detector.py style.

Mocks cv2.VideoCapture (the MJPEG stream) and CatDetector so the suite
runs without a live ESP32-CAM and without loading any model weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from training.edge import auto_label
from training.edge.auto_label import (
    CAT_CLASS_ID,
    _hflip_yolo,
    capture,
    review,
    write_pair,
    xyxy_to_yolo,
    yolo_label_lines,
)
from training.edge.make_dataset_manifest import build_manifest, write_manifest


def test_xyxy_to_yolo_centers_and_dims() -> None:
    cx, cy, w, h = xyxy_to_yolo([0.10, 0.20, 0.30, 0.60])
    assert cx == pytest.approx(0.20)
    assert cy == pytest.approx(0.40)
    assert w == pytest.approx(0.20)
    assert h == pytest.approx(0.40)


def test_yolo_label_lines_class_id_zero_and_format() -> None:
    """class_id MUST be 0 (single-class), four normalized floats per line."""
    detections = [
        {"bbox": [0.10, 0.20, 0.30, 0.60], "confidence": 0.92},
        {"bbox": [0.50, 0.50, 0.80, 0.90], "confidence": 0.81},
    ]
    lines = yolo_label_lines(detections)
    assert len(lines) == 2
    for line in lines:
        parts = line.split()
        assert len(parts) == 5
        assert parts[0] == str(CAT_CLASS_ID)
        # Remaining four are valid floats in [0,1].
        for p in parts[1:]:
            v = float(p)
            assert 0.0 <= v <= 1.0


def test_write_pair_creates_jpg_and_txt(tmp_path: Path) -> None:
    img = np.full((48, 64, 3), 200, dtype=np.uint8)
    detections = [{"bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.9}]
    img_p, lbl_p = write_pair(img, detections, "frame0", tmp_path)
    assert img_p.exists() and img_p.suffix == ".jpg"
    assert lbl_p.exists() and lbl_p.suffix == ".txt"
    contents = lbl_p.read_text().strip().splitlines()
    assert len(contents) == 1
    assert contents[0].split()[0] == "0"


def test_hflip_yolo_mirrors_x_only() -> None:
    flipped = _hflip_yolo([{"bbox": [0.10, 0.20, 0.30, 0.40], "confidence": 0.9}])
    assert flipped[0]["bbox"] == pytest.approx([0.70, 0.20, 0.90, 0.40])


def _fake_video_capture(frames: list[np.ndarray]) -> MagicMock:
    cap = MagicMock()
    cap.isOpened.return_value = True
    seq = list(frames) + [None]
    idx = {"i": 0}

    def _read():
        i = idx["i"]
        idx["i"] += 1
        if i >= len(seq) or seq[i] is None:
            return False, None
        return True, seq[i]

    cap.read.side_effect = _read
    return cap


@patch("training.edge.auto_label.cv2.VideoCapture")
def test_capture_writes_yolo_pairs_when_teacher_detects(mock_vc, tmp_path: Path) -> None:
    frames = [np.full((48, 64, 3), v, dtype=np.uint8) for v in (10, 20, 30)]
    mock_vc.return_value = _fake_video_capture(frames)

    fake_teacher = MagicMock()
    fake_teacher.detect.side_effect = [
        [{"bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.9}],
        [],  # zero-detection frame must be skipped
        [{"bbox": [0.2, 0.3, 0.4, 0.5], "confidence": 0.4}],  # below threshold
    ]
    detector_module = MagicMock()
    detector_module.CatDetector.return_value = fake_teacher

    with patch.dict("sys.modules", {"server.vision.detector": detector_module}):
        summary = capture(
            "http://fake/stream",
            tmp_path,
            target_frames=10,
            min_confidence=0.5,
        )

    assert summary["written"] == 1
    assert summary["skipped_no_detection"] == 1
    assert summary["skipped_low_confidence"] == 1
    pairs = sorted(p.stem for p in tmp_path.glob("*.jpg"))
    labels = sorted(p.stem for p in tmp_path.glob("*.txt"))
    assert pairs == labels
    assert len(pairs) == 1
    label_text = (tmp_path / f"{pairs[0]}.txt").read_text().strip()
    assert label_text.split()[0] == "0"


@patch("training.edge.auto_label.cv2.VideoCapture")
def test_capture_stops_at_target_frames(mock_vc, tmp_path: Path) -> None:
    frames = [np.full((48, 64, 3), v % 255, dtype=np.uint8) for v in range(20)]
    mock_vc.return_value = _fake_video_capture(frames)

    fake_teacher = MagicMock()
    fake_teacher.detect.return_value = [
        {"bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.9}
    ]
    detector_module = MagicMock()
    detector_module.CatDetector.return_value = fake_teacher

    with patch.dict("sys.modules", {"server.vision.detector": detector_module}):
        summary = capture(
            "http://fake/stream",
            tmp_path,
            target_frames=3,
            min_confidence=0.5,
        )

    assert summary["written"] == 3
    assert len(list(tmp_path.glob("*.jpg"))) == 3


def test_review_routes_pairs_by_keypress(tmp_path: Path) -> None:
    in_dir = tmp_path / "raw"
    accept = tmp_path / "labeled"
    reject = tmp_path / "rejected"
    in_dir.mkdir()

    img = np.full((48, 64, 3), 180, dtype=np.uint8)
    for stem, label in (
        ("a", "0 0.5 0.5 0.4 0.4\n"),
        ("r", "0 0.4 0.4 0.2 0.2\n"),
        ("n", "0 0.3 0.3 0.1 0.1\n"),
    ):
        cv2.imwrite(str(in_dir / f"{stem}.jpg"), img)
        (in_dir / f"{stem}.txt").write_text(label)

    keys = iter([ord("a"), ord("n"), ord("r")])  # alphabetical sort: a, n, r
    counts = review(
        in_dir,
        accept,
        reject,
        show_fn=lambda *_a, **_k: None,
        waitkey_fn=lambda: next(keys),
    )
    assert counts["accepted"] == 1
    assert counts["rejected"] == 1
    assert counts["skipped"] == 1
    assert (accept / "a.jpg").exists() and (accept / "a.txt").exists()
    assert (reject / "r.jpg").exists() and (reject / "r.txt").exists()
    assert (in_dir / "n.jpg").exists() and (in_dir / "n.txt").exists()


def test_review_delete_box_clears_label_keeps_image(tmp_path: Path) -> None:
    in_dir = tmp_path / "raw"
    in_dir.mkdir()
    img = np.full((48, 64, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(in_dir / "x.jpg"), img)
    (in_dir / "x.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    counts = review(
        in_dir,
        tmp_path / "accept",
        tmp_path / "reject",
        show_fn=lambda *_a, **_k: None,
        waitkey_fn=lambda: ord("d"),
    )
    assert counts["boxes_deleted"] == 1
    assert (in_dir / "x.jpg").exists()
    assert (in_dir / "x.txt").read_text() == ""


def test_make_dataset_manifest_split_and_checksums(tmp_path: Path) -> None:
    img = np.full((48, 64, 3), 180, dtype=np.uint8)
    for i in range(10):
        cv2.imwrite(str(tmp_path / f"{i:02d}.jpg"), img)
        (tmp_path / f"{i:02d}.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    out = write_manifest(tmp_path, val_fraction=0.2)
    assert out == tmp_path / "manifest.json"
    data = json.loads(out.read_text())
    assert data["count"] == 10
    assert data["train_count"] + data["val_count"] == 10
    assert abs(data["val_count"] / data["count"] - 0.2) <= 0.1
    for entry in data["train"] + data["val"]:
        assert len(entry["sha256"]) == 64


def test_make_dataset_manifest_raises_on_no_pairs(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        build_manifest(tmp_path)


def test_main_dispatch_routes_review_subcommand() -> None:
    with patch.object(auto_label, "_review_main", return_value=0) as mock_review, patch.object(
        auto_label, "_capture_main", return_value=0
    ) as mock_capture:
        rc = auto_label.main(["review", "--in-dir", "/tmp/raw"])
    assert rc == 0
    mock_review.assert_called_once_with(["--in-dir", "/tmp/raw"])
    mock_capture.assert_not_called()


def test_main_dispatch_defaults_to_capture() -> None:
    with patch.object(auto_label, "_capture_main", return_value=0) as mock_capture:
        rc = auto_label.main(["--stream-url", "http://x/stream", "--out-dir", "/tmp/raw"])
    assert rc == 0
    mock_capture.assert_called_once()
