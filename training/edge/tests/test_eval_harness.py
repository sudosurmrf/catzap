"""Tests for training/edge/eval/ — mirrors server/tests/test_detector.py style.

Mocks the model classes so the suite has no heavy ML dependencies.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from training.edge.eval.latency import measure_latency
from training.edge.eval.metrics import compute_map50, iou_xyxy
from training.edge.eval.types import EvalResult


def test_evalresult_json_round_trip(tmp_path: Path) -> None:
    r = EvalResult(
        story_id="US-001-yolov8n-baseline-224",
        model_path="yolov8n.pt",
        model_format="pytorch",
        map50=0.873,
        size_bytes=6_534_387,
        params=3_157_200,
        flops=4_400_000_000,
        input_hw=(224, 224),
        latency_ms_p50=18.4,
        latency_ms_p95=22.1,
        val_images=27,
        notes="silver GT bootstrap from data/cat_photos",
    )

    path = tmp_path / "row.json"
    r.write(path)

    raw = json.loads(path.read_text())
    assert raw["model_format"] == "pytorch"
    assert raw["input_hw"] == [224, 224]

    rt = EvalResult.read(path)
    assert rt == r
    assert isinstance(rt.input_hw, tuple)


def test_latency_p50_matches_known_sleep_duration() -> None:
    """measure_latency should report a p50 close to a mocked predict()'s sleep."""
    sleep_s = 0.005
    calls = {"n": 0}

    def fake_predict(_frame: np.ndarray) -> list[dict]:
        calls["n"] += 1
        time.sleep(sleep_s)
        return []

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    p50, p95 = measure_latency(fake_predict, frame, runs=20, warmup=5)

    assert calls["n"] == 25
    assert 4.0 <= p50 <= 25.0
    assert p95 >= p50


def test_map50_perfect_when_predictions_equal_gt() -> None:
    gt = [
        [[0.10, 0.10, 0.40, 0.40]],
        [[0.50, 0.50, 0.90, 0.90], [0.05, 0.05, 0.20, 0.20]],
    ]
    preds = [
        [{"bbox": gt[0][0], "confidence": 0.99}],
        [
            {"bbox": gt[1][0], "confidence": 0.95},
            {"bbox": gt[1][1], "confidence": 0.80},
        ],
    ]
    assert compute_map50(preds, gt) == pytest.approx(1.0, abs=1e-6)


def test_map50_zero_when_no_predictions() -> None:
    gt = [[[0.10, 0.10, 0.40, 0.40]]]
    assert compute_map50([[]], gt) == 0.0


def test_iou_disjoint_boxes() -> None:
    assert iou_xyxy([0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.6, 0.6]) == 0.0


def test_iou_full_overlap() -> None:
    box = [0.2, 0.2, 0.4, 0.4]
    assert iou_xyxy(box, box) == pytest.approx(1.0)


@patch("training.edge.eval.adapters.pytorch_adapter.YOLO")
@patch("training.edge.eval.adapters.pytorch_adapter.torch")
def test_pytorch_adapter_returns_cat_only_dicts(mock_torch, mock_yolo_class) -> None:
    """Single-class fallback: predictions arrive shaped like CatDetector."""
    mock_yolo_class.return_value = MagicMock()
    boxes = MagicMock()
    boxes.xyxyn = MagicMock()
    boxes.xyxyn.cpu.return_value.numpy.return_value = np.array(
        [[0.1, 0.2, 0.3, 0.5], [0.5, 0.5, 0.8, 0.9]]
    )
    boxes.conf = MagicMock()
    boxes.conf.cpu.return_value.numpy.return_value = np.array([0.92, 0.45])
    boxes.cls = MagicMock()
    boxes.cls.cpu.return_value.numpy.return_value = np.array([15, 0])
    result = MagicMock()
    result.boxes = boxes
    mock_yolo_class.return_value.return_value = [result]
    mock_yolo_class.return_value.names = {0: "person", 15: "cat"}

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__ = MagicMock(return_value=False)
    mock_torch.no_grad.return_value = cm

    from training.edge.eval.adapters.pytorch_adapter import YoloPytorchAdapter

    adapter = YoloPytorchAdapter("ignored.pt", imgsz=224, confidence_threshold=0.5)
    detections = adapter.predict(np.zeros((224, 224, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0]["bbox"] == [0.1, 0.2, 0.3, 0.5]
    assert detections[0]["confidence"] == pytest.approx(0.92)


def test_evaluate_end_to_end_with_mocked_adapter(tmp_path: Path) -> None:
    """Smoke through run_eval.evaluate without touching real model code."""
    val = tmp_path / "val"
    (val / "images").mkdir(parents=True)
    (val / "labels").mkdir()

    import cv2

    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    image_path = val / "images" / "frame0.jpg"
    assert cv2.imwrite(str(image_path), img)
    (val / "labels" / "frame0.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    fake_adapter = MagicMock()
    fake_adapter.predict.return_value = [
        {"bbox": [0.3, 0.3, 0.7, 0.7], "confidence": 0.95}
    ]
    fake_adapter.num_params.return_value = 100
    fake_adapter.num_flops.return_value = 200
    fake_adapter.input_hw = (224, 224)

    model_file = tmp_path / "fake_model.pt"
    model_file.write_bytes(b"x" * 4096)

    with patch("training.edge.eval.run_eval._load_adapter", return_value=fake_adapter):
        from training.edge.eval.run_eval import evaluate

        result = evaluate(
            str(model_file),
            "pytorch",
            "TEST-001",
            val,
            imgsz=224,
            notes="unit test",
        )

    assert result.story_id == "TEST-001"
    assert result.val_images == 1
    assert result.size_bytes == 4096
    assert result.params == 100
    assert result.flops == 200
    assert result.map50 == pytest.approx(1.0, abs=1e-6)
    assert result.latency_ms_p50 >= 0.0
    assert result.latency_ms_p95 >= result.latency_ms_p50
