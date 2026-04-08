import numpy as np
from unittest.mock import patch, MagicMock
from server.vision.detector import CatDetector


def _make_mock_result(boxes_data):
    result = MagicMock()
    boxes = MagicMock()
    if len(boxes_data) == 0:
        boxes.xyxyn = MagicMock()
        boxes.xyxyn.cpu.return_value.numpy.return_value = np.array([]).reshape(0, 4)
        boxes.conf = MagicMock()
        boxes.conf.cpu.return_value.numpy.return_value = np.array([])
        boxes.cls = MagicMock()
        boxes.cls.cpu.return_value.numpy.return_value = np.array([])
    else:
        xyxyn = np.array([[b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]] for b in boxes_data])
        conf = np.array([b["conf"] for b in boxes_data])
        cls = np.array([b["cls"] for b in boxes_data])
        boxes.xyxyn = MagicMock()
        boxes.xyxyn.cpu.return_value.numpy.return_value = xyxyn
        boxes.conf = MagicMock()
        boxes.conf.cpu.return_value.numpy.return_value = conf
        boxes.cls = MagicMock()
        boxes.cls.cpu.return_value.numpy.return_value = cls
    result.boxes = boxes
    return result


@patch("server.vision.detector.YOLO")
def test_detect_cats_returns_cat_detections(mock_yolo_class):
    mock_model = MagicMock()
    mock_yolo_class.return_value = mock_model
    mock_model.return_value = [
        _make_mock_result([
            {"bbox": [0.1, 0.2, 0.3, 0.5], "conf": 0.92, "cls": 15},
            {"bbox": [0.5, 0.5, 0.8, 0.9], "conf": 0.45, "cls": 0},
        ])
    ]

    detector = CatDetector(confidence_threshold=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0]["bbox"] == [0.1, 0.2, 0.3, 0.5]
    assert detections[0]["confidence"] == 0.92


@patch("server.vision.detector.YOLO")
def test_detect_cats_empty_frame(mock_yolo_class):
    mock_model = MagicMock()
    mock_yolo_class.return_value = mock_model
    mock_model.return_value = [_make_mock_result([])]

    detector = CatDetector(confidence_threshold=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert len(detections) == 0


@patch("server.vision.detector.YOLO")
def test_detect_cats_filters_low_confidence(mock_yolo_class):
    mock_model = MagicMock()
    mock_yolo_class.return_value = mock_model
    mock_model.return_value = [
        _make_mock_result([
            {"bbox": [0.1, 0.2, 0.3, 0.5], "conf": 0.3, "cls": 15},
        ])
    ]

    detector = CatDetector(confidence_threshold=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert len(detections) == 0
