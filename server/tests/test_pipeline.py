import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock
from server.vision.pipeline import VisionPipeline


@pytest.fixture
def mock_detector():
    detector = MagicMock()
    detector.detect.return_value = [
        {"bbox": [0.2, 0.2, 0.4, 0.5], "confidence": 0.92}
    ]
    return detector


@pytest.fixture
def mock_actuator():
    actuator = AsyncMock()
    actuator.aim_and_fire.return_value = True
    return actuator


@pytest.fixture
def mock_calibration():
    cal = MagicMock()
    cal.pixel_to_angle.return_value = (90.0, 45.0)
    return cal


@pytest.fixture
def zones():
    return [
        {
            "id": "z1",
            "name": "Kitchen Counter",
            "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.6], [0.1, 0.6]],
            "overlap_threshold": 0.3,
            "cooldown_seconds": 10,
            "enabled": True,
        }
    ]


@pytest.mark.asyncio
async def test_pipeline_detects_violation_and_fires(mock_detector, mock_actuator, mock_calibration, zones):
    on_event = MagicMock()
    pipeline = VisionPipeline(
        detector=mock_detector,
        actuator=mock_actuator,
        calibration=mock_calibration,
        zones=zones,
        armed=True,
        on_event=on_event,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = await pipeline.process_frame(frame)

    assert len(result["detections"]) == 1
    assert len(result["violations"]) == 1
    assert result["fired"] == True
    mock_actuator.aim_and_fire.assert_called_once()
    assert on_event.call_count >= 1


@pytest.mark.asyncio
async def test_pipeline_respects_cooldown(mock_detector, mock_actuator, mock_calibration, zones):
    pipeline = VisionPipeline(
        detector=mock_detector,
        actuator=mock_actuator,
        calibration=mock_calibration,
        zones=zones,
        armed=True,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result1 = await pipeline.process_frame(frame)
    assert result1["fired"] == True

    result2 = await pipeline.process_frame(frame)
    assert result2["fired"] == False


@pytest.mark.asyncio
async def test_pipeline_disarmed_does_not_fire(mock_detector, mock_actuator, mock_calibration, zones):
    pipeline = VisionPipeline(
        detector=mock_detector,
        actuator=mock_actuator,
        calibration=mock_calibration,
        zones=zones,
        armed=False,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = await pipeline.process_frame(frame)

    assert len(result["detections"]) == 1
    assert len(result["violations"]) == 1
    assert result["fired"] == False
    mock_actuator.aim_and_fire.assert_not_called()
