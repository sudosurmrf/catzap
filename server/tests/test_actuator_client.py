import pytest
from unittest.mock import AsyncMock, patch
from server.actuator.client import ActuatorClient


@pytest.mark.asyncio
@patch("server.actuator.client.httpx.AsyncClient.post")
async def test_aim_sends_correct_angles(mock_post):
    mock_post.return_value = AsyncMock(status_code=200)
    client = ActuatorClient(base_url="http://192.168.1.101")
    await client.aim(pan=45.0, tilt=30.0)
    mock_post.assert_called_once_with(
        "http://192.168.1.101/aim",
        json={"pan": 45.0, "tilt": 30.0},
        timeout=2.0,
    )


@pytest.mark.asyncio
@patch("server.actuator.client.httpx.AsyncClient.post")
async def test_fire_sends_request(mock_post):
    mock_post.return_value = AsyncMock(status_code=200)
    client = ActuatorClient(base_url="http://192.168.1.101")
    await client.fire(duration_ms=200)
    mock_post.assert_called_once_with(
        "http://192.168.1.101/fire",
        json={"duration_ms": 200},
        timeout=2.0,
    )


@pytest.mark.asyncio
@patch("server.actuator.client.httpx.AsyncClient.post")
async def test_aim_and_fire_combined(mock_post):
    mock_post.return_value = AsyncMock(status_code=200)
    client = ActuatorClient(base_url="http://192.168.1.101")
    await client.aim_and_fire(pan=90.0, tilt=60.0, duration_ms=300)
    assert mock_post.call_count == 2
