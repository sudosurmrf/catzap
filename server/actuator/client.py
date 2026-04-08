import logging

import httpx

logger = logging.getLogger(__name__)


class ActuatorClient:
    """HTTP client for communicating with the ESP32 DEVKITV1 actuator."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient()

    async def aim(self, pan: float, tilt: float) -> bool:
        try:
            response = await self._client.post(
                f"{self.base_url}/aim",
                json={"pan": pan, "tilt": tilt},
                timeout=2.0,
            )
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.error(f"Failed to send aim command: {e}")
            return False

    async def fire(self, duration_ms: int = 200) -> bool:
        try:
            response = await self._client.post(
                f"{self.base_url}/fire",
                json={"duration_ms": duration_ms},
                timeout=2.0,
            )
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.error(f"Failed to send fire command: {e}")
            return False

    async def aim_and_fire(self, pan: float, tilt: float, duration_ms: int = 200) -> bool:
        aimed = await self.aim(pan, tilt)
        if aimed:
            return await self.fire(duration_ms)
        return False

    async def close(self):
        await self._client.aclose()
