import logging

import httpx

logger = logging.getLogger(__name__)


class Notifier:
    """Sends notifications via webhook (ntfy/Pushover)."""

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url
        self._client = httpx.AsyncClient()

    async def notify(self, title: str, message: str):
        if not self.webhook_url:
            return
        try:
            await self._client.post(
                self.webhook_url,
                json={"title": title, "message": message},
                headers={"Content-Type": "application/json"},
                timeout=5.0,
            )
        except httpx.RequestError as e:
            logger.error(f"Notification failed: {e}")

    async def notify_zap(self, cat_name: str | None, zone_name: str | None):
        cat = cat_name or "Unknown cat"
        zone = zone_name or "unknown zone"
        await self.notify(
            title="CatZap Fired!",
            message=f"{cat} was caught on {zone} and got zapped!",
        )

    async def close(self):
        await self._client.aclose()
