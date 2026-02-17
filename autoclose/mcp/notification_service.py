"""MCP Notification service - webhook and alert dispatch."""

from typing import Any

import httpx

from autoclose.config import get_settings
from autoclose.schemas import ClassificationResult, ProcessingStatus, VisionResult


class NotificationService:
    """Service for sending notifications (webhooks, alerts) on workflow events."""

    def __init__(self, webhook_url: str | None = None) -> None:
        settings = get_settings()
        self.webhook_url = webhook_url or settings.notification_webhook_url

    async def notify_workflow_complete(
        self,
        job_id: str,
        document_id: str,
        status: ProcessingStatus,
        vision_result: VisionResult | None = None,
        classification_result: ClassificationResult | None = None,
        error: str | None = None,
    ) -> bool:
        """Send notification when workflow completes."""
        payload: dict[str, Any] = {
            "event": "workflow_complete",
            "job_id": job_id,
            "document_id": document_id,
            "status": status.value,
        }
        if vision_result:
            payload["vision"] = vision_result.model_dump()
        if classification_result:
            payload["classification"] = classification_result.model_dump()
        if error:
            payload["error"] = error

        return await self._post_webhook(payload)

    async def notify_classification(
        self,
        document_id: str,
        classification: ClassificationResult,
    ) -> bool:
        """Send notification for new classified transaction."""
        payload = {
            "event": "classification",
            "document_id": document_id,
            "classification": classification.model_dump(),
        }
        return await self._post_webhook(payload)

    async def _post_webhook(self, payload: dict[str, Any]) -> bool:
        """Post payload to configured webhook URL."""
        if not self.webhook_url:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                return response.is_success
        except Exception:
            return False
