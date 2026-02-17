"""Webhook notification service."""

import httpx

from utils.config import get_settings
from utils.schemas import ClassificationResult, ProcessingStatus, VisionResult


async def notify_complete(
    job_id: str,
    document_id: str,
    status: ProcessingStatus,
    vision: VisionResult | None = None,
    classification: ClassificationResult | None = None,
    error: str | None = None,
) -> bool:
    url = get_settings().notification_webhook_url
    if not url:
        return False
    payload = {
        "event": "workflow_complete",
        "job_id": job_id,
        "document_id": document_id,
        "status": status.value,
    }
    if vision:
        payload["vision"] = vision.model_dump()
    if classification:
        payload["classification"] = classification.model_dump()
    if error:
        payload["error"] = error
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            return (await client.post(url, json=payload)).is_success
    except Exception:
        return False
