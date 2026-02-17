"""Simulated external API client (ERP/accounting)."""

import httpx

from utils.schemas import ClassificationResult


SIMULATED_URL = "https://httpbin.org/post"


async def post_transaction(document_id: str, classification: ClassificationResult) -> dict:
    """Simulate posting transaction to external API."""
    payload = {
        "document_id": document_id,
        "category": classification.category.value,
        "subcategory": classification.subcategory,
        "amount": classification.amount,
        "description": classification.description,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(SIMULATED_URL, json=payload)
            return {"success": r.is_success, "status_code": r.status_code, "document_id": document_id}
    except Exception as e:
        return {"success": False, "error": str(e), "document_id": document_id}
