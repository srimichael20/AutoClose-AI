"""MCP API Simulator - external accounting/ERP API integration simulation."""

from typing import Any

import httpx

from autoclose.schemas import ClassificationResult


class APISimulator:
    """
    Simulates external API calls (e.g., QuickBooks, Xero) for posting transactions.
    In production, replace with actual ERP/accounting API clients.
    """

    # Simulated endpoint - in production this would be a real API
    SIMULATED_API_URL = "https://httpbin.org/post"

    async def post_transaction(
        self,
        document_id: str,
        classification: ClassificationResult,
    ) -> dict[str, Any]:
        """
        Simulate posting a classified transaction to an external accounting API.
        Returns simulated response.
        """
        payload = {
            "document_id": document_id,
            "category": classification.category.value,
            "subcategory": classification.subcategory,
            "amount": classification.amount,
            "description": classification.description,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    self.SIMULATED_API_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                return {
                    "success": response.is_success,
                    "status_code": response.status_code,
                    "document_id": document_id,
                    "simulated": True,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id,
                "simulated": True,
            }
