"""MCP Integration Agent - Database, File System, API, Notifications."""

from typing import Any

from autoclose.agents.base import BaseAgent
from autoclose.mcp import APISimulator, DatabaseService, FileSystemService, NotificationService
from autoclose.schemas import DocumentType, MCPActionResult, ProcessingStatus, WorkflowState


class MCPIntegrationAgent(BaseAgent):
    """
    Coordinates MCP services: DB persistence, file archival, API posting, notifications.
    Agent decides which integrations to invoke based on workflow context.
    """

    name = "mcp_integration"

    def __init__(
        self,
        database: DatabaseService | None = None,
        file_system: FileSystemService | None = None,
        api_simulator: APISimulator | None = None,
        notification: NotificationService | None = None,
    ):
        self.db = database or DatabaseService()
        self.fs = file_system or FileSystemService()
        self.api = api_simulator or APISimulator()
        self.notification = notification or NotificationService()

    async def process(self, state: WorkflowState) -> dict[str, Any]:
        """Execute MCP integrations and return combined result."""
        document_id = state.document_id
        intake = state.intake_result
        vision = state.vision_result
        classification = state.classification_result

        details: dict[str, Any] = {}
        db_ok = False
        file_ok = False
        api_ok = False
        notify_ok = False

        if intake:
            db_ok = await self.db.store_document(
                document_id=document_id,
                document_type=intake.document_type,
                raw_content=intake.raw_content,
                file_path=intake.file_path,
            )
            details["database"] = "stored" if db_ok else "failed"

        if classification:
            db_tx = await self.db.store_transaction(document_id, classification)
            db_ok = db_ok or db_tx

            api_resp = await self.api.post_transaction(document_id, classification)
            api_ok = api_resp.get("success", False)
            details["api"] = api_resp

        if vision or classification:
            file_path = await self.fs.store_processed_result(
                document_id=document_id,
                vision_result=vision,
                classification_result=classification,
                metadata={"job_id": state.job_id},
            )
            file_ok = True
            details["processed_file"] = file_path

        notify_ok = await self.notification.notify_workflow_complete(
            job_id=state.job_id,
            document_id=document_id,
            status=ProcessingStatus.COMPLETED,
            vision_result=vision,
            classification_result=classification,
        )
        details["notification"] = "sent" if notify_ok else "skipped"

        result = MCPActionResult(
            document_id=document_id,
            database_recorded=db_ok,
            file_stored=file_ok,
            api_called=api_ok,
            notification_sent=notify_ok,
            details=details,
        )

        return {
            "mcp_result": result,
            "current_step": "end",
            "status": ProcessingStatus.COMPLETED,
            "messages": state.messages + ["MCP integration completed"],
        }
