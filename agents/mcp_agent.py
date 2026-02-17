"""MCP Agent - Database, file, API, notifications."""

from typing import Any

from database import Database
from utils.file_storage import FileStorage
from utils.api_client import post_transaction
from utils.notification import notify_complete
from utils.schemas import (
    MCPActionResult,
    ProcessingStatus,
    WorkflowState,
)


class MCPAgent:
    """Orchestrates DB, file, API, and notification integrations."""

    name = "mcp"

    def __init__(
        self,
        db: Database | None = None,
        fs: FileStorage | None = None,
    ):
        self._db = db or Database()
        self._fs = fs or FileStorage()

    async def process(self, state: WorkflowState) -> dict[str, Any]:
        doc_id = state.document_id
        intake = state.intake_result
        vision = state.vision_result
        classification = state.classification_result

        details: dict[str, Any] = {}
        db_ok = False
        file_ok = False
        api_ok = False

        if intake:
            db_ok = await self._db.store_document(
                doc_id,
                intake.document_type,
                intake.raw_content,
                intake.file_path,
            )
            details["db"] = "ok" if db_ok else "fail"

        if classification:
            tx_ok = await self._db.store_transaction(doc_id, classification)
            db_ok = db_ok or tx_ok
            api_res = await post_transaction(doc_id, classification)
            api_ok = api_res.get("success", False)
            details["api"] = api_res

        if vision or classification:
            path = await self._fs.store_result(
                doc_id,
                vision=vision,
                classification=classification,
                metadata={"job_id": state.job_id},
            )
            file_ok = True
            details["file"] = path

        notify_ok = await notify_complete(
            state.job_id,
            doc_id,
            ProcessingStatus.COMPLETED,
            vision=vision,
            classification=classification,
        )
        details["notify"] = "sent" if notify_ok else "skip"

        result = MCPActionResult(
            document_id=doc_id,
            database_recorded=db_ok,
            file_stored=file_ok,
            api_called=api_ok,
            notification_sent=notify_ok,
            details=details,
        )
        return {
            "mcp_result": result,
            "current_step": "summary",
            "status": ProcessingStatus.IN_PROGRESS,
            "messages": state.messages + ["MCP done"],
        }
