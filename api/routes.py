"""API routes."""

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from agents.orchestrator import run_workflow
from utils.file_storage import FileStorage
from utils.schemas import (
    DocumentType,
    ProcessingStatus,
    WorkflowRequest,
    WorkflowResponse,
    IntakeResult,
    VisionResult,
    ClassificationResult,
    MCPActionResult,
)

router = APIRouter()


def _doc_type(ext: str) -> DocumentType:
    if ext in {".pdf"}:
        return DocumentType.PDF
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}:
        return DocumentType.IMAGE
    return DocumentType.TEXT


def _state_to_response(state: dict[str, Any]) -> WorkflowResponse:
    def opt(k, ctor):
        v = state.get(k)
        return ctor(**v) if v and isinstance(v, dict) else None

    return WorkflowResponse(
        job_id=state.get("job_id", ""),
        document_id=state.get("document_id", ""),
        status=ProcessingStatus(state.get("status", "failed")),
        intake_result=opt("intake_result", IntakeResult),
        vision_result=opt("vision_result", VisionResult),
        classification_result=opt("classification_result", ClassificationResult),
        mcp_result=opt("mcp_result", MCPActionResult),
        error=state.get("error"),
    )


@router.post("/workflow/submit", response_model=WorkflowResponse)
async def submit(request: WorkflowRequest) -> WorkflowResponse:
    doc_id = request.document_id or str(uuid.uuid4())
    if not request.content and not request.file_path:
        raise HTTPException(400, "content or file_path required")
    if request.file_path and not Path(request.file_path).exists():
        raise HTTPException(400, f"File not found: {request.file_path}")

    try:
        result = await run_workflow(
            document_id=doc_id,
            document_type=request.document_type,
            file_path=request.file_path,
            raw_content=request.content,
        )
    except Exception as e:
        return WorkflowResponse(
            job_id=str(uuid.uuid4()),
            document_id=doc_id,
            status=ProcessingStatus.FAILED,
            error=str(e),
        )
    return _state_to_response(result)


@router.post("/workflow/upload", response_model=WorkflowResponse)
async def upload(
    file: UploadFile = File(...),
    document_id: str | None = None,
) -> WorkflowResponse:
    doc_id = document_id or str(uuid.uuid4())
    ext = (Path(file.filename or "").suffix or ".txt").lower()
    doc_type = _doc_type(ext)

    fs = FileStorage()
    content = await file.read()
    stored = await fs.store_upload(doc_id, content, doc_type)

    try:
        result = await run_workflow(
            document_id=doc_id,
            document_type=doc_type,
            file_path=stored,
            raw_content=None,
        )
    except Exception as e:
        return WorkflowResponse(
            job_id=str(uuid.uuid4()),
            document_id=doc_id,
            status=ProcessingStatus.FAILED,
            error=str(e),
        )
    return _state_to_response(result)
