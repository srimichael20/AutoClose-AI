"""API routes for workflow submission and status."""

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from autoclose.agents.orchestrator import run_workflow
from autoclose.mcp import FileSystemService
from autoclose.schemas import DocumentType, ProcessingStatus, WorkflowRequest, WorkflowResponse

router = APIRouter()


def _doc_type_from_filename(filename: str) -> DocumentType:
    """Infer document type from file extension."""
    ext = (Path(filename).suffix or "").lower()
    if ext in {".pdf"}:
        return DocumentType.PDF
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}:
        return DocumentType.IMAGE
    return DocumentType.TEXT


@router.post("/workflow/submit", response_model=WorkflowResponse)
async def submit_workflow(request: WorkflowRequest) -> WorkflowResponse:
    """
    Submit a document for processing via the multi-agent workflow.
    Use content for inline text, or file_path for server-accessible files.
    """
    document_id = request.document_id or str(uuid.uuid4())
    doc_type = request.document_type

    if not request.content and not request.file_path:
        raise HTTPException(
            status_code=400,
            detail="Either content or file_path must be provided",
        )

    if request.file_path and not Path(request.file_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"File not found: {request.file_path}",
        )

    try:
        result = await run_workflow(
            document_id=document_id,
            document_type=doc_type,
            file_path=request.file_path,
            raw_content=request.content,
        )
    except Exception as e:
        return WorkflowResponse(
            job_id=str(uuid.uuid4()),
            document_id=document_id,
            status=ProcessingStatus.FAILED,
            error=str(e),
        )

    return _state_to_response(result)


@router.post("/workflow/upload", response_model=WorkflowResponse)
async def upload_and_process(
    file: UploadFile = File(...),
    document_id: str | None = None,
) -> WorkflowResponse:
    """
    Upload a file (PDF, image, text) and process through the multi-agent workflow.
    """
    doc_id = document_id or str(uuid.uuid4())
    doc_type = _doc_type_from_filename(file.filename or "unknown.txt")

    fs = FileSystemService()

    content = await file.read()
    stored_path = await fs.store_upload(
        document_id=doc_id,
        content=content,
        document_type=doc_type,
        original_filename=file.filename,
    )

    try:
        result = await run_workflow(
            document_id=doc_id,
            document_type=doc_type,
            file_path=stored_path,
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


def _state_to_response(state: dict[str, Any]) -> WorkflowResponse:
    """Convert workflow state dict to WorkflowResponse."""
    from autoclose.schemas import (
        ClassificationResult,
        IntakeResult,
        MCPActionResult,
        VisionResult,
    )

    intake = state.get("intake_result")
    if intake and isinstance(intake, dict):
        intake = IntakeResult(**intake)

    vision = state.get("vision_result")
    if vision and isinstance(vision, dict):
        vision = VisionResult(**vision)

    classification = state.get("classification_result")
    if classification and isinstance(classification, dict):
        classification = ClassificationResult(**classification)

    mcp = state.get("mcp_result")
    if mcp and isinstance(mcp, dict):
        mcp = MCPActionResult(**mcp)

    return WorkflowResponse(
        job_id=state.get("job_id", ""),
        document_id=state.get("document_id", ""),
        status=ProcessingStatus(state.get("status", "failed")),
        intake_result=intake,
        vision_result=vision,
        classification_result=classification,
        mcp_result=mcp,
        error=state.get("error"),
    )
