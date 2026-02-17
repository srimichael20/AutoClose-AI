"""Intake Agent - Multi-modal document intake (text, PDF, images)."""

from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader

from autoclose.agents.base import BaseAgent
from autoclose.schemas import (
    DocumentType,
    IntakeResult,
    ProcessingStatus,
    WorkflowState,
)


class IntakeAgent(BaseAgent):
    """
    Handles multi-modal input: text, PDF, images.
    Extracts raw content from text/PDF. For images, marks requires_vision=True.
    NO rule-based logic - uses LLM-free extraction for structured formats.
    """

    name = "intake"

    async def process(self, state: WorkflowState) -> dict[str, Any]:
        """Process incoming document and produce IntakeResult."""
        document_id = state.document_id
        doc_type = state.document_type
        file_path = state.file_path
        raw_content = state.raw_content

        requires_vision = False
        extracted_content: str | None = None
        metadata: dict[str, Any] = {}

        if raw_content and doc_type == DocumentType.TEXT:
            extracted_content = raw_content
        elif file_path:
            path = Path(file_path)
            if not path.exists():
                return {
                    "status": ProcessingStatus.FAILED,
                    "error": f"File not found: {file_path}",
                    "messages": state.messages + [f"Intake failed: file not found"],
                }

            if doc_type == DocumentType.PDF:
                extracted_content, metadata = await self._extract_pdf(path)
                if not extracted_content and path.suffix.lower() == ".pdf":
                    requires_vision = True
                    metadata["fallback_vision"] = "PDF text extraction yielded empty"
            elif doc_type == DocumentType.IMAGE:
                requires_vision = True
                extracted_content = None
                metadata["requires_ocr"] = True
            else:
                extracted_content = await self._read_text_file(path)

        result = IntakeResult(
            document_id=document_id,
            document_type=doc_type,
            raw_content=extracted_content,
            file_path=file_path,
            metadata=metadata,
            requires_vision=requires_vision,
        )

        return {
            "intake_result": result,
            "current_step": "vision" if requires_vision else "classification",
            "context_for_next_agent": extracted_content or f"[Vision required for {doc_type.value}]",
            "status": ProcessingStatus.IN_PROGRESS,
            "messages": state.messages + [f"Intake completed: {doc_type.value}"],
        }

    async def _extract_pdf(self, path: Path) -> tuple[str, dict[str, Any]]:
        """Extract text from PDF using pdfplumber (layout-aware) with pypdf fallback."""
        text_parts: list[str] = []
        metadata: dict[str, Any] = {"pages": 0}

        try:
            with pdfplumber.open(path) as pdf:
                metadata["pages"] = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception:
            try:
                reader = PdfReader(path)
                metadata["pages"] = len(reader.pages)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            except Exception as e:
                return "", {"error": str(e)}

        return "\n\n".join(text_parts), metadata

    async def _read_text_file(self, path: Path) -> str:
        """Read plain text file."""
        return path.read_text(encoding="utf-8", errors="replace")
