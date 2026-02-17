"""Intake Agent - Multi-modal document intake (text, PDF, images)."""

from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader

from utils.schemas import DocumentType, IntakeResult, ProcessingStatus, WorkflowState


class IntakeAgent:
    """Handles text, PDF, and image intake. No LLM calls."""

    name = "intake"

    async def process(self, state: WorkflowState) -> dict[str, Any]:
        document_id = state.document_id
        doc_type = state.document_type
        file_path = state.file_path
        raw_content = state.raw_content

        requires_vision = False
        extracted: str | None = None
        metadata: dict[str, Any] = {}

        if raw_content and doc_type == DocumentType.TEXT:
            extracted = raw_content
        elif file_path:
            path = Path(file_path)
            if not path.exists():
                return {
                    "status": ProcessingStatus.FAILED,
                    "error": f"File not found: {file_path}",
                    "messages": state.messages + ["Intake failed"],
                }
            if doc_type == DocumentType.PDF:
                extracted, metadata = await self._extract_pdf(path)
                if not extracted:
                    requires_vision = True
                    metadata["fallback_vision"] = True
            elif doc_type == DocumentType.IMAGE:
                requires_vision = True
                metadata["requires_ocr"] = True
            else:
                extracted = path.read_text(encoding="utf-8", errors="replace")

        result = IntakeResult(
            document_id=document_id,
            document_type=doc_type,
            raw_content=extracted,
            file_path=file_path,
            metadata=metadata,
            requires_vision=requires_vision,
        )
        return {
            "intake_result": result,
            "current_step": "vision" if requires_vision else "classification",
            "context_for_next_agent": extracted or f"[Vision required: {doc_type.value}]",
            "status": ProcessingStatus.IN_PROGRESS,
            "messages": state.messages + ["Intake done"],
        }

    async def _extract_pdf(self, path: Path) -> tuple[str, dict[str, Any]]:
        text_parts: list[str] = []
        meta: dict[str, Any] = {"pages": 0}
        try:
            with pdfplumber.open(path) as pdf:
                meta["pages"] = len(pdf.pages)
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
        except Exception:
            try:
                reader = PdfReader(path)
                meta["pages"] = len(reader.pages)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
            except Exception as e:
                return "", {"error": str(e)}
        return "\n\n".join(text_parts), meta
