"""Vision Agent - OCR and document context extraction."""

from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from autoclose.agents.base import BaseAgent
from autoclose.schemas import (
    IntakeResult,
    ProcessingStatus,
    VisionResult,
    WorkflowState,
)


class VisionAgent(BaseAgent):
    """
    OCR for images/PDFs + LLM-based document context extraction.
    Uses EasyOCR for OCR, then LLM to extract structured accounting-relevant data.
    """

    name = "vision"

    def __init__(self, llm: BaseChatModel | None = None):
        self._llm = llm

    def _get_llm(self) -> BaseChatModel:
        """Lazy-load LLM based on config."""
        if self._llm is not None:
            return self._llm
        from autoclose.agents.orchestrator import get_llm

        return get_llm()

    async def process(self, state: WorkflowState) -> dict[str, Any]:
        """Run OCR and LLM extraction on document."""
        intake = state.intake_result
        if not intake:
            return {
                "status": ProcessingStatus.FAILED,
                "error": "Intake result required for vision",
            }

        document_id = intake.document_id
        extracted_text = intake.raw_content or ""
        file_path = intake.file_path

        if intake.requires_vision and file_path:
            ocr_text, confidence = await self._run_ocr(Path(file_path))
            extracted_text = ocr_text or extracted_text
        elif not extracted_text:
            return {
                "status": ProcessingStatus.FAILED,
                "error": "No content available for vision processing",
            }

        structured_data = await self._extract_with_llm(extracted_text)

        result = VisionResult(
            document_id=document_id,
            extracted_text=extracted_text,
            structured_data=structured_data,
            confidence_score=confidence if intake.requires_vision else 1.0,
            metadata={"requires_vision": intake.requires_vision},
        )

        return {
            "vision_result": result,
            "current_step": "classification",
            "context_for_next_agent": extracted_text,
            "status": ProcessingStatus.IN_PROGRESS,
            "messages": state.messages + ["Vision extraction completed"],
        }

    async def _run_ocr(self, path: Path) -> tuple[str, float]:
        """Run EasyOCR on image or PDF page."""
        try:
            import easyocr

            reader = easyocr.Reader(["en"], gpu=False)

            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}:
                result = reader.readtext(str(path))
            else:
                try:
                    from pdf2image import convert_from_path

                    images = convert_from_path(path, first_page=1, last_page=1)
                    if not images:
                        return "", 0.0
                    img_path = path.parent / f"{path.stem}_page0.png"
                    images[0].save(img_path)
                    result = reader.readtext(str(img_path))
                    if img_path.exists():
                        img_path.unlink()
                except Exception:
                    return "", 0.0

            lines = [item[1] for item in result]
            confidences = [item[2] for item in result]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            return "\n".join(lines), avg_conf

        except Exception:
            return "", 0.0

    async def _extract_with_llm(self, text: str) -> dict[str, Any]:
        """Use LLM to extract structured accounting data from document text."""
        llm = self._get_llm()
        prompt = f"""Extract accounting-relevant structured data from this document text.
Return a JSON object with keys: amount, date, vendor_or_payee, description, category_hint, line_items (array of {{description, amount}}).
Use null for missing values. Be concise.

Document text:
{text[:8000]}

JSON only, no markdown:"""

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            content = content.strip().replace("```json", "").replace("```", "").strip()
            import json

            return json.loads(content)
        except Exception:
            return {"raw_text_preview": text[:500]}
