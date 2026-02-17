"""Vision Agent - OCR + lightweight LLM extraction."""

import json
from pathlib import Path
from typing import Any

from utils.schemas import IntakeResult, ProcessingStatus, VisionResult, WorkflowState


# Lightweight prompt - minimal tokens
EXTRACT_PROMPT = """Extract accounting data. JSON only: {"amount":number|null,"date":str|null,"vendor":str|null,"description":str|null,"category_hint":str|null}
Text:
"""


class VisionAgent:
    """OCR for images/PDF + single LLM call for structured extraction."""

    name = "vision"

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm:
            return self._llm
        from agents.orchestrator import _get_llm

        return _get_llm()

    async def process(self, state: WorkflowState) -> dict[str, Any]:
        intake: IntakeResult | None = state.intake_result
        if not intake:
            return {"status": ProcessingStatus.FAILED, "error": "No intake"}

        doc_id = intake.document_id
        text = intake.raw_content or ""
        fp = intake.file_path

        if intake.requires_vision and fp:
            text, conf = await self._ocr(Path(fp))
        else:
            conf = 1.0

        if not text:
            return {"status": ProcessingStatus.FAILED, "error": "No content"}

        structured = await self._extract(text)

        result = VisionResult(
            document_id=doc_id,
            extracted_text=text,
            structured_data=structured,
            confidence_score=conf,
            metadata={"requires_vision": intake.requires_vision},
        )
        return {
            "vision_result": result,
            "current_step": "classification",
            "context_for_next_agent": text,
            "status": ProcessingStatus.IN_PROGRESS,
            "messages": state.messages + ["Vision done"],
        }

    async def _ocr(self, path: Path) -> tuple[str, float]:
        try:
            import easyocr

            reader = easyocr.Reader(["en"], gpu=False)
            suff = path.suffix.lower()
            if suff in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}:
                res = reader.readtext(str(path))
            else:
                try:
                    from pdf2image import convert_from_path

                    imgs = convert_from_path(path, first_page=1, last_page=1)
                    if not imgs:
                        return "", 0.0
                    tmp = path.parent / f"{path.stem}_p0.png"
                    imgs[0].save(tmp)
                    res = reader.readtext(str(tmp))
                    tmp.unlink(missing_ok=True)
                except Exception:
                    return "", 0.0
            lines = [r[1] for r in res]
            confs = [r[2] for r in res]
            avg = sum(confs) / len(confs) if confs else 0.0
            return "\n".join(lines), avg
        except Exception:
            return "", 0.0

    async def _extract(self, text: str) -> dict[str, Any]:
        llm = self._get_llm()
        prompt = EXTRACT_PROMPT + text[:4000]
        try:
            from langchain_core.messages import HumanMessage

            r = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = (r.content if hasattr(r, "content") else str(r)).strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception:
            return {"raw": text[:300]}
