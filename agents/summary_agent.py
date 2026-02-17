"""Summary Agent - AI-generated financial summary."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from utils.schemas import (
    ClassificationResult,
    IntakeResult,
    ProcessingStatus,
    SummaryResult,
    VisionResult,
    WorkflowState,
)


SUMMARY_PROMPT = """Generate a concise financial summary for this processed document.
Include: transaction category, amount (if any), brief description, and any accounting notes.
1-3 sentences. Professional tone."""


class SummaryAgent:
    """Generates AI financial summary from workflow results."""

    name = "summary"

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm:
            return self._llm
        from agents.orchestrator import _get_llm

        return _get_llm()

    async def process(self, state: WorkflowState) -> dict[str, Any]:
        doc_id = state.document_id
        intake = state.intake_result
        vision = state.vision_result
        classification = state.classification_result

        context_parts: list[str] = []
        if intake:
            context_parts.append(f"Document type: {intake.document_type.value}")
            if intake.raw_content:
                context_parts.append(f"Content preview: {intake.raw_content[:1500]}...")
        if vision:
            context_parts.append(f"Extracted: {vision.extracted_text[:1500]}...")
        if classification:
            context_parts.append(
                f"Classified: {classification.category.value} | Amount: {classification.amount} | "
                f"Description: {classification.description or 'N/A'}"
            )

        context = "\n".join(context_parts) or "No document content."
        user_prompt = state.user_prompt or ""
        if user_prompt:
            context = f"User request: {user_prompt}\n\n{context}"

        try:
            llm = self._get_llm()
            r = await llm.ainvoke([
                SystemMessage(content=SUMMARY_PROMPT),
                HumanMessage(content=context[:6000]),
            ])
            summary_text = r.content if hasattr(r, "content") else str(r)
        except Exception as e:
            summary_text = f"Summary unavailable: {e}"

        result = SummaryResult(
            document_id=doc_id,
            financial_summary=summary_text.strip(),
            user_prompt=user_prompt or None,
            metadata={"steps_completed": len(state.messages)},
        )
        return {
            "summary_result": result,
            "current_step": "end",
            "status": ProcessingStatus.COMPLETED,
            "messages": state.messages + ["Summary done"],
        }
