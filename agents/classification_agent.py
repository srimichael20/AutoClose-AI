"""Classification Agent - Single LLM call with embedding reuse."""

import json
import re
from typing import Any

from langchain_core.documents import Document

from utils.schemas import (
    ClassificationResult,
    ProcessingStatus,
    TransactionCategory,
    VisionResult,
    WorkflowState,
)
from vector_db import ChromaStore


# Lightweight: if Vision gave category_hint, use short confirm prompt
CLASSIFY_FULL = """Classify SMB transaction. JSON: {"category":"revenue|expense|asset|liability|equity","subcategory":str,"amount":float|null,"description":str,"confidence":0-1,"reasoning":str}
Content:
"""
CLASSIFY_HINT = """Confirm classification. Hint: {hint}. JSON: {"category":"...","subcategory":"...","amount":number|null,"description":"...","confidence":0-1,"reasoning":"brief"}
Content:
"""


class ClassificationAgent:
    """LLM classification + vector store with cached embeddings."""

    name = "classification"

    def __init__(self, vector_store: ChromaStore | None = None):
        self._vs = vector_store or ChromaStore()

    def _get_llm(self):
        from agents.orchestrator import _get_llm

        return _get_llm()

    async def process(self, state: WorkflowState) -> dict[str, Any]:
        doc_id = state.document_id
        context = state.context_for_next_agent

        vision: VisionResult | None = state.vision_result
        if vision:
            context = vision.extracted_text
            hint = vision.structured_data
        else:
            hint = {}

        similar = self._vs.search(context[:500] if context else doc_id, k=3)
        similar_txt = self._fmt_similar(similar)

        use_hint = hint.get("category_hint") and hint.get("amount") is not None
        prompt = (
            CLASSIFY_HINT.format(hint=json.dumps(hint)[:200])
            if use_hint
            else CLASSIFY_FULL
        )
        classification = await self._classify(
            document_id=doc_id,
            content=context or "",
            prompt=prompt,
            similar=similar_txt,
        )

        if context:
            try:
                self._vs.add(
                    doc_id,
                    content=context[:4000],
                    metadata={
                        "category": classification.category.value,
                        "amount": classification.amount,
                        "document_id": doc_id,
                    },
                )
                classification.embeddings_added = True
            except Exception:
                pass

        return {
            "classification_result": classification,
            "current_step": "mcp",
            "status": ProcessingStatus.IN_PROGRESS,
            "messages": state.messages + ["Classification done"],
        }

    def _fmt_similar(self, docs: list[Document]) -> str:
        parts = []
        for d in docs[:3]:
            m = d.metadata
            parts.append(f"cat={m.get('category','?')} amt={m.get('amount','?')} | {d.page_content[:200]}")
        return "\n".join(parts) if parts else ""

    async def _classify(
        self,
        document_id: str,
        content: str,
        prompt: str,
        similar: str,
    ) -> ClassificationResult:
        llm = self._get_llm()
        full = prompt + content[:3000] + "\n\nSimilar:\n" + similar[:500]
        try:
            from langchain_core.messages import HumanMessage

            r = await llm.ainvoke([HumanMessage(content=full)])
            raw = (r.content if hasattr(r, "content") else str(r)).strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
        except Exception:
            data = {"category": "unknown", "subcategory": None, "amount": None, "description": content[:150], "confidence": 0, "reasoning": "parse err"}

        cat_val = (data.get("category") or "unknown").lower()
        category = next((c for c in TransactionCategory if c.value == cat_val), TransactionCategory.UNKNOWN)

        amt = data.get("amount")
        if amt is not None and isinstance(amt, str):
            m = re.search(r"[\d,]+\.?\d*", str(amt).replace(",", ""))
            amt = float(m.group()) if m else None
        elif amt is not None:
            try:
                amt = float(amt)
            except (TypeError, ValueError):
                amt = None

        return ClassificationResult(
            document_id=document_id,
            category=category,
            subcategory=data.get("subcategory"),
            amount=amt,
            description=data.get("description"),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning"),
            embeddings_added=False,
        )
