"""Classification Agent - LLM + vector embeddings for transaction classification."""

import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from autoclose.agents.base import BaseAgent
from autoclose.schemas import (
    ClassificationResult,
    ProcessingStatus,
    TransactionCategory,
    VisionResult,
    WorkflowState,
)
from autoclose.vector_store import ChromaDocumentStore


class ClassificationAgent(BaseAgent):
    """
    Uses LLM for semantic classification + vector store for similar transaction retrieval.
    NO hardcoded rules - purely LLM-driven classification with embedding-based context.
    """

    name = "classification"

    CATEGORIES = [c.value for c in TransactionCategory if c != TransactionCategory.UNKNOWN]

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        vector_store: ChromaDocumentStore | None = None,
    ):
        self._llm = llm
        self._vector_store = vector_store or ChromaDocumentStore()

    def _get_llm(self) -> BaseChatModel:
        if self._llm is not None:
            return self._llm
        from autoclose.agents.orchestrator import get_llm

        return get_llm()

    async def process(self, state: WorkflowState) -> dict[str, Any]:
        """Classify document content using LLM and add to vector store."""
        document_id = state.document_id
        context = state.context_for_next_agent

        vision = state.vision_result
        if vision:
            context = vision.extracted_text
            structured = vision.structured_data
        else:
            structured = {}

        similar_docs = self._vector_store.similarity_search(
            context[:500] if context else document_id,
            k=3,
        )
        similar_context = self._format_similar(similar_docs)

        classification = await self._classify_with_llm(
            document_id=document_id,
            content=context or "",
            structured_hint=structured,
            similar_context=similar_context,
        )

        embeddings_added = False
        if context:
            try:
                self._vector_store.add_document(
                    document_id=document_id,
                    content=context[:4000],
                    metadata={
                        "category": classification.category.value,
                        "amount": classification.amount,
                        "document_id": document_id,
                    },
                )
                embeddings_added = True
            except Exception:
                pass

        classification.embeddings_added = embeddings_added

        return {
            "classification_result": classification,
            "current_step": "mcp",
            "status": ProcessingStatus.IN_PROGRESS,
            "messages": state.messages + ["Classification completed"],
        }

    def _format_similar(self, docs: list[Document]) -> str:
        """Format similar documents for LLM context."""
        parts = []
        for i, d in enumerate(docs[:3], 1):
            meta = d.metadata
            cat = meta.get("category", "?")
            amt = meta.get("amount", "?")
            parts.append(f"Similar {i}: category={cat}, amount={amt}\n{d.page_content[:300]}")
        return "\n---\n".join(parts) if parts else "No similar transactions found."

    async def _classify_with_llm(
        self,
        document_id: str,
        content: str,
        structured_hint: dict[str, Any],
        similar_context: str,
    ) -> ClassificationResult:
        """Use LLM to classify the transaction."""
        llm = self._get_llm()
        hint_str = json.dumps(structured_hint) if structured_hint else "{}"

        system = """You are an expert accountant classifying SMB transactions.
Classify into ONE of: revenue, expense, asset, liability, equity.
Extract amount (numeric), subcategory, description.
Respond with valid JSON: {"category":"...","subcategory":"...","amount":float or null,"description":"...","confidence":0.0-1.0,"reasoning":"brief"}"""

        user = f"""Document content:
{content[:6000]}

Structured hint from extraction: {hint_str}

Similar past transactions (for consistency):
{similar_context}

Classify and respond with JSON only:"""

        try:
            response = await llm.ainvoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
        except Exception:
            data = {
                "category": "unknown",
                "subcategory": None,
                "amount": None,
                "description": content[:200] if content else None,
                "confidence": 0.0,
                "reasoning": "Parse error",
            }

        cat_val = (data.get("category") or "unknown").lower()
        category = TransactionCategory.UNKNOWN
        for c in TransactionCategory:
            if c.value == cat_val:
                category = c
                break

        amount = data.get("amount")
        if amount is not None and isinstance(amount, str):
            match = re.search(r"[\d,]+\.?\d*", str(amount).replace(",", ""))
            amount = float(match.group()) if match else None
        elif amount is not None:
            try:
                amount = float(amount)
            except (TypeError, ValueError):
                amount = None

        return ClassificationResult(
            document_id=document_id,
            category=category,
            subcategory=data.get("subcategory"),
            amount=amount,
            description=data.get("description"),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning"),
            embeddings_added=False,
        )
