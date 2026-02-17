"""
Chroma vector store with embedding cache reuse.
Minimizes redundant embedding computations.
"""

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.config import get_settings
from utils.embeddings import get_cached_embedder


class ChromaStore:
    """Chroma vector store using cached embeddings."""

    def __init__(
        self,
        persist_directory: str | None = None,
    ) -> None:
        settings = get_settings()
        self.persist_dir = persist_directory or settings.chroma_persist_directory
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._collection: Chroma | None = None

    def _get_collection(self) -> Chroma:
        if self._collection is None:
            self._collection = Chroma(
                collection_name="autoclose_docs",
                embedding_function=get_cached_embedder(),
                persist_directory=self.persist_dir,
            )
        return self._collection

    def add(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add document. Uses cached embedding when content was recently embedded."""
        try:
            doc = Document(
                page_content=content[:4000],
                metadata={**(metadata or {}), "document_id": document_id},
            )
            self._get_collection().add_documents(
                documents=[doc],
                ids=[document_id],
                metadatas=[doc.metadata],
            )
            return True
        except Exception:
            return False

    def search(self, query: str, k: int = 3) -> list[Document]:
        """Semantic search. Query embedding uses cache."""
        return self._get_collection().similarity_search(query=query[:8000], k=k)
