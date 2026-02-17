"""Chroma-based vector store for document embeddings and semantic retrieval."""

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from autoclose.config import get_settings


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain Embeddings wrapper for SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()


class ChromaDocumentStore:
    """Chroma vector store for accounting document embeddings and retrieval."""

    def __init__(
        self,
        persist_directory: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self._embedding_model_name = embedding_model or settings.embedding_model
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._collection: Chroma | None = None

    def _get_embeddings_wrapper(self) -> Embeddings:
        """Create LangChain-compatible embeddings wrapper."""
        return SentenceTransformerEmbeddings(model_name=self._embedding_model_name)

    def _get_collection(self) -> Chroma:
        """Get or create Chroma collection."""
        if self._collection is None:
            embeddings = self._get_embeddings_wrapper()
            self._collection = Chroma(
                collection_name="autoclose_accounting",
                embedding_function=embeddings,
                persist_directory=self.persist_directory,
            )
        return self._collection

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add documents to the vector store."""
        collection = self._get_collection()
        return collection.add_documents(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )

    def add_document(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a single document with content and metadata."""
        doc = Document(
            page_content=content,
            metadata=metadata or {},
        )
        ids = self.add_documents(
            documents=[doc],
            ids=[document_id],
            metadatas=[{**(metadata or {}), "document_id": document_id}],
        )
        return ids[0] if ids else ""

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Retrieve similar documents by semantic search."""
        collection = self._get_collection()
        return collection.similarity_search(
            query=query,
            k=k,
            filter=filter_metadata,
        )

    def get_collection(self) -> Chroma:
        """Expose collection for LangChain integration."""
        return self._get_collection()
