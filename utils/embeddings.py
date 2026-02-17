"""
Shared embedding service with in-memory cache.
Reuses embeddings for identical content to minimize API/model calls.
"""

import hashlib
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings


class EmbeddingCache:
    """Cache embeddings by content hash to avoid redundant computation."""

    def __init__(self, embedder: Embeddings, max_size: int = 1000):
        self._embedder = embedder
        self._cache: dict[str, list[float]] = {}
        self._max_size = max_size
        self._keys: list[str] = []

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def embed(self, text: str) -> list[float]:
        """Get embedding, from cache if available."""
        key = self._hash(text)
        if key in self._cache:
            return self._cache[key]
        vec = self._embedder.embed_query(text[:8000])
        if len(self._cache) >= self._max_size and self._keys:
            oldest = self._keys.pop(0)
            self._cache.pop(oldest, None)
        self._cache[key] = vec
        self._keys.append(key)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch, using cache where possible."""
        indexed: list[tuple[int, list[float]]] = []
        to_compute: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            key = self._hash(t[:8000])
            if key in self._cache:
                indexed.append((i, self._cache[key]))
            else:
                to_compute.append((i, t))
        if to_compute:
            batch_texts = [t for _, t in to_compute]
            batch_vecs = self._embedder.embed_documents(batch_texts)
            for (idx, txt), vec in zip(to_compute, batch_vecs):
                self._cache[self._hash(txt[:8000])] = vec
                indexed.append((idx, vec))
        indexed.sort(key=lambda x: x[0])
        return [v for _, v in indexed]


def get_embedder() -> Embeddings:
    """Lazy-load sentence-transformers embedder (local, no API calls)."""
    from utils.config import get_settings

    class _STEmbeddings(Embeddings):
        def __init__(self):
            from sentence_transformers import SentenceTransformer

            settings = get_settings()
            self._model = SentenceTransformer(settings.embedding_model)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._model.encode(texts, convert_to_numpy=True).tolist()

        def embed_query(self, text: str) -> list[float]:
            return self._model.encode(text, convert_to_numpy=True).tolist()

    return _STEmbeddings()


def get_cached_embedder() -> Embeddings:
    """Embedder that uses cache to avoid redundant computations."""
    cache = get_embedding_cache()

    class _CachedEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return cache.embed_batch(texts)

        def embed_query(self, text: str) -> list[float]:
            return cache.embed(text)

    return _CachedEmbeddings()


@lru_cache
def get_embedding_cache() -> EmbeddingCache:
    return EmbeddingCache(get_embedder(), max_size=500)
