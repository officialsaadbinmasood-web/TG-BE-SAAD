import asyncio
import time
import uuid

import chromadb

from src.config import CHROMA_PERSIST_DIR
from src.providers.base import EmbeddingProvider

CACHE_COLLECTION = "tg_query_cache"
SIMILARITY_THRESHOLD = 0.92
CACHE_MAX_ENTRIES = 500
CACHE_TTL_SECONDS = 7 * 24 * 3600


class SemanticCache:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self._embedder = embedding_provider
        self._client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        self._col = self._client.get_or_create_collection(
            name=CACHE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    async def _ensure_collection(self) -> None:
        """Recreate the collection reference if it was deleted while the server is running."""
        try:
            await asyncio.to_thread(self._col.count)
        except Exception:
            self._col = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=CACHE_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )

    async def get(self, query: str) -> str | None:
        await self._ensure_collection()
        count = await asyncio.to_thread(self._col.count)
        if count == 0:
            return None

        embedding = await self._embedder.embed_query(query)
        results = await asyncio.to_thread(
            self._col.query,
            query_embeddings=[embedding],
            n_results=1,
            include=["metadatas", "distances"],
        )

        distances = results["distances"][0]
        if not distances:
            return None

        similarity = 1.0 - distances[0]
        if similarity < SIMILARITY_THRESHOLD:
            return None

        meta = results["metadatas"][0][0]
        cached_at = meta.get("cached_at", 0)
        if time.time() - cached_at > CACHE_TTL_SECONDS:
            await asyncio.to_thread(self._col.delete, ids=[results["ids"][0][0]])
            return None

        return meta["answer"]

    async def set(self, query: str, answer: str) -> None:
        await self._ensure_collection()
        await self._evict_if_full()
        embedding = await self._embedder.embed_query(query)
        await asyncio.to_thread(
            self._col.add,
            ids=[str(uuid.uuid4())],
            documents=[query],
            embeddings=[embedding],
            metadatas=[{"answer": answer, "cached_at": time.time()}],
        )

    async def flush(self) -> None:
        await asyncio.to_thread(self._client.delete_collection, CACHE_COLLECTION)
        self._col = await asyncio.to_thread(
            self._client.create_collection,
            name=CACHE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    async def _evict_if_full(self) -> None:
        count = await asyncio.to_thread(self._col.count)
        if count < CACHE_MAX_ENTRIES:
            return
        all_items = await asyncio.to_thread(
            self._col.get, include=["metadatas", "ids"]
        )
        if not all_items["ids"]:
            return
        oldest_id = min(
            zip(all_items["ids"], all_items["metadatas"]),
            key=lambda x: x[1].get("cached_at", 0),
        )[0]
        await asyncio.to_thread(self._col.delete, ids=[oldest_id])
