import asyncio

import chromadb

from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K
from src.providers.base import EmbeddingProvider


class Retriever:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self._embedder = embedding_provider
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        self._collection = client.get_or_create_collection(COLLECTION_NAME)

    async def retrieve(self, query: str, top_k: int = TOP_K) -> list[str]:
        query_embedding = await self._embedder.embed_query(query)
        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents"],
        )
        docs: list[str] = results["documents"][0] if results["documents"] else []
        return docs
