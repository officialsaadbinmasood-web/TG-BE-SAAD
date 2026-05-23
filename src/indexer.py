"""
Reads knowledge_base.md, splits it into overlapping chunks,
embeds them, and persists to ChromaDB.
Run once on deploy; re-run whenever knowledge_base.md changes.
"""

import re

import chromadb
import tiktoken

from src.config import (
    KNOWLEDGE_BASE_PATH,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.providers.base import EmbeddingProvider


_enc = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_enc.encode(text))


def _split_by_headings(text: str) -> list[str]:
    parts = re.split(r"(?=\n#{1,3} )", text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_paragraphs(section: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\n+", section) if p.strip()]


def chunk_document(text: str) -> list[str]:
    chunks: list[str] = []

    for section in _split_by_headings(text):
        if _token_len(section) <= CHUNK_SIZE:
            chunks.append(section)
            continue

        paragraphs = _split_by_paragraphs(section)
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = _token_len(para)

            if current_len + para_len > CHUNK_SIZE and current:
                chunks.append("\n\n".join(current))
                overlap: list[str] = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + _token_len(p) <= CHUNK_OVERLAP:
                        overlap.insert(0, p)
                        overlap_len += _token_len(p)
                    else:
                        break
                current = overlap
                current_len = overlap_len

            current.append(para)
            current_len += para_len

        if current:
            chunks.append("\n\n".join(current))

    return chunks


async def build_index(embedding_provider: EmbeddingProvider) -> None:
    text = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")
    chunks = chunk_document(text)
    print(f"[indexer] {len(chunks)} chunks from knowledge_base.md")

    embeddings = await embedding_provider.embed_texts(chunks)

    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"chunk_index": i} for i in range(len(chunks))],
    )

    print(f"[indexer] Index built — {len(chunks)} vectors stored in {CHROMA_PERSIST_DIR}")
