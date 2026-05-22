"""
Run this script once to build (or rebuild) the vector index from knowledge_base.md.
Re-run whenever knowledge_base.md is updated.

Usage:
    python index_knowledge_base.py
"""

import asyncio

from src.providers.openai_provider import OpenAIEmbeddingProvider
from src.indexer import build_index

if __name__ == "__main__":
    async def main() -> None:
        embedder = OpenAIEmbeddingProvider()
        await build_index(embedder)
        print("Done. Run `uvicorn api:app --reload` to start the server.")

    asyncio.run(main())
