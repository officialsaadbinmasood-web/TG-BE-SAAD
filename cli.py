"""
Interactive CLI to test the chatbot locally without running the server.
Usage: python cli.py
"""

import asyncio

from src.providers.openai_provider import OpenAIEmbeddingProvider, OpenAILLMProvider
from src.retriever import Retriever
from src.rag_chain import RAGChain


async def main() -> None:
    embedder = OpenAIEmbeddingProvider()
    llm = OpenAILLMProvider()
    retriever = Retriever(embedder)
    chain = RAGChain(retriever, llm)

    print("Technovate Global Chatbot — type 'exit' to quit\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        reply = await chain.answer(query)
        print(f"\nBot: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
