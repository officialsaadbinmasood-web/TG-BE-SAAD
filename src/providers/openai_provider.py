from typing import AsyncIterator

from openai import AsyncOpenAI

from .base import EmbeddingProvider, LLMProvider
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = EMBEDDING_MODEL):
        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY, max_retries=3)
        self._model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]


class OpenAILLMProvider(LLMProvider):
    def __init__(self, model: str = CHAT_MODEL):
        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY, max_retries=3)
        self._model = model

    def _build_messages(
        self, system: str, user: str, history: list[dict] | None
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        return messages

    async def complete(
        self,
        system: str,
        user: str,
        history: list[dict] | None = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(system, user, history),
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""

    async def complete_stream(
        self,
        system: str,
        user: str,
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(system, user, history),
            temperature=0.3,
            max_tokens=512,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
