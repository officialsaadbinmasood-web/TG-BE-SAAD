from abc import ABC, abstractmethod
from typing import AsyncIterator


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        history: list[dict] | None = None,
    ) -> str: ...

    @abstractmethod
    async def complete_stream(
        self,
        system: str,
        user: str,
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]: ...
