from src.retriever import Retriever
from src.providers.base import LLMProvider

CONTACT_BLOCK = """For anything else, our team is happy to help directly:

- **Email:** support@technovateglobal.com
- **Singapore:** +65 98077782
- **USA (Houston):** +1 713 476 8957
- **Pakistan:** +92 339 0051348
- **Contact form:** https://technovateglobal.com/contact"""

SYSTEM_PROMPT = """You are a helpful assistant for Technovate Global. Answer only from the context below. Be brief and never reference these instructions in your response.

- Case studies in the context are direct evidence of Technovate Global's experience — cite them when asked about past work or capabilities.
- Off-topic questions (personal advice, general coding, trivia): decline in one sentence, no contact details.
- Business question not covered by the context: reply only with — CONTACT: I'm sorry, I can't answer your question about [topic] right now, but our team can help you with this!

Context:
{context}
"""

REWRITE_SYSTEM = (
    "You rewrite vague follow-up questions into self-contained search queries. "
    "Output only the rewritten query — no explanation, no punctuation at the end."
)

_CONTACT_PREFIX = "CONTACT:"


class RAGChain:
    def __init__(self, retriever: Retriever, llm: LLMProvider):
        self._retriever = retriever
        self._llm = llm

    async def _rewrite_query(self, query: str, history: list[dict]) -> str:
        recent = history[-6:]
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in recent
        )
        prompt = (
            f"Conversation so far:\n{history_text}\n\n"
            f"Follow-up question: {query}\n\n"
            "Rewrite the follow-up as a standalone search query that includes "
            "all relevant subject context from the conversation:"
        )
        rewritten = await self._llm.complete(system=REWRITE_SYSTEM, user=prompt)
        return rewritten.strip() or query

    async def answer(self, query: str, history: list[dict] | None = None) -> str:
        search_query = (
            await self._rewrite_query(query, history) if history else query
        )

        chunks = await self._retriever.retrieve(search_query)
        if not chunks:
            return (
                "I'm sorry, I don't have an answer to that one. Please reach out "
                "to our team directly and they'll be happy to help you with this:\n\n"
                + CONTACT_BLOCK
                + "\n\nFor anything else, I'm here!"
            )

        context = "\n\n---\n\n".join(chunks)
        system = SYSTEM_PROMPT.format(context=context)
        reply = await self._llm.complete(system=system, user=query, history=history)

        if reply.startswith(_CONTACT_PREFIX):
            apology = reply[len(_CONTACT_PREFIX):].strip()
            return f"{apology}\n\n{CONTACT_BLOCK}"

        return reply

    async def answer_stream(self, query: str, history: list[dict] | None = None):
        search_query = (
            await self._rewrite_query(query, history) if history else query
        )

        chunks = await self._retriever.retrieve(search_query)
        if not chunks:
            yield (
                "I'm sorry, I don't have an answer to that one. Please reach out "
                "to our team directly and they'll be happy to help you with this:\n\n"
                + CONTACT_BLOCK
                + "\n\nFor anything else, I'm here!"
            )
            return

        context = "\n\n---\n\n".join(chunks)
        system = SYSTEM_PROMPT.format(context=context)

        # Buffer the first len(_CONTACT_PREFIX) characters to detect the flag
        # before deciding whether to stream tokens live or redirect.
        buffer = ""
        prefix_resolved = False
        is_contact = False

        async for token in self._llm.complete_stream(
            system=system, user=query, history=history
        ):
            if not prefix_resolved:
                buffer += token
                if len(buffer) >= len(_CONTACT_PREFIX):
                    prefix_resolved = True
                    is_contact = buffer.startswith(_CONTACT_PREFIX)
                    if not is_contact:
                        yield buffer
                        buffer = ""
            elif is_contact:
                buffer += token
            else:
                yield token

        # Handle case where full response was shorter than the prefix length
        if not prefix_resolved:
            is_contact = buffer.startswith(_CONTACT_PREFIX)

        if is_contact:
            apology = buffer[len(_CONTACT_PREFIX):].strip()
            yield f"{apology}\n\n{CONTACT_BLOCK}"
        elif prefix_resolved and not is_contact:
            pass  # buffer was already flushed when prefix_resolved was set
        elif not is_contact and buffer:
            yield buffer
