import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Literal


from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from pythonjsonlogger import jsonlogger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src"))

from providers.openai_provider import OpenAIEmbeddingProvider, OpenAILLMProvider
from retriever import Retriever
from rag_chain import RAGChain
from cache import SemanticCache
from config import ALLOWED_ORIGINS, TRUSTED_PROXIES

_log_handler = logging.StreamHandler()
_log_handler.setFormatter(
    jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
)
logger = logging.getLogger("chatbot")
logger.addHandler(_log_handler)
logger.setLevel(logging.INFO)


def _truncate_ip(ip: str) -> str:
    parts = ip.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.*.*"
    return "ipv6"


MAX_MESSAGE_CHARS = 2_000
MAX_HISTORY_TURNS = 10


def _real_ip(request: Request) -> str:
    direct_ip = get_remote_address(request)
    if direct_ip in TRUSTED_PROXIES:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    return direct_ip


limiter = Limiter(key_func=_real_ip, default_limits=["20/minute"])

app = FastAPI(title="Technovate Global Chatbot")

@app.on_event("startup")
async def startup_event():
    from src.indexer import build_index
    await build_index(_embedder)
    
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        return response


app.add_middleware(SecurityHeadersMiddleware)

FALLBACK_RESPONSE = (
    "Sorry, I'm having trouble right now. Please contact our team directly:\n\n"
    "- **Email:** support@technovateglobal.com\n"
    "- **Singapore:** +65 98077782\n"
    "- **USA (Houston):** +1 713 476 8957\n"
    "- **Pakistan:** +92 339 0051348\n"
    "- **Contact form:** https://technovateglobal.com/contact"
)

_embedder = OpenAIEmbeddingProvider()
_llm = OpenAILLMProvider()
_cache = SemanticCache(_embedder)
_retriever = None
_chain = None

@app.on_event("startup")
async def startup_event():
    global _retriever, _chain
    from src.indexer import build_index
    await build_index(_embedder)
    _retriever = Retriever(_embedder)
    _chain = RAGChain(_retriever, _llm)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[HistoryMessage] = Field(default=[], max_length=MAX_HISTORY_TURNS)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be blank")
        return v.strip()


class ChatResponse(BaseModel):
    reply: str
    cached: bool = False


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    request_id = str(uuid.uuid4())
    ip = _truncate_ip(_real_ip(request))
    t0 = time.perf_counter()

    query = req.message
    is_followup = any(m.role == "user" for m in req.history)

    if not is_followup:
        cached_answer = await _cache.get(query)
        if cached_answer:
            logger.info("chat", extra={
                "request_id": request_id,
                "path": "/chat",
                "cache_hit": True,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "ip": ip,
            })
            return ChatResponse(reply=cached_answer, cached=True)

    history = (
        [{"role": m.role, "content": m.content} for m in req.history]
        if is_followup else None
    )

    try:
        reply = await _chain.answer(query, history=history)
        if not is_followup:
            await _cache.set(query, reply)
    except Exception:
        logger.error("chat_error", extra={
            "request_id": request_id,
            "path": "/chat",
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "ip": ip,
        })
        return ChatResponse(reply=FALLBACK_RESPONSE)

    logger.info("chat", extra={
        "request_id": request_id,
        "path": "/chat",
        "cache_hit": False,
        "latency_ms": round((time.perf_counter() - t0) * 1000),
        "ip": ip,
    })
    return ChatResponse(reply=reply, cached=False)


@app.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    request_id = str(uuid.uuid4())
    ip = _truncate_ip(_real_ip(request))
    t0 = time.perf_counter()

    query = req.message
    is_followup = any(m.role == "user" for m in req.history)
    history = (
        [{"role": m.role, "content": m.content} for m in req.history]
        if is_followup else None
    )

    async def generate():
        if not is_followup:
            cached = await _cache.get(query)
            if cached:
                logger.info("chat_stream", extra={
                    "request_id": request_id,
                    "path": "/chat/stream",
                    "cache_hit": True,
                    "latency_ms": round((time.perf_counter() - t0) * 1000),
                    "ip": ip,
                })
                yield f"data: {json.dumps(cached)}\n\n"
                yield "data: [DONE]\n\n"
                return

        full_reply = ""
        try:
            async for token in _chain.answer_stream(query, history=history):
                full_reply += token
                yield f"data: {json.dumps(token)}\n\n"
        except Exception:
            logger.error("chat_stream_error", extra={
                "request_id": request_id,
                "path": "/chat/stream",
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "ip": ip,
            })
            yield f"data: {json.dumps(FALLBACK_RESPONSE)}\n\n"
            yield "data: [DONE]\n\n"
            return

        yield "data: [DONE]\n\n"

        logger.info("chat_stream", extra={
            "request_id": request_id,
            "path": "/chat/stream",
            "cache_hit": False,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "ip": ip,
        })

        if not is_followup and full_reply:
            await _cache.set(query, full_reply)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/cache/clear")
async def clear_cache(
    request: Request,
    authorization: str = Header(..., alias="Authorization"),
) -> dict:
    expected = f"Bearer {os.environ.get('CACHE_CLEAR_SECRET', '')}"
    if not expected or authorization != expected:
        raise HTTPException(status_code=403, detail="forbidden")
    await _cache.flush()
    return {"status": "cache cleared"}
