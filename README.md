# Technovate Global Chatbot — Backend

A Retrieval-Augmented Generation (RAG) chatbot API for [Technovate Global](https://technovateglobal.com). It answers questions strictly from a curated knowledge base using OpenAI embeddings and GPT-4o Mini, with a semantic cache and per-IP rate limiting.

---

## Architecture

```
User request
    │
    ▼
FastAPI (api.py)
    │  rate limit: 20 req/min per IP
    │  CORS + security headers
    ▼
SemanticCache ──── hit ──▶ cached reply
    │ miss
    ▼
RAGChain (rag_chain.py)
    │  1. rewrite follow-up queries (if history present)
    │  2. retrieve top-5 chunks from ChromaDB
    │  3. call GPT-4o Mini with context
    ▼
reply + cache store
```
test
**Key design choices:**
- Cache is only consulted/written for standalone (no-history) queries. Follow-ups bypass it to prevent context-dependent answers from polluting the cache.
- The query rewriter expands vague follow-ups (e.g. "what about the price?") into self-contained search queries using the last 6 conversation turns.
- Knowledge base is a single Markdown file (`knowledge_base.md`) chunked into ~400-token overlapping segments at index time.

---

## Project Structure

```
chatbot/
├── api.py                   # FastAPI app — endpoints, middleware, singletons
├── cli.py                   # Interactive CLI for local testing (no server needed)
├── index_knowledge_base.py  # One-shot indexing script
├── knowledge_base.md        # Source-of-truth documentation (edit this to update chatbot knowledge)
├── requirements.txt
├── chroma_db/               # Persisted ChromaDB vector store (auto-created)
└── src/
    ├── config.py            # Settings loaded from .env
    ├── indexer.py           # Markdown chunking + embedding + ChromaDB write
    ├── retriever.py         # ChromaDB similarity search
    ├── rag_chain.py         # RAG orchestration and query rewriting
    ├── cache.py             # Semantic cache (cosine similarity, 7-day TTL, 500-entry cap)
    └── providers/
        ├── base.py          # Abstract interfaces for embedding and LLM providers
        └── openai_provider.py  # OpenAI implementation (text-embedding-3-small + gpt-4o-mini)
```

---

## Prerequisites

- Python 3.12+
- An OpenAI API key

---

## Setup

**1. Create and activate a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Create a `.env` file** in the `chatbot/` directory:

```env
OPENAI_API_KEY=sk-...

# Comma-separated allowed origins for CORS (default: http://localhost:3000)
ALLOWED_ORIGINS=http://localhost:3000,https://technovateglobal.com

# Secret for the /cache/clear endpoint (choose any strong random string)
CACHE_CLEAR_SECRET=your-secret-here

# Comma-separated IPs allowed to set X-Forwarded-For (default: 127.0.0.1)
# See "Trusted Proxies" section below before changing this.
TRUSTED_PROXIES=127.0.0.1
```

**4. Build the vector index** (run once, and again whenever `knowledge_base.md` changes):

```bash
python index_knowledge_base.py
```

---

## Running the API

```bash
uvicorn api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

---

## API Reference

### `POST /chat`

Send a message and receive a reply grounded in the knowledge base.

**Request body:**

```json
{
  "message": "What services does Technovate Global offer?",
  "history": [
    { "role": "user",      "content": "Tell me about Recruiter+" },
    { "role": "assistant", "content": "Recruiter+ is ..." }
  ]
}
```

| Field     | Type   | Constraints                            |
|-----------|--------|----------------------------------------|
| `message` | string | 1–2 000 characters, non-blank          |
| `history` | array  | max 10 turns; roles: `user`/`assistant` |

**Response:**

```json
{
  "reply": "Technovate Global offers ...",
  "cached": false
}
```

`cached: true` means the reply was served from the semantic cache without calling the LLM.

**Rate limit:** 20 requests per minute per IP. Returns HTTP 429 when exceeded.

---

### `GET /health`

```json
{ "status": "ok" }
```

---

### `POST /cache/clear`

Wipes all semantic cache entries. Requires a bearer token.

```bash
curl -X POST http://localhost:8000/cache/clear \
  -H "Authorization: Bearer your-secret-here"
```

---

## Local CLI

Test the chatbot interactively without running the server:

```bash
python cli.py
```

Type your question and press Enter. Type `exit` or `quit` to stop.

---

## Updating the Knowledge Base

1. Edit `knowledge_base.md`.
2. Re-run the indexer:
   ```bash
   python index_knowledge_base.py
   ```
3. The old index is replaced automatically. Restart the API server if it's running.

---

## Deployment Notes

### Trusted Proxies & Rate Limiting

The API rate-limits by IP. When running behind a reverse proxy (Nginx, AWS ALB, Cloudflare), the direct connection IP is the proxy's IP, not the visitor's. Proxies set an `X-Forwarded-For` header with the real visitor IP — but blindly trusting that header lets anyone spoof their IP and bypass rate limiting.

`TRUSTED_PROXIES` is a whitelist of IPs whose `X-Forwarded-For` header the server will trust. Only set it to the IP(s) of your actual proxy.

| Deployment | `TRUSTED_PROXIES` value |
|---|---|
| Local dev (no proxy) | `127.0.0.1` (default) |
| Nginx on the same server | `127.0.0.1` |
| AWS ALB / Cloudflare | The ALB or Cloudflare egress IP range |
| Direct internet (no proxy) | Remove `X-Forwarded-For` handling entirely |

### Nginx — SSE Buffering

If serving the `/chat/stream` endpoint through Nginx, add this to your location block to prevent Nginx from buffering the streaming response:

```nginx
proxy_buffering off;
```

The `X-Accel-Buffering: no` response header handles this automatically for most setups, but an explicit config directive is more reliable.

### Before Going Live

1. Set `ALLOWED_ORIGINS` to your production domain only (e.g. `https://technovateglobal.com`)
2. Set `TRUSTED_PROXIES` to match your infrastructure
3. Use a strong random string for `CACHE_CLEAR_SECRET`
4. Never commit `.env` to version control — it is gitignored by default
5. Run `uvicorn api:app --host 0.0.0.0 --port 8000` (drop `--reload` in production)

---

## Configuration Reference

All settings live in `src/config.py` and are overridable via `.env`:

| Setting            | Default                  | Description                            |
|--------------------|--------------------------|----------------------------------------|
| `OPENAI_API_KEY`   | *(required)*             | OpenAI API key                         |
| `ALLOWED_ORIGINS`  | `http://localhost:3000`  | Comma-separated CORS-allowed origins   |
| `CACHE_CLEAR_SECRET` | *(required for endpoint)* | Bearer token for `/cache/clear`      |
| `TRUSTED_PROXIES`  | `127.0.0.1`              | IPs allowed to set `X-Forwarded-For` |
| `EMBEDDING_MODEL`  | `text-embedding-3-small` | OpenAI embedding model                 |
| `CHAT_MODEL`       | `gpt-4o-mini`            | OpenAI chat completion model           |
| `CHUNK_SIZE`       | 400 tokens               | Target chunk size for indexing         |
| `CHUNK_OVERLAP`    | 60 tokens                | Overlap between consecutive chunks    |
| `TOP_K`            | 5                        | Number of chunks retrieved per query   |

**Semantic cache tuning** (in `src/cache.py`):

| Constant               | Default   | Description                                          |
|------------------------|-----------|------------------------------------------------------|
| `SIMILARITY_THRESHOLD` | `0.92`    | Cosine similarity required for a cache hit (0–1)    |
| `CACHE_MAX_ENTRIES`    | `500`     | Max cached pairs; oldest is evicted when exceeded   |
| `CACHE_TTL_SECONDS`    | `604 800` | 7 days; stale entries are evicted on read            |
