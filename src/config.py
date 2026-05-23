from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge_base.md"
CHROMA_PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "tg_knowledge"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 400        # target tokens per chunk
CHUNK_OVERLAP = 60      # overlap tokens between consecutive chunks
TOP_K = 5               # chunks retrieved per query

# Comma-separated list of allowed origins for CORS.
# Override in .env: ALLOWED_ORIGINS=https://technovateglobal.com
ALLOWED_ORIGINS: list[str] = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000"
).split(",")

# Comma-separated list of trusted proxy IPs allowed to set X-Forwarded-For.
# Set in .env when running behind Nginx/ALB: TRUSTED_PROXIES=127.0.0.1
TRUSTED_PROXIES: set[str] = set(
    os.environ.get("TRUSTED_PROXIES", "127.0.0.1").split(",")
)
