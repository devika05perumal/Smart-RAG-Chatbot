"""Central configuration for the Smart RAG Assistant.

Everything tunable lives here. Secrets (the Groq API key) are read from the
environment / a local `.env` file and are never hardcoded.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Branding
APP_NAME = "Smart RAG Assistant"
APP_TAGLINE = "Ask questions about your own documents. Every answer comes with its sources."

# Paths
DATA_DIR = Path(os.getenv("SMART_RAG_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
DB_PATH = DATA_DIR / "chat_history.db"
STYLE_PATH = BASE_DIR / "assets" / "style.css"

# Secrets
_PLACEHOLDER_KEYS = {"", "your_groq_api_key_here", "your-api-key", "changeme"}


def get_api_key() -> str:
    """Return the Groq API key from the environment ('' when missing)."""
    key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    return "" if key.lower() in _PLACEHOLDER_KEYS else key


# LLM
# Groq deprecated llama-3.3-70b-versatile for developer-tier access.
# GPT-OSS 120B is the recommended replacement.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
_FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def configured_models() -> list[str]:
    extra = [m.strip() for m in os.getenv("GROQ_MODELS", "").split(",") if m.strip()]
    seen: list[str] = []
    for model in [DEFAULT_MODEL, *extra, *_FALLBACK_MODELS]:
        if model and model not in seen:
            seen.append(model)
    return seen


DEFAULT_TEMPERATURE = _env_float("TEMPERATURE", 0.2)
MAX_OUTPUT_TOKENS = _env_int("MAX_OUTPUT_TOKENS", 1024)
REQUEST_TIMEOUT_S = _env_int("REQUEST_TIMEOUT_S", 60)

# Embeddings
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "huggingface").strip().lower()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
).strip()

# RAG knobs
DEFAULT_CHUNK_SIZE = _env_int("CHUNK_SIZE", 1000)
DEFAULT_CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 200)
DEFAULT_TOP_K = _env_int("TOP_K", 4)
DEFAULT_SCORE_THRESHOLD = _env_float("SCORE_THRESHOLD", 0.25)
DEFAULT_HISTORY_TURNS = _env_int("HISTORY_TURNS", 6)

# Limits
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".csv")
MAX_FILE_SIZE_MB = _env_int("MAX_FILE_SIZE_MB", 20)
MAX_QUESTION_CHARS = 2000
CSV_ROWS_PER_BLOCK = 20
