"""
Configuration module — loads environment variables and defines app-wide settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ─────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Model Settings ───────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude")  # "claude" or "openai"
# Embedding options: "openai" (paid, higher quality) or "free" (HuggingFace, no API key needed)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "free")

CLAUDE_MODEL_NAME = "claude-sonnet-4-20250514"
OPENAI_MODEL_NAME = "gpt-4o"
OPENAI_EMBEDDING_NAME = "text-embedding-3-small"
FREE_EMBEDDING_NAME = "all-MiniLM-L6-v2"  # HuggingFace sentence-transformers, runs locally

# ── Chunking Settings ────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ── ChromaDB Settings ────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "jobfit_docs"
CHROMA_PERSIST_DIR = "./data/chroma_db"

# ── Scraper Settings ────────────────────────────────────────────────
REQUEST_TIMEOUT = 15
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ── Validation ───────────────────────────────────────────────────────
def validate_keys(model_choice: str) -> tuple[bool, str]:
    """Check that required API keys are present for the chosen model."""
    if model_choice == "claude" and not ANTHROPIC_API_KEY:
        return False, "ANTHROPIC_API_KEY is missing. Add it to your .env file."
    if model_choice == "openai" and not OPENAI_API_KEY:
        return False, "OPENAI_API_KEY is missing. Add it to your .env file."
    if EMBEDDING_MODEL == "openai" and not OPENAI_API_KEY:
        return False, "OPENAI_API_KEY is needed for OpenAI embeddings. Switch to EMBEDDING_MODEL=free in .env to use free local embeddings."
    return True, ""
