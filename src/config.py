"""
Legal AI Assistant — Central Configuration
Loads environment variables and defines all project-wide settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CASE_LAW_DIR = DATA_DIR / "case_law"
CONTRACTS_DIR = DATA_DIR / "contracts"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
CASE_LAW_INDEX_DIR = VECTORSTORE_DIR / "case_law_index"
CONTRACTS_INDEX_DIR = VECTORSTORE_DIR / "contracts_index"

# ── API Keys ───────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")

# ── Provider Selection ─────────────────────────────────────────────────────
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
DEFAULT_EMBEDDING_PROVIDER = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "huggingface")
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")

# ── Model Names ────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
GOOGLE_MODEL = "gemini-2.0-flash"
DEEPSEEK_MODEL = "deepseek-chat"
GROQ_MODEL = "llama-3.3-70b-versatile"
HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# ── Chunking Configuration ────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ── Retrieval Configuration ───────────────────────────────────────────────
RETRIEVAL_TOP_K = 5
SEARCH_TYPE = "similarity"

# ── LLM Configuration ─────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048

# ── API Endpoints (OpenAI-compatible) ─────────────────────────────────────
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_available_providers():
    """Return a list of LLM providers that have valid API keys configured."""
    providers = []
    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
        providers.append("openai")
    if GOOGLE_API_KEY and GOOGLE_API_KEY != "your_google_api_key_here":
        providers.append("google")
    if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your_deepseek_api_key_here":
        providers.append("deepseek")
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        providers.append("groq")
    providers.append("mock")
    return providers


def get_available_embedding_providers():
    """Return a list of embedding providers that are available."""
    providers = ["huggingface"]  # Always available (local)
    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
        providers.append("openai")
    return providers
