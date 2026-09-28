"""Central settings. Everything tunable lives here or in environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env (real environment variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()
RAW_DIR = Path(os.getenv("OLIST_RAW_DIR", ROOT / "data" / "raw"))
DB_PATH = Path(os.getenv("OLIST_DB_PATH", ROOT / "data" / "olist.duckdb"))
RAG_DIR = ROOT / "rag"
SCHEMA_PATH = ROOT / "src" / "schema.md"
EVAL_DIR = ROOT / "eval"

# ---------------------------------------------------------------------------
# Model registry. This is the ONLY place that knows about specific models.
# To add or swap a model, add one line here - nothing else changes.
#   provider: "anthropic" (Claude API), "groq" (open-weight models, OpenAI-compatible API)
#             or "ollama" (open-source, runs locally)
#   open_weights: True if anyone can download and self-host the model
#   price: (USD per 1M input tokens, USD per 1M output tokens), None if free
#   temperature: 0 for repeatable answers, None if the model doesn't accept it
# Claude prices checked Sep 2026 on https://platform.claude.com/docs/en/about-claude/pricing
# Groq prices / model IDs checked Sep 2026 on https://console.groq.com/docs/models
# ---------------------------------------------------------------------------
MODELS = {
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "label": "Claude Haiku 4.5",
        "price": (1.00, 5.00),
        "temperature": None,
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-5",
        "label": "Claude Sonnet 5",
        "price": (2.00, 10.00),
        # Sonnet 5 rejects temperature and thinks adaptively by default
        "temperature": None,
    },
    # --- open-weight models served by Groq (needs GROQ_API_KEY; free tier available) ---
    "gpt-oss-120b": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "label": "GPT-OSS 120B (open-weight)",
        "open_weights": True,
        "price": (0.15, 0.60),
        "temperature": None,
    },
    "gpt-oss-20b": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
        "label": "GPT-OSS 20B (open-weight)",
        "open_weights": True,
        "price": (0.075, 0.30),
        "temperature": None,
    },
    "qwen-groq": {
        # Groq lists this as a preview model: it may be renamed or removed. Check the models page if it errors.
        "provider": "groq",
        "model_id": os.getenv("GROQ_QWEN_MODEL", "qwen/qwen3.8-27b"),
        "label": "Qwen3.8 27B (open-weight, preview)",
        "open_weights": True,
        "price": (0.80, 4.00),
        "temperature": None,
    },
    "qwen-coder": {
        "provider": "ollama",
        "model_id": os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
        "label": "Qwen2.5-Coder 7B (open-source, local)",
        "open_weights": True,
        "price": None,
        "temperature": None,
    },
}

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-haiku")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1")

# Which API key each provider needs (the app hides models whose key is missing)
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY"}

# Guardrails and limits
MAX_ROWS = int(os.getenv("MAX_ROWS", "500"))
QUERY_TIMEOUT_S = float(os.getenv("QUERY_TIMEOUT_S", "15"))
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "10"))

# "Use your own data" uploads
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "25"))  # per file
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "8"))

# Retrieval
RAG_BACKEND = os.getenv("RAG_BACKEND", "auto")  # auto | embeddings | bm25
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE = Path(os.getenv("EMBED_CACHE", ROOT / "data" / "models"))  # downloaded once, ~130 MB
TOP_K_EXAMPLES = int(os.getenv("TOP_K_EXAMPLES", "4"))
TOP_K_TERMS = int(os.getenv("TOP_K_TERMS", "4"))

ALLOWED_TABLES = {
    "orders",
    "order_items",
    "payments",
    "reviews",
    "customers",
    "sellers",
    "products",
    "category_translation",
    "geolocation",
}
