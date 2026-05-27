import os
from pathlib import Path
from botocore.config import Config


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    return int(val)


def _get_list(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if not val:
        return default
    return [x.strip() for x in val.split(",") if x.strip()]


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAQ_PATH = DATA_DIR / "faq.json"

APP_TITLE = os.getenv("APP_TITLE", "Siddh Guide Chatbot")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-south-1"

AWS_BOTO_CONFIG = Config(
    region_name=AWS_REGION,
    retries={"max_attempts": 5, "mode": "standard"},
    connect_timeout=5,
    read_timeout=30,
)

ALLOWED_ORIGINS = _get_list(
    "ALLOWED_ORIGINS",
    [
        "https://siddhantaknowledge.org",
        "https://www.siddhantaknowledge.org",
    ],
)

MAX_MESSAGE_CHARS = _get_int("MAX_MESSAGE_CHARS", 600)
HISTORY_LIMIT = _get_int("HISTORY_LIMIT", 8)
MODEL_HISTORY_MAX_CHARS = _get_int("MODEL_HISTORY_MAX_CHARS", 1200)

ENFORCE_API_KEY = _get_bool("ENFORCE_API_KEY", True)
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "")

USE_HISTORY_FOR_CONTINUITY = _get_bool("USE_HISTORY_FOR_CONTINUITY", True)

ALLOW_DEFAULT_SESSION = _get_bool("ALLOW_DEFAULT_SESSION", False)
SESSION_ID_MAX_LEN = _get_int("SESSION_ID_MAX_LEN", 100)

CHAT_TABLE = os.getenv("CHAT_TABLE", "SiddhGuideChat")
CHAT_TTL_DAYS = _get_int("CHAT_TTL_DAYS", 30)

CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "sidh_guide")
BEDROCK_EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

# You should set this explicitly in env for each environment
NOVA_MODEL_ID = os.getenv("NOVA_MODEL_ID", "").strip()

# For Chroma cosine distance: lower is better. Tune after testing your data.
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.45"))
RAG_DEFAULT_K = _get_int("RAG_DEFAULT_K", 3)
RAG_MAX_CONTEXT_CHARS = _get_int("RAG_MAX_CONTEXT_CHARS", 1500)
RAG_MAX_CHUNK_CHARS = _get_int("RAG_MAX_CHUNK_CHARS", 500)
