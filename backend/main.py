from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chat_store import (
    ChatStoreError,
    get_recent_messages,
    history_to_model_messages,
    put_message,
)
from config import (
    ALLOWED_ORIGINS,
    ALLOW_DEFAULT_SESSION,
    APP_TITLE,
    AUTO_CRAWL,
    CHAT_API_KEY,
    CRAWL_ADMIN_KEY,
    CRAWL_INTERVAL_HOURS,
    CRAWL_ON_STARTUP,
    ENFORCE_API_KEY,
    FAQ_PATH,
    HISTORY_LIMIT,
    LOG_LEVEL,
    MAX_MESSAGE_CHARS,
    MODEL_HISTORY_MAX_CHARS,
    RAG_DEFAULT_K,
    RAG_MAX_CHUNK_CHARS,
    RAG_MAX_CONTEXT_CHARS,
    RAG_MAX_DISTANCE,
    SESSION_ID_MAX_LEN,
    USE_HISTORY_FOR_CONTINUITY,
)
from models import generate_answer
from rag import init_rag, retrieve_hits

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("siddh_guide")

# Per-request flag: set when the conversation store could not be read or written.
# Surfaced to the caller as response["continuity"] == "degraded" so an outage is
# observable from the outside without leaking any internal detail.
_continuity_degraded: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "continuity_degraded", default=False
)

faq_data: list[dict] = []
COURSE_DATA: list[dict] = []
WEBSITE_DATA: list[dict] = []
# Live, crawled Siksha course catalog (from data/courses_catalog.json). When
# populated it is the source of truth for the dynamic course count/list; when
# empty the code falls back to the static courses.json snapshot.
COURSE_CATALOG: list[dict] = []

REFUSAL_MESSAGE = (
    "I can help with Siddhanta course and website information. Please ask about "
    "our courses, syllabus, learning outcomes, course recommendations, enrollment "
    "details shown on the website, or Siddhanta Knowledge Foundation."
)

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,100}$")
TITLE_SEPARATOR_RE = re.compile(r"\s+[\-\u2013\u2014]\s+")

PROMPT_INJECTION_RE = re.compile(
    r"\b(ignore|bypass|forget|override|disregard)\b.*\b(instruction|instructions|rules|context|system|policy|knowledge base)\b"
    r"|\b(from|using)\s+(your\s+)?(own|general|outside)\s+knowledge\b",
    re.IGNORECASE,
)

OUT_OF_DOMAIN_PATTERNS = [
    re.compile(r"\b(pm|prime minister|cm|chief minister|president|governor)\b", re.IGNORECASE),
    re.compile(r"\b(today'?s news|current affairs|latest news|breaking news)\b", re.IGNORECASE),
    re.compile(r"\b(joke|jokes|funny story|entertain me)\b", re.IGNORECASE),
    re.compile(r"\b(weather|temperature|rain|forecast)\b", re.IGNORECASE),
    re.compile(r"\b(device|devices|laptop|mobile|phone|desktop|computer|tablet|browser)\b|\bwatch\b.*\bcourse\b", re.IGNORECASE),
    re.compile(r"\b(virat kohli|cricket score|sports score)\b", re.IGNORECASE),
    re.compile(r"\b(chatgpt|openai|large language model|llm)\b", re.IGNORECASE),
    re.compile(r"\b(stock|stocks|share market|crypto|investment advice|financial advice)\b", re.IGNORECASE),
    re.compile(r"\b(medical advice|diagnose|diagnosis|symptoms|prescription|medicine for)\b", re.IGNORECASE),
    re.compile(r"\b(legal advice|lawsuit|court case|lawyer|contract advice)\b", re.IGNORECASE),
    re.compile(
        r"\b(write|generate|debug|fix|build|create)\b.*\b(code|python|javascript|java|c\+\+|html|css|sql|program|script)\b"
        r"|\b(code|python|javascript|java|c\+\+|html|css|sql)\b.*\b(function|script|program|algorithm)\b",
        re.IGNORECASE,
    ),
]

VALIDATION_TOKEN_STOP_WORDS = {
    "about", "above", "after", "again", "also", "and", "answer", "are", "ask",
    "can", "could", "did", "does", "for", "from", "give", "have", "how", "into",
    "is", "its", "me", "more", "my", "need", "of", "on", "or", "please", "should",
    "show", "tell", "than", "that", "the", "their", "them", "there", "these",
    "this", "those", "to", "today", "using", "was", "what", "when", "where",
    "which", "who", "why", "with", "would", "write", "you", "your",
}

VALIDATION_ALIASES = {
    "enroll": {"enroll", "enrol", "enrollment", "enrolment", "admission", "apply"},
    "enrollment": {"enroll", "enrol", "enrollment", "enrolment", "admission", "apply"},
    "fee": {"fee", "fees", "price", "pricing", "cost"},
    "fees": {"fee", "fees", "price", "pricing", "cost"},
    "price": {"fee", "fees", "price", "pricing", "cost"},
    "certificate": {"certificate", "certification", "certified"},
    "certification": {"certificate", "certification", "certified"},
}

ANSWER_VALIDATION_STOP_WORDS = VALIDATION_TOKEN_STOP_WORDS | {
    "ask", "available", "based", "content", "context", "course", "courses",
    "guide", "information", "knowledge", "provided", "sid", "sidh", "siddh",
    "siddhanta", "the",
}

GREETINGS = {
    "hello",
    "hi",
    "hey",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
}

# Exact-set matching missed anything with the bot's name or a stretched vowel
# attached — "Hi sid", "Hii", "Hari om" all fell through to retrieval and were
# refused. Still anchored at both ends, so a real question that merely opens with
# "hi" ("hi, what is the fee for X?") is not swallowed as a bare greeting.
_GREETING_RE = re.compile(
    r"^(?:hi+|hey+|he?llo+|helo+|namaste|namaskar|hari\s*om|greetings|"
    r"good\s+(?:morning|afternoon|evening|day))"
    r"(?:[\s,!.-]+(?:sid|ask\s*sid|there|team|bot|all|everyone))?"
    r"[\s!.?]*$",
    re.I,
)

ABOUT_BOT_KEYWORDS = [
    "who are you",
    "what are you",
    "what do you do",
    "how will you help",
    "how can you help",
    "what help can you do",
    "what can you do",
    "how will you assist me",
    "about you",
    "about ask sid",
    # Direct name questions. Without these the bot cannot state its own name:
    # they fall through to RAG, the KB has nothing about the assistant, and the
    # visitor gets the generic refusal.
    "your name",
    "call you",
    "introduce yourself",
    "who is ask sid",
    "are you ask sid",
    # Old name kept so visitors who still say it are recognised.
    "about siddh guide",
]

ABOUT_BOT_REPLY = (
    "I’m Ask Sid — a helpful assistant by Siddhanta Knowledge Foundation. "
    "I can help you explore IKS courses, suggest what fits your interests, "
    "and guide you to the right certified programs."
)

CAPABILITY_KEYWORDS = [
    "what can you do",
    "what can you help",
    "how can you help",
    "what do you help with",
    "what help can you do",
    "how will you assist",
    "what can you assist",
    "what can you help us",
]

CAPABILITY_REPLY = (
    "I can help you choose the right IKS-certified courses based on your background, "
    "explain course topics simply, and guide you on eligibility and learning paths. "
    "Tell me your field and your goal."
)

IDK_LINE_RE = re.compile(
    r"^\s*i\s*(do\s*not|don't|dont)\s*know\s*[\.\!\?]*\s*$",
    re.IGNORECASE,
)
IDK_CONTAINS_RE = re.compile(r"\b(i\s*(do\s*not|don't|dont)\s*know)\b", re.IGNORECASE)
UNSUPPORTED_CONTAINS_RE = re.compile(
    r"\b(not available|not present|not provided|not mentioned|not contain|does not contain|not in the context|not in the knowledge base)\b",
    re.IGNORECASE,
)

FOLLOWUP_RE = re.compile(
    r"\b(it|that|this|those|these|they|them|above|earlier|previous|same|continue|more|elaborate)\b",
    re.IGNORECASE,
)

SHORT_FOLLOWUPS = {
    "why", "why?",
    "how", "how?",
    "ok", "okay",
    "yes", "no",
    "more", "continue",
    "details", "explain", "elaborate",
    "tell me more",
}

COURSE_INTENT_RE = re.compile(
    r"\b(course|courses|program|programs|certificate|certification|batch|batches|syllabus)\b",
    re.IGNORECASE,
)

COURSE_LIST_PHRASES = [
    "list of courses",
    "give courses",
    "give the list",
    "all courses",
    "siddhanta courses",
    "courses of siddhanta",
    "courses available",
    "show courses",
    "tell me courses",
]

# Broader catch-all for "list/show/give all the courses", "what courses are
# there", "courses available in the website" etc. Plural "courses" only, so a
# question about one specific course ("show me the Samskrit course") is not
# swept up here.
COURSE_LIST_RE = re.compile(
    r"\b(list|show|display|give|see|view|name)\b[^.?!]{0,25}\bcourses\b"
    r"|\ball\b[^.?!]{0,15}\bcourses\b"
    r"|\bcourses\b[^.?!]{0,15}\b(available|list|offered|there|present)\b"
    r"|\b(what|which)\b[^.?!]{0,15}\bcourses\b",
    re.IGNORECASE,
)

COURSE_COUNT_RE = re.compile(
    r"\b(total|count|number|how many)\b.*\b(course|courses|program|programs)\b"
    r"|\b(course|courses|program|programs)\b.*\b(total|count|number|how many)\b"
    r"|^\s*total\s+count\s*$",
    re.IGNORECASE,
)

LATEST_COURSE_RE = re.compile(
    r"\b(latest|newest|most recent|recently (?:launched|added|released|introduced)|"
    r"just (?:launched|added|released)|new(?:ly)?(?:\s+launched| added)?)\b"
    r".{0,40}\b(course|courses|program|programs)\b"
    r"|\b(course|courses|program|programs)\b.{0,40}"
    r"\b(latest|newest|most recent|recently (?:launched|added|released))\b",
    re.IGNORECASE,
)

WHY_CHOOSE_SIDDHANTA_RE = re.compile(
    r"\bwhy\b.*\b(choose|chose|select|study at|study with)\b.*\b(siddhanta|skf|ask sid|sidh guide|siddh guide)\b"
    r"|\bwhy\b.*\b(siddhanta|skf)\b",
    re.IGNORECASE,
)

COURSE_OUTCOME_OR_CAREER_RE = re.compile(
    r"\b(job|jobs|career|placement|employment|where.*use|where.*apply|what.*get|outcome|benefit|after.*course)\b",
    re.IGNORECASE,
)

VAGUE_COURSE_REFERENCE_RE = re.compile(
    r"\b(this|that|it)\b.*\b(course|use|apply|job|career|get|outcome|benefit)\b",
    re.IGNORECASE,
)

PROCEDURAL_QUERY_RE = re.compile(
    r"\b(enrol|enroll|admission|apply|application|process|procedure|fee|fees|cost|duration|eligibility|certificate|certification|what is|what are|how to|how do)\b",
    re.IGNORECASE,
)

DETAIL_QUERY_RE = re.compile(
    r"\b(detail|details|about|explain|describe|syllabus|curriculum)\b",
    re.IGNORECASE,
)

SINGLE_RECOMMEND_RE = re.compile(
    r"\b(one|any one|one course|single|best)\b",
    re.IGNORECASE,
)

BACKGROUND_RE = re.compile(
    r"\b(i am a|i'm a|im a|i am an|i'm an|im an|my background is|i study|i am from|i'm from|im from)\b",
    re.IGNORECASE,
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str = Field(..., min_length=1, max_length=SESSION_ID_MAX_LEN)


def _load_json_list(path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        logger.exception("Failed to load %s", path)
        return []


def _reload_kb_from_disk() -> None:
    """(Re)load all knowledge-base files into the in-memory globals.

    Used at startup and again after every successful crawl so freshly written
    data is served without a restart.
    """
    global faq_data, COURSE_DATA, WEBSITE_DATA, COURSE_CATALOG

    data_dir = FAQ_PATH.parent
    faq_data = _load_json_list(FAQ_PATH)
    COURSE_DATA = _load_json_list(data_dir / "courses.json")
    WEBSITE_DATA = _load_json_list(data_dir / "website.json")
    COURSE_CATALOG = _load_json_list(data_dir / "courses_catalog.json")
    logger.info(
        "KB loaded: faq=%s courses=%s website=%s live_catalog=%s",
        len(faq_data),
        len(COURSE_DATA),
        len(WEBSITE_DATA),
        len(COURSE_CATALOG),
    )


def _run_refresh_and_reload() -> dict:
    """Blocking crawl + re-embed, then reload globals. Runs in a worker thread."""
    from refresh import refresh_knowledge_base

    summary = refresh_knowledge_base()
    if summary.get("ok"):
        _reload_kb_from_disk()
    return summary


async def _crawl_loop() -> None:
    try:
        if CRAWL_ON_STARTUP:
            logger.info("auto-crawl: running initial refresh in background")
            await asyncio.to_thread(_run_refresh_and_reload)

        interval = max(0, CRAWL_INTERVAL_HOURS) * 3600
        if interval <= 0:
            return

        while True:
            await asyncio.sleep(interval)
            logger.info("auto-crawl: running scheduled refresh")
            await asyncio.to_thread(_run_refresh_and_reload)
    except asyncio.CancelledError:
        logger.info("auto-crawl: loop cancelled on shutdown")
        raise
    except Exception:
        logger.exception("auto-crawl: loop crashed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _reload_kb_from_disk()
    init_rag()
    logger.info("Startup complete: KB loaded + RAG initialized")

    crawl_task: Optional[asyncio.Task] = None
    if AUTO_CRAWL:
        crawl_task = asyncio.create_task(_crawl_loop())
        logger.info(
            "auto-crawl enabled (on_startup=%s interval=%sh)",
            CRAWL_ON_STARTUP,
            CRAWL_INTERVAL_HOURS,
        )

    try:
        yield
    finally:
        if crawl_task is not None:
            crawl_task.cancel()
            try:
                await crawl_task
            except (asyncio.CancelledError, Exception):
                pass


app = FastAPI(title=APP_TITLE, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _norm(text: str) -> str:
    return " ".join((text or "").strip().split())


def _normalize_course_text(text: str) -> str:
    text = _norm(text).lower()
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _course_title_variants(raw_title: str) -> set[str]:
    raw_title = (raw_title or "").strip()
    if not raw_title:
        return set()

    variants = {
        raw_title,
        raw_title.replace("—", "-"),
        raw_title.replace("–", "-"),
        raw_title.replace("-", "—"),
    }

    parts = TITLE_SEPARATOR_RE.split(raw_title, maxsplit=1)
    if parts:
        variants.add(parts[0].strip())

    normalized = {_normalize_course_text(v) for v in variants if v.strip()}
    return {v for v in normalized if v}


def _extract_primary_title(raw_title: str) -> str:
    raw_title = (raw_title or "").strip()
    if not raw_title:
        return ""
    parts = TITLE_SEPARATOR_RE.split(raw_title, maxsplit=1)
    return parts[0].strip() if parts else raw_title


def _check_api_key(x_api_key: Optional[str]) -> None:
    if not ENFORCE_API_KEY:
        return

    if not CHAT_API_KEY:
        logger.error("ENFORCE_API_KEY is enabled but CHAT_API_KEY is missing")
        raise HTTPException(status_code=500, detail="Server auth misconfigured")

    if x_api_key != CHAT_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _validate_session_id(session_id: str) -> str:
    sid = _norm(session_id)

    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")

    if sid == "default" and not ALLOW_DEFAULT_SESSION:
        raise HTTPException(status_code=400, detail="default session_id is not allowed")

    if len(sid) > SESSION_ID_MAX_LEN:
        raise HTTPException(status_code=400, detail="session_id too long")

    if not SESSION_ID_RE.match(sid):
        raise HTTPException(
            status_code=400,
            detail="session_id contains invalid characters",
        )

    return sid


def _looks_like_followup(message: str) -> bool:
    msg = _norm(message).lower()

    if msg in SHORT_FOLLOWUPS:
        return True

    if FOLLOWUP_RE.search(msg):
        return True

    if msg.startswith(("what about", "and ", "then ", "so ", "ok ", "okay ")):
        return True

    return False


def is_course_count_question(message: str) -> bool:
    return bool(COURSE_COUNT_RE.search(_norm(message).lower()))


def is_latest_course_question(message: str) -> bool:
    return bool(LATEST_COURSE_RE.search(_norm(message).lower()))


def _latest_courses(limit: int = 3) -> list[dict]:
    dated = [c for c in COURSE_CATALOG if (c.get("published") or "").strip()]
    dated.sort(key=lambda c: c.get("published", ""), reverse=True)
    return dated[:limit]


def is_course_list_intent(message: str) -> bool:
    msg = _norm(message).lower()
    if any(phrase in msg for phrase in COURSE_LIST_PHRASES):
        return True
    return bool(COURSE_LIST_RE.search(msg))


def is_why_choose_siddhanta(message: str) -> bool:
    return bool(WHY_CHOOSE_SIDDHANTA_RE.search(_norm(message).lower()))


def is_course_outcome_or_career_query(message: str) -> bool:
    msg = _norm(message).lower()
    return bool(
        COURSE_OUTCOME_OR_CAREER_RE.search(msg)
        or VAGUE_COURSE_REFERENCE_RE.search(msg)
    )


def is_procedural_query(message: str) -> bool:
    return bool(PROCEDURAL_QUERY_RE.search(_norm(message)))


def wants_single_recommendation(message: str) -> bool:
    return bool(SINGLE_RECOMMEND_RE.search(_norm(message)))


def wants_course_details(message: str) -> bool:
    return bool(DETAIL_QUERY_RE.search(_norm(message)))


def is_course_recommendation_intent(message: str) -> bool:
    msg = _norm(message).lower()

    if is_course_list_intent(msg):
        return True

    if COURSE_INTENT_RE.search(msg) and not is_procedural_query(msg):
        return True

    if BACKGROUND_RE.search(msg):
        return True

    recommendation_patterns = [
        r"\bfor law\b",
        r"\bfor architect\b",
        r"\bfor architecture\b",
        r"\bfor management\b",
        r"\bfor education\b",
        r"\bfor teacher\b",
        r"\bfor student\b",
        r"\blaw student\b",
        r"\barchitecture student\b",
        r"\bmanagement student\b",
        r"\beducation student\b",
    ]
    return any(re.search(p, msg) for p in recommendation_patterns)


def is_greeting(message: str) -> bool:
    msg = _norm(message)
    return msg.lower() in GREETINGS or bool(_GREETING_RE.match(msg))


def is_about_bot(message: str) -> bool:
    msg = _norm(message).lower()
    return any(k in msg for k in ABOUT_BOT_KEYWORDS)


def is_capability_question(message: str) -> bool:
    msg = _norm(message).lower()
    return any(k in msg for k in CAPABILITY_KEYWORDS)


def _looks_like_idk(answer: str) -> bool:
    if not answer:
        return True
    a = answer.strip()
    if REFUSAL_MESSAGE.lower() in a.lower():
        return True
    if IDK_LINE_RE.match(a):
        return True
    if IDK_CONTAINS_RE.search(a):
        return True
    if UNSUPPORTED_CONTAINS_RE.search(a):
        return True
    if len(a) < 6:
        return True
    return False


def _strip_embedded_refusal(answer: str) -> str:
    answer = (answer or "").strip()
    if not answer:
        return ""

    if REFUSAL_MESSAGE in answer:
        answer = answer.replace(REFUSAL_MESSAGE, "").strip()

    lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip()
        and "i can help with siddhanta course and website information" not in line.lower()
    ]
    return "\n".join(lines).strip()


def is_prompt_injection(message: str) -> bool:
    return bool(PROMPT_INJECTION_RE.search(_norm(message).lower()))


def is_out_of_domain_query(message: str) -> bool:
    msg = _norm(message).lower()
    return any(pattern.search(msg) for pattern in OUT_OF_DOMAIN_PATTERNS)


def _content_tokens(text: str, stop_words: set[str]) -> list[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [
        w
        for w in words
        if (len(w) >= 3 or w in {"ai", "cm", "pm", "ug", "pg"})
        and w not in stop_words
    ]


def _validation_terms(question: str) -> set[str]:
    msg = _norm(question).lower()
    terms = set(_content_tokens(msg, VALIDATION_TOKEN_STOP_WORDS))

    if re.search(r"\bpm\b|\bprime minister\b", msg):
        terms.update({"prime", "minister"})
        terms.discard("pm")

    if re.search(r"\bcm\b|\bchief minister\b", msg):
        terms.update({"chief", "minister"})
        terms.discard("cm")

    return terms


def _term_matches_context(term: str, context_tokens: set[str], context_text: str) -> bool:
    aliases = VALIDATION_ALIASES.get(term, {term})
    return any(alias in context_tokens or alias in context_text for alias in aliases)


def _context_is_relevant(question: str, context_text: str) -> bool:
    terms = _validation_terms(question)
    if not terms:
        # No content terms (e.g. "how do I do this?") — the vector-distance gate
        # already established similarity, so trust it rather than refuse.
        return True

    context_text = (context_text or "").lower()
    context_tokens = set(re.findall(r"[a-z0-9]+", context_text))
    matched = {term for term in terms if _term_matches_context(term, context_tokens, context_text)}

    if is_out_of_domain_query(question):
        # Out-of-domain questions are allowed only when the KB text explicitly
        # contains the key topic terms; loose semantic similarity is not enough.
        return len(matched) == len(terms)

    # In-domain: the retrieval distance threshold (RAG_MAX_DISTANCE) already
    # gated semantic similarity, and the model is instructed to answer only from
    # this context (and every answer is re-validated against it). So require just
    # a light keyword overlap to drop obviously-unrelated chunks, and otherwise
    # trust the embeddings. This lets natural phrasings ("how can I access the
    # courses") match KB text that words it differently ("Click To Enroll").
    return len(matched) >= 1


def _validate_retrieved_hits(question: str, hits: list[dict]) -> list[dict]:
    usable: list[dict] = []

    for hit in hits or []:
        doc = (hit.get("document") or "").strip()
        source = (hit.get("source") or "").strip()
        distance = hit.get("distance")

        if not doc or not source or distance is None:
            continue

        try:
            if float(distance) > RAG_MAX_DISTANCE:
                continue
        except (TypeError, ValueError):
            continue

        usable.append(hit)

    if not usable:
        return []

    joined_context = "\n".join((hit.get("document") or "") for hit in usable)
    if not _context_is_relevant(question, joined_context):
        return []

    return usable


def _citations_from_hits(hits: list[dict]) -> list[str]:
    citations: list[str] = []
    seen = set()

    for hit in hits:
        source = (hit.get("source") or "").strip()
        title = (hit.get("title") or "").strip()
        if not source:
            continue

        label = f"{source} | {title}" if title else source
        if label not in seen:
            citations.append(label)
            seen.add(label)

    return citations


def _build_context_from_hits(
    hits: list[dict],
    max_chars: int = RAG_MAX_CONTEXT_CHARS,
    max_chunk_chars: int = RAG_MAX_CHUNK_CHARS,
) -> str:
    chunks = []
    total = 0

    for hit in hits:
        src = (hit.get("source") or "").strip()
        title = (hit.get("title") or "").strip()
        doc = (hit.get("document") or "").strip()
        if not src or not doc:
            continue

        header = f"[source={src}{' | ' + title if title else ''}]"
        if len(doc) > max_chunk_chars:
            doc = doc[:max_chunk_chars].rsplit(" ", 1)[0] + "..."

        chunk = f"{header}\n{doc}"
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining <= 0:
                break
            chunk = chunk[:remaining].rsplit(" ", 1)[0] + "..."
            chunks.append(chunk)
            break

        chunks.append(chunk)
        total += len(chunk)

    return "\n\n---\n\n".join(chunks).strip()


def _retrieve_validated_context(
    question: str,
    k: int = RAG_DEFAULT_K,
    max_chars: int = RAG_MAX_CONTEXT_CHARS,
    max_chunk_chars: int = RAG_MAX_CHUNK_CHARS,
) -> tuple[str, list[str]]:
    hits = retrieve_hits(question, k=k)
    hits = _validate_retrieved_hits(question, hits)
    citations = _citations_from_hits(hits)

    if not hits or not citations:
        return "", []

    context = _build_context_from_hits(
        hits,
        max_chars=max_chars,
        max_chunk_chars=max_chunk_chars,
    )
    if not context:
        return "", []

    return context, citations


def _answer_supported_by_context(
    answer: str, context: str, extra_allowed_tokens: frozenset = frozenset()
) -> bool:
    """Is every content word in the answer traceable to the retrieved context?

    `extra_allowed_tokens` carries words the *user* supplied this session (e.g.
    "engineering" after "I am an engineer"). Echoing the user's own word back is
    not a hallucination, so it must not trip the grounding check — but it widens
    nothing else: course facts, numbers and codes are still matched against the
    KB context alone, below.
    """
    if _looks_like_idk(answer) or not context.strip():
        return False

    context_text = context.lower()
    context_tokens = set(re.findall(r"[a-z0-9]+", context_text)) | set(extra_allowed_tokens)
    answer_tokens = _content_tokens(answer, ANSWER_VALIDATION_STOP_WORDS)

    if answer_tokens:
        matched = [
            token
            for token in answer_tokens
            if token in context_tokens or token in context_text
        ]
        if len(answer_tokens) <= 3 and len(matched) < len(answer_tokens):
            return False
        if len(answer_tokens) > 3 and (len(matched) < 2 or len(matched) / len(answer_tokens) < 0.25):
            return False

    # Codes, dates, numbers, and acronyms are high-risk factual claims.
    claim_markers = re.findall(r"\b[A-Z]{2,}[A-Z0-9-]*\b|\b\d+(?:\.\d+)?\b", answer)
    for marker in claim_markers:
        if marker.lower() in {"iks", "ug", "pg"}:
            continue
        if marker.lower() not in context_text:
            return False

    for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
        sentence_tokens = _content_tokens(sentence, ANSWER_VALIDATION_STOP_WORDS)
        if len(sentence_tokens) >= 4:
            sentence_matches = [
                token
                for token in sentence_tokens
                if token in context_tokens or token in context_text
            ]
            if not sentence_matches:
                return False

    return True


def _generated_answer_or_refusal(
    user_message: str,
    context: str,
    citations: list[str],
    history_messages: list[dict],
    session_context: str = "",
    extra_allowed_tokens: frozenset = frozenset(),
) -> tuple[str, bool]:
    if not context or not citations:
        return REFUSAL_MESSAGE, False

    answer = generate_answer(
        user_message=user_message,
        context=context,
        history_messages=history_messages,
        session_context=session_context,
    )
    answer = _strip_embedded_refusal(answer)

    if not _answer_supported_by_context(
        answer, context, extra_allowed_tokens=extra_allowed_tokens
    ):
        return REFUSAL_MESSAGE, False

    return answer, True


def _tokenize_query(text: str) -> list[str]:
    stop_words = {
        "course", "courses", "program", "programs", "certificate", "certification",
        "give", "tell", "show", "find", "need", "want", "any", "one", "for", "me",
        "the", "a", "an", "i", "am", "im", "i'm", "student", "about", "please",
        "suggest", "recommend", "required",
    }
    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    return [w for w in words if len(w) >= 3 and w not in stop_words]


def _expand_query_terms(tokens: list[str]) -> set[str]:
    expanded = set(tokens)

    synonym_map = {
        "law": {"law", "legal", "jurisprudence", "ethics", "governance", "reasoning", "debating", "nyaya"},
        "legal": {"law", "legal", "jurisprudence", "ethics", "governance", "reasoning", "debating", "nyaya"},
        "architect": {"architect", "architecture", "design", "visualization", "creative", "temple", "vastu"},
        "architecture": {"architect", "architecture", "design", "visualization", "creative", "temple", "vastu"},
        "management": {"management", "leadership", "governance", "organization"},
        "education": {"education", "teaching", "learning", "pedagogy"},
        "teacher": {"teacher", "teaching", "learning", "pedagogy", "education"},
        "design": {"design", "visualization", "creative", "architecture", "temple"},
        "student": {"beginner", "foundation", "fundamentals"},
    }

    for token in tokens:
        expanded.update(synonym_map.get(token, set()))

    return expanded


def find_exact_course_title_match(message: str, course_data: list[dict]) -> Optional[dict]:
    msg = _normalize_course_text(message)
    if not msg:
        return None

    for item in course_data:
        raw_title = (
            item.get("title")
            or item.get("course_name")
            or item.get("name")
            or ""
        ).strip()

        if not raw_title:
            continue

        variants = _course_title_variants(raw_title)
        if msg in variants:
            return {
                "title": raw_title,
                "display_title": _extract_primary_title(raw_title),
                "content": (item.get("content") or "").strip(),
            }

    return None


def _unique_course_titles(course_data: list[dict]) -> list[str]:
    titles: list[str] = []
    seen = set()

    for item in course_data:
        raw_title = (
            item.get("title")
            or item.get("course_name")
            or item.get("name")
            or ""
        ).strip()

        title = _extract_primary_title(raw_title)
        key = _normalize_course_text(title)
        if title and key not in seen:
            titles.append(title)
            seen.add(key)

    return titles


def find_course_title_in_message(message: str, course_data: list[dict]) -> Optional[dict]:
    msg = _normalize_course_text(message)
    if not msg:
        return None

    best: Optional[dict] = None
    best_len = 0

    for item in course_data:
        raw_title = (
            item.get("title")
            or item.get("course_name")
            or item.get("name")
            or ""
        ).strip()
        if not raw_title:
            continue

        primary_title = _extract_primary_title(raw_title)
        variants = _course_title_variants(raw_title) | _course_title_variants(primary_title)

        for variant in variants:
            if len(variant) < 8:
                continue
            if variant in msg and len(variant) > best_len:
                best = {
                    "title": raw_title,
                    "display_title": primary_title,
                    "content": (item.get("content") or "").strip(),
                }
                best_len = len(variant)

    return best


def _course_context_for_title(title: str, course_data: list[dict]) -> tuple[str, list[str]]:
    display_title = _extract_primary_title(title)
    title_key = _normalize_course_text(display_title)
    related: list[dict] = []

    for item in course_data:
        raw_title = (
            item.get("title")
            or item.get("course_name")
            or item.get("name")
            or ""
        ).strip()
        if _normalize_course_text(_extract_primary_title(raw_title)) == title_key:
            related.append(item)

    if not related:
        return "", []

    def priority(item: dict) -> int:
        title_text = (item.get("title") or "").lower()
        if "objective" in title_text or "outcome" in title_text:
            return 0
        if "curriculum" in title_text:
            return 1
        if "overview" in title_text:
            return 2
        return 3

    chunks = [f"[source=courses.json | {display_title}]\nTitle: {display_title}"]
    total = len(chunks[0])

    for item in sorted(related, key=priority):
        raw_title = (item.get("title") or display_title).strip()
        content = (item.get("content") or "").strip()
        if not content:
            continue
        chunk = f"Section: {raw_title}\n{content}"
        if total + len(chunk) > RAG_MAX_CONTEXT_CHARS:
            remaining = RAG_MAX_CONTEXT_CHARS - total
            if remaining <= 80:
                break
            chunk = chunk[:remaining].rsplit(" ", 1)[0] + "..."
            chunks.append(chunk)
            break
        chunks.append(chunk)
        total += len(chunk)

    return "\n\n".join(chunks), [f"courses.json | {display_title}"]


# --------------------------------------------------------------------------- #
# Conversational continuity.
#
# A short follow-up ("how much is it?", "who is it for?", "tell me more") only
# makes sense relative to the course already under discussion. On its own it
# retrieves nothing and the bot refuses. These helpers detect such follow-ups
# and recover the last course the session was talking about so the thread is
# kept instead of dropped.
# --------------------------------------------------------------------------- #
_FOLLOWUP_PRONOUN_RE = re.compile(
    r"\b(it|its|it's|this|that|these|those|the same|the course|one)\b", re.I
)
_FOLLOWUP_ATTR_RE = re.compile(
    r"\b(how much|price|priced|cost|costs|fee|fees|duration|how long|"
    r"hours|credits?|who is it for|who's it for|audience|eligibility|"
    r"prerequisites?|syllabus|curriculum|outcomes?|objectives?|"
    r"tell me more|more details?|know more|level)\b",
    re.I,
)


def is_followup_about_previous(message: str) -> bool:
    """True when the message reads like a context-dependent follow-up that only
    makes sense relative to a course already under discussion."""
    msg = _norm(message).lower()
    if not msg:
        return False

    words = msg.split()
    if len(words) > 8:
        return False

    # If the message itself names a course, it is a fresh course query, not a
    # bare follow-up — let the normal course routing handle it.
    if COURSE_DATA and find_course_title_in_message(message, COURSE_DATA):
        return False

    has_attr = bool(_FOLLOWUP_ATTR_RE.search(msg))
    has_pronoun = bool(_FOLLOWUP_PRONOUN_RE.search(msg))

    # This runs only AFTER greetings, anchors (count/list) and the FAQ have all
    # declined — so a short message arriving here, in an ongoing session, is
    # almost always a follow-up ("what about electrical engineering", "can you
    # provide the link?", "and the url?"). Treat short messages as follow-ups
    # too, not just ones with an explicit pronoun/attribute cue. The generation
    # step stays KB-grounded, so a genuinely off-topic short question still
    # refuses safely. Cap the "short" rule at 5 words so self-contained 6+ word
    # questions (e.g. "what does the Ayush course cover?") are NOT treated as
    # follow-ups and stay un-polluted by prior context.
    return has_attr or has_pronoun or len(words) <= 5


def _last_course_context_from_history(messages: list[dict]) -> Optional[tuple[str, list[str]]]:
    """(context, citations) for the most recent course discussed, newest-first.

    Live per-course facts live in website.json (titles like
    "Samskrit 1: Thinking in Samskrit"). courses.json uses different titles
    ("Sanskrit I - Thinking in Samskrit — Overview") that do not match the
    website-sourced answers, so it is only a fallback. Returns the same
    (context, citations) shape the rest of the pipeline uses.
    """
    for m in reversed(messages or []):
        text = (m.get("text") or "").strip()
        if not text:
            continue

        # Preferred: the website course entry this turn was about (matches both a
        # user question naming the course and the assistant reply that echoes the
        # full title).
        if WEBSITE_DATA:
            entry = _website_course_entry_in_message(text)
            content = (entry.get("content") or "").strip() if entry else ""
            if entry and content:
                title = (entry.get("title") or "").strip()
                content = content[:RAG_MAX_CONTEXT_CHARS]
                context = f"[source=website.json | {title}]\nTitle: {title}\nContent: {content}"
                return context, [f"website.json | {title}"]

        # Fallback: courses.json.
        if COURSE_DATA:
            match = (
                find_course_title_in_message(text, COURSE_DATA)
                or find_exact_course_title_match(text, COURSE_DATA)
            )
            if match:
                context, citations = _course_context_for_title(match["title"], COURSE_DATA)
                if context and citations:
                    return context, citations

    return None


def rank_course_candidates(user_message: str, course_data: list[dict], limit: int = 8) -> list[dict]:
    tokens = _tokenize_query(user_message)
    expanded_terms = _expand_query_terms(tokens)

    ranked: list[dict] = []
    seen = set()

    for item in course_data:
        raw_title = (
            item.get("title")
            or item.get("course_name")
            or item.get("name")
            or ""
        ).strip()
        if not raw_title:
            continue

        title = _extract_primary_title(raw_title)
        if not title or title.lower() in seen:
            continue

        content = (item.get("content") or "").strip()
        searchable_title = _normalize_course_text(raw_title)
        searchable_content = content.lower()

        score = 0
        matched_terms = 0

        for term in expanded_terms:
            in_title = term in searchable_title
            in_content = term in searchable_content

            if in_title:
                score += 8
                matched_terms += 1
            elif in_content:
                score += 3
                matched_terms += 1

        score += min(matched_terms, 5)

        if matched_terms > 0 and all(term not in searchable_title for term in expanded_terms):
            score -= 1

        if score > 0:
            ranked.append(
                {
                    "title": title,
                    "raw_title": raw_title,
                    "content": content,
                    "score": score,
                }
            )
            seen.add(title.lower())

    ranked.sort(key=lambda x: (-x["score"], x["title"].lower()))
    return ranked[:limit]


def get_faq_answer(message: str) -> Optional[str]:
    message_lower = _norm(message).lower()

    # never let FAQ hijack exact course title selections
    exact_course = find_exact_course_title_match(message, COURSE_DATA) if COURSE_DATA else None
    if exact_course:
        return None

    blocked_generic_keywords = {
        "about", "company", "organization", "profile",
        "mission", "vision", "siddhanta", "course", "courses",
        "knowledge", "meta"
    }

    for item in faq_data:
        for keyword in item.get("keywords", []):
            kw = _norm(keyword).lower()
            if not kw:
                continue

            # skip dangerous generic keywords
            if kw in blocked_generic_keywords:
                continue

            # for multi-word phrases, match the whole phrase anywhere in the
            # message (word-boundary), so natural questions like "what is
            # Siddhanta Knowledge Foundation?" hit the intended FAQ entry, while
            # a bare substring inside a larger word still cannot false-match.
            if " " in kw:
                if re.search(r"\b" + re.escape(kw) + r"\b", message_lower):
                    return item.get("answer")
                continue

            # for single-word keywords, only allow if message itself is short/simple
            if len(message_lower.split()) <= 3 and re.search(r"\b" + re.escape(kw) + r"\b", message_lower):
                return item.get("answer")

    return None


def get_why_choose_siddhanta_answer() -> Optional[str]:
    for item in faq_data:
        answer = (item.get("answer") or "").strip()
        if "Siddhanta Knowledge Foundation" not in answer:
            continue

        return (
            "Siddhanta Knowledge Foundation works to revive, nurture, and develop "
            "Indian Knowledge Systems through education, research, and "
            "technology-enabled platforms. Siddhanta collaborates with premier "
            "Indian institutions, is developing over 100 IKS-based courses for "
            "learners across disciplines, and creates learning resources for "
            "different age groups."
        )

    return None


def _website_entry_by_id(raw_id: str) -> Optional[dict]:
    for item in WEBSITE_DATA:
        if (item.get("id") or "").strip() == raw_id:
            return item
    return None


def _is_website_course_entry(item: dict) -> bool:
    if (item.get("category") or "").strip().lower() != "course":
        return False

    raw_id = (item.get("id") or "").strip().lower()
    content = (item.get("content") or "").lower()
    return (
        raw_id.startswith(("course-", "mdp-"))
        or "course title:" in content
        or "program title:" in content
    )


def _website_course_entry_in_message(message: str) -> Optional[dict]:
    msg = _normalize_course_text(message)
    if not msg:
        return None

    best: Optional[dict] = None
    best_len = 0
    msg_numbers = set(re.findall(r"\d+", msg))

    for item in WEBSITE_DATA:
        if not _is_website_course_entry(item):
            continue

        raw_title = (item.get("title") or "").strip()

        # Match ONLY on the course's own title — never on generic, shared
        # keyword tokens ("Samskrit", "Foundation", "Indian", ...), which collide
        # across many courses and produced confidently-wrong answers. Also allow
        # the short prefix before a colon ("Samskrit 1: Thinking in Samskrit" ->
        # "Samskrit 1") so users can name a course by its short form.
        variants = _course_title_variants(raw_title)
        if ":" in raw_title:
            prefix = raw_title.split(":", 1)[0].strip()
            if prefix:
                variants.add(prefix)

        # A numbered course ("Samskrit 1") must not be matched by a message that
        # names a different number ("Samskrit 2"): if the title carries a number,
        # that number has to appear in the message too.
        title_numbers = set(re.findall(r"\d+", raw_title))

        for variant in variants:
            if len(variant) < 8:
                continue
            if variant.lower() not in msg:
                continue
            if title_numbers and not (title_numbers & msg_numbers):
                continue
            if len(variant) > best_len:
                best = item
                best_len = len(variant)

    return best


def _website_entry_for_query(message: str) -> Optional[dict]:
    msg = _norm(message).lower()
    if not msg:
        return None

    course_entry = _website_course_entry_in_message(message)
    if course_entry:
        return course_entry

    raw_id = None

    if re.search(r"\b(refund|refunds|return|returns|cancel|cancellation)\b", msg):
        raw_id = "refund-policy"
    elif re.search(r"\b(contact|support|help|reach|phone|email|message form|get in touch)\b", msg):
        raw_id = "contact-support"
    elif re.search(r"\b(enrol|enroll|enrollment|enrolment|admission|apply|join|register|registration)\b", msg):
        raw_id = "enrollment-overview"
    elif re.search(r"\b(price|pricing|fee|fees|cost|amount|gst)\b", msg):
        raw_id = "pricing-information"
    elif "what siddhanta does" in msg or "what does siddhanta do" in msg:
        raw_id = "about-what-siddhanta-does"
    elif re.search(r"\b(siksha|shiksha)\b", msg):
        raw_id = "siksha-platform-overview"
    elif re.search(r"\baajivan\b", msg):
        raw_id = "aajivan-overview"
    elif re.search(r"\b(prakashan|publication|publications|book|books|astadhyayi|astadhyayi pravesha)\b", msg):
        raw_id = "prakashan-overview"
    elif re.search(r"\b(event|events|webinar|webinars|seminar|conference|lecture)\b", msg):
        raw_id = "events-overview"
    elif re.search(r"\b(blog|blogs|article|articles|post|posts)\b", msg):
        raw_id = "blogs-overview"
    elif re.search(r"\b(siddhanta vijnan|siddhantavijnan|vijnan)\b", msg):
        raw_id = "siddhantavijnan-overview"
    # sandhaan / shodha sub-pages must be checked before the generic overviews
    elif "siddhanta kosha" in msg or "siddhanta kosa" in msg:
        raw_id = "siddhanta-kosha-overview"
    elif "shastra maps" in msg or "shaastra maps" in msg:
        raw_id = "shastra-maps-overview"
    elif re.search(r"\b(linguistics|vyakarana|mahabhasya)\b", msg):
        raw_id = "sandhaan-linguistics"
    elif re.search(r"\b(jyotisha|jyotish|predictive|laghu jatakam)\b", msg) and "sandhaan" in msg:
        raw_id = "sandhaan-jyotisha"
    elif re.search(r"\byoga\b", msg) and "sandhaan" in msg:
        raw_id = "sandhaan-yoga"
    elif re.search(r"\bsandhaan\b", msg):
        raw_id = "sandhaan-technology-overview"
    elif "siddhanta prastuti" in msg or "siddhanta-prastuti" in msg or "prastuti" in msg:
        raw_id = "shodha-siddhanta-prastuti"
    elif "indic thought model" in msg or "thought models" in msg:
        raw_id = "shodha-indic-thought-models"
    elif "conscious enterprise" in msg or "enterprise management" in msg:
        raw_id = "shodha-conscious-enterprise-management"
    elif re.search(r"\bshodha\b", msg):
        raw_id = "shodha-research-overview"
    elif "privacy policy" in msg:
        raw_id = "privacy-policy"
    elif "terms of use" in msg or "terms and conditions" in msg:
        raw_id = "terms-of-use-overview"
    elif "copyright policy" in msg:
        raw_id = "copyright-policy"

    return _website_entry_by_id(raw_id) if raw_id else None


def _extract_labeled_value(content: str, label: str) -> Optional[str]:
    pattern = (
        rf"\b{re.escape(label)}\s*:\s*(.*?)"
        r"(?=\.\s+(?:Course title|Duration|Applicable audience|Category shown|"
        r"Categories shown|Price shown|The course|This course|The page|"
        r"Enrollment steps|Program title|Program type|Users should)|$)"
    )
    match = re.search(pattern, content or "", re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    value = _norm(match.group(1)).strip(" .")
    return value or None


def _website_course_answer_for_entry(entry: dict, message: str) -> Optional[str]:
    content = (entry.get("content") or "").strip()
    title = (
        _extract_labeled_value(content, "Course title")
        or (entry.get("title") or "").strip()
    )
    msg = _norm(message).lower()

    price = _extract_labeled_value(content, "Price shown")
    duration = _extract_labeled_value(content, "Duration")
    audience = _extract_labeled_value(content, "Applicable audience")
    categories = (
        _extract_labeled_value(content, "Categories shown")
        or _extract_labeled_value(content, "Category shown")
    )

    if re.search(r"\b(price|pricing|fee|fees|cost|amount|gst)\b", msg):
        if price:
            return f"The visible price for {title} is {price}."
        return f"The accessed course page for {title} does not show a visible price."

    if re.search(r"\b(duration|hours|time|how long)\b", msg):
        if duration:
            return f"The visible duration for {title} is {duration}."
        return f"The accessed course page for {title} does not show a visible duration."

    if re.search(r"\b(eligible|eligibility|audience|who can|suitable)\b", msg):
        if audience:
            return f"The applicable audience shown for {title} is {audience}."
        return f"The accessed course page for {title} does not show eligibility details."

    if re.search(r"\b(category|categories|stream|area)\b", msg):
        if categories:
            return f"The categories shown for {title} are {categories}."
        return f"The accessed course page for {title} does not show categories."

    if re.search(r"\b(enrol|enroll|enrollment|enrolment|admission|apply|join|register)\b", msg):
        return (
            f"For {title}, the accessed course page shows a Click To Enroll "
            "button. It does not show detailed enrollment steps beyond that button."
        )

    parts = [f"{title} is listed as an Ask Sid/Siksha course."]
    if duration:
        parts.append(f"Duration: {duration}.")
    if audience:
        parts.append(f"Applicable audience: {audience}.")
    if categories:
        parts.append(f"Categories shown: {categories}.")
    if price:
        parts.append(f"Visible price: {price}.")

    return " ".join(parts)


def _website_answer_for_entry(entry: dict, message: str) -> Optional[str]:
    raw_id = (entry.get("id") or "").strip()
    content = (entry.get("content") or "").strip()
    if not content:
        return None

    if _is_website_course_entry(entry):
        return _website_course_answer_for_entry(entry, message)

    if raw_id == "enrollment-overview":
        return (
            "To enroll, open the course page and use the Click To Enroll button "
            "shown near the course price. The website also shows course discovery "
            "links such as Join A Course, Explore, Read More, and View All. "
            "The accessed pages do not show a detailed step-by-step payment, login, "
            "or enrollment workflow."
        )

    if raw_id == "refund-policy":
        return (
            "Siddhanta Knowledge Foundation's refund policy says all purchases of "
            "courses and related educational materials are final. Refunds are "
            "generally not offered once a purchase has been made. In exceptional "
            "circumstances, users may contact the support team with a detailed "
            "explanation, and the request will be reviewed case by case."
        )

    if raw_id == "contact-support":
        return (
            "You can contact Siddhanta through the Contact page form. The visible "
            "fields are Name, Email, and Comment or Message, followed by a Contact "
            "Us button. The accessed Contact page does not show a phone number or "
            "direct support email."
        )

    if raw_id == "pricing-information":
        return (
            "Course prices are shown on individual course pages, not as one single "
            "general pricing table. Several Siksha course pages show Rs. 2,500.00 "
            "with GST additional, some show $75.00 with fee additional, and the "
            "Aajivan Arthasastra course shows Rs. 99.00 including GST as the current "
            "price. Ask about a specific course for its visible price."
        )

    if raw_id == "siksha-platform-overview":
        return (
            "Siksha is Siddhanta's education initiative for exploring Indic "
            "wisdom through online courses. The website lists course areas such "
            "as Foundation, Agriculture, Arts and Humanities, Education, Law, "
            "Management, Medicine, and STEM."
        )

    if raw_id == "aajivan-overview":
        return (
            "Aajivan is Siddhanta's set of 1-hour capsule learning experiences "
            "anchored in Indian Knowledge Systems and Indic traditions. The page "
            "describes these courses as rooted in ancient insights and designed "
            "for modern living and applications."
        )

    if raw_id == "about-what-siddhanta-does":
        return (
            "Siddhanta creates courses, textbooks, educational videos, "
            "publications, and films on Shastras, Ancient Indian Knowledge "
            "System and Heritage, Indian History, and Civilisation. The website "
            "also says Siddhanta uses technology tools to make this wisdom "
            "accessible in contemporary formats."
        )

    return content


def get_website_answer(message: str) -> Optional[tuple[str, list[str]]]:
    entry = _website_entry_for_query(message)
    if not entry:
        return None

    answer = _website_answer_for_entry(entry, message)
    if not answer:
        return None

    title = (entry.get("title") or entry.get("id") or "Website").strip()
    return answer, [f"website.json | {title}"]


def _response(
    answer: str,
    sources: list[str],
    context_used: int,
    request_id: str,
    citations: Optional[list[str]] = None,
    status: str = "ok",
) -> Dict:
    return {
        "answer": answer,
        "status": status,
        "sources": sources,
        "context_used": context_used,
        "citations": citations or [],
        "request_id": request_id,
        # "degraded" means this turn was not fully recorded / history was not
        # readable, so follow-ups may not resolve. Never claim "ok" in that case.
        "continuity": "degraded" if _continuity_degraded.get() else "ok",
    }


def _safe_put_message(**kwargs) -> bool:
    """Persist a turn. Returns False and flags the request as degraded on failure.

    A failed write must not 500 the visitor — they still get their answer — but
    it must never be silent either, because the next turn in this session will
    be missing context because of it.
    """
    try:
        put_message(**kwargs)
        return True
    except ChatStoreError:
        # chat_store already logged the cause with a stack trace.
        _continuity_degraded.set(True)
        return False
    except Exception:
        logger.exception(
            "CONTINUITY: unexpected chat-store write failure session=%s role=%s",
            kwargs.get("session_id"),
            kwargs.get("role"),
        )
        _continuity_degraded.set(True)
        return False


def _load_recent_messages(session_id: str, limit: int) -> list[dict]:
    """Ordered history for exactly this session, or [] with the request flagged."""
    try:
        return get_recent_messages(session_id=session_id, limit=limit)
    except ChatStoreError:
        # chat_store already logged the cause; flag it so the response cannot
        # claim continuity that did not happen.
        _continuity_degraded.set(True)
        return []
    except Exception:
        logger.exception(
            "CONTINUITY: unexpected chat-store read failure session=%s", session_id
        )
        _continuity_degraded.set(True)
        return []


def _store_assistant_message(
    session_id: str,
    answer: str,
    request_id: str,
    sources: list[str],
    context_used: int,
) -> None:
    _safe_put_message(
        session_id=session_id,
        role="assistant",
        text=answer,
        request_id=request_id,
        sources=sources,
        context_used=context_used,
    )


def _refusal_response(session_id: str, request_id: str) -> Dict:
    # KB-only enforcement: when retrieval is empty, weak, irrelevant, or
    # uncited, the model is not called and the standard refusal is returned.
    sources = ["kb_refusal"]
    _store_assistant_message(
        session_id=session_id,
        answer=REFUSAL_MESSAGE,
        request_id=request_id,
        sources=sources,
        context_used=0,
    )
    return _response(
        REFUSAL_MESSAGE,
        sources,
        0,
        request_id,
        citations=[],
        status="refused",
    )


# --------------------------------------------------------------------------- #
# Course recommendation: match the user's subject/field against the real course
# catalog and let the model pick from it. Kept narrow (needs an explicit
# recommend/suggest verb) so it never hijacks course-detail questions, which
# stay on RAG.
# --------------------------------------------------------------------------- #
_RECOMMEND_VERB_RE = re.compile(
    # explicit recommend/suggest phrasings ...
    r"\b(?:recommend|recomend|recommendation|recommendations|suggest|suggestion|"
    r"suggestions|which course|what course|best course|good course|"
    r"help me (?:choose|pick|find) a course)\b"
    # ... OR a field/subject-scoped course query, e.g. "courses on economics",
    # "any course relevant to tamil", "courses for engineering students",
    # "course related to robotics". These should match against the catalog, not
    # fall through to a refusal. (A named course is still excluded below.)
    r"|\bcourses?\s+(?:on|about|related to|relevant to|regarding|for|in|of)\b"
    r"|\bany\s+courses?\b"
    # ... OR a plainly stated intent to study something: "i would like to take
    # sanskrit", "i want to learn ayurveda", "looking for something in law".
    # These carry a subject but no recommend/suggest verb, so they used to fall
    # through to retrieval and get refused.
    r"|\bi\s*(?:'d|\s+would)?\s*(?:like|want|wish|wanna|need)\s*(?:to)?\s*"
    r"(?:take|do|join|learn|study|buy|enrol|enroll|pursue|start)\b"
    r"|\b(?:i\s*(?:am|'m)\s+)?looking\s+for\b",
    re.I,
)

_REC_GENERIC_TOKENS = {
    "subject", "subjects", "field", "fields", "based", "interest", "interests",
    "topic", "topics", "area", "areas", "stream", "background", "domain",
}


# --------------------------------------------------------------------------- #
# Temporary current-session context.
#
# Facts the visitor states about themselves ("I am an engineer") are used to
# personalise recommendations and follow-up explanations for the remainder of
# the loaded page, and nothing more. They are:
#   - derived deterministically here rather than left to the model to infer;
#   - read only from messages carrying the current session_id, so a refreshed
#     page (new id) starts genuinely blank;
#   - never treated as KB facts — course names, fees, duration, eligibility and
#     links still come only from the catalog/KB.
# --------------------------------------------------------------------------- #

# Occupation words that imply a study field. Kept small and explicit; anything
# unrecognised falls through to the free-text patterns below.
_ROLE_TO_FIELD = {
    "engineer": "engineering",
    "engineering": "engineering",
    "teacher": "education",
    "lecturer": "education",
    "professor": "education",
    "doctor": "medicine",
    "nurse": "healthcare",
    "lawyer": "law",
    "advocate": "law",
    "architect": "architecture",
    "accountant": "commerce",
    "manager": "management",
    "designer": "design",
    "farmer": "agriculture",
    "scientist": "science",
    "developer": "software",
    "programmer": "software",
    "artist": "arts",
    "musician": "music",
    "psychologist": "psychology",
}

# Trailing noise to trim off a captured field ("engineering student" -> "engineering").
_FIELD_TAIL_RE = re.compile(
    r"\b(student|students|graduate|graduates|background|degree|field|stream|"
    r"course|courses|major|professional|guy|person)\b.*$",
    re.I,
)

_FIELD_PATTERNS = [
    # "I am an engineer", "I'm a teacher"
    re.compile(
        r"\bi\s*(?:am|'m|m)\s+(?:an?\s+)?(" + "|".join(_ROLE_TO_FIELD) + r")\b",
        re.I,
    ),
    # "my stream is commerce", "my background is in law", "my field: design"
    re.compile(
        r"\bmy\s+(?:stream|field|background|subject|discipline|domain|branch|major)\s*"
        r"(?:is|was|:)?\s*(?:in\s+)?([a-z][a-z &/'-]{2,40})",
        re.I,
    ),
    # "I study architecture", "I studied commerce"
    re.compile(r"\bi\s+(?:study|studied|am\s+studying)\s+([a-z][a-z &/'-]{2,40})", re.I),
    # "I work in finance", "I work as an architect"
    re.compile(
        r"\bi\s+work\s+(?:in|as)\s+(?:an?\s+)?([a-z][a-z &/'-]{2,40})", re.I
    ),
    # "I am from an engineering background"
    re.compile(r"\bfrom\s+(?:an?\s+)?([a-z][a-z &/'-]{2,40})\s+background\b", re.I),
    # "I am an engineering student" (role word not in the map, but field-shaped)
    re.compile(r"\bi\s*(?:am|'m|m)\s+(?:an?\s+)?([a-z][a-z &/'-]{2,40})\s+student\b", re.I),
]

_INTEREST_PATTERNS = [
    re.compile(r"\b(?:i\s*(?:am|'m|m)\s+)?interested\s+in\s+([a-z][a-z &/'-]{2,40})", re.I),
    re.compile(r"\bi\s+(?:like|love|enjoy)\s+([a-z][a-z &/'-]{2,40})", re.I),
]

# Format/style preferences ("I prefer a short course"). Recorded so the model can
# honour them, but deliberately NOT used to pick the recommendation subject —
# "short" is not a field of study.
_PREFERENCE_PATTERNS = [
    re.compile(r"\bi\s+(?:prefer|want|need|am\s+looking\s+for)\s+([a-z][a-z &/'-]{2,40})", re.I),
]


def _clean_field(raw: str) -> str:
    field = _FIELD_TAIL_RE.sub("", (raw or "")).strip(" .,:;!?-&/'")
    field = re.sub(r"^(?:a|an|the|some|any)\s+", "", field.strip(), flags=re.I)
    field = re.sub(r"\s+", " ", field)
    # Reject filler captures like "interested", "looking", "not sure".
    if len(field) < 3 or field.lower() in _REC_GENERIC_TOKENS:
        return ""
    if field.lower() in {"interested", "looking", "sure", "not", "the", "a", "an"}:
        return ""
    return _ROLE_TO_FIELD.get(field.lower(), field.lower())


def _session_profile(recent_messages: Optional[list[dict]]) -> dict:
    """Temporary preferences stated by the user in THIS session.

    Returns {"field": str, "interests": [str], "preferred_course": str}. Scans
    newest-first so a newly stated field replaces an earlier one.
    """
    profile = {"field": "", "interests": [], "preferences": [], "preferred_course": ""}
    if not recent_messages:
        return profile

    for m in reversed(recent_messages):
        role = (m.get("role") or "").strip().lower()
        text = (m.get("text") or "").strip()
        if not text:
            continue

        # Course under discussion: taken from either side of the conversation,
        # but only ever accepted if it is a real catalog title.
        if not profile["preferred_course"] and COURSE_CATALOG:
            for c in COURSE_CATALOG:
                title = (c.get("title") or "").strip()
                if title and title.lower() in text.lower():
                    profile["preferred_course"] = title
                    break

        # Self-described facts come from the user only. Assistant text is model
        # output and must never become a "fact about the user".
        if role != "user":
            continue

        if not profile["field"]:
            for pattern in _FIELD_PATTERNS:
                found = pattern.search(text)
                if found:
                    field = _clean_field(found.group(1))
                    if field:
                        profile["field"] = field
                        break

        for pattern in _INTEREST_PATTERNS:
            for found in pattern.finditer(text):
                interest = _clean_field(found.group(1))
                if interest and interest not in profile["interests"]:
                    profile["interests"].append(interest)

        for pattern in _PREFERENCE_PATTERNS:
            for found in pattern.finditer(text):
                pref = _clean_field(found.group(1))
                if pref and pref not in profile["preferences"]:
                    profile["preferences"].append(pref)

    return profile


# "What is my stream?" — a question about what the visitor told us in THIS
# session. It is never a KB question, so it must not reach retrieval: on a fresh
# page there is nothing to recall and the honest answer is to say so.
_ASK_OWN_PROFILE_RE = re.compile(
    r"\b(?:what|which|do you (?:know|remember))\b[^.?!]{0,30}\bmy\b[^.?!]{0,20}"
    r"\b(stream|field|background|subject|discipline|branch|profession|occupation)\b"
    r"|\bwhat\s+did\s+i\s+(?:say|tell)\b",
    re.I,
)

_UNKNOWN_PROFILE_REPLY = (
    "I don’t know your stream yet. Please tell me your field or background."
)


# "which course suits my background?" mentions the background but is asking for a
# course, not for what we remember. Anything naming a course/program is routed
# normally so the recommendation and RAG paths keep those questions.
_MENTIONS_COURSE_RE = re.compile(
    r"\b(course|courses|program|programs|programme|programmes|class|classes|"
    r"certificate|certification|syllabus)\b",
    re.I,
)


# "list them" / "show them" right after a course answer. These carry no noun of
# their own, so the normal course-list matcher (which needs the word "courses")
# misses them and they dead-end in a refusal.
_BARE_LIST_RE = re.compile(
    r"^\s*(?:can|could|would|will)?\s*(?:you\s+)?(?:pls|plz|please|kindly)?\s*"
    r"(?:list|show|name|give|display)\s*"
    r"(?:me\s+)?(?:them|it|those|these|all|out)?\s*[.?!]*\s*$",
    re.I,
)


def is_bare_list_followup(message: str, recent_messages: Optional[list[dict]]) -> bool:
    """True when a bare 'list them' follows a turn that was about courses."""
    if not _BARE_LIST_RE.match(_norm(message)):
        return False
    for m in reversed(recent_messages or []):
        text = (m.get("text") or "").lower()
        if not text:
            continue
        if "course" in text or "siksha" in text:
            return True
    return False


def is_own_profile_question(message: str) -> bool:
    msg = _norm(message)
    if not _ASK_OWN_PROFILE_RE.search(msg):
        return False
    if _MENTIONS_COURSE_RE.search(msg):
        return False
    return True


def _own_profile_answer(profile: dict) -> str:
    if profile.get("field"):
        return (
            f"You told me your field is {profile['field']}. Tell me if that has "
            "changed and I'll use the new one."
        )
    if profile.get("interests"):
        return (
            f"You haven't told me your stream, but you mentioned an interest in "
            f"{profile['interests'][0]}. What is your field or background?"
        )
    return _UNKNOWN_PROFILE_REPLY


def _session_context_block(profile: dict) -> str:
    """Render the profile for the model, clearly labelled as untrusted and temporary."""
    bits = []
    if profile.get("field"):
        bits.append(f"- Stated field/background: {profile['field']}")
    if profile.get("interests"):
        bits.append(f"- Stated interests: {', '.join(profile['interests'][:4])}")
    if profile.get("preferences"):
        bits.append(f"- Stated preferences: {', '.join(profile['preferences'][:4])}")
    if profile.get("preferred_course"):
        bits.append(f"- Course currently under discussion: {profile['preferred_course']}")
    if not bits:
        return ""
    return (
        "TEMPORARY CURRENT-SESSION DETAILS (stated by the user in this "
        "conversation; use only to personalise wording and course selection, "
        "never as factual course information):\n" + "\n".join(bits)
    )


def _profile_allowed_tokens(profile: dict) -> frozenset:
    """Tokens the user themselves supplied, so echoing them is not a hallucination."""
    words: set[str] = set()
    for value in [profile.get("field", ""), profile.get("preferred_course", "")] + list(
        profile.get("interests") or []
    ):
        for token in re.findall(r"[a-z0-9]+", str(value).lower()):
            if len(token) > 2:
                words.add(token)
    return frozenset(words)


def _is_recommendation_request(message: str) -> bool:
    if not _RECOMMEND_VERB_RE.search(_norm(message)):
        return False
    # If a specific course is named, it's a detail question, not a recommendation.
    if COURSE_DATA and find_course_title_in_message(message, COURSE_DATA):
        return False
    return True


def states_own_field_only(message: str) -> bool:
    """True for a bare self-description like "I am an engineer".

    On its own this is not a question, so it used to reach retrieval and get
    refused — a dead end right after the visitor told us the most useful thing
    about themselves. Treated as a recommendation request instead, so the reply
    is generated from the real catalog for that field.
    """
    msg = _norm(message)
    if not msg or "?" in msg or len(msg.split()) > 12:
        return False
    # Anything naming a course/programme is a real query; leave it to the normal routes.
    if _MENTIONS_COURSE_RE.search(msg):
        return False

    profile = _session_profile([{"role": "user", "text": msg}])
    return bool(profile.get("field") or profile.get("interests"))


def _recommendation_subject_terms(message: str) -> list[str]:
    """Meaningful subject tokens in a recommendation request. Empty means no
    subject was given (e.g. 'recommend a course based on my subject')."""
    return [t for t in _tokenize_query(message) if t not in _REC_GENERIC_TOKENS]


def _resolve_recommendation_subject(
    message: str, profile: dict
) -> tuple[list[str], str]:
    """(subject terms, where it came from) for a recommendation request.

    Priority, per the required behaviour:
      1. a subject named explicitly in the current request — a fresh topic must
         never be overridden by an older one;
      2. otherwise the most recent field/interest stated in this session, so
         "suggest me a course" after "I am an engineer" recommends for
         engineering;
      3. otherwise nothing, and the caller asks the user for their field.
    """
    explicit = _recommendation_subject_terms(message)
    if explicit:
        return explicit, "request"

    if profile.get("field"):
        return _tokenize_query(profile["field"]) or [profile["field"]], "session"

    if profile.get("interests"):
        first = profile["interests"][0]
        return _tokenize_query(first) or [first], "session"

    return [], "none"


def _recommend_courses_response(
    user_message: str,
    history_messages: list[dict],
    session_id: str,
    request_id: str,
    recent_messages: Optional[list[dict]] = None,
) -> Dict:
    if not COURSE_CATALOG:
        return _rag_answer_response(
            user_message,
            history_messages,
            session_id,
            request_id,
            recent_messages=recent_messages,
        )

    profile = _session_profile(recent_messages)
    subject_terms, subject_source = _resolve_recommendation_subject(user_message, profile)

    # No subject in the request AND none stated earlier this session -> ask.
    if not subject_terms:
        answer = (
            "Sure — which subject or field are you interested in? For example: "
            "management, law, architecture, Sanskrit, agriculture, psychology, or "
            "education. Tell me the area and I'll suggest matching courses."
        )
        _safe_put_message(
            session_id=session_id, role="assistant", text=answer,
            request_id=request_id, sources=["local"], context_used=0,
        )
        return _response(answer, ["local"], 0, request_id)

    # Give the model the real catalog (title + categories) and the user's request,
    # then let it pick the best-fitting courses. Including the request text keeps
    # the user's own terms (e.g. "MBA") in the grounded context.
    lines = []
    for c in COURSE_CATALOG:
        title = (c.get("title") or "").strip()
        if not title:
            continue
        cats = c.get("categories") or []
        cat_text = ", ".join(str(x) for x in cats) if isinstance(cats, list) else str(cats)
        lines.append(f"- {title}" + (f" (categories: {cat_text})" if cat_text else ""))

    session_block = _session_context_block(profile)

    context = (
        "[source=courses_catalog.json | Siksha course catalog]\n"
        f"User request: {user_message}\n"
        + (f"Subject to match: {' '.join(subject_terms)}\n" if subject_terms else "")
        + "Available Siksha courses:\n"
        + "\n".join(lines)
    )
    citations = ["courses_catalog.json | Siksha live catalog"]

    # When the subject came from earlier in the conversation rather than from
    # this message, say so, so the reply reads as "based on your engineering
    # background" instead of appearing to guess.
    carried = (
        f"The user did not repeat their field in this message. Earlier in this "
        f"same conversation they said their field/interest is "
        f"'{profile.get('field') or (profile.get('interests') or [''])[0]}'. "
        f"Use that to choose the courses and refer to it naturally.\n"
        if subject_source == "session"
        else ""
    )

    try:
        answer = generate_answer(
            user_message=(
                f"The user asked for a course recommendation: {user_message}\n"
                + carried
                + "From the Available Siksha courses listed in the context, choose the 1-3 "
                "that best fit the user's subject, field, or interest, and briefly say why "
                "each fits. Only name courses that appear in the list; never invent a course. "
                "Do not promise jobs, salaries, or placement outcomes unless the context "
                "states them."
            ),
            context=context,
            history_messages=history_messages,
            session_context=session_block,
        )
    except Exception:
        logger.exception("[%s] recommendation generation failed", request_id)
        return _refusal_response(session_id, request_id)

    answer = _strip_embedded_refusal(answer)
    # Guard against hallucination: the answer must name a real catalog course.
    names_real_course = any(
        (c.get("title") or "").strip() and (c.get("title") or "").strip().lower() in answer.lower()
        for c in COURSE_CATALOG
    )
    if _looks_like_idk(answer) or not names_real_course:
        return _refusal_response(session_id, request_id)

    _safe_put_message(
        session_id=session_id, role="assistant", text=answer,
        request_id=request_id, sources=["courses_catalog.json", "nova"], context_used=1,
    )
    return _response(answer, ["courses_catalog.json", "nova"], 1, request_id, citations=citations)


def _retrieval_query(user_message: str, recent_messages: Optional[list[dict]] = None) -> str:
    """Build the text used for vector retrieval.

    A bare follow-up ("how much is it?", "who is it for?") carries no topic on
    its own, so it retrieves nothing. When the current turn looks like a
    follow-up, anchor it with the most recent user question so retrieval lands on
    the course/topic under discussion. Fresh, self-contained questions are used
    as-is so old context never pollutes them. This replaces the old regex
    follow-up router with a standard history-aware retriever.
    """
    if not recent_messages or not is_followup_about_previous(user_message):
        return user_message

    # Anchor the follow-up with the last couple of turns (user AND assistant),
    # so the course/topic under discussion is carried into retrieval. The
    # assistant turn usually names the exact course ("... Indian System of Music
    # ..."), which is what makes "can you provide the link?" resolve correctly.
    tail: list[str] = []
    for m in reversed(recent_messages):
        text = (m.get("text") or "").strip()
        if not text or text.lower() == user_message.lower():
            continue
        tail.append(text[:200])
        if len(tail) >= 2:
            break

    if not tail:
        return user_message

    tail.reverse()
    return " ".join(tail) + ". " + user_message


def _catalog_context_for_course(course_title: str) -> tuple[str, list[str]]:
    """Grounded context built from the live catalog row for one course.

    Used when vector retrieval finds nothing for a follow-up like "what is its
    duration?". The row is handed to the model as ordinary context so the reply
    is written dynamically, exactly like every other answer — no canned
    sentences. Blank fields are stated as not listed, so the model can say so
    truthfully instead of inventing a value.
    """
    title = (course_title or "").strip()
    if not title or not COURSE_CATALOG:
        return "", []

    entry = next(
        (c for c in COURSE_CATALOG if (c.get("title") or "").strip().lower() == title.lower()),
        None,
    )
    if not entry:
        return "", []

    real_title = (entry.get("title") or title).strip()
    lines = [f"Course title: {real_title}"]
    for field in ("duration", "price", "audience", "categories", "url", "published"):
        value = entry.get(field)
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        value = str(value).strip() if value else ""
        lines.append(f"{field}: {value if value else 'not listed on the website'}")

    context = f"[source=courses_catalog.json | {real_title}]\n" + "\n".join(lines)
    return context, [f"courses_catalog.json | {real_title}"]


def _course_aware_answer(
    user_message: str,
    history_messages: list[dict],
    recent_messages: Optional[list[dict]],
    profile: dict,
) -> Optional[tuple[str, list[str]]]:
    """A course-specific answer for an FAQ-shaped follow-up, or None.

    "How do I enroll?" straight after discussing a course should answer for THAT
    course rather than giving the generic enrollment blurb. This deliberately
    stores nothing and returns None on any miss, so the caller can fall back to
    the curated FAQ and still write exactly one assistant turn — two writes would
    corrupt the order history is replayed in.
    """
    course = profile.get("preferred_course") or ""
    query = _retrieval_query(user_message, recent_messages)
    if course and course.lower() not in query.lower():
        query = f"{course}. {query}"

    context, citations = _retrieve_validated_context(
        query,
        k=RAG_DEFAULT_K,
        max_chars=RAG_MAX_CONTEXT_CHARS,
        max_chunk_chars=RAG_MAX_CHUNK_CHARS,
    )
    if not context or not citations:
        return None

    answer, supported = _generated_answer_or_refusal(
        user_message=user_message,
        context=context,
        citations=citations,
        history_messages=history_messages,
        session_context=_session_context_block(profile),
        extra_allowed_tokens=_profile_allowed_tokens(profile),
    )
    if not supported:
        return None

    return answer, citations


def _rag_answer_response(
    user_message: str,
    history_messages: list[dict],
    session_id: str,
    request_id: str,
    recent_messages: Optional[list[dict]] = None,
) -> Dict:
    # Retrieve against a history-aware query so follow-ups resolve, but always
    # answer the user's actual message.
    query = _retrieval_query(user_message, recent_messages)
    context, citations = _retrieve_validated_context(
        query,
        k=RAG_DEFAULT_K,
        max_chars=RAG_MAX_CONTEXT_CHARS,
        max_chunk_chars=RAG_MAX_CHUNK_CHARS,
    )

    # Temporary details the user stated this session, so "why did you pick that
    # course?" can be answered in terms of their background without those
    # details ever becoming a source of course facts.
    profile = _session_profile(recent_messages)

    def _answer_from(ctx: str, cits: list[str]) -> tuple[str, bool]:
        return _generated_answer_or_refusal(
            user_message=user_message,
            context=ctx,
            citations=cits,
            history_messages=history_messages,
            session_context=_session_context_block(profile),
            extra_allowed_tokens=_profile_allowed_tokens(profile),
        )

    answer, supported = ("", False)
    if context and citations:
        answer, supported = _answer_from(context, citations)

    if not supported:
        # Retrieval found nothing usable, but a course is under discussion — give
        # the model that course's catalog row and let it answer from there. Blank
        # fields are marked "not listed on the website" in the context, so it can
        # say so truthfully rather than invent a value.
        cat_ctx, cat_cits = _catalog_context_for_course(profile.get("preferred_course", ""))
        if not cat_ctx:
            return _refusal_response(session_id, request_id)

        answer, supported = _answer_from(cat_ctx, cat_cits)
        if not supported:
            return _refusal_response(session_id, request_id)
        citations = cat_cits

    sources = ["rag", "nova"]
    _store_assistant_message(
        session_id=session_id,
        answer=answer,
        request_id=request_id,
        sources=sources,
        context_used=1,
    )
    return _response(answer, sources, 1, request_id, citations=citations)


@app.get("/health")
def health() -> Dict:
    return {
        "status": "ok",
        "courses_live": len(COURSE_CATALOG),
        "website_entries": len(WEBSITE_DATA),
        "auto_crawl": AUTO_CRAWL,
    }


def _check_admin_key(x_api_key: Optional[str]) -> None:
    expected = CRAWL_ADMIN_KEY or CHAT_API_KEY
    if not expected:
        raise HTTPException(status_code=500, detail="Admin auth not configured")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/admin/refresh")
async def admin_refresh(
    wait: bool = False,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> Dict:
    """Manually trigger a live crawl + re-embed (no redeploy needed).

    Runs in the background by default; pass ?wait=1 to block for the summary.
    """
    _check_admin_key(x_api_key)

    if wait:
        summary = await asyncio.to_thread(_run_refresh_and_reload)
        return {"status": "completed", "summary": summary}

    asyncio.create_task(asyncio.to_thread(_run_refresh_and_reload))
    return {"status": "refresh_started"}


@app.get("/history/{session_id}")
def history(
    session_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> Dict:
    _check_api_key(x_api_key)
    sid = _validate_session_id(session_id)
    try:
        msgs = get_recent_messages(session_id=sid, limit=50)
    except ChatStoreError:
        # Already logged with a stack trace. Report unavailability rather than
        # an empty transcript, which would look like a valid empty session.
        raise HTTPException(status_code=503, detail="history temporarily unavailable")
    return {"session_id": sid, "messages": msgs}


@app.post("/chat")
async def chat(
    request: ChatRequest,
    req: Request,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> Dict:
    request_id = str(uuid.uuid4())[:8]
    _continuity_degraded.set(False)

    try:
        _check_api_key(x_api_key)

        user_message = _norm(request.message).replace("knowledgebase", "knowledge base")
        session_id = _validate_session_id(request.session_id)

        if not user_message:
            raise HTTPException(status_code=400, detail="message is required")

        if len(user_message) > MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"message too long (max {MAX_MESSAGE_CHARS} chars)",
            )

        origin = req.headers.get("origin", "")
        logger.info(
            "[%s] /chat origin=%s session=%s msg_len=%s",
            request_id,
            origin,
            session_id,
            len(user_message),
        )

        # Ordered turns for THIS session id only. A different page/tab generates
        # a different id, so its turns are unreachable from here.
        recent_before = _load_recent_messages(session_id, HISTORY_LIMIT)
        has_prior_history = len(recent_before) > 0

        history_messages = []
        if USE_HISTORY_FOR_CONTINUITY and has_prior_history:
            history_messages = history_to_model_messages(
                recent_before,
                max_chars=MODEL_HISTORY_MAX_CHARS,
            )

        _safe_put_message(
            session_id=session_id,
            role="user",
            text=user_message,
            request_id=request_id,
        )

        if is_greeting(user_message):
            answer = "Hello! How can I assist you today?"
            _safe_put_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                request_id=request_id,
                sources=["local"],
                context_used=0,
            )
            return _response(answer, ["local"], 0, request_id)

        if is_about_bot(user_message):
            answer = ABOUT_BOT_REPLY
            _safe_put_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                request_id=request_id,
                sources=["local"],
                context_used=0,
            )
            return _response(answer, ["local"], 0, request_id)

        if is_capability_question(user_message):
            answer = CAPABILITY_REPLY
            _safe_put_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                request_id=request_id,
                sources=["local"],
                context_used=0,
            )
            return _response(answer, ["local"], 0, request_id)

        # "What is my stream?" is about this session, not the knowledge base.
        # Answered from the current session's own turns, so a refreshed page
        # (new session id, no turns) correctly says it does not know.
        if is_own_profile_question(user_message):
            answer = _own_profile_answer(_session_profile(recent_before))
            _safe_put_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                request_id=request_id,
                sources=["session"],
                context_used=0,
            )
            return _response(answer, ["session"], 0, request_id)

        # KB-only enforcement: legacy cache entries do not carry retrieval
        # context or citations, so they are not trusted as answer sources.

        if is_course_count_question(user_message):
            # Prefer the live, crawled Siksha catalog for an accurate count; fall
            # back to the static courses.json snapshot only when it is absent.
            if COURSE_CATALOG:
                count = len(COURSE_CATALOG)
                answer = (
                    f"There are {count} courses currently listed on Siddhanta's "
                    "Siksha platform (siksha.siddhantaknowledge.org)."
                )
                sources = ["courses_catalog.json"]
                citations = ["courses_catalog.json | Siksha live catalog"]
            else:
                titles = _unique_course_titles(COURSE_DATA) if COURSE_DATA else []
                if not titles:
                    return _refusal_response(session_id, request_id)
                answer = (
                    f"There are {len(titles)} courses available in the current "
                    "Ask Sid course database."
                )
                sources = ["courses.json"]
                citations = ["courses.json"]

            _safe_put_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                request_id=request_id,
                sources=sources,
                context_used=1,
            )
            return _response(answer, sources, 1, request_id, citations=citations)

        if is_latest_course_question(user_message) and COURSE_CATALOG:
            newest = _latest_courses(limit=3)
            if newest:
                top = newest[0]
                parts = [
                    f"The most recently added course on Siddhanta's Siksha platform is "
                    f"“{top['title']}” (listed {top['published']})."
                ]
                if top.get("categories"):
                    parts.append(f"Category: {', '.join(top['categories'])}.")
                if top.get("price"):
                    parts.append(f"Price shown: {top['price']}.")
                if len(newest) > 1:
                    parts.append(
                        "Other recent additions: "
                        + "; ".join(f"{c['title']} ({c['published']})" for c in newest[1:])
                        + "."
                    )
                answer = " ".join(parts)
                _safe_put_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    request_id=request_id,
                    sources=["courses_catalog.json"],
                    context_used=1,
                )
                return _response(
                    answer,
                    ["courses_catalog.json"],
                    1,
                    request_id,
                    citations=["courses_catalog.json | Siksha live catalog"],
                )

        if is_course_list_intent(user_message) or is_bare_list_followup(
            user_message, recent_before
        ):
            # Prefer the live Siksha catalog so the listed courses stay current.
            if COURSE_CATALOG:
                titles = [
                    (c.get("title") or "").strip()
                    for c in COURSE_CATALOG
                    if (c.get("title") or "").strip()
                ]
                source = "courses_catalog.json"
                citation = "courses_catalog.json | Siksha live catalog"
            else:
                ranked = rank_course_candidates(user_message, COURSE_DATA, limit=20) if COURSE_DATA else []
                titles = [r["title"] for r in ranked] if ranked else [
                    _extract_primary_title(
                        (item.get("title") or item.get("course_name") or item.get("name") or "").strip()
                    )
                    for item in COURSE_DATA[:20]
                    if (item.get("title") or item.get("course_name") or item.get("name"))
                ]
                source = "courses.json"
                citation = "courses.json"

            titles = [t for t in titles if t]

            if not titles:
                return _refusal_response(session_id, request_id)

            shown = titles[:45]
            more = len(titles) - len(shown)
            answer = (
                f"Here are the {len(titles)} courses currently available on Siksha:\n\n"
                + "\n".join([f"• {t}" for t in shown])
                + (f"\n\n…and {more} more." if more > 0 else "")
                + "\n\nWhich one would you like details about?"
            )

            _safe_put_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                request_id=request_id,
                sources=[source],
                context_used=1,
            )
            return _response(
                answer,
                [source],
                1,
                request_id,
                citations=[citation],
            )

        if is_prompt_injection(user_message):
            return _refusal_response(session_id, request_id)

        # Curated FAQ layer: a small set of common, stable questions (enrollment,
        # contact, pricing, certificates, org facts) answered deterministically
        # from faq.json. Short intent questions like "how do I enroll" sit right
        # at the RAG retrieval borderline and flicker; the curated FAQ makes them
        # reliable. Course questions carry none of these keywords, so they fall
        # through to RAG unchanged — this is a thin FAQ+RAG hybrid, not the old
        # brittle keyword-routing maze.
        faq_answer = get_faq_answer(user_message)
        if faq_answer:
            # When a course is already under discussion, an FAQ-shaped follow-up
            # ("how do I enroll?") is really about THAT course. Try the
            # course-specific answer first and fall back to the curated FAQ if
            # retrieval cannot support one, so this is never worse than before.
            profile = _session_profile(recent_before)
            course_specific = None
            if profile.get("preferred_course") and is_followup_about_previous(user_message):
                try:
                    course_specific = _course_aware_answer(
                        user_message, history_messages, recent_before, profile
                    )
                except Exception:
                    logger.exception(
                        "[%s] course-aware FAQ attempt failed; using curated FAQ",
                        request_id,
                    )

            if course_specific:
                answer, citations = course_specific
                sources = ["rag", "nova"]
                _safe_put_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    request_id=request_id,
                    sources=sources,
                    context_used=1,
                )
                return _response(answer, sources, 1, request_id, citations=citations)

            _safe_put_message(
                session_id=session_id,
                role="assistant",
                text=faq_answer,
                request_id=request_id,
                sources=["faq"],
                context_used=0,
            )
            return _response(faq_answer, ["faq"], 0, request_id, citations=["faq"])

        # Course recommendation ("recommend a course for MBA"): match the subject
        # against the real catalog. Needs an explicit recommend/suggest verb and
        # no specific course named, so course-detail questions stay on RAG.
        if _is_recommendation_request(user_message) or states_own_field_only(user_message):
            return _recommend_courses_response(
                user_message,
                history_messages,
                session_id,
                request_id,
                recent_messages=recent_before,
            )

        # RAG-first: every remaining question is answered by retrieving the most
        # relevant knowledge-base content and letting the model write a grounded,
        # cited answer — or refuse if nothing relevant is found. Retrieval is
        # history-aware, so a follow-up ("how much is it?") resolves against the
        # course under discussion. This replaces the old maze of keyword/intent
        # routers that mis-classified natural questions.
        return _rag_answer_response(
            user_message,
            history_messages,
            session_id,
            request_id,
            recent_messages=recent_before,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("[%s] Unhandled error in /chat", request_id)
        raise HTTPException(status_code=500, detail="Internal error")
