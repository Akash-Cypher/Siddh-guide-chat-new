from __future__ import annotations

import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chat_store import get_recent_messages, history_to_model_messages, put_message
from config import (
    ALLOWED_ORIGINS,
    ALLOW_DEFAULT_SESSION,
    APP_TITLE,
    CHAT_API_KEY,
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

faq_data: list[dict] = []
COURSE_DATA: list[dict] = []

REFUSAL_MESSAGE = (
    "I can answer only from the Sidh Guide knowledge base. I don’t have this "
    "information in the available course or website content. Please ask about "
    "our courses, platform, enrollment, or learning content."
)

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,100}$")

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
    "available", "based", "content", "context", "course", "courses", "guide",
    "information", "knowledge", "provided", "sidh", "siddh", "siddhanta", "the",
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
    "about siddh guide",
]

ABOUT_BOT_REPLY = (
    "I’m Siddh Guide — a helpful assistant by Siddhanta Knowledge Foundation. "
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global faq_data, COURSE_DATA

    with FAQ_PATH.open("r", encoding="utf-8") as f:
        faq_data = json.load(f)

    courses_path = FAQ_PATH.parent / "courses.json"
    if courses_path.exists():
        try:
            with courses_path.open("r", encoding="utf-8") as f:
                COURSE_DATA = json.load(f)
            logger.info("Loaded %s courses from courses.json", len(COURSE_DATA))
        except Exception:
            logger.exception("Failed to load courses.json")
            COURSE_DATA = []
    else:
        logger.warning("courses.json not found in data directory")
        COURSE_DATA = []

    init_rag()
    logger.info("Startup complete: FAQ loaded + RAG initialized + Courses loaded")
    yield


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
    text = text.replace("—", "-")
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
        raw_title.replace("-", "—"),
    }

    parts = re.split(r"\s*[—-]\s*", raw_title, maxsplit=1)
    if parts:
        variants.add(parts[0].strip())

    normalized = {_normalize_course_text(v) for v in variants if v.strip()}
    return {v for v in normalized if v}


def _extract_primary_title(raw_title: str) -> str:
    raw_title = (raw_title or "").strip()
    if not raw_title:
        return ""
    parts = re.split(r"\s*[—-]\s*", raw_title, maxsplit=1)
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


def is_course_list_intent(message: str) -> bool:
    msg = _norm(message).lower()
    return any(phrase in msg for phrase in COURSE_LIST_PHRASES)


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
    return _norm(message).lower() in GREETINGS


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
    if IDK_LINE_RE.match(a):
        return True
    if IDK_CONTAINS_RE.search(a):
        return True
    if len(a) < 6:
        return True
    return False


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
        return False

    context_text = (context_text or "").lower()
    context_tokens = set(re.findall(r"[a-z0-9]+", context_text))
    matched = {term for term in terms if _term_matches_context(term, context_tokens, context_text)}

    if is_out_of_domain_query(question):
        # Out-of-domain questions are allowed only when the KB text explicitly
        # contains the key topic terms; loose semantic similarity is not enough.
        return len(matched) == len(terms)

    if len(terms) <= 2:
        return len(matched) == len(terms)

    return len(matched) >= 2 and (len(matched) / len(terms)) >= 0.35


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


def _answer_supported_by_context(answer: str, context: str) -> bool:
    if _looks_like_idk(answer) or not context.strip():
        return False

    context_text = context.lower()
    context_tokens = set(re.findall(r"[a-z0-9]+", context_text))
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
) -> tuple[str, bool]:
    if not context or not citations:
        return REFUSAL_MESSAGE, False

    answer = generate_answer(
        user_message=user_message,
        context=context,
        history_messages=history_messages,
    )

    if not _answer_supported_by_context(answer, context):
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

            # for multi-word phrases, prefer exact message match
            if " " in kw:
                if message_lower == kw:
                    return item.get("answer")
                continue

            # for single-word keywords, only allow if message itself is short/simple
            if len(message_lower.split()) <= 3 and re.search(r"\b" + re.escape(kw) + r"\b", message_lower):
                return item.get("answer")

    return None

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
    }


def _store_assistant_message(
    session_id: str,
    answer: str,
    request_id: str,
    sources: list[str],
    context_used: int,
) -> None:
    put_message(
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


def _rag_answer_response(
    user_message: str,
    history_messages: list[dict],
    session_id: str,
    request_id: str,
) -> Dict:
    context, citations = _retrieve_validated_context(
        user_message,
        k=RAG_DEFAULT_K,
        max_chars=RAG_MAX_CONTEXT_CHARS,
        max_chunk_chars=RAG_MAX_CHUNK_CHARS,
    )

    if not context or not citations:
        return _refusal_response(session_id, request_id)

    answer, supported = _generated_answer_or_refusal(
        user_message=user_message,
        context=context,
        citations=citations,
        history_messages=history_messages,
    )

    if not supported:
        return _refusal_response(session_id, request_id)

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
    return {"status": "ok"}


@app.get("/history/{session_id}")
def history(
    session_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> Dict:
    _check_api_key(x_api_key)
    sid = _validate_session_id(session_id)
    msgs = get_recent_messages(session_id=sid, limit=50)
    return {"session_id": sid, "messages": msgs}


@app.post("/chat")
async def chat(
    request: ChatRequest,
    req: Request,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> Dict:
    request_id = str(uuid.uuid4())[:8]

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

        recent_before = get_recent_messages(session_id=session_id, limit=HISTORY_LIMIT)
        has_prior_history = len(recent_before) > 0

        history_messages = []
        if USE_HISTORY_FOR_CONTINUITY and has_prior_history:
            history_messages = history_to_model_messages(
                recent_before,
                max_chars=MODEL_HISTORY_MAX_CHARS,
            )

        put_message(
            session_id=session_id,
            role="user",
            text=user_message,
            request_id=request_id,
        )

        if is_greeting(user_message):
            answer = "Hello! How can I assist you today?"
            put_message(
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
            put_message(
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
            put_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                request_id=request_id,
                sources=["local"],
                context_used=0,
            )
            return _response(answer, ["local"], 0, request_id)

        # KB-only enforcement: legacy cache entries do not carry retrieval
        # context or citations, so they are not trusted as answer sources.

        if is_course_list_intent(user_message):
            ranked = rank_course_candidates(user_message, COURSE_DATA, limit=20) if COURSE_DATA else []
            titles = [r["title"] for r in ranked] if ranked else [
                _extract_primary_title(
                    (item.get("title") or item.get("course_name") or item.get("name") or "").strip()
                )
                for item in COURSE_DATA[:20]
                if (item.get("title") or item.get("course_name") or item.get("name"))
            ]

            titles = [t for t in titles if t]

            if not titles:
                return _refusal_response(session_id, request_id)

            if titles:
                answer = (
                    "Here are some available courses:\n\n"
                    + "\n".join([f"• {t}" for t in titles[:20]])
                    + "\n\nWhich one do you want details for?"
                )
            else:
                answer = "I’m not seeing course titles in the current data."

            put_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                request_id=request_id,
                sources=["courses.json"],
                context_used=1 if titles else 0,
            )
            return _response(
                answer,
                ["courses.json"],
                1 if titles else 0,
                request_id,
                citations=["courses.json"] if titles else [],
            )

        faq_answer = get_faq_answer(user_message)
        if faq_answer:
            put_message(
                session_id=session_id,
                role="assistant",
                text=faq_answer,
                request_id=request_id,
                sources=["faq"],
                context_used=0,
            )
            return _response(faq_answer, ["faq"], 0, request_id, citations=["faq"])

        if is_prompt_injection(user_message):
            return _refusal_response(session_id, request_id)

        if is_out_of_domain_query(user_message):
            return _rag_answer_response(user_message, history_messages, session_id, request_id)

        if is_procedural_query(user_message):
            return _rag_answer_response(user_message, history_messages, session_id, request_id)

        if COURSE_DATA:
            exact_course = find_exact_course_title_match(user_message, COURSE_DATA)
            if exact_course:
                context = (
                    f"[source=courses.json]\n"
                    f"Title: {exact_course['title']}\n"
                    f"Content: {exact_course['content']}"
                )
                citations = [f"courses.json | {exact_course['title']}"]

                answer, supported = _generated_answer_or_refusal(
                    user_message=(
                        f"The user selected this course title: {exact_course['display_title']}. "
                        "Give a short and direct overview based only on the provided course content. "
                        "If available, mention what the course covers, who it suits, and key themes."
                    ),
                    context=context,
                    citations=citations,
                    history_messages=history_messages,
                )

                if not supported:
                    return _refusal_response(session_id, request_id)

                put_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    request_id=request_id,
                    sources=["courses.json", "nova"],
                    context_used=1,
                )
                return _response(answer, ["courses.json", "nova"], 1, request_id, citations=citations)

        if is_course_recommendation_intent(user_message):
            ranked = rank_course_candidates(user_message, COURSE_DATA, limit=8) if COURSE_DATA else []

            if len(ranked) < 3:
                hits = _validate_retrieved_hits(user_message, retrieve_hits(user_message, k=8))
                seen_titles = {r["title"].lower() for r in ranked}

                for hit in hits:
                    title = (hit.get("title") or "").strip()
                    if not title:
                        first_line = hit.get("document", "").splitlines()[0].strip()
                        if 4 <= len(first_line) <= 120:
                            title = first_line

                    title = _extract_primary_title(title)
                    if title and title.lower() not in seen_titles:
                        ranked.append(
                            {
                                "title": title,
                                "raw_title": title,
                                "content": hit.get("document") or "",
                                "score": 1,
                            }
                        )
                        seen_titles.add(title.lower())

                    if len(ranked) >= 8:
                        break

            if ranked:
                if wants_single_recommendation(user_message) or len(ranked) == 1:
                    top = ranked[0]
                    context = f"[source=courses.json]\nTitle: {top['title']}\nContent: {top['content']}"
                    citations = [f"courses.json | {top['title']}"]
                    answer, supported = _generated_answer_or_refusal(
                        user_message=(
                            f"The user asked: {user_message}\n"
                            "Recommend only the single best-fit course from the provided context "
                            "and explain briefly why it suits the user."
                        ),
                        context=context,
                        citations=citations,
                        history_messages=history_messages,
                    )
                    if not supported:
                        return _refusal_response(session_id, request_id)

                    put_message(
                        session_id=session_id,
                        role="assistant",
                        text=answer,
                        request_id=request_id,
                        sources=["courses.json", "nova"],
                        context_used=1,
                    )
                    return _response(answer, ["courses.json", "nova"], 1, request_id, citations=citations)

                if wants_course_details(user_message):
                    top = ranked[0]
                    context = f"[source=courses.json]\nTitle: {top['title']}\nContent: {top['content']}"
                    citations = [f"courses.json | {top['title']}"]
                    answer, supported = _generated_answer_or_refusal(
                        user_message=user_message,
                        context=context,
                        citations=citations,
                        history_messages=history_messages,
                    )
                    if not supported:
                        return _refusal_response(session_id, request_id)

                    put_message(
                        session_id=session_id,
                        role="assistant",
                        text=answer,
                        request_id=request_id,
                        sources=["courses.json", "nova"],
                        context_used=1,
                    )
                    return _response(answer, ["courses.json", "nova"], 1, request_id, citations=citations)

                answer = (
                    "Here are some relevant courses:\n"
                    + "\n".join([f"- {r['title']}" for r in ranked[:5]])
                    + "\n\nWhich one do you want details for?"
                )
                put_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    request_id=request_id,
                    sources=["courses.json"],
                    context_used=1,
                )
                return _response(answer, ["courses.json"], 1, request_id, citations=["courses.json"])

            return _refusal_response(session_id, request_id)

        return _rag_answer_response(user_message, history_messages, session_id, request_id)

    except HTTPException:
        raise
    except Exception:
        logger.exception("[%s] Unhandled error in /chat", request_id)
        raise HTTPException(status_code=500, detail="Internal error")
