from __future__ import annotations

import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cache import get_cached, put_cached
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
    SESSION_ID_MAX_LEN,
    USE_HISTORY_FOR_CONTINUITY,
)
from models import generate_answer
from rag import init_rag, retrieve_context, retrieve_hits

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("siddh_guide")

faq_data: list[dict] = []
COURSE_DATA: list[dict] = []

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,100}$")

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

        title = raw_title.split("—")[0].strip()
        if not title or title.lower() in seen:
            continue

        content = (item.get("content") or "").strip()
        searchable_title = title.lower()
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

        # slight penalty for extremely broad matches with no title relevance
        if matched_terms > 0 and all(term not in searchable_title for term in expanded_terms):
            score -= 1

        if score > 0:
            ranked.append(
                {
                    "title": title,
                    "content": content,
                    "score": score,
                }
            )
            seen.add(title.lower())

    ranked.sort(key=lambda x: (-x["score"], x["title"].lower()))
    return ranked[:limit]


def _fallback_reply(user_message: str, context: str) -> str:
    msg = (user_message or "").strip().lower()

    suggestions: list[str] = []
    if context and context.strip():
        lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
        for ln in lines:
            if ln.startswith("[source="):
                continue
            if any(
                k in ln.lower()
                for k in [
                    "course",
                    "law",
                    "jurisprudence",
                    "governance",
                    "philosophy",
                    "ethics",
                    "iks",
                    "design",
                    "education",
                ]
            ):
                clean = ln
                if len(clean) > 90:
                    clean = clean[:90].rsplit(" ", 1)[0] + "…"
                suggestions.append(clean)
            if len(suggestions) >= 3:
                break

    if suggestions:
        bullets = "\n".join([f"- {s}" for s in suggestions[:3]])
        return (
            "I don’t have that exact detail in the current course data I’m using.\n"
            f"{bullets}\n"
            "Do you want recommendations, syllabus details, or eligibility?"
        )

    if "law" in msg or "llb" in msg or "juris" in msg:
        return (
            "I can help with law-related IKS course recommendations. "
            "Do you want the best course to start with or a short list?"
        )

    if "architect" in msg or "architecture" in msg or "design" in msg:
        return (
            "I can help with design and architecture-related IKS course recommendations. "
            "Do you want one best-fit course or a short list?"
        )

    return (
        "I can help with course recommendations, syllabus details, eligibility, or enrolment guidance. "
        "Tell me your field or goal."
    )


def get_faq_answer(message: str) -> Optional[str]:
    message_lower = (message or "").lower()

    for item in faq_data:
        for keyword in item.get("keywords", []):
            kw = (keyword or "").strip().lower()
            if not kw:
                continue

            if " " in kw:
                if kw in message_lower:
                    return item.get("answer")
                continue

            if re.search(r"\b" + re.escape(kw) + r"\b", message_lower):
                return item.get("answer")

    return None


def _should_cache(answer: str) -> bool:
    ans = (answer or "").strip().lower()
    if not ans:
        return False
    if IDK_CONTAINS_RE.search(ans):
        return False
    if "i don’t have that" in ans or "i don't have that" in ans:
        return False
    return True


def _response(answer: str, sources: list[str], context_used: int, request_id: str) -> Dict:
    return {
        "answer": answer,
        "sources": sources,
        "context_used": context_used,
        "request_id": request_id,
    }


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

        if not has_prior_history and not _looks_like_followup(user_message):
            cached = get_cached(user_message)
            if cached:
                put_message(
                    session_id=session_id,
                    role="assistant",
                    text=cached,
                    request_id=request_id,
                    sources=["cache"],
                    context_used=0,
                )
                return _response(cached, ["cache"], 0, request_id)

        # Exact course-list style asks
        if is_course_list_intent(user_message):
            ranked = rank_course_candidates(user_message, COURSE_DATA, limit=20) if COURSE_DATA else []
            titles = [r["title"] for r in ranked] if ranked else [
                (item.get("title") or item.get("course_name") or item.get("name") or "").strip()
                for item in COURSE_DATA[:20]
                if (item.get("title") or item.get("course_name") or item.get("name"))
            ]

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
            return _response(answer, ["courses.json"], 1 if titles else 0, request_id)

        # FAQ and procedural queries should not be swallowed by recommendation logic
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
            return _response(faq_answer, ["faq"], 0, request_id)

        if is_procedural_query(user_message):
            context = retrieve_context(
                user_message,
                k=RAG_DEFAULT_K,
                max_chars=RAG_MAX_CONTEXT_CHARS,
                max_chunk_chars=RAG_MAX_CHUNK_CHARS,
            )
            context_used = 1 if context else 0
            sources = ["rag", "nova"] if context_used else ["nova"]

            answer = generate_answer(
                user_message=user_message,
                context=context,
                history_messages=history_messages,
            )

            if _looks_like_idk(answer):
                answer = _fallback_reply(user_message, context)

            put_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                request_id=request_id,
                sources=sources,
                context_used=context_used,
            )

            if (not has_prior_history) and (not _looks_like_followup(user_message)) and _should_cache(answer):
                put_cached(user_message, answer)

            return _response(answer, sources, context_used, request_id)

        # Dynamic recommendation flow
        if is_course_recommendation_intent(user_message):
            ranked = rank_course_candidates(user_message, COURSE_DATA, limit=8) if COURSE_DATA else []

            # Supplement with RAG only if direct ranking is weak
            if len(ranked) < 3:
                hits = retrieve_hits(user_message, k=8)
                seen_titles = {r["title"].lower() for r in ranked}

                for hit in hits:
                    title = (hit.get("title") or "").strip()
                    if not title:
                        first_line = hit.get("document", "").splitlines()[0].strip()
                        if 4 <= len(first_line) <= 80:
                            title = first_line

                    title = title.split("—")[0].strip()
                    if title and title.lower() not in seen_titles:
                        ranked.append(
                            {
                                "title": title,
                                "content": hit.get("document") or "",
                                "score": 1,
                            }
                        )
                        seen_titles.add(title.lower())

                    if len(ranked) >= 8:
                        break

            if ranked:
                # user asked for only one best-fit course
                if wants_single_recommendation(user_message) or len(ranked) == 1:
                    top = ranked[0]
                    context = f"[source=courses.json]\nTitle: {top['title']}\nContent: {top['content']}"
                    answer = generate_answer(
                        user_message=(
                            f"The user asked: {user_message}\n"
                            "Recommend only the single best-fit course from the provided context "
                            "and explain briefly why it suits the user."
                        ),
                        context=context,
                        history_messages=history_messages,
                    )
                    put_message(
                        session_id=session_id,
                        role="assistant",
                        text=answer,
                        request_id=request_id,
                        sources=["courses.json", "nova"],
                        context_used=1,
                    )
                    return _response(answer, ["courses.json", "nova"], 1, request_id)

                # user asked for details
                if wants_course_details(user_message):
                    top = ranked[0]
                    context = f"[source=courses.json]\nTitle: {top['title']}\nContent: {top['content']}"
                    answer = generate_answer(
                        user_message=user_message,
                        context=context,
                        history_messages=history_messages,
                    )
                    put_message(
                        session_id=session_id,
                        role="assistant",
                        text=answer,
                        request_id=request_id,
                        sources=["courses.json", "nova"],
                        context_used=1,
                    )
                    return _response(answer, ["courses.json", "nova"], 1, request_id)

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
                return _response(answer, ["courses.json"], 1, request_id)

            answer = (
                "I can help with course recommendations, but I’m not seeing strong matches yet. "
                "Tell me your field like Law, Architecture, Management, Education, or Sanskrit."
            )
            put_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                request_id=request_id,
                sources=["courses.json"],
                context_used=0,
            )
            return _response(answer, ["courses.json"], 0, request_id)

        # General RAG + model flow
        context = retrieve_context(
            user_message,
            k=RAG_DEFAULT_K,
            max_chars=RAG_MAX_CONTEXT_CHARS,
            max_chunk_chars=RAG_MAX_CHUNK_CHARS,
        )
        context_used = 1 if context else 0
        sources = ["rag", "nova"] if context_used else ["nova"]

        answer = generate_answer(
            user_message=user_message,
            context=context,
            history_messages=history_messages,
        )

        if _looks_like_idk(answer):
            answer = _fallback_reply(user_message, context)

        put_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            request_id=request_id,
            sources=sources,
            context_used=context_used,
        )

        if (not has_prior_history) and (not _looks_like_followup(user_message)) and _should_cache(answer):
            put_cached(user_message, answer)

        return _response(answer, sources, context_used, request_id)

    except HTTPException:
        raise
    except Exception:
        logger.exception("[%s] Unhandled error in /chat", request_id)
        raise HTTPException(status_code=500, detail="Internal error")
