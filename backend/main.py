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

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,100}$")

GREETINGS = {"hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"}

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

IDK_LINE_RE = re.compile(r"^\s*i\s*(do\s*not|don't|dont)\s*know\s*[\.\!\?]*\s*$", re.IGNORECASE)
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
    r"\b(course|courses|program|programs|certificate|certification|batch|batches|enrol|enroll|admission|syllabus)\b",
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


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str = Field(..., min_length=1, max_length=SESSION_ID_MAX_LEN)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global faq_data

    with FAQ_PATH.open("r", encoding="utf-8") as f:
        faq_data = json.load(f)

    init_rag()
    logger.info("Startup complete: FAQ loaded + RAG initialized")
    yield


app = FastAPI(title=APP_TITLE, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
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


def is_course_intent(message: str) -> bool:
    msg = _norm(message).lower()
    if any(p in msg for p in COURSE_LIST_PHRASES):
        return True
    return bool(COURSE_INTENT_RE.search(msg))


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


def _fallback_reply(user_message: str, context: str) -> str:
    msg = (user_message or "").strip().lower()

    suggestions: list[str] = []
    if context and context.strip():
        lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
        for ln in lines:
            if ln.startswith("[source="):
                continue
            if any(k in ln.lower() for k in ["course", "law", "jurisprudence", "governance", "philosophy", "ethics", "iks"]):
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
            "What’s your goal—practice, judiciary prep, or research?"
        )

    if "law" in msg or "llb" in msg or "juris" in msg:
        return (
            "I don’t have that specific detail in my current course list yet. "
            "Are you looking for law-focused IKS, legal philosophy, or governance and ethics?"
        )

    return (
        "I don’t have that in my current course list yet. "
        "Are you trying to find course recommendations, eligibility, or syllabus details?"
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

        # Store user message after validation
        put_message(
            session_id=session_id,
            role="user",
            text=user_message,
            request_id=request_id,
        )

        if is_greeting(user_message):
            answer = "Hello! How can I assist you today?"
            put_message(session_id=session_id, role="assistant", text=answer, request_id=request_id, sources=["local"], context_used=0)
            return _response(answer, ["local"], 0, request_id)

        if is_about_bot(user_message):
            answer = ABOUT_BOT_REPLY
            put_message(session_id=session_id, role="assistant", text=answer, request_id=request_id, sources=["local"], context_used=0)
            return _response(answer, ["local"], 0, request_id)

        if is_capability_question(user_message):
            answer = CAPABILITY_REPLY
            put_message(session_id=session_id, role="assistant", text=answer, request_id=request_id, sources=["local"], context_used=0)
            return _response(answer, ["local"], 0, request_id)

        # For correctness, only use cache for standalone turns
        if not has_prior_history and not _looks_like_followup(user_message):
            cached = get_cached(user_message)
            if cached:
                put_message(session_id=session_id, role="assistant", text=cached, request_id=request_id, sources=["cache"], context_used=0)
                return _response(cached, ["cache"], 0, request_id)

        if is_course_intent(user_message):
            hits = retrieve_hits(user_message, k=8)
            titles: list[str] = []
            seen = set()

            for hit in hits:
                title = (hit.get("title") or "").strip()
                if not title:
                    first_line = hit.get("document", "").splitlines()[0].strip()
                    if 4 <= len(first_line) <= 80:
                        title = first_line

                if title and title.lower() not in seen:
                    titles.append(title)
                    seen.add(title.lower())

                if len(titles) >= 10:
                    break

            if titles:
                answer = (
                    "Here are some courses I can see:\n"
                    + "\n".join([f"- {t}" for t in titles])
                    + "\n\nWhich one do you want details for?"
                )
            else:
                answer = (
                    "I can share the course list, but I’m not seeing clear course titles in the current data. "
                    "Which domain do you want—Law, Management, Education, or Sanskrit and Grammar?"
                )

            put_message(session_id=session_id, role="assistant", text=answer, request_id=request_id, sources=["rag_list"], context_used=1)
            return _response(answer, ["rag_list"], 1, request_id)

        faq_answer = get_faq_answer(user_message)
        if faq_answer:
            put_message(session_id=session_id, role="assistant", text=faq_answer, request_id=request_id, sources=["faq"], context_used=0)
            return _response(faq_answer, ["faq"], 0, request_id)

        context = retrieve_context(
            user_message,
            k=RAG_DEFAULT_K,
            max_chars=RAG_MAX_CONTEXT_CHARS,
            max_chunk_chars=RAG_MAX_CHUNK_CHARS,
        )
        context_used = 1 if context else 0

        answer = generate_answer(
            user_message=user_message,
            context=context,
            history_messages=history_messages,
        )

        if _looks_like_idk(answer):
            answer = _fallback_reply(user_message, context)

        sources = ["rag", "nova"] if context_used else ["nova"]

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)