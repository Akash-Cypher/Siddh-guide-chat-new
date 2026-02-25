from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cache import get_cached, put_cached
from models import generate_answer
from rag import init_rag, retrieve_context
from chat_store import put_message, get_recent_messages, format_history_for_model

app = FastAPI(title="Siddh Guide Chatbot")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("siddh_guide")

ALLOWED_ORIGINS = [
    "https://siddhantaknowledge.org",
    "https://www.siddhantaknowledge.org",
]

MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "600"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "8"))

ENFORCE_API_KEY = os.getenv("ENFORCE_API_KEY", "1") == "1"
USE_HISTORY_FOR_CONTINUITY = os.getenv("USE_HISTORY_FOR_CONTINUITY", "1") == "1"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

faq_data: list[dict] = []

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
    "I’m Siddh Guide 🤝 — a helpful assistant by Siddhanta Knowledge Foundation. "
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
    "Tell me your field (like law, education, management) and your goal."
)

IDK_LINE_RE = re.compile(r"^\s*i\s*(do\s*not|don't|dont)\s*know\s*[\.\!\?]*\s*$", re.IGNORECASE)
IDK_CONTAINS_RE = re.compile(r"\b(i\s*(do\s*not|don't|dont)\s*know)\b", re.IGNORECASE)

def _norm(text: str) -> str:
    return " ".join((text or "").strip().split())

# -------------------------
# Follow-up detector (FIXED)
# -------------------------
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

def _looks_like_followup(message: str) -> bool:
    msg = _norm(message).lower()

    if msg in SHORT_FOLLOWUPS:
        return True

    if FOLLOWUP_RE.search(msg):
        return True

    if msg.startswith(("what about", "and ", "then ", "so ", "ok ", "okay ")):
        return True

    return False

# -------------------------
# Course intent override
# -------------------------
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

def is_course_intent(message: str) -> bool:
    msg = _norm(message).lower()
    if any(p in msg for p in COURSE_LIST_PHRASES):
        return True
    if COURSE_INTENT_RE.search(msg):
        return True
    return False

def extract_course_titles_from_context(context: str, max_items: int = 10) -> list[str]:
    if not context:
        return []

    titles: list[str] = []
    seen = set()

    for m in re.findall(r"\"([^\"]{3,120})\"", context):
        t = _norm(m)
        if t and t.lower() not in seen:
            titles.append(t)
            seen.add(t.lower())
        if len(titles) >= max_items:
            return titles

    for line in context.splitlines():
        line = line.strip()
        if not line or line.startswith("[source="):
            continue
        if 4 <= len(line) <= 80:
            t = _norm(line)
            if t.lower() not in seen:
                titles.append(t)
                seen.add(t.lower())
        if len(titles) >= max_items:
            break

    return titles

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
            "I don’t have that exact detail in the current course data I’m using. "
            "Here are a few relevant options I can suggest:\n"
            f"{bullets}\n"
            "What’s your goal—practice, judiciary prep, or research?"
        )

    if "law" in msg or "llb" in msg or "juris" in msg:
        return (
            "I don’t have that specific detail in my current course list yet. "
            "Are you looking for law-focused IKS, legal philosophy, or governance/ethics?"
        )

    return (
        "I don’t have that in my current course list yet. "
        "What are you trying to find—course recommendations, eligibility, or syllabus details?"
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

def _check_api_key(x_api_key: str | None):
    chat_api_key = os.getenv("CHAT_API_KEY")
    if ENFORCE_API_KEY and chat_api_key:
        if not x_api_key or x_api_key != chat_api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")

@app.on_event("startup")
async def startup_event():
    global faq_data
    with open("data/faq.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    init_rag()
    logger.info("Startup complete: FAQ loaded + RAG initialized")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/history/{session_id}")
def history(
    session_id: str,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> Dict:
    _check_api_key(x_api_key)
    sid = _norm(session_id) or "default"
    msgs = get_recent_messages(session_id=sid, limit=50)
    return {"session_id": sid, "messages": msgs}

@app.post("/chat")
async def chat(
    request: ChatRequest,
    req: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> Dict:
    request_id = str(uuid.uuid4())[:8]

    try:
        _check_api_key(x_api_key)

        user_message = _norm(request.message)
        session_id = _norm(request.session_id) or "default"

        if not user_message:
            raise HTTPException(status_code=400, detail="message is required")

        if len(user_message) > MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"message too long (max {MAX_MESSAGE_CHARS} chars)",
            )

        user_message = user_message.replace("knowledgebase", "knowledge base")

        origin = req.headers.get("origin", "")
        logger.info(f"[{request_id}] /chat origin={origin} session={session_id} msg_len={len(user_message)}")

        # ---- fetch prior history BEFORE writing new user message
        recent_before = get_recent_messages(session_id=session_id, limit=HISTORY_LIMIT)
        has_prior_history = len(recent_before) > 0

        history_text = ""
        if USE_HISTORY_FOR_CONTINUITY and has_prior_history:
            history_text = format_history_for_model(recent_before, max_chars=1200)

        # ---- store user message
        put_message(session_id=session_id, role="user", text=user_message, request_id=request_id)

        # 1) Greetings
        if is_greeting(user_message):
            answer = "Hello! How can I assist you today?"
            put_message(session_id=session_id, role="assistant", text=answer, request_id=request_id, sources=["local"], context_used=0)
            return {"answer": answer, "sources": ["local"], "context_used": 0, "request_id": request_id}

        # 1.5) About bot
        if is_about_bot(user_message):
            answer = ABOUT_BOT_REPLY
            put_message(session_id=session_id, role="assistant", text=answer, request_id=request_id, sources=["local"], context_used=0)
            return {"answer": answer, "sources": ["local"], "context_used": 0, "request_id": request_id}

        # 1.6) Capability
        if is_capability_question(user_message):
            answer = CAPABILITY_REPLY
            put_message(session_id=session_id, role="assistant", text=answer, request_id=request_id, sources=["local"], context_used=0)
            return {"answer": answer, "sources": ["local"], "context_used": 0, "request_id": request_id}

        # 2) Cache read (safe only if no history OR not follow-up)
        cached = get_cached(user_message)
        safe_to_use_cache = (not has_prior_history) or (not _looks_like_followup(user_message))

        if cached and safe_to_use_cache:
            answer = cached
            put_message(session_id=session_id, role="assistant", text=answer, request_id=request_id, sources=["cache"], context_used=0)
            return {"answer": answer, "sources": ["cache"], "context_used": 0, "request_id": request_id}

        # 3) Course intent override
        if is_course_intent(user_message):
            context_for_list = retrieve_context(user_message, k=8, max_chars=2800, max_chunk_chars=500)
            titles = extract_course_titles_from_context(context_for_list, max_items=10)

            if titles:
                answer = (
                    "Here are some courses I can see:\n"
                    + "\n".join([f"- {t}" for t in titles])
                    + "\n\nWhich one do you want details for?"
                )
            else:
                answer = (
                    "I can share the course list, but I’m not seeing course titles in my current data yet. "
                    "Which domain do you want—Law, Management, Education, or Sanskrit/Grammar?"
                )

            put_message(session_id=session_id, role="assistant", text=answer, request_id=request_id, sources=["rag_list"], context_used=1)
            return {"answer": answer, "sources": ["rag_list"], "context_used": 1, "request_id": request_id}

        # 4) FAQ
        faq_answer = get_faq_answer(user_message)
        if faq_answer:
            answer = faq_answer
            put_message(session_id=session_id, role="assistant", text=answer, request_id=request_id, sources=["FAQ"], context_used=0)
            return {"answer": answer, "sources": ["FAQ"], "context_used": 0, "request_id": request_id}

        # 5) RAG
        context = retrieve_context(user_message, k=3, max_chars=1500, max_chunk_chars=500)
        context_used = 1 if (context and context.strip()) else 0

        # 6) Model prompt WITH continuity
        prompt_for_model = user_message
        if USE_HISTORY_FOR_CONTINUITY and history_text:
            prompt_for_model = f"Conversation so far:\n{history_text}\n\nUser: {user_message}"

        answer = generate_answer(prompt_for_model, context=context)

        if _looks_like_idk(answer):
            answer = _fallback_reply(user_message, context)

        sources = ["nova"]
        if context_used:
            sources = ["rag", "nova"]

        put_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            request_id=request_id,
            sources=sources,
            context_used=context_used,
        )

        # 7) Cache write (cache only reusable, standalone questions)
        if _should_cache(answer) and (not _looks_like_followup(user_message)):
            put_cached(user_message, answer)

        return {"answer": answer, "sources": sources, "context_used": context_used, "request_id": request_id}

    except HTTPException:
        raise
    except Exception:
        logger.exception(f"[{request_id}] Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)