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

# -------------------------
# App + Logging
# -------------------------
app = FastAPI(title="Siddh Guide Chatbot")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("siddh_guide")

# -------------------------
# Config
# -------------------------
ALLOWED_ORIGINS = [
    "https://siddhantaknowledge.org",
    "https://www.siddhantaknowledge.org",
]

MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "600"))
ENFORCE_API_KEY = os.getenv("ENFORCE_API_KEY", "1") == "1"

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request schema
# -------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


faq_data: list[dict] = []

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
    "I’m Siddh Guide 🤝 — a helpful assistant by Siddhanta Knowledge Foundation. "
    "I can help you explore IKS courses, suggest what fits your interests, "
    "and guide you to the right certified programs."
)

# -------------------------
# NEW: Capability intent (local reply)
# -------------------------
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

# -------------------------
# Anti “I don’t know” guardrail
# -------------------------
IDK_LINE_RE = re.compile(
    r"^\s*i\s*(do\s*not|don't|dont)\s*know\s*[\.\!\?]*\s*$",
    re.IGNORECASE,
)

IDK_CONTAINS_RE = re.compile(
    r"\b(i\s*(do\s*not|don't|dont)\s*know)\b",
    re.IGNORECASE,
)

def _looks_like_idk(answer: str) -> bool:
    """
    Catch standalone and low-effort answers that ruin UX.
    """
    if not answer:
        return True

    a = answer.strip()
    if IDK_LINE_RE.match(a):
        return True

    # If model includes it inside a sentence, we still treat it as bad.
    if IDK_CONTAINS_RE.search(a):
        return True

    # Super short replies often mean it failed.
    if len(a) < 6:
        return True

    return False


def _fallback_reply(user_message: str, context: str) -> str:
    """
    Fallback that never says "I don't know" and moves the conversation forward.
    If context exists, try to suggest up to 3 course-like lines from context.
    """
    msg = (user_message or "").strip().lower()

    # If we have context, try extracting up to 3 "course-ish" items
    suggestions: list[str] = []
    if context and context.strip():
        lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
        for ln in lines:
            if ln.startswith("[source="):
                continue
            # Heuristic: course lines often contain these keywords or look like titles
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

    # No context or no usable suggestions
    if "law" in msg or "llb" in msg or "juris" in msg:
        return (
            "I don’t have that specific detail in my current course list yet. "
            "Are you looking for law-focused IKS, legal philosophy, or governance/ethics?"
        )

    return (
        "I don’t have that in my current course list yet. "
        "What are you trying to find—course recommendations, eligibility, or syllabus details?"
    )


@app.on_event("startup")
async def startup_event():
    global faq_data
    with open("data/faq.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    init_rag()
    logger.info("Startup complete: FAQ loaded + RAG initialized")


def _norm(text: str) -> str:
    return " ".join((text or "").strip().split())


def is_greeting(message: str) -> bool:
    return _norm(message).lower() in GREETINGS


def is_about_bot(message: str) -> bool:
    msg = _norm(message).lower()
    return any(k in msg for k in ABOUT_BOT_KEYWORDS)


# -------------------------
# NEW: capability checker
# -------------------------
def is_capability_question(message: str) -> bool:
    msg = _norm(message).lower()
    return any(k in msg for k in CAPABILITY_KEYWORDS)


def get_faq_answer(message: str) -> Optional[str]:
    """
    Matches:
    - multi-word keywords via substring (e.g. 'siddhanta knowledge foundation')
    - single-word keywords via word-boundary regex
    """
    message_lower = (message or "").lower()

    for item in faq_data:
        for keyword in item.get("keywords", []):
            kw = (keyword or "").strip().lower()
            if not kw:
                continue

            # multi-word phrase: simple substring match
            if " " in kw:
                if kw in message_lower:
                    return item.get("answer")
                continue

            # single word: word boundary match
            if re.search(r"\b" + re.escape(kw) + r"\b", message_lower):
                return item.get("answer")

    return None


def _should_cache(answer: str) -> bool:
    """
    Don't cache fallback / refusal-ish / low-signal answers.
    """
    ans = (answer or "").strip().lower()
    if not ans:
        return False

    # block "i don't know" variants
    if IDK_CONTAINS_RE.search(ans):
        return False

    # block our own fallback style too (so cache doesn’t become generic)
    if "i don’t have that" in ans or "i don't have that" in ans:
        return False

    return True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(
    request: ChatRequest,
    req: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> Dict:
    """
    Flow:
    greeting -> about bot -> FAQ -> cache -> RAG -> Bedrock -> cache write
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        # -------------------------
        # API key auth
        # -------------------------
        chat_api_key = os.getenv("CHAT_API_KEY")  # read at runtime
        if ENFORCE_API_KEY and chat_api_key:
            if not x_api_key or x_api_key != chat_api_key:
                raise HTTPException(status_code=401, detail="Unauthorized")

        user_message = _norm(request.message)
        if not user_message:
            raise HTTPException(status_code=400, detail="message is required")

        if len(user_message) > MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"message too long (max {MAX_MESSAGE_CHARS} chars)",
            )

        # normalize common variant
        user_message = user_message.replace("knowledgebase", "knowledge base")

        origin = req.headers.get("origin", "")
        logger.info(
            f"[{request_id}] /chat origin={origin} session={request.session_id} msg_len={len(user_message)}"
        )

        # 1) Greetings -> local
        if is_greeting(user_message):
            return {
                "answer": "Hello! How can I assist you today?",
                "sources": ["local"],
                "context_used": 0,
                "request_id": request_id,
            }

        # 1.5) About bot -> local
        if is_about_bot(user_message):
            return {
                "answer": ABOUT_BOT_REPLY,
                "sources": ["local"],
                "context_used": 0,
                "request_id": request_id,
            }

        # 1.6) Capability -> local (NEW)
        if is_capability_question(user_message):
            return {
                "answer": CAPABILITY_REPLY,
                "sources": ["local"],
                "context_used": 0,
                "request_id": request_id,
            }

        # 2) FAQ -> local
        faq_answer = get_faq_answer(user_message)
        if faq_answer:
            return {
                "answer": faq_answer,
                "sources": ["FAQ"],
                "context_used": 0,
                "request_id": request_id,
            }

        # 3) Cache -> exact match
        cached = get_cached(user_message)
        if cached:
            return {
                "answer": cached,
                "sources": ["cache"],
                "context_used": 0,
                "request_id": request_id,
            }

        # 4) RAG retrieve context (clamped)
        context = retrieve_context(
            user_message,
            k=3,
            max_chars=1500,
            max_chunk_chars=500,
        )
        context_used = 1 if (context and context.strip()) else 0

        # 5) Bedrock/Nova with context
        answer = generate_answer(user_message, context=context)

        # 5.5) Guardrail: block "I don't know" outputs
        if _looks_like_idk(answer):
            answer = _fallback_reply(user_message, context)

        # 6) Store in cache (skip fallback answers)
        if _should_cache(answer):
            put_cached(user_message, answer)

        return {
            "answer": answer,
            "sources": ["rag", "nova"],
            "context_used": context_used,
            "request_id": request_id,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception(f"[{request_id}] Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
