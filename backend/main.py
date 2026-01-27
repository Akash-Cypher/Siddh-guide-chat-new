from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import json
import re
import logging
import os
import uuid

from models import generate_answer
from rag import init_rag, retrieve_context
from cache import get_cached, put_cached

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

# Cost + abuse control
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "600"))  # keep reasonable
ENFORCE_API_KEY = os.getenv("ENFORCE_API_KEY", "1") == "1"     # default ON

# -------------------------
# CORS (ok for testing; WP proxy makes this less critical)
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

faq_data = []

GREETINGS = [
    "hello", "hi", "hey", "greetings",
    "good morning", "good afternoon", "good evening"
]

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
    "I guide you through IKS courses, explain what suits your background, "
    "and help you choose the right certified programs under the Ministry of Education."
)

@app.on_event("startup")
async def startup_event():
    global faq_data
    with open("data/faq.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    init_rag()
    logger.info("Startup complete: FAQ loaded + RAG initialized")

def is_greeting(message: str) -> bool:
    return message.strip().lower() in GREETINGS

def is_about_bot(message: str) -> bool:
    msg = (message or "").strip().lower()
    return any(k in msg for k in ABOUT_BOT_KEYWORDS)

def get_faq_answer(message: str) -> str | None:
    message_lower = message.lower()
    for item in faq_data:
        for keyword in item.get("keywords", []):
            if re.search(r"\b" + re.escape(keyword) + r"\b", message_lower):
                return item.get("answer")
    return None

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
    Main chatbot endpoint:
    greeting -> about bot -> FAQ -> cache -> RAG -> Bedrock -> cache write

    API key:
    - Enforced if ENFORCE_API_KEY=1 and CHAT_API_KEY is set in env (Secrets Manager in App Runner).
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        # -------------------------
        # API key auth (recommended when using WP proxy)
        # -------------------------
        chat_api_key = os.getenv("CHAT_API_KEY")  # read at runtime
        if ENFORCE_API_KEY and chat_api_key:
            if not x_api_key or x_api_key != chat_api_key:
                raise HTTPException(status_code=401, detail="Unauthorized")

        user_message = (request.message or "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="message is required")

        if len(user_message) > MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"message too long (max {MAX_MESSAGE_CHARS} chars)"
            )

        # Optional normalize common variant
        user_message = user_message.replace("knowledgebase", "knowledge base")

        # Log request (avoid logging full user message)
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
                "request_id": request_id
            }

        # 1.5) About bot -> local (THIS IS THE MISSING PART)
        if is_about_bot(user_message):
            return {
                "answer": ABOUT_BOT_REPLY,
                "sources": ["local"],
                "context_used": 0,
                "request_id": request_id
            }

        # 2) FAQ -> local
        faq_answer = get_faq_answer(user_message)
        if faq_answer:
            return {
                "answer": faq_answer,
                "sources": ["FAQ"],
                "context_used": 0,
                "request_id": request_id
            }

        # 3) Cache -> exact match
        cached = get_cached(user_message)
        if cached:
            return {
                "answer": cached,
                "sources": ["cache"],
                "context_used": 0,
                "request_id": request_id
            }

        # 4) RAG retrieve context (clamped for cost + shorter replies)
        context = retrieve_context(user_message, k=3, max_chars=1500, max_chunk_chars=500)
        context_used = 1 if (context and context.strip()) else 0

        # 5) Nova with context
        answer = generate_answer(user_message, context=context)

        # 6) Store in cache (skip useless answers)
        if answer and answer.strip().lower() not in ["i don't know.", "i dont know."]:
            put_cached(user_message, answer)

        return {
            "answer": answer,
            "sources": ["rag", "nova"],
            "context_used": context_used,
            "request_id": request_id
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception(f"[{request_id}] Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
