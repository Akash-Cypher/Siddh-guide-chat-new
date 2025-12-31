from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import init_rag, query_rag, ingest_data
from models import generate_answer
from typing import List, Dict
import json
import re
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Siddh Guide Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

faq_data = []
GREETINGS = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
qa_cache = {}

@app.on_event("startup")
async def startup_event():
    global faq_data
    init_rag()
    ingest_data()
    with open('data/faq.json', 'r') as f:
        faq_data = json.load(f)

def is_greeting(message: str) -> bool:
    return message.lower().strip() in GREETINGS

def get_faq_answer(message: str) -> str | None:
    message_lower = message.lower()
    for item in faq_data:
        if any(re.search(r'\b' + keyword + r'\b', message_lower) for keyword in item['keywords']):
            return item['answer']
    return None

@app.post("/chat")
async def chat(request: ChatRequest) -> Dict:
    try:
        user_message = request.message.strip()

        # 1. Check for greetings
        if is_greeting(user_message):
            return {
                "answer": "Hello! How can I assist you today?",
                "sources": [],
                "context_used": 0
            }

        # 2. Check cache for previously answered questions
        if user_message in qa_cache:
            return qa_cache[user_message]

        # 3. RAG query for course-related questions
        docs = query_rag(user_message)
        if docs:
            context = "\n".join([doc['text'] for doc in docs])
            answer = generate_answer(user_message, context)
            sources = [doc['metadata']['title'] for doc in docs]
            
            response = {
                "answer": answer,
                "sources": sources,
                "context_used": len(docs)
            }
            
            # Store the new answer in the cache
            qa_cache[user_message] = response
            return response

        # 4. Check for FAQ
        faq_answer = get_faq_answer(user_message)
        if faq_answer:
            return {
                "answer": faq_answer,
                "sources": ["FAQ"],
                "context_used": 0
            }

        # 5. Fallback response
        return {
            "answer": "I can answer questions about our courses and common topics like services, hiring, and pricing. How can I help you with those?",
            "sources": [],
            "context_used": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest():
    ingest_data()
    return {"status": "ingested"}

# Mount the frontend directory containing the UI
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
