from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import init_rag, query_rag, ingest_data
from models import generate_answer
from typing import List, Dict

app = FastAPI(title="Sidh Guide Chatbot")

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

@app.on_event("startup")
async def startup_event():
    init_rag()
    ingest_data()

@app.post("/chat")
async def chat(request: ChatRequest) -> Dict:
    try:
        # RAG query
        docs = query_rag(request.message)
        context = "\n".join([doc['text'] for doc in docs])
        
        # Generate answer
        answer = generate_answer(request.message, context)
        sources = [doc['metadata']['title'] for doc in docs]
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest():
    ingest_data()
    return {"status": "ingested"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
