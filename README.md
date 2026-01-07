flowchart TD
  U[User / WordPress Frontend] -->|HTTP POST /chat| API[FastAPI Backend (Uvicorn)]
  U -->|HTTP GET /health| API

  API --> G{Greeting?}
  G -->|Yes| L1[Local Greeting Response]
  G -->|No| F{FAQ Match?}
  F -->|Yes| L2[FAQ Answer from data/faq.json]
  F -->|No| C{Cached Q/A?}

  C -->|Yes| L3[Return Cached Answer]
  C -->|No| RAG[RAG Retrieve Context]
  RAG --> VDB[(Chroma Vector DB\nchroma_db/ persistent)]
  RAG --> EMB[Amazon Titan Embeddings\n(encode query + docs)]

  VDB --> CTX[Top-K Context Chunks]
  CTX --> LLM[Amazon Bedrock Titan Text\n(generate final answer)]
  LLM --> API
  API -->|Store Q/A in cache| CACHE[(Cache Store\n(in-memory/Redis optional))]
  API --> U
