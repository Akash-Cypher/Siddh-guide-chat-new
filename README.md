flowchart TD
  U[User / WordPress Frontend] -->|HTTP POST /chat| API[FastAPI Backend (Uvicorn)]
  U -->|HTTP GET /health| API

  API --> G{Greeting?}
  G -->|Yes| L1[Local Greeting Response]
  G -->|No| F{FAQ Match?}
  F -->|Yes| L2[FAQ Answer from data/faq.json]
  F -->|No| RAG[RAG Retrieve Hits]

  RAG --> VDB[(Chroma Vector DB\nchroma_db/ persistent)]
  RAG --> EMB[Amazon Titan Embeddings\n(encode query + docs)]

  VDB --> VAL{Validated KB context\nwith citations?}
  VAL -->|No| REF[KB-only refusal]
  VAL -->|Yes| LLM[Amazon Bedrock Nova\n(generate grounded answer)]
  LLM --> OUT{Answer supported by KB?}
  OUT -->|No| REF
  OUT -->|Yes| API
  REF --> API
  API --> U
