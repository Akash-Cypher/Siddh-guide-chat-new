import os
import json
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "sidh_guide")

_client = None
_collection = None


def init_rag():
    global _client, _collection
    _client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )

    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )

def build_index_from_json_folder(json_folder: str = "data"):
    """
    Run this ONE TIME when docs change (or during docker build).
    It (re)creates the vector index in CHROMA_PATH.
    """
    init_rag()

    # Clear existing docs (optional)
    try:
        existing = _collection.get(include=[])
        if existing and existing.get("ids"):
            _collection.delete(ids=existing["ids"])
    except Exception:
        pass

    docs = []
    for filename in os.listdir(json_folder):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(json_folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Supports list-of-objects json (your current format)
        if isinstance(data, list):
            for i, item in enumerate(data):
                content = (
                    item.get("content")
                    or item.get("answer")
                    or json.dumps(item, ensure_ascii=False)
                )
                doc_id = item.get("id") or f"{filename}:{i}"
                meta = {
                    "source": filename,
                    "id": doc_id,
                    "title": item.get("title", "")
                }
                docs.append((doc_id, content, meta))

        # Supports dict json too
        elif isinstance(data, dict):
            doc_id = filename
            content = json.dumps(data, ensure_ascii=False)
            meta = {"source": filename, "id": doc_id, "title": ""}
            docs.append((doc_id, content, meta))

    if not docs:
        print("No JSON docs found to ingest.")
        return

    ids = [d[0] for d in docs]
    documents = [d[1] for d in docs]
    metadatas = [d[2] for d in docs]

    _collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Ingested {len(ids)} docs into Chroma at {CHROMA_PATH}")

def retrieve_context(question: str, k: int = 3) -> str:
    """
    Returns a context string to feed Titan.
    """
    if _collection is None:
        init_rag()

    results = _collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    chunks = []
    for d, m in zip(docs, metas):
        src = (m or {}).get("source", "doc")
        title = (m or {}).get("title", "")
        header = f"[source={src}{' | ' + title if title else ''}]"
        chunks.append(f"{header}\n{d}")

    return "\n\n---\n\n".join(chunks).strip()
