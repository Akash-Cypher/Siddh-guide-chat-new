import os
import json
from typing import Optional

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


def _make_unique_id(filename: str, raw_id: Optional[str], i: int) -> str:
    """
    Guaranteed-unique Chroma ID:
    - includes filename (prevents clashes across files)
    - includes raw_id if present
    - includes row index i (prevents clashes even if content duplicates)
    """
    file_key = os.path.basename(filename).replace(" ", "_")
    base = (raw_id or "").strip() or "row"
    return f"{file_key}::{base}::{i}"


def build_index_from_json_folder(json_folder: str = "data"):
    """
    Run when docs change (or during docker build).
    Rebuilds the vector index in CHROMA_PATH.
    """
    init_rag()

    # Clear existing docs
    try:
        existing = _collection.get(include=[])
        if existing and existing.get("ids"):
            _collection.delete(ids=existing["ids"])
    except Exception:
        pass

    docs = []
    used_ids = set()

    for filename in os.listdir(json_folder):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(json_folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    content = (
                        item.get("content")
                        or item.get("answer")
                        or json.dumps(item, ensure_ascii=False)
                    )
                    raw_id = item.get("id")
                    title = item.get("title", "") or ""
                else:
                    content = json.dumps(item, ensure_ascii=False)
                    raw_id = None
                    title = ""

                doc_id = _make_unique_id(filename, raw_id, i)

                # Ultra-safety: if somehow still duplicated, keep bumping i
                bump = i
                while doc_id in used_ids:
                    bump += 1
                    doc_id = _make_unique_id(filename, raw_id, bump)

                used_ids.add(doc_id)

                meta = {
                    "source": filename,
                    "raw_id": raw_id or "",
                    "title": title
                }

                docs.append((doc_id, content, meta))

        elif isinstance(data, dict):
            content = json.dumps(data, ensure_ascii=False)
            raw_id = data.get("id") if isinstance(data.get("id"), str) else None
            doc_id = _make_unique_id(filename, raw_id, 0)

            if doc_id in used_ids:
                # rare but possible if multiple dict files same name etc
                bump = 1
                while doc_id in used_ids:
                    doc_id = _make_unique_id(filename, raw_id, bump)
                    bump += 1

            used_ids.add(doc_id)

            meta = {
                "source": filename,
                "raw_id": raw_id or "",
                "title": data.get("title", "") or ""
            }

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
    Returns a context string to feed the LLM.
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
