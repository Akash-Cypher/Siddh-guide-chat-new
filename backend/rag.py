import os
import json
from typing import Optional, List

import boto3
import chromadb
from chromadb.config import Settings

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "sidh_guide")

# Bedrock embedding model (set via env)
# Good default for Bedrock embeddings:
# amazon.titan-embed-text-v2:0 (recommended) OR amazon.titan-embed-text-v1
BEDROCK_EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
BEDROCK_REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "ap-south-1"

_client = None
_collection = None
_bedrock = None


def _get_bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    return _bedrock


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings using Bedrock Titan Embeddings.
    """
    br = _get_bedrock()
    vectors: List[List[float]] = []

    for t in texts:
        t = (t or "").strip()
        if not t:
            vectors.append([])
            continue

        # Titan v2 expects "inputText"
        body = {"inputText": t}

        resp = br.invoke_model(
            modelId=BEDROCK_EMBED_MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )

        payload = json.loads(resp["body"].read())

        # Titan embedding response keys can differ by model version
        # v2: {"embedding": [...]}
        # v1: {"embedding": [...]}
        emb = payload.get("embedding")
        if not emb:
            # fallback keys (rare)
            emb = payload.get("vector") or payload.get("embeddings")

        if not emb:
            raise RuntimeError(f"Bedrock embedding failed: {payload}")

        vectors.append(emb)

    return vectors


def init_rag():
    global _client, _collection

    _client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    # IMPORTANT: do NOT pass embedding_function to collection.
    # We will manually provide embeddings on add/query.
    _collection = _client.get_or_create_collection(name=COLLECTION_NAME)


def _make_unique_id(filename: str, raw_id: Optional[str], i: int) -> str:
    file_key = os.path.basename(filename).replace(" ", "_")
    base = (raw_id or "").strip() or "row"
    return f"{file_key}::{base}::{i}"


def build_index_from_json_folder(json_folder: str = "data"):
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

                bump = i
                while doc_id in used_ids:
                    bump += 1
                    doc_id = _make_unique_id(filename, raw_id, bump)

                used_ids.add(doc_id)

                meta = {"source": filename, "raw_id": raw_id or "", "title": title}
                docs.append((doc_id, content, meta))

        elif isinstance(data, dict):
            content = json.dumps(data, ensure_ascii=False)
            raw_id = data.get("id") if isinstance(data.get("id"), str) else None
            doc_id = _make_unique_id(filename, raw_id, 0)

            if doc_id in used_ids:
                bump = 1
                while doc_id in used_ids:
                    doc_id = _make_unique_id(filename, raw_id, bump)
                    bump += 1

            used_ids.add(doc_id)
            meta = {"source": filename, "raw_id": raw_id or "", "title": data.get("title", "") or ""}
            docs.append((doc_id, content, meta))

    if not docs:
        print("No JSON docs found to ingest.")
        return

    ids = [d[0] for d in docs]
    documents = [d[1] for d in docs]
    metadatas = [d[2] for d in docs]

    # Embed all documents (batching optional; keep simple first)
    embeddings = _embed_texts(documents)

    _collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print(f"Ingested {len(ids)} docs into Chroma at {CHROMA_PATH}")


def retrieve_context(
    question: str,
    k: int = 3,
    max_chars: int = 1500,
    max_chunk_chars: int = 500
) -> str:
    if _collection is None:
        init_rag()

    q_emb = _embed_texts([question])[0]

    results = _collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    chunks = []
    total = 0

    for d, m in zip(docs, metas):
        if not d:
            continue

        src = (m or {}).get("source", "doc")
        title = (m or {}).get("title", "")
        header = f"[source={src}{' | ' + title if title else ''}]"

        d = d.strip().replace("\n\n", "\n")
        if len(d) > max_chunk_chars:
            d = d[:max_chunk_chars].rsplit(" ", 1)[0] + "…"

        chunk = f"{header}\n{d}"

        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining <= 0:
                break
            chunk = chunk[:remaining].rsplit(" ", 1)[0] + "…"
            chunks.append(chunk)
            break

        chunks.append(chunk)
        total += len(chunk)

    return "\n\n---\n\n".join(chunks).strip()