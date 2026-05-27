import json
import logging
import os
from typing import Dict, List, Optional

import boto3
import chromadb
from chromadb.config import Settings

from config import (
    AWS_BOTO_CONFIG,
    AWS_REGION,
    BEDROCK_EMBED_MODEL_ID,
    CHROMA_COLLECTION,
    CHROMA_PATH,
    RAG_MAX_DISTANCE,
)

logger = logging.getLogger("siddh_guide.rag")

_client = None
_collection = None
_bedrock = None


def _get_bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=AWS_BOTO_CONFIG)
    return _bedrock


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def init_rag() -> None:
    _get_collection()


def _embed_texts(texts: List[str]) -> List[List[float]]:
    br = _get_bedrock()
    vectors: List[List[float]] = []

    for t in texts:
        t = (t or "").strip()
        if not t:
            vectors.append([])
            continue

        body = {"inputText": t}

        resp = br.invoke_model(
            modelId=BEDROCK_EMBED_MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )

        payload = json.loads(resp["body"].read())
        emb = payload.get("embedding") or payload.get("vector") or payload.get("embeddings")

        if not emb:
            raise RuntimeError(f"Bedrock embedding failed: {payload}")

        vectors.append(emb)

    return vectors


def _make_unique_id(filename: str, raw_id: Optional[str], i: int) -> str:
    file_key = os.path.basename(filename).replace(" ", "_")
    base = (raw_id or "").strip() or "row"
    return f"{file_key}::{base}::{i}"


def build_index_from_json_folder(json_folder: str = "data") -> None:
    collection = _get_collection()

    try:
        existing = collection.get(include=[])
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception:
        logger.exception("failed clearing existing collection before re-index")

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
                    content = item.get("content") or item.get("answer") or json.dumps(item, ensure_ascii=False)
                    raw_id = item.get("id")
                    title = (item.get("title") or "").strip()
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
            title = (data.get("title") or "").strip()
            doc_id = _make_unique_id(filename, raw_id, 0)

            bump = 1
            while doc_id in used_ids:
                doc_id = _make_unique_id(filename, raw_id, bump)
                bump += 1

            used_ids.add(doc_id)
            meta = {"source": filename, "raw_id": raw_id or "", "title": title}
            docs.append((doc_id, content, meta))

    if not docs:
        logger.warning("No JSON docs found to ingest from %s", json_folder)
        return

    ids = [d[0] for d in docs]
    documents = [d[1] for d in docs]
    metadatas = [d[2] for d in docs]

    embeddings = _embed_texts(documents)

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    logger.info("Ingested %s docs into Chroma at %s", len(ids), CHROMA_PATH)


def retrieve_hits(
    question: str,
    k: int = 3,
    max_distance: float = RAG_MAX_DISTANCE,
) -> List[Dict]:
    question = (question or "").strip()
    if not question:
        return []

    collection = _get_collection()
    q_emb = _embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=max(1, k),
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    hits: List[Dict] = []

    for doc, meta, dist in zip(docs, metas, distances):
        if not doc:
            continue

        distance = float(dist) if dist is not None else 999.0
        if distance > max_distance:
            continue

        meta = meta or {}
        hits.append(
            {
                "document": doc.strip(),
                "source": meta.get("source", "doc"),
                "title": (meta.get("title") or "").strip(),
                "raw_id": meta.get("raw_id", ""),
                "distance": distance,
            }
        )

    return hits


