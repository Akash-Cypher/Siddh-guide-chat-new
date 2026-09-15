import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
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
    EMBED_BATCH_SIZE,
    EMBED_CONCURRENCY,
    RAG_MAX_DISTANCE,
)
from errors import ModelBackendError

logger = logging.getLogger("siddh_guide.rag")

_client = None
_collection = None
_bedrock = None
_collection_lock = threading.Lock()

# Result of the last embedding self-check, surfaced on /health. Starts as
# "not checked" so a deploy that never ran the check is distinguishable from
# one where the check passed.
_embed_status: Dict = {
    "checked": False,
    "ok": False,
    "model": BEDROCK_EMBED_MODEL_ID,
    "family": None,
    "dimension": None,
    "error_type": None,
    "error_hint": None,
}
_SELF_CHECK_TEXT = "Ask Sid embedding self-check"


def _get_bedrock():
    global _bedrock
    if _bedrock is None:
        try:
            _bedrock = boto3.client(
                "bedrock-runtime", region_name=AWS_REGION, config=AWS_BOTO_CONFIG
            )
        except Exception as exc:
            logger.exception("could not construct the Bedrock client for embeddings")
            raise ModelBackendError(
                f"Bedrock client unavailable: {type(exc).__name__}: {exc}"
            ) from exc
    return _bedrock


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _collection_metadata(model_id: str, dimension: Optional[int]) -> dict:
    meta = {"hnsw:space": "cosine", "embed_model": model_id}
    if dimension:
        meta["embed_dim"] = int(dimension)
    return meta


def _stored_dimension(collection) -> Optional[int]:
    """The vector width the collection actually holds, or None if empty/unknown."""
    meta = collection.metadata or {}
    if meta.get("embed_dim"):
        return int(meta["embed_dim"])
    try:
        if collection.count() == 0:
            return None
        sample = collection.peek(limit=1)
        embeddings = sample.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            return len(embeddings[0])
    except Exception:
        logger.exception("could not inspect the stored embedding dimension")
    return None


def _collection_is_stale(collection, model_id: str, dimension: Optional[int]) -> Optional[str]:
    """Why this collection cannot take vectors from `model_id`, or None if it can.

    Chroma fixes a collection's vector width on the first insert and rejects
    every later insert of a different width. A model change (Titan v1 -> v2, or
    Titan -> Cohere) therefore has to recreate the collection rather than fail
    inserts against it. The model id is recorded in metadata so the common case
    is a cheap string compare; the dimension check catches collections written
    before the metadata existed.
    """
    meta = collection.metadata or {}
    stored_model = meta.get("embed_model")
    if stored_model and stored_model != model_id:
        return f"embedding model changed ({stored_model} -> {model_id})"

    if dimension:
        stored_dim = _stored_dimension(collection)
        if stored_dim and stored_dim != dimension:
            return f"embedding dimension changed ({stored_dim} -> {dimension})"

    return None


def _open_collection(dimension: Optional[int] = None):
    """Open the collection, recreating it if it cannot hold the current model's vectors."""
    global _collection
    client = _get_client()
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata=_collection_metadata(BEDROCK_EMBED_MODEL_ID, dimension),
    )

    reason = _collection_is_stale(collection, BEDROCK_EMBED_MODEL_ID, dimension)
    if reason:
        logger.warning(
            "recreating Chroma collection %r: %s (dropping %s stored vectors)",
            CHROMA_COLLECTION, reason, collection.count(),
        )
        client.delete_collection(CHROMA_COLLECTION)
        collection = client.create_collection(
            name=CHROMA_COLLECTION,
            metadata=_collection_metadata(BEDROCK_EMBED_MODEL_ID, dimension),
        )

    _collection = collection
    return collection


def _get_collection():
    if _collection is None:
        with _collection_lock:
            if _collection is None:
                try:
                    _open_collection()
                except Exception as exc:
                    logger.exception("could not open the Chroma collection at %s", CHROMA_PATH)
                    raise ModelBackendError(
                        f"vector store unavailable: {type(exc).__name__}: {exc}"
                    ) from exc
    return _collection


def _ensure_collection_for(dimension: int):
    """The collection that will accept `dimension`-wide vectors, recreated if needed."""
    with _collection_lock:
        try:
            return _open_collection(dimension)
        except Exception as exc:
            logger.exception("could not prepare the Chroma collection for %s-dim vectors", dimension)
            raise ModelBackendError(
                f"vector store unavailable: {type(exc).__name__}: {exc}"
            ) from exc


def init_rag() -> None:
    _get_collection()


def index_count() -> Optional[int]:
    """How many vectors the index holds, or None if it cannot be read.

    This — not the crawl counts — is what says whether retrieval can work.
    """
    try:
        return int(_get_collection().count())
    except Exception:
        logger.exception("could not count the Chroma collection")
        return None


# --------------------------------------------------------------------------- #
# Embedding providers.
#
# Amazon and Cohere disagree on every field name in both the request and the
# response, and they differ on batching too: Cohere embeds up to 96 texts per
# call, Titan exactly one. The family is decided from the model id in one place
# so the id and the payload can never drift apart again, and an id that matches
# no family is refused loudly rather than sent Titan's body on the off-chance.
# --------------------------------------------------------------------------- #
_FAMILY_PREFIXES = (
    ("titan", "amazon.titan-embed"),
    ("cohere", "cohere.embed"),
)


def embedding_family(model_id: str) -> str:
    """'titan' or 'cohere', or ModelBackendError naming the offending setting."""
    bare = (model_id or "").strip()
    # Accept a full model ARN too: the family is in the last path segment.
    if "/" in bare:
        bare = bare.rsplit("/", 1)[1]
    lowered = bare.lower()
    for family, prefix in _FAMILY_PREFIXES:
        if lowered.startswith(prefix):
            return family
    raise ModelBackendError(
        f"BEDROCK_EMBED_MODEL_ID={model_id!r} is not a supported embedding model: "
        "expected an amazon.titan-embed-* or cohere.embed-* model id"
    )


def _embed_request_body(model_id: str, texts: List[str], input_type: str) -> dict:
    """The request shape this embedding model expects for `texts`."""
    family = embedding_family(model_id)
    if family == "cohere":
        # input_type is not optional for Cohere v3+: a passage and the question
        # that should retrieve it are embedded differently, and using one type
        # for both measurably degrades retrieval.
        return {
            "texts": list(texts),
            "input_type": input_type,
            "embedding_types": ["float"],
        }
    if len(texts) != 1:
        raise ModelBackendError("Titan embeds exactly one text per request")
    return {"inputText": texts[0]}


def _embeddings_from_payload(payload: dict) -> List[List[float]]:
    """Every vector out of whichever response shape the model returned, in order."""
    emb = payload.get("embedding")                    # Titan: one vector
    if emb:
        return [emb]

    embeddings = payload.get("embeddings")
    if isinstance(embeddings, dict):                  # Cohere with embedding_types
        floats = embeddings.get("float") or embeddings.get("float_")
        if floats:
            return list(floats)
    elif isinstance(embeddings, list) and embeddings:  # Cohere legacy
        first = embeddings[0]
        return list(embeddings) if isinstance(first, list) else [embeddings]

    vector = payload.get("vector")
    return [vector] if vector else []


def _invoke_embedding(br, texts: List[str], input_type: str) -> List[List[float]]:
    """One Bedrock call for `texts`; returns exactly len(texts) vectors."""
    body = _embed_request_body(BEDROCK_EMBED_MODEL_ID, texts, input_type)
    try:
        resp = br.invoke_model(
            modelId=BEDROCK_EMBED_MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read())
    except Exception as exc:
        logger.exception(
            "Bedrock embedding call failed (model=%s, texts=%s)",
            BEDROCK_EMBED_MODEL_ID, len(texts),
        )
        raise ModelBackendError(
            f"Bedrock embedding call failed: {_error_code(exc)}: {exc}"
        ) from exc

    vectors = _embeddings_from_payload(payload)
    if len(vectors) != len(texts) or any(not v for v in vectors):
        raise ModelBackendError(
            f"Bedrock returned {len(vectors)} embeddings for {len(texts)} texts: "
            f"{str(payload)[:200]}"
        )
    return vectors


def _embed_texts(
    texts: List[str], input_type: str = "search_document"
) -> List[List[float]]:
    """Embed each text, preserving order; blank texts come back as [].

    `input_type` is honoured by Cohere and ignored by Titan. Cohere is called
    in batches of EMBED_BATCH_SIZE; Titan is one text per call, so those calls
    run EMBED_CONCURRENCY at a time. Any failure raises ModelBackendError.
    """
    br = _get_bedrock()
    family = embedding_family(BEDROCK_EMBED_MODEL_ID)

    cleaned = [(t or "").strip() for t in texts]
    todo = [(i, t) for i, t in enumerate(cleaned) if t]
    vectors: List[List[float]] = [[] for _ in cleaned]
    if not todo:
        return vectors

    if family == "cohere":
        size = max(1, EMBED_BATCH_SIZE)
        for start in range(0, len(todo), size):
            chunk = todo[start:start + size]
            for (i, _), vec in zip(chunk, _invoke_embedding(br, [t for _, t in chunk], input_type)):
                vectors[i] = vec
        return vectors

    workers = max(1, min(EMBED_CONCURRENCY, len(todo)))
    if workers == 1:
        for i, t in todo:
            vectors[i] = _invoke_embedding(br, [t], input_type)[0]
        return vectors

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda item: _invoke_embedding(br, [item[1]], input_type)[0], todo)
        for (i, _), vec in zip(todo, results):
            vectors[i] = vec
    return vectors


# --------------------------------------------------------------------------- #
# Startup self-check.
#
# A denied, misnamed or unavailable embedding model used to fail only inside the
# background re-index, where refresh.py logged it and moved on. /health kept
# saying "ok" because it reported crawl counts, not the index. Embedding one
# probe string at boot makes the failure — and its exact AWS error code — the
# first thing in the logs and a field on /health.
# --------------------------------------------------------------------------- #
def _error_code(exc: BaseException) -> str:
    """The AWS error code behind an exception chain, else the class name.

    botocore raises ClientError for every service error; the code that tells an
    IAM denial from a bad request body lives in exc.response, and the app wraps
    it in ModelBackendError, so walk the chain.
    """
    seen = 0
    cur: Optional[BaseException] = exc
    root: BaseException = exc
    while cur is not None and seen < 6:
        response = getattr(cur, "response", None)
        if isinstance(response, dict):
            code = (response.get("Error") or {}).get("Code")
            if code:
                return str(code)
        root = cur
        cur = cur.__cause__ or cur.__context__
        seen += 1
    # No AWS code anywhere in the chain: name the innermost exception, which is
    # the real failure (NoCredentialsError, ReadTimeoutError...), not the
    # ModelBackendError the app wrapped it in.
    return type(root).__name__


def _error_hint(code: str, message: str) -> str:
    """Which fix the error points at. Stable tokens, safe to expose on /health."""
    msg = (message or "").lower()
    if code == "AccessDeniedException":
        if "not authorized to perform" in msg:
            return "iam_policy_denies_bedrock_invokemodel_on_this_model"
        if "access to the model" in msg or "model access" in msg:
            return "bedrock_model_access_not_granted_in_region"
        return "access_denied"
    if code == "ValidationException":
        return "request_body_does_not_match_model_family"
    if code == "ResourceNotFoundException":
        return "model_id_not_available_in_region"
    if code in ("ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"):
        return "quota_throttled"
    if code in ("NoCredentialsError", "CredentialRetrievalError", "PartialCredentialsError"):
        return "no_aws_credentials_for_instance"
    if "not a supported embedding model" in msg:
        return "unknown_model_family"
    return "unexpected"


def embedding_self_check() -> Dict:
    """Embed one probe string; record and log the outcome. Never raises."""
    status: Dict = {
        "checked": True,
        "ok": False,
        "model": BEDROCK_EMBED_MODEL_ID,
        "family": None,
        "dimension": None,
        "error_type": None,
        "error_hint": None,
    }
    try:
        status["family"] = embedding_family(BEDROCK_EMBED_MODEL_ID)
        vector = _embed_texts([_SELF_CHECK_TEXT], input_type="search_query")[0]
        status["ok"] = True
        status["dimension"] = len(vector)
        logger.info(
            "embedding self-check OK: model=%s family=%s dimension=%s",
            BEDROCK_EMBED_MODEL_ID, status["family"], status["dimension"],
        )
        # The dimension is now known, so a collection left by a different model
        # can be replaced before the first re-index tries to insert into it.
        try:
            _ensure_collection_for(status["dimension"])
        except ModelBackendError:
            logger.exception("embedding self-check: collection could not be prepared")
    except Exception as exc:
        code = _error_code(exc)
        status["error_type"] = code
        status["error_hint"] = _error_hint(code, str(exc))
        logger.error(
            "embedding self-check FAILED: model=%s error=%s hint=%s detail=%s",
            BEDROCK_EMBED_MODEL_ID, code, status["error_hint"], exc,
        )

    _embed_status.update(status)
    return dict(_embed_status)


def embedding_status() -> Dict:
    return dict(_embed_status)


def _make_unique_id(filename: str, raw_id: Optional[str], i: int) -> str:
    file_key = os.path.basename(filename).replace(" ", "_")
    base = (raw_id or "").strip() or "row"
    return f"{file_key}::{base}::{i}"


def _enrich_for_embedding(title: str, keywords, content: str) -> str:
    """Prepend the curated title + keywords to the content before embedding.

    Retrieval matches the *embedding* of this text. Entries carry hand-written
    keywords (e.g. enrollment-overview -> "how to enroll", "click to enroll")
    that describe the questions they answer, but embedding content alone misses
    them, so short intent questions land too far away. Including title + keywords
    pulls those questions close to the right entry.
    """
    parts = []
    if title:
        parts.append(title.strip())
    if isinstance(keywords, list):
        kw = " ".join(str(k).strip() for k in keywords if str(k).strip())
        if kw:
            parts.append(kw)
    elif isinstance(keywords, str) and keywords.strip():
        parts.append(keywords.strip())
    parts.append((content or "").strip())
    return "\n".join(p for p in parts if p).strip()


def build_index_from_json_folder(json_folder: str = "data") -> None:
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
                    base_content = item.get("content") or item.get("answer") or json.dumps(item, ensure_ascii=False)
                    raw_id = item.get("id")
                    title = (item.get("title") or "").strip()
                    # Embed the curated title + keywords alongside the content so
                    # short intent questions ("how do I enroll") match entries
                    # whose keywords already say "how to enroll" / "enrollment".
                    content = _enrich_for_embedding(title, item.get("keywords"), base_content)
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
                # The page URL travels with the chunk so the model can cite the real
                # link. Without it the context had no URL at all and the model
                # invented plausible-looking ones.
                url = ""
                if isinstance(item, dict):
                    url = (item.get("source_url") or item.get("url") or item.get("link") or "").strip()
                meta = {"source": filename, "raw_id": raw_id or "", "title": title, "url": url}
                docs.append((doc_id, content, meta))

        elif isinstance(data, dict):
            base_content = data.get("content") or data.get("answer") or json.dumps(data, ensure_ascii=False)
            raw_id = data.get("id") if isinstance(data.get("id"), str) else None
            title = (data.get("title") or "").strip()
            content = _enrich_for_embedding(title, data.get("keywords"), base_content)
            doc_id = _make_unique_id(filename, raw_id, 0)

            bump = 1
            while doc_id in used_ids:
                doc_id = _make_unique_id(filename, raw_id, bump)
                bump += 1

            used_ids.add(doc_id)
            url = (data.get("source_url") or data.get("url") or data.get("link") or "").strip()
            meta = {"source": filename, "raw_id": raw_id or "", "title": title, "url": url}
            docs.append((doc_id, content, meta))

    if not docs:
        logger.warning("No JSON docs found to ingest from %s", json_folder)
        return

    ids = [d[0] for d in docs]
    documents = [d[1] for d in docs]
    metadatas = [d[2] for d in docs]

    # Embed BEFORE clearing. The old order deleted every document first and only
    # then called Bedrock, so any embedding failure - a denied model, a throttle,
    # a timeout - left the collection permanently EMPTY, and every subsequent
    # startup crawl wiped it again. Retrieval then found nothing and the
    # assistant answered from no knowledge at all. Building the new vectors
    # first means a failed refresh leaves the previous index serving.
    embeddings = _embed_texts(documents, input_type="search_document")

    # The collection must be able to take vectors of THIS width. If the
    # embedding model changed since it was written, a plain add would raise on
    # the dimension mismatch - after the delete below had already emptied it.
    dimension = next((len(v) for v in embeddings if v), 0)
    if not dimension:
        raise ModelBackendError("re-index produced no embeddings")
    collection = _ensure_collection_for(dimension)

    try:
        existing = collection.get(include=[])
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception:
        logger.exception("failed clearing existing collection before re-index")

    try:
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    except Exception as exc:
        logger.exception("Chroma insert failed after re-embedding %s docs", len(ids))
        raise ModelBackendError(
            f"vector store insert failed: {type(exc).__name__}: {exc}"
        ) from exc
    logger.info(
        "Ingested %s docs into Chroma at %s (model=%s, dimension=%s)",
        len(ids), CHROMA_PATH, BEDROCK_EMBED_MODEL_ID, dimension,
    )
    # A full re-index just embedded every document, which is a stronger proof
    # than the boot probe. Record it, so an IAM fix followed by /admin/refresh
    # turns /health green without waiting for the next restart.
    _embed_status.update({
        "checked": True, "ok": True,
        "model": BEDROCK_EMBED_MODEL_ID,
        "family": embedding_family(BEDROCK_EMBED_MODEL_ID),
        "dimension": dimension,
        "error_type": None, "error_hint": None,
    })


def retrieve_hits(
    question: str,
    k: int = 3,
    max_distance: float = RAG_MAX_DISTANCE,
) -> List[Dict]:
    question = (question or "").strip()
    if not question:
        return []

    collection = _get_collection()
    q_emb = _embed_texts([question], input_type="search_query")[0]

    try:
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=max(1, k),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.exception("vector search failed")
        raise ModelBackendError(
            f"vector search failed: {type(exc).__name__}: {exc}"
        ) from exc

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
                "url": (meta.get("url") or "").strip(),
                "distance": distance,
            }
        )

    return hits


