"""Every external AI failure is typed at the boundary.

`main` can only degrade gracefully if it gets one exception type back. These
pin that `models` and `rag` translate the real failure modes -- a missing model
id, a botocore ClientError, a socket timeout, an unreadable payload -- into
ModelBackendError rather than letting the raw exception escape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import json

from botocore.exceptions import ClientError, ReadTimeoutError

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import models  # noqa: E402
import rag  # noqa: E402
from errors import ModelBackendError  # noqa: E402


ACCESS_DENIED = ClientError(
    {"Error": {"Code": "AccessDeniedException", "Message": "not authorized"}},
    "InvokeModel",
)
THROTTLED = ClientError(
    {"Error": {"Code": "ThrottlingException", "Message": "rate exceeded"}},
    "InvokeModel",
)


class _Client:
    def __init__(self, exc):
        self._exc = exc

    def invoke_model(self, **kwargs):
        raise self._exc


def test_missing_nova_model_id_is_a_backend_error(monkeypatch):
    monkeypatch.setattr(models, "NOVA_MODEL_ID", "")

    with pytest.raises(ModelBackendError) as excinfo:
        models.generate_answer(user_message="hi", context="some context")

    # The message has to name the variable: this is the whole diagnosis.
    assert "NOVA_MODEL_ID" in str(excinfo.value)


def test_missing_nova_model_id_is_a_backend_error_for_social(monkeypatch):
    monkeypatch.setattr(models, "NOVA_MODEL_ID", "")

    with pytest.raises(ModelBackendError):
        models.generate_social_reply("hey", "say hello")


@pytest.mark.parametrize("exc", [ACCESS_DENIED, THROTTLED, ReadTimeoutError(endpoint_url="x")])
def test_bedrock_invocation_failures_are_typed(monkeypatch, exc):
    monkeypatch.setattr(models, "NOVA_MODEL_ID", "amazon.nova-lite-v1:0")
    monkeypatch.setattr(models, "_get_bedrock_client", lambda: _Client(exc))

    with pytest.raises(ModelBackendError):
        models.generate_answer(user_message="hi", context="some context")


def test_missing_context_is_still_a_value_error(monkeypatch):
    """A caller bug must not be laundered into an outage."""
    monkeypatch.setattr(models, "NOVA_MODEL_ID", "amazon.nova-lite-v1:0")

    with pytest.raises(ValueError):
        models.generate_answer(user_message="hi", context="")


@pytest.mark.parametrize("exc", [ACCESS_DENIED, THROTTLED])
def test_embedding_failures_are_typed(monkeypatch, exc):
    monkeypatch.setattr(rag, "_get_bedrock", lambda: _Client(exc))

    with pytest.raises(ModelBackendError):
        rag._embed_texts(["what courses are available"])


def test_empty_embedding_payload_is_typed(monkeypatch):
    class _Empty:
        def invoke_model(self, **kwargs):
            class _Body:
                @staticmethod
                def read():
                    return b"{}"

            return {"body": _Body()}

    monkeypatch.setattr(rag, "_get_bedrock", lambda: _Empty())

    with pytest.raises(ModelBackendError):
        rag._embed_texts(["what courses are available"])


def test_vector_search_failure_is_typed(monkeypatch):
    class _Collection:
        def query(self, **kwargs):
            raise RuntimeError("chroma segment missing")

    monkeypatch.setattr(rag, "_embed_texts", lambda texts, **kw: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr(rag, "_get_collection", lambda: _Collection())

    with pytest.raises(ModelBackendError):
        rag.retrieve_hits("what courses are available")


def test_model_backend_error_is_a_runtime_error():
    """Pre-existing broad handlers keep catching it."""
    assert issubclass(ModelBackendError, RuntimeError)


# --------------------------------------------------------------------------- #
# Embedding provider portability.
#
# BEDROCK_EMBED_MODEL_ID was only nominally configurable: the request body was
# hardcoded to Amazon's field names, so if no Titan embedding model could be
# invoked in the region there was no working value to switch to.
# --------------------------------------------------------------------------- #
class _Recorder:
    def __init__(self, payload):
        self.payload = payload
        self.body = None

    def invoke_model(self, **kwargs):
        self.body = json.loads(kwargs["body"])

        class _Body:
            @staticmethod
            def read():
                return json.dumps(_Recorder_payload[0]).encode()

        _Recorder_payload[0] = self.payload
        return {"body": _Body()}


_Recorder_payload = [None]


def _embed_with(monkeypatch, model_id, payload, **kwargs):
    rec = _Recorder(payload)
    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", model_id)
    monkeypatch.setattr(rag, "_get_bedrock", lambda: rec)
    vectors = rag._embed_texts(["what courses are available"], **kwargs)
    return rec.body, vectors


def test_titan_request_and_response_shape(monkeypatch):
    body, vectors = _embed_with(
        monkeypatch, "amazon.titan-embed-text-v2:0", {"embedding": [0.1, 0.2]}
    )

    assert body == {"inputText": "what courses are available"}
    assert vectors == [[0.1, 0.2]]


def test_cohere_request_and_response_shape(monkeypatch):
    body, vectors = _embed_with(
        monkeypatch,
        "cohere.embed-v4:0",
        {"embeddings": {"float": [[0.3, 0.4]]}},
        input_type="search_query",
    )

    # Cohere names every field differently; sending Titan's body silently
    # produced a validation error rather than an embedding.
    assert body == {
        "texts": ["what courses are available"],
        "input_type": "search_query",
        "embedding_types": ["float"],
    }
    assert vectors == [[0.3, 0.4]]


def test_cohere_legacy_list_response_shape(monkeypatch):
    _, vectors = _embed_with(
        monkeypatch, "cohere.embed-english-v3", {"embeddings": [[0.5, 0.6]]}
    )

    assert vectors == [[0.5, 0.6]]


def test_a_question_is_embedded_as_a_query_not_a_passage(monkeypatch):
    """Cohere embeds a question and the passage answering it differently."""
    captured = {}

    def fake_embed(texts, input_type="search_document"):
        captured["input_type"] = input_type
        return [[0.1, 0.2, 0.3]]

    class _Collection:
        def query(self, **kwargs):
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    monkeypatch.setattr(rag, "_embed_texts", fake_embed)
    monkeypatch.setattr(rag, "_get_collection", lambda: _Collection())

    rag.retrieve_hits("what courses are available")

    assert captured["input_type"] == "search_query"


# --------------------------------------------------------------------------- #
# A failed re-index must not destroy the index that is serving.
# --------------------------------------------------------------------------- #
def test_failed_reindex_leaves_the_existing_index_intact(monkeypatch, tmp_path):
    """The old order deleted every document, THEN called Bedrock.

    So one denied or throttled embedding call emptied the collection, and every
    later startup crawl emptied it again - retrieval found nothing at all.
    """
    (tmp_path / "site.json").write_text(
        json.dumps([{"id": "a", "title": "Siksha", "content": "About Siksha."}]),
        encoding="utf-8",
    )

    class _Collection:
        def __init__(self):
            self.deleted = False

        def get(self, **kwargs):
            return {"ids": ["old-1", "old-2"]}

        def delete(self, **kwargs):
            self.deleted = True

        def add(self, **kwargs):
            raise AssertionError("add must not be reached when embedding fails")

    collection = _Collection()
    monkeypatch.setattr(rag, "_get_collection", lambda: collection)
    monkeypatch.setattr(
        rag, "_embed_texts", lambda *a, **k: (_ for _ in ()).throw(ModelBackendError("denied"))
    )

    with pytest.raises(ModelBackendError):
        rag.build_index_from_json_folder(str(tmp_path))

    assert not collection.deleted, "existing vectors were destroyed by a failed re-index"
