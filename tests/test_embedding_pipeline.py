"""The embedding pipeline: provider families, batching, the boot self-check,
and a collection that survives a model change.

Background: every knowledge-base answer failed in production while /health said
"ok". The crawl succeeded, the embedding step did not, and nothing surfaced the
Bedrock error outside a background log line. These pin that the error code is
now named on /health, that the request shape can never drift from the model id,
and that a model change recreates the Chroma collection instead of failing every
insert against a stale one.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402
import rag  # noqa: E402
from errors import ModelBackendError  # noqa: E402


def _client_error(code: str, message: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "InvokeModel")


class _Bedrock:
    """Answers every invoke_model with vectors of `dim`, recording each request."""

    def __init__(self, dim: int = 4, exc: Exception | None = None):
        self.dim = dim
        self.exc = exc
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def invoke_model(self, **kwargs):
        body = json.loads(kwargs["body"])
        with self._lock:
            self.calls.append(body)
        if self.exc is not None:
            raise self.exc

        if "texts" in body:                                    # Cohere
            vecs = [[float(hash(t) % 97) / 97] * self.dim for t in body["texts"]]
            payload = {"embeddings": {"float": vecs}}
        else:                                                  # Titan
            payload = {"embedding": [float(hash(body["inputText"]) % 97) / 97] * self.dim}

        raw = json.dumps(payload).encode()

        class _Body:
            @staticmethod
            def read():
                return raw

        return {"body": _Body()}


@pytest.fixture(autouse=True)
def _fresh_embed_status():
    before = dict(rag._embed_status)
    yield
    rag._embed_status.clear()
    rag._embed_status.update(before)


# --------------------------------------------------------------------------- #
# Provider family is decided in one place, and an unknown id is refused.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model_id, family",
    [
        ("amazon.titan-embed-text-v2:0", "titan"),
        ("amazon.titan-embed-text-v1", "titan"),
        ("cohere.embed-multilingual-v3", "cohere"),
        ("cohere.embed-english-v3", "cohere"),
        ("cohere.embed-v4:0", "cohere"),
        ("arn:aws:bedrock:ap-south-1::foundation-model/amazon.titan-embed-text-v2:0", "titan"),
        ("  Cohere.Embed-Multilingual-V3 ", "cohere"),
    ],
)
def test_known_embedding_families(model_id, family):
    assert rag.embedding_family(model_id) == family


@pytest.mark.parametrize(
    "model_id",
    ["", "amazon.nova-micro-v1:0", "anthropic.claude-3-haiku", "titan", "cohere.command-r"],
)
def test_unknown_embedding_model_is_refused_by_name(model_id):
    """Silently sending Titan's body to a non-Titan model is how the outage hid."""
    with pytest.raises(ModelBackendError) as excinfo:
        rag.embedding_family(model_id)

    assert "BEDROCK_EMBED_MODEL_ID" in str(excinfo.value)


def test_unknown_model_fails_at_embed_time_before_any_call(monkeypatch):
    br = _Bedrock()
    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "anthropic.claude-3-haiku")
    monkeypatch.setattr(rag, "_get_bedrock", lambda: br)

    with pytest.raises(ModelBackendError):
        rag._embed_texts(["hello"])

    assert br.calls == [], "no request should be sent for an unsupported model"


# --------------------------------------------------------------------------- #
# Batching.
# --------------------------------------------------------------------------- #
def test_cohere_is_batched(monkeypatch):
    br = _Bedrock(dim=3)
    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "cohere.embed-multilingual-v3")
    monkeypatch.setattr(rag, "EMBED_BATCH_SIZE", 96)
    monkeypatch.setattr(rag, "_get_bedrock", lambda: br)

    texts = [f"passage {i}" for i in range(100)]
    vectors = rag._embed_texts(texts, input_type="search_document")

    assert len(br.calls) == 2, "100 texts should be 96 + 4, not 100 calls"
    assert [len(c["texts"]) for c in br.calls] == [96, 4]
    assert all(c["input_type"] == "search_document" for c in br.calls)
    assert len(vectors) == 100 and all(len(v) == 3 for v in vectors)


def test_titan_runs_concurrently_and_keeps_order(monkeypatch):
    br = _Bedrock(dim=2)
    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setattr(rag, "EMBED_CONCURRENCY", 4)
    monkeypatch.setattr(rag, "_get_bedrock", lambda: br)

    texts = [f"passage {i}" for i in range(20)]
    vectors = rag._embed_texts(texts)

    assert len(br.calls) == 20, "Titan takes one text per request"
    # Order must survive the thread pool: vector i is the embedding of text i.
    expected = [[float(hash(t) % 97) / 97] * 2 for t in texts]
    assert vectors == expected


def test_blank_texts_are_skipped_but_keep_their_slot(monkeypatch):
    br = _Bedrock(dim=2)
    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "cohere.embed-english-v3")
    monkeypatch.setattr(rag, "_get_bedrock", lambda: br)

    vectors = rag._embed_texts(["a", "", "   ", "b"])

    assert br.calls[0]["texts"] == ["a", "b"]
    assert vectors[1] == [] and vectors[2] == []
    assert len(vectors[0]) == 2 and len(vectors[3]) == 2


def test_short_response_is_an_error_not_a_misaligned_index(monkeypatch):
    """Fewer vectors than texts would silently pair passages with the wrong vector."""
    class _Short:
        def invoke_model(self, **kwargs):
            raw = json.dumps({"embeddings": {"float": [[0.1, 0.2]]}}).encode()

            class _Body:
                @staticmethod
                def read():
                    return raw

            return {"body": _Body()}

    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "cohere.embed-english-v3")
    monkeypatch.setattr(rag, "_get_bedrock", lambda: _Short())

    with pytest.raises(ModelBackendError):
        rag._embed_texts(["a", "b"])


# --------------------------------------------------------------------------- #
# The boot self-check names the real AWS error.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "exc, code, hint",
    [
        (
            _client_error(
                "AccessDeniedException",
                "User: arn:aws:sts::1:assumed-role/r/x is not authorized to perform: "
                "bedrock:InvokeModel on resource: arn:aws:bedrock:ap-south-1::foundation-model/m",
            ),
            "AccessDeniedException",
            "iam_policy_denies_bedrock_invokemodel_on_this_model",
        ),
        (
            _client_error(
                "AccessDeniedException",
                "You don't have access to the model with the specified model ID.",
            ),
            "AccessDeniedException",
            "bedrock_model_access_not_granted_in_region",
        ),
        (
            _client_error("ValidationException", "Malformed input request"),
            "ValidationException",
            "request_body_does_not_match_model_family",
        ),
        (
            _client_error("ResourceNotFoundException", "Could not resolve the foundation model"),
            "ResourceNotFoundException",
            "model_id_not_available_in_region",
        ),
        (
            _client_error("ThrottlingException", "Too many requests"),
            "ThrottlingException",
            "quota_throttled",
        ),
        (NoCredentialsError(), "NoCredentialsError", "no_aws_credentials_for_instance"),
        (
            # Verbatim message from the production Cohere failure. It contains
            # "not authorized to perform" - the same phrase the generic IAM
            # case matches on - which is exactly what made this misclassify as
            # an IAM problem in production and cost hours of debugging a policy
            # that was never wrong.
            _client_error(
                "AccessDeniedException",
                "Model access is denied due to IAM user or service role is not "
                "authorized to perform the required AWS Marketplace actions "
                "(aws-marketplace:ViewSubscriptions, aws-marketplace:Subscribe) "
                "to enable access to this model. ... Your AWS Marketplace "
                "subscription for this model cannot be completed at this time.",
            ),
            "AccessDeniedException",
            "marketplace_subscription_required_for_third_party_model",
        ),
    ],
)
def test_self_check_reports_the_aws_error_code(monkeypatch, exc, code, hint):
    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setattr(rag, "_get_bedrock", lambda: _Bedrock(exc=exc))

    status = rag.embedding_self_check()

    assert status["checked"] is True
    assert status["ok"] is False
    # "ClientError" would be useless here - the AWS code is the diagnosis.
    assert status["error_type"] == code
    assert status["error_hint"] == hint
    assert status["dimension"] is None


# --------------------------------------------------------------------------- #
# Regression: an AWS Marketplace subscription denial must never read as an IAM
# policy problem. This is the exact misdiagnosis that shipped to production -
# Cohere's Marketplace-denial message contains "not authorized to perform",
# which is also the generic IAM-denial phrase, and the IAM branch used to be
# checked first. The fix must be Marketplace-vocabulary-first, not phrase-first.
# --------------------------------------------------------------------------- #
_MARKETPLACE_DENIAL = (
    "Model access is denied due to IAM user or service role is not authorized "
    "to perform the required AWS Marketplace actions (aws-marketplace:"
    "ViewSubscriptions, aws-marketplace:Subscribe) to enable access to this "
    "model. ... Your AWS Marketplace subscription for this model cannot be "
    "completed at this time."
)


def test_marketplace_denial_is_never_misread_as_an_iam_policy_problem():
    """The literal defect this task exists to fix: same code, same substring
    ("not authorized to perform"), two completely different required fixes."""
    hint = rag._error_hint("AccessDeniedException", _MARKETPLACE_DENIAL)

    assert hint == "marketplace_subscription_required_for_third_party_model"
    assert hint != "iam_policy_denies_bedrock_invokemodel_on_this_model"


def test_plain_iam_denial_without_marketplace_wording_still_gets_the_iam_hint():
    """The fix for the regression above must not swallow the real IAM case."""
    hint = rag._error_hint(
        "AccessDeniedException",
        "User: arn:aws:sts::1:assumed-role/r/x is not authorized to perform: "
        "bedrock:InvokeModel on resource: arn:aws:bedrock:ap-south-1::foundation-model/m",
    )

    assert hint == "iam_policy_denies_bedrock_invokemodel_on_this_model"


def test_self_check_reports_unknown_model_family(monkeypatch):
    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "amazon.nova-micro-v1:0")
    monkeypatch.setattr(rag, "_get_bedrock", lambda: _Bedrock())

    status = rag.embedding_self_check()

    assert status["ok"] is False
    assert status["error_hint"] == "unknown_model_family"


def test_self_check_success_records_family_and_dimension(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "cohere.embed-multilingual-v3")
    monkeypatch.setattr(rag, "CHROMA_PATH", str(tmp_path))
    monkeypatch.setattr(rag, "_client", None)
    monkeypatch.setattr(rag, "_collection", None)
    monkeypatch.setattr(rag, "_get_bedrock", lambda: _Bedrock(dim=1024))

    status = rag.embedding_self_check()

    assert status["ok"] is True
    assert status["family"] == "cohere"
    assert status["dimension"] == 1024
    assert status["error_type"] is None


def test_self_check_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("client construction exploded")

    monkeypatch.setattr(rag, "_get_bedrock", boom)

    status = rag.embedding_self_check()      # must not propagate

    assert status["ok"] is False
    assert status["error_type"] == "RuntimeError"


# --------------------------------------------------------------------------- #
# /health carries the index count and the self-check, not just crawl counts.
# --------------------------------------------------------------------------- #
def test_health_exposes_index_count_and_embedding_status(monkeypatch):
    monkeypatch.setattr(main, "ENFORCE_API_KEY", False)
    monkeypatch.setattr(main, "AUTO_CRAWL", False)
    monkeypatch.setattr(main, "init_rag", lambda: None)
    monkeypatch.setattr(main, "_reload_kb_from_disk", lambda: None)
    monkeypatch.setattr(main, "index_count", lambda: 0)
    monkeypatch.setattr(
        main,
        "embedding_status",
        lambda: {
            "checked": True, "ok": False, "model": "cohere.embed-multilingual-v3",
            "family": "cohere", "dimension": None,
            "error_type": "AccessDeniedException",
            "error_hint": "iam_policy_denies_bedrock_invokemodel_on_this_model",
        },
    )

    with TestClient(main.app) as client:
        body = client.get("/health").json()

    # The crawl counts still say "healthy"; these two say retrieval cannot work.
    assert body["index_documents"] == 0
    assert body["retrieval_ready"] is False
    assert body["embedding"]["error_type"] == "AccessDeniedException"
    assert body["embedding"]["error_hint"] == "iam_policy_denies_bedrock_invokemodel_on_this_model"
    assert body["embedding"]["ok"] is False


def test_health_retrieval_ready_when_index_populated_and_probe_passed(monkeypatch):
    monkeypatch.setattr(main, "ENFORCE_API_KEY", False)
    monkeypatch.setattr(main, "AUTO_CRAWL", False)
    monkeypatch.setattr(main, "init_rag", lambda: None)
    monkeypatch.setattr(main, "_reload_kb_from_disk", lambda: None)
    monkeypatch.setattr(main, "index_count", lambda: 371)
    monkeypatch.setattr(
        main,
        "embedding_status",
        lambda: {"checked": True, "ok": True, "model": "amazon.titan-embed-text-v2:0",
                 "family": "titan", "dimension": 1024, "error_type": None, "error_hint": None},
    )

    with TestClient(main.app) as client:
        body = client.get("/health").json()

    assert body["index_documents"] == 371
    assert body["retrieval_ready"] is True
    assert body["embedding"]["dimension"] == 1024


# --------------------------------------------------------------------------- #
# A model change recreates the collection instead of failing every insert.
# --------------------------------------------------------------------------- #
def _reset_chroma(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "CHROMA_PATH", str(tmp_path))
    monkeypatch.setattr(rag, "CHROMA_COLLECTION", "sidh_guide")
    monkeypatch.setattr(rag, "_client", None)
    monkeypatch.setattr(rag, "_collection", None)


def _write_kb(tmp_path, n=3):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "site.json").write_text(
        json.dumps([{"id": f"p{i}", "title": f"Page {i}", "content": f"About page {i}."} for i in range(n)]),
        encoding="utf-8",
    )
    return str(data_dir)


def test_reindex_after_model_change_recreates_collection(monkeypatch, tmp_path):
    """Titan v1 wrote 1536-wide vectors; switching to a 1024-wide model must not
    leave the collection empty with every insert rejected."""
    _reset_chroma(monkeypatch, tmp_path)
    kb = _write_kb(tmp_path)

    # First model: 1536-dim.
    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v1")
    monkeypatch.setattr(rag, "_get_bedrock", lambda: _Bedrock(dim=1536))
    rag.build_index_from_json_folder(kb)
    assert rag.index_count() == 3

    # Model changes to a 1024-dim one - same process, same on-disk store.
    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "cohere.embed-multilingual-v3")
    monkeypatch.setattr(rag, "_get_bedrock", lambda: _Bedrock(dim=1024))
    rag.build_index_from_json_folder(kb)

    assert rag.index_count() == 3, "the re-index must land, not fail on a stale dimension"
    assert rag._get_collection().metadata.get("embed_model") == "cohere.embed-multilingual-v3"
    assert rag._get_collection().metadata.get("embed_dim") == 1024


def test_successful_reindex_clears_a_failed_boot_probe(monkeypatch, tmp_path):
    """IAM gets fixed, /admin/refresh runs: /health must go green without a restart."""
    _reset_chroma(monkeypatch, tmp_path)
    kb = _write_kb(tmp_path)
    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "cohere.embed-multilingual-v3")

    monkeypatch.setattr(rag, "_get_bedrock", lambda: _Bedrock(exc=_client_error(
        "AccessDeniedException", "is not authorized to perform: bedrock:InvokeModel")))
    assert rag.embedding_self_check()["ok"] is False

    monkeypatch.setattr(rag, "_get_bedrock", lambda: _Bedrock(dim=1024))
    rag.build_index_from_json_folder(kb)

    status = rag.embedding_status()
    assert status["ok"] is True
    assert status["dimension"] == 1024
    assert status["error_type"] is None


def test_collection_survives_a_restart_on_the_same_model(monkeypatch, tmp_path):
    """Recreation is for model changes only. A plain restart must keep serving
    the existing vectors."""
    _reset_chroma(monkeypatch, tmp_path)
    kb = _write_kb(tmp_path, n=5)
    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setattr(rag, "_get_bedrock", lambda: _Bedrock(dim=1024))
    rag.build_index_from_json_folder(kb)

    # "Restart": drop the in-process handles, reopen from disk with the same model.
    monkeypatch.setattr(rag, "_client", None)
    monkeypatch.setattr(rag, "_collection", None)
    rag.embedding_self_check()

    assert rag.index_count() == 5


def test_stale_collection_without_metadata_is_detected_by_dimension(monkeypatch, tmp_path):
    """Collections written before embed_model metadata existed carry only vectors."""
    _reset_chroma(monkeypatch, tmp_path)
    client = rag._get_client()
    legacy = client.create_collection("sidh_guide", metadata={"hnsw:space": "cosine"})
    legacy.add(ids=["old"], documents=["old doc"], embeddings=[[0.1] * 1536])

    monkeypatch.setattr(rag, "BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
    collection = rag._ensure_collection_for(1024)

    assert collection.count() == 0, "the 1536-dim legacy collection should have been replaced"
    assert collection.metadata.get("embed_dim") == 1024
