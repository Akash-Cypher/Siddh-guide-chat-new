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

    monkeypatch.setattr(rag, "_embed_texts", lambda texts: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr(rag, "_get_collection", lambda: _Collection())

    with pytest.raises(ModelBackendError):
        rag.retrieve_hits("what courses are available")


def test_model_backend_error_is_a_runtime_error():
    """Pre-existing broad handlers keep catching it."""
    assert issubclass(ModelBackendError, RuntimeError)
