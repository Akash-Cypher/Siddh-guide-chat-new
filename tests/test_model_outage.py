"""The model backend is down. The visitor must never see a 500.

Before this, `retrieve_hits()` and `generate_answer()` were the only Bedrock
calls in the request path with no error handling, so any Bedrock failure -- an
unset NOVA_MODEL_ID, denied model access, a throttle, a socket timeout -- left
`/chat` raising, FastAPI answering 500, and the widget printing "Sorry,
something went wrong. Please try again."

The routes that DID handle it were not much better: they returned the standard
refusal, which claims the assistant can help with anything on the site. During a
full outage that is untrue, and it made the outage look like ordinary traffic in
metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402
from errors import ModelBackendError  # noqa: E402


OUTAGE = main.SERVICE_UNAVAILABLE_MESSAGE


@pytest.fixture(autouse=True)
def isolated_backend(monkeypatch):
    monkeypatch.setattr(main, "ENFORCE_API_KEY", False)
    monkeypatch.setattr(main, "ALLOW_DEFAULT_SESSION", True)
    monkeypatch.setattr(main, "faq_data", [])
    monkeypatch.setattr(main, "COURSE_DATA", [])
    monkeypatch.setattr(main, "WEBSITE_DATA", [])
    monkeypatch.setattr(main, "COURSE_CATALOG", [])
    monkeypatch.setattr(main, "get_recent_messages", lambda *a, **k: [])
    monkeypatch.setattr(main, "history_to_model_messages", lambda *a, **k: [])
    monkeypatch.setattr(main, "put_message", lambda *a, **k: None)


@pytest.fixture()
def client():
    # raise_server_exceptions=False so an unhandled error surfaces as the 500 a
    # real visitor would get, instead of exploding inside the test.
    return TestClient(main.app, raise_server_exceptions=False)


def post_chat(client: TestClient, message: str):
    return client.post("/chat", json={"message": message, "session_id": "outage-test"})


def dead_model(*args, **kwargs):
    raise ModelBackendError("NOVA_MODEL_ID is not set")


def one_hit(*args, **kwargs):
    return [
        {
            "document": (
                "Siksha is the Siddhanta online learning platform offering "
                "IKS certified courses."
            ),
            "source": "website.json",
            "title": "Siksha",
            "raw_id": "siksha",
            "url": "https://siddhantaknowledge.org/siksha",
            "distance": 0.2,
        }
    ]


# --------------------------------------------------------------------------- #
# The exact failure from the bug report.
# --------------------------------------------------------------------------- #
def test_generation_outage_on_a_real_question_is_not_a_500(client, monkeypatch):
    """"about siksha?" -- retrieval succeeds, generation is down."""
    monkeypatch.setattr(main, "retrieve_hits", one_hit)
    monkeypatch.setattr(main, "generate_answer", dead_model)

    response = post_chat(client, "about siksha?")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unavailable"
    assert data["answer"] == OUTAGE
    assert data["model"] == "degraded"
    assert data["citations"] == []


def test_retrieval_outage_is_not_a_500(client, monkeypatch):
    """The embedding call or the vector store is down."""
    monkeypatch.setattr(main, "retrieve_hits", dead_model)
    monkeypatch.setattr(main, "generate_answer", dead_model)

    response = post_chat(client, "what does the research programme cover?")

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert response.json()["answer"] == OUTAGE


def test_outage_answer_is_not_the_generic_refusal(client, monkeypatch):
    """An outage must not claim the assistant can help with anything on the site."""
    monkeypatch.setattr(main, "retrieve_hits", one_hit)
    monkeypatch.setattr(main, "generate_answer", dead_model)

    data = post_chat(client, "about siksha?").json()

    assert data["answer"] != main.REFUSAL_MESSAGE
    assert data["status"] != "refused"


def test_recommendation_outage_reports_unavailable(client, monkeypatch):
    """The recommendation route used to dress an outage up as a refusal."""
    monkeypatch.setattr(
        main,
        "COURSE_CATALOG",
        [{"title": "Introduction to Indian Knowledge Systems",
          "categories": ["IKS"], "price": "Free", "published": "2026-01-01"}],
    )
    monkeypatch.setattr(main, "retrieve_hits", one_hit)
    monkeypatch.setattr(main, "generate_answer", dead_model)

    data = post_chat(client, "recommend a course for engineering").json()

    assert data["status"] == "unavailable"
    assert data["answer"] == OUTAGE


# --------------------------------------------------------------------------- #
# Routes with truthful canned text keep answering -- but say they are degraded.
# --------------------------------------------------------------------------- #
def test_greeting_still_answers_during_an_outage_but_flags_it(client, monkeypatch):
    monkeypatch.setattr(main, "generate_social_reply", dead_model)
    monkeypatch.setattr(main, "retrieve_hits", one_hit)
    monkeypatch.setattr(main, "generate_answer", dead_model)

    response = post_chat(client, "hey")

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Hello! How can I assist you today?"
    # The reply looked perfectly healthy before; only this field distinguishes a
    # model-written greeting from the fixed one.
    assert data["model"] == "degraded"


def test_healthy_request_is_not_flagged_as_degraded(client, monkeypatch):
    monkeypatch.setattr(main, "generate_social_reply", lambda *a, **k: "Hi there!")

    data = post_chat(client, "hey").json()

    assert data["answer"] == "Hi there!"
    assert data["model"] == "ok"


# --------------------------------------------------------------------------- #
# Genuine bugs must stay loud.
# --------------------------------------------------------------------------- #
def test_a_real_bug_still_returns_500(client, monkeypatch):
    """Only dependency failures degrade. A TypeError is a defect, not an outage."""
    def broken(*args, **kwargs):
        raise TypeError("programming error")

    monkeypatch.setattr(main, "retrieve_hits", one_hit)
    monkeypatch.setattr(main, "generate_answer", broken)

    assert post_chat(client, "about siksha?").status_code == 500


# --------------------------------------------------------------------------- #
# The misconfiguration is visible without reading a log.
# --------------------------------------------------------------------------- #
def test_health_reports_whether_the_model_is_configured(client):
    body = client.get("/health").json()

    assert "nova_model_configured" in body
    assert isinstance(body["nova_model_configured"], bool)
    # Booleans only -- /health is unauthenticated, so no values are exposed.
    assert main.NOVA_MODEL_ID not in str(body) or not main.NOVA_MODEL_ID


# --------------------------------------------------------------------------- #
# Startup must not claim a component came up when it did not.
# --------------------------------------------------------------------------- #
def test_startup_does_not_claim_rag_initialized_when_it_failed(monkeypatch, caplog):
    """init_rag() failure no longer aborts boot, so the success line must adapt.

    Before the outage fix a broken vector store propagated out of lifespan and
    startup died, so this log line was unreachable after a failure. Now that the
    failure is swallowed to avoid a crash-loop, an unconditional "RAG
    initialized" would tell a deploy check the service came up clean while every
    knowledge-base answer is the outage notice.
    """
    def dead_init():
        raise ModelBackendError("vector store unavailable")

    monkeypatch.setattr(main, "init_rag", dead_init)
    monkeypatch.setattr(main, "AUTO_CRAWL", False)
    monkeypatch.setattr(main, "_reload_kb_from_disk", lambda: None)

    with caplog.at_level("INFO", logger="siddh_guide"):
        with TestClient(main.app):
            pass

    startup_lines = [r.getMessage() for r in caplog.records if "Startup complete" in r.getMessage()]
    assert startup_lines, "the startup line should still be emitted"
    assert "RAG initialized" not in startup_lines[-1]
    assert "UNAVAILABLE" in startup_lines[-1]


def test_startup_line_is_unchanged_on_a_healthy_boot(monkeypatch, caplog):
    """Existing log scans and deploy checks must keep matching."""
    monkeypatch.setattr(main, "init_rag", lambda: None)
    monkeypatch.setattr(main, "AUTO_CRAWL", False)
    monkeypatch.setattr(main, "_reload_kb_from_disk", lambda: None)

    with caplog.at_level("INFO", logger="siddh_guide"):
        with TestClient(main.app):
            pass

    startup_lines = [r.getMessage() for r in caplog.records if "Startup complete" in r.getMessage()]
    assert startup_lines[-1] == "Startup complete: KB loaded + RAG initialized"
