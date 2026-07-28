"""Current-session conversation continuity.

The shipped suite (test_kb_only_chat.py) stubs get_recent_messages to [], so it
never exercises multi-turn behaviour. These tests run against a real in-memory
chat store so history is genuinely written, ordered, read back and used.
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

import chat_store  # noqa: E402
import main  # noqa: E402


CATALOG = [
    {"title": "Indian Knowledge Systems for Engineers",
     "categories": ["engineering", "technology"], "published": "2025-01-10"},
    {"title": "Vedic Mathematics Foundations",
     "categories": ["mathematics", "engineering"], "published": "2025-02-01"},
    {"title": "Samskrit 1: Thinking in Samskrit",
     "categories": ["language"], "published": "2025-03-05"},
    {"title": "Indic Business and Commerce Traditions",
     "categories": ["commerce", "management"], "published": "2025-04-02"},
]

KB_DOC = (
    "Indian Knowledge Systems for Engineers is a certified online course. "
    "The duration is 12 weeks. It covers Indic approaches to design and problem "
    "solving for technical students."
)


class Recorder:
    """Captures what the model layer was handed on each call."""

    def __init__(self):
        self.calls = []
        self.queries = []
        self.answer = "The duration is 12 weeks."

    def generate_answer(self, user_message, context="", history_messages=None,
                        session_context=""):
        self.calls.append({
            "user_message": user_message,
            "context": context,
            "history": history_messages or [],
            "session_context": session_context,
        })
        return self.answer

    def retrieve_hits(self, query, k=5, **kwargs):
        self.queries.append(query)
        if "engineer" in query.lower() or "duration" in query.lower():
            return [{"document": KB_DOC, "source": "website.json",
                     "title": "IKS for Engineers", "distance": 0.05}]
        return []


@pytest.fixture()
def rec():
    return Recorder()


@pytest.fixture()
def store(monkeypatch):
    """Real ordered in-memory store, standing in for DynamoDB."""
    data: dict[str, list[dict]] = {}
    clock = {"ms": 1_700_000_000_000}

    def put_message(session_id, role, text, request_id="", sources=None, context_used=0):
        # Same millisecond for several writes on purpose: ordering must come from
        # the sort key's sequence, not from luck.
        ts = clock["ms"]
        data.setdefault(session_id, []).append({
            "session_id": session_id,
            "sk": chat_store._make_sort_key(ts),
            "ts": ts,
            "role": role,
            "text": text,
        })

    def get_recent_messages(session_id, limit=8):
        items = list(data.get(session_id, []))
        items.sort(key=lambda x: (int(x["ts"]), str(x["sk"])))
        return items[-limit:]

    monkeypatch.setattr(main, "put_message", put_message)
    monkeypatch.setattr(main, "get_recent_messages", get_recent_messages)
    monkeypatch.setattr(main, "history_to_model_messages",
                        chat_store.history_to_model_messages)
    return data


@pytest.fixture()
def client(monkeypatch, rec, store):
    monkeypatch.setattr(main, "ENFORCE_API_KEY", False)
    monkeypatch.setattr(main, "ALLOW_DEFAULT_SESSION", True)
    monkeypatch.setattr(main, "faq_data", [])
    monkeypatch.setattr(main, "COURSE_DATA", [])
    monkeypatch.setattr(main, "WEBSITE_DATA", [])
    monkeypatch.setattr(main, "COURSE_CATALOG", CATALOG)
    monkeypatch.setattr(main, "generate_answer", rec.generate_answer)
    monkeypatch.setattr(main, "retrieve_hits", rec.retrieve_hits)
    return TestClient(main.app)


def ask(client, message, session="page-1"):
    r = client.post("/chat", json={"message": message, "session_id": session})
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------- 6, 10, 15

def test_stated_field_drives_a_later_bare_recommendation(client, rec):
    """6. 'I am an engineer' then 'suggest me a course' must use engineering."""
    rec.answer = "Indian Knowledge Systems for Engineers fits your background."

    ask(client, "I am an engineer.")
    out = ask(client, "Suggest me a course.")

    assert out["status"] == "ok"
    call = rec.calls[-1]
    assert "engineering" in call["context"].lower()
    assert "engineering" in call["session_context"].lower()
    assert "Indian Knowledge Systems for Engineers" in out["answer"]


def test_bare_recommendation_without_any_field_asks_for_one(client, rec):
    """3rd priority: no subject anywhere -> ask, never guess."""
    out = ask(client, "Suggest me a course.")
    assert "which subject" in out["answer"].lower()
    assert rec.calls == [], "must not call the model when no subject is known"


def test_explicit_subject_beats_session_field(client, rec):
    """8 (ordering rule): a self-contained topic is not polluted by older context."""
    rec.answer = "Samskrit 1: Thinking in Samskrit is a language course."

    ask(client, "I am an engineer.")
    ask(client, "Suggest me a course on samskrit.")

    context = rec.calls[-1]["context"]
    assert "Subject to match: samskrit" in context
    assert "Subject to match: engineering" not in context


def test_newest_field_replaces_the_older_one(client, rec):
    """10. A new explicit field wins."""
    rec.answer = "Indic Business and Commerce Traditions suits commerce."

    ask(client, "I am an engineer.")
    ask(client, "Actually my stream is commerce.")
    ask(client, "Suggest me a course.")

    session_context = rec.calls[-1]["session_context"].lower()
    assert "commerce" in session_context
    assert "engineering" not in session_context


def test_recommendation_must_name_a_real_catalog_course(client, rec):
    """15. A fabricated course name is refused, not passed through."""
    rec.answer = "I recommend Advanced Quantum Ayurveda, a great fit for engineers."

    ask(client, "I am an engineer.")
    out = ask(client, "Suggest me a course.")

    assert out["status"] == "refused"
    assert out["answer"] == main.REFUSAL_MESSAGE


# ------------------------------------------------------------------- 7, 8

def test_why_that_course_resolves_against_the_recommendation(client, rec):
    """7. 'Why did you select that course?' knows the course and the reason."""
    rec.answer = "Indian Knowledge Systems for Engineers fits your background."
    ask(client, "I am an engineer.")
    ask(client, "Suggest me a course.")

    rec.answer = "It covers Indic approaches to design for technical students."
    out = ask(client, "Why did you select that course?")

    assert out["status"] == "ok"
    call = rec.calls[-1]
    # Both halves of the required understanding are present.
    assert "Indian Knowledge Systems for Engineers" in call["session_context"]
    assert "engineering" in call["session_context"].lower()
    assert call["history"], "the prior turns must reach the model"


def test_its_duration_resolves_against_the_same_course(client, rec):
    """8. 'What is its duration?' answers for the course under discussion."""
    rec.answer = "Indian Knowledge Systems for Engineers fits your background."
    ask(client, "I am an engineer.")
    ask(client, "Suggest me a course.")

    rec.answer = "The duration is 12 weeks."
    out = ask(client, "What is its duration?")

    assert out["status"] == "ok"
    assert "12 weeks" in out["answer"]
    # Retrieval was anchored on the conversation, not the bare pronoun.
    assert rec.queries[-1] != "What is its duration?"
    assert "engineer" in rec.queries[-1].lower()


# ---------------------------------------------------------------- 9, 11

def test_fresh_session_cannot_recall_a_previous_stream(client):
    """9. After a refresh (new session id) the stream is genuinely unknown."""
    ask(client, "I am an engineer.", session="page-1")

    out = ask(client, "What is my stream?", session="page-2-after-refresh")

    assert out["answer"] == main._UNKNOWN_PROFILE_REPLY
    assert "engineer" not in out["answer"].lower()


def test_same_session_can_recall_the_stream(client):
    """The flip side: within one page the detail is retained."""
    ask(client, "I am an engineer.", session="page-1")
    out = ask(client, "What is my stream?", session="page-1")
    assert "engineering" in out["answer"].lower()


@pytest.mark.parametrize(
    "message",
    [
        "what is my stream?",
        "what is my field",
        "do you remember my background",
    ],
)
def test_profile_questions_are_answered_from_the_session(message):
    assert main.is_own_profile_question(message)


@pytest.mark.parametrize(
    "message",
    [
        "which course suits my background?",
        "what courses are in my field?",
        "which course is best for my field",
        "which courses match my background",
        "suggest a course for my stream",
        "what is the duration of my course",
    ],
)
def test_course_requests_are_not_hijacked_by_the_profile_router(message):
    """These name a course/programme: they are requests, not memory questions."""
    assert not main.is_own_profile_question(message)


def test_course_request_mentioning_background_still_recommends(client, rec):
    rec.answer = "Indian Knowledge Systems for Engineers fits your background."
    ask(client, "I am an engineer.")
    out = ask(client, "Which course suits my background?")

    assert out["status"] == "ok"
    assert "Indian Knowledge Systems for Engineers" in out["answer"]
    assert out["answer"] != main._UNKNOWN_PROFILE_REPLY


def test_two_sessions_never_share_context(client, rec):
    """11. Distinct ids are fully isolated."""
    ask(client, "I am an engineer.", session="tab-a")
    rec.calls.clear()

    out = ask(client, "Suggest me a course.", session="tab-b")

    assert "which subject" in out["answer"].lower()
    assert rec.calls == [], "tab-b must not inherit tab-a's field"


def test_history_is_scoped_to_one_session(client, store):
    ask(client, "hello", session="s-one")
    ask(client, "hello", session="s-two")
    assert set(store) == {"s-one", "s-two"}
    for session_id, items in store.items():
        assert all(i["session_id"] == session_id for i in items)


# ------------------------------------------------------------------- 13

def test_history_is_chronologically_ordered(client, store):
    """13. Ordered even when several writes share a millisecond."""
    ask(client, "I am an engineer.", session="ordered")
    ask(client, "What is my stream?", session="ordered")

    items = main.get_recent_messages(session_id="ordered", limit=50)
    assert [i["role"] for i in items] == ["user", "assistant", "user", "assistant"]
    assert [i["text"] for i in items][0] == "I am an engineer."
    # Sort keys are strictly increasing despite the identical timestamp.
    keys = [i["sk"] for i in items]
    assert keys == sorted(keys)
    assert len({i["ts"] for i in items}) == 1, "fixture writes share one millisecond"


def test_model_receives_history_oldest_first(client, rec):
    rec.answer = "Indian Knowledge Systems for Engineers fits your background."
    ask(client, "I am an engineer.")
    ask(client, "Suggest me a course.")

    roles = [m["role"] for m in rec.calls[-1]["history"]]
    texts = [m["content"][0]["text"] for m in rec.calls[-1]["history"]]
    assert roles[0] == "user"
    assert texts[0] == "I am an engineer."


# ------------------------------------------------------------------- 14

def test_store_read_failure_is_observable(client, monkeypatch):
    """14. A read outage must not masquerade as an empty conversation."""
    def boom(*a, **k):
        raise chat_store.ChatStoreError("dynamodb unavailable")

    monkeypatch.setattr(main, "get_recent_messages", boom)

    out = ask(client, "hello")

    assert out["continuity"] == "degraded"
    assert out["status"] == "ok", "the visitor still gets an answer"
    # No internal detail leaks to the browser.
    assert "dynamodb" not in str(out).lower()


def test_store_write_failure_is_observable(client, monkeypatch):
    def boom(*a, **k):
        raise chat_store.ChatStoreError("dynamodb unavailable")

    monkeypatch.setattr(main, "put_message", boom)

    out = ask(client, "hello")

    assert out["continuity"] == "degraded"
    assert "dynamodb" not in str(out).lower()


def test_healthy_store_reports_ok(client):
    assert ask(client, "hello")["continuity"] == "ok"


def test_store_failure_is_logged_with_the_cause(monkeypatch, caplog):
    """The outage must be diagnosable from the logs, not just flagged."""
    from botocore.exceptions import ClientError

    class DeadTable:
        def query(self, **kwargs):
            raise ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": "no table"}},
                "Query",
            )

    monkeypatch.setattr(chat_store, "_get_table", lambda: DeadTable())

    with caplog.at_level("ERROR", logger="siddh_guide.chat_store"):
        with pytest.raises(chat_store.ChatStoreError):
            chat_store.get_recent_messages("sess-x")

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "CONTINUITY" in messages
    assert "sess-x" in messages
    # The underlying exception is attached for diagnosis.
    assert any(r.exc_info for r in caplog.records)


def test_chat_store_raises_instead_of_returning_empty(monkeypatch):
    """The store itself must distinguish 'no history' from 'unavailable'."""
    from botocore.exceptions import ClientError

    class DeadTable:
        def query(self, **kwargs):
            raise ClientError({"Error": {"Code": "ResourceNotFoundException",
                                         "Message": "no table"}}, "Query")

        def put_item(self, **kwargs):
            raise ClientError({"Error": {"Code": "ResourceNotFoundException",
                                         "Message": "no table"}}, "PutItem")

    monkeypatch.setattr(chat_store, "_get_table", lambda: DeadTable())

    with pytest.raises(chat_store.ChatStoreError):
        chat_store.get_recent_messages("s")
    with pytest.raises(chat_store.ChatStoreError):
        chat_store.put_message("s", "user", "hi")


# ------------------------------------------------- persistence of every path

@pytest.mark.parametrize(
    "message",
    [
        "hello",                     # greeting
        "who are you",               # about-bot
        "what can you do",           # capability
        "what is my stream?",        # session profile
        "how many courses are there",  # course count
        "list all courses",          # course list
        "suggest me a course",       # recommendation (asks for subject)
        "what is its duration?",     # RAG
        "what is the weather",       # refusal
    ],
)
def test_every_path_persists_both_turns(client, store, message):
    ask(client, message, session="persist")
    roles = [i["role"] for i in store["persist"]]
    assert roles == ["user", "assistant"], f"{message!r} stored {roles}"


# ------------------------------------------- FAQ vs the course under discussion

# Mirrors the real data/faq.json entry: multi-word keywords, matched anywhere.
FAQ_ENROLL = [{"keywords": ["how do i enroll", "how to enroll", "enroll"],
               "answer": "Open the course page and use the Click To Enroll button."}]


def test_faq_answer_used_when_no_course_is_under_discussion(client, monkeypatch, rec):
    monkeypatch.setattr(main, "faq_data", FAQ_ENROLL)

    out = ask(client, "how do i enroll")

    assert out["sources"] == ["faq"]
    assert "Click To Enroll" in out["answer"]


def test_followup_prefers_the_course_specific_answer(client, monkeypatch, rec):
    """'How do I enroll?' right after a course must answer for that course."""
    monkeypatch.setattr(main, "faq_data", FAQ_ENROLL)

    rec.answer = "Indian Knowledge Systems for Engineers fits your background."
    ask(client, "I am an engineer.")
    ask(client, "Suggest me a course.")

    rec.answer = "Enrollment for Indian Knowledge Systems for Engineers runs 12 weeks."
    out = ask(client, "how do i enroll")

    assert out["sources"] == ["rag", "nova"], "should not fall back to the generic FAQ"
    assert "Indian Knowledge Systems for Engineers" in out["answer"]


def test_falls_back_to_faq_and_writes_one_turn_when_unsupported(client, monkeypatch, rec):
    """A failed course-specific attempt must not leave a stray assistant turn."""
    monkeypatch.setattr(main, "faq_data", FAQ_ENROLL)

    rec.answer = "Indian Knowledge Systems for Engineers fits your background."
    ask(client, "I am an engineer.", session="fb")
    ask(client, "Suggest me a course.", session="fb")

    # Ungrounded answer -> the course-specific attempt is rejected.
    rec.answer = "Enrollment costs 4321 rupees and closes on 9 September."
    out = ask(client, "how do i enroll", session="fb")

    assert out["sources"] == ["faq"]
    assert "Click To Enroll" in out["answer"]

    history = main.get_recent_messages(session_id="fb", limit=50)
    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant"] * 3, f"history not write-once: {roles}"


def test_grounding_is_not_weakened_by_session_context(client, rec):
    """Session details may personalise wording, never invent course facts."""
    ask(client, "I am an engineer.")
    rec.answer = "The fee is 9999 rupees and it runs for 3 days."
    out = ask(client, "What is its duration?")
    assert out["status"] == "refused", "unsupported numbers must still be refused"
