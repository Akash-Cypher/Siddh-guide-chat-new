"""_current_faq_data() falls back to the bundled faq_data whenever the SKF
Admin Console's override is unset (the normal case - no SKF_ADMIN_CONFIG_TABLE
is set in this test suite, so config_overlay.get() always returns its
caller's default), and honors the override when one is set. get_faq_answer()
and get_why_choose_siddhanta_answer() are unchanged otherwise - this only
covers the new indirection.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402


def test_falls_back_to_bundled_faq_data_when_no_override(monkeypatch):
    bundled = [{"keywords": ["refund"], "answer": "bundled answer"}]
    monkeypatch.setattr(main, "faq_data", bundled)
    monkeypatch.setattr(main.config_overlay, "get", lambda *a, **k: None)
    assert main._current_faq_data() == bundled


def test_uses_the_override_when_set_as_json_string(monkeypatch):
    monkeypatch.setattr(main, "faq_data", [{"keywords": ["refund"], "answer": "bundled"}])
    override = '[{"keywords": ["refund"], "answer": "overridden answer"}]'
    monkeypatch.setattr(main.config_overlay, "get", lambda *a, **k: override)
    assert main._current_faq_data() == [{"keywords": ["refund"], "answer": "overridden answer"}]


def test_get_faq_answer_reads_through_the_override(monkeypatch):
    monkeypatch.setattr(main, "faq_data", [])
    monkeypatch.setattr(main, "COURSE_DATA", [])
    override = '[{"keywords": ["refund policy"], "answer": "Refunds take 5 days."}]'
    monkeypatch.setattr(main.config_overlay, "get", lambda *a, **k: override)
    assert main.get_faq_answer("what is your refund policy") == "Refunds take 5 days."


def test_malformed_override_falls_back_to_bundled_faq_data(monkeypatch):
    bundled = [{"keywords": ["refund"], "answer": "bundled answer"}]
    monkeypatch.setattr(main, "faq_data", bundled)
    monkeypatch.setattr(main.config_overlay, "get", lambda *a, **k: "not json")
    assert main._current_faq_data() == bundled


def test_override_that_is_not_a_list_falls_back_to_bundled_faq_data(monkeypatch):
    bundled = [{"keywords": ["refund"], "answer": "bundled answer"}]
    monkeypatch.setattr(main, "faq_data", bundled)
    monkeypatch.setattr(main.config_overlay, "get", lambda *a, **k: '{"not": "a list"}')
    assert main._current_faq_data() == bundled
