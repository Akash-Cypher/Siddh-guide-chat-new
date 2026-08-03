"""Frontend session-lifecycle contracts.

A conversation must live exactly as long as the loaded page: one id generated in
memory at init, reused for every message, gone on refresh.

These assert against the shipped widget sources. They are static contract checks,
not a browser run — there is no JS runtime in this project's toolchain — but they
pin the properties that actually regress: someone reintroducing sessionStorage,
dropping the in-flight lock, or adding a history fetch on load.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WIDGET_JS = ROOT / "frontend" / "siddh-guide-chat" / "assets" / "siddh-guide-chat-widget.js"
TEST_PAGE = ROOT / "frontend" / "siddh-guide-chat" / "siddh-guide-chat-test.html"
PLUGIN_PHP = ROOT / "frontend" / "siddh-guide-chat" / "siddh-guide-chat.php"


def strip_comments(source: str) -> str:
    """Drop // and /* */ comments so prose about storage is not mistaken for code."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    source = re.sub(r"(?m)^\s*//.*$", "", source)
    source = re.sub(r"(?m)//.*$", "", source)
    return source


@pytest.fixture(scope="module")
def widget_code() -> str:
    return strip_comments(WIDGET_JS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def test_page_code() -> str:
    html = TEST_PAGE.read_text(encoding="utf-8")
    script = html.split("<script>")[-1].split("</script>")[0]
    return strip_comments(script)


ALL_WIDGETS = ["widget_code", "test_page_code"]


# 1. One id per page lifecycle, reused for every message.
@pytest.mark.parametrize("fixture_name", ALL_WIDGETS)
def test_session_id_generated_once_at_init(fixture_name, request):
    code = request.getfixturevalue(fixture_name)

    assert re.search(r"const\s+SESSION_ID\s*=\s*generateSessionId\(\)", code), (
        "SESSION_ID must be generated once at init and be const"
    )
    # No reassignment anywhere (that would silently split a conversation).
    assert not re.search(r"(?<!const\s)\bSESSION_ID\s*=(?!=)", code)
    assert len(re.findall(r"\bSESSION_ID\s*=(?!=)", code)) == 1


@pytest.mark.parametrize("fixture_name", ALL_WIDGETS)
def test_every_message_sends_that_same_id(fixture_name, request):
    code = request.getfixturevalue(fixture_name)
    sends = re.findall(r"session_id\s*:\s*([A-Za-z_$][\w$]*)", code)
    assert sends, "the request body must carry a session_id"
    assert set(sends) == {"SESSION_ID"}, f"unexpected session id source: {sends}"


# 2. A refresh must produce a completely new id.
# 3. Closing/reopening the bubble must retain it.
# 4. No storage-backed session may be restored.
@pytest.mark.parametrize("fixture_name", ALL_WIDGETS)
def test_no_persistent_storage_anywhere(fixture_name, request):
    code = request.getfixturevalue(fixture_name)
    for api in ("sessionStorage", "localStorage", "document.cookie", "indexedDB"):
        assert api not in code, (
            f"{api} would survive a refresh and leak the previous conversation"
        )


@pytest.mark.parametrize("fixture_name", ALL_WIDGETS)
def test_id_derives_only_from_generator(fixture_name, request):
    """Refresh isolation follows from this: the id has no source but the RNG, so
    re-running the script cannot reproduce the previous value."""
    code = request.getfixturevalue(fixture_name)
    assert re.search(r"function\s+generateSessionId", code)
    assert "randomUUID" in code
    assert "getItem" not in code, "reading any store would defeat refresh isolation"
    assert "setItem" not in code


def test_bubble_toggle_does_not_touch_the_session(widget_code):
    """Closing/reopening the bubble is pure UI - it must not reset the id."""
    handler = widget_code.split("chatToggleButton.addEventListener")[1].split("});")[0]
    assert "SESSION_ID" not in handler
    assert "generateSessionId" not in handler


# 5. No history restoration on page init.
@pytest.mark.parametrize("fixture_name", ALL_WIDGETS)
def test_no_history_request_on_init(fixture_name, request):
    code = request.getfixturevalue(fixture_name)
    assert "/history" not in code, "the widget must never fetch prior transcripts"

    fetches = re.findall(r"fetch\(\s*`([^`]+)`", code)
    assert fetches, "expected at least the chat request"
    assert all(url.endswith("/chat") for url in fetches), fetches


@pytest.mark.parametrize("fixture_name", ALL_WIDGETS)
def test_no_new_chat_button(fixture_name, request):
    code = request.getfixturevalue(fixture_name)
    assert "resetChatSession" not in code
    assert not re.search(r"new\s+chat", code, re.I)


# 12. Overlapping submissions are blocked.
@pytest.mark.parametrize("fixture_name", ALL_WIDGETS)
def test_only_one_request_in_flight(fixture_name, request):
    code = request.getfixturevalue(fixture_name)

    assert re.search(r"let\s+requestInFlight\s*=\s*false", code)
    # The submit handler bails out while a request is open.
    submit = code.split("chatForm.addEventListener")[1]
    assert re.search(r"if\s*\(\s*requestInFlight\s*\)\s*return", submit)

    # Both controls are disabled while busy ...
    assert re.search(r"userInput\.disabled\s*=\s*busy", code)
    assert re.search(r"submitButton\.disabled\s*=\s*busy", code)
    # ... and always restored, including on failure.
    assert re.search(r"finally\s*\{\s*(?://[^\n]*\n\s*)*setBusy\(false\)", code), (
        "controls must be restored in a finally block, or an error locks the widget"
    )


@pytest.mark.parametrize("fixture_name", ALL_WIDGETS)
def test_busy_is_set_before_the_request(fixture_name, request):
    code = request.getfixturevalue(fixture_name)
    body = code.split("function fetchBotResponse")[1]
    assert body.index("setBusy(true)") < body.index("fetch("), (
        "the lock must close before the request starts, not after"
    )


# WordPress cache-busting: a stale cached copy would keep the old
# sessionStorage build alive and silently undo refresh isolation.
def test_release_zip_matches_the_sources():
    """The handover zip is what actually reaches WordPress.

    A zip built before a widget change ships the old script and silently undoes
    the session-lifecycle fix, so it must be byte-identical to the sources.
    """
    import zipfile

    zip_path = ROOT / "siddh-guide-chat.zip"
    if not zip_path.exists():
        pytest.skip("no release zip built yet")

    expected = {
        "siddh-guide-chat/siddh-guide-chat.php": PLUGIN_PHP,
        "siddh-guide-chat/siddh-guide-chat-test.html": TEST_PAGE,
        "siddh-guide-chat/assets/siddh-guide-chat-widget.js": WIDGET_JS,
        "siddh-guide-chat/assets/style.css": WIDGET_JS.parent / "style.css",
        "siddh-guide-chat/assets/siddhanta-logo.png": WIDGET_JS.parent / "siddhanta-logo.png",
    }

    with zipfile.ZipFile(zip_path) as z:
        assert set(z.namelist()) == set(expected), (
            "the zip must contain exactly the plugin folder, nothing else"
        )
        for entry, source in expected.items():
            assert z.read(entry) == source.read_bytes(), (
                f"{entry} in the zip is stale - rebuild it before handing over"
            )


def test_bundled_test_page_is_not_indexable():
    """It ships inside the plugin, so it is live-reachable. Keep it out of search."""
    html = TEST_PAGE.read_text(encoding="utf-8")
    assert re.search(
        r'<meta\s+name="robots"\s+content="[^"]*noindex', html, re.I
    ), "the bundled test page must carry a noindex robots meta"


def test_asset_version_is_file_derived_not_hardcoded():
    php = PLUGIN_PHP.read_text(encoding="utf-8")

    assert "function siddh_guide_chat_asset_version" in php
    assert "filemtime" in php
    # The enqueue calls and the version filter must both use it.
    assert php.count("siddh_guide_chat_asset_version(") >= 4
    assert not re.search(r"add_query_arg\(\s*'v'\s*,\s*'1\.0\.0'", php), (
        "a hardcoded asset version lets caches pin visitors to the old widget"
    )
