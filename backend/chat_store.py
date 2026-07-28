import itertools
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from config import AWS_BOTO_CONFIG, AWS_REGION, CHAT_TABLE, CHAT_TTL_DAYS

logger = logging.getLogger("siddh_guide.chat_store")


class ChatStoreError(RuntimeError):
    """The conversation store could not be read or written.

    Raised instead of returning an empty list, because "no history" and "history
    unavailable" are different states: the first is a genuinely new conversation,
    the second is an outage. Swallowing the difference makes every follow-up
    silently lose its context while the API keeps reporting success.
    """


_dynamodb = None
_table = None

# Monotonic tiebreaker. Two writes inside the same millisecond would otherwise
# sort by the random uuid suffix, which can invert a user/assistant pair and
# corrupt the order history is replayed in.
_seq_lock = threading.Lock()
_seq_counter = itertools.count()


def _get_table():
    global _dynamodb, _table
    if _table is None:
        _dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION, config=AWS_BOTO_CONFIG)
        _table = _dynamodb.Table(CHAT_TABLE)
    return _table


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ttl_epoch() -> int:
    return int(time.time()) + (CHAT_TTL_DAYS * 24 * 60 * 60)


def _make_sort_key(ts_ms: int) -> str:
    # <ms>#<seq>#<uuid> — the millisecond keeps the existing ordering and key
    # width, the sequence makes same-millisecond writes deterministic, and the
    # uuid keeps the key unique across processes.
    with _seq_lock:
        seq = next(_seq_counter) % 1_000_000
    return f"{ts_ms:013d}#{seq:06d}#{uuid.uuid4().hex}"


def put_message(
    session_id: str,
    role: str,
    text: str,
    request_id: str = "",
    sources: Optional[list] = None,
    context_used: int = 0,
) -> None:
    session_id = (session_id or "").strip()
    if not session_id:
        raise ValueError("session_id is required")

    role = (role or "").strip().lower()
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")

    ts_ms = _now_ms()

    item = {
        "session_id": session_id,
        "sk": _make_sort_key(ts_ms),
        "ts": ts_ms,
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "role": role,
        "text": (text or "").strip(),
        "request_id": request_id or "",
        "context_used": int(context_used),
        "ttl": _ttl_epoch(),
    }

    if sources is not None:
        item["sources"] = sources

    table = _get_table()

    try:
        table.put_item(Item=item)
    except ClientError as exc:
        logger.error(
            "CONTINUITY: put_message failed session=%s role=%s table=%s — "
            "this turn will be missing from the conversation history",
            session_id,
            role,
            CHAT_TABLE,
            exc_info=True,
        )
        raise ChatStoreError("could not persist chat message") from exc


def get_recent_messages(session_id: str, limit: int = 8) -> List[Dict]:
    session_id = (session_id or "").strip()
    if not session_id:
        return []

    table = _get_table()

    try:
        resp = table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=False,
            Limit=max(1, limit),
        )
    except ClientError as exc:
        logger.error(
            "CONTINUITY: get_recent_messages failed session=%s table=%s — "
            "follow-ups in this session cannot be resolved",
            session_id,
            CHAT_TABLE,
            exc_info=True,
        )
        raise ChatStoreError("could not read chat history") from exc

    items = resp.get("Items", [])
    # Oldest-first, so the model receives the conversation in the order it
    # happened. DynamoDB returned it newest-first (ScanIndexForward=False).
    items.sort(key=lambda x: (int(x.get("ts", 0) or 0), str(x.get("sk", ""))))
    return items


def history_to_model_messages(messages: List[Dict], max_chars: int = 1200) -> List[Dict]:
    if not messages:
        return []

    selected: List[Dict] = []
    used = 0

    for m in reversed(messages):
        role = (m.get("role") or "").strip().lower()
        text = (m.get("text") or "").strip()

        if role not in {"user", "assistant"} or not text:
            continue

        needed = len(text)
        if selected and used + needed > max_chars:
            break

        selected.append(
            {
                "role": role,
                "content": [{"text": text}],
            }
        )
        used += needed

    selected.reverse()
    return selected