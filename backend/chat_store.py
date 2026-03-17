import logging
import time
import uuid
from typing import Dict, List, Optional

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from config import AWS_BOTO_CONFIG, AWS_REGION, CHAT_TABLE, CHAT_TTL_DAYS

logger = logging.getLogger("siddh_guide.chat_store")

_dynamodb = None
_table = None


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
    return f"{ts_ms:013d}#{uuid.uuid4().hex}"


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
    except ClientError:
        logger.exception("put_message failed")


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
    except ClientError:
        logger.exception("get_recent_messages failed")
        return []

    items = resp.get("Items", [])
    items.sort(key=lambda x: (x.get("ts", 0), x.get("sk", "")))
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