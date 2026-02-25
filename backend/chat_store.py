import os
import time
from typing import List, Dict, Optional

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "ap-south-1"
CHAT_TABLE = os.getenv("CHAT_TABLE", "SiddhGuideChat")
CHAT_TTL_DAYS = int(os.getenv("CHAT_TTL_DAYS", "30"))  # default 30 days

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(CHAT_TABLE)

def _now_ms() -> int:
    return int(time.time() * 1000)

def _ttl_epoch() -> int:
    return int(time.time()) + (CHAT_TTL_DAYS * 24 * 60 * 60)

def put_message(
    session_id: str,
    role: str,
    text: str,
    request_id: str = "",
    sources: Optional[list] = None,
    context_used: int = 0,
):
    if not session_id:
        session_id = "default"

    item = {
        "session_id": session_id,
        "ts": _now_ms(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "role": role,
        "text": (text or "").strip(),
        "request_id": request_id or "",
        "context_used": int(context_used),
        "ttl": _ttl_epoch(),
    }

    if sources is not None:
        item["sources"] = sources

    try:
        table.put_item(Item=item)
    except ClientError as e:
        # Fail-open: history should never break chat
        print(f"[chat_store] put_message failed: {e}")
        return

def get_recent_messages(session_id: str, limit: int = 8) -> List[Dict]:
    if not session_id:
        session_id = "default"

    try:
        resp = table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=False,  # newest first
            Limit=limit,
        )
    except ClientError as e:
        print(f"[chat_store] get_recent_messages failed: {e}")
        return []

    items = resp.get("Items", [])
    items.sort(key=lambda x: x.get("ts", 0))  # oldest -> newest
    return items

def format_history_for_model(messages: List[Dict], max_chars: int = 1200) -> str:
    if not messages:
        return ""

    lines = []
    for m in messages:
        role = (m.get("role") or "").lower()
        text = (m.get("text") or "").strip()
        if not text:
            continue

        if role == "user":
            lines.append(f"User: {text}")
        else:
            lines.append(f"Assistant: {text}")

    history = "\n".join(lines).strip()

    if len(history) > max_chars:
        history = history[-max_chars:]
        if "\n" in history:
            history = history.split("\n", 1)[-1]

    return history