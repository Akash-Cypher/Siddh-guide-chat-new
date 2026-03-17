import hashlib
import logging
import re
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from config import (
    AWS_BOTO_CONFIG,
    AWS_REGION,
    CACHE_TABLE,
    CACHE_TTL_SECONDS,
    CACHE_VERSION,
    MAX_CACHE_ANSWER_CHARS,
)

logger = logging.getLogger("siddh_guide.cache")

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_dynamodb = None
_table = None


def _get_table():
    global _dynamodb, _table
    if _table is None:
        _dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION, config=AWS_BOTO_CONFIG)
        _table = _dynamodb.Table(CACHE_TABLE)
    return _table


def _norm(q: str) -> str:
    q = (q or "").lower().strip()
    q = _PUNCT_RE.sub("", q)
    return " ".join(q.split())


def make_key(q: str, version: str = CACHE_VERSION) -> str:
    raw = f"{version}:{_norm(q)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached(q: str) -> Optional[str]:
    key = make_key(q)
    table = _get_table()

    try:
        resp = table.get_item(Key={"pk": key})
    except ClientError:
        logger.exception("cache get_item failed")
        return None

    item = resp.get("Item")
    if not item:
        return None

    ans = item.get("answer")
    return ans if isinstance(ans, str) and ans.strip() else None


def put_cached(q: str, answer: str) -> None:
    if not isinstance(answer, str) or not answer.strip():
        return

    ans = answer.strip()
    if len(ans) > MAX_CACHE_ANSWER_CHARS:
        ans = ans[:MAX_CACHE_ANSWER_CHARS].rsplit(" ", 1)[0] + "…"

    key = make_key(q)
    ttl = int(time.time()) + CACHE_TTL_SECONDS
    table = _get_table()

    try:
        table.put_item(
            Item={
                "pk": key,
                "answer": ans,
                "ttl": ttl,
            }
        )
    except ClientError:
        logger.exception("cache put_item failed")