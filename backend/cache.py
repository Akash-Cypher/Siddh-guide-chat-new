import os
import time
import hashlib
import boto3
import re
from botocore.exceptions import ClientError

CACHE_TABLE = os.getenv("CACHE_TABLE", "SiddhGuideCache")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "259200"))  # 3 days default

REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "ap-south-1"

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(CACHE_TABLE)

# Safety: DynamoDB item max is 400KB, keep a buffer
MAX_ANSWER_CHARS = int(os.getenv("MAX_CACHE_ANSWER_CHARS", "12000"))

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

def _norm(q: str) -> str:
    q = (q or "").lower().strip()
    q = _PUNCT_RE.sub("", q)  # remove punctuation so "IKS?" == "IKS"
    return " ".join(q.split())

def make_key(q: str, version: str = "v1") -> str:
    raw = f"{version}:{_norm(q)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def get_cached(q: str) -> str | None:
    key = make_key(q)
    try:
        resp = table.get_item(Key={"pk": key})
    except ClientError:
        # Fail-open: cache should never break chat
        return None

    item = resp.get("Item")
    if not item:
        return None

    ans = item.get("answer")
    return ans if isinstance(ans, str) and ans.strip() else None

def put_cached(q: str, answer: str):
    if not isinstance(answer, str) or not answer.strip():
        return

    # Clamp overly long answers so put_item doesn't fail
    ans = answer.strip()
    if len(ans) > MAX_ANSWER_CHARS:
        ans = ans[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + "…"

    key = make_key(q)
    ttl = int(time.time()) + CACHE_TTL_SECONDS

    try:
        table.put_item(
            Item={
                "pk": key,
                "answer": ans,
                "ttl": ttl,
            }
        )
    except ClientError:
        # Fail-open
        return