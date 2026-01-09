import os
import time
import hashlib
import boto3

CACHE_TABLE = os.getenv("CACHE_TABLE", "SiddhGuideCache")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "259200"))  # 3 days default

REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "ap-south-1"

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(CACHE_TABLE)

def _norm(q: str) -> str:
    return " ".join((q or "").lower().strip().split())

def make_key(q: str, version: str = "v1") -> str:
    raw = f"{version}:{_norm(q)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def get_cached(q: str) -> str | None:
    key = make_key(q)
    resp = table.get_item(Key={"pk": key})
    item = resp.get("Item")
    return item.get("answer") if item else None

def put_cached(q: str, answer: str):
    key = make_key(q)
    ttl = int(time.time()) + CACHE_TTL_SECONDS
    table.put_item(Item={
        "pk": key,
        "q": q,
        "answer": answer,
        "ttl": ttl
    })
