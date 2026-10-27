"""skf_admin_config overlay reader - vendored into Sid Guru, Ask Sid and the
WhatsApp/Zoho support bot so an operator can change settings from the SKF
Admin Console without a deploy.

Contract (see skf-admin-console/docs/adr/0002-config-overlay.md):

    get(system, key, default) -> value

* Reads one DynamoDB item from the table named by SKF_ADMIN_CONFIG_TABLE.
  When that variable is unset the overlay is OFF and every call returns
  `default` immediately - which is exactly how each system behaved before the
  overlay existed.
* Results (including "not found") are cached for SKF_ADMIN_CONFIG_TTL_SECONDS
  (default 60) per warm process, so the hot path costs one GetItem a minute.
* ANY failure returns `default` and is logged once per process. The console
  being down can never make the system it configures worse.
* Values are JSON-decoded when they parse as JSON, otherwise returned as the
  raw string (markdown documents). Typed helpers coerce to the default's type
  and fall back to `default` on a bad value.

Python 3.11+ (Ask Sid runs 3.11). Depends only on boto3, which every one of the
three runtimes already has.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger("skf_admin_config")

TABLE_ENV = "SKF_ADMIN_CONFIG_TABLE"
TTL_ENV = "SKF_ADMIN_CONFIG_TTL_SECONDS"
REGION_ENV = "SKF_ADMIN_CONFIG_REGION"

_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, Any]] = {}
_client = None
_failure_logged = False
_MISSING = object()


def table_name() -> str:
    return os.environ.get(TABLE_ENV, "").strip()


def _ttl() -> float:
    try:
        return float(os.environ.get(TTL_ENV, "60"))
    except ValueError:
        return 60.0


def _get_client():
    global _client
    if _client is None:
        import boto3

        region = os.environ.get(REGION_ENV) or os.environ.get("AWS_REGION") \
            or os.environ.get("AWS_DEFAULT_REGION") or "ap-south-1"
        _client = boto3.client("dynamodb", region_name=region)
    return _client


def _parse(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _fetch(system: str, key: str) -> Any:
    """One GetItem. Returns the decoded value or _MISSING."""
    resp = _get_client().get_item(
        TableName=table_name(),
        Key={"system": {"S": system}, "key": {"S": key}},
    )
    item = resp.get("Item")
    if not item or "value" not in item:
        return _MISSING
    return _parse(item["value"].get("S", ""))


def get(system: str, key: str, default: Any = None) -> Any:
    """Return the overlay value for (system, key), or `default`."""
    global _failure_logged
    if not table_name():
        return default

    cache_key = (system, key)
    now = time.monotonic()
    with _lock:
        hit = _cache.get(cache_key)
    if hit and now - hit[0] < _ttl():
        value = hit[1]
        return default if value is _MISSING else value

    try:
        value = _fetch(system, key)
    except Exception as e:  # noqa: BLE001 - the overlay must never raise
        if not _failure_logged:
            logger.warning("skf_admin_config unavailable (%s); using defaults", type(e).__name__)
            _failure_logged = True
        value = _MISSING
    with _lock:
        _cache[cache_key] = (now, value)
    return default if value is _MISSING else value


# ---- typed helpers ---------------------------------------------------------

def get_str(system: str, key: str, default: str) -> str:
    value = get(system, key, default)
    return value if isinstance(value, str) else default


def get_int(system: str, key: str, default: int) -> int:
    value = get(system, key, default)
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_float(system: str, key: str, default: float) -> float:
    value = get(system, key, default)
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_bool(system: str, key: str, default: bool) -> bool:
    value = get(system, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def get_list(system: str, key: str, default: list) -> list:
    value = get(system, key, default)
    return list(value) if isinstance(value, (list, tuple)) else default


def get_dict(system: str, key: str, default: dict) -> dict:
    value = get(system, key, default)
    return dict(value) if isinstance(value, dict) else default


def setting(system: str, name: str, default: Any) -> Any:
    """Convenience for the common shape: one JSON object under key "settings"
    holding many named values. Returns default when the name is absent."""
    values = get_dict(system, "settings", {})
    return values.get(name, default)


def invalidate() -> None:
    """Drop the cache (tests, or after a manual refresh endpoint)."""
    global _failure_logged
    with _lock:
        _cache.clear()
    _failure_logged = False
