"""
Knowledge-base refresh orchestration.

`refresh_knowledge_base()` crawls the live sites (crawler.py), rewrites the JSON
data files, and re-embeds them into Chroma (rag.py). It is deliberately tolerant:
any failure leaves the last-good data + index in place so the chatbot keeps
serving. This is the single entry point used by both the scheduled background
refresh and the manual /admin/refresh endpoint.
"""

from __future__ import annotations

import logging
import threading

from config import CRAWL_REBUILD_INDEX, DATA_DIR
from crawler import crawl_to_files

logger = logging.getLogger("siddh_guide.refresh")

# Guarantees only one crawl/re-embed runs at a time across startup + schedule +
# manual triggers, so we never delete/re-add the Chroma collection concurrently.
_refresh_lock = threading.Lock()


def refresh_knowledge_base(rebuild_index: bool | None = None) -> dict:
    if rebuild_index is None:
        rebuild_index = CRAWL_REBUILD_INDEX

    if not _refresh_lock.acquire(blocking=False):
        logger.info("refresh: another refresh is already running; skipping")
        return {"ok": False, "skipped": "already_running"}

    try:
        summary = crawl_to_files(str(DATA_DIR))

        if summary.get("ok") and rebuild_index:
            try:
                from rag import build_index_from_json_folder

                build_index_from_json_folder(str(DATA_DIR))
                summary["reindexed"] = True
                logger.info("refresh: Chroma index rebuilt")
            except Exception:
                logger.exception("refresh: reindex failed (data files were still updated)")
                summary["reindexed"] = False

        return summary
    except Exception:
        logger.exception("refresh: crawl failed; existing data left untouched")
        return {"ok": False, "error": "crawl_failed"}
    finally:
        _refresh_lock.release()
