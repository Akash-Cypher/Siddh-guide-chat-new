"""Suite-wide fixtures.

The app embeds a probe string against Bedrock at boot. Off here so no test ever
reaches AWS: CI has no credentials, and botocore's credential search alone adds
a metadata-service timeout to every app boot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402


@pytest.fixture(autouse=True)
def _no_boot_time_bedrock_calls(monkeypatch):
    monkeypatch.setattr(main, "EMBED_SELF_CHECK", False)
