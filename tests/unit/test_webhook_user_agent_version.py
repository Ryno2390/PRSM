"""Webhook User-Agent header version not stale (sprint 130).

Pre-fix: WebhookDeliverer hardcoded `User-Agent: prsm-node/
0.24.0` — same stale version that sprint 112 fixed in the
FastAPI app config. Receivers integrating against PRSM
webhooks would see the wrong version forever.

Caught via real dogfood: the user inspected the actual webhook
landing at webhook.site and noticed the stale User-Agent. No
unit test would have caught it (we never asserted on the
User-Agent value).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.node.webhook_delivery import WebhookDeliverer


@pytest.mark.asyncio
async def test_user_agent_uses_package_version():
    captured = {}

    async def capture_post(url, body, headers):
        captured["headers"] = headers
        return 200, "ok"

    deliverer = WebhookDeliverer()
    await deliverer.deliver(
        url="https://hook.example.com",
        event="webhook.test",
        payload={"x": 1},
        post_fn=capture_post,
    )
    ua = captured["headers"]["User-Agent"]
    assert ua.startswith("prsm-node/")
    # Version field must NOT be the stale 0.24.0
    version = ua.split("/", 1)[1]
    assert version != "0.24.0"


@pytest.mark.asyncio
async def test_user_agent_matches_pyproject():
    from pathlib import Path
    repo = Path(__file__).parent.parent.parent
    expected = None
    for line in (repo / "pyproject.toml").read_text().splitlines():
        if line.startswith("version = "):
            expected = line.split("=", 1)[1].strip().strip('"')
            break
    if expected is None:
        pytest.skip("Could not read pyproject.toml")

    captured = {}

    async def capture_post(url, body, headers):
        captured["headers"] = headers
        return 200, "ok"

    await WebhookDeliverer().deliver(
        url="https://hook.example.com",
        event="webhook.test",
        payload={},
        post_fn=capture_post,
    )
    ua = captured["headers"]["User-Agent"]
    # Either the canonical version OR "unknown" (source-only run)
    assert ua in (
        f"prsm-node/{expected}",
        "prsm-node/unknown",
    )
