"""key.released webhook delivery — KeyDistributionWatcher fires
key.released webhook when an on-chain KeyReleased event is
observed. Symmetric to slash.* (sprint 85) and
distribution.distributed (sprint 88).
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.key_distribution import KeyReleasedEvent
from prsm.node.node import _build_key_distribution_watcher_or_none


@pytest.fixture
def opted_in_env():
    with patch.dict(os.environ, {
        "PRSM_KEY_DISTRIBUTION_WATCHER_ENABLED": "1",
    }):
        yield


@pytest.fixture
def deliverer():
    d = MagicMock()
    async def fake_deliver(**kwargs):
        d.last_kwargs = kwargs
        result = MagicMock()
        result.success = True
        return result
    d.deliver = MagicMock(side_effect=fake_deliver)
    return d


def _event():
    return KeyReleasedEvent(
        content_hash=b"\x11" * 32,
        recipient="0xRECIPIENT",
        encrypted_key=b"\xab\xcd\xef",
    )


def test_dispatches_webhook_on_key_released(opted_in_env, deliverer):
    watcher = _build_key_distribution_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=deliverer,
        webhook_url="https://hook.example.com",
        webhook_secret="s3cret",
    )
    asyncio.run(watcher._invoke_cb(watcher._on_released, _event()))

    assert deliverer.deliver.called
    args = deliverer.last_kwargs
    assert args["url"] == "https://hook.example.com"
    assert args["event"] == "key.released"
    assert args["secret"] == "s3cret"
    assert args["payload"]["content_hash"] == "0x" + "11" * 32
    assert args["payload"]["recipient"] == "0xRECIPIENT"


def test_no_deliverer_does_not_crash(opted_in_env):
    watcher = _build_key_distribution_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=None,
        webhook_url=None,
    )
    asyncio.run(watcher._invoke_cb(watcher._on_released, _event()))


def test_deliverer_failure_isolated(opted_in_env):
    bad = MagicMock()
    async def boom(**kwargs):
        raise RuntimeError("simulated")
    bad.deliver = MagicMock(side_effect=boom)
    watcher = _build_key_distribution_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=bad,
        webhook_url="https://hook.example.com",
    )
    # Must NOT raise
    asyncio.run(watcher._invoke_cb(watcher._on_released, _event()))


def test_deliverer_without_url_skips(opted_in_env, deliverer):
    watcher = _build_key_distribution_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=deliverer,
        webhook_url=None,
    )
    asyncio.run(watcher._invoke_cb(watcher._on_released, _event()))
    assert not deliverer.deliver.called
