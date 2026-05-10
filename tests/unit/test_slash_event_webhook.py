"""Slash event webhook delivery — when StorageSlashingWatcher
observes a slash, the WebhookDeliverer fires a slash.* event
(in addition to logging + ring-buffer recording).

Two new webhook event names:
  * slash.proof_failure_slashed
  * slash.heartbeat_missing_slashed

Same delivery pipeline as daemon.crashed/recovered: fire-and-
log on failure, never raise from callback. Three-tier sink
isolation continues — log + ring + webhook are independent;
one failing doesn't take down others.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.storage_slashing import (
    HeartbeatMissingSlashedEvent, ProofFailureSlashedEvent,
)
from prsm.node.node import _build_storage_slashing_watcher_or_none


@pytest.fixture
def opted_in_env():
    with patch.dict(os.environ, {
        "PRSM_STORAGE_SLASHING_WATCHER_ENABLED": "1",
    }):
        yield


@pytest.fixture
def deliverer():
    """A stub WebhookDeliverer matching the .deliver() interface."""
    d = MagicMock()
    # Mark as async so iscoroutine check works
    async def fake_deliver(**kwargs):
        d.last_kwargs = kwargs
        result = MagicMock()
        result.success = True
        return result
    d.deliver = MagicMock(side_effect=fake_deliver)
    return d


def test_proof_failure_slash_dispatches_webhook(opted_in_env, deliverer):
    watcher = _build_storage_slashing_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=deliverer,
        webhook_url="https://hook.example.com",
        webhook_secret="s3cret",
    )
    event = ProofFailureSlashedEvent(
        provider="0xPROV", challenger="0xCHAL",
        shard_id=b"\x11" * 32, evidence_hash=b"\x22" * 32,
        slash_id=b"\x33" * 32,
    )
    asyncio.run(watcher._invoke_cb(watcher._on_proof, event))

    assert deliverer.deliver.called
    args = deliverer.last_kwargs
    assert args["url"] == "https://hook.example.com"
    assert args["event"] == "slash.proof_failure_slashed"
    assert args["secret"] == "s3cret"
    assert args["payload"]["provider"] == "0xPROV"
    assert args["payload"]["challenger"] == "0xCHAL"


def test_heartbeat_missing_slash_dispatches_webhook(
    opted_in_env, deliverer,
):
    watcher = _build_storage_slashing_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=deliverer,
        webhook_url="https://hook.example.com",
    )
    event = HeartbeatMissingSlashedEvent(
        provider="0xPROV", challenger="0xCHAL",
        last_heartbeat_at=1700000000,
        slash_id=b"\x44" * 32,
    )
    asyncio.run(watcher._invoke_cb(watcher._on_missing, event))

    assert deliverer.deliver.called
    args = deliverer.last_kwargs
    assert args["event"] == "slash.heartbeat_missing_slashed"
    assert args["payload"]["last_heartbeat_at"] == 1700000000


def test_no_deliverer_arg_does_not_crash(opted_in_env):
    """Without webhook_deliverer, callback still runs (logging
    + ring continue working)."""
    watcher = _build_storage_slashing_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=None,
        webhook_url=None,
    )
    event = HeartbeatMissingSlashedEvent(
        provider="0xPROV", challenger="0xCHAL",
        last_heartbeat_at=0, slash_id=b"\x55" * 32,
    )
    asyncio.run(watcher._invoke_cb(watcher._on_missing, event))


def test_deliverer_failure_isolated_from_callback(opted_in_env):
    """If deliver() raises, the callback completes (logging
    side effect runs, no exception propagates to watcher)."""
    bad_deliverer = MagicMock()
    async def boom(**kwargs):
        raise RuntimeError("simulated deliverer failure")
    bad_deliverer.deliver = MagicMock(side_effect=boom)

    watcher = _build_storage_slashing_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=bad_deliverer,
        webhook_url="https://hook.example.com",
    )
    event = HeartbeatMissingSlashedEvent(
        provider="0xPROV", challenger="0xCHAL",
        last_heartbeat_at=0, slash_id=b"\x66" * 32,
    )
    # MUST NOT raise — watcher's _invoke_cb wraps in try/except
    asyncio.run(watcher._invoke_cb(watcher._on_missing, event))


def test_deliverer_without_url_skips_dispatch(opted_in_env, deliverer):
    """If deliverer is set but URL is unset, skip dispatch
    (mirrors the DaemonWatchdog pattern)."""
    watcher = _build_storage_slashing_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=deliverer,
        webhook_url=None,
    )
    event = HeartbeatMissingSlashedEvent(
        provider="0xPROV", challenger="0xCHAL",
        last_heartbeat_at=0, slash_id=b"\x77" * 32,
    )
    asyncio.run(watcher._invoke_cb(watcher._on_missing, event))
    assert not deliverer.deliver.called
