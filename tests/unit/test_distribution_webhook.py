"""distribution.distributed webhook delivery — when
CompensationDistributorWatcher observes a Distributed event,
fire webhook so operators get paged on emission round
landings.

Symmetric to slash.proof_failure_slashed /
slash.heartbeat_missing_slashed (sprint 85).
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.compensation_distributor import DistributedEvent
from prsm.node.node import (
    _build_compensation_distributor_watcher_or_none,
)


@pytest.fixture
def opted_in_env():
    with patch.dict(os.environ, {
        "PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED": "1",
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
    return DistributedEvent(
        to_creator=100, to_operator=50, to_grant=20,
    )


def test_dispatches_webhook_on_distributed(opted_in_env, deliverer):
    watcher = _build_compensation_distributor_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=deliverer,
        webhook_url="https://hook.example.com",
        webhook_secret="s3cret",
    )
    asyncio.run(watcher._invoke_cb(_event()))

    assert deliverer.deliver.called
    args = deliverer.last_kwargs
    assert args["url"] == "https://hook.example.com"
    assert args["event"] == "distribution.distributed"
    assert args["secret"] == "s3cret"
    assert args["payload"]["to_creator"] == 100
    assert args["payload"]["to_operator"] == 50
    assert args["payload"]["to_grant"] == 20


def test_no_deliverer_does_not_crash(opted_in_env):
    watcher = _build_compensation_distributor_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=None,
        webhook_url=None,
    )
    asyncio.run(watcher._invoke_cb(_event()))


def test_deliverer_failure_isolated(opted_in_env):
    bad = MagicMock()
    async def boom(**kwargs):
        raise RuntimeError("simulated")
    bad.deliver = MagicMock(side_effect=boom)
    watcher = _build_compensation_distributor_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=bad,
        webhook_url="https://hook.example.com",
    )
    # Must NOT raise
    asyncio.run(watcher._invoke_cb(_event()))


def test_deliverer_without_url_skips(opted_in_env, deliverer):
    watcher = _build_compensation_distributor_watcher_or_none(
        client=MagicMock(),
        webhook_deliverer=deliverer,
        webhook_url=None,
    )
    asyncio.run(watcher._invoke_cb(_event()))
    assert not deliverer.deliver.called
