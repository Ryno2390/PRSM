"""Libp2pGossip subscribe-before-start handling (sprint 122).

Pre-fix: subscribe() called transport._lib.PrsmSubscribe with
handle=-1 (libp2p host not yet started). C bridge logged
`PrsmSubscribe: handle -1 not found` and returned failure, but
Python-side _subscribed_topics added the topic anyway —
permanently desyncing the GossipSub layer.

Post-fix: subscribe() defers to _pending_topics when handle<0;
start() flushes pending subscriptions once the host is up.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.node.libp2p_gossip import Libp2pGossip


def _gossip(handle=-1):
    transport = MagicMock()
    transport._handle = handle
    transport._lib = MagicMock()
    transport.on_message = MagicMock()
    return Libp2pGossip(transport=transport, identity=MagicMock())


class TestPreStartSubscribe:
    def test_subscribe_with_handle_negative_defers(self):
        g = _gossip(handle=-1)
        g.subscribe("test_topic", lambda *a, **kw: None)
        # Topic queued, NOT marked subscribed
        assert "test_topic" not in g._subscribed_topics
        topic_full = g._topic_name("test_topic")
        assert topic_full in g._pending_topics
        # PrsmSubscribe NOT called — guard prevented C-bridge noise
        g.transport._lib.PrsmSubscribe.assert_not_called()
        # Callback IS registered (will fire once subscribed)
        assert "test_topic" in g._callbacks

    def test_subscribe_with_handle_ready_subscribes_immediately(self):
        g = _gossip(handle=42)  # handle >= 0
        g.subscribe("test_topic", lambda *a, **kw: None)
        topic_full = g._topic_name("test_topic")
        assert topic_full in g._subscribed_topics
        assert topic_full not in g._pending_topics
        g.transport._lib.PrsmSubscribe.assert_called_once()


class TestPendingFlush:
    @pytest.mark.asyncio
    async def test_start_flushes_pending(self):
        g = _gossip(handle=-1)
        g.subscribe("a", lambda *a, **kw: None)
        g.subscribe("b", lambda *a, **kw: None)
        assert len(g._pending_topics) == 2
        # Flip handle to ready, then start()
        g.transport._handle = 42
        await g.start()
        # Both topics now subscribed, pending drained
        assert len(g._pending_topics) == 0
        assert len(g._subscribed_topics) == 2
        # PrsmSubscribe called exactly twice (once per pending)
        assert g.transport._lib.PrsmSubscribe.call_count == 2

    @pytest.mark.asyncio
    async def test_start_handles_subscribe_failure(self):
        g = _gossip(handle=-1)
        g.subscribe("a", lambda *a, **kw: None)
        g.transport._handle = 42
        g.transport._lib.PrsmSubscribe.side_effect = RuntimeError(
            "C bridge failed",
        )
        # Should NOT raise — log + continue
        await g.start()
        # Topic stays out of subscribed set since C call failed
        assert g._topic_name("a") not in g._subscribed_topics
        # Pending was drained (we don't retry on failure within
        # start() flush)
        assert g._topic_name("a") not in g._pending_topics


class TestSubscribeFailureRetry:
    def test_failed_subscribe_retries_next_call(self):
        """When PrsmSubscribe raises, the topic should NOT be
        marked subscribed — leaving room for the next subscribe()
        call to retry. Pre-fix bug: failed subs were marked
        subscribed, locking out forever."""
        g = _gossip(handle=42)
        g.transport._lib.PrsmSubscribe.side_effect = RuntimeError("boom")
        g.subscribe("a", lambda *a, **kw: None)
        # Topic NOT in subscribed set
        assert g._topic_name("a") not in g._subscribed_topics

        # Recover and retry
        g.transport._lib.PrsmSubscribe.side_effect = None
        g.subscribe("a", lambda *a, **kw: None)
        # Now subscribed
        assert g._topic_name("a") in g._subscribed_topics
        # Called twice total (initial + retry)
        assert g.transport._lib.PrsmSubscribe.call_count == 2
