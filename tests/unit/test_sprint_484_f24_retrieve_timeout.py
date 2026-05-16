"""Sprint 484 — F24 ContentProvider retrieve timeout-propagation pin.

F24 surfaced during the sprint 484 semi-fresh dogfood pass.
Sprint 480's F22 fix made `ProvenanceQueries.load_all_for_node`
hydrate `_local_content` correctly from the DB after restart.
That exposed a latent bug: BT-publisher's `_published_paths`
is per-session, so `local_publish_path()` returns None for
hydrated CIDs even though `_local_content` says we have them.

The retriever then fell through to
`bt_requester.request_content(timeout=None)` — and because
`_fetch_local_via_bt` didn't propagate the caller's timeout,
the BT swarm wait was UNBOUNDED. With 0 peers (single-node
dev), `/content/retrieve/{cid}` would hang forever for any
hydrated CID.

Operator-visible symptom: `curl /content/retrieve/{hydrated_cid}`
returns no response — not even the 30s HTTP timeout fires
because aiohttp keeps the connection open.

Sprint 484 fix: propagate `timeout` through `_fetch_local` →
`_fetch_local_via_bt` → `retriever.fetch(timeout=...)`. Plus
defense-in-depth via `asyncio.wait_for` so even a buggy
fetch implementation can't hang beyond the caller's timeout.

Live-verified pre-fix: 30s+ hang. Post-fix: 2s response with
clean `not_found` envelope.

These pins defend the timeout-propagation contract.
"""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path


def test_fetch_local_accepts_timeout_kwarg():
    """`_fetch_local` must accept a timeout parameter so the
    caller (`request_content`) can bound the BT-fallback
    path. Pre-fix signature: `(self, content_id)` — no
    timeout."""
    from prsm.node.content_provider import ContentProvider
    sig = inspect.signature(ContentProvider._fetch_local)
    assert "timeout" in sig.parameters, (
        "_fetch_local must accept timeout kwarg — F24 fix"
    )


def test_fetch_local_via_bt_accepts_timeout_kwarg():
    """Same invariant on the BT-fallback layer."""
    from prsm.node.content_provider import ContentProvider
    sig = inspect.signature(
        ContentProvider._fetch_local_via_bt,
    )
    assert "timeout" in sig.parameters, (
        "_fetch_local_via_bt must accept timeout kwarg — "
        "F24 fix"
    )


def test_request_content_propagates_timeout_to_fetch_local():
    """The integration point: `request_content(cid, timeout=T)`
    must pass T into `_fetch_local`. Pin the source so a
    refactor can't silently drop the propagation."""
    src = (
        Path(__file__).resolve().parents[2]
        / "prsm" / "node" / "content_provider.py"
    ).read_text()
    # Find the request_content body where _fetch_local is
    # called.
    idx = src.find("async def request_content")
    assert idx >= 0
    body = src[idx:idx + 4000]
    # The call MUST pass timeout — pre-F24 it was
    # `self._fetch_local(cid)` only.
    assert "_fetch_local(cid, timeout=timeout)" in body, (
        "request_content must propagate timeout into "
        "_fetch_local — F24 regression risk"
    )


def test_fetch_local_via_bt_uses_asyncio_wait_for():
    """Defense-in-depth: `_fetch_local_via_bt` wraps the
    `retriever.fetch(...)` call in `asyncio.wait_for(...,
    timeout=...)` when timeout is provided. This guarantees
    bounded latency even if a future retriever implementation
    silently ignores its own timeout parameter."""
    src = (
        Path(__file__).resolve().parents[2]
        / "prsm" / "node" / "content_provider.py"
    ).read_text()
    idx = src.find("async def _fetch_local_via_bt")
    assert idx >= 0
    body = src[idx:idx + 3000]
    assert "asyncio.wait_for" in body, (
        "_fetch_local_via_bt must use asyncio.wait_for as "
        "defense-in-depth — F24 regression risk"
    )


def test_timeout_propagation_under_simulated_hang():
    """Integration: a retriever that NEVER returns must
    raise asyncio.TimeoutError within `timeout` seconds —
    not hang forever. Live-verifies the asyncio.wait_for
    defense-in-depth.

    Implementation note: uses asyncio.Event.wait() instead
    of asyncio.sleep(60) because some test harnesses
    fast-forward sleep. Event.wait is a true block."""
    from prsm.node.content_provider import ContentProvider

    class HangingRetriever:
        async def fetch(self, cid, *, timeout=None):
            # Block until an event that will never fire.
            ev = asyncio.Event()
            await ev.wait()
            return b"never returned"

    p = ContentProvider.__new__(ContentProvider)
    p._local_content = {"hangcid": {}}
    p.content_retriever = HangingRetriever()

    async def run():
        loop = asyncio.get_event_loop()
        start = loop.time()
        result = await p._fetch_local_via_bt(
            "hangcid", timeout=0.5,
        )
        elapsed = loop.time() - start
        return result, elapsed

    result, elapsed = asyncio.run(run())
    assert result is None, (
        f"hanging retriever should return None on timeout; "
        f"got {result!r}"
    )
    assert elapsed < 5.0, (
        f"asyncio.wait_for didn't enforce 0.5s timeout — "
        f"took {elapsed:.2f}s"
    )
