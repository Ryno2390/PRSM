"""Sprint 837 — PublisherKeyAnchor.lookup() retry with backoff.

Multi-host re-attest 2026-05-24 (sp836 follow-on) surfaced
that a single transient Base RPC error from the free
mainnet.base.org endpoint bricked chain inference dispatch
with `lookup(<peer>) RPC failed`. Sprint 836 fixed the pool
admission gap; sprint 837 hardens the next step (anchor pubkey
lookup before stake verification) against transient RPC blips.

Sprint 837 adds:
- PRSM_ANCHOR_LOOKUP_RETRY_ATTEMPTS (default 3): total
  attempt count (1 initial + 2 retries).
- PRSM_ANCHOR_LOOKUP_RETRY_BACKOFF_S (default 0.5): exp
  progression — 0.5s, 1.0s, 2.0s.
- ValueError NOT retried (deterministic input error).
- Successful lookups still cache normally (TTL unchanged).
- Cached lookups bypass the retry path entirely.

Pin tests:
- Single transient failure → succeeds on retry
- Two transient failures → succeeds on third attempt
- All attempts fail → AnchorRPCError mentions attempt count
- ValueError on first attempt → not retried
- Successful first attempt → no sleeping (regression guard)
- Env vars adjust retry count + backoff
- Successful lookup result cached (regression — sp837 didn't
  break sp273's positive-cache contract)
- Env vars registered in sp696 readiness CLI
"""
from __future__ import annotations

import os
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env():
    """Snapshot+restore the sp837 env vars between tests."""
    keys = (
        "PRSM_ANCHOR_LOOKUP_RETRY_ATTEMPTS",
        "PRSM_ANCHOR_LOOKUP_RETRY_BACKOFF_S",
    )
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k in keys:
        os.environ.pop(k, None)
        if saved[k] is not None:
            os.environ[k] = saved[k]


def _make_client(**env_overrides):
    """Build a PublisherKeyAnchorClient with a mock contract.
    env_overrides set via os.environ BEFORE construction (the
    client reads env once at init time)."""
    for k, v in env_overrides.items():
        os.environ[k] = v
    from prsm.security.publisher_key_anchor.client import (
        PublisherKeyAnchorClient,
    )
    contract = MagicMock()
    return PublisherKeyAnchorClient(
        contract_address="0x" + "a" * 40,
        contract=contract,
    )


# 32-hex-char node_id (the test target)
_NID = "0123456789abcdef" * 2


# ---- Retry behavior ------------------------------------------


def test_single_transient_failure_succeeds_on_retry():
    """One RPC blip then success → caller sees the result."""
    client = _make_client()
    # 32-byte pubkey
    pubkey_bytes = b"\x01" * 32
    side = [RuntimeError("rate limit"), pubkey_bytes]
    with patch.object(client, "_call_lookup", side_effect=side), \
         patch("time.sleep") as sleep_mock:
        result = client.lookup(_NID)
    assert result is not None
    # Slept once between attempts
    assert sleep_mock.call_count == 1


def test_two_transient_failures_succeed_on_third():
    """Default attempts=3 → 2 retries possible."""
    client = _make_client()
    side = [
        RuntimeError("blip1"),
        RuntimeError("blip2"),
        b"\x02" * 32,
    ]
    with patch.object(client, "_call_lookup", side_effect=side), \
         patch("time.sleep") as sleep_mock:
        result = client.lookup(_NID)
    assert result is not None
    assert sleep_mock.call_count == 2


def test_all_attempts_fail_raises_with_count():
    """When all retries exhausted, AnchorRPCError mentions
    attempt count so operators can triage."""
    from prsm.security.publisher_key_anchor.client import (
        AnchorRPCError,
    )
    client = _make_client()
    side = [
        RuntimeError("blip1"),
        RuntimeError("blip2"),
        RuntimeError("blip3"),
    ]
    with patch.object(client, "_call_lookup", side_effect=side), \
         patch("time.sleep"):
        with pytest.raises(AnchorRPCError) as exc_info:
            client.lookup(_NID)
    msg = str(exc_info.value)
    assert "3 attempt" in msg
    assert "blip3" in msg


def test_value_error_not_retried():
    """ValueError is a deterministic input error — must
    propagate immediately without retry."""
    client = _make_client()
    side = [ValueError("malformed"), b"\x03" * 32]
    with patch.object(client, "_call_lookup", side_effect=side), \
         patch("time.sleep") as sleep_mock:
        with pytest.raises(ValueError):
            client.lookup(_NID)
    sleep_mock.assert_not_called()


def test_successful_first_attempt_no_sleep():
    """Regression guard: a happy-path call MUST NOT introduce
    latency from the sp837 sleep helper."""
    client = _make_client()
    with patch.object(
        client, "_call_lookup", return_value=b"\x04" * 32,
    ), patch("time.sleep") as sleep_mock:
        client.lookup(_NID)
    sleep_mock.assert_not_called()


# ---- Env-tuned retry config ----------------------------------


def test_env_attempts_override():
    """ATTEMPTS=5 → 4 retries possible."""
    client = _make_client(
        PRSM_ANCHOR_LOOKUP_RETRY_ATTEMPTS="5",
    )
    side = [RuntimeError("x")] * 4 + [b"\x05" * 32]
    with patch.object(client, "_call_lookup", side_effect=side), \
         patch("time.sleep") as sleep_mock:
        client.lookup(_NID)
    assert sleep_mock.call_count == 4


def test_env_attempts_clamped_to_minimum_1():
    """ATTEMPTS=0 (or negative) → clamped to 1 (one attempt,
    no retries)."""
    client = _make_client(
        PRSM_ANCHOR_LOOKUP_RETRY_ATTEMPTS="0",
    )
    side = [RuntimeError("x")]
    with patch.object(client, "_call_lookup", side_effect=side), \
         patch("time.sleep") as sleep_mock:
        from prsm.security.publisher_key_anchor.client import (
            AnchorRPCError,
        )
        with pytest.raises(AnchorRPCError):
            client.lookup(_NID)
    # One attempt → no sleep between attempts
    sleep_mock.assert_not_called()


def test_env_attempts_malformed_defaults_to_3():
    """Non-int ATTEMPTS env → falls back to 3."""
    client = _make_client(
        PRSM_ANCHOR_LOOKUP_RETRY_ATTEMPTS="not-an-int",
    )
    side = [RuntimeError("x")] * 2 + [b"\x06" * 32]
    with patch.object(client, "_call_lookup", side_effect=side), \
         patch("time.sleep") as sleep_mock:
        client.lookup(_NID)
    assert sleep_mock.call_count == 2


def test_env_backoff_progression():
    """Backoff = base * 2**attempt: with base=0.1 → 0.1, 0.2."""
    client = _make_client(
        PRSM_ANCHOR_LOOKUP_RETRY_BACKOFF_S="0.1",
        PRSM_ANCHOR_LOOKUP_RETRY_ATTEMPTS="3",
    )
    side = [
        RuntimeError("x"),
        RuntimeError("y"),
        b"\x07" * 32,
    ]
    with patch.object(client, "_call_lookup", side_effect=side), \
         patch("time.sleep") as sleep_mock:
        client.lookup(_NID)
    delays = [c.args[0] for c in sleep_mock.call_args_list]
    assert delays == [0.1, 0.2]


# ---- Positive-cache contract still holds --------------------


def test_successful_lookup_caches_result():
    """sp273's positive-cache must keep working — second lookup
    of same node_id MUST NOT call _call_lookup again."""
    client = _make_client()
    call_count = {"n": 0}

    def _impl(_):
        call_count["n"] += 1
        return b"\x08" * 32

    with patch.object(client, "_call_lookup", side_effect=_impl):
        first = client.lookup(_NID)
        second = client.lookup(_NID)
    assert first == second
    assert call_count["n"] == 1


# ---- Env vars surfaced in readiness CLI ---------------------


def test_retry_env_vars_in_readiness_registry():
    from prsm.cli import _PARALLAX_ENV_REGISTRY
    names = [t[0] for t in _PARALLAX_ENV_REGISTRY]
    assert "PRSM_ANCHOR_LOOKUP_RETRY_ATTEMPTS" in names
    assert "PRSM_ANCHOR_LOOKUP_RETRY_BACKOFF_S" in names
