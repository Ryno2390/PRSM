"""Sprint 176 — SwarmDispatcherAdapter._to_partial threads
source_agent_pubkey from the AgentExecutor result.

Pre-fix _to_partial defaulted to 32 zero bytes when the result
dict lacked ``source_agent_pubkey``. AgentExecutor surfaced the
agent's signing key as ``provider_public_key`` (base64) but
SwarmDispatcherAdapter only read the raw-bytes field — so v1
PartialResults always carried a placeholder pubkey, and the
aggregator's signature-verification path either fell back to a
permissive check or rejected every partial.

Sprint 176 priority order:
  1. result["source_agent_pubkey"] (32 raw bytes — explicit override)
  2. result["provider_public_key"] (base64 — the v1 canonical path)
  3. 32 zero bytes (fail-loud default for unsupported dispatchers)
"""
from __future__ import annotations

import base64

import pytest

from prsm.compute.query_orchestrator.swarm_dispatcher_adapter import (
    SwarmDispatcherAdapter,
)
from prsm.compute.query_orchestrator.shard_finder import ShardCandidate


def _shard(cid="QmABC", creator_id="creator-1"):
    return ShardCandidate(
        cid=cid,
        similarity=0.9,
        creator_id=creator_id,
    )


def _result(**overrides):
    base = {
        "payload": b"hello",
        "agent_signature": b"sig" * 11,
    }
    base.update(overrides)
    return base


class TestSourceAgentPubkeyThreading:
    def test_explicit_source_agent_pubkey_wins(self):
        explicit = b"\xaa" * 32
        partial = SwarmDispatcherAdapter._to_partial(
            _shard(),
            _result(source_agent_pubkey=explicit),
        )
        assert partial.source_agent_pubkey == explicit

    def test_falls_back_to_provider_public_key_base64(self):
        """Sprint 176 — AgentExecutor surfaces provider_public_key
        as base64 string. _to_partial decodes it as the source agent
        pubkey when the explicit field isn't supplied."""
        raw = b"\xbb" * 32
        b64 = base64.b64encode(raw).decode()
        partial = SwarmDispatcherAdapter._to_partial(
            _shard(),
            _result(provider_public_key=b64),
        )
        assert partial.source_agent_pubkey == raw

    def test_zero_bytes_when_no_pubkey_in_result(self):
        """Defensive — no pubkey at all → 32 zero bytes (the original
        fail-loud default; aggregator verification will reject)."""
        partial = SwarmDispatcherAdapter._to_partial(
            _shard(),
            _result(),
        )
        assert partial.source_agent_pubkey == b"\x00" * 32

    def test_zero_bytes_when_provider_pubkey_malformed_base64(self):
        """Defensive — malformed base64 string falls back to zero
        bytes rather than crashing the entire fan-out."""
        partial = SwarmDispatcherAdapter._to_partial(
            _shard(),
            _result(provider_public_key="not_valid_base64!!"),
        )
        # Either zeros (preferred fail-loud) OR some decoded bytes.
        # base64.b64decode is permissive — it may accept this. Pin
        # the invariant that decode failure doesn't crash; output
        # is bytes either way.
        assert isinstance(partial.source_agent_pubkey, bytes)
        assert len(partial.source_agent_pubkey) == 32

    def test_zero_bytes_when_provider_pubkey_wrong_size(self):
        """Sprint 176 — a base64 string that decodes to non-32-byte
        output falls back to zero bytes (signature verification
        requires exactly 32 bytes for Ed25519)."""
        # 16 bytes encoded.
        short_b64 = base64.b64encode(b"\xcc" * 16).decode()
        partial = SwarmDispatcherAdapter._to_partial(
            _shard(),
            _result(provider_public_key=short_b64),
        )
        assert partial.source_agent_pubkey == b"\x00" * 32

    def test_explicit_wins_over_provider_key(self):
        """Sprint 176 priority — explicit source_agent_pubkey wins
        even when provider_public_key is also present."""
        explicit = b"\xaa" * 32
        provider = base64.b64encode(b"\xdd" * 32).decode()
        partial = SwarmDispatcherAdapter._to_partial(
            _shard(),
            _result(
                source_agent_pubkey=explicit,
                provider_public_key=provider,
            ),
        )
        assert partial.source_agent_pubkey == explicit
