"""Phase 3.x.11.y Task 6 — ``RollbackCacheRequest`` server-side
handler tests.

Covers:
  - Server.handle dispatches RollbackCacheRequest →
    sharded_runner.rollback_cache → KVCacheManager.rollback;
    returns RollbackCacheResponse with rolled_back +
    actual_dropped.
  - Happy path: rollback by N positions, manager truncates +
    decrements tokens_generated.
  - Idempotent paths: zero-drop, drop-past-tokens-generated,
    unknown-request-id all return rolled_back=False (or
    rolled_back=True with actual_dropped <= n_positions).
  - Server without kv_cache_manager → INTERNAL_ERROR.
  - Server without sharded_runner → INTERNAL_ERROR.
  - Model missing truncate_cache → MALFORMED_REQUEST (caller
    bug, distinguished from internal crash).
  - Multi-stage broadcast: same RollbackCacheRequest hits each
    stage independently (each stage's manager truncates its
    own handle).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

from prsm.compute.chain_rpc.kv_cache import KVCacheManager
from prsm.compute.chain_rpc.protocol import (
    ChainRpcMalformedError,
    DecodeMode,
    RollbackCacheRequest,
    RollbackCacheResponse,
    StageError,
    StageErrorCode,
    encode_message,
    parse_message,
)
from prsm.compute.chain_rpc.server import LayerStageServer
from prsm.compute.inference.autoregressive_runner import SamplingDefaults
from prsm.compute.inference.sharded_runner import (
    ShardedAutoregressiveRunner,
)
from prsm.compute.tee.models import TEEType
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


class _MinimalRunner:
    def run_layer_range(self, **kwargs):  # noqa: ARG002
        raise NotImplementedError


class _MinimalRegistry:
    def get(self, model_id):  # noqa: ARG002
        return None


class _MinimalTEERuntime:
    tee_type = TEEType.SOFTWARE
    tee_attestation = b"\x00" * 32

    def get_attestation_bytes(self) -> bytes:
        return self.tee_attestation


class _FakeAnchor:
    def lookup(self, node_id):  # noqa: ARG002
        return None


class _TruncateCapableModel:
    """Model that exposes truncate_cache. Records every call. The
    payload is a List[str] of cached-position labels; truncation
    drops the last N entries."""

    def __init__(self) -> None:
        self.truncate_calls: List[Tuple[int, int]] = []

    def forward_prefill(self, **kwargs):  # noqa: ARG002
        raise NotImplementedError

    def forward_incremental(self, **kwargs):  # noqa: ARG002
        raise NotImplementedError

    def truncate_cache(self, payload: Any, n_positions: int) -> Any:
        self.truncate_calls.append((id(payload), n_positions))
        # Drop the last N entries from the list-payload.
        if isinstance(payload, list):
            if n_positions >= len(payload):
                payload.clear()
            else:
                del payload[-n_positions:]
            return payload
        return payload


class _NoTruncateModel:
    """Model that does NOT expose truncate_cache. Server should
    map MissingVerifyCapabilityError → MALFORMED_REQUEST."""

    def forward_prefill(self, **kwargs):  # noqa: ARG002
        raise NotImplementedError

    def forward_incremental(self, **kwargs):  # noqa: ARG002
        raise NotImplementedError


def _make_runner(
    *,
    model: Any,
    cache: KVCacheManager,
) -> ShardedAutoregressiveRunner:
    return ShardedAutoregressiveRunner(
        model=model,
        layer_range=(0, 4),
        kv_cache_manager=cache,
        tee_attestation=b"x" * 32,
        tee_type=TEEType.SOFTWARE,
    )


def _make_server(
    *,
    kv_cache_manager: Optional[KVCacheManager] = None,
    sharded_runner: Any = None,
) -> LayerStageServer:
    identity = generate_node_identity("server")
    return LayerStageServer(
        identity=identity,
        registry=_MinimalRegistry(),
        runner=_MinimalRunner(),
        tee_runtime=_MinimalTEERuntime(),
        anchor=_FakeAnchor(),
        kv_cache_manager=kv_cache_manager,
        sharded_runner=sharded_runner,
    )


def _seed_handle(
    cache: KVCacheManager,
    request_id: str,
    *,
    n_payload: int = 5,
    tokens_generated: int = 5,
) -> None:
    """Helper: allocate + populate a handle so rollback has
    something to drop."""
    handle = cache.allocate(request_id, n_layers=4)
    handle.payload = [f"P{i}" for i in range(n_payload)]
    handle.tokens_generated = tokens_generated


# ──────────────────────────────────────────────────────────────────────────
# Server handler
# ──────────────────────────────────────────────────────────────────────────


class TestRollbackCacheHandler:
    def test_happy_path_drops_n_positions(self):
        cache = KVCacheManager()
        model = _TruncateCapableModel()
        runner = _make_runner(model=model, cache=cache)
        _seed_handle(cache, "req-1", n_payload=5, tokens_generated=5)
        server = _make_server(
            kv_cache_manager=cache, sharded_runner=runner,
        )

        wire = encode_message(
            RollbackCacheRequest(
                request_id="req-1", n_positions_to_drop=2,
            )
        )
        parsed = parse_message(server.handle(wire))

        assert isinstance(parsed, RollbackCacheResponse)
        assert parsed.request_id == "req-1"
        assert parsed.rolled_back is True
        assert parsed.actual_dropped == 2
        # Manager actually truncated.
        h = cache.get("req-1")
        assert h.tokens_generated == 3
        assert h.payload == ["P0", "P1", "P2"]
        # Model.truncate_cache was called with n=2.
        assert len(model.truncate_calls) == 1
        assert model.truncate_calls[0][1] == 2

    def test_drop_past_tokens_generated_clamps(self):
        # Asking to drop more than tokens_generated drops everything
        # generated — actual_dropped == prior tokens_generated.
        cache = KVCacheManager()
        model = _TruncateCapableModel()
        runner = _make_runner(model=model, cache=cache)
        _seed_handle(cache, "req-1", n_payload=3, tokens_generated=3)
        server = _make_server(
            kv_cache_manager=cache, sharded_runner=runner,
        )

        wire = encode_message(
            RollbackCacheRequest(
                request_id="req-1", n_positions_to_drop=10,
            )
        )
        parsed = parse_message(server.handle(wire))
        assert isinstance(parsed, RollbackCacheResponse)
        assert parsed.rolled_back is True
        assert parsed.actual_dropped == 3
        h = cache.get("req-1")
        assert h.tokens_generated == 0
        assert h.payload == []

    def test_zero_drop_is_idempotent_no_op(self):
        # n_positions_to_drop=0 is a documented idempotent no-op
        # (mirrors the manager's contract).
        cache = KVCacheManager()
        model = _TruncateCapableModel()
        runner = _make_runner(model=model, cache=cache)
        _seed_handle(cache, "req-1", n_payload=4, tokens_generated=4)
        server = _make_server(
            kv_cache_manager=cache, sharded_runner=runner,
        )
        wire = encode_message(
            RollbackCacheRequest(
                request_id="req-1", n_positions_to_drop=0,
            )
        )
        parsed = parse_message(server.handle(wire))
        assert isinstance(parsed, RollbackCacheResponse)
        assert parsed.rolled_back is False
        assert parsed.actual_dropped == 0
        # Cache untouched.
        h = cache.get("req-1")
        assert h.tokens_generated == 4
        # truncate_cache not called.
        assert model.truncate_calls == []

    def test_unknown_request_id_returns_rolled_back_false(self):
        cache = KVCacheManager()
        model = _TruncateCapableModel()
        runner = _make_runner(model=model, cache=cache)
        server = _make_server(
            kv_cache_manager=cache, sharded_runner=runner,
        )
        wire = encode_message(
            RollbackCacheRequest(
                request_id="never-allocated", n_positions_to_drop=2,
            )
        )
        parsed = parse_message(server.handle(wire))
        assert isinstance(parsed, RollbackCacheResponse)
        assert parsed.rolled_back is False
        assert parsed.actual_dropped == 0
        # truncate_cache not called.
        assert model.truncate_calls == []

    def test_handler_without_manager_rejects(self):
        # No kv_cache_manager wired → INTERNAL_ERROR.
        server = _make_server(kv_cache_manager=None, sharded_runner=None)
        wire = encode_message(
            RollbackCacheRequest(
                request_id="req-1", n_positions_to_drop=1,
            )
        )
        parsed = parse_message(server.handle(wire))
        assert isinstance(parsed, StageError)
        assert parsed.code == StageErrorCode.INTERNAL_ERROR.value
        assert "kv_cache_manager" in parsed.message

    def test_handler_without_runner_rejects(self):
        # kv_cache_manager wired but no sharded_runner → can't
        # provide truncate_cache delegate.
        cache = KVCacheManager()
        server = _make_server(
            kv_cache_manager=cache, sharded_runner=None,
        )
        wire = encode_message(
            RollbackCacheRequest(
                request_id="req-1", n_positions_to_drop=1,
            )
        )
        parsed = parse_message(server.handle(wire))
        assert isinstance(parsed, StageError)
        assert parsed.code == StageErrorCode.INTERNAL_ERROR.value
        assert "sharded_runner" in parsed.message

    def test_handler_with_no_truncate_model_returns_malformed(self):
        # Runner wired with a model that doesn't expose
        # truncate_cache → MALFORMED_REQUEST (caller bug, not
        # internal crash).
        cache = KVCacheManager()
        model = _NoTruncateModel()
        runner = _make_runner(model=model, cache=cache)
        _seed_handle(cache, "req-1", n_payload=3, tokens_generated=3)
        server = _make_server(
            kv_cache_manager=cache, sharded_runner=runner,
        )
        wire = encode_message(
            RollbackCacheRequest(
                request_id="req-1", n_positions_to_drop=1,
            )
        )
        parsed = parse_message(server.handle(wire))
        assert isinstance(parsed, StageError)
        assert parsed.code == StageErrorCode.MALFORMED_REQUEST.value
        assert "truncate_cache" in parsed.message

    def test_handler_propagates_request_id(self):
        cache = KVCacheManager()
        model = _TruncateCapableModel()
        runner = _make_runner(model=model, cache=cache)
        _seed_handle(cache, "custom-uuid-a1", n_payload=2, tokens_generated=2)
        server = _make_server(
            kv_cache_manager=cache, sharded_runner=runner,
        )
        wire = encode_message(
            RollbackCacheRequest(
                request_id="custom-uuid-a1", n_positions_to_drop=1,
            )
        )
        parsed = parse_message(server.handle(wire))
        assert isinstance(parsed, RollbackCacheResponse)
        assert parsed.request_id == "custom-uuid-a1"

    def test_handler_does_not_leak_other_requests(self):
        cache = KVCacheManager()
        model = _TruncateCapableModel()
        runner = _make_runner(model=model, cache=cache)
        _seed_handle(cache, "req-1", n_payload=3, tokens_generated=3)
        _seed_handle(cache, "req-2", n_payload=3, tokens_generated=3)
        server = _make_server(
            kv_cache_manager=cache, sharded_runner=runner,
        )
        wire = encode_message(
            RollbackCacheRequest(
                request_id="req-1", n_positions_to_drop=2,
            )
        )
        parsed = parse_message(server.handle(wire))
        assert isinstance(parsed, RollbackCacheResponse)
        assert parsed.actual_dropped == 2
        # req-1 truncated.
        assert cache.get("req-1").tokens_generated == 1
        # req-2 untouched.
        assert cache.get("req-2").tokens_generated == 3
        assert cache.get("req-2").payload == ["P0", "P1", "P2"]


# ──────────────────────────────────────────────────────────────────────────
# Multi-stage broadcast simulation
# ──────────────────────────────────────────────────────────────────────────


class TestMultiStageBroadcast:
    def test_same_request_hits_each_stage_independently(self):
        # Simulate the executor's broadcast: encode ONE
        # RollbackCacheRequest, send to TWO independent servers
        # (each with its own cache + runner). Both should truncate
        # their local handle by the same N.
        cache_a = KVCacheManager()
        cache_b = KVCacheManager()
        model_a = _TruncateCapableModel()
        model_b = _TruncateCapableModel()
        runner_a = _make_runner(model=model_a, cache=cache_a)
        runner_b = _make_runner(model=model_b, cache=cache_b)
        _seed_handle(cache_a, "req-1", n_payload=4, tokens_generated=4)
        _seed_handle(cache_b, "req-1", n_payload=4, tokens_generated=4)
        server_a = _make_server(
            kv_cache_manager=cache_a, sharded_runner=runner_a,
        )
        server_b = _make_server(
            kv_cache_manager=cache_b, sharded_runner=runner_b,
        )

        # Single broadcast envelope.
        wire = encode_message(
            RollbackCacheRequest(
                request_id="req-1", n_positions_to_drop=2,
            )
        )
        # Each server processes the same bytes.
        ack_a = parse_message(server_a.handle(wire))
        ack_b = parse_message(server_b.handle(wire))

        assert isinstance(ack_a, RollbackCacheResponse)
        assert isinstance(ack_b, RollbackCacheResponse)
        assert ack_a.actual_dropped == ack_b.actual_dropped == 2
        # Both caches truncated by 2.
        assert cache_a.get("req-1").tokens_generated == 2
        assert cache_b.get("req-1").tokens_generated == 2
        # Each model.truncate_cache called exactly once.
        assert len(model_a.truncate_calls) == 1
        assert len(model_b.truncate_calls) == 1
