"""Phase 3.x.11 Task 6 — ``EvictCacheRequest`` wire message + handler.

Covers:
  - EvictCacheRequest / EvictCacheResponse round-trip
  - Validation (request_id non-empty; evicted must be bool)
  - LayerStageServer.handle dispatches EvictCacheRequest →
    KVCacheManager.evict; returns EvictCacheResponse
  - evicted=True when handle exists; evicted=False on idempotent
    re-evict; evicted=False on never-allocated
  - Server without kv_cache_manager rejects with INTERNAL_ERROR
  - Round-trip: Executor encodes EvictCacheRequest →
    server.handle parses → manager.evict → server encodes
    EvictCacheResponse → executor parses ack
"""

from __future__ import annotations

from typing import Any, List, Optional

import pytest

from prsm.compute.chain_rpc.kv_cache import KVCacheManager
from prsm.compute.chain_rpc.protocol import (
    ChainRpcMalformedError,
    EvictCacheRequest,
    EvictCacheResponse,
    StageError,
    StageErrorCode,
    encode_message,
    parse_message,
)
from prsm.compute.chain_rpc.server import LayerStageServer
from prsm.compute.tee.models import TEEType
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Wire round-trip + validation
# ──────────────────────────────────────────────────────────────────────────


class TestEvictCacheRequestRoundTrip:
    def test_round_trip(self):
        req = EvictCacheRequest(request_id="abc-123")
        wire = encode_message(req)
        parsed = parse_message(wire)
        assert isinstance(parsed, EvictCacheRequest)
        assert parsed.request_id == "abc-123"

    def test_rejects_empty_request_id(self):
        with pytest.raises(ChainRpcMalformedError, match="request_id"):
            EvictCacheRequest(request_id="")

    def test_rejects_non_str_request_id(self):
        with pytest.raises(ChainRpcMalformedError, match="request_id"):
            EvictCacheRequest(request_id=123)  # type: ignore[arg-type]


class TestEvictCacheResponseRoundTrip:
    def test_round_trip_evicted_true(self):
        resp = EvictCacheResponse(request_id="abc", evicted=True)
        wire = encode_message(resp)
        parsed = parse_message(wire)
        assert isinstance(parsed, EvictCacheResponse)
        assert parsed.request_id == "abc"
        assert parsed.evicted is True

    def test_round_trip_evicted_false(self):
        resp = EvictCacheResponse(request_id="abc", evicted=False)
        wire = encode_message(resp)
        parsed = parse_message(wire)
        assert parsed.evicted is False

    def test_rejects_non_bool_evicted(self):
        with pytest.raises(ChainRpcMalformedError, match="evicted"):
            EvictCacheResponse(
                request_id="abc", evicted="yes",  # type: ignore[arg-type]
            )

    def test_from_dict_rejects_int_for_bool_evicted(self):
        # Wire-level: int 1/0 must NOT slip through as bool —
        # the dataclass __post_init__ catches at construction
        # time, so the parse path also rejects.
        bad_wire = encode_message(
            EvictCacheResponse(request_id="x", evicted=True),
        )
        # Manually poison the JSON to set evicted=1 (int).
        import json
        d = json.loads(bad_wire)
        d["evicted"] = 1
        poisoned = json.dumps(d).encode("utf-8")
        with pytest.raises(ChainRpcMalformedError, match="evicted"):
            parse_message(poisoned)


# ──────────────────────────────────────────────────────────────────────────
# Server handler
# ──────────────────────────────────────────────────────────────────────────


class _MinimalRunner:
    """Minimal LayerSliceRunner stub — never actually called by
    EvictCacheRequest handling; needed only because the
    LayerStageServer constructor validates ``runner`` shape."""

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


def _make_server(
    *,
    kv_cache_manager: Optional[KVCacheManager] = None,
) -> LayerStageServer:
    identity = generate_node_identity("server")
    return LayerStageServer(
        identity=identity,
        registry=_MinimalRegistry(),
        runner=_MinimalRunner(),
        tee_runtime=_MinimalTEERuntime(),
        anchor=_FakeAnchor(),
        kv_cache_manager=kv_cache_manager,
    )


class TestServerEvictHandler:
    def test_handler_evicts_existing_cache(self):
        manager = KVCacheManager()
        manager.allocate("req-1", n_layers=4)
        assert "req-1" in manager
        server = _make_server(kv_cache_manager=manager)

        wire = encode_message(EvictCacheRequest(request_id="req-1"))
        response_bytes = server.handle(wire)
        parsed = parse_message(response_bytes)

        assert isinstance(parsed, EvictCacheResponse)
        assert parsed.request_id == "req-1"
        assert parsed.evicted is True
        assert "req-1" not in manager

    def test_handler_idempotent_returns_evicted_false(self):
        # Re-evict after a successful evict — returns evicted=False
        # without raising. Mirrors the manager's
        # idempotent-evict contract.
        manager = KVCacheManager()
        manager.allocate("req-1", n_layers=4)
        server = _make_server(kv_cache_manager=manager)

        wire = encode_message(EvictCacheRequest(request_id="req-1"))
        first = parse_message(server.handle(wire))
        assert isinstance(first, EvictCacheResponse)
        assert first.evicted is True
        # Same wire bytes again.
        second = parse_message(server.handle(wire))
        assert isinstance(second, EvictCacheResponse)
        assert second.evicted is False

    def test_handler_unknown_request_id_returns_evicted_false(self):
        manager = KVCacheManager()
        server = _make_server(kv_cache_manager=manager)
        wire = encode_message(
            EvictCacheRequest(request_id="never-allocated"),
        )
        parsed = parse_message(server.handle(wire))
        assert isinstance(parsed, EvictCacheResponse)
        assert parsed.evicted is False

    def test_handler_without_manager_rejects_with_internal_error(self):
        # Server constructed without kv_cache_manager — eviction
        # request rejects with structured StageError.
        server = _make_server(kv_cache_manager=None)
        wire = encode_message(EvictCacheRequest(request_id="req-1"))
        parsed = parse_message(server.handle(wire))
        assert isinstance(parsed, StageError)
        assert parsed.code == StageErrorCode.INTERNAL_ERROR.value
        assert "kv_cache_manager" in parsed.message

    def test_handler_propagates_request_id_in_response(self):
        # Server populates the response.request_id with the
        # incoming request's id (not the server's own id).
        manager = KVCacheManager()
        server = _make_server(kv_cache_manager=manager)
        wire = encode_message(
            EvictCacheRequest(request_id="custom-uuid-x7"),
        )
        parsed = parse_message(server.handle(wire))
        assert isinstance(parsed, EvictCacheResponse)
        assert parsed.request_id == "custom-uuid-x7"

    def test_handler_does_not_leak_other_requests(self):
        # Evicting req-1 must NOT touch req-2's handle.
        manager = KVCacheManager()
        manager.allocate("req-1", n_layers=4)
        manager.allocate("req-2", n_layers=4)
        server = _make_server(kv_cache_manager=manager)
        wire = encode_message(EvictCacheRequest(request_id="req-1"))
        parsed = parse_message(server.handle(wire))
        assert isinstance(parsed, EvictCacheResponse)
        assert parsed.evicted is True
        assert "req-1" not in manager
        # req-2 still in pool.
        assert "req-2" in manager


class TestServerConstructorValidation:
    def test_rejects_non_kv_cache_manager(self):
        with pytest.raises(RuntimeError, match="kv_cache_manager"):
            LayerStageServer(
                identity=generate_node_identity("server"),
                registry=_MinimalRegistry(),
                runner=_MinimalRunner(),
                tee_runtime=_MinimalTEERuntime(),
                anchor=_FakeAnchor(),
                kv_cache_manager="not a manager",  # type: ignore[arg-type]
            )

    def test_default_kv_cache_manager_is_none(self):
        # Existing servers (Phase 3.x.7+ non-sharded) construct
        # without kv_cache_manager — back-compat.
        server = _make_server()
        assert server is not None
