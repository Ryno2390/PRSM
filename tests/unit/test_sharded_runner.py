"""Phase 3.x.11 Task 3 — unit tests for ``ShardedAutoregressiveRunner``
(non-tail variant).

Covers:
  - Constructor validation (model shape, layer_range, manager,
    tee_attestation, tee_type)
  - PREFILL: allocates cache, stores payload on handle, returns
    boundary hidden state, decode_mode field correct
  - INCREMENTAL: uses existing cache, updates payload, model gets
    the prior-call payload as kv_cache_payload kwarg
  - INCREMENTAL without prior PREFILL → MalformedCacheStateError
  - layer_range respected (model receives the runner's range)
  - Cache survives across multiple INCREMENTAL calls (10 iterations)
  - Explicit eviction returns True/False per existence
  - Non-tail v1 rejects ``is_final_stage=True`` (Task 4 deferral)
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pytest

from prsm.compute.chain_rpc.kv_cache import KVCacheManager
from prsm.compute.chain_rpc.protocol import DecodeMode
from prsm.compute.inference.sharded_runner import (
    LayerSliceResult,
    MalformedCacheStateError,
    ShardedAutoregressiveRunner,
)
from prsm.compute.tee.models import TEEType


# ──────────────────────────────────────────────────────────────────────────
# Test fakes
# ──────────────────────────────────────────────────────────────────────────


class _FakeShardedModel:
    """Deterministic ``ShardedLayerForward`` impl. Records every
    call so tests can assert on layer_range, payload threading,
    + call counts.

    Forward semantics: returns ``hidden`` = a small np.ndarray
    that encodes the call type + a counter. The KV-cache payload
    is a list of strings ``["layer_i_pos_j", ...]`` representing
    cached positions; PREFILL initializes it to the prompt
    length; INCREMENTAL appends one entry."""

    def __init__(self) -> None:
        self.prefill_calls: List[dict] = []
        self.incremental_calls: List[dict] = []

    def forward_prefill(
        self,
        *,
        input_or_hidden: Any,
        layer_range: Tuple[int, int],
    ) -> Tuple[np.ndarray, List[str]]:
        self.prefill_calls.append({
            "input_or_hidden": input_or_hidden,
            "layer_range": layer_range,
        })
        # Pretend prompt has N positions; build a payload entry
        # per cached position per layer in range.
        if isinstance(input_or_hidden, list):
            n_positions = len(input_or_hidden)
        else:
            # np.ndarray — assume shape [batch=1, seq_len, hidden]
            # or [seq_len, hidden]; either way, pull seq_len.
            arr = np.asarray(input_or_hidden)
            n_positions = arr.shape[-2] if arr.ndim >= 2 else arr.shape[0]
        start, end = layer_range
        payload = [
            f"L{layer}_P{pos}"
            for layer in range(start, end)
            for pos in range(n_positions)
        ]
        # Hidden state: encode call counter + range size for
        # determinism in tests.
        hidden = np.array(
            [[len(self.prefill_calls), end - start, n_positions]],
            dtype=np.float32,
        )
        return hidden, payload

    def forward_incremental(
        self,
        *,
        input_or_hidden: Any,
        layer_range: Tuple[int, int],
        kv_cache_payload: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        self.incremental_calls.append({
            "input_or_hidden": input_or_hidden,
            "layer_range": layer_range,
            "incoming_payload_len": len(kv_cache_payload),
            "incoming_payload_id": id(kv_cache_payload),
        })
        # Mutate the payload in place — append one new position
        # per layer in range. Returns the same reference (matches
        # the runner's "may mutate in place + return same ref"
        # contract).
        start, end = layer_range
        # Existing layout was payload[ (layer_idx * n_positions) + pos ]
        # for compactness across layers. Just append per-layer
        # to the end (the model's internal indexing is not the
        # runner's concern).
        for layer in range(start, end):
            kv_cache_payload.append(
                f"L{layer}_INC{len(self.incremental_calls)}",
            )
        hidden = np.array(
            [[len(self.incremental_calls), end - start, 1]],
            dtype=np.float32,
        )
        return hidden, kv_cache_payload


def _make_runner(
    *,
    layer_range: Tuple[int, int] = (0, 6),
    cache_kwargs: dict = None,
) -> Tuple[ShardedAutoregressiveRunner, _FakeShardedModel, KVCacheManager]:
    """Helper: spin up runner + fake model + manager."""
    model = _FakeShardedModel()
    cache = KVCacheManager(**(cache_kwargs or {}))
    runner = ShardedAutoregressiveRunner(
        model=model,
        layer_range=layer_range,
        kv_cache_manager=cache,
        tee_attestation=b"fake-attestation-32-bytes-payload",
        tee_type=TEEType.SOFTWARE,
    )
    return runner, model, cache


# ──────────────────────────────────────────────────────────────────────────
# Constructor validation
# ──────────────────────────────────────────────────────────────────────────


class TestConstructorValidation:
    def test_rejects_none_model(self):
        cache = KVCacheManager()
        with pytest.raises(RuntimeError, match="model is required"):
            ShardedAutoregressiveRunner(
                model=None,
                layer_range=(0, 6),
                kv_cache_manager=cache,
                tee_attestation=b"x" * 32,
                tee_type=TEEType.SOFTWARE,
            )

    def test_rejects_model_missing_forward_prefill(self):
        class _Bad:
            def forward_incremental(self, **kw):
                ...
        cache = KVCacheManager()
        with pytest.raises(RuntimeError, match="forward_prefill"):
            ShardedAutoregressiveRunner(
                model=_Bad(),
                layer_range=(0, 6),
                kv_cache_manager=cache,
                tee_attestation=b"x" * 32,
                tee_type=TEEType.SOFTWARE,
            )

    def test_rejects_model_missing_forward_incremental(self):
        class _Bad:
            def forward_prefill(self, **kw):
                ...
        cache = KVCacheManager()
        with pytest.raises(RuntimeError, match="forward_incremental"):
            ShardedAutoregressiveRunner(
                model=_Bad(),
                layer_range=(0, 6),
                kv_cache_manager=cache,
                tee_attestation=b"x" * 32,
                tee_type=TEEType.SOFTWARE,
            )

    def test_rejects_inverted_layer_range(self):
        cache = KVCacheManager()
        with pytest.raises(RuntimeError, match="layer_range"):
            ShardedAutoregressiveRunner(
                model=_FakeShardedModel(),
                layer_range=(6, 0),
                kv_cache_manager=cache,
                tee_attestation=b"x" * 32,
                tee_type=TEEType.SOFTWARE,
            )

    def test_rejects_negative_layer_range_start(self):
        cache = KVCacheManager()
        with pytest.raises(RuntimeError, match="layer_range"):
            ShardedAutoregressiveRunner(
                model=_FakeShardedModel(),
                layer_range=(-1, 6),
                kv_cache_manager=cache,
                tee_attestation=b"x" * 32,
                tee_type=TEEType.SOFTWARE,
            )

    def test_rejects_bool_layer_range_start(self):
        cache = KVCacheManager()
        with pytest.raises(RuntimeError, match="layer_range"):
            ShardedAutoregressiveRunner(
                model=_FakeShardedModel(),
                layer_range=(True, 6),  # type: ignore[arg-type]
                kv_cache_manager=cache,
                tee_attestation=b"x" * 32,
                tee_type=TEEType.SOFTWARE,
            )

    def test_rejects_non_tuple_layer_range(self):
        cache = KVCacheManager()
        with pytest.raises(RuntimeError, match="layer_range"):
            ShardedAutoregressiveRunner(
                model=_FakeShardedModel(),
                layer_range=[0, 6],  # type: ignore[arg-type]
                kv_cache_manager=cache,
                tee_attestation=b"x" * 32,
                tee_type=TEEType.SOFTWARE,
            )

    def test_rejects_non_manager(self):
        with pytest.raises(RuntimeError, match="kv_cache_manager"):
            ShardedAutoregressiveRunner(
                model=_FakeShardedModel(),
                layer_range=(0, 6),
                kv_cache_manager="not a manager",  # type: ignore[arg-type]
                tee_attestation=b"x" * 32,
                tee_type=TEEType.SOFTWARE,
            )

    def test_rejects_non_bytes_attestation(self):
        cache = KVCacheManager()
        with pytest.raises(RuntimeError, match="tee_attestation"):
            ShardedAutoregressiveRunner(
                model=_FakeShardedModel(),
                layer_range=(0, 6),
                kv_cache_manager=cache,
                tee_attestation="not bytes",  # type: ignore[arg-type]
                tee_type=TEEType.SOFTWARE,
            )

    def test_rejects_non_tee_type(self):
        cache = KVCacheManager()
        with pytest.raises(RuntimeError, match="tee_type"):
            ShardedAutoregressiveRunner(
                model=_FakeShardedModel(),
                layer_range=(0, 6),
                kv_cache_manager=cache,
                tee_attestation=b"x" * 32,
                tee_type="software",  # type: ignore[arg-type]
            )

    def test_accepts_bytearray_attestation(self):
        # bytearray accepted (coerced to bytes internally).
        cache = KVCacheManager()
        runner = ShardedAutoregressiveRunner(
            model=_FakeShardedModel(),
            layer_range=(0, 6),
            kv_cache_manager=cache,
            tee_attestation=bytearray(b"x" * 32),
            tee_type=TEEType.SOFTWARE,
        )
        assert runner.tee_attestation == b"x" * 32

    def test_exposes_layer_range_and_n_layers(self):
        runner, _, _ = _make_runner(layer_range=(2, 5))
        assert runner.layer_range == (2, 5)
        assert runner.n_layers == 3


# ──────────────────────────────────────────────────────────────────────────
# PREFILL
# ──────────────────────────────────────────────────────────────────────────


class TestPrefill:
    def test_prefill_allocates_cache(self):
        runner, _, cache = _make_runner()
        assert "req-1" not in cache
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3, 4],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        assert isinstance(result, LayerSliceResult)
        assert result.decode_mode == DecodeMode.PREFILL
        assert "req-1" in cache

    def test_prefill_stores_payload_on_handle(self):
        runner, _, cache = _make_runner(layer_range=(0, 2))
        runner.run_layer_slice_unary(
            activation_or_input_ids=[10, 20, 30],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        handle = cache.get("req-1")
        assert handle is not None
        # 2 layers × 3 positions = 6 payload entries
        assert handle.payload == [
            "L0_P0", "L0_P1", "L0_P2",
            "L1_P0", "L1_P1", "L1_P2",
        ]

    def test_prefill_returns_boundary_hidden_state(self):
        runner, _, _ = _make_runner(layer_range=(0, 6))
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3, 4, 5],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        # FakeModel encodes [call_count, n_layers, n_positions]
        assert result.hidden_state.tolist() == [[1, 6, 5]]
        assert result.n_layers_run == 6

    def test_prefill_passes_input_through_to_model(self):
        runner, model, _ = _make_runner()
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        assert len(model.prefill_calls) == 1
        assert model.prefill_calls[0]["input_or_hidden"] == [1, 2, 3]

    def test_prefill_double_for_same_id_raises(self):
        # Manager raises CacheAlreadyAllocatedError, runner does
        # not silently swallow.
        runner, _, _ = _make_runner()
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        from prsm.compute.chain_rpc.kv_cache import (
            CacheAlreadyAllocatedError,
        )
        with pytest.raises(CacheAlreadyAllocatedError):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[1, 2, 3],
                request_id="req-1",
                decode_mode=DecodeMode.PREFILL,
            )


# ──────────────────────────────────────────────────────────────────────────
# INCREMENTAL
# ──────────────────────────────────────────────────────────────────────────


class TestIncremental:
    def test_incremental_uses_existing_cache(self):
        runner, model, _ = _make_runner(layer_range=(0, 2))
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[42],
            request_id="req-1",
            decode_mode=DecodeMode.INCREMENTAL,
        )
        # Model should have received the PREFILL's payload as
        # kv_cache_payload kwarg.
        assert len(model.incremental_calls) == 1
        # The 6-entry PREFILL payload (2 layers × 3 positions)
        # was the incoming payload.
        assert model.incremental_calls[0]["incoming_payload_len"] == 6

    def test_incremental_without_prior_prefill_raises(self):
        runner, _, _ = _make_runner()
        with pytest.raises(
            MalformedCacheStateError, match="no prior PREFILL"
        ):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[42],
                request_id="never-prefilled",
                decode_mode=DecodeMode.INCREMENTAL,
            )

    def test_incremental_payload_threaded_across_calls(self):
        # PREFILL → 6 positions cached. 5 INCREMENTAL calls
        # should each see the prior call's payload + extend it.
        runner, model, cache = _make_runner(layer_range=(0, 2))
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        for _ in range(5):
            runner.run_layer_slice_unary(
                activation_or_input_ids=np.array([[42]]),
                request_id="req-1",
                decode_mode=DecodeMode.INCREMENTAL,
            )
        # Each INCREMENTAL adds 2 entries (layers 0+1).
        # Initial 6 + 5 × 2 = 16.
        handle = cache.get("req-1")
        assert handle is not None
        assert len(handle.payload) == 16
        # Confirm the LAST incremental call saw 14 incoming
        # entries (initial 6 + 4 previous incrementals × 2 = 14).
        assert (
            model.incremental_calls[-1]["incoming_payload_len"] == 14
        )

    def test_incremental_returns_decode_mode_field(self):
        runner, _, _ = _make_runner()
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=np.array([[42]]),
            request_id="req-1",
            decode_mode=DecodeMode.INCREMENTAL,
        )
        assert result.decode_mode == DecodeMode.INCREMENTAL

    def test_incremental_after_evict_raises(self):
        # Evict mid-stream → next INCREMENTAL hits the malformed
        # path. Models the TTL-expiry-out-from-under-request race.
        runner, _, _ = _make_runner()
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        assert runner.evict("req-1") is True
        with pytest.raises(MalformedCacheStateError):
            runner.run_layer_slice_unary(
                activation_or_input_ids=np.array([[42]]),
                request_id="req-1",
                decode_mode=DecodeMode.INCREMENTAL,
            )


# ──────────────────────────────────────────────────────────────────────────
# layer_range respected
# ──────────────────────────────────────────────────────────────────────────


class TestLayerRangeRespected:
    def test_prefill_passes_layer_range_to_model(self):
        runner, model, _ = _make_runner(layer_range=(2, 5))
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        assert model.prefill_calls[0]["layer_range"] == (2, 5)

    def test_incremental_passes_layer_range_to_model(self):
        runner, model, _ = _make_runner(layer_range=(2, 5))
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=np.array([[42]]),
            request_id="req-1",
            decode_mode=DecodeMode.INCREMENTAL,
        )
        assert model.incremental_calls[0]["layer_range"] == (2, 5)

    def test_prefill_n_layers_run_matches_range(self):
        runner, _, _ = _make_runner(layer_range=(2, 5))
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        assert result.n_layers_run == 3


# ──────────────────────────────────────────────────────────────────────────
# Cache survives across many incremental calls
# ──────────────────────────────────────────────────────────────────────────


class TestCacheSurvival:
    def test_cache_survives_10_incremental_iterations(self):
        runner, _, cache = _make_runner(layer_range=(0, 4))
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        # 10 successive INCREMENTAL calls — same request_id, same
        # handle.
        for _ in range(10):
            assert "req-1" in cache
            runner.run_layer_slice_unary(
                activation_or_input_ids=np.array([[42]]),
                request_id="req-1",
                decode_mode=DecodeMode.INCREMENTAL,
            )
        # Initial 4 layers × 3 positions = 12; plus 10 increments
        # × 4 layers = 40; total 52.
        handle = cache.get("req-1")
        assert handle is not None
        assert len(handle.payload) == 52

    def test_cache_survives_concurrent_distinct_ids(self):
        # Two requests interleave PREFILL + INCREMENTAL. Caches
        # must stay independent — req-1's payload doesn't leak
        # into req-2.
        runner, _, cache = _make_runner(layer_range=(0, 2))
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],  # 3 positions
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[10, 20, 30, 40, 50],  # 5 positions
            request_id="req-2",
            decode_mode=DecodeMode.PREFILL,
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=np.array([[7]]),
            request_id="req-1",
            decode_mode=DecodeMode.INCREMENTAL,
        )
        h1 = cache.get("req-1")
        h2 = cache.get("req-2")
        assert h1 is not None and h2 is not None
        # req-1: 6 PREFILL + 2 INC = 8
        assert len(h1.payload) == 8
        # req-2: 10 PREFILL + 0 INC = 10 (untouched)
        assert len(h2.payload) == 10


# ──────────────────────────────────────────────────────────────────────────
# Eviction
# ──────────────────────────────────────────────────────────────────────────


class TestEviction:
    def test_evict_returns_true_when_handle_exists(self):
        runner, _, _ = _make_runner()
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        assert runner.evict("req-1") is True

    def test_evict_idempotent_returns_false(self):
        runner, _, _ = _make_runner()
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        runner.evict("req-1")
        assert runner.evict("req-1") is False

    def test_evict_unknown_returns_false(self):
        runner, _, _ = _make_runner()
        assert runner.evict("never-allocated") is False


# ──────────────────────────────────────────────────────────────────────────
# Tail rejected (Task 4 deferral)
# ──────────────────────────────────────────────────────────────────────────


class TestTailRejected:
    def test_is_final_stage_true_raises(self):
        runner, _, _ = _make_runner()
        with pytest.raises(RuntimeError, match="is_final_stage"):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[1, 2, 3],
                request_id="req-1",
                decode_mode=DecodeMode.PREFILL,
                is_final_stage=True,
            )

    def test_is_final_stage_false_dispatches_normally(self):
        runner, _, _ = _make_runner()
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=False,
        )
        assert result.next_token_id is None
        assert result.is_terminal is False


# ──────────────────────────────────────────────────────────────────────────
# Dispatch validation
# ──────────────────────────────────────────────────────────────────────────


class TestDispatchValidation:
    def test_rejects_empty_request_id(self):
        runner, _, _ = _make_runner()
        with pytest.raises(RuntimeError, match="request_id"):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[1, 2, 3],
                request_id="",
                decode_mode=DecodeMode.PREFILL,
            )

    def test_rejects_non_decode_mode(self):
        runner, _, _ = _make_runner()
        with pytest.raises(RuntimeError, match="decode_mode"):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[1, 2, 3],
                request_id="req-1",
                decode_mode="prefill",  # type: ignore[arg-type]
            )
