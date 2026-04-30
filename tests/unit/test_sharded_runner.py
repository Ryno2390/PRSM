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

from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

from prsm.compute.chain_rpc.kv_cache import KVCacheManager
from prsm.compute.chain_rpc.protocol import DecodeMode
from prsm.compute.inference.autoregressive_runner import SamplingDefaults
from prsm.compute.inference.sharded_runner import (
    LayerSliceResult,
    MalformedCacheStateError,
    MissingTailCapabilityError,
    MissingVerifyCapabilityError,
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
    def test_non_tail_runner_rejects_is_final_stage_true(self):
        # Non-tail-only construction (no sampling_defaults) ->
        # tail dispatch raises MissingTailCapabilityError.
        runner, _, _ = _make_runner()
        with pytest.raises(
            MissingTailCapabilityError, match="tail-capable"
        ):
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


# ──────────────────────────────────────────────────────────────────────────
# Tail variant — Task 4
# ──────────────────────────────────────────────────────────────────────────


class _FakeTailShardedModel(_FakeShardedModel):
    """``_FakeShardedModel`` extended with the tail-only
    ``apply_lm_head_and_sample`` method. Drives sampling via
    a configurable script of token ids returned in order; tests
    inject specific scripts for greedy / EOS / max_tokens
    coverage."""

    def __init__(self, sample_script: List[int]) -> None:
        super().__init__()
        self._script = list(sample_script)
        self._cursor = 0
        self.sample_calls: List[dict] = []

    def apply_lm_head_and_sample(
        self,
        *,
        hidden_state: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        self.sample_calls.append({
            "hidden_state": hidden_state,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })
        if self._cursor >= len(self._script):
            raise AssertionError(
                "_FakeTailShardedModel: sample script exhausted "
                "(test bug — pad the script or shorten max_tokens)"
            )
        tok = self._script[self._cursor]
        self._cursor += 1
        return tok


def _make_tail_runner(
    *,
    sample_script: List[int],
    layer_range: Tuple[int, int] = (0, 6),
    eos_token_id: int = 999,
    sampling_defaults: SamplingDefaults = None,
) -> Tuple[ShardedAutoregressiveRunner, _FakeTailShardedModel, KVCacheManager]:
    model = _FakeTailShardedModel(sample_script)
    cache = KVCacheManager()
    runner = ShardedAutoregressiveRunner(
        model=model,
        layer_range=layer_range,
        kv_cache_manager=cache,
        tee_attestation=b"x" * 32,
        tee_type=TEEType.SOFTWARE,
        sampling_defaults=sampling_defaults or SamplingDefaults(
            max_tokens=512, temperature=1.0, top_k=50, top_p=0.95,
        ),
        eos_token_id=eos_token_id,
    )
    return runner, model, cache


class _FakeRequest:
    """Minimal request shim — duck-typed for getattr access to
    ``max_tokens`` + ``temperature``."""

    def __init__(
        self,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature


class TestTailVariant:
    def test_tail_samples_next_token_id_on_prefill(self):
        runner, _, _ = _make_tail_runner(sample_script=[42])
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=10),
        )
        assert result.next_token_id == 42
        assert result.is_terminal is False

    def test_tail_passes_temperature_and_top_kp_to_model(self):
        runner, model, _ = _make_tail_runner(
            sample_script=[42],
            sampling_defaults=SamplingDefaults(
                max_tokens=10, temperature=0.7, top_k=40, top_p=0.9,
            ),
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(temperature=0.0),  # request override
        )
        assert len(model.sample_calls) == 1
        call = model.sample_calls[0]
        # Request override on temperature reaches the model.
        assert call["temperature"] == 0.0
        # Defaults reach the model for top_k/top_p (no wire field).
        assert call["top_k"] == 40
        assert call["top_p"] == 0.9

    def test_tail_greedy_temperature_zero_deterministic(self):
        # Greedy script: always returns 7. Two identical dispatches
        # should produce identical token sequences.
        runner_a, _, _ = _make_tail_runner(sample_script=[7, 7, 7])
        runner_b, _, _ = _make_tail_runner(sample_script=[7, 7, 7])
        req = _FakeRequest(max_tokens=3, temperature=0.0)
        out_a = []
        out_b = []
        for runner, out, rid in [
            (runner_a, out_a, "ra"),
            (runner_b, out_b, "rb"),
        ]:
            r1 = runner.run_layer_slice_unary(
                activation_or_input_ids=[1, 2, 3],
                request_id=rid,
                decode_mode=DecodeMode.PREFILL,
                is_final_stage=True,
                request=req,
            )
            out.append(r1.next_token_id)
            for _ in range(2):
                r = runner.run_layer_slice_unary(
                    activation_or_input_ids=np.array([[r1.next_token_id]]),
                    request_id=rid,
                    decode_mode=DecodeMode.INCREMENTAL,
                    is_final_stage=True,
                    request=req,
                )
                out.append(r.next_token_id)
        assert out_a == out_b == [7, 7, 7]

    def test_tail_max_tokens_one_returns_terminal_on_prefill(self):
        runner, _, _ = _make_tail_runner(sample_script=[42])
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=1),
        )
        assert result.next_token_id == 42
        assert result.is_terminal is True

    def test_tail_max_tokens_three_terminates_on_third_call(self):
        # PREFILL produces token #1; INCREMENTAL #1 produces token
        # #2; INCREMENTAL #2 produces token #3 — that one should
        # be is_terminal=True.
        runner, _, _ = _make_tail_runner(sample_script=[10, 20, 30])
        req = _FakeRequest(max_tokens=3)
        r1 = runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=req,
        )
        assert r1.next_token_id == 10
        assert r1.is_terminal is False
        r2 = runner.run_layer_slice_unary(
            activation_or_input_ids=np.array([[10]]),
            request_id="req-1",
            decode_mode=DecodeMode.INCREMENTAL,
            is_final_stage=True,
            request=req,
        )
        assert r2.next_token_id == 20
        assert r2.is_terminal is False
        r3 = runner.run_layer_slice_unary(
            activation_or_input_ids=np.array([[20]]),
            request_id="req-1",
            decode_mode=DecodeMode.INCREMENTAL,
            is_final_stage=True,
            request=req,
        )
        assert r3.next_token_id == 30
        assert r3.is_terminal is True

    def test_tail_eos_returns_terminal(self):
        # EOS token 999. Sample script: token 50, then EOS.
        runner, _, _ = _make_tail_runner(
            sample_script=[50, 999], eos_token_id=999,
        )
        req = _FakeRequest(max_tokens=10)
        r1 = runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=req,
        )
        assert r1.next_token_id == 50
        assert r1.is_terminal is False
        r2 = runner.run_layer_slice_unary(
            activation_or_input_ids=np.array([[50]]),
            request_id="req-1",
            decode_mode=DecodeMode.INCREMENTAL,
            is_final_stage=True,
            request=req,
        )
        assert r2.next_token_id == 999
        assert r2.is_terminal is True

    def test_tail_no_eos_token_id_set_only_max_tokens_terminates(self):
        # eos_token_id=None — runner ignores any matching id.
        runner, _, _ = _make_tail_runner(
            sample_script=[999, 5], eos_token_id=None,
        )
        req = _FakeRequest(max_tokens=2)
        r1 = runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=req,
        )
        # Token "999" looks like EOS but EOS is disabled.
        assert r1.next_token_id == 999
        assert r1.is_terminal is False
        r2 = runner.run_layer_slice_unary(
            activation_or_input_ids=np.array([[999]]),
            request_id="req-1",
            decode_mode=DecodeMode.INCREMENTAL,
            is_final_stage=True,
            request=req,
        )
        # max_tokens=2 reached — terminates.
        assert r2.is_terminal is True

    def test_non_tail_dispatch_on_tail_capable_runner_leaves_token_none(self):
        # Even a tail-capable runner, when dispatched with
        # is_final_stage=False, behaves like the non-tail variant
        # (no sampling, no token bump).
        runner, model, cache = _make_tail_runner(
            sample_script=[42],
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=False,
            request=_FakeRequest(max_tokens=10),
        )
        assert result.next_token_id is None
        assert result.is_terminal is False
        # Sampling NOT invoked.
        assert model.sample_calls == []
        # tokens_generated stays 0 on the handle.
        h = cache.get("req-1")
        assert h is not None
        assert h.tokens_generated == 0

    def test_tail_falls_back_to_runner_defaults_when_request_none(self):
        # request=None path — runner uses defaults exclusively.
        # max_tokens default 1 means PREFILL is_terminal=True.
        runner, model, _ = _make_tail_runner(
            sample_script=[42],
            sampling_defaults=SamplingDefaults(
                max_tokens=1, temperature=0.0, top_k=50, top_p=0.95,
            ),
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=None,
        )
        assert result.next_token_id == 42
        assert result.is_terminal is True
        # Defaults reach model.
        assert model.sample_calls[0]["temperature"] == 0.0


class TestTailConstructorValidation:
    def test_tail_capable_requires_apply_lm_head_method(self):
        # Non-tail _FakeShardedModel doesn't have
        # apply_lm_head_and_sample; tail-capable construction
        # rejects.
        cache = KVCacheManager()
        with pytest.raises(
            RuntimeError, match="apply_lm_head_and_sample"
        ):
            ShardedAutoregressiveRunner(
                model=_FakeShardedModel(),
                layer_range=(0, 6),
                kv_cache_manager=cache,
                tee_attestation=b"x" * 32,
                tee_type=TEEType.SOFTWARE,
                sampling_defaults=SamplingDefaults(),
                eos_token_id=0,
            )

    def test_rejects_non_sampling_defaults(self):
        cache = KVCacheManager()
        with pytest.raises(
            RuntimeError, match="sampling_defaults"
        ):
            ShardedAutoregressiveRunner(
                model=_FakeTailShardedModel([1]),
                layer_range=(0, 6),
                kv_cache_manager=cache,
                tee_attestation=b"x" * 32,
                tee_type=TEEType.SOFTWARE,
                sampling_defaults={"max_tokens": 10},  # type: ignore[arg-type]
                eos_token_id=0,
            )

    def test_rejects_negative_eos_token_id(self):
        cache = KVCacheManager()
        with pytest.raises(RuntimeError, match="eos_token_id"):
            ShardedAutoregressiveRunner(
                model=_FakeTailShardedModel([1]),
                layer_range=(0, 6),
                kv_cache_manager=cache,
                tee_attestation=b"x" * 32,
                tee_type=TEEType.SOFTWARE,
                sampling_defaults=SamplingDefaults(),
                eos_token_id=-1,
            )

    def test_rejects_bool_eos_token_id(self):
        cache = KVCacheManager()
        with pytest.raises(RuntimeError, match="eos_token_id"):
            ShardedAutoregressiveRunner(
                model=_FakeTailShardedModel([1]),
                layer_range=(0, 6),
                kv_cache_manager=cache,
                tee_attestation=b"x" * 32,
                tee_type=TEEType.SOFTWARE,
                sampling_defaults=SamplingDefaults(),
                eos_token_id=True,  # type: ignore[arg-type]
            )


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.y Task 4 — VERIFY (speculative-decoding) variant
# ──────────────────────────────────────────────────────────────────────────


class _FakeVerifyShardedModel(_FakeShardedModel):
    """``_FakeShardedModel`` extended with the VERIFY-only
    ``forward_verify`` method. Records every VERIFY call.
    Forward semantics: returns hidden = an ndarray shaped
    ``[K+1, hidden=3]`` with deterministic content; appends K+1
    cache entries per layer to the kv_cache_payload."""

    def __init__(self) -> None:
        super().__init__()
        self.verify_calls: List[dict] = []

    def forward_verify(
        self,
        *,
        input_or_hidden: Any,
        layer_range: Tuple[int, int],
        kv_cache_payload: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        self.verify_calls.append({
            "input_or_hidden": input_or_hidden,
            "layer_range": layer_range,
            "incoming_payload_len": len(kv_cache_payload),
            "incoming_payload_id": id(kv_cache_payload),
        })
        # Resolve K+1 from input shape.
        if isinstance(input_or_hidden, list):
            n_positions = len(input_or_hidden)
        else:
            arr = np.asarray(input_or_hidden)
            n_positions = arr.shape[-2]
        start, end = layer_range
        # Append K+1 cache entries per layer.
        for layer in range(start, end):
            for j in range(n_positions):
                kv_cache_payload.append(
                    f"L{layer}_VRF{len(self.verify_calls)}_P{j}",
                )
        # Hidden state shape [K+1, 3] — distinct per position.
        hidden = np.array(
            [
                [len(self.verify_calls), end - start, p]
                for p in range(n_positions)
            ],
            dtype=np.float32,
        )
        return hidden, kv_cache_payload


class _FakeVerifyTailShardedModel(_FakeVerifyShardedModel):
    """Tail-capable VERIFY model. Adds ``apply_lm_head_and_sample``
    (for non-VERIFY tail dispatches in mixed tests) +
    ``apply_lm_head_and_sample_batch`` (the VERIFY-tail batch
    sampler). Both driven by configurable scripts of token ids."""

    def __init__(
        self,
        *,
        sample_script: List[int],
        verify_batch_script: List[List[int]],
    ) -> None:
        super().__init__()
        self._sample_script = list(sample_script)
        self._sample_cursor = 0
        self._verify_batch_script = [list(b) for b in verify_batch_script]
        self._verify_batch_cursor = 0
        self.sample_calls: List[dict] = []
        self.batch_sample_calls: List[dict] = []

    def apply_lm_head_and_sample(
        self,
        *,
        hidden_state: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        self.sample_calls.append({
            "hidden_state": hidden_state,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })
        if self._sample_cursor >= len(self._sample_script):
            raise AssertionError(
                "_FakeVerifyTailShardedModel: sample script "
                "exhausted (test bug)"
            )
        tok = self._sample_script[self._sample_cursor]
        self._sample_cursor += 1
        return tok

    def apply_lm_head_and_sample_batch(
        self,
        *,
        hidden_state_batch: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> List[int]:
        self.batch_sample_calls.append({
            "hidden_state_batch": hidden_state_batch,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })
        if self._verify_batch_cursor >= len(self._verify_batch_script):
            raise AssertionError(
                "_FakeVerifyTailShardedModel: verify batch script "
                "exhausted (test bug)"
            )
        out = self._verify_batch_script[self._verify_batch_cursor]
        self._verify_batch_cursor += 1
        return list(out)


def _make_verify_runner(
    *,
    layer_range: Tuple[int, int] = (0, 6),
) -> Tuple[
    ShardedAutoregressiveRunner, _FakeVerifyShardedModel, KVCacheManager,
]:
    """Helper: spin up a non-tail VERIFY-capable runner."""
    model = _FakeVerifyShardedModel()
    cache = KVCacheManager()
    runner = ShardedAutoregressiveRunner(
        model=model,
        layer_range=layer_range,
        kv_cache_manager=cache,
        tee_attestation=b"x" * 32,
        tee_type=TEEType.SOFTWARE,
    )
    return runner, model, cache


def _make_verify_tail_runner(
    *,
    sample_script: List[int],
    verify_batch_script: List[List[int]],
    layer_range: Tuple[int, int] = (0, 6),
    eos_token_id: int = 999,
    sampling_defaults: SamplingDefaults = None,
) -> Tuple[
    ShardedAutoregressiveRunner,
    _FakeVerifyTailShardedModel,
    KVCacheManager,
]:
    model = _FakeVerifyTailShardedModel(
        sample_script=sample_script,
        verify_batch_script=verify_batch_script,
    )
    cache = KVCacheManager()
    runner = ShardedAutoregressiveRunner(
        model=model,
        layer_range=layer_range,
        kv_cache_manager=cache,
        tee_attestation=b"x" * 32,
        tee_type=TEEType.SOFTWARE,
        sampling_defaults=sampling_defaults or SamplingDefaults(
            max_tokens=512, temperature=0.0, top_k=50, top_p=0.95,
        ),
        eos_token_id=eos_token_id,
    )
    return runner, model, cache


class TestVerifyNonTail:
    def test_verify_requires_prior_prefill(self):
        runner, _, _ = _make_verify_runner()
        with pytest.raises(MalformedCacheStateError, match="VERIFY"):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[1, 2, 3, 4, 5],
                request_id="req-no-prefill",
                decode_mode=DecodeMode.VERIFY,
            )

    def test_verify_happy_path_appends_cache_and_returns_batch(self):
        runner, model, cache = _make_verify_runner(layer_range=(0, 2))
        runner.run_layer_slice_unary(
            activation_or_input_ids=[10, 20, 30],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        prefill_payload_len = len(cache.get("req-1").payload)
        # VERIFY with parent + 4 drafts (K=4, K+1=5).
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101, 102, 103],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
        )
        assert result.decode_mode == DecodeMode.VERIFY
        # K+1 hidden states returned.
        assert result.hidden_state.shape == (5, 3)
        # Non-tail leaves verify-only response fields None.
        assert result.next_token_id is None
        assert result.is_terminal is False
        assert result.verified_token_ids is None
        assert result.accepted_count is None
        # Cache extended with K+1 positions per layer (2 layers, 5
        # positions = 10 new entries).
        new_payload_len = len(cache.get("req-1").payload)
        assert new_payload_len == prefill_payload_len + 2 * 5
        # forward_verify called with the prior payload.
        assert len(model.verify_calls) == 1
        assert (
            model.verify_calls[0]["incoming_payload_len"]
            == prefill_payload_len
        )

    def test_verify_rejects_input_below_two_positions(self):
        # K+1 must be >= 2 (parent + at least one draft).
        runner, _, _ = _make_verify_runner()
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        with pytest.raises(RuntimeError, match="at least 2"):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[42],
                request_id="req-1",
                decode_mode=DecodeMode.VERIFY,
            )

    def test_verify_rejects_input_above_cap(self):
        from prsm.compute.chain_rpc.protocol import (
            MAX_VERIFY_BATCH_TOKENS,
        )
        runner, _, _ = _make_verify_runner()
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        # K+1 == cap+1 is rejected.
        too_long = list(range(MAX_VERIFY_BATCH_TOKENS + 1))
        with pytest.raises(RuntimeError, match="exceeds cap"):
            runner.run_layer_slice_unary(
                activation_or_input_ids=too_long,
                request_id="req-1",
                decode_mode=DecodeMode.VERIFY,
            )

    def test_verify_accepts_ndarray_batch_input(self):
        # Stage > 1 input — ndarray of shape [K+1, hidden].
        runner, model, _ = _make_verify_runner(layer_range=(2, 4))
        runner.run_layer_slice_unary(
            activation_or_input_ids=np.zeros(
                (3, 8), dtype=np.float32,
            ),
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        # K+1 = 5 positions, hidden = 8.
        batch_in = np.ones((5, 8), dtype=np.float32)
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=batch_in,
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
        )
        assert result.hidden_state.shape == (5, 3)
        assert model.verify_calls[0]["input_or_hidden"] is batch_in

    def test_verify_missing_forward_verify_method_raises(self):
        # Plain non-VERIFY model. PREFILL ok (model has
        # forward_prefill). VERIFY raises
        # MissingVerifyCapabilityError.
        cache = KVCacheManager()
        runner = ShardedAutoregressiveRunner(
            model=_FakeShardedModel(),
            layer_range=(0, 6),
            kv_cache_manager=cache,
            tee_attestation=b"x" * 32,
            tee_type=TEEType.SOFTWARE,
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        with pytest.raises(
            MissingVerifyCapabilityError,
            match="forward_verify",
        ):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[42, 100, 101],
                request_id="req-1",
                decode_mode=DecodeMode.VERIFY,
            )

    def test_proposed_token_ids_rejected_outside_verify(self):
        runner, _, _ = _make_verify_runner()
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
        )
        with pytest.raises(
            RuntimeError, match="proposed_token_ids"
        ):
            runner.run_layer_slice_unary(
                activation_or_input_ids=np.array([[1.0, 2.0, 3.0]]),
                request_id="req-1",
                decode_mode=DecodeMode.INCREMENTAL,
                proposed_token_ids=[5, 6],
            )


class TestVerifyTail:
    def test_tail_verify_all_accepted_emits_k_plus_one_tokens(self):
        # Drafts = [100, 101, 102, 103]; verifier returns
        # [100, 101, 102, 103, 999_bonus] — all 4 match → emit
        # 5 tokens (the K+1 batch verifies fully + bonus).
        runner, model, cache = _make_verify_tail_runner(
            sample_script=[42],
            verify_batch_script=[[100, 101, 102, 103, 50]],
        )
        # PREFILL.
        r1 = runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
        )
        assert r1.next_token_id == 42
        assert cache.get("req-1").tokens_generated == 1
        # VERIFY. parent=42; drafts=[100,101,102,103] (K=4).
        r2 = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101, 102, 103],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
            proposed_token_ids=[100, 101, 102, 103],
        )
        assert r2.decode_mode == DecodeMode.VERIFY
        assert r2.verified_token_ids == (100, 101, 102, 103, 50)
        assert r2.accepted_count == 4  # all 4 drafts accepted
        # Last emitted = verified[4] = 50 (bonus).
        assert r2.next_token_id == 50
        assert r2.is_terminal is False
        # tokens_generated bumped by 5 (K+1 emitted).
        assert cache.get("req-1").tokens_generated == 1 + 5
        # Sampling params reached the model.
        assert len(model.batch_sample_calls) == 1
        assert model.batch_sample_calls[0]["temperature"] == 0.0

    def test_tail_verify_zero_accepted_emits_one_correction(self):
        # Drafts = [100, 101]; verifier returns [200, 201, 202] —
        # 0 match → emit 1 token (the verifier's correction at
        # position 0).
        runner, _, cache = _make_verify_tail_runner(
            sample_script=[42],
            verify_batch_script=[[200, 201, 202]],
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
        )
        r2 = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
            proposed_token_ids=[100, 101],
        )
        assert r2.accepted_count == 0
        assert r2.next_token_id == 200  # correction
        assert r2.verified_token_ids == (200, 201, 202)
        assert r2.is_terminal is False
        # Only 1 token emitted (the correction).
        assert cache.get("req-1").tokens_generated == 1 + 1

    def test_tail_verify_partial_accept_two_of_four(self):
        # Drafts = [100, 101, 102, 103]; verifier returns
        # [100, 101, 999, 998, 997] — first 2 match, then
        # divergence → emit 3 tokens (verified[0..2]).
        runner, _, cache = _make_verify_tail_runner(
            sample_script=[42],
            verify_batch_script=[[100, 101, 999, 998, 997]],
            eos_token_id=12345,  # not 999
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
        )
        r2 = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101, 102, 103],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
            proposed_token_ids=[100, 101, 102, 103],
        )
        assert r2.accepted_count == 2
        # Emitted = verified[0..2] = (100, 101, 999); last = 999.
        assert r2.next_token_id == 999
        assert r2.is_terminal is False
        assert cache.get("req-1").tokens_generated == 1 + 3

    def test_tail_verify_eos_in_emitted_terminates(self):
        # eos=999. verifier returns [100, 999, 102, 103, 104];
        # drafts [100, 999, 102, 103] match all → all accepted,
        # and 999 is in emitted → is_terminal=True.
        runner, _, _ = _make_verify_tail_runner(
            sample_script=[42],
            verify_batch_script=[[100, 999, 102, 103, 104]],
            eos_token_id=999,
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
        )
        r2 = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 999, 102, 103],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
            proposed_token_ids=[100, 999, 102, 103],
        )
        assert r2.accepted_count == 4  # all match
        assert r2.is_terminal is True

    def test_tail_verify_max_tokens_cap_terminates(self):
        # max_tokens=3. PREFILL emits token 1. VERIFY emits 2
        # more (accepted_count=1, K=2 drafts, partial accept) →
        # tokens_generated = 1 + 2 = 3 → is_terminal=True.
        runner, _, cache = _make_verify_tail_runner(
            sample_script=[42],
            # Drafts will be [100, 101]. Verifier returns
            # [100, 999, 998] — first matches, then divergence.
            verify_batch_script=[[100, 999, 998]],
            eos_token_id=12345,  # not in emitted
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=3, temperature=0.0),
        )
        r2 = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=3, temperature=0.0),
            proposed_token_ids=[100, 101],
        )
        assert r2.accepted_count == 1
        assert cache.get("req-1").tokens_generated == 3
        assert r2.is_terminal is True

    def test_tail_verify_missing_proposed_raises(self):
        runner, _, _ = _make_verify_tail_runner(
            sample_script=[42],
            verify_batch_script=[[1, 2, 3]],
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
        )
        with pytest.raises(
            RuntimeError, match="proposed_token_ids"
        ):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[42, 100, 101],
                request_id="req-1",
                decode_mode=DecodeMode.VERIFY,
                is_final_stage=True,
                request=_FakeRequest(
                    max_tokens=100, temperature=0.0,
                ),
                # proposed_token_ids omitted
            )

    def test_tail_verify_proposed_length_mismatch_raises(self):
        # Verifier returns 5 tokens (K+1=5 → K=4) but
        # proposed_token_ids is length 3. Length mismatch is
        # caller bug.
        runner, _, _ = _make_verify_tail_runner(
            sample_script=[42],
            verify_batch_script=[[100, 101, 102, 103, 104]],
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
        )
        with pytest.raises(
            RuntimeError, match="K\\+1"
        ):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[42, 100, 101, 102, 103],
                request_id="req-1",
                decode_mode=DecodeMode.VERIFY,
                is_final_stage=True,
                request=_FakeRequest(
                    max_tokens=100, temperature=0.0,
                ),
                proposed_token_ids=[100, 101, 102],  # wrong K
            )

    def test_tail_verify_rejects_bool_in_proposed(self):
        runner, _, _ = _make_verify_tail_runner(
            sample_script=[42],
            verify_batch_script=[[100, 101]],
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
        )
        with pytest.raises(RuntimeError, match="proposed_token_ids"):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[42, 100],
                request_id="req-1",
                decode_mode=DecodeMode.VERIFY,
                is_final_stage=True,
                request=_FakeRequest(
                    max_tokens=100, temperature=0.0,
                ),
                proposed_token_ids=[True],  # type: ignore[list-item]
            )

    def test_tail_verify_no_eos_token_id_only_max_tokens_terminates(self):
        # eos_token_id=None — emitted tokens that look like EOS
        # don't trigger termination; only max_tokens does.
        runner, _, cache = _make_verify_tail_runner(
            sample_script=[42],
            verify_batch_script=[[999, 998, 997]],
            eos_token_id=None,
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
        )
        r2 = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
            proposed_token_ids=[100, 101],
        )
        # 0 accepted; emitted = (999,); 999 not treated as EOS.
        assert r2.accepted_count == 0
        assert r2.next_token_id == 999
        assert r2.is_terminal is False
        assert cache.get("req-1").tokens_generated == 2

    def test_tail_verify_request_temperature_override(self):
        # Default temperature is 0.0 in _make_verify_tail_runner;
        # request supplies 0.0 explicitly. Both reach the model
        # via the batch sampler — NOT the per-token sampler.
        runner, model, _ = _make_verify_tail_runner(
            sample_script=[42],
            verify_batch_script=[[100, 101]],
            sampling_defaults=SamplingDefaults(
                max_tokens=100, temperature=1.0,
                top_k=40, top_p=0.9,
            ),
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(temperature=0.0),
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(temperature=0.0),
            proposed_token_ids=[100],
        )
        # request.temperature override reaches batch sampler.
        assert model.batch_sample_calls[0]["temperature"] == 0.0
        assert model.batch_sample_calls[0]["top_k"] == 40
        assert model.batch_sample_calls[0]["top_p"] == 0.9


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.y.x Task 4 — v2 stochastic VERIFY routing
# ──────────────────────────────────────────────────────────────────────────


class _FakeV2VerifyTailShardedModel(_FakeVerifyTailShardedModel):
    """Tail-capable v2 VERIFY model. Adds
    ``apply_lm_head_and_sample_batch_with_rejection`` driven by a
    scripted ``(verified_token_ids, accepted_count)`` per round.
    Tests inject scripts that exercise specific accept/reject
    paths deterministically.
    """

    def __init__(
        self,
        *,
        sample_script: List[int],
        verify_batch_script: List[List[int]],
        rejection_script: Optional[
            List[Tuple[List[int], int]]
        ] = None,
    ) -> None:
        super().__init__(
            sample_script=sample_script,
            verify_batch_script=verify_batch_script,
        )
        self._rejection_script = list(rejection_script or [])
        self._rejection_cursor = 0
        self.rejection_calls: List[dict] = []

    def apply_lm_head_and_sample_batch_with_rejection(
        self,
        *,
        hidden_state_batch: np.ndarray,
        proposed_token_ids: List[int],
        proposed_token_probs: List[float],
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[List[int], int]:
        self.rejection_calls.append({
            "hidden_state_batch": hidden_state_batch,
            "proposed_token_ids": list(proposed_token_ids),
            "proposed_token_probs": list(proposed_token_probs),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })
        if self._rejection_cursor >= len(self._rejection_script):
            raise AssertionError(
                "_FakeV2VerifyTailShardedModel: rejection_script "
                "exhausted (test bug)"
            )
        out = self._rejection_script[self._rejection_cursor]
        self._rejection_cursor += 1
        ids, ac = out
        return list(ids), int(ac)


def _make_v2_tail_runner(
    *,
    rejection_script: List[Tuple[List[int], int]],
    verify_batch_script: List[List[int]] = None,
    sample_script: List[int] = None,
    layer_range: Tuple[int, int] = (0, 6),
    eos_token_id: int = 999,
    sampling_defaults: SamplingDefaults = None,
    constant_k_commitment: bool = False,
) -> Tuple[
    ShardedAutoregressiveRunner,
    _FakeV2VerifyTailShardedModel,
    KVCacheManager,
]:
    model = _FakeV2VerifyTailShardedModel(
        sample_script=sample_script or [42],
        verify_batch_script=verify_batch_script or [],
        rejection_script=rejection_script,
    )
    cache = KVCacheManager()
    runner = ShardedAutoregressiveRunner(
        model=model,
        layer_range=layer_range,
        kv_cache_manager=cache,
        tee_attestation=b"x" * 32,
        tee_type=TEEType.SOFTWARE,
        sampling_defaults=sampling_defaults or SamplingDefaults(
            max_tokens=512, temperature=0.7, top_k=50, top_p=0.95,
        ),
        eos_token_id=eos_token_id,
        constant_k_commitment=constant_k_commitment,
    )
    return runner, model, cache


class TestV2StochasticVerifyRouting:
    """Phase 3.x.11.y.x Task 4 — v2 stochastic routing tests.

    Routing rule: probs set + temperature > 0 → v2 stochastic
    (Leviathan-2023). Either alone falls back to v1 greedy.
    """

    def test_v1_greedy_path_unchanged_when_probs_unset(self):
        # No probs → v1 greedy (apply_lm_head_and_sample_batch)
        # regardless of temperature value. K+1 verified tokens.
        runner, model, _ = _make_v2_tail_runner(
            rejection_script=[],   # never invoked
            verify_batch_script=[[10, 20, 30]],  # K+1 = 3 → K=2
            sample_script=[42],
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.7),
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.7),
            proposed_token_ids=[100, 101],
            # NOTE: no proposed_token_probs → v1 greedy path
        )
        # v1 greedy: K+1=3 verified tokens.
        assert result.verified_token_ids == (10, 20, 30)
        assert result.accepted_count == 0  # 10 != 100
        # v1 path used apply_lm_head_and_sample_batch, NOT
        # apply_lm_head_and_sample_batch_with_rejection.
        assert len(model.batch_sample_calls) == 1
        assert len(model.rejection_calls) == 0

    def test_v2_stochastic_path_when_probs_and_temperature_set(self):
        # probs set + temperature > 0 → v2 stochastic. Tail
        # returns (ids, accepted_count) directly.
        runner, model, _ = _make_v2_tail_runner(
            rejection_script=[
                # K=2 partial accept: accept 1, then correction.
                ([100, 999], 1),
            ],
            sample_script=[42],
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.7),
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.7),
            proposed_token_ids=[100, 101],
            proposed_token_probs=[0.6, 0.5],
        )
        # v2: verified has accepted_count + 1 = 2 entries.
        assert result.verified_token_ids == (100, 999)
        assert result.accepted_count == 1
        # v2 path used apply_lm_head_and_sample_batch_with_rejection.
        assert len(model.rejection_calls) == 1
        assert len(model.batch_sample_calls) == 0
        # Probs threaded through to the model.
        assert model.rejection_calls[0]["proposed_token_probs"] == [0.6, 0.5]

    def test_v1_path_when_probs_set_but_temperature_zero(self):
        # Probs set but temperature == 0 → falls back to v1
        # greedy. Defends against legacy callers that send probs
        # for forward-compat but use temp=0 (degenerate).
        runner, model, _ = _make_v2_tail_runner(
            rejection_script=[],
            verify_batch_script=[[100, 200, 300]],
            sample_script=[42],
            sampling_defaults=SamplingDefaults(
                max_tokens=100, temperature=0.0,  # default temp=0
                top_k=50, top_p=0.95,
            ),
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.0),
            proposed_token_ids=[100, 101],
            proposed_token_probs=[0.5, 0.5],   # set but ignored
        )
        # v1 path fired (temperature=0 forces greedy).
        assert len(model.batch_sample_calls) == 1
        assert len(model.rejection_calls) == 0
        # K+1 = 3 verified entries.
        assert len(result.verified_token_ids) == 3

    def test_v2_full_accept_K_plus_one_emit(self):
        # All K accepted → emit accepted_count + 1 = K + 1 tokens
        # (last is the bonus from p_K).
        runner, model, _ = _make_v2_tail_runner(
            rejection_script=[
                ([100, 101, 99], 2),  # K=2 all accepted + bonus 99
            ],
            sample_script=[42],
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.7),
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.7),
            proposed_token_ids=[100, 101],
            proposed_token_probs=[0.6, 0.6],
        )
        assert result.verified_token_ids == (100, 101, 99)
        assert result.accepted_count == 2
        assert result.next_token_id == 99  # last emitted (bonus)

    def test_v2_zero_accept_emits_one_correction(self):
        # accepted_count=0 → emit 1 token (correction).
        runner, _, _ = _make_v2_tail_runner(
            rejection_script=[
                ([777], 0),  # K=2; correction = 777 at position 0.
            ],
            sample_script=[42],
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.7),
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.7),
            proposed_token_ids=[100, 101],
            proposed_token_probs=[0.5, 0.5],
        )
        assert result.verified_token_ids == (777,)
        assert result.accepted_count == 0
        assert result.next_token_id == 777

    def test_v2_path_missing_method_raises_missing_verify_capability(self):
        # Model lacks apply_lm_head_and_sample_batch_with_rejection;
        # v2 dispatch should map to MissingVerifyCapabilityError.
        # Use the v1-only fake (no rejection method).
        runner, _, _ = _make_verify_tail_runner(
            sample_script=[42],
            verify_batch_script=[[100, 101]],
            sampling_defaults=SamplingDefaults(
                max_tokens=100, temperature=0.7,
                top_k=50, top_p=0.95,
            ),
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.7),
        )
        with pytest.raises(
            MissingVerifyCapabilityError,
            match="with_rejection",
        ):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[42, 100],
                request_id="req-1",
                decode_mode=DecodeMode.VERIFY,
                is_final_stage=True,
                request=_FakeRequest(
                    max_tokens=100, temperature=0.7,
                ),
                proposed_token_ids=[100],
                proposed_token_probs=[0.5],
            )

    def test_probs_without_ids_rejected(self):
        # proposed_token_probs without proposed_token_ids = caller bug.
        runner, _, _ = _make_v2_tail_runner(
            rejection_script=[],
            verify_batch_script=[[100]],
        )
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.7),
        )
        with pytest.raises(
            RuntimeError, match="proposed_token_probs",
        ):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[42, 100],
                request_id="req-1",
                decode_mode=DecodeMode.VERIFY,
                is_final_stage=True,
                request=_FakeRequest(temperature=0.7),
                proposed_token_probs=[0.5],
                # ids missing
            )

    def test_probs_length_mismatch_rejected(self):
        runner, _, _ = _make_v2_tail_runner(rejection_script=[])
        runner.run_layer_slice_unary(
            activation_or_input_ids=[1, 2, 3],
            request_id="req-1",
            decode_mode=DecodeMode.PREFILL,
            is_final_stage=True,
            request=_FakeRequest(max_tokens=100, temperature=0.7),
        )
        with pytest.raises(
            RuntimeError, match="length",
        ):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[42, 100, 101],
                request_id="req-1",
                decode_mode=DecodeMode.VERIFY,
                is_final_stage=True,
                request=_FakeRequest(temperature=0.7),
                proposed_token_ids=[100, 101],
                proposed_token_probs=[0.5],   # |probs|=1 vs |ids|=2
            )


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.q.y Task 3 — constant-K commitment
# ──────────────────────────────────────────────────────────────────────────


class TestConstantKCommitment:
    """Phase 3.x.11.q.y constant-K commitment under v2 stochastic
    VERIFY. The runner pads verified_token_ids up to K+1 by
    greedy-continuing from the §2.2 corrected position; the
    wire-side accepted_count is fixed to K regardless of actual
    acceptance.
    """

    def test_constant_k_pads_partial_accept_to_k_plus_one(self):
        # K=4, actual ac=1 (rejection at position 1). Without
        # constant-K, runner returns (verified[:2], 1). Under
        # constant-K, runner pads with v1-greedy argmaxes for
        # positions 2..K and returns (K+1 entries, K).
        K = 4
        # Rejection helper returns ac=1 with 2 entries:
        #   [accepted_d_0, correction_at_pos_1]
        rejection_script = [([100, 999], 1)]
        # v1 batched argmax helper returns K+1=5 deterministic
        # argmaxes (verify_batch_script).
        verify_batch_script = [[100, 999, 200, 201, 202]]
        runner, model, _ = _make_v2_tail_runner(
            rejection_script=rejection_script,
            verify_batch_script=verify_batch_script,
            constant_k_commitment=True,
        )
        # Seed the cache as if PREFILL ran.
        from prsm.compute.chain_rpc.kv_cache import KVCacheHandle
        runner._cache._handles["req-1"] = KVCacheHandle(
            request_id="req-1",
            payload=[], n_layers=6,
            cached_positions=3,
            tokens_generated=0,
            last_touch_time=1000.0,
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101, 102, 103],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(temperature=0.7),
            proposed_token_ids=[100, 101, 102, 103],
            proposed_token_probs=[0.9, 0.5, 0.5, 0.5],
        )
        # accepted_count is K=4 on the wire (constant-byte).
        assert result.accepted_count == K
        # verified_token_ids has K+1=5 entries (constant-byte).
        assert len(result.verified_token_ids) == K + 1
        # Position 0 is the §2.2 accepted draft; position 1 is the
        # §2.2 correction (both from rejection helper). Positions
        # 2..K are v1 greedy argmaxes (200, 201, 202).
        assert result.verified_token_ids[0] == 100  # accepted
        assert result.verified_token_ids[1] == 999  # §2.2 correction
        assert result.verified_token_ids[2] == 200  # greedy pad
        assert result.verified_token_ids[3] == 201  # greedy pad
        assert result.verified_token_ids[4] == 202  # greedy pad

    def test_constant_k_no_op_when_actual_ac_equals_k(self):
        # When the rejection helper already returned ac=K (full
        # accept), no padding is needed. Output is unchanged.
        K = 3
        rejection_script = [([10, 20, 30, 999], K)]
        runner, model, _ = _make_v2_tail_runner(
            rejection_script=rejection_script,
            verify_batch_script=[[10, 20, 30, 999]],
            constant_k_commitment=True,
        )
        from prsm.compute.chain_rpc.kv_cache import KVCacheHandle
        runner._cache._handles["req-1"] = KVCacheHandle(
            request_id="req-1",
            payload=[], n_layers=6,
            cached_positions=2,
            tokens_generated=0,
            last_touch_time=1000.0,
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 10, 20, 30],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(temperature=0.7),
            proposed_token_ids=[10, 20, 30],
            proposed_token_probs=[0.9, 0.9, 0.9],
        )
        assert result.accepted_count == K
        assert len(result.verified_token_ids) == K + 1
        assert result.verified_token_ids == (10, 20, 30, 999)

    def test_constant_k_zero_accept_pads_full_k(self):
        # K=2, ac=0 (rejection at position 0). Padding fills all
        # K positions after the §2.2 correction.
        K = 2
        rejection_script = [([777], 0)]  # only correction
        verify_batch_script = [[42, 100, 200]]  # K+1=3 argmaxes
        runner, model, _ = _make_v2_tail_runner(
            rejection_script=rejection_script,
            verify_batch_script=verify_batch_script,
            constant_k_commitment=True,
        )
        from prsm.compute.chain_rpc.kv_cache import KVCacheHandle
        runner._cache._handles["req-1"] = KVCacheHandle(
            request_id="req-1",
            payload=[], n_layers=6,
            cached_positions=2,
            tokens_generated=0,
            last_touch_time=1000.0,
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 5, 6],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(temperature=0.7),
            proposed_token_ids=[5, 6],
            proposed_token_probs=[0.5, 0.5],
        )
        assert result.accepted_count == K
        assert len(result.verified_token_ids) == K + 1
        # Position 0: §2.2 correction (777). Positions 1..K: greedy
        # pad from v1 helper (verify_batch_script[0][1:] = [100, 200]).
        assert result.verified_token_ids[0] == 777
        assert result.verified_token_ids[1] == 100
        assert result.verified_token_ids[2] == 200

    def test_constant_k_disabled_returns_natural_ac(self):
        # Backwards-compat: when constant_k_commitment=False
        # (default), runner returns (verified[:ac+1], ac) — the
        # natural narrowed semantic. Tier A/B ungated path.
        K = 4
        rejection_script = [([100, 999], 1)]
        runner, model, _ = _make_v2_tail_runner(
            rejection_script=rejection_script,
            verify_batch_script=[[100, 999, 200, 201, 202]],
            constant_k_commitment=False,  # default
        )
        from prsm.compute.chain_rpc.kv_cache import KVCacheHandle
        runner._cache._handles["req-1"] = KVCacheHandle(
            request_id="req-1",
            payload=[], n_layers=6,
            cached_positions=3,
            tokens_generated=0,
            last_touch_time=1000.0,
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 100, 101, 102, 103],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(temperature=0.7),
            proposed_token_ids=[100, 101, 102, 103],
            proposed_token_probs=[0.9, 0.5, 0.5, 0.5],
        )
        # Natural ac=1, len=2. NOT padded.
        assert result.accepted_count == 1
        assert len(result.verified_token_ids) == 2

    def test_constant_k_does_not_affect_v1_greedy_path(self):
        # Sanity: constant-K commitment is gated on v2 stochastic
        # (probs set + temperature > 0). v1 greedy path
        # (proposed_token_probs is None) returns K+1 entries
        # naturally; constant-K is a no-op even when flag is set.
        K = 3
        runner, model, _ = _make_v2_tail_runner(
            rejection_script=[],  # not used in v1
            verify_batch_script=[[10, 20, 30, 999]],
            constant_k_commitment=True,
        )
        from prsm.compute.chain_rpc.kv_cache import KVCacheHandle
        runner._cache._handles["req-1"] = KVCacheHandle(
            request_id="req-1",
            payload=[], n_layers=6,
            cached_positions=2,
            tokens_generated=0,
            last_touch_time=1000.0,
        )
        result = runner.run_layer_slice_unary(
            activation_or_input_ids=[42, 10, 20, 30],
            request_id="req-1",
            decode_mode=DecodeMode.VERIFY,
            is_final_stage=True,
            request=_FakeRequest(temperature=0.0),  # v1 greedy
            proposed_token_ids=[10, 20, 30],
            proposed_token_probs=None,  # v1 path
        )
        # v1 returns K+1 + accepted_count = longest-prefix-match.
        # For (proposed=10,20,30) vs (verifier_argmax=10,20,30,999),
        # all 3 prefix-match → ac=3=K.
        assert result.accepted_count == K
        assert len(result.verified_token_ids) == K + 1

    def test_constant_k_post_condition_invariant(self):
        # Defense-in-depth: the runner asserts len == K+1 after
        # padding. A buggy v1 helper that returned the wrong
        # length should surface as a clear RuntimeError, not
        # silently produce wrong-length output.
        rejection_script = [([100, 999], 1)]
        # v1 helper returns wrong-length list (K+1=4 expected, 3 returned)
        verify_batch_script = [[100, 999, 200]]  # only 3 entries
        runner, model, _ = _make_v2_tail_runner(
            rejection_script=rejection_script,
            verify_batch_script=verify_batch_script,
            constant_k_commitment=True,
        )
        from prsm.compute.chain_rpc.kv_cache import KVCacheHandle
        runner._cache._handles["req-1"] = KVCacheHandle(
            request_id="req-1",
            payload=[], n_layers=6,
            cached_positions=3,
            tokens_generated=0,
            last_touch_time=1000.0,
        )
        with pytest.raises(
            RuntimeError, match="must return list of K\\+1",
        ):
            runner.run_layer_slice_unary(
                activation_or_input_ids=[42, 100, 101, 102],
                request_id="req-1",
                decode_mode=DecodeMode.VERIFY,
                is_final_stage=True,
                request=_FakeRequest(temperature=0.7),
                proposed_token_ids=[100, 101, 102],
                proposed_token_probs=[0.9, 0.5, 0.5],
            )
