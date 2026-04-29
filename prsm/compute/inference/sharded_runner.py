"""Phase 3.x.11 Task 3 — ``ShardedAutoregressiveRunner`` (non-tail variant).

Sibling to ``AutoregressiveStreamingRunner`` (Phase 3.x.10) that
runs ONLY its assigned ``layer_range`` with KV-cache management.
Each per-token wire dispatch is a unary
``RunLayerSliceRequest`` → ``RunLayerSliceResponse`` cycle; the
executor's per-token chain loop (Phase 3.x.11 Task 5) drives the
full chain forward each token.

Two dispatch modes:
  - **prefill**: full ``input_ids`` (Stage 1) or full
    activation tensor (Stage > 1) forward through ``layer_range``;
    allocates a cache handle via the manager; emits hidden state
    at the layer-range exit boundary.
  - **incremental**: single-position forward with cached KV; uses
    the prior PREFILL's payload from the manager; emits
    single-position hidden state and updates the handle's
    payload (the model may mutate the payload in place + return
    the same reference).

**Tail variant deferred to Task 4.** v1 non-tail rejects
``is_final_stage=True`` at the dispatch boundary so the existing
tail-only ``AutoregressiveStreamingRunner`` stays the load-bearing
tail path until Task 4's LM-head + sampling + EOS / max_tokens
detection wires in.

Model abstraction (``ShardedLayerForward`` Protocol): the runner
depends on a duck-typed model that exposes ``forward_prefill`` +
``forward_incremental``. Production wiring (deferred to Task 5/6
+ a follow-up factory) adapts a real HF transformer block range
to that Protocol; tests inject deterministic fakes. Decoupling
the runner from any specific model layout means the runner stays
testable without HF + torch in the inner loop, and a future
non-HF model (e.g., GGUF/llama.cpp adapter) drops in by
implementing the same Protocol.

Cache lifecycle ownership:
  - The MANAGER (``KVCacheManager``, Phase 3.x.11 Task 2) owns
    the handle's lifecycle (allocate / get / evict + LRU + TTL).
  - The MODEL owns the payload shape (per-layer K + V tensors).
  - The RUNNER mediates: it tells the manager when to allocate,
    drives the model's forward, and stores the model's returned
    payload on the handle.
  - On an INCREMENTAL with no prior PREFILL handle, the runner
    raises ``MalformedCacheStateError`` (caller bug or cache
    evicted out from under the request — at the wire layer this
    maps to ``MALFORMED_REQUEST``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Tuple, Union

import numpy as np

from prsm.compute.chain_rpc.kv_cache import KVCacheManager
from prsm.compute.chain_rpc.protocol import (
    MAX_VERIFY_BATCH_TOKENS,
    DecodeMode,
)
from prsm.compute.inference.autoregressive_runner import SamplingDefaults
from prsm.compute.tee.models import TEEType


__all__ = [
    "LayerSliceResult",
    "MalformedCacheStateError",
    "MissingTailCapabilityError",
    "MissingVerifyCapabilityError",
    "ShardedAutoregressiveRunner",
    "ShardedLayerForward",
]


class MalformedCacheStateError(RuntimeError):
    """Raised on INCREMENTAL dispatch with no prior PREFILL cache.
    Maps to ``MALFORMED_REQUEST`` at the wire layer (the caller —
    typically the executor's per-token chain loop — must restart
    with a fresh PREFILL or surface the error to the user)."""


class MissingTailCapabilityError(RuntimeError):
    """Raised on tail dispatch (``is_final_stage=True``) when the
    runner was constructed without tail capability (no
    ``sampling_defaults`` / no ``apply_lm_head_and_sample`` on the
    model). Caller bug — operators must construct the runner with
    tail args set when the stage is the chain tail."""


class MissingVerifyCapabilityError(RuntimeError):
    """Raised on ``decode_mode == VERIFY`` dispatch when the runner's
    model does not implement ``forward_verify`` (or, for tail
    dispatch, does not also implement ``apply_lm_head_and_sample_batch``).
    Caller bug — operators must wire a verify-capable model when
    serving speculative-decoding chains. Maps to
    ``MALFORMED_REQUEST`` at the wire layer."""


# ──────────────────────────────────────────────────────────────────────────
# ShardedLayerForward — duck-typed model Protocol
# ──────────────────────────────────────────────────────────────────────────


class ShardedLayerForward(Protocol):
    """Duck-typed Protocol for the per-stage layer-range forward.

    Production wiring (Task 5/6 follow-up) wraps a real HF
    transformer-block range; tests inject deterministic fakes.
    The KV-cache payload is opaque to the runner — whatever shape
    the model produces (typically a list-of-(K,V)-tensor-pairs
    indexed by layer).
    """

    def forward_prefill(
        self,
        *,
        input_or_hidden: Any,
        layer_range: Tuple[int, int],
    ) -> Tuple[np.ndarray, Any]:
        """Run full forward through ``layer_range``. ``input_or_hidden``
        is ``List[int]`` (input_ids) for Stage 1 or
        ``np.ndarray`` (hidden state from prior stage) for Stage > 1.
        Returns ``(hidden_at_exit, kv_cache_payload)``."""
        ...

    def forward_incremental(
        self,
        *,
        input_or_hidden: Any,
        layer_range: Tuple[int, int],
        kv_cache_payload: Any,
    ) -> Tuple[np.ndarray, Any]:
        """Single-position forward with cached KV. Returns
        ``(hidden_at_exit, updated_kv_cache_payload)``. The
        returned payload may be the same reference as the input,
        mutated in place — the runner just stores whatever comes
        back."""
        ...

    def apply_lm_head_and_sample(
        self,
        *,
        hidden_state: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        """Tail-only — project hidden state through the LM head
        to vocab logits + sample the next token id per sampling
        params. ``temperature == 0`` triggers greedy
        (``argmax``); positive triggers temperature-scaled
        softmax with optional top-k / top-p filtering. Returns
        the sampled token id.

        Non-tail stage models can omit this method; the runner
        guards it at construction time when tail capability is
        opted-in."""
        ...

    def forward_verify(
        self,
        *,
        input_or_hidden: Any,
        layer_range: Tuple[int, int],
        kv_cache_payload: Any,
    ) -> Tuple[np.ndarray, Any]:
        """Phase 3.x.11.y — speculative-decoding VERIFY forward.
        Batched K+1-position forward through ``layer_range`` with
        cached KV (the cache covers positions before the K+1
        speculative tokens). ``input_or_hidden`` is
        ``List[int]`` of length K+1 for Stage 1 or
        ``np.ndarray`` shaped ``[K+1, hidden]`` (or
        ``[1, K+1, hidden]``) for Stage > 1. Returns
        ``(hidden_at_exit, updated_kv_cache_payload)`` where
        ``hidden_at_exit`` is shaped ``[K+1, hidden]`` (one
        position per input). Cache is mutated to cover all K+1
        new positions; the caller (executor) issues
        ``RollbackCacheRequest`` to drop the rejected suffix.

        Models that don't support speculation can omit this
        method; the runner guards it at dispatch time per
        decode_mode."""
        ...

    def apply_lm_head_and_sample_batch(
        self,
        *,
        hidden_state_batch: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> List[int]:
        """Phase 3.x.11.y — tail-only — project K+1 hidden states
        through the LM head + sample one token per position.
        ``hidden_state_batch`` shape is ``[K+1, hidden]``;
        returns ``[K+1]`` token ids (one per position;
        ``verified[i]`` is the model's argmax/sample for the
        token following input position i). Greedy under
        ``temperature == 0`` (used by v1 — sampling-correct
        speculation under temperature > 0 requires the
        Leviathan-2023 correction, deferred to Phase 3.x.11.y.x).

        Non-tail / non-speculation models can omit this method;
        the runner guards it at dispatch time per
        decode_mode/is_final_stage."""
        ...

    def truncate_cache(
        self,
        payload: Any,
        n_positions: int,
    ) -> Any:
        """Phase 3.x.11.y — drop the LAST ``n_positions`` from the
        KV-cache payload and return the updated payload. Called
        under ``KVCacheManager``'s lock during rollback (the
        executor's RollbackCacheRequest broadcast routes here via
        ``ShardedAutoregressiveRunner.rollback_cache``).

        ``n_positions > 0`` is guaranteed; the manager passes the
        already-clamped count (capped at the cache's
        ``tokens_generated``). The model is free to mutate the
        payload in place + return the same reference, or return
        a fresh payload — the manager just stores what comes
        back.

        Models that don't support speculation can omit this
        method; the runner guards it at dispatch time per
        decode_mode/operation."""
        ...


# ──────────────────────────────────────────────────────────────────────────
# LayerSliceResult — runner return type
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerSliceResult:
    """Return value of ``run_layer_slice_unary``.

    The caller (server's unary handler) maps these into a
    ``RunLayerSliceResponse`` — ``hidden_state`` becomes the wire
    activation; ``next_token_id`` + ``is_terminal`` populate
    tail-only fields (None / False on non-tail responses);
    ``verified_token_ids`` + ``accepted_count`` populate
    Phase 3.x.11.y speculative-decoding tail-only fields (None on
    PREFILL / INCREMENTAL / non-tail VERIFY).

    ``n_layers_run`` is the size of the layer range this dispatch
    actually executed; useful for receipt + metrics.
    """

    hidden_state: np.ndarray
    duration_seconds: float
    decode_mode: DecodeMode
    n_layers_run: int
    next_token_id: Optional[int] = None
    is_terminal: bool = False
    verified_token_ids: Optional[Tuple[int, ...]] = None
    accepted_count: Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────
# ShardedAutoregressiveRunner
# ──────────────────────────────────────────────────────────────────────────


class ShardedAutoregressiveRunner:
    """Per-stage layer-range runner with KV-cache lifecycle.

    Constructor args:
      model              ``ShardedLayerForward``-shaped object.
      layer_range        ``(start, end_exclusive)`` — half-open.
      kv_cache_manager   Per-server cache lifecycle
                         (``KVCacheManager``, Task 2).
      tee_attestation    Bytes the stage signs over (TEE-bound
                         identity); same shape as the streaming
                         runner.
      tee_type           ``TEEType``.

    Tail capability is opt-in at construction. When the stage is
    the chain tail (``is_final_stage=True`` at dispatch), the
    runner additionally:
      - Calls ``model.apply_lm_head_and_sample`` to project the
        boundary hidden state through the LM head + sample the
        next token id per the request's sampling params.
      - Tracks ``tokens_generated`` on the cache handle to detect
        the ``max_tokens`` cap.
      - Sets ``is_terminal=True`` on the result when EOS is
        sampled OR ``tokens_generated >= max_tokens``.

    Tail capability requires ``sampling_defaults`` AND a model
    with ``apply_lm_head_and_sample``. ``eos_token_id`` is
    optional — if ``None``, EOS detection is disabled and only
    ``max_tokens`` triggers termination. Operators wire
    ``eos_token_id`` from the tokenizer at the factory layer.
    """

    def __init__(
        self,
        *,
        model: ShardedLayerForward,
        layer_range: Tuple[int, int],
        kv_cache_manager: KVCacheManager,
        tee_attestation: bytes,
        tee_type: TEEType,
        sampling_defaults: Optional[SamplingDefaults] = None,
        eos_token_id: Optional[int] = None,
    ) -> None:
        if model is None:
            raise RuntimeError(
                "ShardedAutoregressiveRunner: model is required "
                "(ShardedLayerForward Protocol)"
            )
        for attr in ("forward_prefill", "forward_incremental"):
            if not callable(getattr(model, attr, None)):
                raise RuntimeError(
                    f"ShardedAutoregressiveRunner: model must "
                    f"implement {attr}(...) per ShardedLayerForward "
                    f"Protocol"
                )
        if (
            not isinstance(layer_range, tuple)
            or len(layer_range) != 2
        ):
            raise RuntimeError(
                "ShardedAutoregressiveRunner: layer_range must be a "
                "(start, end_exclusive) tuple"
            )
        start, end = layer_range
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise RuntimeError(
                f"ShardedAutoregressiveRunner: invalid layer_range "
                f"{layer_range!r} — must be 0 <= start < end (ints)"
            )
        if not isinstance(kv_cache_manager, KVCacheManager):
            raise RuntimeError(
                "ShardedAutoregressiveRunner: kv_cache_manager must "
                "be a KVCacheManager instance"
            )
        if not isinstance(tee_attestation, (bytes, bytearray)):
            raise RuntimeError(
                "ShardedAutoregressiveRunner: tee_attestation must "
                "be bytes"
            )
        if not isinstance(tee_type, TEEType):
            raise RuntimeError(
                f"ShardedAutoregressiveRunner: tee_type must be "
                f"TEEType, got {type(tee_type).__name__}"
            )
        # Tail capability validation. Either both are wired (full
        # tail-capable) or neither is (non-tail-only). A
        # half-wired runner is a configuration bug.
        if sampling_defaults is not None and not isinstance(
            sampling_defaults, SamplingDefaults,
        ):
            raise RuntimeError(
                "ShardedAutoregressiveRunner: sampling_defaults "
                "must be a SamplingDefaults instance or None"
            )
        if eos_token_id is not None and (
            isinstance(eos_token_id, bool)
            or not isinstance(eos_token_id, int)
            or eos_token_id < 0
        ):
            raise RuntimeError(
                f"ShardedAutoregressiveRunner: eos_token_id must "
                f"be a non-negative int or None, got "
                f"{eos_token_id!r}"
            )
        # Tail capability requires the model to expose
        # ``apply_lm_head_and_sample``. Non-tail-only construction
        # (``sampling_defaults=None``) skips the check — the model
        # is allowed to omit the method.
        self._tail_capable = sampling_defaults is not None
        if self._tail_capable and not callable(
            getattr(model, "apply_lm_head_and_sample", None),
        ):
            raise RuntimeError(
                "ShardedAutoregressiveRunner: tail-capable "
                "construction (sampling_defaults set) requires "
                "model.apply_lm_head_and_sample(...) per "
                "ShardedLayerForward Protocol"
            )

        self._model = model
        self._layer_range = (int(start), int(end))
        self._n_layers = int(end - start)
        self._cache = kv_cache_manager
        self._tee_attestation = bytes(tee_attestation)
        self._tee_type = tee_type
        self._sampling_defaults = sampling_defaults
        self._eos_token_id = eos_token_id

    # ── public read-only accessors ────────────────────────────────────

    @property
    def layer_range(self) -> Tuple[int, int]:
        return self._layer_range

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def tee_attestation(self) -> bytes:
        return self._tee_attestation

    @property
    def tee_type(self) -> TEEType:
        return self._tee_type

    # ── unary dispatch ────────────────────────────────────────────────

    def run_layer_slice_unary(
        self,
        *,
        activation_or_input_ids: Union[np.ndarray, List[int]],
        request_id: str,
        decode_mode: DecodeMode,
        is_final_stage: bool = False,
        request: Any = None,
        proposed_token_ids: Optional[List[int]] = None,
    ) -> LayerSliceResult:
        """Dispatch one PREFILL / INCREMENTAL / VERIFY forward
        through ``layer_range``.

        PREFILL:
          - Allocate fresh cache handle via the manager (LRU-evicts
            oldest if at ``max_cached_requests`` cap).
          - Drive ``model.forward_prefill``.
          - Store the returned KV-cache payload on the handle.
          - Return the boundary hidden state.

        INCREMENTAL:
          - Look up the existing handle via the manager.
          - Raise ``MalformedCacheStateError`` if no handle exists
            (caller bug or cache evicted out from under request).
          - Drive ``model.forward_incremental(kv_cache_payload=...)``.
          - Update the handle's payload (model may mutate in place
            + return the same reference; runner stores what comes
            back).
          - Return the single-position hidden state.

        VERIFY (Phase 3.x.11.y — speculative decoding):
          - Look up the existing handle (must exist; raises
            ``MalformedCacheStateError`` if absent — VERIFY is
            never the first dispatch).
          - Validate the K+1-position input (Stage 1: list of
            K+1 ints; Stage > 1: ndarray with K+1 batch dim).
            Cap K+1 at ``MAX_VERIFY_BATCH_TOKENS`` (defends
            against malformed peer claiming a huge speculation
            depth that explodes server-side memory).
          - Drive ``model.forward_verify(kv_cache_payload=...)``;
            cache is extended with K+1 new positions (executor
            issues ``RollbackCacheRequest`` afterward to drop
            the rejected suffix).
          - If tail: sample K+1 logits via
            ``model.apply_lm_head_and_sample_batch``; compute
            ``accepted_count`` as the longest matching prefix
            between ``proposed_token_ids`` and the verified
            argmaxes; bump ``handle.tokens_generated`` by
            ``accepted_count + 1`` (only emitted tokens count
            against ``max_tokens``); set ``next_token_id`` to
            the LAST emitted (``verified_token_ids[accepted_count]``).
          - Tail-stage termination: ``is_terminal=True`` if any
            emitted token equals ``eos_token_id`` OR
            ``handle.tokens_generated >= max_tokens``.
        """
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError(
                "ShardedAutoregressiveRunner.run_layer_slice_unary: "
                "request_id must be a non-empty string"
            )
        if not isinstance(decode_mode, DecodeMode):
            raise RuntimeError(
                f"ShardedAutoregressiveRunner.run_layer_slice_unary: "
                f"decode_mode must be DecodeMode, got "
                f"{type(decode_mode).__name__}"
            )
        if isinstance(is_final_stage, bool) is False:
            raise RuntimeError(
                f"ShardedAutoregressiveRunner.run_layer_slice_unary: "
                f"is_final_stage must be bool, got "
                f"{type(is_final_stage).__name__}"
            )
        if is_final_stage and not self._tail_capable:
            raise MissingTailCapabilityError(
                "ShardedAutoregressiveRunner: is_final_stage=True "
                "requires tail-capable construction "
                "(sampling_defaults + model.apply_lm_head_and_sample) "
                "— this runner was constructed non-tail-only"
            )
        # ``proposed_token_ids`` is meaningful only on tail VERIFY
        # dispatch; reject as caller bug if set on any other path.
        if proposed_token_ids is not None and decode_mode != DecodeMode.VERIFY:
            raise RuntimeError(
                "ShardedAutoregressiveRunner.run_layer_slice_unary: "
                "proposed_token_ids is only meaningful for "
                "decode_mode=VERIFY"
            )

        start_ts = time.time()
        if decode_mode == DecodeMode.PREFILL:
            handle = self._cache.allocate(
                request_id, n_layers=self._n_layers,
            )
            hidden, payload = self._model.forward_prefill(
                input_or_hidden=activation_or_input_ids,
                layer_range=self._layer_range,
            )
            handle.payload = payload
        elif decode_mode == DecodeMode.INCREMENTAL:
            handle = self._cache.get(request_id)
            if handle is None:
                raise MalformedCacheStateError(
                    f"ShardedAutoregressiveRunner: incremental "
                    f"dispatch for request_id={request_id!r} but "
                    f"no prior PREFILL cache exists (caller must "
                    f"run PREFILL first or cache was evicted by "
                    f"TTL/LRU)"
                )
            hidden, updated_payload = self._model.forward_incremental(
                input_or_hidden=activation_or_input_ids,
                layer_range=self._layer_range,
                kv_cache_payload=handle.payload,
            )
            handle.payload = updated_payload
        else:
            # VERIFY — Phase 3.x.11.y
            handle = self._cache.get(request_id)
            if handle is None:
                raise MalformedCacheStateError(
                    f"ShardedAutoregressiveRunner: VERIFY dispatch "
                    f"for request_id={request_id!r} but no prior "
                    f"PREFILL cache exists (caller must run PREFILL "
                    f"first or cache was evicted by TTL/LRU)"
                )
            if not callable(getattr(self._model, "forward_verify", None)):
                raise MissingVerifyCapabilityError(
                    "ShardedAutoregressiveRunner: decode_mode=VERIFY "
                    "requires model.forward_verify(...) per "
                    "ShardedLayerForward Protocol — this model omits it"
                )
            n_positions = self._verify_input_n_positions(
                activation_or_input_ids,
            )
            if n_positions < 2:
                raise RuntimeError(
                    f"ShardedAutoregressiveRunner: VERIFY input must "
                    f"contain at least 2 positions (parent + at "
                    f"least one draft), got {n_positions}"
                )
            if n_positions > MAX_VERIFY_BATCH_TOKENS:
                raise RuntimeError(
                    f"ShardedAutoregressiveRunner: VERIFY input "
                    f"length {n_positions} exceeds cap "
                    f"{MAX_VERIFY_BATCH_TOKENS} — defends against "
                    f"malformed peer claiming huge speculation depth"
                )
            hidden, updated_payload = self._model.forward_verify(
                input_or_hidden=activation_or_input_ids,
                layer_range=self._layer_range,
                kv_cache_payload=handle.payload,
            )
            handle.payload = updated_payload

        next_token_id: Optional[int] = None
        is_terminal = False
        verified_token_ids: Optional[Tuple[int, ...]] = None
        accepted_count: Optional[int] = None
        if is_final_stage:
            if decode_mode == DecodeMode.VERIFY:
                (
                    next_token_id,
                    is_terminal,
                    verified_token_ids,
                    accepted_count,
                ) = self._sample_tail_verify(
                    hidden_state_batch=hidden,
                    handle=handle,
                    request=request,
                    proposed_token_ids=proposed_token_ids,
                )
            else:
                next_token_id, is_terminal = self._sample_tail(
                    hidden_state=hidden,
                    handle=handle,
                    request=request,
                )

        duration = time.time() - start_ts
        return LayerSliceResult(
            hidden_state=hidden,
            duration_seconds=duration,
            decode_mode=decode_mode,
            n_layers_run=self._n_layers,
            next_token_id=next_token_id,
            is_terminal=is_terminal,
            verified_token_ids=verified_token_ids,
            accepted_count=accepted_count,
        )

    @staticmethod
    def _verify_input_n_positions(
        activation_or_input_ids: Union[np.ndarray, List[int]],
    ) -> int:
        """Extract K+1 (the position count) from a VERIFY input.
        Stage 1 input is ``List[int]`` of length K+1; Stage > 1
        input is an ndarray whose seq-len axis is K+1 (shape
        ``[K+1, hidden]`` or ``[1, K+1, hidden]``)."""
        if isinstance(activation_or_input_ids, list):
            return len(activation_or_input_ids)
        arr = np.asarray(activation_or_input_ids)
        if arr.ndim < 2:
            raise RuntimeError(
                f"ShardedAutoregressiveRunner: VERIFY ndarray input "
                f"must have ndim >= 2, got shape {arr.shape}"
            )
        # [K+1, hidden] or [1, K+1, hidden] — K+1 is always the
        # second-to-last axis (per the streaming runner convention).
        return int(arr.shape[-2])

    # ── tail-only sampling ────────────────────────────────────────────

    def _sample_tail(
        self,
        *,
        hidden_state: np.ndarray,
        handle: Any,  # KVCacheHandle — Any to keep imports clean
        request: Any,
    ) -> Tuple[int, bool]:
        """Project the boundary hidden state through the LM head,
        sample the next token id, bump the per-request token
        counter on the cache handle, and decide ``is_terminal``
        from EOS detection + ``max_tokens`` cap.

        Returns ``(next_token_id, is_terminal)``. The runner
        guarantees ``self._tail_capable`` before calling — caller
        must check ``is_final_stage`` first.
        """
        defaults = self._sampling_defaults
        assert defaults is not None  # tail_capable guarantees this
        # Resolve sampling params from request, falling back to
        # runner defaults. Mirrors AutoregressiveStreamingRunner's
        # _effective_max_tokens / _effective_temperature pattern.
        rmax = getattr(request, "max_tokens", None) if request is not None else None
        max_tokens = int(rmax) if rmax is not None else defaults.max_tokens
        rtemp = getattr(request, "temperature", None) if request is not None else None
        temperature = float(rtemp) if rtemp is not None else defaults.temperature
        # top_k / top_p are runner-level config (no wire field).
        top_k = defaults.top_k
        top_p = defaults.top_p

        next_token_id = int(self._model.apply_lm_head_and_sample(
            hidden_state=hidden_state,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ))

        # Bump the per-request token counter on the cache handle.
        # PREFILL produces token #1 (the first generated token);
        # subsequent INCREMENTALs produce tokens #2, #3, ...
        handle.tokens_generated += 1

        # Termination: EOS sampled OR token count reached the
        # request-level cap.
        is_terminal = False
        if (
            self._eos_token_id is not None
            and next_token_id == self._eos_token_id
        ):
            is_terminal = True
        if handle.tokens_generated >= max_tokens:
            is_terminal = True

        return next_token_id, is_terminal

    # ── tail-only VERIFY sampling ─────────────────────────────────────

    def _sample_tail_verify(
        self,
        *,
        hidden_state_batch: np.ndarray,
        handle: Any,  # KVCacheHandle
        request: Any,
        proposed_token_ids: Optional[List[int]],
    ) -> Tuple[int, bool, Tuple[int, ...], int]:
        """Tail-only VERIFY sampling. Sample K+1 logits from the
        batched hidden state, compute ``accepted_count`` as the
        longest matching prefix between ``proposed_token_ids``
        and the verified argmaxes, bump
        ``handle.tokens_generated`` by ``accepted_count + 1``
        (only emitted tokens count against ``max_tokens``),
        return the per-iteration outcome.

        Returns
        ``(next_token_id, is_terminal, verified_token_ids, accepted_count)``.
        ``next_token_id`` is the LAST emitted token —
        ``verified_token_ids[accepted_count]`` — the seed for the
        next speculation round. ``is_terminal`` triggers on EOS
        in any emitted position OR
        ``tokens_generated >= max_tokens``.
        """
        defaults = self._sampling_defaults
        assert defaults is not None  # tail_capable guarantees this
        if not callable(
            getattr(self._model, "apply_lm_head_and_sample_batch", None),
        ):
            raise MissingVerifyCapabilityError(
                "ShardedAutoregressiveRunner: tail-stage VERIFY "
                "dispatch requires "
                "model.apply_lm_head_and_sample_batch(...) per "
                "ShardedLayerForward Protocol — this model omits it"
            )
        if proposed_token_ids is None:
            raise RuntimeError(
                "ShardedAutoregressiveRunner: tail-stage VERIFY "
                "dispatch requires proposed_token_ids — caller "
                "(executor's speculation loop) must pass the K "
                "draft tokens for accepted_count comparison"
            )
        if not isinstance(proposed_token_ids, list):
            raise RuntimeError(
                f"ShardedAutoregressiveRunner: proposed_token_ids "
                f"must be list, got {type(proposed_token_ids).__name__}"
            )
        for tok in proposed_token_ids:
            if isinstance(tok, bool) or not isinstance(tok, int):
                raise RuntimeError(
                    f"ShardedAutoregressiveRunner: proposed_token_ids "
                    f"entries must be int, got {type(tok).__name__}"
                )
        # Resolve sampling params (mirrors _sample_tail).
        rmax = getattr(request, "max_tokens", None) if request is not None else None
        max_tokens = int(rmax) if rmax is not None else defaults.max_tokens
        rtemp = getattr(request, "temperature", None) if request is not None else None
        temperature = float(rtemp) if rtemp is not None else defaults.temperature
        top_k = defaults.top_k
        top_p = defaults.top_p

        sampled = self._model.apply_lm_head_and_sample_batch(
            hidden_state_batch=hidden_state_batch,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if not isinstance(sampled, list):
            raise RuntimeError(
                f"ShardedAutoregressiveRunner: "
                f"apply_lm_head_and_sample_batch must return list, "
                f"got {type(sampled).__name__}"
            )
        verified_token_ids = tuple(int(t) for t in sampled)
        # K+1 verified positions; K proposed.
        k = len(proposed_token_ids)
        if len(verified_token_ids) != k + 1:
            raise RuntimeError(
                f"ShardedAutoregressiveRunner: VERIFY tail expected "
                f"{k + 1} verified token ids (K+1 with K="
                f"{k}), got {len(verified_token_ids)}"
            )

        # accepted_count = longest matching prefix between
        # proposed and verified[:K]. verified[i] is the model's
        # argmax for the position FOLLOWING input i; verified[0]
        # is what comes after parent (should match proposed[0]
        # = d_1 if accepted). verified[K] is the bonus token
        # emitted when all drafts accept.
        accepted_count = 0
        for i in range(k):
            if verified_token_ids[i] == proposed_token_ids[i]:
                accepted_count += 1
            else:
                break

        # Emitted = verified_token_ids[: accepted_count + 1] —
        # accepted draft positions plus one correction/bonus.
        emitted = verified_token_ids[: accepted_count + 1]

        # Bump tokens_generated by emitted count (only emitted
        # tokens count against max_tokens; speculatively-cached-
        # then-rejected positions don't).
        handle.tokens_generated += len(emitted)

        # Termination: EOS in any emitted position OR
        # tokens_generated reached max_tokens.
        is_terminal = False
        if self._eos_token_id is not None:
            for tok in emitted:
                if tok == self._eos_token_id:
                    is_terminal = True
                    break
        if handle.tokens_generated >= max_tokens:
            is_terminal = True

        next_token_id = int(emitted[-1])
        return next_token_id, is_terminal, verified_token_ids, accepted_count

    # ── eviction passthrough ──────────────────────────────────────────

    def rollback_cache(
        self,
        request_id: str,
        n_positions: int,
    ) -> Tuple[bool, int]:
        """Phase 3.x.11.y — public rollback entrypoint. Wraps the
        manager's ``rollback`` with this runner's ``model.truncate_cache``
        as the ``truncate_fn``. Wired by ``LayerStageServer``'s
        ``_handle_rollback_cache``.

        Returns ``(rolled_back, actual_dropped)``:
          - ``rolled_back=True`` iff at least one position was
            dropped.
          - ``actual_dropped`` is the count actually removed (may
            be less than ``n_positions`` if the cache had fewer
            generated tokens; idempotent over-drop semantics
            mirror ``KVCacheManager.rollback``'s contract).

        Raises ``MissingVerifyCapabilityError`` if the model
        doesn't expose ``truncate_cache`` — server maps this to
        ``MALFORMED_REQUEST`` so the executor can distinguish
        caller bug from internal crash.
        """
        if not callable(getattr(self._model, "truncate_cache", None)):
            raise MissingVerifyCapabilityError(
                "ShardedAutoregressiveRunner: rollback requires "
                "model.truncate_cache(...) per ShardedLayerForward "
                "Protocol — this model omits it (typically signals "
                "a non-speculation-capable model wired into a "
                "speculation-enabled chain)"
            )
        return self._cache.rollback(
            request_id=request_id,
            n_positions=n_positions,
            truncate_fn=self._model.truncate_cache,
        )

    def evict(self, request_id: str) -> bool:
        """Explicit cache eviction. Idempotent (returns ``False``
        when no handle existed). Wired by the executor's
        terminal-cleanup signal (Task 6 ``EvictCacheRequest``)."""
        return self._cache.evict(request_id)
