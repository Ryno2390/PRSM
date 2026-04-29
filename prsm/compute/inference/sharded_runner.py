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
from prsm.compute.chain_rpc.protocol import DecodeMode
from prsm.compute.inference.autoregressive_runner import SamplingDefaults
from prsm.compute.tee.models import TEEType


__all__ = [
    "LayerSliceResult",
    "MalformedCacheStateError",
    "MissingTailCapabilityError",
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


# ──────────────────────────────────────────────────────────────────────────
# LayerSliceResult — runner return type
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerSliceResult:
    """Return value of ``run_layer_slice_unary``.

    The caller (server's unary handler) maps these into a
    ``RunLayerSliceResponse`` — ``hidden_state`` becomes the wire
    activation; ``next_token_id`` + ``is_terminal`` populate
    tail-only fields (None / False on non-tail responses).

    ``n_layers_run`` is the size of the layer range this dispatch
    actually executed; useful for receipt + metrics.
    """

    hidden_state: np.ndarray
    duration_seconds: float
    decode_mode: DecodeMode
    n_layers_run: int
    next_token_id: Optional[int] = None
    is_terminal: bool = False


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
        request: Any = None,  # noqa: ARG002 — Task 4 wires sampling
    ) -> LayerSliceResult:
        """Dispatch one PREFILL or INCREMENTAL forward through
        ``layer_range``.

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

        Tail (Task 4) extends this with LM-head + sampling. v1
        non-tail returns ``next_token_id=None`` +
        ``is_terminal=False`` always.
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
        else:
            # INCREMENTAL
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

        next_token_id: Optional[int] = None
        is_terminal = False
        if is_final_stage:
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
        )

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

    # ── eviction passthrough ──────────────────────────────────────────

    def evict(self, request_id: str) -> bool:
        """Explicit cache eviction. Idempotent (returns ``False``
        when no handle existed). Wired by the executor's
        terminal-cleanup signal (Task 6 ``EvictCacheRequest``)."""
        return self._cache.evict(request_id)
