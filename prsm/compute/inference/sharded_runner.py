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
from prsm.compute.tee.models import TEEType


__all__ = [
    "LayerSliceResult",
    "MalformedCacheStateError",
    "ShardedAutoregressiveRunner",
    "ShardedLayerForward",
]


class MalformedCacheStateError(RuntimeError):
    """Raised on INCREMENTAL dispatch with no prior PREFILL cache.
    Maps to ``MALFORMED_REQUEST`` at the wire layer (the caller —
    typically the executor's per-token chain loop — must restart
    with a fresh PREFILL or surface the error to the user)."""


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

    Tail-only contract: v1 of this class is the **non-tail**
    variant — ``is_final_stage=True`` raises at dispatch. Task 4
    extends to handle LM-head + sampling. Until Task 4 lands, the
    tail keeps using ``AutoregressiveStreamingRunner`` over the
    streaming wire path; sharded decode is opt-in via
    ``enable_sharded_decode`` on the executor (Task 5).
    """

    def __init__(
        self,
        *,
        model: ShardedLayerForward,
        layer_range: Tuple[int, int],
        kv_cache_manager: KVCacheManager,
        tee_attestation: bytes,
        tee_type: TEEType,
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

        self._model = model
        self._layer_range = (int(start), int(end))
        self._n_layers = int(end - start)
        self._cache = kv_cache_manager
        self._tee_attestation = bytes(tee_attestation)
        self._tee_type = tee_type

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
        if is_final_stage:
            # Phase 3.x.11 Task 3 honest scope — non-tail variant
            # only. Task 4 extends this with LM-head + sampling.
            # Until then, the tail keeps using
            # AutoregressiveStreamingRunner over the streaming
            # wire path.
            raise RuntimeError(
                "ShardedAutoregressiveRunner: is_final_stage=True is "
                "reserved for the Task 4 tail variant; v1 non-tail "
                "rejects tail dispatch — operators must keep the "
                "tail on AutoregressiveStreamingRunner until Task 4 "
                "lands"
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

        duration = time.time() - start_ts
        return LayerSliceResult(
            hidden_state=hidden,
            duration_seconds=duration,
            decode_mode=decode_mode,
            n_layers_run=self._n_layers,
        )

    # ── eviction passthrough ──────────────────────────────────────────

    def evict(self, request_id: str) -> bool:
        """Explicit cache eviction. Idempotent (returns ``False``
        when no handle existed). Wired by the executor's
        terminal-cleanup signal (Task 6 ``EvictCacheRequest``)."""
        return self._cache.evict(request_id)
