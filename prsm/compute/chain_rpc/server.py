"""Phase 3.x.7 Task 2 — Cross-host LayerStageServer.

Per-node server that receives a ``RunLayerSliceRequest`` from an
upstream stage (or the executor itself for the chain head), validates
it through the 8-step gate, executes the requested contiguous layer
slice via a local layer runner, and returns a stage-signed
``RunLayerSliceResponse``.

8-step validation order (matches design plan §3.4):

  1. Parse the request                  → MALFORMED_REQUEST
  2. verify_with_anchor on token        → INVALID_TOKEN
  3. token.deadline_unix > clock()      → DEADLINE_EXCEEDED
                                          (also checks request deadline)
  4. registry.get(model_id)             → MODEL_NOT_FOUND
  5. layer_range covered by local shards → SHARD_MISSING
  6. privacy_tier vs tee_runtime.tee_type → TIER_GATE
  7. Decode activation → run layer slice → encode output → handle TIMEOUT
  8. Sign the response under self.identity

"Never raises" invariant: every failure is mapped to a structured
``StageError`` and returned as encoded bytes. Caller transport never
sees an exception. Mirrors Phase 3.x.5 ``ManifestDHTServer.handle()``
and Phase 3.x.6 ``ProfileDHTServer.handle()``.

This module is intentionally narrow: the public surface is the
``LayerStageServer.handle(bytes) → bytes`` callable plus the
``LayerSliceRunner`` Protocol that callers inject for production
wiring (Task 6 wraps Phase 2 Ring 8's ``TensorParallelExecutor``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Tuple

import numpy as np

from prsm.compute.chain_rpc.protocol import (
    ChainRpcMalformedError,
    ChainRpcProtocolError,
    ChainRpcUnknownTypeError,
    ChainRpcVersionMismatchError,
    HandoffToken,
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    StageError,
    StageErrorCode,
    encode_message,
    parse_message,
)
from prsm.compute.tee.models import HARDWARE_TEE_TYPES, PrivacyLevel, TEEType
from prsm.node.identity import NodeIdentity


logger = logging.getLogger(__name__)


_UNKNOWN_REQUEST_ID = "<unknown>"


# ──────────────────────────────────────────────────────────────────────────
# LayerSliceRunner Protocol — what the server calls to actually run layers
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerSliceResult:
    """Per-stage output bundled for the response.

    Mirrors the shape that ``RunLayerSliceResponse.sign(...)`` expects.
    Production wiring (Task 6) wraps the existing
    ``TensorParallelExecutor`` to satisfy this Protocol; tests inject
    a fake.
    """

    output: np.ndarray
    duration_seconds: float
    tee_attestation: bytes
    tee_type: TEEType
    epsilon_spent: float


class LayerSliceRunner(Protocol):
    """Adapter that runs a contiguous layer range on a local model.

    The runner is responsible for:
      - Loading the requested layer slice's weights into the local
        accelerator (or reusing if already loaded).
      - Running the forward pass for those layers on the input
        ``activation`` ndarray.
      - Applying DP noise + producing the TEE attestation IF this is
        the chain's tail stage (the server passes ``apply_dp=True``
        only when it knows from the request layout that this is the
        final stage; v1 always sets it based on the layer_range
        covering the model's last layer).
      - Returning a ``LayerSliceResult`` with the post-layer-slice
        activation, timing, attestation, and any DP epsilon spent.

    Errors propagate. The server wraps the call in try/except and
    maps to ``StageError(INTERNAL_ERROR)``.
    """

    def run_layer_range(
        self,
        *,
        model: Any,
        layer_range: Tuple[int, int],
        activation: np.ndarray,
        privacy_tier: PrivacyLevel,
        is_final_stage: bool,
    ) -> LayerSliceResult:
        ...


# ──────────────────────────────────────────────────────────────────────────
# Activation codec helpers (inline; Task 3 will extend into a module)
# ──────────────────────────────────────────────────────────────────────────


def _decode_activation(
    blob: bytes,
    shape: Tuple[int, ...],
    dtype_str: str,
) -> np.ndarray:
    """Reconstruct a numpy array from raw bytes + shape + dtype.

    Raises ``ValueError`` on unsupported dtype or shape/blob mismatch.
    Server maps these to ``StageError(ACTIVATION_INVALID)``.
    """
    try:
        dtype = np.dtype(dtype_str)
    except TypeError as exc:
        raise ValueError(f"unsupported dtype {dtype_str!r}: {exc}") from exc
    expected_size = int(np.prod(shape)) * dtype.itemsize
    if expected_size != len(blob):
        raise ValueError(
            f"activation blob size {len(blob)} does not match "
            f"shape {shape} × dtype {dtype_str} (expected {expected_size} bytes)"
        )
    return np.frombuffer(blob, dtype=dtype).reshape(shape).copy()


def _encode_activation(arr: np.ndarray) -> Tuple[bytes, Tuple[int, ...], str]:
    """Inverse of ``_decode_activation``. Forces C-contiguous layout
    so the output bytes are deterministic regardless of upstream
    striding."""
    contig = np.ascontiguousarray(arr)
    return contig.tobytes(), tuple(int(d) for d in contig.shape), str(contig.dtype)


# ──────────────────────────────────────────────────────────────────────────
# Layer-coverage helpers
# ──────────────────────────────────────────────────────────────────────────


def _shards_cover_range(model: Any, layer_range: Tuple[int, int]) -> bool:
    """Return True iff every layer in [start, end) is covered by at
    least one shard's ``layer_range`` on the local model.

    Tolerates models where shards have ``layer_range == (0, 0)`` —
    that's the registry's "layer assignment unset" sentinel from
    Phase 3.x.2 Task 5; production allocators set real ranges before
    publishing. When EVERY shard has the sentinel, we treat the
    model as covering all layers (single-stage fallback) to preserve
    the existing single-host code path.
    """
    if not hasattr(model, "shards"):
        return False
    shards = list(model.shards)
    if not shards:
        return False
    sentinel = all(
        getattr(s, "layer_range", (0, 0)) == (0, 0) for s in shards
    )
    if sentinel:
        return True
    start, end = layer_range
    if start >= end:
        return False
    covered = [False] * (end - start)
    for shard in shards:
        s_start, s_end = getattr(shard, "layer_range", (0, 0))
        if s_start == 0 and s_end == 0:
            continue
        for layer in range(max(s_start, start), min(s_end, end)):
            covered[layer - start] = True
    return all(covered)


def _is_final_stage(model: Any, layer_range: Tuple[int, int]) -> bool:
    """Heuristic: this stage is the tail iff the requested
    layer_range's end is at or beyond the model's last layer.

    For shards using the (0, 0) sentinel from Phase 3.x.2, we fall
    back to assuming non-final (DP noise applied at the executor
    level instead). v1 conservative — final-stage detection is best-
    effort signal for the LayerSliceRunner.
    """
    if not hasattr(model, "shards") or not model.shards:
        return False
    max_end = 0
    for shard in model.shards:
        s_start, s_end = getattr(shard, "layer_range", (0, 0))
        if s_start == 0 and s_end == 0:
            continue
        max_end = max(max_end, s_end)
    if max_end == 0:
        return False
    return layer_range[1] >= max_end


# ──────────────────────────────────────────────────────────────────────────
# LayerStageServer
# ──────────────────────────────────────────────────────────────────────────


class LayerStageServer:
    """Per-node handler for ``RunLayerSliceRequest``.

    Constructor args:
      identity      The stage's NodeIdentity (used to sign responses).
                    The same identity is what callers register on the
                    Phase 3.x.3 anchor.
      registry      ``ModelRegistry`` for ``model_id → ShardedModel``
                    lookup. Production: ``FilesystemModelRegistry``.
      runner        ``LayerSliceRunner`` that actually executes the
                    layer slice.
      tee_runtime   Phase 2 Ring 8 ``TEERuntime``. The server reads
                    ``tee_runtime.tee_type`` to enforce the privacy-
                    tier gate.
      anchor        Phase 3.x.3 anchor for verifying upstream tokens.
      clock         Injected for tests; defaults to ``time.time``.
    """

    def __init__(
        self,
        *,
        identity: NodeIdentity,
        registry: Any,
        runner: LayerSliceRunner,
        tee_runtime: Any,
        anchor: Any,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if identity is None or not hasattr(identity, "node_id"):
            raise RuntimeError(
                "LayerStageServer requires a NodeIdentity for response signing"
            )
        if registry is None or not hasattr(registry, "get"):
            raise RuntimeError(
                "LayerStageServer requires a ModelRegistry with .get(model_id)"
            )
        if runner is None or not hasattr(runner, "run_layer_range"):
            raise RuntimeError(
                "LayerStageServer requires a LayerSliceRunner with "
                ".run_layer_range(...)"
            )
        if tee_runtime is None or not hasattr(tee_runtime, "tee_type"):
            raise RuntimeError(
                "LayerStageServer requires a tee_runtime with .tee_type"
            )
        if anchor is None or not hasattr(anchor, "lookup"):
            raise RuntimeError(
                "LayerStageServer requires an anchor with .lookup(node_id)"
            )

        self._identity = identity
        self._registry = registry
        self._runner = runner
        self._tee_runtime = tee_runtime
        self._anchor = anchor
        self._clock = clock

    # ── public API ────────────────────────────────────────────────────

    def handle(self, request_bytes: bytes) -> bytes:
        """Parse → validate → run → encode response. NEVER raises.

        Wraps the entire pipeline in defensive try/except so any
        failure path emits a structured ``StageError`` rather than
        propagating an exception through the transport.
        """
        # Step 1: parse.
        try:
            request = parse_message(request_bytes)
        except ChainRpcVersionMismatchError as exc:
            return self._error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.UNSUPPORTED_VERSION,
                str(exc),
            )
        except (ChainRpcMalformedError, ChainRpcUnknownTypeError) as exc:
            return self._error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.MALFORMED_REQUEST,
                str(exc),
            )
        except ChainRpcProtocolError as exc:
            return self._error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.MALFORMED_REQUEST,
                str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("LayerStageServer.handle: unexpected parse failure")
            return self._error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.INTERNAL_ERROR,
                f"internal error during parse: {exc.__class__.__name__}",
            )

        if not isinstance(request, RunLayerSliceRequest):
            return self._error(
                getattr(request, "request_id", _UNKNOWN_REQUEST_ID),
                StageErrorCode.MALFORMED_REQUEST,
                f"server only handles RunLayerSliceRequest; got "
                f"{type(request).__name__}",
            )

        # Step 2-8 in a try/except so any unexpected raise gets mapped
        # to INTERNAL_ERROR rather than propagating.
        try:
            return self._dispatch(request)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer.handle: unexpected dispatch failure for "
                "request_id=%r",
                request.request_id,
            )
            return self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"internal error during dispatch: {exc.__class__.__name__}",
            )

    # ── pipeline ──────────────────────────────────────────────────────

    def _dispatch(self, request: RunLayerSliceRequest) -> bytes:
        # Step 2: anchor-verify the upstream token.
        if not request.upstream_token.verify_with_anchor(self._anchor):
            return self._error(
                request.request_id,
                StageErrorCode.INVALID_TOKEN,
                f"upstream_token failed anchor verification (settler="
                f"{request.upstream_token.settler_node_id!r})",
            )

        # Step 3: deadline check. The token's deadline AND the
        # request's own deadline both bound the work; enforce both.
        now = self._clock()
        if request.upstream_token.deadline_unix <= now:
            return self._error(
                request.request_id,
                StageErrorCode.DEADLINE_EXCEEDED,
                f"token deadline {request.upstream_token.deadline_unix} "
                f"already past clock {now}",
            )
        if request.deadline_unix <= now:
            return self._error(
                request.request_id,
                StageErrorCode.DEADLINE_EXCEEDED,
                f"request deadline {request.deadline_unix} already past "
                f"clock {now}",
            )

        # Step 4: registry lookup. The registry's get() raises on
        # ManifestVerificationError or ModelNotFoundError; map both
        # to specific codes.
        try:
            model = self._registry.get(request.model_id)
        except Exception as exc:  # noqa: BLE001
            # Distinguish missing vs verification failure by class
            # name to avoid hard-coding registry imports here (keeps
            # this module dependency-light).
            class_name = exc.__class__.__name__
            if "NotFound" in class_name:
                return self._error(
                    request.request_id,
                    StageErrorCode.MODEL_NOT_FOUND,
                    f"model {request.model_id!r} not in local registry",
                )
            if "Verification" in class_name:
                return self._error(
                    request.request_id,
                    StageErrorCode.MODEL_NOT_FOUND,
                    f"model {request.model_id!r} failed registry "
                    f"verification: {exc}",
                )
            # Unexpected — surface as internal.
            logger.exception(
                "LayerStageServer: registry.get raised unexpectedly"
            )
            return self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"registry error: {class_name}",
            )

        # Step 5: layer-range coverage.
        if not _shards_cover_range(model, request.layer_range):
            return self._error(
                request.request_id,
                StageErrorCode.SHARD_MISSING,
                f"local shards do not cover layer_range "
                f"{request.layer_range} for model {request.model_id!r}",
            )

        # Step 6: privacy-tier gate against the local TEE runtime.
        if request.privacy_tier != PrivacyLevel.NONE:
            tee_type = self._tee_runtime.tee_type
            if not self._is_hardware_tee(tee_type):
                return self._error(
                    request.request_id,
                    StageErrorCode.TIER_GATE,
                    f"privacy_tier={request.privacy_tier.value} requires "
                    f"hardware TEE; local runtime is {tee_type.value}",
                )

        # Step 7: decode → run → encode.
        try:
            activation = _decode_activation(
                request.activation_blob,
                request.activation_shape,
                request.activation_dtype,
            )
        except ValueError as exc:
            return self._error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                str(exc),
            )

        try:
            result = self._runner.run_layer_range(
                model=model,
                layer_range=request.layer_range,
                activation=activation,
                privacy_tier=request.privacy_tier,
                is_final_stage=_is_final_stage(model, request.layer_range),
            )
        except TimeoutError as exc:
            return self._error(
                request.request_id,
                StageErrorCode.TIMEOUT,
                str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer: runner.run_layer_range raised for "
                "request_id=%r",
                request.request_id,
            )
            return self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"layer-runner failure: {exc.__class__.__name__}",
            )

        try:
            output_blob, output_shape, output_dtype = _encode_activation(
                result.output
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer: output encode failed for request_id=%r",
                request.request_id,
            )
            return self._error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"output encode failure: {exc.__class__.__name__}",
            )

        # Step 8: sign and return.
        response = RunLayerSliceResponse.sign(
            identity=self._identity,
            request_id=request.request_id,
            activation_blob=output_blob,
            activation_shape=output_shape,
            activation_dtype=output_dtype,
            duration_seconds=result.duration_seconds,
            tee_attestation=result.tee_attestation,
            tee_type=result.tee_type,
            epsilon_spent=result.epsilon_spent,
        )
        return encode_message(response)

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_hardware_tee(tee_type: TEEType) -> bool:
        """Same hardware-TEE policy used by Phase 3.x.6 Task 5
        ``TierGateAdapter``: anything in ``HARDWARE_TEE_TYPES`` counts;
        ``SOFTWARE`` does not."""
        return tee_type.value in HARDWARE_TEE_TYPES

    def _error(
        self,
        request_id: str,
        code: StageErrorCode,
        message: str,
    ) -> bytes:
        """Encode a ``StageError`` for return; never raises (helpers
        within encode_message are pure JSON serialization)."""
        return encode_message(
            StageError(
                request_id=request_id or _UNKNOWN_REQUEST_ID,
                code=code.value,
                message=message,
            )
        )
