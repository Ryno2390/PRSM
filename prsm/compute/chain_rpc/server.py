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
from typing import Any, Callable, Iterable, List, Optional, Protocol, Tuple

import numpy as np

from prsm.compute.chain_rpc.activation_codec import (
    DEFAULT_CHUNK_BYTES_ACTIVATION,
    ActivationCodecError,
    chunk_activation,
    decode_activation,
    encode_activation,
    reassemble_chunked,
)
from prsm.compute.chain_rpc.protocol import (
    ActivationChunk,
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
from prsm.node.shard_streaming import ShardChunk
from prsm.compute.tee.models import HARDWARE_TEE_TYPES, PrivacyLevel, TEEType
from prsm.node.identity import NodeIdentity


logger = logging.getLogger(__name__)


_UNKNOWN_REQUEST_ID = "<unknown>"


@dataclass(frozen=True)
class _GateResult:
    """Outcome of ``_run_validation_gates``. Either ``model`` is
    populated (all gates pass) or ``error`` is populated (first gate
    that failed)."""

    model: Optional[Any] = None
    error: Optional[Tuple[StageErrorCode, str]] = None


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
        chunk_bytes: int = DEFAULT_CHUNK_BYTES_ACTIVATION,
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
        if chunk_bytes <= 0:
            raise ValueError(
                f"chunk_bytes must be positive, got {chunk_bytes}"
            )

        self._identity = identity
        self._registry = registry
        self._runner = runner
        self._tee_runtime = tee_runtime
        self._anchor = anchor
        self._clock = clock
        self._chunk_bytes = int(chunk_bytes)

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

    # ── streamed-path public entry ────────────────────────────────────

    def handle_streamed(
        self,
        manifest_bytes: bytes,
        chunk_iter: Iterable[bytes],
    ) -> Tuple[bytes, Iterable[bytes]]:
        """Parse manifest → validate → assemble chunks → run → chunk
        output → sign + return.

        Mirrors ``handle()`` for the v2 streamed path. Validation
        gates (token verify, deadline, registry, shard coverage, tier
        gate) fire BEFORE chunk consumption so a peer that exhausts
        the stream without a valid token wastes only the manifest
        parse cost.

        NEVER raises. Returns ``(error_bytes, empty_iter)`` on any
        failure path so the streaming transport never sees an
        exception.
        """
        # Step 1: parse the manifest.
        try:
            request = parse_message(manifest_bytes)
        except ChainRpcVersionMismatchError as exc:
            return self._streamed_error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.UNSUPPORTED_VERSION,
                str(exc),
            )
        except (ChainRpcMalformedError, ChainRpcUnknownTypeError) as exc:
            return self._streamed_error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.MALFORMED_REQUEST,
                str(exc),
            )
        except ChainRpcProtocolError as exc:
            return self._streamed_error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.MALFORMED_REQUEST,
                str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer.handle_streamed: parse raised"
            )
            return self._streamed_error(
                _UNKNOWN_REQUEST_ID,
                StageErrorCode.INTERNAL_ERROR,
                f"internal error during parse: {exc.__class__.__name__}",
            )

        if not isinstance(request, RunLayerSliceRequest):
            return self._streamed_error(
                getattr(request, "request_id", _UNKNOWN_REQUEST_ID),
                StageErrorCode.MALFORMED_REQUEST,
                f"streamed handler only accepts RunLayerSliceRequest; got "
                f"{type(request).__name__}",
            )

        # Streamed-handler contract: the request manifest MUST carry
        # an activation_manifest. A peer using the streamed RPC method
        # for an inline-formatted request is a protocol violation;
        # reject explicitly.
        if request.activation_manifest is None:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.MALFORMED_REQUEST,
                "streamed dispatch: request lacks activation_manifest "
                "(use handle() for inline requests)",
            )

        try:
            return self._dispatch_streamed(request, chunk_iter)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer.handle_streamed: dispatch raised "
                "for request_id=%r",
                request.request_id,
            )
            return self._streamed_error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"internal error during dispatch: {exc.__class__.__name__}",
            )

    # ── pipeline ──────────────────────────────────────────────────────

    def _dispatch(self, request: RunLayerSliceRequest) -> bytes:
        # Streamed requests routed to handle_streamed; reject explicitly
        # if a streamed request reaches the inline path.
        if request.activation_manifest is not None:
            return self._error(
                request.request_id,
                StageErrorCode.MALFORMED_REQUEST,
                "inline dispatch: request carries activation_manifest "
                "(use handle_streamed() for streamed requests)",
            )

        # Steps 2-6: shared validation gates.
        gate_result = self._run_validation_gates(request)
        if gate_result.error is not None:
            return self._error(
                request.request_id, gate_result.error[0], gate_result.error[1]
            )
        model = gate_result.model

        # Step 7a: decode inline activation.
        try:
            activation = decode_activation(
                request.activation_blob,
                request.activation_shape,
                request.activation_dtype,
            )
        except ActivationCodecError as exc:
            return self._error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                str(exc),
            )

        # Step 7b: run + encode + sign (inline).
        result_or_error = self._run_layer_slice(request, model, activation)
        if isinstance(result_or_error, tuple):
            return self._error(
                request.request_id, result_or_error[0], result_or_error[1]
            )
        result = result_or_error

        try:
            output_blob, output_shape, output_dtype = encode_activation(
                result.output
            )
        except ActivationCodecError as exc:
            return self._error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"output encode failure: {exc}",
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

    # ── streamed dispatch ─────────────────────────────────────────────

    def _dispatch_streamed(
        self,
        request: RunLayerSliceRequest,
        chunk_iter: Iterable[bytes],
    ) -> Tuple[bytes, Iterable[bytes]]:
        """Streamed-path body. Validation BEFORE chunk consumption so
        rejected requests don't pay assembly cost."""
        # Steps 2-6: shared validation gates (BEFORE consuming chunks).
        gate_result = self._run_validation_gates(request)
        if gate_result.error is not None:
            return self._streamed_error(
                request.request_id, gate_result.error[0], gate_result.error[1]
            )
        model = gate_result.model

        # Step 7a-streamed: assemble chunks → decode activation. Both
        # ShardAssembler errors and decode errors map to ACTIVATION_INVALID.
        try:
            shard_chunks = self._reassemble_inbound_chunks(
                chunk_iter,
                expected_request_id=request.request_id,
                manifest_shard_id=request.activation_manifest.shard_id,
            )
        except ActivationCodecError as exc:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"streamed input chunk parse failed: {exc}",
            )

        from prsm.compute.chain_rpc.activation_codec import (
            ChunkedActivation as _CA,
        )
        chunked_in = _CA(
            manifest=request.activation_manifest,
            chunks=shard_chunks,
            shape=request.activation_shape,
            dtype_str=request.activation_dtype,
        )
        try:
            activation = reassemble_chunked(chunked_in, chunks=shard_chunks)
        except ActivationCodecError as exc:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"streamed input reassembly failed: {exc}",
            )

        # Step 7b: run.
        result_or_error = self._run_layer_slice(request, model, activation)
        if isinstance(result_or_error, tuple):
            return self._streamed_error(
                request.request_id, result_or_error[0], result_or_error[1]
            )
        result = result_or_error

        # Step 7c-streamed: chunk the output for streaming back. v1
        # contract: streamed request → streamed response (always).
        try:
            chunked_out = chunk_activation(
                result.output,
                activation_id=f"{request.request_id}::resp",
                chunk_bytes=self._chunk_bytes,
            )
        except ActivationCodecError as exc:
            return self._streamed_error(
                request.request_id,
                StageErrorCode.ACTIVATION_INVALID,
                f"streamed output encode failed: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer: streamed output encode failed for "
                "request_id=%r",
                request.request_id,
            )
            return self._streamed_error(
                request.request_id,
                StageErrorCode.INTERNAL_ERROR,
                f"streamed output encode failure: {exc.__class__.__name__}",
            )

        # Step 8: sign + emit. The signing payload commits to
        # manifest.payload_sha256 (v2 conditional encoding); response
        # activation_blob is empty.
        response = RunLayerSliceResponse.sign(
            identity=self._identity,
            request_id=request.request_id,
            activation_blob=b"",
            activation_shape=chunked_out.shape,
            activation_dtype=chunked_out.dtype_str,
            duration_seconds=result.duration_seconds,
            tee_attestation=result.tee_attestation,
            tee_type=result.tee_type,
            epsilon_spent=result.epsilon_spent,
            activation_manifest=chunked_out.manifest,
        )
        response_manifest_bytes = encode_message(response)

        # Encode each chunk as an ActivationChunk wire frame. Bind to
        # request.request_id (relay-defense — chunks can't be spliced
        # across streams).
        response_chunk_frames: List[bytes] = []
        for shard_chunk in chunked_out.chunks:
            frame = ActivationChunk(
                request_id=request.request_id,
                sequence=shard_chunk.sequence,
                data=shard_chunk.data,
                chunk_sha256=shard_chunk.chunk_sha256,
            )
            response_chunk_frames.append(encode_message(frame))

        return response_manifest_bytes, iter(response_chunk_frames)

    # ── shared validation + run ───────────────────────────────────────

    def _run_validation_gates(
        self, request: RunLayerSliceRequest
    ) -> "_GateResult":
        """Steps 2-6: token verify → deadline → registry → shard
        coverage → tier gate. Shared by inline + streamed paths.
        Returns a ``_GateResult`` with either ``model`` populated (all
        gates pass) or ``error`` populated (first failure)."""
        # Step 2: anchor-verify the upstream token.
        if not request.upstream_token.verify_with_anchor(self._anchor):
            return _GateResult(error=(
                StageErrorCode.INVALID_TOKEN,
                f"upstream_token failed anchor verification (settler="
                f"{request.upstream_token.settler_node_id!r})",
            ))

        # Step 3: deadline check.
        now = self._clock()
        if request.upstream_token.deadline_unix <= now:
            return _GateResult(error=(
                StageErrorCode.DEADLINE_EXCEEDED,
                f"token deadline {request.upstream_token.deadline_unix} "
                f"already past clock {now}",
            ))
        if request.deadline_unix <= now:
            return _GateResult(error=(
                StageErrorCode.DEADLINE_EXCEEDED,
                f"request deadline {request.deadline_unix} already past "
                f"clock {now}",
            ))

        # Step 4: registry lookup.
        try:
            model = self._registry.get(request.model_id)
        except Exception as exc:  # noqa: BLE001
            class_name = exc.__class__.__name__
            if "NotFound" in class_name:
                return _GateResult(error=(
                    StageErrorCode.MODEL_NOT_FOUND,
                    f"model {request.model_id!r} not in local registry",
                ))
            if "Verification" in class_name:
                return _GateResult(error=(
                    StageErrorCode.MODEL_NOT_FOUND,
                    f"model {request.model_id!r} failed registry "
                    f"verification: {exc}",
                ))
            logger.exception(
                "LayerStageServer: registry.get raised unexpectedly"
            )
            return _GateResult(error=(
                StageErrorCode.INTERNAL_ERROR,
                f"registry error: {class_name}",
            ))

        # Step 5: layer-range coverage.
        if not _shards_cover_range(model, request.layer_range):
            return _GateResult(error=(
                StageErrorCode.SHARD_MISSING,
                f"local shards do not cover layer_range "
                f"{request.layer_range} for model {request.model_id!r}",
            ))

        # Step 6: privacy-tier gate against the local TEE runtime.
        if request.privacy_tier != PrivacyLevel.NONE:
            tee_type = self._tee_runtime.tee_type
            if not self._is_hardware_tee(tee_type):
                return _GateResult(error=(
                    StageErrorCode.TIER_GATE,
                    f"privacy_tier={request.privacy_tier.value} requires "
                    f"hardware TEE; local runtime is {tee_type.value}",
                ))

        return _GateResult(model=model)

    def _run_layer_slice(
        self,
        request: RunLayerSliceRequest,
        model: Any,
        activation: np.ndarray,
    ):
        """Wraps runner.run_layer_range with TIMEOUT / INTERNAL_ERROR
        mapping. Returns a ``LayerSliceResult`` on success or a
        ``(StageErrorCode, message)`` tuple on failure."""
        try:
            return self._runner.run_layer_range(
                model=model,
                layer_range=request.layer_range,
                activation=activation,
                privacy_tier=request.privacy_tier,
                is_final_stage=_is_final_stage(model, request.layer_range),
            )
        except TimeoutError as exc:
            return (StageErrorCode.TIMEOUT, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LayerStageServer: runner.run_layer_range raised for "
                "request_id=%r",
                request.request_id,
            )
            return (
                StageErrorCode.INTERNAL_ERROR,
                f"layer-runner failure: {exc.__class__.__name__}",
            )

    @staticmethod
    def _reassemble_inbound_chunks(
        frame_iter: Iterable[bytes],
        *,
        expected_request_id: str,
        manifest_shard_id: str,
    ) -> List[ShardChunk]:
        """Decode incoming ActivationChunk frames back to Phase 6
        ``ShardChunk`` objects. Validates the relay-defense binding
        (chunk.request_id must match the parent request); rewraps the
        shard_id to match the inbound manifest so the assembler's
        per-chunk validation succeeds.
        """
        out: List[ShardChunk] = []
        for raw in frame_iter:
            try:
                msg = parse_message(raw)
            except (
                ChainRpcMalformedError,
                ChainRpcUnknownTypeError,
                ChainRpcProtocolError,
                ChainRpcVersionMismatchError,
            ) as exc:
                raise ActivationCodecError(
                    f"streamed chunk frame parse failed: {exc}"
                ) from exc
            if not isinstance(msg, ActivationChunk):
                raise ActivationCodecError(
                    f"streamed chunk frame: expected ActivationChunk, "
                    f"got {type(msg).__name__}"
                )
            if msg.request_id != expected_request_id:
                raise ActivationCodecError(
                    f"streamed chunk request_id mismatch: got "
                    f"{msg.request_id!r}, expected {expected_request_id!r}"
                )
            out.append(ShardChunk(
                shard_id=manifest_shard_id,
                sequence=msg.sequence,
                data=msg.data,
                chunk_sha256=msg.chunk_sha256,
            ))
        return out

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

    def _streamed_error(
        self,
        request_id: str,
        code: StageErrorCode,
        message: str,
    ) -> Tuple[bytes, Iterable[bytes]]:
        """Streamed-path failure: encode StageError as the manifest
        return slot + empty chunk iterator. Caller never sees a raise."""
        return self._error(request_id, code, message), iter(())
