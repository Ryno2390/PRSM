"""Phase 3.x.7 Task 4 — Cross-host RpcChainExecutor (client-side orchestrator).

Implements the Phase 3.x.6 ``ChainExecutor`` Protocol. Given a
``GPUChain`` (the router's output) and an ``InferenceRequest``, the
executor:

  1. Encodes the prompt into an initial activation tensor via the
     injected ``PromptEncoder``.
  2. For each stage in chain order, mints a ``HandoffToken`` under
     the settler identity, sends a ``RunLayerSliceRequest`` via the
     injected transport, parses the response, anchor-verifies the
     stage signature, and threads the decoded output activation as
     input to the next stage.
  3. Decodes the final stage's activation into the user-facing string
     via the injected ``OutputDecoder``.
  4. Aggregates per-stage TEE attestations + chooses worst-case
     ``TEEType`` (Task 5 will refine the attestation list format on
     the receipt side).
  5. Returns a ``ChainExecutionResult`` ready for
     ``ParallaxScheduledExecutor`` to wrap in a signed
     ``InferenceReceipt``.

Orchestrator model (vs relay): the executor explicitly calls each
stage in sequence rather than letting the head stage forward through
the chain. Trades 2N round-trips for N for stronger isolation +
easier debugging + per-stage signature verification at one place. The
relay model is a Phase 3.x.7.x perf optimization (per design plan
§2.2 + §6 risk register).

Failure handling — ALL paths surface as ``ChainExecutionError`` so
``ParallaxScheduledExecutor.execute()`` already maps to
``InferenceResult.failure(...)``:

  - Stage returns ``StageError``    → ``code`` = StageErrorCode value
  - Stage signature fails verify     → ``code`` = "INVALID_STAGE_SIGNATURE"
                                       (stronger than output divergence
                                       — the response was UNAUTHENTIC)
  - Transport raises (timeout etc.)  → ``code`` = "TRANSPORT_ERROR"
  - Wire-format parse error          → ``code`` = "MALFORMED_RESPONSE"
  - Codec error decoding output      → ``code`` = "ACTIVATION_INVALID"
  - Wrong response type              → ``code`` = "MALFORMED_RESPONSE"

The executor itself does NOT raise to the transport caller; it raises
``ChainExecutionError`` to its own caller (the
``ParallaxScheduledExecutor``). The transport-side "never raises"
invariant lives on the server (Task 2).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Protocol, Tuple, Union

import numpy as np

from prsm.compute.chain_rpc.activation_codec import (
    CHUNK_THRESHOLD_BYTES,
    DEFAULT_CHUNK_BYTES_ACTIVATION,
    ActivationCodecError,
    chunk_activation,
    decode_activation,
    encode_activation,
    reassemble_chunked,
    should_chunk,
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
    StreamFinalFrame,
    TokenFrame,
    encode_message,
    parse_message,
)
from prsm.node.shard_streaming import ShardChunk
from prsm.compute.inference.models import InferenceRequest
from prsm.compute.inference.multi_stage_attestation import (
    StageAttestation,
    encode_multi_stage_attestation,
    worst_case_tee_type,
)
from prsm.compute.inference.parallax_executor import ChainExecutionResult
from prsm.compute.parallax_scheduling.prsm_request_router import GPUChain
from prsm.compute.tee.models import TEEType
from prsm.node.identity import NodeIdentity


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────


SendMessage = Callable[[str, bytes], bytes]
"""Transport-layer callable: ``(stage_address, request_bytes) → response_bytes``.
Production wires this to Phase 6 ``TransportAdapter``. Tests inject
a fake that maps stage_address → server.handle for in-process round-
trips. May raise transport-level exceptions; the executor maps these
to ``ChainExecutionError(code='TRANSPORT_ERROR')``."""

StreamedSendMessage = Callable[
    [str, bytes, Iterable[bytes]],
    Tuple[bytes, Iterable[bytes]],
]
"""Streaming transport-layer callable: ``(stage_address, manifest_bytes,
chunk_bytes_iter) → (response_manifest_bytes, response_chunk_bytes_iter)``.

Phase 3.x.7.1 v2 streamed path. Production wires this to a Phase 6
gRPC bidi-streaming RPC method: send the manifest as the first frame,
iterate chunks as subsequent frames; receive the response manifest as
the first reply frame, iterate response chunks. Tests inject a fake
that processes manifest + chunks in-process via LayerStageServer.

May raise transport-level exceptions; the executor maps these to
``ChainExecutionError(code='TRANSPORT_ERROR')``. Optional —
``RpcChainExecutor`` falls back to ``ACTIVATION_TOO_LARGE`` when an
activation exceeds the inline threshold but no streaming transport
was wired."""

TokenStreamSendMessage = Callable[[str, bytes], Iterable[bytes]]
"""Server-streaming transport-layer callable for Phase 3.x.8 token
output: ``(stage_address, request_bytes) → Iterable[response_frames]``.

Production wires this to a Phase 6 gRPC SERVER-streaming RPC method
(one request, many response frames). Each yielded ``bytes`` is a
wire-encoded ``TokenFrame`` (incremental) or the terminal
``StreamFinalFrame`` carrying the signed
``RunLayerSliceResponse``. Tests inject a fake that drives
``LayerStageServer.handle_token_stream`` in-process.

May raise transport-level exceptions; the executor maps these to
``ChainExecutionError(code='TRANSPORT_ERROR')``. Optional —
``RpcChainExecutor.execute_chain_streaming`` raises
``ExecutorErrorCode.STREAMING_NOT_WIRED`` when called without a
streaming transport configured."""

PromptEncoder = Callable[[str], np.ndarray]
"""Convert the user-facing prompt string into the chain head's input
activation tensor. Production wraps the model's tokenizer + embedding
layer. Tests inject deterministic fakes."""

OutputDecoder = Callable[[np.ndarray], str]
"""Inverse of ``PromptEncoder`` for the chain tail's output. Production
de-tokenizes; tests use a deterministic representation."""

AddressResolver = Callable[[str], str]
"""Map a chain stage's ``node_id`` to a transport address. Default
identity (``node_id == address``) suffices for the in-process test
harness; production wires to Phase 6 peer registry."""


# ──────────────────────────────────────────────────────────────────────────
# Output dataclasses
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageOutcome:
    """Per-stage execution record. Aggregated into the final
    ``ChainExecutionResult`` so callers can audit each stage's
    contribution to the chain."""

    stage_index: int
    stage_node_id: str
    duration_seconds: float
    tee_attestation: bytes
    tee_type: TEEType
    epsilon_spent: float


# ``ChainExecutionResult`` is shared with Phase 3.x.6's
# ``ParallaxScheduledExecutor`` — both produce + consume the same
# Protocol-shape from ``prsm.compute.inference.parallax_executor``.
# This client returns that exact dataclass; ``StageOutcome`` is an
# internal aggregation record consumed by the helpers below to derive
# the public fields per the design plan §3.7 worst-case policy:
#   duration_seconds = sum over stages
#   tee_type         = worst case (SOFTWARE drags down hardware)
#   epsilon_spent    = sum over stages (DP applied only at the tail
#                      per the runner's is_final_stage flag, so
#                      non-tail stages contribute 0.0)
#   tee_attestation  = length-prefixed concatenation of per-stage
#                      attestations; Task 5 will swap this for a
#                      JSON-encoded list at the InferenceReceipt
#                      layer.


# ──────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────


class ChainExecutionError(Exception):
    """Structured failure raised by ``RpcChainExecutor`` to its caller
    (``ParallaxScheduledExecutor``). The caller wraps this in
    ``InferenceResult.failure(...)`` with the executor's standard
    "chain execution" reason prefix."""

    def __init__(
        self,
        *,
        stage_index: int,
        stage_node_id: str,
        code: str,
        message: str,
    ):
        super().__init__(
            f"chain stage {stage_index} ({stage_node_id!r}): "
            f"{code} — {message}"
        )
        self.stage_index = stage_index
        self.stage_node_id = stage_node_id
        self.code = code
        self.message = message


# Internal codes layered on top of the wire protocol's StageErrorCode.
# These cover failures that happen at the executor's side (bad
# response, transport exception) rather than ones the server reports.

class ExecutorErrorCode:
    INVALID_STAGE_SIGNATURE = "INVALID_STAGE_SIGNATURE"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    ACTIVATION_INVALID = "ACTIVATION_INVALID"
    PROMPT_ENCODE_ERROR = "PROMPT_ENCODE_ERROR"
    OUTPUT_DECODE_ERROR = "OUTPUT_DECODE_ERROR"
    EMPTY_CHAIN = "EMPTY_CHAIN"
    SHAPE_MISMATCH = "SHAPE_MISMATCH"
    # Phase 3.x.7.1: activation exceeds inline-path threshold but no
    # streamed transport was wired. Caller fix: pass
    # streamed_send_message= to make_rpc_chain_executor / RpcChainExecutor.
    ACTIVATION_TOO_LARGE = "ACTIVATION_TOO_LARGE"
    # Phase 3.x.8: caller invoked execute_chain_streaming() but no
    # token_stream_send_message= was wired. Caller fix: provide a
    # streaming transport at executor construction time.
    STREAMING_NOT_WIRED = "STREAMING_NOT_WIRED"


# ──────────────────────────────────────────────────────────────────────────
# StreamToken — caller-facing incremental output from execute_chain_streaming
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StreamToken:
    """Caller-facing incremental output yielded by
    ``RpcChainExecutor.execute_chain_streaming``. Wraps a wire-format
    ``TokenFrame`` minus the protocol-level fields (request_id +
    protocol_version) — those are the executor's concern, not the
    caller's.

    Multiple ``StreamToken``s per stream, ordered strictly by
    ``sequence_index``. ``finish_reason`` is non-None on the LAST
    token; the executor follows the last token with a
    ``ChainExecutionResult`` (the single non-StreamToken yield).
    """

    sequence_index: int
    text_delta: str
    token_id: Optional[int] = None
    finish_reason: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────
# RpcChainExecutor
# ──────────────────────────────────────────────────────────────────────────


class RpcChainExecutor:
    """Client-side orchestrator. Implements the Phase 3.x.6
    ``ChainExecutor`` Protocol.

    Constructor args:
      settler_identity         The settling node's ``NodeIdentity``.
                               Used to mint ``HandoffToken``s; same
                               identity that signs the final
                               ``InferenceReceipt`` at the
                               ``ParallaxScheduledExecutor`` layer.
      send_message             Transport callable (Phase 6).
      anchor                   ``AnchorLookup`` for verifying each
                               stage's response signature.
      prompt_encoder           prompt → initial activation tensor.
      output_decoder           final activation tensor → output string.
      address_resolver         node_id → transport address. Default
                               identity (node_id == address) — fine
                               for in-process tests + simple Phase 6
                               peer registries.
      default_deadline_seconds Per-request deadline budget when the
                               ``InferenceRequest`` doesn't expose
                               its own. Default 30s.
      clock                    Injected for tests; defaults to
                               ``time.time``.
    """

    def __init__(
        self,
        *,
        settler_identity: NodeIdentity,
        send_message: SendMessage,
        anchor: object,
        prompt_encoder: PromptEncoder,
        output_decoder: OutputDecoder,
        address_resolver: Optional[AddressResolver] = None,
        streamed_send_message: Optional[StreamedSendMessage] = None,
        token_stream_send_message: Optional[TokenStreamSendMessage] = None,
        chunk_threshold_bytes: int = CHUNK_THRESHOLD_BYTES,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES_ACTIVATION,
        max_streamed_payload_bytes: int = 1 * 1024 * 1024 * 1024,
        default_deadline_seconds: float = 30.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if settler_identity is None or not hasattr(settler_identity, "node_id"):
            raise RuntimeError(
                "RpcChainExecutor requires a NodeIdentity for settler signing"
            )
        if send_message is None or not callable(send_message):
            raise RuntimeError(
                "RpcChainExecutor requires a callable send_message(addr, bytes)"
            )
        if streamed_send_message is not None and not callable(streamed_send_message):
            raise RuntimeError(
                "RpcChainExecutor: streamed_send_message must be callable "
                "if provided"
            )
        if token_stream_send_message is not None and not callable(
            token_stream_send_message
        ):
            raise RuntimeError(
                "RpcChainExecutor: token_stream_send_message must be "
                "callable if provided"
            )
        if anchor is None or not hasattr(anchor, "lookup"):
            raise RuntimeError(
                "RpcChainExecutor requires an anchor with .lookup(node_id)"
            )
        if prompt_encoder is None or not callable(prompt_encoder):
            raise RuntimeError(
                "RpcChainExecutor requires a callable prompt_encoder"
            )
        if output_decoder is None or not callable(output_decoder):
            raise RuntimeError(
                "RpcChainExecutor requires a callable output_decoder"
            )
        if default_deadline_seconds <= 0:
            raise ValueError(
                f"default_deadline_seconds must be positive, "
                f"got {default_deadline_seconds}"
            )
        if chunk_threshold_bytes <= 0:
            raise ValueError(
                f"chunk_threshold_bytes must be positive, "
                f"got {chunk_threshold_bytes}"
            )
        if chunk_bytes <= 0:
            raise ValueError(
                f"chunk_bytes must be positive, got {chunk_bytes}"
            )
        if max_streamed_payload_bytes <= 0:
            raise ValueError(
                f"max_streamed_payload_bytes must be positive, got "
                f"{max_streamed_payload_bytes}"
            )

        self._settler = settler_identity
        self._send = send_message
        self._streamed_send = streamed_send_message
        self._token_stream_send = token_stream_send_message
        self._anchor = anchor
        self._prompt_encoder = prompt_encoder
        self._output_decoder = output_decoder
        self._resolve_address = address_resolver or (lambda nid: nid)
        self._chunk_threshold_bytes = int(chunk_threshold_bytes)
        self._chunk_bytes = int(chunk_bytes)
        self._max_streamed_payload_bytes = int(max_streamed_payload_bytes)
        self._default_deadline_seconds = float(default_deadline_seconds)
        self._clock = clock

    # ── ChainExecutor Protocol ────────────────────────────────────────

    def execute_chain(
        self,
        *,
        request: InferenceRequest,
        chain: GPUChain,
    ) -> ChainExecutionResult:
        if not chain.stages:
            raise ChainExecutionError(
                stage_index=-1,
                stage_node_id="",
                code=ExecutorErrorCode.EMPTY_CHAIN,
                message="GPUChain has no stages — router produced an empty chain",
            )
        if len(chain.stages) != len(chain.layer_ranges):
            raise ChainExecutionError(
                stage_index=-1,
                stage_node_id="",
                code=ExecutorErrorCode.SHAPE_MISMATCH,
                message=(
                    f"chain stages count ({len(chain.stages)}) != "
                    f"layer_ranges count ({len(chain.layer_ranges)})"
                ),
            )

        # Step 1: prompt → initial activation.
        try:
            activation = self._prompt_encoder(request.prompt)
        except Exception as exc:  # noqa: BLE001
            raise ChainExecutionError(
                stage_index=-1,
                stage_node_id="",
                code=ExecutorErrorCode.PROMPT_ENCODE_ERROR,
                message=f"prompt_encoder raised: {exc.__class__.__name__}: {exc}",
            ) from exc

        # Compute deadline once for the whole chain. Per-stage tokens
        # all carry this deadline; stages enforce it locally.
        deadline_unix = self._clock() + self._default_deadline_seconds
        chain_total = len(chain.stages)
        outcomes: List[StageOutcome] = []

        # Step 2: walk the chain.
        for stage_index, (stage_node_id, layer_range) in enumerate(
            zip(chain.stages, chain.layer_ranges)
        ):
            # _dispatch_stage returns the verified response AND the
            # decoded next-stage activation. The streaming path
            # assembles chunks internally; the inline path decodes the
            # response.activation_blob. Both produce the same numpy
            # array shape — execute_chain stays mode-opaque.
            response, activation = self._dispatch_stage(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                layer_range=tuple(layer_range),
                activation=activation,
                request=request,
                chain_total=chain_total,
                deadline_unix=deadline_unix,
            )

            outcomes.append(StageOutcome(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                duration_seconds=response.duration_seconds,
                tee_attestation=response.tee_attestation,
                tee_type=response.tee_type,
                epsilon_spent=response.epsilon_spent,
            ))

        # Step 4: decode final activation → output string.
        try:
            output_text = self._output_decoder(activation)
        except Exception as exc:  # noqa: BLE001
            raise ChainExecutionError(
                stage_index=chain_total - 1,
                stage_node_id=chain.stages[-1],
                code=ExecutorErrorCode.OUTPUT_DECODE_ERROR,
                message=f"output_decoder raised: {exc.__class__.__name__}: {exc}",
            ) from exc

        # Step 5: aggregate per-stage signals into the Phase 3.x.6
        # ChainExecutionResult Protocol shape. Per-stage TEE
        # attestations ride inside the tee_attestation field via the
        # Phase 3.x.7 Task 5 multi-stage envelope; the receipt
        # signature commits to all per-stage attestations because
        # signing_payload() hex-encodes the full bytes.
        stage_attestations = [
            StageAttestation(
                stage_index=outcome.stage_index,
                stage_node_id=outcome.stage_node_id,
                tee_type=outcome.tee_type,
                attestation=outcome.tee_attestation,
            )
            for outcome in outcomes
        ]
        return ChainExecutionResult(
            output=output_text,
            duration_seconds=sum(s.duration_seconds for s in outcomes),
            tee_attestation=encode_multi_stage_attestation(stage_attestations),
            tee_type=worst_case_tee_type(stage_attestations),
            epsilon_spent=sum(s.epsilon_spent for s in outcomes),
        )

    # ── streaming-token API (Phase 3.x.8) ────────────────────────────

    def execute_chain_streaming(
        self,
        *,
        request: InferenceRequest,
        chain: GPUChain,
    ) -> Iterator[Union[StreamToken, ChainExecutionResult]]:
        """Streaming counterpart to ``execute_chain``. Yields
        ``StreamToken`` objects as the tail stage produces them; the
        LAST yielded item is a ``ChainExecutionResult`` carrying the
        signed multi-stage receipt over the joined output text.

        Phases:
          1. Prefill — non-tail stages run their existing one-shot
             forward pass via ``_dispatch_stage`` (inline OR
             3.x.7.1-streamed activations both supported on the way
             through). Each yields a ``StageOutcome`` for the final
             aggregation.
          2. Decode — the tail stage is dispatched via
             ``token_stream_send_message`` with ``streaming=True``;
             each ``TokenFrame`` becomes a ``StreamToken`` yielded to
             the caller; the terminal ``StreamFinalFrame`` carries
             the signed ``RunLayerSliceResponse`` whose
             ``activation_blob`` is the UTF-8-encoded joined output.
             The H2 invariant is preserved: the tail's signature
             must verify under the EXPECTED ``stage_node_id``
             (caller-supplied), not the response self-claim.

        Caller MUST exhaust the iterator (or call its ``.close()``)
        otherwise the tail stage's stream leaks. Phase 3.x.8 Task 6
        (cancellation propagation) wires the .close() → tail
        cancellation signal end-to-end.

        Failure modes (raised as ``ChainExecutionError`` to the
        caller — same surface as ``execute_chain``):
          - empty chain / shape mismatch → EMPTY_CHAIN /
            SHAPE_MISMATCH
          - prompt encode raises → PROMPT_ENCODE_ERROR
          - no token_stream_send_message wired →
            STREAMING_NOT_WIRED
          - non-tail prefill stage failure → forwarded from
            ``_dispatch_stage``
          - transport failure on tail dispatch → TRANSPORT_ERROR
          - garbage / wrong-type frame → MALFORMED_RESPONSE
          - server-side StageError → forwarded with its code
          - tail signature fails anchor verify →
            INVALID_STAGE_SIGNATURE
          - joined deltas don't match signed activation_blob →
            MALFORMED_RESPONSE
        """
        if not chain.stages:
            raise ChainExecutionError(
                stage_index=-1,
                stage_node_id="",
                code=ExecutorErrorCode.EMPTY_CHAIN,
                message="GPUChain has no stages — router produced an empty chain",
            )
        if len(chain.stages) != len(chain.layer_ranges):
            raise ChainExecutionError(
                stage_index=-1,
                stage_node_id="",
                code=ExecutorErrorCode.SHAPE_MISMATCH,
                message=(
                    f"chain stages count ({len(chain.stages)}) != "
                    f"layer_ranges count ({len(chain.layer_ranges)})"
                ),
            )
        if self._token_stream_send is None:
            raise ChainExecutionError(
                stage_index=-1,
                stage_node_id="",
                code=ExecutorErrorCode.STREAMING_NOT_WIRED,
                message=(
                    "execute_chain_streaming requires "
                    "token_stream_send_message= at executor "
                    "construction time"
                ),
            )

        # Step 1: prompt → initial activation.
        try:
            activation = self._prompt_encoder(request.prompt)
        except Exception as exc:  # noqa: BLE001
            raise ChainExecutionError(
                stage_index=-1,
                stage_node_id="",
                code=ExecutorErrorCode.PROMPT_ENCODE_ERROR,
                message=f"prompt_encoder raised: {exc.__class__.__name__}: {exc}",
            ) from exc

        deadline_unix = self._clock() + self._default_deadline_seconds
        chain_total = len(chain.stages)
        outcomes: List[StageOutcome] = []

        # Step 2: prefill non-tail stages via the existing unary path.
        # This composes with 3.x.7.1 chunked-streaming on the prefill
        # automatically (large activations between non-tail stages
        # ride through _dispatch_stage's branch).
        for stage_index in range(chain_total - 1):
            stage_node_id = chain.stages[stage_index]
            layer_range = tuple(chain.layer_ranges[stage_index])
            response, activation = self._dispatch_stage(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                layer_range=layer_range,
                activation=activation,
                request=request,
                chain_total=chain_total,
                deadline_unix=deadline_unix,
            )
            outcomes.append(StageOutcome(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                duration_seconds=response.duration_seconds,
                tee_attestation=response.tee_attestation,
                tee_type=response.tee_type,
                epsilon_spent=response.epsilon_spent,
            ))

        # Step 3: dispatch the tail stage with streaming=True. Yields
        # StreamToken... StreamToken... ChainExecutionResult.
        tail_index = chain_total - 1
        tail_node_id = chain.stages[-1]
        tail_layer_range = tuple(chain.layer_ranges[-1])
        yield from self._dispatch_streaming_tail(
            stage_index=tail_index,
            stage_node_id=tail_node_id,
            layer_range=tail_layer_range,
            activation=activation,
            request=request,
            chain_total=chain_total,
            deadline_unix=deadline_unix,
            outcomes_so_far=outcomes,
        )

    # ── streaming tail dispatch (Phase 3.x.8) ────────────────────────

    def _dispatch_streaming_tail(
        self,
        *,
        stage_index: int,
        stage_node_id: str,
        layer_range: tuple,
        activation: np.ndarray,
        request: InferenceRequest,
        chain_total: int,
        deadline_unix: float,
        outcomes_so_far: List[StageOutcome],
    ) -> Iterator[Union[StreamToken, ChainExecutionResult]]:
        """Tail-stage streaming dispatch. Encodes the activation
        inline, mints a tail-bound token, sends a streaming request
        via ``token_stream_send_message``, parses each response
        frame, yields a ``StreamToken`` per ``TokenFrame`` and a
        terminal ``ChainExecutionResult`` after anchor-verifying
        the embedded response.

        v1: inline activations only on the streaming-input side
        (matches the server-side handle_token_stream contract). If
        the activation exceeds the inline threshold, surfaces as
        ``ACTIVATION_TOO_LARGE`` — chunked-input + streaming-output
        composition is a Task 7 follow-up.
        """
        # Encode activation for the wire.
        try:
            blob, shape, dtype_str = encode_activation(activation)
        except ActivationCodecError as exc:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.ACTIVATION_INVALID,
                message=f"tail-stage input encode failed: {exc}",
            ) from exc

        # v1: streaming + chunked-input composition not yet wired.
        if should_chunk(blob, threshold=self._chunk_threshold_bytes):
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.ACTIVATION_TOO_LARGE,
                message=(
                    f"tail-stage activation {len(blob)} bytes exceeds "
                    f"inline threshold {self._chunk_threshold_bytes}; "
                    f"streaming + chunked-input composition is a "
                    f"Phase 3.x.8 Task 7 follow-up"
                ),
            )

        # Mint a token bound to the tail stage.
        token = HandoffToken.sign(
            identity=self._settler,
            request_id=request.request_id,
            chain_stage_index=stage_index,
            chain_total_stages=chain_total,
            deadline_unix=deadline_unix,
        )

        # Build the streaming RunLayerSliceRequest (streaming=True).
        wire_request = RunLayerSliceRequest(
            request_id=request.request_id,
            model_id=request.model_id,
            layer_range=layer_range,
            privacy_tier=request.privacy_tier,
            content_tier=request.content_tier,
            activation_blob=blob,
            activation_shape=shape,
            activation_dtype=dtype_str,
            upstream_token=token,
            deadline_unix=deadline_unix,
            streaming=True,
        )
        request_bytes = encode_message(wire_request)

        # Send via the token-stream transport.
        address = self._resolve_address(stage_node_id)
        try:
            frame_iter = self._token_stream_send(address, request_bytes)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.TRANSPORT_ERROR,
                message=(
                    f"token-stream transport raised: "
                    f"{exc.__class__.__name__}: {exc}"
                ),
            ) from exc

        # Iterate response frames. Track sequence + joined text for
        # the integrity check against the signed StreamFinalFrame.
        expected_seq = 0
        joined_parts: List[str] = []
        final_frame: Optional[StreamFinalFrame] = None

        # Phase 3.x.8 Task 6 — cancellation cleanup. The outer
        # try/finally ensures the upstream frame_iter is .close()'d
        # on EVERY exit path: caller GeneratorExit (stops iterating
        # this executor generator), ChainExecutionError, or normal
        # completion. Without this, a caller closing the executor's
        # generator could leave the upstream stream alive until the
        # next gc cycle — undesirable for production transports
        # holding network sockets / KV cache state.
        #
        # v1 honest scope: cancellation = clean upstream cleanup,
        # NOT delivery of a partial-output ChainExecutionResult.
        # Python's GeneratorExit semantics forbid yielding additional
        # values after .close(); a partial-receipt-on-cancel pathway
        # would require either an in-band cancel-sentinel via
        # .send() or a side-channel inspection API — both deferred
        # to a Phase 3.x.8.x follow-up.
        try:
            for raw_frame in frame_iter:
                msg = self._parse_stream_frame(
                    stage_index=stage_index,
                    stage_node_id=stage_node_id,
                    raw=raw_frame,
                )

                # Server-side error terminal — forward.
                if isinstance(msg, StageError):
                    raise ChainExecutionError(
                        stage_index=stage_index,
                        stage_node_id=stage_node_id,
                        code=msg.code,
                        message=msg.message,
                    )

                if isinstance(msg, TokenFrame):
                    # Cross-field check: TokenFrame must carry the
                    # parent request_id (relay defense — a network
                    # relay can't splice frames from one stream into
                    # another).
                    if msg.request_id != request.request_id:
                        raise ChainExecutionError(
                            stage_index=stage_index,
                            stage_node_id=stage_node_id,
                            code=ExecutorErrorCode.MALFORMED_RESPONSE,
                            message=(
                                f"TokenFrame.request_id "
                                f"{msg.request_id!r} != sent "
                                f"{request.request_id!r}"
                            ),
                        )
                    # Sequence-index invariant: 0-indexed, strictly
                    # increasing, no gaps.
                    if msg.sequence_index != expected_seq:
                        raise ChainExecutionError(
                            stage_index=stage_index,
                            stage_node_id=stage_node_id,
                            code=ExecutorErrorCode.MALFORMED_RESPONSE,
                            message=(
                                f"TokenFrame sequence_index="
                                f"{msg.sequence_index} (expected "
                                f"{expected_seq})"
                            ),
                        )
                    expected_seq += 1
                    joined_parts.append(msg.text_delta)
                    yield StreamToken(
                        sequence_index=msg.sequence_index,
                        text_delta=msg.text_delta,
                        token_id=msg.token_id,
                        finish_reason=msg.finish_reason,
                    )
                    continue

                if isinstance(msg, StreamFinalFrame):
                    final_frame = msg
                    # Don't break — defensive: continue iterating to
                    # ensure the iterator is exhausted, but reject
                    # any frames that arrive AFTER the terminal.
                    # Actually: a well-behaved server emits exactly
                    # one StreamFinalFrame and stops. Treat extra
                    # frames as MALFORMED_RESPONSE.
                    break

                # Unreachable given parse_stream_frame's whitelist,
                # but defense-in-depth.
                raise ChainExecutionError(
                    stage_index=stage_index,
                    stage_node_id=stage_node_id,
                    code=ExecutorErrorCode.MALFORMED_RESPONSE,
                    message=(
                        f"unexpected wire type in token-stream: "
                        f"{type(msg).__name__}"
                    ),
                )

            # If frame_iter yields more after StreamFinalFrame, the
            # server is sending stale frames — reject.
            if final_frame is not None:
                for stray in frame_iter:
                    raise ChainExecutionError(
                        stage_index=stage_index,
                        stage_node_id=stage_node_id,
                        code=ExecutorErrorCode.MALFORMED_RESPONSE,
                        message=(
                            "frames received after StreamFinalFrame; "
                            "server should terminate the stream at the "
                            "terminal frame"
                        ),
                    )
        except ChainExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.TRANSPORT_ERROR,
                message=(
                    f"token-stream iterator raised: "
                    f"{exc.__class__.__name__}: {exc}"
                ),
            ) from exc
        finally:
            # Explicit close on every exit path. Mirrors the
            # server-side _dispatch_token_stream cleanup. Generator
            # iterators support .close() natively; non-generator
            # iterators that implement the protocol get the call
            # forwarded; iterators without .close() are silently
            # skipped via getattr.
            close = getattr(frame_iter, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    # Cleanup-time exceptions are swallowed —
                    # never propagate through GeneratorExit. By
                    # the time this fires the caller has already
                    # decided to stop iterating; a close-time
                    # exception is non-actionable.
                    logger.debug(
                        "RpcChainExecutor: token-stream frame_iter "
                        "close() raised during cleanup "
                        "(stage_node_id=%r); swallowed",
                        stage_node_id,
                    )

        if final_frame is None:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.MALFORMED_RESPONSE,
                message=(
                    "token-stream ended without a StreamFinalFrame "
                    "(no signed receipt material received)"
                ),
            )

        # Cross-field check: response.request_id must echo our request.
        response = final_frame.response
        if response.request_id != request.request_id:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.MALFORMED_RESPONSE,
                message=(
                    f"StreamFinalFrame.response.request_id "
                    f"{response.request_id!r} != sent "
                    f"{request.request_id!r}"
                ),
            )

        # H2 invariant — anchor-verify under the EXPECTED stage_node_id
        # (caller-supplied), NOT the response's self-claim. Any
        # anchor-registered peer impersonating another is rejected at
        # verify_with_anchor's cross-field check.
        if not response.verify_with_anchor(
            self._anchor, expected_stage_node_id=stage_node_id
        ):
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.INVALID_STAGE_SIGNATURE,
                message=(
                    f"tail-stage StreamFinalFrame.response signature "
                    f"failed anchor verification (expected "
                    f"stage_node_id={stage_node_id!r}, claimed="
                    f"{response.stage_node_id!r})"
                ),
            )

        # Joined-text invariant: the deltas we received MUST match
        # what the stage signed in activation_blob. Defense-in-depth
        # against a relay that tampers TokenFrame.text_delta — even
        # though the server enforces this BEFORE signing, the
        # consumer re-checks against the signed bytes.
        joined = "".join(joined_parts)
        if joined.encode("utf-8") != response.activation_blob:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.MALFORMED_RESPONSE,
                message=(
                    "joined TokenFrame text_deltas do not match "
                    "StreamFinalFrame.response.activation_blob (a "
                    "relay may have tampered an intermediate token "
                    "frame)"
                ),
            )

        # Build + yield the terminal ChainExecutionResult.
        tail_outcome = StageOutcome(
            stage_index=stage_index,
            stage_node_id=stage_node_id,
            duration_seconds=response.duration_seconds,
            tee_attestation=response.tee_attestation,
            tee_type=response.tee_type,
            epsilon_spent=response.epsilon_spent,
        )
        all_outcomes = list(outcomes_so_far) + [tail_outcome]
        stage_attestations = [
            StageAttestation(
                stage_index=outcome.stage_index,
                stage_node_id=outcome.stage_node_id,
                tee_type=outcome.tee_type,
                attestation=outcome.tee_attestation,
            )
            for outcome in all_outcomes
        ]
        yield ChainExecutionResult(
            output=joined,
            duration_seconds=sum(s.duration_seconds for s in all_outcomes),
            tee_attestation=encode_multi_stage_attestation(stage_attestations),
            tee_type=worst_case_tee_type(stage_attestations),
            epsilon_spent=sum(s.epsilon_spent for s in all_outcomes),
        )

    def _parse_stream_frame(
        self,
        *,
        stage_index: int,
        stage_node_id: str,
        raw: bytes,
    ) -> Union[TokenFrame, StreamFinalFrame, StageError]:
        """Parse a single token-stream wire frame; map protocol-level
        parse errors to ``ChainExecutionError``."""
        try:
            msg = parse_message(raw)
        except ChainRpcVersionMismatchError as exc:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.UNSUPPORTED_VERSION,
                message=str(exc),
            ) from exc
        except (
            ChainRpcMalformedError,
            ChainRpcUnknownTypeError,
            ChainRpcProtocolError,
        ) as exc:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.MALFORMED_RESPONSE,
                message=str(exc),
            ) from exc
        if not isinstance(msg, (TokenFrame, StreamFinalFrame, StageError)):
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.MALFORMED_RESPONSE,
                message=(
                    f"unexpected wire type in token-stream: "
                    f"{type(msg).__name__}"
                ),
            )
        return msg

    # ── stage dispatch ────────────────────────────────────────────────

    def _dispatch_stage(
        self,
        *,
        stage_index: int,
        stage_node_id: str,
        layer_range: tuple,
        activation: np.ndarray,
        request: InferenceRequest,
        chain_total: int,
        deadline_unix: float,
    ) -> Tuple[RunLayerSliceResponse, np.ndarray]:
        """Mint token → encode request → send → parse + verify response
        → decode output activation.

        Branches between the inline path (``_dispatch_inline``) and the
        v2 streamed path (``_dispatch_streamed``) based on the encoded
        blob size. Returns the verified response paired with the
        decoded next-stage activation array — execute_chain stays
        mode-opaque.

        Every failure path raises ``ChainExecutionError`` with a
        specific code.
        """
        # Encode activation for the wire (cheap: numpy copy + tobytes).
        try:
            blob, shape, dtype_str = encode_activation(activation)
        except ActivationCodecError as exc:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.ACTIVATION_INVALID,
                message=f"input encode failed: {exc}",
            ) from exc

        # Mint a token bound to this stage.
        token = HandoffToken.sign(
            identity=self._settler,
            request_id=request.request_id,
            chain_stage_index=stage_index,
            chain_total_stages=chain_total,
            deadline_unix=deadline_unix,
        )

        if should_chunk(blob, threshold=self._chunk_threshold_bytes):
            return self._dispatch_streamed(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                layer_range=layer_range,
                request=request,
                token=token,
                deadline_unix=deadline_unix,
                blob=blob,
                shape=shape,
                dtype_str=dtype_str,
            )
        return self._dispatch_inline(
            stage_index=stage_index,
            stage_node_id=stage_node_id,
            layer_range=layer_range,
            request=request,
            token=token,
            deadline_unix=deadline_unix,
            blob=blob,
            shape=shape,
            dtype_str=dtype_str,
        )

    # ── inline path ──────────────────────────────────────────────────

    def _dispatch_inline(
        self,
        *,
        stage_index: int,
        stage_node_id: str,
        layer_range: tuple,
        request: InferenceRequest,
        token: HandoffToken,
        deadline_unix: float,
        blob: bytes,
        shape: tuple,
        dtype_str: str,
    ) -> Tuple[RunLayerSliceResponse, np.ndarray]:
        """v1 inline path: activation rides hex-encoded inside the JSON
        envelope; response activation rides the same way."""
        wire_request = RunLayerSliceRequest(
            request_id=request.request_id,
            model_id=request.model_id,
            layer_range=layer_range,
            privacy_tier=request.privacy_tier,
            content_tier=request.content_tier,
            activation_blob=blob,
            activation_shape=shape,
            activation_dtype=dtype_str,
            upstream_token=token,
            deadline_unix=deadline_unix,
        )
        request_bytes = encode_message(wire_request)

        # Resolve transport address + send.
        address = self._resolve_address(stage_node_id)
        try:
            response_bytes = self._send(address, request_bytes)
        except Exception as exc:  # noqa: BLE001
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.TRANSPORT_ERROR,
                message=f"transport raised: {exc.__class__.__name__}: {exc}",
            ) from exc

        response = self._parse_and_verify_response(
            stage_index=stage_index,
            stage_node_id=stage_node_id,
            request=request,
            response_bytes=response_bytes,
        )

        # Decode the inline activation blob → next-stage input.
        try:
            output_activation = decode_activation(
                response.activation_blob,
                response.activation_shape,
                response.activation_dtype,
            )
        except ActivationCodecError as exc:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.ACTIVATION_INVALID,
                message=str(exc),
            ) from exc

        return response, output_activation

    # ── streamed path (v2) ───────────────────────────────────────────

    def _dispatch_streamed(
        self,
        *,
        stage_index: int,
        stage_node_id: str,
        layer_range: tuple,
        request: InferenceRequest,
        token: HandoffToken,
        deadline_unix: float,
        blob: bytes,
        shape: tuple,
        dtype_str: str,
    ) -> Tuple[RunLayerSliceResponse, np.ndarray]:
        """v2 streamed path: activation chunked via Phase 6
        ``ShardChunker``; chunks ride out-of-band over the streaming
        transport. Manifest carries the cryptographic commitment to
        the to-be-assembled bytes; stage signature commits to the
        manifest's ``payload_sha256`` via the v2 signing payload."""
        if self._streamed_send is None:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.ACTIVATION_TOO_LARGE,
                message=(
                    f"activation blob {len(blob)} bytes exceeds inline "
                    f"threshold {self._chunk_threshold_bytes}; streamed "
                    f"transport not wired (pass streamed_send_message= "
                    f"to the executor / make_rpc_chain_executor factory)"
                ),
            )

        # Chunk the activation. Bound the chunk_id to (request_id,
        # stage_index) so a relay can't splice chunks from one stream
        # into another.
        activation_id = f"{request.request_id}::stage-{stage_index}::out-from-prev"
        try:
            chunked = chunk_activation(
                np.frombuffer(blob, dtype=np.dtype(dtype_str)).reshape(shape),
                activation_id=activation_id,
                chunk_bytes=self._chunk_bytes,
            )
        except ActivationCodecError as exc:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.ACTIVATION_INVALID,
                message=f"chunk encoding failed: {exc}",
            ) from exc

        # Build the streamed RunLayerSliceRequest: empty inline blob,
        # manifest carrying the commitment.
        wire_request = RunLayerSliceRequest(
            request_id=request.request_id,
            model_id=request.model_id,
            layer_range=layer_range,
            privacy_tier=request.privacy_tier,
            content_tier=request.content_tier,
            activation_blob=b"",
            activation_shape=shape,
            activation_dtype=dtype_str,
            upstream_token=token,
            deadline_unix=deadline_unix,
            activation_manifest=chunked.manifest,
        )
        manifest_bytes = encode_message(wire_request)

        # Encode each chunk as an ActivationChunk wire frame.
        chunk_frames: List[bytes] = []
        for shard_chunk in chunked.chunks:
            frame = ActivationChunk(
                request_id=request.request_id,
                sequence=shard_chunk.sequence,
                data=shard_chunk.data,
                chunk_sha256=shard_chunk.chunk_sha256,
            )
            chunk_frames.append(encode_message(frame))

        # Send via the streaming transport.
        address = self._resolve_address(stage_node_id)
        try:
            response_manifest_bytes, response_chunk_frames = self._streamed_send(
                address, manifest_bytes, iter(chunk_frames)
            )
        except Exception as exc:  # noqa: BLE001
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.TRANSPORT_ERROR,
                message=(
                    f"streamed transport raised: "
                    f"{exc.__class__.__name__}: {exc}"
                ),
            ) from exc

        response = self._parse_and_verify_response(
            stage_index=stage_index,
            stage_node_id=stage_node_id,
            request=request,
            response_bytes=response_manifest_bytes,
        )

        # Streamed responses MUST carry a manifest. If the server sent
        # an inline response on a streamed request, that's a protocol
        # violation; we reject rather than silently consume bytes.
        if response.activation_manifest is None:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.MALFORMED_RESPONSE,
                message=(
                    "streamed dispatch: server returned an inline "
                    "response (no activation_manifest). Streamed "
                    "requests require streamed responses in v1.0."
                ),
            )

        # H1 + H2 round-1 remediation: validate the response envelope
        # BEFORE consuming any response chunk frames. Even though the
        # response is anchor-verified above (so an authentic stage
        # can't lie about envelope shape — that's the H2 server-side
        # signing fix), defense-in-depth: a future weakness in either
        # signing layer or a buggy stage shouldn't translate into the
        # client allocating unbounded memory.
        envelope_err = self._validate_streamed_response_envelope(response)
        if envelope_err is not None:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.ACTIVATION_INVALID,
                message=envelope_err,
            )

        # Reassemble the response chunks. ShardAssembler (Phase 6)
        # enforces strict in-order delivery + per-chunk + overall
        # sha256 — failures map to ACTIVATION_INVALID. ActivationChunk
        # frames carry the executor's request_id (relay defense), but
        # the assembler needs each ShardChunk to carry the response
        # manifest's shard_id; we rewrap accordingly.
        try:
            shard_chunks = list(
                self._iter_response_chunks(
                    response_chunk_frames,
                    expected_request_id=request.request_id,
                    manifest_shard_id=response.activation_manifest.shard_id,
                    expected_total_chunks=response.activation_manifest.total_chunks,
                )
            )
        except ChainExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.TRANSPORT_ERROR,
                message=(
                    f"streamed response chunk-iter raised: "
                    f"{exc.__class__.__name__}: {exc}"
                ),
            ) from exc

        # Wrap response.activation_manifest + response chunks +
        # response.activation_shape/dtype in a ChunkedActivation-shaped
        # tuple so reassemble_chunked has what it needs.
        from prsm.compute.chain_rpc.activation_codec import ChunkedActivation
        chunked_response = ChunkedActivation(
            manifest=response.activation_manifest,
            chunks=shard_chunks,
            shape=response.activation_shape,
            dtype_str=response.activation_dtype,
        )
        try:
            output_activation = reassemble_chunked(
                chunked_response, chunks=shard_chunks
            )
        except ActivationCodecError as exc:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.ACTIVATION_INVALID,
                message=f"streamed reassembly failed: {exc}",
            ) from exc

        return response, output_activation

    def _validate_streamed_response_envelope(
        self, response: RunLayerSliceResponse
    ) -> Optional[str]:
        """H1 + H2 round-1 remediation: client-side envelope sanity
        check, mirrors ``LayerStageServer._validate_streamed_envelope``.
        Returns ``None`` if the response manifest's envelope is
        consistent with the response's shape/dtype AND below the
        client's ``max_streamed_payload_bytes`` ceiling. Otherwise
        returns an error message string.

        Defense-in-depth against a peer (or future signing-layer
        weakness) that ships an inflated envelope to coerce client-
        side memory allocation during reassembly.
        """
        m = response.activation_manifest
        if m is None:  # caller already checked, but be explicit.
            return "streamed response missing activation_manifest"
        try:
            dtype = np.dtype(response.activation_dtype)
        except TypeError:
            return (
                f"streamed response: unrecognized activation_dtype "
                f"{response.activation_dtype!r}"
            )
        expected_payload_bytes = (
            int(np.prod(response.activation_shape)) * dtype.itemsize
        )
        if m.payload_bytes != expected_payload_bytes:
            return (
                f"streamed response: manifest.payload_bytes "
                f"({m.payload_bytes}) does not match shape "
                f"{response.activation_shape} × dtype "
                f"{response.activation_dtype} ({expected_payload_bytes})"
            )
        if m.payload_bytes > self._max_streamed_payload_bytes:
            return (
                f"streamed response: manifest.payload_bytes "
                f"({m.payload_bytes}) exceeds max_streamed_payload_bytes "
                f"({self._max_streamed_payload_bytes})"
            )
        if m.total_chunks > 0:
            expected_total_chunks = (
                m.payload_bytes + m.chunk_bytes - 1
            ) // m.chunk_bytes
            if m.total_chunks != expected_total_chunks:
                return (
                    f"streamed response: manifest.total_chunks "
                    f"({m.total_chunks}) inconsistent with payload_bytes "
                    f"({m.payload_bytes}) / chunk_bytes ({m.chunk_bytes}); "
                    f"expected {expected_total_chunks}"
                )
        elif m.payload_bytes > 0:
            return (
                f"streamed response: manifest.total_chunks == 0 but "
                f"payload_bytes == {m.payload_bytes}"
            )
        return None

    @staticmethod
    def _iter_response_chunks(
        frame_iter: Iterable[bytes],
        *,
        expected_request_id: str,
        manifest_shard_id: str,
        expected_total_chunks: int,
    ) -> Iterable[ShardChunk]:
        """Decode each ActivationChunk wire frame back to a Phase 6
        ShardChunk.

        Two bindings:
          - ActivationChunk.request_id MUST match the executor's
            request_id (relay-defense — a network relay can't splice
            chunks from one stream into another).
          - The reconstructed ShardChunk.shard_id is set to the
            response manifest's shard_id so ShardAssembler's per-chunk
            validation can correlate manifest ↔ chunks.

        H1 round-1 remediation: bounded by ``expected_total_chunks``.
        A peer that keeps streaming frames past the manifest-promised
        count gets ``ActivationCodecError("excess chunks")`` rather
        than unbounded memory consumption.
        """
        produced = 0
        for raw in frame_iter:
            if produced >= expected_total_chunks:
                raise ActivationCodecError(
                    f"streamed response: peer shipped excess chunks "
                    f"(expected_total_chunks={expected_total_chunks})"
                )
            try:
                msg = parse_message(raw)
            except (
                ChainRpcMalformedError,
                ChainRpcUnknownTypeError,
                ChainRpcProtocolError,
                ChainRpcVersionMismatchError,
            ) as exc:
                raise ActivationCodecError(
                    f"streamed response chunk frame parse failed: {exc}"
                ) from exc
            if not isinstance(msg, ActivationChunk):
                raise ActivationCodecError(
                    f"streamed response chunk frame: expected "
                    f"ActivationChunk, got {type(msg).__name__}"
                )
            if msg.request_id != expected_request_id:
                raise ActivationCodecError(
                    f"streamed response chunk request_id mismatch: "
                    f"got {msg.request_id!r}, expected "
                    f"{expected_request_id!r}"
                )
            yield ShardChunk(
                shard_id=manifest_shard_id,
                sequence=msg.sequence,
                data=msg.data,
                chunk_sha256=msg.chunk_sha256,
            )

    # ── shared response parse + verify ───────────────────────────────

    def _parse_and_verify_response(
        self,
        *,
        stage_index: int,
        stage_node_id: str,
        request: InferenceRequest,
        response_bytes: bytes,
    ) -> RunLayerSliceResponse:
        """Parse + cross-field check + anchor-verify the response.
        Shared by inline + streamed paths."""
        try:
            response = parse_message(response_bytes)
        except ChainRpcVersionMismatchError as exc:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.UNSUPPORTED_VERSION,
                message=str(exc),
            ) from exc
        except (
            ChainRpcMalformedError,
            ChainRpcUnknownTypeError,
            ChainRpcProtocolError,
        ) as exc:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.MALFORMED_RESPONSE,
                message=str(exc),
            ) from exc

        # Server returned a structured StageError → forward as-is.
        if isinstance(response, StageError):
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=response.code,
                message=response.message,
            )

        # Anything other than RunLayerSliceResponse at this point is
        # a protocol-violation by the peer.
        if not isinstance(response, RunLayerSliceResponse):
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.MALFORMED_RESPONSE,
                message=(
                    f"server returned {type(response).__name__}; "
                    f"expected RunLayerSliceResponse"
                ),
            )

        # Cross-field consistency: server's response must echo our
        # request_id. Mismatch could be a server bug or an adversarial
        # relay swapping responses between concurrent requests.
        if response.request_id != request.request_id:
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.MALFORMED_RESPONSE,
                message=(
                    f"response request_id {response.request_id!r} != "
                    f"sent {request.request_id!r}"
                ),
            )

        # Anchor-verify against the EXPECTED identity (Phase 3.x.7 H2
        # invariant). Substitution by any anchor-registered peer
        # rejected at the cross-field check inside verify_with_anchor.
        if not response.verify_with_anchor(
            self._anchor, expected_stage_node_id=stage_node_id
        ):
            raise ChainExecutionError(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                code=ExecutorErrorCode.INVALID_STAGE_SIGNATURE,
                message=(
                    f"stage response signature failed anchor verification "
                    f"(expected stage_node_id={stage_node_id!r}, "
                    f"claimed={response.stage_node_id!r})"
                ),
            )

        return response


# Aggregation helpers (worst-case TEE selection + envelope encoding)
# live in prsm.compute.inference.multi_stage_attestation per Phase 3.x.7
# Task 5 — kept module-level so the receipt-side verification helper
# can reuse the same encoding without depending on this client module.
