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
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Iterable, Iterator, List, Optional, Protocol, Tuple, Union

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
    MAX_VERIFY_BATCH_TOKENS,
    ActivationChunk,
    ChainRpcMalformedError,
    ChainRpcProtocolError,
    ChainRpcUnknownTypeError,
    ChainRpcVersionMismatchError,
    DecodeMode,
    EvictCacheRequest,
    EvictCacheResponse,
    HandoffToken,
    RollbackCacheRequest,
    RollbackCacheResponse,
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
    IterationAttestation,
    StageAttestation,
    encode_multi_iteration_attestation,
    encode_multi_stage_attestation,
    worst_case_tee_type,
    worst_case_tee_type_across_iterations,
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
    # Phase 3.x.11: tail stage in sharded mode returned a response
    # missing next_token_id (caller bug — tail server-side runner
    # not tail-capable, or response signing predates 3.x.11).
    TAIL_TOKEN_MISSING = "TAIL_TOKEN_MISSING"


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
        # ── Phase 3.x.11 sharded-decode opt-in ─────────────────────────
        enable_sharded_decode: bool = False,
        tokenizer: Optional[object] = None,
        cache_evict_send_message: Optional[SendMessage] = None,
        sharded_default_max_tokens: int = 512,
        # ── Phase 3.x.11.y speculative-decoding opt-in ─────────────────
        # When ``draft_model`` is wired, the executor's sharded-
        # decode streaming path branches to the speculative loop:
        # draft.propose → chain VERIFY → accept → rollback → emit.
        # When None (default), behaves exactly as Phase 3.x.11
        # single-token decode (greedy-equivalent output).
        # ``speculation_depth`` is the K in K+1 (the count of draft
        # tokens proposed per round); the chain forwards K+1 with
        # the parent. Cap at MAX_VERIFY_BATCH_TOKENS - 1 so the
        # K+1 batch fits the response-side cap. Greedy-only at v1
        # — temperature > 0 raises (Leviathan-2023 sampling-correct
        # speculation deferred to Phase 3.x.11.y.x).
        draft_model: Optional[Any] = None,
        speculation_depth: int = 4,
        rollback_cache_send_message: Optional[SendMessage] = None,
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
        # Phase 3.x.11 sharded-decode validation. Tail-only sharded
        # decode opt-in requires a tokenizer at the executor boundary
        # (encode prompt → input_ids; decode token_id → text_delta).
        # ``cache_evict_send_message`` is optional — when wired, the
        # executor broadcasts EvictCacheRequest (Task 6 wire format)
        # to every chain stage on terminal/cancellation; when not
        # wired, eviction is a no-op + the operator-side TTL sweeper
        # bounds the leak window.
        if enable_sharded_decode:
            if tokenizer is None:
                raise RuntimeError(
                    "RpcChainExecutor: enable_sharded_decode=True "
                    "requires tokenizer= (used to encode prompt → "
                    "input_ids + decode token_id → text_delta)"
                )
            if (
                not hasattr(tokenizer, "encode")
                or not callable(getattr(tokenizer, "encode", None))
                or not hasattr(tokenizer, "decode")
                or not callable(getattr(tokenizer, "decode", None))
            ):
                raise RuntimeError(
                    "RpcChainExecutor: tokenizer must expose .encode + "
                    ".decode (HF AutoTokenizer-shaped)"
                )
        if cache_evict_send_message is not None and not callable(
            cache_evict_send_message,
        ):
            raise RuntimeError(
                "RpcChainExecutor: cache_evict_send_message must be "
                "callable if provided"
            )
        if (
            isinstance(sharded_default_max_tokens, bool)
            or not isinstance(sharded_default_max_tokens, int)
            or sharded_default_max_tokens <= 0
        ):
            raise ValueError(
                f"sharded_default_max_tokens must be positive int, "
                f"got {sharded_default_max_tokens!r}"
            )
        # Phase 3.x.11.y — speculative-decoding validation.
        if draft_model is not None:
            if not enable_sharded_decode:
                raise RuntimeError(
                    "RpcChainExecutor: draft_model= requires "
                    "enable_sharded_decode=True (speculation is a "
                    "sharded-decode optimization)"
                )
            for attr in ("reset", "propose", "commit", "evict"):
                if not callable(getattr(draft_model, attr, None)):
                    raise RuntimeError(
                        f"RpcChainExecutor: draft_model must "
                        f"implement {attr}(...) per DraftModel "
                        f"Protocol — see prsm.compute.inference.draft_model"
                    )
        if (
            isinstance(speculation_depth, bool)
            or not isinstance(speculation_depth, int)
            or speculation_depth <= 0
        ):
            raise ValueError(
                f"speculation_depth must be positive int, got "
                f"{speculation_depth!r}"
            )
        if speculation_depth >= MAX_VERIFY_BATCH_TOKENS:
            raise ValueError(
                f"speculation_depth {speculation_depth} exceeds K cap "
                f"{MAX_VERIFY_BATCH_TOKENS - 1} (K drafts produce K+1 "
                f"verified positions; K must satisfy K+1 <= "
                f"{MAX_VERIFY_BATCH_TOKENS})"
            )
        if rollback_cache_send_message is not None and not callable(
            rollback_cache_send_message,
        ):
            raise RuntimeError(
                "RpcChainExecutor: rollback_cache_send_message must "
                "be callable if provided"
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
        self._enable_sharded_decode = bool(enable_sharded_decode)
        self._tokenizer = tokenizer
        self._cache_evict_send = cache_evict_send_message
        self._sharded_default_max_tokens = int(sharded_default_max_tokens)
        self._draft_model = draft_model
        self._speculation_depth = int(speculation_depth)
        self._rollback_cache_send = rollback_cache_send_message

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
        # Phase 3.x.11 sharded-decode branch — per-token chain
        # redispatch with KV-cache handoff. Skips the
        # token_stream_send_message wire entirely (sharded uses
        # only unary dispatches). Phase 3.x.11.y: when
        # ``draft_model`` is wired, branch further into the
        # speculative-decoding loop (chain VERIFY + rollback).
        if self._enable_sharded_decode:
            if self._draft_model is not None:
                yield from self._execute_chain_streaming_sharded_speculative(
                    request=request, chain=chain,
                )
            else:
                yield from self._execute_chain_streaming_sharded(
                    request=request, chain=chain,
                )
            return
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

    # ── Phase 3.x.11 sharded-decode path ─────────────────────────────

    def _execute_chain_streaming_sharded(
        self,
        *,
        request: InferenceRequest,
        chain: GPUChain,
    ) -> Iterator[Union[StreamToken, ChainExecutionResult]]:
        """Sharded-decode streaming. Per-token chain redispatch with
        KV-cache handoff: each chain pass produces one new token.

        Wire flow per token:
          - Stage 1 input  = input_ids (np.int64; full prompt on
                             PREFILL, single previous token on
                             INCREMENTAL)
          - Stages 2..T    = hidden state (np.float* per the model)
          - Tail response  = activation (residual hidden state for
                             the receipt) + ``next_token_id`` +
                             ``is_terminal``

        The executor:
          1. Encodes prompt → input_ids (tokenizer at executor
             boundary; the model on each stage operates on
             tensors).
          2. PREFILL pass: walks the chain once; tail returns the
             FIRST generated token id.
          3. Decode loop: repeats unary chain passes with
             ``decode_mode=INCREMENTAL`` and the last token as
             input; stops on ``is_terminal=True`` from the tail or
             when ``request.max_tokens`` is reached.
          4. Yields ``StreamToken`` per token, ``ChainExecutionResult``
             with the signed multi-stage receipt at terminal.
          5. ``finally`` block broadcasts ``EvictCacheRequest`` to
             every chain stage on terminal/cancellation. Eviction
             is best-effort — if the wire isn't wired
             (``cache_evict_send_message=None``), it's a no-op and
             operators rely on the server-side TTL sweeper for
             leaked-handle cleanup.

        Honest scope:
          - Per-token wire tax: T network round-trips per token.
            Pipelining is Phase 3.x.11.x.
          - Tier C: not yet supported (each per-token dispatch is
            a fresh timing surface). Phase 3.x.11.q.
        """
        # Encode prompt → input_ids.
        try:
            initial_input_ids = self._tokenizer.encode(request.prompt)
        except Exception as exc:  # noqa: BLE001
            raise ChainExecutionError(
                stage_index=-1,
                stage_node_id="",
                code=ExecutorErrorCode.PROMPT_ENCODE_ERROR,
                message=(
                    f"tokenizer.encode raised: "
                    f"{exc.__class__.__name__}: {exc}"
                ),
            ) from exc
        if not isinstance(initial_input_ids, list) or not initial_input_ids:
            raise ChainExecutionError(
                stage_index=-1,
                stage_node_id="",
                code=ExecutorErrorCode.PROMPT_ENCODE_ERROR,
                message=(
                    f"tokenizer.encode must return a non-empty list of "
                    f"int token ids; got "
                    f"{type(initial_input_ids).__name__}"
                ),
            )

        deadline_unix = self._clock() + self._default_deadline_seconds
        chain_total = len(chain.stages)
        # Phase 3.x.11.x Task 2: per-iteration accumulation. Each
        # inner list is one chain pass (PREFILL + each INCREMENTAL).
        # Receipt-emit step encodes via the multi-iteration envelope
        # so verifiers can confirm "stage K served EVERY dispatch,
        # not just the last one".
        per_iteration_outcomes: List[List[StageOutcome]] = []
        per_iteration_decode_modes: List[DecodeMode] = []
        output_text_parts: List[str] = []
        max_tokens = (
            int(getattr(request, "max_tokens", None) or 0)
            or self._sharded_default_max_tokens
        )

        try:
            # Step 1: PREFILL pass — produces token #1.
            next_token_id, is_terminal, prefill_outcomes = (
                self._run_chain_iteration_sharded(
                    request=request,
                    chain=chain,
                    decode_mode=DecodeMode.PREFILL,
                    input_ids=initial_input_ids,
                    deadline_unix=deadline_unix,
                )
            )
            per_iteration_outcomes.append(prefill_outcomes)
            per_iteration_decode_modes.append(DecodeMode.PREFILL)
            text_delta = self._decode_token_to_text(next_token_id)
            output_text_parts.append(text_delta)
            yield StreamToken(
                sequence_index=0,
                text_delta=text_delta,
                token_id=next_token_id,
                finish_reason=("stop" if is_terminal else None),
            )

            # Step 2: INCREMENTAL decode loop.
            sequence_idx = 1
            while not is_terminal and sequence_idx < max_tokens:
                next_token_id, is_terminal, inc_outcomes = (
                    self._run_chain_iteration_sharded(
                        request=request,
                        chain=chain,
                        decode_mode=DecodeMode.INCREMENTAL,
                        input_ids=[next_token_id],
                        deadline_unix=deadline_unix,
                    )
                )
                per_iteration_outcomes.append(inc_outcomes)
                per_iteration_decode_modes.append(DecodeMode.INCREMENTAL)
                text_delta = self._decode_token_to_text(next_token_id)
                output_text_parts.append(text_delta)
                yield StreamToken(
                    sequence_index=sequence_idx,
                    text_delta=text_delta,
                    token_id=next_token_id,
                    finish_reason=(
                        "stop" if is_terminal
                        else (
                            "max_tokens"
                            if sequence_idx + 1 >= max_tokens
                            else None
                        )
                    ),
                )
                sequence_idx += 1

            # Step 3: emit ChainExecutionResult with per-iteration
            # attestation envelope. Phase 3.x.11.x Task 2 — closes
            # the "no per-iteration cryptographic commitment" gap
            # from the Phase 3.x.11 threat-model addendum §3.2.
            yield self._build_sharded_chain_result(
                output_text="".join(output_text_parts),
                per_iteration_outcomes=per_iteration_outcomes,
                per_iteration_decode_modes=per_iteration_decode_modes,
            )
        finally:
            # Broadcast eviction on EVERY exit path (terminal or
            # caller GeneratorExit). Best-effort — never fail the
            # main flow. Operator-side TTL sweeper bounds the
            # leak window if eviction misses.
            self._evict_cache_on_stages(
                chain=chain, request_id=request.request_id,
            )

    def _run_chain_iteration_sharded(
        self,
        *,
        request: InferenceRequest,
        chain: GPUChain,
        decode_mode: DecodeMode,
        input_ids: List[int],
        deadline_unix: float,
    ) -> Tuple[int, bool, List[StageOutcome]]:
        """One forward pass of the chain. Stage 1 receives
        ``input_ids`` (encoded as ``np.int64``); stages 2..T
        receive the prior stage's hidden state.

        Returns ``(next_token_id, is_terminal, per_stage_outcomes)``.
        Tail must populate ``next_token_id`` — TAIL_TOKEN_MISSING
        if it doesn't (server-side runner wasn't tail-capable).
        """
        chain_total = len(chain.stages)
        outcomes: List[StageOutcome] = []
        # Stage 1 input: input_ids as np.int64 array. The model
        # adapter on the server side detects "dtype=int64 +
        # decode_mode=PREFILL/INCREMENTAL" and routes to the
        # token-embedding path before the layer slice; downstream
        # stages see floating-point hidden states.
        activation = np.array(input_ids, dtype=np.int64)
        next_token_id: int = -1
        is_terminal = False
        last_response: Optional[RunLayerSliceResponse] = None

        for stage_index in range(chain_total):
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
                decode_mode=decode_mode,
                include_sampling_fields=True,
            )
            outcomes.append(StageOutcome(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                duration_seconds=response.duration_seconds,
                tee_attestation=response.tee_attestation,
                tee_type=response.tee_type,
                epsilon_spent=response.epsilon_spent,
            ))
            last_response = response

        # Tail must populate next_token_id. Non-tail responses
        # leave it None — that's expected; only the TAIL is
        # required to fill it. We check after the loop because the
        # tail is by definition the LAST stage.
        if last_response is None or last_response.next_token_id is None:
            tail_node = chain.stages[-1]
            raise ChainExecutionError(
                stage_index=chain_total - 1,
                stage_node_id=tail_node,
                code=ExecutorErrorCode.TAIL_TOKEN_MISSING,
                message=(
                    f"sharded decode: tail stage {tail_node!r} "
                    f"returned a response with next_token_id=None; "
                    f"server-side runner must be tail-capable "
                    f"(see ShardedAutoregressiveRunner sampling_defaults)"
                ),
            )
        next_token_id = int(last_response.next_token_id)
        is_terminal = bool(last_response.is_terminal)
        return next_token_id, is_terminal, outcomes

    # ── Phase 3.x.11.y speculative-decoding path ─────────────────────

    def _execute_chain_streaming_sharded_speculative(
        self,
        *,
        request: InferenceRequest,
        chain: GPUChain,
    ) -> Iterator[Union[StreamToken, ChainExecutionResult]]:
        """Speculative-decoding sharded streaming. Same shape as
        ``_execute_chain_streaming_sharded`` (PREFILL → loop →
        terminal receipt) but the loop dispatches VERIFY rounds
        instead of single-token INCREMENTALs.

        Per VERIFY round (cost = T network round-trips for
        accepted_count + 1 emitted tokens):
          1. ``draft.propose(parent=last_emitted, k=K)`` — draft
             model produces K candidate tokens.
          2. Dispatch ``decode_mode=VERIFY`` with input
             ``[parent, d_1, ..., d_K]`` to Stage 1; intermediate
             stages forward batched K+1 hidden states; tail
             returns ``verified_token_ids`` (K+1 argmaxes) +
             ``accepted_count`` (longest matching prefix).
          3. Emit ``verified[: accepted_count + 1]`` —
             accepted_count + 1 tokens (last is verifier's
             correction or bonus).
          4. ``draft.commit(accepted_token_ids=emitted)`` — draft
             rolls its state forward.
          5. If ``accepted_count < K``: broadcast
             ``RollbackCacheRequest(K - accepted_count)`` to all
             chain stages (drop the speculatively-cached-but-
             rejected suffix).
          6. Continue with last emitted token as next parent.

        Routing (Phase 3.x.11.y.x):
          - ``temperature == 0.0`` (or unset) → v1 (greedy)
            path. Calls ``draft.propose`` (no probs); chain
            VERIFY returns K+1 verified ids; tail's
            ``apply_lm_head_and_sample_batch`` argmaxes; emits
            longest accepted prefix + greedy correction/bonus.
            Bit-identical to non-speculative greedy.
          - ``temperature > 0`` → v2 (Leviathan-2023 stochastic)
            path. Requires ``draft.propose_with_probs`` capability
            (raised if absent). Chain VERIFY carries
            ``proposed_token_probs``; tail's
            ``apply_lm_head_and_sample_batch_with_rejection``
            performs §2.2 rejection-sampling and returns
            ``accepted_count + 1`` verified ids (last is
            correction OR bonus). Marginal output distribution
            matches the non-speculative softmax(p/T) sampling.

        Adaptive K (Phase 3.x.11.y.x §Task 5):
          - Per-request rolling-window of last 4 rounds'
            ``(accepted_count, K)`` pairs. Once the window is
            full, every round recomputes the smoothed accept-
            rate ``Σ ac / Σ K`` and adapts K for the NEXT round:
              · rate < 0.25 → K //= 2 (floor 1)
              · rate > 0.75 → K *= 2 (cap MAX_VERIFY_BATCH_TOKENS - 1)
            K stays unchanged in the [0.25, 0.75] band. Initial
            K is the constructor's ``speculation_depth``.

        v1+v2 shared honest scope:
          - **Per-iteration accept-rate timing surface.**
            ``accepted_count`` is observable on the wire — the
            operator/network observer learns the acceptance-rate
            distribution. Tier C remains structurally denied for
            sharded decode (Phase 3.x.11.q deferred).
        """
        # Routing decision. v1 (greedy) preserved bit-identical
        # via temp == 0.0 path; v2 (stochastic) requires
        # propose_with_probs capability + tail-side v2 routing.
        request_temp = getattr(request, "temperature", None)
        use_stochastic = (
            request_temp is not None and float(request_temp) > 0.0
        )
        if use_stochastic and not callable(
            getattr(self._draft_model, "propose_with_probs", None),
        ):
            raise ChainExecutionError(
                stage_index=-1,
                stage_node_id="",
                code=ExecutorErrorCode.PROMPT_ENCODE_ERROR,
                message=(
                    f"speculative decoding at temperature={request_temp} "
                    f"requires DraftModel.propose_with_probs(...) "
                    f"capability (Phase 3.x.11.y.x §Task 2); the "
                    f"configured draft_model does not implement it. "
                    f"Either set temperature=0.0 (v1 greedy path) or "
                    f"upgrade the draft model — see "
                    f"prsm.compute.inference.draft_model.HFDraftModel."
                ),
            )

        # Encode prompt → input_ids (same as non-speculative path).
        try:
            initial_input_ids = self._tokenizer.encode(request.prompt)
        except Exception as exc:  # noqa: BLE001
            raise ChainExecutionError(
                stage_index=-1,
                stage_node_id="",
                code=ExecutorErrorCode.PROMPT_ENCODE_ERROR,
                message=(
                    f"tokenizer.encode raised: "
                    f"{exc.__class__.__name__}: {exc}"
                ),
            ) from exc
        if not isinstance(initial_input_ids, list) or not initial_input_ids:
            raise ChainExecutionError(
                stage_index=-1,
                stage_node_id="",
                code=ExecutorErrorCode.PROMPT_ENCODE_ERROR,
                message=(
                    f"tokenizer.encode must return a non-empty list of "
                    f"int token ids; got "
                    f"{type(initial_input_ids).__name__}"
                ),
            )

        deadline_unix = self._clock() + self._default_deadline_seconds
        per_iteration_outcomes: List[List[StageOutcome]] = []
        per_iteration_decode_modes: List[DecodeMode] = []
        output_text_parts: List[str] = []
        max_tokens = (
            int(getattr(request, "max_tokens", None) or 0)
            or self._sharded_default_max_tokens
        )
        k = self._speculation_depth
        # Adaptive K rolling-window state. Each entry is
        # (accepted_count, K_at_that_round). Once full (4 rounds),
        # smoothed accept-rate Σ ac / Σ K drives next-round K.
        accept_history: Deque[Tuple[int, int]] = deque(maxlen=4)
        k_max = MAX_VERIFY_BATCH_TOKENS - 1

        # Initialize draft model BEFORE PREFILL dispatch. Caller
        # cancellation between PREFILL and the first yield is rare
        # but real (transport stall, deadline elapse): doing reset
        # up-front means the finally-block draft.evict always has
        # state to clean up. Mirrors the chain-side cache: the
        # server allocates on PREFILL receipt; the draft is the
        # client-side mirror of that.
        self._draft_model.reset(
            request_id=request.request_id,
            prompt_input_ids=initial_input_ids,
        )
        draft_reset_done = True
        try:
            # Step 1: PREFILL pass — produces token #1.
            next_token_id, is_terminal, prefill_outcomes = (
                self._run_chain_iteration_sharded(
                    request=request,
                    chain=chain,
                    decode_mode=DecodeMode.PREFILL,
                    input_ids=initial_input_ids,
                    deadline_unix=deadline_unix,
                )
            )
            per_iteration_outcomes.append(prefill_outcomes)
            per_iteration_decode_modes.append(DecodeMode.PREFILL)
            text_delta = self._decode_token_to_text(next_token_id)
            output_text_parts.append(text_delta)
            yield StreamToken(
                sequence_index=0,
                text_delta=text_delta,
                token_id=next_token_id,
                finish_reason=("stop" if is_terminal else None),
            )
            tokens_emitted = 1

            # Step 2: speculative VERIFY loop.
            while not is_terminal and tokens_emitted < max_tokens:
                # Step 2a: draft proposes K tokens. v1 path uses
                # propose (greedy, probs implicit); v2 path uses
                # propose_with_probs and forwards q(d_i) on the
                # wire so the tail can run §2.2 rejection-sampling.
                proposed_probs: Optional[List[float]] = None
                if use_stochastic:
                    propose_with_probs = (
                        self._draft_model.propose_with_probs
                    )
                    proposed, probs_out = propose_with_probs(
                        request_id=request.request_id,
                        parent_token_id=next_token_id,
                        k=k,
                        temperature=float(request_temp),
                    )
                    proposed_probs = (
                        [float(p) for p in probs_out]
                        if probs_out is not None else None
                    )
                else:
                    proposed = self._draft_model.propose(
                        request_id=request.request_id,
                        parent_token_id=next_token_id,
                        k=k,
                        temperature=0.0,
                    )
                if not isinstance(proposed, list) or not proposed:
                    raise ChainExecutionError(
                        stage_index=-1,
                        stage_node_id="",
                        code=ExecutorErrorCode.MALFORMED_RESPONSE,
                        message=(
                            f"draft_model.propose returned "
                            f"{type(proposed).__name__}; expected "
                            f"non-empty list of int token ids"
                        ),
                    )
                if (
                    use_stochastic
                    and (
                        proposed_probs is None
                        or len(proposed_probs) != len(proposed)
                    )
                ):
                    raise ChainExecutionError(
                        stage_index=-1,
                        stage_node_id="",
                        code=ExecutorErrorCode.MALFORMED_RESPONSE,
                        message=(
                            f"draft_model.propose_with_probs returned "
                            f"probs of length "
                            f"{None if proposed_probs is None else len(proposed_probs)} "
                            f"but proposed K={len(proposed)}; the two "
                            f"must be co-set with equal length"
                        ),
                    )
                # Step 2b: chain VERIFY.
                k_round = len(proposed)
                (
                    verified,
                    accepted_count,
                    chain_terminal,
                    verify_outcomes,
                ) = self._run_chain_iteration_sharded_verify(
                    request=request,
                    chain=chain,
                    parent_token_id=next_token_id,
                    proposed_token_ids=proposed,
                    proposed_token_probs=proposed_probs,
                    deadline_unix=deadline_unix,
                )
                per_iteration_outcomes.append(verify_outcomes)
                per_iteration_decode_modes.append(DecodeMode.VERIFY)

                # Step 2c: emit accepted_count + 1 tokens. Cap
                # against max_tokens — if we'd overshoot, truncate
                # the emit + flag terminal; the rollback below
                # accounts for the truncation.
                # Round-1 MEDIUM-2 remediation: split the
                # cap-bound check (>=, terminates the loop) from
                # the actual-truncation flag (>, drives
                # finish_reason="max_tokens"). At exact-cap-hit
                # (==), the natural "stop" path applies if the
                # chain flagged terminal; otherwise "max_tokens"
                # is still the correct semantic since we're at
                # the cap. ``cap_hit_mid_emit`` only fires when
                # we ACTUALLY truncated the emitted slice.
                emitted = list(verified[: accepted_count + 1])
                cap_hit_mid_emit = False
                if tokens_emitted + len(emitted) > max_tokens:
                    emitted = emitted[: max_tokens - tokens_emitted]
                    cap_hit_mid_emit = True
                cap_reached = (
                    tokens_emitted + len(emitted) >= max_tokens
                )
                for tok in emitted:
                    text_delta = self._decode_token_to_text(tok)
                    output_text_parts.append(text_delta)
                    finish = None
                    if (
                        chain_terminal
                        and tok == emitted[-1]
                    ):
                        finish = "stop"
                    elif (
                        cap_reached
                        and tok == emitted[-1]
                    ):
                        finish = "max_tokens"
                    yield StreamToken(
                        sequence_index=tokens_emitted,
                        text_delta=text_delta,
                        token_id=int(tok),
                        finish_reason=finish,
                    )
                    tokens_emitted += 1

                # Step 2d: draft.commit on the actually-emitted
                # tokens (handles the cap_hit_mid_emit truncation
                # case correctly — draft history reflects what
                # the user actually saw).
                if emitted:
                    self._draft_model.commit(
                        request_id=request.request_id,
                        accepted_token_ids=[int(t) for t in emitted],
                    )

                # Step 2e: rollback the rejected suffix on every
                # chain stage. Stages cached K+1 positions during
                # the verify forward (regardless of v1 vs v2 — the
                # K+1 batched forward populates K+1 KV positions).
                # ``cached_extra`` is K+1 minus what we actually
                # kept (= len(emitted)). NOTE: in v2, len(verified)
                # == accepted_count + 1 (NOT K+1), so we MUST NOT
                # use len(verified) as the cached count — it would
                # under-count the rollback in the v2 partial-accept
                # case. Compute against k_round (this round's K)
                # instead. cap_hit_mid_emit case: emitted shorter
                # than accepted_count + 1, so cached_extra grows.
                cached_extra = (k_round + 1) - len(emitted)
                if cached_extra > 0:
                    self._rollback_cache_on_stages(
                        chain=chain,
                        request_id=request.request_id,
                        n_positions_to_drop=cached_extra,
                    )

                # Step 2g: adaptive K — v2-only to preserve v1's
                # bit-identical-to-Phase-3.x.11.y regression
                # (round-1 review L3). Record this round's
                # (accepted_count, K) and once the rolling window
                # is full, recompute smoothed accept-rate and
                # adjust K for the NEXT round. Halve below 25%,
                # double above 75%, hold in [25%, 75%]. Cap at
                # MAX_VERIFY_BATCH_TOKENS - 1; floor at 1. K is
                # an executor-side parameter; the chain stages
                # adapt naturally to the new batch size on the
                # next VERIFY dispatch.
                if use_stochastic:
                    accept_history.append((accepted_count, k_round))
                    if len(accept_history) == accept_history.maxlen:
                        total_ac = sum(a for a, _ in accept_history)
                        total_k = sum(kk for _, kk in accept_history)
                        rate = (
                            (total_ac / total_k) if total_k > 0 else 0.0
                        )
                        if rate < 0.25:
                            k = max(1, k // 2)
                        elif rate > 0.75:
                            k = min(k_max, k * 2)

                # Step 2f: terminal logic. Chain reports terminal
                # iff EOS in emitted OR tokens_generated reached
                # max_tokens server-side. We also enforce the
                # client-side cap.
                if cap_reached:
                    is_terminal = True
                elif chain_terminal:
                    is_terminal = True
                else:
                    # Continue speculation with the LAST emitted
                    # token as parent (= verified[accepted_count]
                    # in the no-cap case; = the truncated last in
                    # cap_hit case but cap_reached terminates anyway).
                    next_token_id = int(emitted[-1])

            # Step 3: emit ChainExecutionResult.
            yield self._build_sharded_chain_result(
                output_text="".join(output_text_parts),
                per_iteration_outcomes=per_iteration_outcomes,
                per_iteration_decode_modes=per_iteration_decode_modes,
            )
        finally:
            # Eviction on every exit path — terminal, cancellation,
            # exception. Best-effort (server-side TTL bounds the
            # leak window if broadcast misses).
            if draft_reset_done:
                try:
                    self._draft_model.evict(
                        request_id=request.request_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "RpcChainExecutor: draft_model.evict raised "
                        "%s: %s — proceeding with chain eviction",
                        exc.__class__.__name__, exc,
                    )
            self._evict_cache_on_stages(
                chain=chain, request_id=request.request_id,
            )

    def _run_chain_iteration_sharded_verify(
        self,
        *,
        request: InferenceRequest,
        chain: GPUChain,
        parent_token_id: int,
        proposed_token_ids: List[int],
        proposed_token_probs: Optional[List[float]],
        deadline_unix: float,
    ) -> Tuple[Tuple[int, ...], int, bool, List[StageOutcome]]:
        """One VERIFY pass through the chain.

        Stage 1 input is ``[parent_token_id, d_1, ..., d_K]`` — K+1
        token ids encoded as ``np.int64``. Intermediate stages
        forward the K+1 batch of hidden states. Tail returns
        ``verified_token_ids`` (K+1 argmaxes) + ``accepted_count``
        (longest-prefix match) + ``next_token_id`` (last emitted)
        + ``is_terminal``.

        Returns
        ``(verified_token_ids, accepted_count, is_terminal,
        per_stage_outcomes)``. Tail must populate
        ``verified_token_ids`` + ``accepted_count`` —
        ``MALFORMED_RESPONSE`` if absent (server-side runner
        wasn't VERIFY-tail-capable).
        """
        chain_total = len(chain.stages)
        outcomes: List[StageOutcome] = []
        # Stage 1 input: K+1 token ids as np.int64. Server
        # adapter detects "dtype=int64 + decode_mode=VERIFY"
        # and routes to the token-embedding path before the
        # batched K+1 forward.
        verify_input_ids = [int(parent_token_id)] + [
            int(t) for t in proposed_token_ids
        ]
        activation = np.array(verify_input_ids, dtype=np.int64)
        proposed_tuple: Tuple[int, ...] = tuple(
            int(t) for t in proposed_token_ids
        )
        proposed_probs_tuple: Optional[Tuple[float, ...]] = (
            tuple(float(p) for p in proposed_token_probs)
            if proposed_token_probs is not None else None
        )
        last_response: Optional[RunLayerSliceResponse] = None

        for stage_index in range(chain_total):
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
                decode_mode=DecodeMode.VERIFY,
                include_sampling_fields=True,
                proposed_token_ids=proposed_tuple,
                proposed_token_probs=proposed_probs_tuple,
            )
            outcomes.append(StageOutcome(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                duration_seconds=response.duration_seconds,
                tee_attestation=response.tee_attestation,
                tee_type=response.tee_type,
                epsilon_spent=response.epsilon_spent,
            ))
            last_response = response

        if (
            last_response is None
            or last_response.verified_token_ids is None
            or last_response.accepted_count is None
        ):
            tail_node = chain.stages[-1]
            raise ChainExecutionError(
                stage_index=chain_total - 1,
                stage_node_id=tail_node,
                code=ExecutorErrorCode.MALFORMED_RESPONSE,
                message=(
                    f"sharded VERIFY: tail stage {tail_node!r} "
                    f"returned response without "
                    f"verified_token_ids/accepted_count; server-"
                    f"side runner must be VERIFY-tail-capable "
                    f"(see ShardedAutoregressiveRunner forward_verify "
                    f"+ apply_lm_head_and_sample_batch)"
                ),
            )
        verified = tuple(int(t) for t in last_response.verified_token_ids)
        accepted_count = int(last_response.accepted_count)
        # v1 (greedy / proposed_token_probs is None): expect K+1
        # verified entries (longest-prefix match + correction or
        # bonus). v2 (Leviathan-2023 stochastic): expect
        # accepted_count + 1 entries (rejection-sampled correction
        # OR bonus is the last entry; the rejected proposals are
        # NOT echoed back). Validate the right invariant per path.
        if proposed_token_probs is None:
            expected_len = len(proposed_token_ids) + 1
            if len(verified) != expected_len:
                tail_node = chain.stages[-1]
                raise ChainExecutionError(
                    stage_index=chain_total - 1,
                    stage_node_id=tail_node,
                    code=ExecutorErrorCode.MALFORMED_RESPONSE,
                    message=(
                        f"sharded VERIFY (v1 greedy): tail returned "
                        f"{len(verified)} verified tokens but expected "
                        f"K+1={expected_len}"
                    ),
                )
        else:
            expected_len = accepted_count + 1
            if (
                accepted_count < 0
                or accepted_count > len(proposed_token_ids)
                or len(verified) != expected_len
            ):
                tail_node = chain.stages[-1]
                raise ChainExecutionError(
                    stage_index=chain_total - 1,
                    stage_node_id=tail_node,
                    code=ExecutorErrorCode.MALFORMED_RESPONSE,
                    message=(
                        f"sharded VERIFY (v2 stochastic): tail returned "
                        f"accepted_count={accepted_count} with "
                        f"{len(verified)} verified tokens; expected "
                        f"accepted_count in [0, K={len(proposed_token_ids)}] "
                        f"and len(verified) == accepted_count + 1"
                    ),
                )
        is_terminal = bool(last_response.is_terminal)
        return verified, accepted_count, is_terminal, outcomes

    def _rollback_cache_on_stages(
        self,
        *,
        chain: GPUChain,
        request_id: str,
        n_positions_to_drop: int,
    ) -> None:
        """Best-effort RollbackCacheRequest broadcast. Encodes a
        ``RollbackCacheRequest`` and sends it via
        ``rollback_cache_send_message`` to every chain stage.
        Server-side handler routes to ``KVCacheManager.rollback``
        (Phase 3.x.11.y Task 6).

        Never raises — rollback is best-effort by design (server-
        side TTL sweeper bounds the leak window if a broadcast
        misses; the manager is idempotent so duplicate broadcasts
        from retried close paths are safe).
        """
        if self._rollback_cache_send is None or n_positions_to_drop <= 0:
            return
        try:
            rollback_bytes = encode_message(
                RollbackCacheRequest(
                    request_id=request_id,
                    n_positions_to_drop=int(n_positions_to_drop),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RpcChainExecutor: failed to encode "
                "RollbackCacheRequest for request_id=%r: %s: %s — "
                "relying on server-side TTL sweeper",
                request_id, exc.__class__.__name__, exc,
            )
            return
        for stage_node_id in chain.stages:
            try:
                address = self._resolve_address(stage_node_id)
                response_bytes = self._rollback_cache_send(
                    address, rollback_bytes,
                )
                if response_bytes:
                    try:
                        ack = parse_message(response_bytes)
                        if not isinstance(ack, RollbackCacheResponse):
                            logger.warning(
                                "RpcChainExecutor: stage_node_id=%r "
                                "returned non-RollbackCacheResponse "
                                "to rollback broadcast (%s) — TTL "
                                "sweeper will close the gap",
                                stage_node_id, type(ack).__name__,
                            )
                    except ChainRpcProtocolError as exc:
                        logger.warning(
                            "RpcChainExecutor: stage_node_id=%r "
                            "rollback ack failed to parse: %s",
                            stage_node_id, exc,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RpcChainExecutor: rollback broadcast to "
                    "stage_node_id=%r raised %s: %s — relying on "
                    "server-side TTL sweeper",
                    stage_node_id, exc.__class__.__name__, exc,
                )

    def _build_sharded_chain_result(
        self,
        *,
        output_text: str,
        per_iteration_outcomes: List[List[StageOutcome]],
        per_iteration_decode_modes: List[DecodeMode],
    ) -> ChainExecutionResult:
        """Phase 3.x.11.x Task 2: build the terminal
        ``ChainExecutionResult`` for a sharded-decode stream.

        The receipt's ``tee_attestation`` field carries a
        multi-iteration envelope (Phase 3.x.11.x Task 1)
        committing to one ``StageAttestation`` per stage PER
        iteration. ``tee_type`` is the worst-case across ALL
        iterations' stages — a single SOFTWARE stage in any
        iteration drags the whole receipt to SOFTWARE.
        ``duration_seconds`` + ``epsilon_spent`` aggregate
        across every per-iteration outcome.

        Tolerates the empty-iterations case (caller may invoke
        from cancellation paths where PREFILL never completed)
        by emitting an empty-output result with the existing
        single-stage envelope encoder. The non-sharded receipt
        path (``execute_chain``) is unchanged — this helper is
        sharded-only.
        """
        if not per_iteration_outcomes:
            return ChainExecutionResult(
                output=output_text,
                duration_seconds=0.0,
                tee_attestation=b"",
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
            )

        iteration_records: List[IterationAttestation] = []
        total_duration = 0.0
        total_epsilon = 0.0
        for it_idx, outcomes in enumerate(per_iteration_outcomes):
            decode_mode = per_iteration_decode_modes[it_idx]
            ordered = sorted(outcomes, key=lambda o: o.stage_index)
            stage_records = [
                StageAttestation(
                    stage_index=outcome.stage_index,
                    stage_node_id=outcome.stage_node_id,
                    tee_type=outcome.tee_type,
                    attestation=outcome.tee_attestation,
                )
                for outcome in ordered
            ]
            iteration_records.append(IterationAttestation(
                iteration_index=it_idx,
                decode_mode=decode_mode,
                stage_records=stage_records,
            ))
            for outcome in outcomes:
                total_duration += outcome.duration_seconds
                total_epsilon += outcome.epsilon_spent

        return ChainExecutionResult(
            output=output_text,
            duration_seconds=total_duration,
            tee_attestation=encode_multi_iteration_attestation(
                iteration_records,
            ),
            tee_type=worst_case_tee_type_across_iterations(
                iteration_records,
            ),
            epsilon_spent=total_epsilon,
        )

    def _decode_token_to_text(self, token_id: int) -> str:
        """tokenizer.decode([token_id], skip_special_tokens=True)
        with a defensive fallback. v1: decode each token in
        isolation — sufficient for ASCII / single-token whole-
        word output. Multi-byte (emoji / CJK) cumulative-decode
        ordering is a Phase 3.x.11.x follow-up; production
        operators serving such workloads should keep tokenizer
        state via a wrapper that tracks the cumulative-decode
        cursor (mirrors AutoregressiveStreamingRunner's
        _HFStreamerAdapter pattern)."""
        try:
            return self._tokenizer.decode(
                [int(token_id)], skip_special_tokens=True,
            )
        except TypeError:
            # Tokenizers that don't accept skip_special_tokens kwarg.
            return self._tokenizer.decode([int(token_id)])

    def _evict_cache_on_stages(
        self,
        *,
        chain: GPUChain,
        request_id: str,
    ) -> None:
        """Best-effort eviction broadcast. Encodes an
        ``EvictCacheRequest`` (Phase 3.x.11 Task 6 wire message)
        and sends it via ``cache_evict_send_message`` to every
        chain stage. Server-side handler routes to
        ``KVCacheManager.evict``; the ack is parsed for logging
        but the executor doesn't fail the main flow on bad acks.

        Never raises — eviction is best-effort by design (server-
        side TTL sweeper bounds the leak window if a broadcast
        misses; the manager is idempotent so duplicate broadcasts
        from retried close paths are safe).
        """
        if self._cache_evict_send is None:
            return
        try:
            evict_request_bytes = encode_message(
                EvictCacheRequest(request_id=request_id),
            )
        except Exception as exc:  # noqa: BLE001
            # Encoding shouldn't realistically fail (request_id is
            # already validated upstream), but if it does, we
            # can't broadcast — fall back to TTL.
            logger.warning(
                "RpcChainExecutor: failed to encode EvictCacheRequest "
                "for request_id=%r: %s: %s — relying on server-side "
                "TTL sweeper",
                request_id, exc.__class__.__name__, exc,
            )
            return
        for stage_node_id in chain.stages:
            try:
                address = self._resolve_address(stage_node_id)
                response_bytes = self._cache_evict_send(
                    address, evict_request_bytes,
                )
                # Parse the ack opportunistically. A None /
                # empty / mis-shaped response is logged but
                # doesn't fail the broadcast.
                if response_bytes:
                    try:
                        ack = parse_message(response_bytes)
                        if not isinstance(ack, EvictCacheResponse):
                            logger.warning(
                                "RpcChainExecutor: stage_node_id=%r "
                                "returned non-EvictCacheResponse to "
                                "eviction broadcast (%s) — TTL sweeper "
                                "will close the gap",
                                stage_node_id,
                                type(ack).__name__,
                            )
                    except ChainRpcProtocolError as exc:
                        logger.warning(
                            "RpcChainExecutor: stage_node_id=%r "
                            "eviction ack failed to parse: %s",
                            stage_node_id, exc,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RpcChainExecutor: cache eviction broadcast to "
                    "stage_node_id=%r raised %s: %s — relying on "
                    "server-side TTL sweeper",
                    stage_node_id, exc.__class__.__name__, exc,
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
        # Phase 3.x.10.x: forward the originating
        # InferenceRequest's sampling overrides on the wire so the
        # downstream server's StreamingSamplingShim can hand them to
        # the runner. None values pass through (omit-when-None
        # canonical encoding preserves byte-equivalence with
        # pre-3.x.10.x signed bytes for callers that don't override).
        # Streaming-only: the unary RunLayerSliceRequest construction
        # paths leave these fields unset because non-tail stages have
        # no autoregressive decode to override.
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
            max_tokens=request.max_tokens,
            temperature=request.temperature,
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
        decode_mode: DecodeMode = DecodeMode.PREFILL,
        include_sampling_fields: bool = False,
        proposed_token_ids: Optional[Tuple[int, ...]] = None,
        proposed_token_probs: Optional[Tuple[float, ...]] = None,
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
            # Phase 3.x.11.x Task 4: chunked + sharded PREFILL is
            # supported (server's _dispatch_streamed routes to
            # _dispatch_streamed_sharded when sharded_runner is
            # wired AND decode_mode=PREFILL). INCREMENTAL stays
            # unary-only — single-position activations don't
            # benefit from chunking; surface
            # ACTIVATION_TOO_LARGE with the unary-only message.
            if (
                self._enable_sharded_decode
                and decode_mode != DecodeMode.PREFILL
            ):
                raise ChainExecutionError(
                    stage_index=stage_index,
                    stage_node_id=stage_node_id,
                    code=ExecutorErrorCode.ACTIVATION_TOO_LARGE,
                    message=(
                        f"sharded INCREMENTAL is unary-only (design "
                        f"plan §3.4); activation {len(blob)} bytes "
                        f"exceeds inline threshold "
                        f"{self._chunk_threshold_bytes}. Single-position "
                        f"INCREMENTAL activations don't benefit from "
                        f"chunking; if your hidden_dim produces > "
                        f"{self._chunk_threshold_bytes}-byte single-"
                        f"position activations the model is structurally "
                        f"incompatible with sharded decode."
                    ),
                )
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
                decode_mode=decode_mode,
                include_sampling_fields=include_sampling_fields,
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
            decode_mode=decode_mode,
            include_sampling_fields=include_sampling_fields,
            proposed_token_ids=proposed_token_ids,
            proposed_token_probs=proposed_token_probs,
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
        decode_mode: DecodeMode = DecodeMode.PREFILL,
        include_sampling_fields: bool = False,
        proposed_token_ids: Optional[Tuple[int, ...]] = None,
        proposed_token_probs: Optional[Tuple[float, ...]] = None,
    ) -> Tuple[RunLayerSliceResponse, np.ndarray]:
        """v1 inline path: activation rides hex-encoded inside the JSON
        envelope; response activation rides the same way.

        ``include_sampling_fields`` opt-in bridges Phase 3.x.10.x's
        streaming-tail-only sampling pattern to Phase 3.x.11
        sharded decode where every chain stage's response carries
        the request's sampling params. Default False preserves the
        Phase 3.x.10.x non-streaming-tail invariant (unary requests
        keep max_tokens/temperature absent on the wire — matches
        the existing TestStreamingSamplingPropagation pin).

        ``proposed_token_ids`` (Phase 3.x.11.y) carries K draft
        tokens on VERIFY dispatches. None on non-VERIFY paths
        (the wire field omits-when-None to preserve byte-
        equivalence with pre-3.x.11.y signed bytes).
        """
        max_tokens = (
            getattr(request, "max_tokens", None)
            if include_sampling_fields else None
        )
        temperature = (
            getattr(request, "temperature", None)
            if include_sampling_fields else None
        )
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
            decode_mode=decode_mode,
            max_tokens=max_tokens,
            temperature=temperature,
            proposed_token_ids=proposed_token_ids,
            proposed_token_probs=proposed_token_probs,
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
        decode_mode: DecodeMode = DecodeMode.PREFILL,
        include_sampling_fields: bool = False,
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
            decode_mode=decode_mode,
            max_tokens=(
                getattr(request, "max_tokens", None)
                if include_sampling_fields else None
            ),
            temperature=(
                getattr(request, "temperature", None)
                if include_sampling_fields else None
            ),
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
