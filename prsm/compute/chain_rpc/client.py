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
from typing import Callable, List, Optional, Protocol

import numpy as np

from prsm.compute.chain_rpc.activation_codec import (
    ActivationCodecError,
    decode_activation,
    encode_activation,
)
from prsm.compute.chain_rpc.protocol import (
    ChainRpcMalformedError,
    ChainRpcProtocolError,
    ChainRpcUnknownTypeError,
    ChainRpcVersionMismatchError,
    HandoffToken,
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    StageError,
    encode_message,
    parse_message,
)
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

        self._settler = settler_identity
        self._send = send_message
        self._anchor = anchor
        self._prompt_encoder = prompt_encoder
        self._output_decoder = output_decoder
        self._resolve_address = address_resolver or (lambda nid: nid)
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
            response = self._dispatch_stage(
                stage_index=stage_index,
                stage_node_id=stage_node_id,
                layer_range=tuple(layer_range),
                activation=activation,
                request=request,
                chain_total=chain_total,
                deadline_unix=deadline_unix,
            )

            # Step 3: thread output activation into next stage.
            try:
                activation = decode_activation(
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
    ) -> RunLayerSliceResponse:
        """Mint token → encode request → send → parse + verify response.

        Every failure path raises ``ChainExecutionError`` with a
        specific code. The caller (``execute_chain``) lets it
        propagate.
        """
        # Encode activation for the wire.
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

        # Parse the response. Mirrors the server's parse-error
        # taxonomy (version mismatch is its own bucket).
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
        # request_id. Mismatch could be a server bug or an
        # adversarial relay swapping responses between concurrent
        # requests.
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

        # Stage signature must verify under the EXPECTED stage's
        # anchor pubkey — i.e., the node we dispatched to, not
        # whatever identity the response self-declares. The
        # ``expected_stage_node_id`` kwarg makes the API impossible to
        # misuse: a substituted response signed by Mallory (any anchor-
        # registered identity) fails because Alice's pubkey doesn't
        # verify Mallory's signature, regardless of whether Mallory
        # IS anchor-registered. Failure here is the strongest signal
        # the server returned an unauthentic response — caller may
        # choose to fire a Phase 7.1 challenge for this code.
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
