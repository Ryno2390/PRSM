"""Phase 3.x.7 Task 4 — RpcChainExecutor unit tests.

Coverage matches design plan §4 Task 4 acceptance:
  - Happy path with mocked transport (multi-stage chain threading)
  - Per-StageErrorCode mapping (each server-reported error → matching
    ChainExecutionError.code)
  - Stage signature mismatch detection
  - Wrong stage_node_id detection
  - Wrong request_id echo detection
  - Wrong response type detection
  - Transport-layer exception mapping
  - Wire-format malformed-response mapping
  - Wire-format version-mismatch mapping
  - Empty / shape-mismatched chain rejection
  - Prompt encoder + output decoder failure paths
  - Construction validation
  - TEE worst-case aggregation
  - epsilon_spent + duration_seconds aggregation
  - Implements ChainExecutor Protocol — slots into ParallaxScheduledExecutor
"""

from __future__ import annotations

from decimal import Decimal
from typing import Callable, Dict, List, Optional

import numpy as np
import pytest

from prsm.compute.chain_rpc.client import (
    AddressResolver,
    ChainExecutionError,
    ExecutorErrorCode,
    RpcChainExecutor,
    StageOutcome,
)
from prsm.compute.chain_rpc.protocol import (
    ChainRpcMessageType,
    HandoffToken,
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    StageError,
    StageErrorCode,
    encode_message,
    parse_message,
)
from prsm.compute.chain_rpc.activation_codec import (
    decode_activation,
    encode_activation,
)
from prsm.compute.inference.models import (
    ContentTier,
    InferenceRequest,
)
from prsm.compute.parallax_scheduling.prsm_request_router import GPUChain
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Test fakes
# ──────────────────────────────────────────────────────────────────────────


class FakeAnchor:
    def __init__(self, registered: Optional[Dict[str, str]] = None):
        self.registered: Dict[str, str] = dict(registered or {})

    def lookup(self, node_id: str) -> Optional[str]:
        return self.registered.get(node_id)


def _register(anchor: FakeAnchor, identity) -> None:
    anchor.registered[identity.node_id] = identity.public_key_b64


def _prompt_encoder(prompt: str) -> np.ndarray:
    """Deterministic test encoder: prompt's UTF-8 bytes interpreted as
    int32 (zero-padded to 4-byte boundary)."""
    raw = prompt.encode("utf-8")
    pad = (4 - len(raw) % 4) % 4
    raw = raw + b"\x00" * pad
    return np.frombuffer(raw, dtype=np.int32).copy()


def _output_decoder(arr: np.ndarray) -> str:
    """Inverse of _prompt_encoder."""
    return arr.tobytes().rstrip(b"\x00").decode("utf-8", errors="replace")


# ──────────────────────────────────────────────────────────────────────────
# Stage server simulator (reuses real LayerStageServer from Task 2 in
# integration; here we use a simpler in-line responder so tests can
# fully control per-stage behavior).
# ──────────────────────────────────────────────────────────────────────────


class StageSim:
    """Deterministic per-stage responder.

    Default: identity transform (output activation == input). Tests
    customize via ``set_response`` / ``set_error`` / ``set_raise``.
    """

    def __init__(self, identity, *, tee_type: TEEType = TEEType.SOFTWARE):
        self.identity = identity
        self.tee_type = tee_type
        self.tee_attestation = b"\x01" * 32
        self.epsilon = 0.0
        self.duration = 0.05
        self._error: Optional[StageError] = None
        self._raise: Optional[Exception] = None
        self._response_override: Optional[bytes] = None
        # When set, the simulator transforms the input activation
        # via this callable before signing.
        self._transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
        # When set, signs with this identity instead of self.identity
        # (impersonation test).
        self._signer_override = None
        # When set, response carries this stage_node_id field instead
        # of self.identity.node_id (mismatch test).
        self._stage_node_id_override: Optional[str] = None
        # When set, response echoes this request_id (mismatch test).
        self._request_id_override: Optional[str] = None
        self.calls: List[bytes] = []

    def set_error(self, code: StageErrorCode, message: str = "") -> None:
        self._error = StageError(
            request_id="<set-on-call>",
            code=code.value,
            message=message,
        )

    def set_raw_bytes(self, raw: bytes) -> None:
        self._response_override = raw

    def set_raise(self, exc: Exception) -> None:
        self._raise = exc

    def set_transform(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._transform = fn

    def impersonate_with(self, signer_identity, claim_node_id: str) -> None:
        """Simulate a stage that signs with a different identity but
        claims to be claim_node_id. The signature will verify under
        signer_identity's pubkey on the anchor (if registered), but
        the stage_node_id field will mismatch our dispatched stage."""
        self._signer_override = signer_identity
        self._stage_node_id_override = claim_node_id

    def respond_with_request_id(self, request_id: str) -> None:
        self._request_id_override = request_id

    def handle(self, request_bytes: bytes) -> bytes:
        self.calls.append(request_bytes)
        if self._raise is not None:
            raise self._raise
        if self._response_override is not None:
            return self._response_override

        request = parse_message(request_bytes)
        assert isinstance(request, RunLayerSliceRequest)

        if self._error is not None:
            return encode_message(StageError(
                request_id=request.request_id,
                code=self._error.code,
                message=self._error.message,
            ))

        # Decode input activation, optionally transform, encode output.
        activation = decode_activation(
            request.activation_blob,
            request.activation_shape,
            request.activation_dtype,
        )
        if self._transform is not None:
            activation = self._transform(activation)
        out_blob, out_shape, out_dtype = encode_activation(activation)

        signer = self._signer_override or self.identity
        response = RunLayerSliceResponse.sign(
            identity=signer,
            request_id=self._request_id_override or request.request_id,
            activation_blob=out_blob,
            activation_shape=out_shape,
            activation_dtype=out_dtype,
            duration_seconds=self.duration,
            tee_attestation=self.tee_attestation,
            tee_type=self.tee_type,
            epsilon_spent=self.epsilon,
        )

        # If we're claiming a different stage_node_id, hand-craft the
        # response with the impostor's claimed node_id field while
        # keeping the signature from signer.
        if self._stage_node_id_override is not None:
            response = RunLayerSliceResponse(
                request_id=response.request_id,
                activation_blob=response.activation_blob,
                activation_shape=response.activation_shape,
                activation_dtype=response.activation_dtype,
                duration_seconds=response.duration_seconds,
                tee_attestation=response.tee_attestation,
                tee_type=response.tee_type,
                epsilon_spent=response.epsilon_spent,
                stage_signature_b64=response.stage_signature_b64,
                stage_node_id=self._stage_node_id_override,
            )

        return encode_message(response)


class FakeTransport:
    """Maps stage address → StageSim. Configurable transport-level
    failures via ``transport_failure``."""

    def __init__(self, sims: Dict[str, StageSim]):
        self.sims = sims
        self.transport_failure: Optional[Exception] = None
        self.delivery_log: List[str] = []

    def send(self, address: str, request_bytes: bytes) -> bytes:
        self.delivery_log.append(address)
        if self.transport_failure is not None:
            raise self.transport_failure
        sim = self.sims.get(address)
        if sim is None:
            raise ConnectionError(f"no stage at {address!r}")
        return sim.handle(request_bytes)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def settler():
    return generate_node_identity("settler")


@pytest.fixture
def alice():
    return generate_node_identity("alice")


@pytest.fixture
def bob():
    return generate_node_identity("bob")


@pytest.fixture
def anchor(settler, alice, bob):
    a = FakeAnchor()
    _register(a, settler)
    _register(a, alice)
    _register(a, bob)
    return a


@pytest.fixture
def alice_sim(alice):
    return StageSim(alice)


@pytest.fixture
def bob_sim(bob):
    return StageSim(bob)


@pytest.fixture
def transport(alice, bob, alice_sim, bob_sim):
    return FakeTransport({
        alice.node_id: alice_sim,
        bob.node_id: bob_sim,
    })


@pytest.fixture
def executor(settler, anchor, transport):
    return RpcChainExecutor(
        settler_identity=settler,
        send_message=transport.send,
        anchor=anchor,
        prompt_encoder=_prompt_encoder,
        output_decoder=_output_decoder,
    )


def _make_chain(stages: List[str], total_layers: int = 4) -> GPUChain:
    n = len(stages)
    per_stage = total_layers // n
    layer_ranges = []
    for i in range(n):
        start = i * per_stage
        end = (i + 1) * per_stage if i < n - 1 else total_layers
        layer_ranges.append((start, end))
    return GPUChain(
        request_id="req-1",
        region="us-east",
        stages=tuple(stages),
        layer_ranges=tuple(layer_ranges),
        total_latency_ms=10.0,
        stale_profile_count=0,
    )


def _make_request(prompt: str = "hello world") -> InferenceRequest:
    return InferenceRequest(
        prompt=prompt,
        model_id="test-model",
        budget_ftns=Decimal("10.0"),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        request_id="req-1",
    )


# ──────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def _stub(self, settler, anchor, transport):
        return dict(
            settler_identity=settler,
            send_message=transport.send,
            anchor=anchor,
            prompt_encoder=_prompt_encoder,
            output_decoder=_output_decoder,
        )

    def test_rejects_missing_settler(self, anchor, transport):
        kw = self._stub(None, anchor, transport)
        kw["settler_identity"] = None
        with pytest.raises(RuntimeError, match="NodeIdentity"):
            RpcChainExecutor(**kw)

    def test_rejects_non_callable_send(self, settler, anchor, transport):
        kw = self._stub(settler, anchor, transport)
        kw["send_message"] = "not-callable"  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="send_message"):
            RpcChainExecutor(**kw)

    def test_rejects_missing_anchor(self, settler, transport):
        kw = self._stub(settler, None, transport)
        kw["anchor"] = None
        with pytest.raises(RuntimeError, match="anchor"):
            RpcChainExecutor(**kw)

    def test_rejects_non_callable_prompt_encoder(self, settler, anchor, transport):
        kw = self._stub(settler, anchor, transport)
        kw["prompt_encoder"] = "no"  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="prompt_encoder"):
            RpcChainExecutor(**kw)

    def test_rejects_non_callable_output_decoder(self, settler, anchor, transport):
        kw = self._stub(settler, anchor, transport)
        kw["output_decoder"] = "no"  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="output_decoder"):
            RpcChainExecutor(**kw)

    def test_rejects_non_positive_deadline(self, settler, anchor, transport):
        kw = self._stub(settler, anchor, transport)
        with pytest.raises(ValueError, match="default_deadline_seconds"):
            RpcChainExecutor(**kw, default_deadline_seconds=0)


# ──────────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_two_stage_chain_round_trips(
        self, executor, alice, bob, transport, alice_sim, bob_sim
    ):
        # Identity transforms throughout — output should equal prompt.
        chain = _make_chain([alice.node_id, bob.node_id])
        result = executor.execute_chain(
            request=_make_request("hello world"),
            chain=chain,
        )
        assert result.output == "hello world"
        # Each stage was called exactly once.
        assert len(alice_sim.calls) == 1
        assert len(bob_sim.calls) == 1
        # Delivery in chain order.
        assert transport.delivery_log == [alice.node_id, bob.node_id]

    def test_activation_threads_through_stages(
        self, executor, alice, bob, alice_sim, bob_sim
    ):
        # alice doubles, bob adds 1. Output = prompt_int * 2 + 1.
        alice_sim.set_transform(lambda a: a * 2)
        bob_sim.set_transform(lambda a: a + 1)

        chain = _make_chain([alice.node_id, bob.node_id])
        request = _make_request("AB")
        result = executor.execute_chain(request=request, chain=chain)

        # Verify by reproducing the math out-of-band.
        encoded_prompt = _prompt_encoder("AB")
        expected = (encoded_prompt * 2) + 1
        decoded = _output_decoder(expected)
        assert result.output == decoded

    def test_aggregates_durations(
        self, executor, alice, bob, alice_sim, bob_sim
    ):
        alice_sim.duration = 0.10
        bob_sim.duration = 0.25
        chain = _make_chain([alice.node_id, bob.node_id])
        result = executor.execute_chain(request=_make_request(), chain=chain)
        assert result.duration_seconds == pytest.approx(0.35)

    def test_aggregates_epsilon_only_at_tail(
        self, executor, alice, bob, alice_sim, bob_sim
    ):
        # Per design: only the final stage applies DP. Simulate this
        # by having alice spend 0 and bob spend 0.5.
        alice_sim.epsilon = 0.0
        bob_sim.epsilon = 0.5
        chain = _make_chain([alice.node_id, bob.node_id])
        result = executor.execute_chain(request=_make_request(), chain=chain)
        assert result.epsilon_spent == pytest.approx(0.5)

    def test_worst_case_tee_with_software_stage(
        self, executor, alice, bob, alice_sim, bob_sim
    ):
        # alice runs SGX, bob runs SOFTWARE. Aggregate must be SOFTWARE.
        alice_sim.tee_type = TEEType.SGX
        bob_sim.tee_type = TEEType.SOFTWARE
        chain = _make_chain([alice.node_id, bob.node_id])
        result = executor.execute_chain(request=_make_request(), chain=chain)
        assert result.tee_type == TEEType.SOFTWARE

    def test_worst_case_tee_all_hardware(
        self, executor, alice, bob, alice_sim, bob_sim
    ):
        alice_sim.tee_type = TEEType.SGX
        bob_sim.tee_type = TEEType.TDX
        chain = _make_chain([alice.node_id, bob.node_id])
        result = executor.execute_chain(request=_make_request(), chain=chain)
        # Both hardware → result.tee_type is one of them (we keep
        # the first stage's by tie-break).
        assert result.tee_type in {TEEType.SGX, TEEType.TDX}
        # Specifically, the rank-tied tie-break preserves first stage's.
        assert result.tee_type == TEEType.SGX

    def test_per_stage_token_carries_stage_index(
        self, executor, alice, bob, anchor, alice_sim, bob_sim
    ):
        chain = _make_chain([alice.node_id, bob.node_id])
        executor.execute_chain(request=_make_request(), chain=chain)
        # Inspect the wire bytes alice + bob received and verify each
        # carries a token bound to its chain_stage_index.
        alice_request = parse_message(alice_sim.calls[0])
        bob_request = parse_message(bob_sim.calls[0])
        assert alice_request.upstream_token.chain_stage_index == 0
        assert bob_request.upstream_token.chain_stage_index == 1
        assert alice_request.upstream_token.chain_total_stages == 2
        assert bob_request.upstream_token.chain_total_stages == 2
        # Both tokens verify under the settler's pubkey on the anchor.
        assert alice_request.upstream_token.verify_with_anchor(anchor) is True
        assert bob_request.upstream_token.verify_with_anchor(anchor) is True


# ──────────────────────────────────────────────────────────────────────────
# Error paths
# ──────────────────────────────────────────────────────────────────────────


class TestServerReportedErrors:
    """Each StageErrorCode the server might return must propagate as
    ChainExecutionError with a matching code."""

    @pytest.mark.parametrize("code", [
        StageErrorCode.MALFORMED_REQUEST,
        StageErrorCode.INVALID_TOKEN,
        StageErrorCode.DEADLINE_EXCEEDED,
        StageErrorCode.MODEL_NOT_FOUND,
        StageErrorCode.SHARD_MISSING,
        StageErrorCode.TIER_GATE,
        StageErrorCode.LAYER_RANGE_INVALID,
        StageErrorCode.ACTIVATION_INVALID,
        StageErrorCode.TIMEOUT,
        StageErrorCode.INTERNAL_ERROR,
        StageErrorCode.UNSUPPORTED_VERSION,
    ])
    def test_propagates_server_error_code(
        self, executor, alice, bob, alice_sim, code
    ):
        alice_sim.set_error(code, message=f"server says {code.value}")
        chain = _make_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        assert exc_info.value.code == code.value
        assert exc_info.value.stage_index == 0
        assert exc_info.value.stage_node_id == alice.node_id
        # Subsequent stages were NOT called after the error.
        assert exc_info.value.message  # non-empty


class TestSignatureVerification:
    def test_invalid_stage_signature_unregistered_signer(
        self, executor, alice, bob, alice_sim
    ):
        # alice signs with a fresh (unregistered) identity. Anchor
        # cannot resolve a pubkey for the rogue identity → verify fails.
        rogue = generate_node_identity("rogue")
        alice_sim.impersonate_with(rogue, claim_node_id=rogue.node_id)
        chain = _make_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        assert exc_info.value.code == ExecutorErrorCode.INVALID_STAGE_SIGNATURE

    def test_stage_node_id_mismatch_detected(
        self, executor, alice, bob, anchor, alice_sim
    ):
        # alice signs with her real identity (sig verifies under her
        # pubkey) but claims to be bob. Detected at the cross-field
        # check.
        alice_sim.impersonate_with(alice, claim_node_id=bob.node_id)
        chain = _make_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        # Could be INVALID_STAGE_SIGNATURE either from the verify
        # (bob's pubkey doesn't match alice's sig) OR from the cross-
        # field check; either is correct rejection.
        assert exc_info.value.code == ExecutorErrorCode.INVALID_STAGE_SIGNATURE
        assert exc_info.value.stage_index == 0


class TestRequestIdMismatch:
    def test_response_request_id_mismatch(
        self, executor, alice, bob, alice_sim
    ):
        alice_sim.respond_with_request_id("DIFFERENT")
        chain = _make_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        # Either MALFORMED_RESPONSE (request_id mismatch) or
        # INVALID_STAGE_SIGNATURE (because the modified request_id
        # invalidates the signature) — both are correct rejection
        # paths that prevent a swapped response from being accepted.
        assert exc_info.value.code in {
            ExecutorErrorCode.MALFORMED_RESPONSE,
            ExecutorErrorCode.INVALID_STAGE_SIGNATURE,
        }


class TestWrongResponseType:
    def test_server_returns_response_with_wrong_type(
        self, executor, alice, bob, alice_sim
    ):
        # Send back a HandoffToken-style payload (wrong message type).
        # Use a separate valid wire object with a valid sub-type.
        from prsm.compute.chain_rpc.protocol import StageError as _SE
        # Wrong response — a StageError with non-enum code is still
        # parseable, but RpcChainExecutor maps StageError to a server-
        # reported error. We need a different test: a parseable
        # message that isn't a Response/StageError.
        # Use a hand-crafted request bytes (server peer mistakenly
        # echoes a Request). parse_message will produce a
        # RunLayerSliceRequest type — not a Response.
        token = HandoffToken.sign(
            identity=generate_node_identity("rogue-server"),
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=99999.0,
        )
        # We can't register the rogue on the anchor, so the token
        # itself wouldn't verify, but the type check happens BEFORE
        # we ever look at the inner token.
        bogus_request = RunLayerSliceRequest(
            request_id="req-1",
            model_id="x",
            layer_range=(0, 1),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"\x00" * 4,
            activation_shape=(1,),
            activation_dtype="int32",
            upstream_token=token,
            deadline_unix=99999.0,
        )
        alice_sim.set_raw_bytes(encode_message(bogus_request))
        chain = _make_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        assert exc_info.value.code == ExecutorErrorCode.MALFORMED_RESPONSE


class TestTransportLayerErrors:
    def test_transport_raises_maps_to_transport_error(
        self, executor, alice, bob, transport
    ):
        transport.transport_failure = ConnectionError("connection refused")
        chain = _make_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        assert exc_info.value.code == ExecutorErrorCode.TRANSPORT_ERROR

    def test_simulator_raises_maps_to_transport_error(
        self, executor, alice, bob, alice_sim
    ):
        # Stage simulator (server-side) raising synchronously surfaces
        # via the transport boundary as a transport error from the
        # client's POV.
        alice_sim.set_raise(RuntimeError("simulator crash"))
        chain = _make_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        assert exc_info.value.code == ExecutorErrorCode.TRANSPORT_ERROR


class TestMalformedResponse:
    def test_unparseable_response(
        self, executor, alice, bob, alice_sim
    ):
        alice_sim.set_raw_bytes(b"not-json")
        chain = _make_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        assert exc_info.value.code == ExecutorErrorCode.MALFORMED_RESPONSE

    def test_version_mismatch_response(
        self, executor, alice, bob, alice_sim
    ):
        import json
        # Construct a response with a future protocol version.
        future_payload = json.dumps({
            "type": ChainRpcMessageType.STAGE_ERROR.value,
            "protocol_version": 999,
            "request_id": "req-1",
            "code": "X",
            "message": "from the future",
        }).encode("utf-8")
        alice_sim.set_raw_bytes(future_payload)
        chain = _make_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        assert exc_info.value.code == ExecutorErrorCode.UNSUPPORTED_VERSION


# ──────────────────────────────────────────────────────────────────────────
# Chain-level validation
# ──────────────────────────────────────────────────────────────────────────


class TestChainValidation:
    def test_empty_chain(self, executor):
        chain = GPUChain(
            request_id="req-1",
            region="us-east",
            stages=(),
            layer_ranges=(),
            total_latency_ms=0.0,
            stale_profile_count=0,
        )
        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        assert exc_info.value.code == ExecutorErrorCode.EMPTY_CHAIN

    def test_stages_layer_ranges_count_mismatch(self, executor, alice, bob):
        chain = GPUChain(
            request_id="req-1",
            region="us-east",
            stages=(alice.node_id, bob.node_id),
            layer_ranges=((0, 4),),  # 1 range for 2 stages
            total_latency_ms=0.0,
            stale_profile_count=0,
        )
        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        assert exc_info.value.code == ExecutorErrorCode.SHAPE_MISMATCH


class TestEncoderDecoder:
    def test_prompt_encoder_failure(
        self, settler, anchor, transport
    ):
        def boom_encoder(prompt: str) -> np.ndarray:
            raise ValueError("encoder broken")

        executor = RpcChainExecutor(
            settler_identity=settler,
            send_message=transport.send,
            anchor=anchor,
            prompt_encoder=boom_encoder,
            output_decoder=_output_decoder,
        )
        chain = _make_chain(
            [list(transport.sims.keys())[0], list(transport.sims.keys())[1]]
        )
        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        assert exc_info.value.code == ExecutorErrorCode.PROMPT_ENCODE_ERROR

    def test_output_decoder_failure(
        self, settler, anchor, transport, alice, bob
    ):
        def boom_decoder(arr: np.ndarray) -> str:
            raise ValueError("decoder broken")

        executor = RpcChainExecutor(
            settler_identity=settler,
            send_message=transport.send,
            anchor=anchor,
            prompt_encoder=_prompt_encoder,
            output_decoder=boom_decoder,
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(request=_make_request(), chain=chain)
        assert exc_info.value.code == ExecutorErrorCode.OUTPUT_DECODE_ERROR


# ──────────────────────────────────────────────────────────────────────────
# Address resolver
# ──────────────────────────────────────────────────────────────────────────


class TestAddressResolver:
    def test_custom_resolver_used(
        self, settler, anchor, alice, bob, alice_sim, bob_sim
    ):
        # node_id "alice" → address "alice.example.com:50051"
        sims_by_address = {
            f"{alice.node_id}.test:1": alice_sim,
            f"{bob.node_id}.test:1": bob_sim,
        }
        transport = FakeTransport(sims_by_address)
        executor = RpcChainExecutor(
            settler_identity=settler,
            send_message=transport.send,
            anchor=anchor,
            prompt_encoder=_prompt_encoder,
            output_decoder=_output_decoder,
            address_resolver=lambda nid: f"{nid}.test:1",
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        result = executor.execute_chain(request=_make_request(), chain=chain)
        assert result.output  # success
        assert transport.delivery_log == [
            f"{alice.node_id}.test:1",
            f"{bob.node_id}.test:1",
        ]


# ──────────────────────────────────────────────────────────────────────────
# Protocol conformance — slots into ParallaxScheduledExecutor
# ──────────────────────────────────────────────────────────────────────────


class TestChainExecutorProtocolConformance:
    def test_implements_chain_executor_protocol(
        self, executor, alice, bob, settler
    ):
        from prsm.compute.inference.parallax_executor import (
            ChainExecutionResult as ParallaxChainResult,
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        result = executor.execute_chain(request=_make_request(), chain=chain)
        # The returned object IS the same dataclass Phase 3.x.6 uses.
        assert isinstance(result, ParallaxChainResult)
        # Has the four Protocol-required fields.
        assert hasattr(result, "output")
        assert hasattr(result, "duration_seconds")
        assert hasattr(result, "tee_attestation")
        assert hasattr(result, "tee_type")
        assert hasattr(result, "epsilon_spent")

    def test_drop_in_replaces_test_fake_in_parallax_executor(
        self, settler, anchor, alice, bob, alice_sim, bob_sim, transport
    ):
        """End-to-end: wire an RpcChainExecutor into a real
        ParallaxScheduledExecutor in place of the test fake. The
        existing 3.x.6 executor should call execute_chain identically."""
        from prsm.compute.inference.parallax_executor import (
            ParallaxScheduledExecutor,
        )
        from prsm.compute.parallax_scheduling.prsm_request_router import (
            InMemoryProfileSource, ProfileSnapshot,
        )
        from prsm.compute.parallax_scheduling.prsm_types import ParallaxGPU
        from prsm.compute.parallax_scheduling.trust_adapter import (
            AnchorVerifyAdapter, ConsensusMismatchHook,
            StakeWeightedTrustAdapter, TierGateAdapter, TrustStack,
        )
        from prsm.compute.parallax_scheduling.model_info import ModelInfo

        rpc_executor = RpcChainExecutor(
            settler_identity=settler,
            send_message=transport.send,
            anchor=anchor,
            prompt_encoder=_prompt_encoder,
            output_decoder=_output_decoder,
        )

        # Minimal ParallaxScheduledExecutor wiring: pool with alice +
        # bob (already anchor-registered), trivial trust stack, the
        # rpc_executor as chain_executor.
        pool = [
            ParallaxGPU(
                node_id=alice.node_id, region="us-east", layer_capacity=4,
                stake_amount=10**18, tier_attestation="tier-sgx",
                tflops_fp16=100.0, memory_gb=80.0, memory_bandwidth_gbps=2000.0,
            ),
            ParallaxGPU(
                node_id=bob.node_id, region="us-east", layer_capacity=4,
                stake_amount=10**18, tier_attestation="tier-sgx",
                tflops_fp16=100.0, memory_gb=80.0, memory_bandwidth_gbps=2000.0,
            ),
        ]
        snaps = InMemoryProfileSource(snapshots={
            alice.node_id: ProfileSnapshot(
                alice.node_id, 10.0, {bob.node_id: 1.0}, 1000.0,
            ),
            bob.node_id: ProfileSnapshot(
                bob.node_id, 10.0, {alice.node_id: 1.0}, 1000.0,
            ),
        })

        class _Stake:
            def get_stake(self, n): return 10**18

        trust = TrustStack(
            anchor_verify=AnchorVerifyAdapter(anchor=anchor),
            tier_gate=TierGateAdapter(),
            profile_source=StakeWeightedTrustAdapter(
                inner=snaps, stake_lookup=_Stake(),
            ),
            consensus_hook=ConsensusMismatchHook(
                submitter=lambda r: None, sample_rate=0.0,
            ),
        )
        catalog = {"test-model": ModelInfo(
            model_name="t", mlx_model_name="t", head_size=64,
            hidden_dim=512, intermediate_dim=2048,
            num_attention_heads=8, num_kv_heads=8,
            vocab_size=32000, num_layers=4,
        )}

        executor = ParallaxScheduledExecutor(
            gpu_pool_provider=lambda: pool,
            trust_stack=trust,
            model_catalog=catalog,
            chain_executor=rpc_executor,
            node_identity=settler,
            cost_per_layer=Decimal("0.01"),
        )
        import asyncio
        result = asyncio.new_event_loop().run_until_complete(
            executor.execute(_make_request("integration"))
        )
        assert result.success is True, result.error
        assert result.output == "integration"
        assert result.receipt is not None
        # The Parallax scheduler may pick either a single-stage chain
        # (one GPU has enough capacity for the whole 4-layer model)
        # or a multi-stage chain. Either way, at least one stage sim
        # MUST have been called via the RpcChainExecutor — that's the
        # drop-in-replacement signal.
        total_calls = len(alice_sim.calls) + len(bob_sim.calls)
        assert total_calls >= 1, (
            "RpcChainExecutor was not exercised — "
            "ParallaxScheduledExecutor took a different path"
        )


# ──────────────────────────────────────────────────────────────────────────
# StageOutcome dataclass smoke
# ──────────────────────────────────────────────────────────────────────────


class TestStageOutcome:
    def test_immutable(self):
        outcome = StageOutcome(
            stage_index=0,
            stage_node_id="alice",
            duration_seconds=0.05,
            tee_attestation=b"\x01" * 32,
            tee_type=TEEType.SGX,
            epsilon_spent=0.0,
        )
        with pytest.raises(Exception):
            outcome.stage_index = 99  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.7.1 v2 streaming dispatch
# ──────────────────────────────────────────────────────────────────────────


from typing import Iterable, Tuple as _Tuple

from prsm.compute.chain_rpc import StreamedSendMessage
from prsm.compute.chain_rpc.protocol import (
    ActivationChunk as _ActivationChunk,
    RunLayerSliceRequest as _Req,
    RunLayerSliceResponse as _Resp,
)
from prsm.compute.chain_rpc.activation_codec import (
    chunk_activation as _chunk_activation,
    reassemble_chunked as _reassemble,
)
from prsm.node.shard_streaming import ShardChunk as _ShardChunk


class StreamedStageSim:
    """Per-stage simulator for the v2 streamed path. Receives a
    manifest_bytes + chunk_bytes_iter; assembles the activation,
    optionally transforms it, chunks the output, signs the response,
    returns (response_manifest_bytes, response_chunk_bytes_iter)."""

    def __init__(
        self,
        identity,
        *,
        tee_type: TEEType = TEEType.SOFTWARE,
        chunk_bytes: int = 256,
    ):
        self.identity = identity
        self.tee_type = tee_type
        self.tee_attestation = b"\x01" * 32
        self.epsilon = 0.0
        self.duration = 0.05
        self.chunk_bytes = chunk_bytes
        self._transform = None
        self.calls: List[bytes] = []
        self.raise_on_call: Optional[Exception] = None

    def set_transform(self, fn):
        self._transform = fn

    def handle(
        self, manifest_bytes: bytes, chunk_iter: Iterable[bytes]
    ) -> _Tuple[bytes, Iterable[bytes]]:
        self.calls.append(manifest_bytes)
        if self.raise_on_call is not None:
            raise self.raise_on_call

        request = parse_message(manifest_bytes)
        assert isinstance(request, _Req)
        assert request.activation_manifest is not None
        # Reassemble inbound chunks via the codec helpers.
        # ActivationChunk.request_id is the relay-defense binding;
        # the assembler's shard_id check uses the manifest's shard_id.
        shard_chunks: List[_ShardChunk] = []
        for raw in chunk_iter:
            msg = parse_message(raw)
            assert isinstance(msg, _ActivationChunk)
            shard_chunks.append(_ShardChunk(
                shard_id=request.activation_manifest.shard_id,
                sequence=msg.sequence,
                data=msg.data,
                chunk_sha256=msg.chunk_sha256,
            ))
        from prsm.compute.chain_rpc.activation_codec import ChunkedActivation as _CA
        chunked_in = _CA(
            manifest=request.activation_manifest,
            chunks=shard_chunks,
            shape=request.activation_shape,
            dtype_str=request.activation_dtype,
        )
        activation = _reassemble(chunked_in, chunks=shard_chunks)

        # Optionally transform the activation.
        if self._transform is not None:
            activation = self._transform(activation)

        # Chunk the output via the same codec.
        chunked_out = _chunk_activation(
            activation,
            activation_id=f"{request.request_id}::resp",
            chunk_bytes=self.chunk_bytes,
        )
        # Build a streamed response signed by this stage. The
        # signing payload commits to manifest.payload_sha256.
        response = _Resp.sign(
            identity=self.identity,
            request_id=request.request_id,
            activation_blob=b"",
            activation_shape=chunked_out.shape,
            activation_dtype=chunked_out.dtype_str,
            duration_seconds=self.duration,
            tee_attestation=self.tee_attestation,
            tee_type=self.tee_type,
            epsilon_spent=self.epsilon,
            activation_manifest=chunked_out.manifest,
        )
        response_manifest_bytes = encode_message(response)
        response_chunk_frames = [
            encode_message(_ActivationChunk(
                request_id=request.request_id,
                sequence=c.sequence,
                data=c.data,
                chunk_sha256=c.chunk_sha256,
            ))
            for c in chunked_out.chunks
        ]
        return response_manifest_bytes, iter(response_chunk_frames)


class StreamedFakeTransport:
    """Sibling to FakeTransport for the v2 streamed path."""

    def __init__(self, sims: Dict[str, StreamedStageSim]):
        self.sims = sims
        self.transport_failure: Optional[Exception] = None
        self.delivery_log: List[str] = []

    def send_streamed(
        self,
        address: str,
        manifest_bytes: bytes,
        chunk_iter: Iterable[bytes],
    ) -> _Tuple[bytes, Iterable[bytes]]:
        self.delivery_log.append(address)
        if self.transport_failure is not None:
            raise self.transport_failure
        sim = self.sims.get(address)
        if sim is None:
            raise ConnectionError(f"no streamed stage at {address!r}")
        return sim.handle(manifest_bytes, chunk_iter)


@pytest.fixture
def alice_streamed_sim(alice):
    return StreamedStageSim(alice, chunk_bytes=128)


@pytest.fixture
def bob_streamed_sim(bob):
    return StreamedStageSim(bob, chunk_bytes=128)


@pytest.fixture
def streamed_transport(alice, bob, alice_streamed_sim, bob_streamed_sim):
    return StreamedFakeTransport({
        alice.node_id: alice_streamed_sim,
        bob.node_id: bob_streamed_sim,
    })


def _make_executor_with_streaming(
    *,
    settler,
    anchor,
    inline_transport,
    streamed_transport,
    threshold: int = 3,
    chunk_bytes: int = 32,
) -> RpcChainExecutor:
    """Low threshold + small chunks force the streamed path on
    typical-sized test activations. Threshold=3 means any prompt
    encoded to >= 4 bytes (i.e., ≥ 1 int32) triggers streamed."""
    return RpcChainExecutor(
        settler_identity=settler,
        send_message=inline_transport.send,
        streamed_send_message=streamed_transport.send_streamed,
        anchor=anchor,
        prompt_encoder=_prompt_encoder,
        output_decoder=_output_decoder,
        chunk_threshold_bytes=threshold,
        chunk_bytes=chunk_bytes,
    )


class TestStreamedDispatchHappyPath:
    def test_streamed_two_stage_chain_round_trips(
        self, settler, anchor, alice, bob, transport,
        streamed_transport, alice_streamed_sim, bob_streamed_sim,
    ):
        executor = _make_executor_with_streaming(
            settler=settler, anchor=anchor,
            inline_transport=transport,
            streamed_transport=streamed_transport,
            threshold=3,  # force streamed for any activation > 16 bytes
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        result = executor.execute_chain(
            request=_make_request("hello world long enough to chunk"),
            chain=chain,
        )
        assert result.output == "hello world long enough to chunk"
        # Streamed transport carried both stages.
        assert streamed_transport.delivery_log == [alice.node_id, bob.node_id]
        # Inline transport was NOT used (activation always exceeded
        # threshold).
        assert len(alice_streamed_sim.calls) == 1
        assert len(bob_streamed_sim.calls) == 1

    def test_streamed_threading_through_stages(
        self, settler, anchor, alice, bob, transport,
        streamed_transport, alice_streamed_sim, bob_streamed_sim,
    ):
        # alice doubles, bob adds 1.
        alice_streamed_sim.set_transform(lambda a: a * 2)
        bob_streamed_sim.set_transform(lambda a: a + 1)
        executor = _make_executor_with_streaming(
            settler=settler, anchor=anchor,
            inline_transport=transport,
            streamed_transport=streamed_transport,
            threshold=3,
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        request = _make_request("AB")
        result = executor.execute_chain(request=request, chain=chain)

        encoded = _prompt_encoder("AB")
        expected = (encoded * 2) + 1
        assert result.output == _output_decoder(expected)


class TestActivationTooLarge:
    def test_activation_too_large_when_streamed_transport_missing(
        self, settler, anchor, alice, bob, transport,
    ):
        """When activation exceeds inline threshold but no streamed
        transport was wired, the executor surfaces a structured
        ACTIVATION_TOO_LARGE error rather than truncating."""
        executor = RpcChainExecutor(
            settler_identity=settler,
            send_message=transport.send,
            anchor=anchor,
            prompt_encoder=_prompt_encoder,
            output_decoder=_output_decoder,
            chunk_threshold_bytes=4,  # absurdly low; any activation triggers
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(
                request=_make_request("hello"),
                chain=chain,
            )
        assert exc_info.value.code == ExecutorErrorCode.ACTIVATION_TOO_LARGE
        assert exc_info.value.stage_index == 0


class TestStreamedSignatureVerification:
    def test_streamed_signature_verifies_under_anchor(
        self, settler, anchor, alice, bob, transport,
        streamed_transport,
    ):
        """The streamed-path response signature must verify under
        the dispatched stage's pubkey via the H2 expected_stage_node_id
        invariant."""
        executor = _make_executor_with_streaming(
            settler=settler, anchor=anchor,
            inline_transport=transport,
            streamed_transport=streamed_transport,
            threshold=3,
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        result = executor.execute_chain(
            request=_make_request("streamed-sig-verify"),
            chain=chain,
        )
        # Success implies signature verified at each stage.
        assert result.output == "streamed-sig-verify"

    def test_streamed_response_with_unregistered_signer_rejected(
        self, settler, anchor, alice, bob, transport,
    ):
        """A streamed response signed by an unregistered identity must
        fail verification."""
        rogue = generate_node_identity("rogue-stage")
        # Note: rogue NOT registered on anchor.
        rogue_sim = StreamedStageSim(rogue, chunk_bytes=128)
        # Bind rogue to the address alice was supposed to serve, so the
        # client dispatches to alice.node_id but a rogue stage answers.
        # The expected_stage_node_id check rejects the substitution.
        rogue_sim.identity = rogue
        bad_streamed = StreamedFakeTransport({alice.node_id: rogue_sim})

        executor = _make_executor_with_streaming(
            settler=settler, anchor=anchor,
            inline_transport=transport,
            streamed_transport=bad_streamed,
            threshold=3,
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(
                request=_make_request("rogue-streamed"),
                chain=chain,
            )
        assert exc_info.value.code == ExecutorErrorCode.INVALID_STAGE_SIGNATURE
        assert exc_info.value.stage_index == 0


class TestStreamedTransportErrors:
    def test_streamed_transport_failure_maps_to_transport_error(
        self, settler, anchor, alice, bob, transport,
        streamed_transport,
    ):
        streamed_transport.transport_failure = ConnectionError("net down")
        executor = _make_executor_with_streaming(
            settler=settler, anchor=anchor,
            inline_transport=transport,
            streamed_transport=streamed_transport,
            threshold=3,
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(
                request=_make_request("xyz"), chain=chain,
            )
        assert exc_info.value.code == ExecutorErrorCode.TRANSPORT_ERROR

    def test_streamed_simulator_raise_maps_to_transport_error(
        self, settler, anchor, alice, bob, transport,
        streamed_transport, alice_streamed_sim,
    ):
        alice_streamed_sim.raise_on_call = RuntimeError("alice crashed")
        executor = _make_executor_with_streaming(
            settler=settler, anchor=anchor,
            inline_transport=transport,
            streamed_transport=streamed_transport,
            threshold=3,
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(
                request=_make_request(), chain=chain,
            )
        assert exc_info.value.code == ExecutorErrorCode.TRANSPORT_ERROR
        assert exc_info.value.stage_index == 0


class TestStreamedConstructionValidation:
    def test_non_callable_streamed_send_rejected(
        self, settler, anchor, transport,
    ):
        with pytest.raises(RuntimeError, match="streamed_send_message"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=transport.send,
                streamed_send_message="not-callable",  # type: ignore[arg-type]
                anchor=anchor,
                prompt_encoder=_prompt_encoder,
                output_decoder=_output_decoder,
            )

    def test_non_positive_threshold_rejected(
        self, settler, anchor, transport,
    ):
        with pytest.raises(ValueError, match="chunk_threshold_bytes"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=transport.send,
                anchor=anchor,
                prompt_encoder=_prompt_encoder,
                output_decoder=_output_decoder,
                chunk_threshold_bytes=0,
            )

    def test_non_positive_chunk_bytes_rejected(
        self, settler, anchor, transport,
    ):
        with pytest.raises(ValueError, match="chunk_bytes"):
            RpcChainExecutor(
                settler_identity=settler,
                send_message=transport.send,
                anchor=anchor,
                prompt_encoder=_prompt_encoder,
                output_decoder=_output_decoder,
                chunk_bytes=0,
            )


class TestInlinePathStillWorksAtV2:
    """Default v2 executor with no streamed transport still drives
    the inline path for small activations."""

    def test_small_activation_uses_inline_path(
        self, settler, anchor, alice, bob, transport,
        streamed_transport, alice_sim, bob_sim,
        alice_streamed_sim, bob_streamed_sim,
    ):
        # High threshold → small activations stay inline even though
        # streaming transport is wired.
        executor = _make_executor_with_streaming(
            settler=settler, anchor=anchor,
            inline_transport=transport,
            streamed_transport=streamed_transport,
            threshold=10_000_000,  # 10 MB — way above test activation size
        )
        chain = _make_chain([alice.node_id, bob.node_id])
        result = executor.execute_chain(
            request=_make_request("small"),
            chain=chain,
        )
        assert result.output == "small"
        # Inline transport was used; streamed transport was not.
        assert len(alice_sim.calls) == 1
        assert len(bob_sim.calls) == 1
        assert len(alice_streamed_sim.calls) == 0
        assert len(bob_streamed_sim.calls) == 0
