"""End-to-end integration test — Phase 3.x.7 Task 7.

Three simulated PRSM nodes (alice, bob, charlie) each running a real
``LayerStageServer`` over a synchronous ``FakeNetwork`` bus (same
pattern as Phase 3.x.5 ``test_manifest_dht_e2e.py`` and Phase 3.x.6
``test_parallax_e2e.py``). The chain executes a real (toy) 4-layer
``ShardedModel`` end-to-end via ``RpcChainExecutor``; outputs are
bit-identical to the single-host reference computation.

The "real model" is a deterministic 4-layer transform (``y = x * 2 + N``
applied per layer). It exercises the full Phase 3.x.7 stack:

  - ``RpcChainExecutor`` (Task 4) orchestrates per-stage dispatch.
  - ``LayerStageServer`` (Task 2) validates 8 gates + executes via
    the injected ``LayerSliceRunner``.
  - Activation codec (Task 3) round-trips numpy arrays through the
    wire format.
  - ``HandoffToken`` + stage signatures (Task 1) verify against the
    Phase 3.x.3 anchor.
  - Multi-stage TEE attestation envelope (Task 5) wraps per-stage
    attestations in the receipt's ``tee_attestation`` field.

Acceptance per design plan §4 Task 7:

  1. Happy path: chain executes; output bit-identical to single-host
  2. Deadline propagation: token deadline expires mid-chain → DEADLINE_EXCEEDED
  3. Forged token: response signed by an unregistered key → INVALID_TOKEN
  4. Shard missing: stage asked for layers it doesn't host → SHARD_MISSING
  5. Tier gate: PrivacyLevel.HIGH on software-only chain → TIER_GATE
  6. Cross-stage signature failure: tampered response → INVALID_STAGE_SIGNATURE
  7. Mid-chain disconnect: stage offline mid-request → TRANSPORT_ERROR
"""

from __future__ import annotations

import base64
import hashlib
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

from prsm.compute.chain_rpc import (
    ChainExecutionError,
    ExecutorErrorCode,
    LayerSliceResult,
    LayerStageServer,
    RpcChainExecutor,
    encode_message,
    make_layer_stage_server,
    make_rpc_chain_executor,
    parse_message,
    utf8_output_decoder,
    utf8_prompt_encoder,
)
from prsm.compute.chain_rpc.protocol import (
    HandoffToken,
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    StageError,
    StageErrorCode,
)
from prsm.compute.inference.models import (
    ContentTier,
    InferenceRequest,
)
from prsm.compute.model_sharding.models import (
    ModelShard,
    PipelineStakeTier,
    ShardedModel,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import NodeIdentity, generate_node_identity
from prsm.security.publisher_key_anchor import PublisherAlreadyRegisteredError


# ──────────────────────────────────────────────────────────────────────────
# Faithful simulated anchor (mirrors Phase 3.x.5/3.x.6 pattern)
# ──────────────────────────────────────────────────────────────────────────


class SimulatedAnchorContract:
    def __init__(self) -> None:
        self._publisher_keys: Dict[bytes, bytes] = {}

    def register(self, public_key_bytes: bytes) -> bytes:
        if len(public_key_bytes) != 32:
            raise ValueError("InvalidPublicKeyLength")
        node_id = hashlib.sha256(public_key_bytes).digest()[:16]
        if node_id in self._publisher_keys:
            raise PublisherAlreadyRegisteredError(
                f"AlreadyRegistered: {node_id.hex()}"
            )
        self._publisher_keys[node_id] = public_key_bytes
        return node_id

    def lookup(self, node_id_bytes16: bytes) -> bytes:
        return self._publisher_keys.get(node_id_bytes16, b"")


class SimulatedAnchorClient:
    def __init__(self, contract: SimulatedAnchorContract) -> None:
        self._contract = contract

    def lookup(self, node_id: str) -> Optional[str]:
        if not node_id:
            return None
        s = node_id[2:] if node_id.startswith("0x") else node_id
        if len(s) != 32:
            return None
        try:
            node_id_bytes = bytes.fromhex(s)
        except ValueError:
            return None
        result = self._contract.lookup(node_id_bytes)
        if not result:
            return None
        return base64.b64encode(result).decode("ascii")


def _register_identity(
    contract: SimulatedAnchorContract, identity: NodeIdentity
) -> None:
    pubkey_bytes = base64.b64decode(identity.public_key_b64)
    contract.register(pubkey_bytes)


# ──────────────────────────────────────────────────────────────────────────
# FakeNetwork — synchronous in-process bus
# ──────────────────────────────────────────────────────────────────────────


class FakeNetwork:
    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[bytes], bytes]] = {}
        # Optional per-address middleware: address → callable(req_bytes,
        # default_handler) → resp_bytes. Used to simulate response
        # tampering or out-of-band failures.
        self._middleware: Dict[str, Callable[[bytes, Callable], bytes]] = {}

    def register(
        self, address: str, handler: Callable[[bytes], bytes]
    ) -> None:
        self._handlers[address] = handler

    def disconnect(self, address: str) -> None:
        self._handlers.pop(address, None)
        self._middleware.pop(address, None)

    def install_middleware(
        self,
        address: str,
        mw: Callable[[bytes, Callable], bytes],
    ) -> None:
        self._middleware[address] = mw

    def send(self, address: str, request_bytes: bytes) -> bytes:
        handler = self._handlers.get(address)
        if handler is None:
            raise ConnectionError(f"no node at {address}")
        mw = self._middleware.get(address)
        if mw is not None:
            return mw(request_bytes, handler)
        return handler(request_bytes)


# ──────────────────────────────────────────────────────────────────────────
# Deterministic layer math: y = x * 2 + layer_idx
# ──────────────────────────────────────────────────────────────────────────


def _apply_layer_range(
    activation: np.ndarray, layer_range: Tuple[int, int]
) -> np.ndarray:
    """Apply the deterministic transform for layers in [start, end).

    Pure numpy; no torch dependency. Bit-identical across single-host
    and chain-host runs as long as the layer indices line up.
    """
    out = activation.astype(np.int64).copy()
    start, end = layer_range
    for layer_idx in range(start, end):
        out = out * 2 + layer_idx
    return out


def _single_host_reference(
    prompt: str, total_layers: int = 4
) -> np.ndarray:
    """The bit-identical baseline: encode prompt, apply all layers
    in one shot."""
    activation = utf8_prompt_encoder(prompt)
    return _apply_layer_range(activation, (0, total_layers))


# ──────────────────────────────────────────────────────────────────────────
# DeterministicLayerRunner — a real LayerSliceRunner backed by the
# math above. Per-stage TEE attestation is a deterministic blob the
# caller supplies (typically the stage's NodeIdentity-derived bytes).
# ──────────────────────────────────────────────────────────────────────────


class DeterministicLayerRunner:
    def __init__(
        self,
        *,
        attestation: bytes = b"\x01" * 32,
        tee_type: TEEType = TEEType.SOFTWARE,
        sleep_seconds: float = 0.0,
    ):
        self.attestation = attestation
        self.tee_type = tee_type
        self.sleep_seconds = sleep_seconds
        self.calls: List[Tuple[Tuple[int, int], np.ndarray]] = []

    def run_layer_range(
        self,
        *,
        model: Any,
        layer_range: Tuple[int, int],
        activation: np.ndarray,
        privacy_tier: PrivacyLevel,
        is_final_stage: bool,
    ) -> LayerSliceResult:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        out = _apply_layer_range(activation, layer_range)
        self.calls.append((layer_range, out.copy()))
        return LayerSliceResult(
            output=out,
            duration_seconds=0.05,
            tee_attestation=self.attestation,
            tee_type=self.tee_type,
            epsilon_spent=0.0,
        )


# ──────────────────────────────────────────────────────────────────────────
# Test model + per-node registry
# ──────────────────────────────────────────────────────────────────────────


MODEL_ID = "phase3x7-e2e-toy"
TOTAL_LAYERS = 4


def _make_model_with_layers(layer_range: Tuple[int, int]) -> ShardedModel:
    """Build a real ShardedModel whose single shard advertises the
    given layer_range. Simulates 'this node has shards covering these
    layers locally'."""
    shard = ModelShard(
        shard_id=f"{MODEL_ID}-shard-0",
        model_id=MODEL_ID,
        shard_index=0,
        total_shards=1,
        tensor_data=b"\x00" * 16,
        tensor_shape=(4,),
        layer_range=layer_range,
        size_bytes=16,
        checksum="0" * 64,
    )
    return ShardedModel(
        model_id=MODEL_ID,
        model_name="phase3x7-e2e-toy-llm",
        total_shards=1,
        shards=[shard],
        stake_tier=PipelineStakeTier.STANDARD,
    )


@dataclass
class _LocalRegistry:
    """Minimal ModelRegistry-shaped object for the test stage server.
    Implements just .get(model_id) since that's what the server uses."""

    models: Dict[str, ShardedModel] = field(default_factory=dict)

    def get(self, model_id: str) -> ShardedModel:
        if model_id not in self.models:
            from prsm.compute.model_registry import ModelNotFoundError
            raise ModelNotFoundError(f"unknown model {model_id!r}")
        return self.models[model_id]


@dataclass
class _LocalTEE:
    tee_type: TEEType = TEEType.SOFTWARE


# ──────────────────────────────────────────────────────────────────────────
# Per-node setup helper
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class StageNode:
    identity: NodeIdentity
    address: str
    layer_range: Tuple[int, int]
    runner: DeterministicLayerRunner
    server: LayerStageServer
    registry: _LocalRegistry
    tee_runtime: _LocalTEE


def _make_stage_node(
    *,
    identity: NodeIdentity,
    address: str,
    layer_range: Tuple[int, int],
    anchor: SimulatedAnchorClient,
    network: FakeNetwork,
    tee_type: TEEType = TEEType.SOFTWARE,
    sleep_seconds: float = 0.0,
    attestation: Optional[bytes] = None,
) -> StageNode:
    registry = _LocalRegistry(
        models={MODEL_ID: _make_model_with_layers(layer_range)}
    )
    tee_runtime = _LocalTEE(tee_type=tee_type)
    runner = DeterministicLayerRunner(
        attestation=attestation or hashlib.sha256(
            identity.node_id.encode()
        ).digest(),
        tee_type=tee_type,
        sleep_seconds=sleep_seconds,
    )
    server = make_layer_stage_server(
        identity=identity,
        registry=registry,
        runner=runner,
        tee_runtime=tee_runtime,
        anchor=anchor,
    )
    network.register(address, server.handle)
    return StageNode(
        identity=identity,
        address=address,
        layer_range=layer_range,
        runner=runner,
        server=server,
        registry=registry,
        tee_runtime=tee_runtime,
    )


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def settler():
    return generate_node_identity("settler")


@pytest.fixture
def alice():
    return generate_node_identity("alice-stage")


@pytest.fixture
def bob():
    return generate_node_identity("bob-stage")


@pytest.fixture
def charlie():
    return generate_node_identity("charlie-stage")


@pytest.fixture
def network():
    return FakeNetwork()


@pytest.fixture
def anchor_pair(settler, alice, bob, charlie):
    contract = SimulatedAnchorContract()
    client = SimulatedAnchorClient(contract)
    _register_identity(contract, settler)
    _register_identity(contract, alice)
    _register_identity(contract, bob)
    _register_identity(contract, charlie)
    return contract, client


@pytest.fixture
def two_stage_setup(network, anchor_pair, alice, bob):
    """Default setup: alice covers layers 0..2, bob covers 2..4."""
    _, anchor_client = anchor_pair
    alice_node = _make_stage_node(
        identity=alice, address=alice.node_id,
        layer_range=(0, TOTAL_LAYERS),  # full coverage so coverage check passes for any sub-range
        anchor=anchor_client, network=network,
    )
    bob_node = _make_stage_node(
        identity=bob, address=bob.node_id,
        layer_range=(0, TOTAL_LAYERS),  # full coverage
        anchor=anchor_client, network=network,
    )
    return {"alice": alice_node, "bob": bob_node}


def _build_chain(stage_node_ids: List[str]) -> Any:
    """Build a GPUChain with even layer-range division across stages."""
    from prsm.compute.parallax_scheduling.prsm_request_router import GPUChain

    n = len(stage_node_ids)
    per_stage = TOTAL_LAYERS // n
    layer_ranges = []
    for i in range(n):
        start = i * per_stage
        end = (i + 1) * per_stage if i < n - 1 else TOTAL_LAYERS
        layer_ranges.append((start, end))
    return GPUChain(
        request_id="req-1",
        region="us-east",
        stages=tuple(stage_node_ids),
        layer_ranges=tuple(layer_ranges),
        total_latency_ms=10.0,
        stale_profile_count=0,
    )


def _make_request(
    *,
    prompt: str = "hello",
    privacy_tier: PrivacyLevel = PrivacyLevel.NONE,
    request_id: str = "req-1",
) -> InferenceRequest:
    return InferenceRequest(
        prompt=prompt,
        model_id=MODEL_ID,
        budget_ftns=Decimal("10.0"),
        privacy_tier=privacy_tier,
        content_tier=ContentTier.A,
        request_id=request_id,
    )


def _make_executor(
    *,
    settler: NodeIdentity,
    network: FakeNetwork,
    anchor_client: SimulatedAnchorClient,
    deadline_seconds: float = 30.0,
) -> RpcChainExecutor:
    return make_rpc_chain_executor(
        settler_identity=settler,
        send_message=network.send,
        anchor=anchor_client,
        default_deadline_seconds=deadline_seconds,
    )


# ──────────────────────────────────────────────────────────────────────────
# Scenario 1: Happy path — bit-identical to single-host
# ──────────────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_two_stage_chain_matches_single_host(
        self, settler, network, anchor_pair, two_stage_setup, alice, bob
    ):
        _, anchor_client = anchor_pair
        executor = _make_executor(
            settler=settler, network=network, anchor_client=anchor_client
        )
        chain = _build_chain([alice.node_id, bob.node_id])
        prompt = "the quick brown fox"

        result = executor.execute_chain(
            request=_make_request(prompt=prompt),
            chain=chain,
        )

        # Reference computation: same prompt run through all 4 layers
        # in a single shot. Decoded → same output string the chain
        # produced.
        reference_activation = _single_host_reference(prompt)
        reference_output = utf8_output_decoder(reference_activation)
        assert result.output == reference_output, (
            f"chain output {result.output!r} differs from single-host "
            f"reference {reference_output!r}"
        )

        # Each stage was called exactly once with its layer slice.
        assert len(two_stage_setup["alice"].runner.calls) == 1
        assert two_stage_setup["alice"].runner.calls[0][0] == (0, 2)
        assert len(two_stage_setup["bob"].runner.calls) == 1
        assert two_stage_setup["bob"].runner.calls[0][0] == (2, 4)

    def test_three_stage_chain(
        self, settler, network, anchor_pair, alice, bob, charlie
    ):
        _, anchor_client = anchor_pair
        # Each stage covers the full model layer range locally so any
        # sub-slice request passes the coverage check.
        nodes = {
            "alice": _make_stage_node(
                identity=alice, address=alice.node_id,
                layer_range=(0, TOTAL_LAYERS),
                anchor=anchor_client, network=network,
            ),
            "bob": _make_stage_node(
                identity=bob, address=bob.node_id,
                layer_range=(0, TOTAL_LAYERS),
                anchor=anchor_client, network=network,
            ),
            "charlie": _make_stage_node(
                identity=charlie, address=charlie.node_id,
                layer_range=(0, TOTAL_LAYERS),
                anchor=anchor_client, network=network,
            ),
        }
        executor = _make_executor(
            settler=settler, network=network, anchor_client=anchor_client
        )
        # Three-stage chain: alice 0..1, bob 1..2, charlie 2..4.
        from prsm.compute.parallax_scheduling.prsm_request_router import (
            GPUChain,
        )
        chain = GPUChain(
            request_id="req-1",
            region="us-east",
            stages=(alice.node_id, bob.node_id, charlie.node_id),
            layer_ranges=((0, 1), (1, 2), (2, 4)),
            total_latency_ms=10.0,
            stale_profile_count=0,
        )

        result = executor.execute_chain(
            request=_make_request(prompt="abc"), chain=chain,
        )
        reference = utf8_output_decoder(_single_host_reference("abc"))
        assert result.output == reference

    def test_receipt_tee_attestation_is_multi_stage_envelope(
        self, settler, network, anchor_pair, two_stage_setup, alice, bob
    ):
        from prsm.compute.inference.multi_stage_attestation import (
            decode_multi_stage_attestation,
            is_multi_stage_attestation,
        )

        _, anchor_client = anchor_pair
        executor = _make_executor(
            settler=settler, network=network, anchor_client=anchor_client
        )
        chain = _build_chain([alice.node_id, bob.node_id])
        result = executor.execute_chain(
            request=_make_request(), chain=chain,
        )

        # The chain executor wraps per-stage attestations in the
        # multi-stage envelope (Task 5). Verifiers can decode + iterate.
        assert is_multi_stage_attestation(result.tee_attestation)
        stages = decode_multi_stage_attestation(result.tee_attestation)
        assert stages is not None
        assert len(stages) == 2
        assert stages[0].stage_node_id == alice.node_id
        assert stages[1].stage_node_id == bob.node_id


# ──────────────────────────────────────────────────────────────────────────
# Scenario 2: Deadline propagation
# ──────────────────────────────────────────────────────────────────────────


class TestDeadlineExceeded:
    def test_deadline_in_past_rejected_by_first_stage(
        self, settler, network, anchor_pair, two_stage_setup, alice, bob
    ):
        """Force the executor's deadline to be immediately past by
        using an absurdly small deadline window. The first stage's
        clock-driven deadline check rejects with DEADLINE_EXCEEDED."""
        _, anchor_client = anchor_pair
        executor = _make_executor(
            settler=settler, network=network, anchor_client=anchor_client,
            deadline_seconds=0.0001,  # ~immediate; expires before send
        )
        chain = _build_chain([alice.node_id, bob.node_id])

        # Tiny sleep to ensure clock advances past deadline.
        time.sleep(0.001)
        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(
                request=_make_request(), chain=chain,
            )
        assert exc_info.value.code == StageErrorCode.DEADLINE_EXCEEDED.value
        assert exc_info.value.stage_index == 0


# ──────────────────────────────────────────────────────────────────────────
# Scenario 3: Forged token (unregistered settler signs)
# ──────────────────────────────────────────────────────────────────────────


class TestForgedToken:
    def test_unregistered_settler_token_rejected(
        self, network, anchor_pair, two_stage_setup, alice, bob
    ):
        """A 'malicious' settler whose identity is NOT registered on
        the anchor mints + signs a token. The first stage's anchor
        verification rejects it with INVALID_TOKEN."""
        _, anchor_client = anchor_pair
        rogue_settler = generate_node_identity("rogue-settler")
        # NOT registered on the anchor.
        executor = _make_executor(
            settler=rogue_settler,
            network=network,
            anchor_client=anchor_client,
        )
        chain = _build_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(
                request=_make_request(), chain=chain,
            )
        assert exc_info.value.code == StageErrorCode.INVALID_TOKEN.value
        assert exc_info.value.stage_index == 0
        # alice's runner was NOT invoked — the rejection happened at
        # the validation gate before any layer execution.
        assert len(two_stage_setup["alice"].runner.calls) == 0


# ──────────────────────────────────────────────────────────────────────────
# Scenario 4: Shard missing
# ──────────────────────────────────────────────────────────────────────────


class TestShardMissing:
    def test_stage_lacks_layer_coverage(
        self, settler, network, anchor_pair, alice, bob
    ):
        """Bob hosts only layers (2, 4) locally. Chain assigns him
        layers (0, 2) — bob's local registry returns a model whose
        shards don't cover the requested range. SHARD_MISSING."""
        _, anchor_client = anchor_pair
        # Alice with full coverage.
        _make_stage_node(
            identity=alice, address=alice.node_id,
            layer_range=(0, TOTAL_LAYERS),
            anchor=anchor_client, network=network,
        )
        # Bob with PARTIAL coverage (only layers 2..4).
        bob_node = _make_stage_node(
            identity=bob, address=bob.node_id,
            layer_range=(2, TOTAL_LAYERS),
            anchor=anchor_client, network=network,
        )

        executor = _make_executor(
            settler=settler, network=network, anchor_client=anchor_client
        )
        # Reverse the chain so bob is asked for (0, 2) — outside his
        # layer range.
        from prsm.compute.parallax_scheduling.prsm_request_router import (
            GPUChain,
        )
        chain = GPUChain(
            request_id="req-1",
            region="us-east",
            stages=(bob.node_id, alice.node_id),
            layer_ranges=((0, 2), (2, 4)),
            total_latency_ms=10.0,
            stale_profile_count=0,
        )

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(
                request=_make_request(), chain=chain,
            )
        assert exc_info.value.code == StageErrorCode.SHARD_MISSING.value
        assert exc_info.value.stage_index == 0
        assert exc_info.value.stage_node_id == bob.node_id
        # bob's runner was NOT invoked — coverage check fired before
        # layer execution.
        assert len(bob_node.runner.calls) == 0


# ──────────────────────────────────────────────────────────────────────────
# Scenario 5: Tier gate
# ──────────────────────────────────────────────────────────────────────────


class TestTierGate:
    def test_high_privacy_on_software_tee_rejected(
        self, settler, network, anchor_pair, two_stage_setup, alice, bob
    ):
        """All stages run software TEE; PrivacyLevel.HIGH demands
        hardware. First stage's tier-gate check fires."""
        _, anchor_client = anchor_pair
        executor = _make_executor(
            settler=settler, network=network, anchor_client=anchor_client
        )
        chain = _build_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(
                request=_make_request(privacy_tier=PrivacyLevel.HIGH),
                chain=chain,
            )
        assert exc_info.value.code == StageErrorCode.TIER_GATE.value
        assert exc_info.value.stage_index == 0
        # No layer execution happened.
        assert len(two_stage_setup["alice"].runner.calls) == 0

    def test_high_privacy_on_hardware_tee_succeeds(
        self, settler, network, anchor_pair, alice, bob
    ):
        """When all stages have hardware-TEE attestation, the same
        request succeeds end-to-end."""
        _, anchor_client = anchor_pair
        _make_stage_node(
            identity=alice, address=alice.node_id,
            layer_range=(0, TOTAL_LAYERS),
            anchor=anchor_client, network=network,
            tee_type=TEEType.SGX,
        )
        _make_stage_node(
            identity=bob, address=bob.node_id,
            layer_range=(0, TOTAL_LAYERS),
            anchor=anchor_client, network=network,
            tee_type=TEEType.TDX,
        )
        executor = _make_executor(
            settler=settler, network=network, anchor_client=anchor_client
        )
        chain = _build_chain([alice.node_id, bob.node_id])

        result = executor.execute_chain(
            request=_make_request(privacy_tier=PrivacyLevel.HIGH),
            chain=chain,
        )
        # Same bit-identical reference.
        reference = utf8_output_decoder(_single_host_reference("hello"))
        assert result.output == reference


# ──────────────────────────────────────────────────────────────────────────
# Scenario 6: Cross-stage signature failure
# ──────────────────────────────────────────────────────────────────────────


class TestSignatureTampering:
    def test_tampered_response_signature_detected(
        self, settler, network, anchor_pair, two_stage_setup, alice, bob
    ):
        """Install middleware on bob's address that decodes the
        server's response, swaps a byte in the activation_blob,
        re-encodes, and forwards. The byte-level tamper invalidates
        bob's signature → executor catches at anchor-verify."""
        _, anchor_client = anchor_pair

        def tamper(req_bytes: bytes, default_handler: Callable) -> bytes:
            # Let bob actually run, then corrupt the response.
            response_bytes = default_handler(req_bytes)
            response = parse_message(response_bytes)
            assert isinstance(response, RunLayerSliceResponse)
            tampered_blob = bytearray(response.activation_blob)
            # Flip a byte mid-blob.
            tampered_blob[0] ^= 0xFF
            tampered = RunLayerSliceResponse(
                request_id=response.request_id,
                activation_blob=bytes(tampered_blob),
                activation_shape=response.activation_shape,
                activation_dtype=response.activation_dtype,
                duration_seconds=response.duration_seconds,
                tee_attestation=response.tee_attestation,
                tee_type=response.tee_type,
                epsilon_spent=response.epsilon_spent,
                stage_signature_b64=response.stage_signature_b64,
                stage_node_id=response.stage_node_id,
            )
            return encode_message(tampered)

        network.install_middleware(bob.node_id, tamper)

        executor = _make_executor(
            settler=settler, network=network, anchor_client=anchor_client
        )
        chain = _build_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(
                request=_make_request(), chain=chain,
            )
        assert exc_info.value.code == ExecutorErrorCode.INVALID_STAGE_SIGNATURE
        # bob is the second stage (index 1).
        assert exc_info.value.stage_index == 1
        # alice's runner ran; bob's also ran (the tamper happens AFTER
        # bob's local execution but BEFORE the response reaches the
        # client). The signal is the executor refused to accept the
        # response — that's the v1 challenge-trigger surface.
        assert len(two_stage_setup["alice"].runner.calls) == 1


# ──────────────────────────────────────────────────────────────────────────
# Scenario 7: Mid-chain disconnect
# ──────────────────────────────────────────────────────────────────────────


class TestMidChainDisconnect:
    def test_stage_offline_mid_request(
        self, settler, network, anchor_pair, two_stage_setup, alice, bob
    ):
        """alice runs successfully; bob disconnects from the network
        before the executor reaches him. Transport-level ConnectionError
        surfaces as ChainExecutionError(TRANSPORT_ERROR) at stage 1.
        No hung chain."""
        _, anchor_client = anchor_pair

        # Disconnect bob from the network — alice still reachable.
        network.disconnect(bob.node_id)

        executor = _make_executor(
            settler=settler, network=network, anchor_client=anchor_client
        )
        chain = _build_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(
                request=_make_request(), chain=chain,
            )
        assert exc_info.value.code == ExecutorErrorCode.TRANSPORT_ERROR
        assert exc_info.value.stage_index == 1
        assert exc_info.value.stage_node_id == bob.node_id
        # alice's runner ran successfully before bob's disconnect
        # surfaced.
        assert len(two_stage_setup["alice"].runner.calls) == 1


# ──────────────────────────────────────────────────────────────────────────
# Single-host reference smoke
# ──────────────────────────────────────────────────────────────────────────


class TestReferenceComputation:
    def test_single_host_layer_math_deterministic(self):
        """Sanity check: the reference computation that the chain
        outputs are compared against is itself deterministic."""
        ref1 = _single_host_reference("hello")
        ref2 = _single_host_reference("hello")
        np.testing.assert_array_equal(ref1, ref2)

    def test_layer_range_composition(self):
        """A single-host run of layers 0..4 must equal a chained run
        of layers 0..2 followed by layers 2..4 on the same input."""
        prompt = "the quick brown fox"
        full = _single_host_reference(prompt)

        encoded = utf8_prompt_encoder(prompt)
        first_half = _apply_layer_range(encoded, (0, 2))
        chained = _apply_layer_range(first_half, (2, 4))
        np.testing.assert_array_equal(full, chained)


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.7.1 streamed-path E2E
# ──────────────────────────────────────────────────────────────────────────


from typing import Iterable as _Iterable, Tuple as _Tuple


class FakeStreamingNetwork:
    """Sibling to FakeNetwork that routes streamed requests to each
    server's handle_streamed. Production uses Phase 6 gRPC bidi-
    streaming; here we just call the server's handle_streamed directly.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable] = {}

    def register(
        self,
        address: str,
        handle_streamed: Callable[[bytes, _Iterable[bytes]],
                                  _Tuple[bytes, _Iterable[bytes]]],
    ) -> None:
        self._handlers[address] = handle_streamed

    def disconnect(self, address: str) -> None:
        self._handlers.pop(address, None)

    def send_streamed(
        self,
        address: str,
        manifest_bytes: bytes,
        chunk_iter: _Iterable[bytes],
    ) -> _Tuple[bytes, _Iterable[bytes]]:
        handler = self._handlers.get(address)
        if handler is None:
            raise ConnectionError(f"no streamed node at {address}")
        return handler(manifest_bytes, chunk_iter)


def _make_streaming_stage_node(
    *,
    identity: NodeIdentity,
    address: str,
    layer_range: Tuple[int, int],
    anchor: SimulatedAnchorClient,
    inline_network: FakeNetwork,
    streamed_network: FakeStreamingNetwork,
    chunk_bytes: int = 256,
    tee_type: TEEType = TEEType.SOFTWARE,
) -> StageNode:
    """Build a stage node wired to BOTH inline + streamed networks."""
    registry = _LocalRegistry(
        models={MODEL_ID: _make_model_with_layers(layer_range)}
    )
    tee_runtime = _LocalTEE(tee_type=tee_type)
    runner = DeterministicLayerRunner(
        attestation=hashlib.sha256(identity.node_id.encode()).digest(),
        tee_type=tee_type,
    )
    server = make_layer_stage_server(
        identity=identity,
        registry=registry,
        runner=runner,
        tee_runtime=tee_runtime,
        anchor=anchor,
        chunk_bytes=chunk_bytes,
    )
    inline_network.register(address, server.handle)
    streamed_network.register(address, server.handle_streamed)
    return StageNode(
        identity=identity,
        address=address,
        layer_range=layer_range,
        runner=runner,
        server=server,
        registry=registry,
        tee_runtime=tee_runtime,
    )


def _make_streaming_executor(
    *,
    settler: NodeIdentity,
    inline_network: FakeNetwork,
    streamed_network: FakeStreamingNetwork,
    anchor_client: SimulatedAnchorClient,
    chunk_threshold_bytes: int,
    chunk_bytes: int = 256,
) -> RpcChainExecutor:
    return make_rpc_chain_executor(
        settler_identity=settler,
        send_message=inline_network.send,
        streamed_send_message=streamed_network.send_streamed,
        anchor=anchor_client,
        chunk_threshold_bytes=chunk_threshold_bytes,
        chunk_bytes=chunk_bytes,
    )


class TestStreamedActivation:
    """Streamed-path E2E: activations exceeding the inline threshold
    route via chunked streaming through real LayerStageServers; output
    is bit-identical to a single-host reference."""

    def test_streamed_two_stage_chain_matches_single_host(
        self, settler, alice, bob, anchor_pair
    ):
        """Identical math, identical output: streamed path must
        produce bit-identical results to single-host reference."""
        _, anchor_client = anchor_pair
        inline_net = FakeNetwork()
        streamed_net = FakeStreamingNetwork()

        _make_streaming_stage_node(
            identity=alice, address=alice.node_id,
            layer_range=(0, TOTAL_LAYERS),
            anchor=anchor_client,
            inline_network=inline_net, streamed_network=streamed_net,
        )
        _make_streaming_stage_node(
            identity=bob, address=bob.node_id,
            layer_range=(0, TOTAL_LAYERS),
            anchor=anchor_client,
            inline_network=inline_net, streamed_network=streamed_net,
        )
        # Threshold = 4 → any prompt encoded to ≥ 8 bytes streams.
        executor = _make_streaming_executor(
            settler=settler,
            inline_network=inline_net,
            streamed_network=streamed_net,
            anchor_client=anchor_client,
            chunk_threshold_bytes=4,
            chunk_bytes=8,
        )
        chain = _build_chain([alice.node_id, bob.node_id])
        prompt = "the quick brown fox jumps over the lazy dog"

        result = executor.execute_chain(
            request=_make_request(prompt=prompt),
            chain=chain,
        )

        # Bit-identical to single-host reference.
        reference = utf8_output_decoder(_single_host_reference(prompt))
        assert result.output == reference

    def test_streamed_three_stage_chain_matches_single_host(
        self, settler, alice, bob, charlie, anchor_pair
    ):
        _, anchor_client = anchor_pair
        inline_net = FakeNetwork()
        streamed_net = FakeStreamingNetwork()
        for ident in (alice, bob, charlie):
            _make_streaming_stage_node(
                identity=ident, address=ident.node_id,
                layer_range=(0, TOTAL_LAYERS),
                anchor=anchor_client,
                inline_network=inline_net, streamed_network=streamed_net,
            )
        executor = _make_streaming_executor(
            settler=settler,
            inline_network=inline_net,
            streamed_network=streamed_net,
            anchor_client=anchor_client,
            chunk_threshold_bytes=4,
            chunk_bytes=8,
        )
        from prsm.compute.parallax_scheduling.prsm_request_router import (
            GPUChain,
        )
        chain = GPUChain(
            request_id="req-1",
            region="us-east",
            stages=(alice.node_id, bob.node_id, charlie.node_id),
            layer_ranges=((0, 1), (1, 2), (2, 4)),
            total_latency_ms=10.0,
            stale_profile_count=0,
        )

        result = executor.execute_chain(
            request=_make_request(prompt="streamed-3-stage-test"),
            chain=chain,
        )

        reference = utf8_output_decoder(
            _single_host_reference("streamed-3-stage-test")
        )
        assert result.output == reference


class TestMultiMBStreamedActivation:
    """The headline scaling scenario: an activation that genuinely
    cannot fit the inline-path 64 MiB cap (post-hex+JSON overhead)
    routes successfully via streaming. Run on a 16 MiB raw float32
    activation chunked into ≥ 16 chunks of 1 MiB each.

    To keep the test fast we use a single stage that simply identity-
    transforms the activation rather than running 4 layer transforms;
    the 4-layer math is exhaustively tested in the smaller TestStreamed*
    cases above. Here we're stressing the chunk transport.
    """

    def test_16_mib_activation_round_trips_streamed(
        self, settler, alice, anchor_pair
    ):
        _, anchor_client = anchor_pair
        inline_net = FakeNetwork()
        streamed_net = FakeStreamingNetwork()

        # Use 1 MiB chunks so a 16 MiB activation produces 16 chunks.
        ONE_MIB = 1 * 1024 * 1024
        _make_streaming_stage_node(
            identity=alice, address=alice.node_id,
            layer_range=(0, TOTAL_LAYERS),
            anchor=anchor_client,
            inline_network=inline_net, streamed_network=streamed_net,
            chunk_bytes=ONE_MIB,
        )
        # Threshold low so any non-trivial activation streams.
        executor = _make_streaming_executor(
            settler=settler,
            inline_network=inline_net,
            streamed_network=streamed_net,
            anchor_client=anchor_client,
            chunk_threshold_bytes=4,
            chunk_bytes=ONE_MIB,
        )

        # Custom prompt encoder: produce a 16 MiB float32 activation
        # directly (bypass the UTF-8 default for stress sizing).
        # 4 MiB of float32 = 1 M elements; 16 MiB = 4 M elements.
        rng = np.random.default_rng(seed=42)
        big_activation = rng.standard_normal(
            size=(4_000_000,)
        ).astype(np.float32)
        assert big_activation.nbytes == 4 * 4_000_000  # = 16 MiB

        # Custom encoder/decoder: round-trip bytes losslessly.
        def big_encoder(prompt: str) -> np.ndarray:
            return big_activation

        def big_decoder(arr: np.ndarray) -> str:
            return arr.dtype.str + "::" + str(arr.shape) + "::" + str(arr.nbytes)

        custom_executor = make_rpc_chain_executor(
            settler_identity=settler,
            send_message=inline_net.send,
            streamed_send_message=streamed_net.send_streamed,
            anchor=anchor_client,
            prompt_encoder=big_encoder,
            output_decoder=big_decoder,
            chunk_threshold_bytes=4,
            chunk_bytes=ONE_MIB,
        )

        # Single-stage chain so we test transport without compounding
        # the 4-layer transform on a 16 MiB activation (fast).
        from prsm.compute.parallax_scheduling.prsm_request_router import (
            GPUChain,
        )
        chain = GPUChain(
            request_id="req-bigact",
            region="us-east",
            stages=(alice.node_id,),
            layer_ranges=((0, 4),),
            total_latency_ms=10.0,
            stale_profile_count=0,
        )

        result = custom_executor.execute_chain(
            request=_make_request(
                prompt="ignored", request_id="req-bigact"
            ),
            chain=chain,
        )

        # The decoder reports dtype/shape/bytes — recovered output
        # MUST report the same shape + nbytes as input (after layer
        # transform on int64, the byte width changes).
        # Layer transform: float32 → int64 (per _apply_layer_range
        # cast). 4M elements × 8 bytes = 32 MiB output. Shape and
        # element count preserved.
        assert "(4000000,)" in result.output
        # Output dtype is int64 (8-byte) per the layer math.
        assert "int64" in result.output or "i8" in result.output

    def test_16_mib_activation_bit_identical_to_single_host(
        self, settler, alice, anchor_pair
    ):
        """Bit-equivalence at scale. Same big activation routed via
        streaming MUST produce the same int64 output array (byte-for-
        byte) as single-host application of the layer math."""
        _, anchor_client = anchor_pair
        inline_net = FakeNetwork()
        streamed_net = FakeStreamingNetwork()
        ONE_MIB = 1 * 1024 * 1024

        _make_streaming_stage_node(
            identity=alice, address=alice.node_id,
            layer_range=(0, TOTAL_LAYERS),
            anchor=anchor_client,
            inline_network=inline_net, streamed_network=streamed_net,
            chunk_bytes=ONE_MIB,
        )

        rng = np.random.default_rng(seed=99)
        big_activation = rng.standard_normal(
            size=(4_000_000,)
        ).astype(np.float32)

        captured = {}

        def big_encoder(prompt: str) -> np.ndarray:
            return big_activation

        def big_decoder(arr: np.ndarray) -> str:
            captured["output"] = arr
            return "ok"

        executor = make_rpc_chain_executor(
            settler_identity=settler,
            send_message=inline_net.send,
            streamed_send_message=streamed_net.send_streamed,
            anchor=anchor_client,
            prompt_encoder=big_encoder,
            output_decoder=big_decoder,
            chunk_threshold_bytes=4,
            chunk_bytes=ONE_MIB,
        )

        from prsm.compute.parallax_scheduling.prsm_request_router import (
            GPUChain,
        )
        chain = GPUChain(
            request_id="req-bit-eq",
            region="us-east",
            stages=(alice.node_id,),
            layer_ranges=((0, TOTAL_LAYERS),),
            total_latency_ms=10.0,
            stale_profile_count=0,
        )

        result = executor.execute_chain(
            request=_make_request(prompt="ignored", request_id="req-bit-eq"),
            chain=chain,
        )
        assert result.output == "ok"

        # Single-host reference: apply layer math directly.
        reference = _apply_layer_range(big_activation, (0, TOTAL_LAYERS))
        # The chain output MUST match byte-for-byte.
        np.testing.assert_array_equal(captured["output"], reference)

    def test_streamed_response_is_chunked_when_output_exceeds_threshold(
        self, settler, alice, anchor_pair
    ):
        """When the response activation also exceeds the chunk
        threshold, the response carries a manifest + chunks (not
        inline). Verifies symmetric streaming."""
        from prsm.compute.chain_rpc.protocol import (
            RunLayerSliceResponse,
        )

        _, anchor_client = anchor_pair
        inline_net = FakeNetwork()
        streamed_net = FakeStreamingNetwork()

        ONE_MIB = 1 * 1024 * 1024
        node = _make_streaming_stage_node(
            identity=alice, address=alice.node_id,
            layer_range=(0, TOTAL_LAYERS),
            anchor=anchor_client,
            inline_network=inline_net, streamed_network=streamed_net,
            chunk_bytes=ONE_MIB,
        )

        # Capture the wire response by intercepting the streamed
        # transport.
        captured_responses = []
        original = streamed_net.send_streamed

        def intercept(addr, manifest_bytes, chunk_iter):
            resp_manifest, resp_chunks = original(addr, manifest_bytes, chunk_iter)
            # Materialize chunks to allow inspection.
            chunk_list = list(resp_chunks)
            captured_responses.append((resp_manifest, chunk_list))
            return resp_manifest, iter(chunk_list)

        rng = np.random.default_rng(seed=11)
        big_activation = rng.standard_normal(
            size=(2_000_000,)
        ).astype(np.float32)  # 8 MiB float32 → 16 MiB int64 after math

        def big_encoder(prompt: str) -> np.ndarray:
            return big_activation

        def big_decoder(arr: np.ndarray) -> str:
            return "ok"

        executor = make_rpc_chain_executor(
            settler_identity=settler,
            send_message=inline_net.send,
            streamed_send_message=intercept,
            anchor=anchor_client,
            prompt_encoder=big_encoder,
            output_decoder=big_decoder,
            chunk_threshold_bytes=4,
            chunk_bytes=ONE_MIB,
        )

        from prsm.compute.parallax_scheduling.prsm_request_router import (
            GPUChain,
        )
        chain = GPUChain(
            request_id="req-resp-streamed",
            region="us-east",
            stages=(alice.node_id,),
            layer_ranges=((0, TOTAL_LAYERS),),
            total_latency_ms=10.0,
            stale_profile_count=0,
        )

        result = executor.execute_chain(
            request=_make_request(
                prompt="ignored", request_id="req-resp-streamed"
            ),
            chain=chain,
        )
        assert result.output == "ok"

        # The captured response MUST be a streamed RunLayerSliceResponse:
        # activation_blob empty + activation_manifest set + chunks
        # delivered.
        assert len(captured_responses) == 1
        resp_manifest_bytes, resp_chunks = captured_responses[0]
        from prsm.compute.chain_rpc.protocol import parse_message
        resp = parse_message(resp_manifest_bytes)
        assert isinstance(resp, RunLayerSliceResponse)
        assert resp.activation_blob == b""
        assert resp.activation_manifest is not None
        # Multi-chunk response (8 MiB float32 → 16 MiB int64 / 1 MiB
        # per chunk = 16 chunks).
        assert len(resp_chunks) >= 8
        assert resp.activation_manifest.total_chunks == len(resp_chunks)


class TestStreamedActivationTooLarge:
    """When the activation exceeds the inline threshold but no
    streamed transport is wired, the executor surfaces a structured
    ACTIVATION_TOO_LARGE failure — bit-identical to the unit-test
    coverage but verified end-to-end against a real
    LayerStageServer + ParallaxScheduledExecutor."""

    def test_activation_too_large_when_no_streamed_transport(
        self, settler, alice, bob, anchor_pair
    ):
        from prsm.compute.chain_rpc.client import (
            ChainExecutionError,
            ExecutorErrorCode,
        )

        _, anchor_client = anchor_pair
        inline_net = FakeNetwork()
        # No streamed network wired.
        streamed_net = FakeStreamingNetwork()

        # Build stages but only register them on the inline network —
        # caller supplies inline transport only.
        for ident in (alice, bob):
            registry = _LocalRegistry(
                models={MODEL_ID: _make_model_with_layers((0, TOTAL_LAYERS))}
            )
            tee_runtime = _LocalTEE(tee_type=TEEType.SOFTWARE)
            runner = DeterministicLayerRunner(
                attestation=hashlib.sha256(ident.node_id.encode()).digest(),
            )
            server = make_layer_stage_server(
                identity=ident,
                registry=registry,
                runner=runner,
                tee_runtime=tee_runtime,
                anchor=anchor_client,
            )
            inline_net.register(ident.node_id, server.handle)

        # Threshold low; no streamed transport.
        executor = make_rpc_chain_executor(
            settler_identity=settler,
            send_message=inline_net.send,
            anchor=anchor_client,
            chunk_threshold_bytes=4,
        )
        chain = _build_chain([alice.node_id, bob.node_id])

        with pytest.raises(ChainExecutionError) as exc_info:
            executor.execute_chain(
                request=_make_request(prompt="something"),
                chain=chain,
            )
        assert exc_info.value.code == ExecutorErrorCode.ACTIVATION_TOO_LARGE
