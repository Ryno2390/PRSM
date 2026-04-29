"""Phase 3.x.11.x Task 3 — server-side chunked + sharded PREFILL
composition tests.

Covers ``LayerStageServer._dispatch_streamed_sharded`` (the new
method that lifts the Phase 3.x.11 Task 9 M1 guard for PREFILL
when ``sharded_runner`` is wired). INCREMENTAL streamed requests
stay rejected (single-position activations don't benefit from
chunking). Tier C structurally denied (mirrors
``_dispatch_sharded`` policy).

Test coverage:
  - Happy-path PREFILL with chunked activation roundtrips through
    the sharded runner; response carries next_token_id +
    is_terminal AND is itself chunked back
  - INCREMENTAL streamed → MALFORMED_REQUEST with message
    pointing at the unary-only design
  - Tier C streamed → TIER_GATE (mirrors _dispatch_sharded)
  - Envelope validation: inflated payload_bytes → ACTIVATION_INVALID
    (reuses Phase 3.x.7.1 H1 + M1 defence)
  - Chunk reassembly error (corrupt chunk) → ACTIVATION_INVALID
  - Sharded runner exception → INTERNAL_ERROR (caught + mapped;
    server's "never raises" invariant preserved)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pytest

from prsm.compute.chain_rpc.activation_codec import (
    ChunkedActivation,
    chunk_activation,
    reassemble_chunked,
)
from prsm.compute.chain_rpc.kv_cache import KVCacheManager
from prsm.compute.chain_rpc.protocol import (
    ActivationChunk,
    DecodeMode,
    HandoffToken,
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    StageError,
    StageErrorCode,
    encode_message,
    parse_message,
)
from prsm.compute.chain_rpc.server import (
    LayerSliceResult,
    LayerSliceRunner,
    LayerStageServer,
)
from prsm.compute.inference.autoregressive_runner import SamplingDefaults
from prsm.compute.inference.models import ContentTier
from prsm.compute.inference.sharded_runner import (
    ShardedAutoregressiveRunner,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity
from prsm.node.shard_streaming import ShardChunk


# ──────────────────────────────────────────────────────────────────────────
# Test fakes
# ──────────────────────────────────────────────────────────────────────────


class _FakeAnchor:
    def __init__(self) -> None:
        self.registered: Dict[str, str] = {}

    def lookup(self, node_id: str) -> Optional[str]:
        return self.registered.get(node_id)


class _Shard:
    def __init__(self, layer_range: Tuple[int, int]) -> None:
        self.layer_range = layer_range


class _Model:
    def __init__(self) -> None:
        self.shards = [_Shard((0, 3))]


class _Registry:
    def __init__(self) -> None:
        self._model = _Model()

    def get(self, model_id: str) -> _Model:
        if model_id != "test-model":
            raise _ModelNotFoundError(model_id)
        return self._model


class _ModelNotFoundError(Exception):
    pass


class _TEERuntime:
    tee_type = TEEType.SOFTWARE

    def get_attestation_bytes(self) -> bytes:
        return b"\x07" * 32


class _PassthroughRunner(LayerSliceRunner):
    """Required by the LayerStageServer constructor — sharded
    decode never reaches this runner."""

    def run_layer_range(
        self, *, model, layer_range, activation, privacy_tier,
        is_final_stage,
    ) -> LayerSliceResult:
        return LayerSliceResult(
            output=activation.copy(),
            duration_seconds=0.001,
            tee_attestation=b"\x07" * 32,
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )


class _FakeShardedModel:
    """ShardedLayerForward Protocol impl. PREFILL returns the
    input as-is (cast to float32 + 3-D shape); cache is a marker
    list. Tail-capable when configured."""

    def __init__(self, *, sample_token: int = 42) -> None:
        self._sample_token = sample_token
        self.prefill_calls: List[Tuple[Any, Tuple[int, int]]] = []

    def forward_prefill(
        self, *, input_or_hidden: Any,
        layer_range: Tuple[int, int],
    ) -> Tuple[np.ndarray, List[str]]:
        self.prefill_calls.append((input_or_hidden, layer_range))
        if isinstance(input_or_hidden, list):
            arr = np.array(input_or_hidden, dtype=np.float32)
        else:
            arr = np.asarray(input_or_hidden, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :, np.newaxis]
        elif arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        # Return a non-trivially-sized hidden state so the response
        # also exercises chunked output. Replicate the input
        # ~10x to push past chunk_bytes=128.
        out = np.tile(arr, (1, 10, 1)).astype(np.float32)
        cache = ["cache-marker-0", "cache-marker-1", "cache-marker-2"]
        return out, cache

    def forward_incremental(
        self, *, input_or_hidden, layer_range, kv_cache_payload,
    ):
        # Phase 3.x.11.x Task 3 only exercises PREFILL; INCREMENTAL
        # streamed is rejected upstream.
        raise AssertionError(
            "_FakeShardedModel.forward_incremental should not be "
            "called by the chunked-PREFILL path"
        )

    def apply_lm_head_and_sample(
        self, *, hidden_state, temperature, top_k, top_p,
    ) -> int:
        return self._sample_token


# ──────────────────────────────────────────────────────────────────────────
# Server build
# ──────────────────────────────────────────────────────────────────────────


def _build_server(
    *,
    is_tail: bool = False,
    sample_token: int = 42,
    chunk_bytes_override: int = 128,
):
    identity = generate_node_identity("stage")
    settler = generate_node_identity("settler")
    anchor = _FakeAnchor()
    anchor.registered[identity.node_id] = identity.public_key_b64
    anchor.registered[settler.node_id] = settler.public_key_b64

    kv_cache = KVCacheManager()
    fake_model = _FakeShardedModel(sample_token=sample_token)

    sharded_kwargs = {
        "model": fake_model,
        "layer_range": (0, 3),
        "kv_cache_manager": kv_cache,
        "tee_attestation": b"\x07" * 32,
        "tee_type": TEEType.SOFTWARE,
    }
    if is_tail:
        sharded_kwargs["sampling_defaults"] = SamplingDefaults(
            max_tokens=4, temperature=0.0, top_k=50, top_p=0.95,
        )
        sharded_kwargs["eos_token_id"] = 999
    sharded_runner = ShardedAutoregressiveRunner(**sharded_kwargs)

    server = LayerStageServer(
        identity=identity,
        registry=_Registry(),
        runner=_PassthroughRunner(),
        tee_runtime=_TEERuntime(),
        anchor=anchor,
        clock=lambda: 1000.0,
        chunk_bytes=chunk_bytes_override,
        kv_cache_manager=kv_cache,
        sharded_runner=sharded_runner,
    )
    return server, settler, identity, fake_model, kv_cache


def _build_streamed_request(
    *,
    settler,
    decode_mode: DecodeMode = DecodeMode.PREFILL,
    content_tier: ContentTier = ContentTier.A,
    chain_stage_index: int = 0,
    chain_total_stages: int = 2,
    activation: np.ndarray = None,
    payload_bytes_override: Optional[int] = None,
    chunk_bytes: int = 128,
) -> Tuple[bytes, List[bytes], np.ndarray]:
    """Build a streamed RunLayerSliceRequest manifest + chunk frames
    suitable for handle_streamed. Returns
    (manifest_bytes, chunk_frames, activation_input)."""
    if activation is None:
        # Large-ish activation that forces chunking at chunk_bytes=128.
        activation = np.arange(64, dtype=np.float32).reshape(2, 32)
    chunked = chunk_activation(
        activation,
        activation_id=f"req-streamed::stage-0",
        chunk_bytes=chunk_bytes,
    )
    manifest = chunked.manifest
    # M1 envelope-tamper: optionally inflate payload_bytes.
    if payload_bytes_override is not None:
        from prsm.node.shard_streaming import ShardManifest
        manifest = ShardManifest(
            shard_id=manifest.shard_id,
            payload_sha256=manifest.payload_sha256,
            payload_bytes=payload_bytes_override,
            total_chunks=manifest.total_chunks,
            chunk_bytes=manifest.chunk_bytes,
        )
    deadline = 2000.0
    token = HandoffToken.sign(
        identity=settler,
        request_id="req-streamed",
        chain_stage_index=chain_stage_index,
        chain_total_stages=chain_total_stages,
        deadline_unix=deadline,
    )
    request = RunLayerSliceRequest(
        request_id="req-streamed",
        model_id="test-model",
        layer_range=(0, 3),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=content_tier,
        activation_blob=b"",
        activation_shape=chunked.shape,
        activation_dtype=chunked.dtype_str,
        upstream_token=token,
        deadline_unix=deadline,
        activation_manifest=manifest,
        decode_mode=decode_mode,
    )
    chunk_frames = [
        encode_message(ActivationChunk(
            request_id="req-streamed",
            sequence=c.sequence,
            data=c.data,
            chunk_sha256=c.chunk_sha256,
        ))
        for c in chunked.chunks
    ]
    return encode_message(request), chunk_frames, activation


def _decode_streamed_response(
    response_manifest_bytes: bytes,
    response_chunk_iter: Iterable[bytes],
):
    """Parse manifest + chunks, reassemble. Returns
    (recovered_array, response_or_error_dataclass)."""
    response = parse_message(response_manifest_bytes)
    if isinstance(response, StageError):
        return None, response
    assert isinstance(response, RunLayerSliceResponse)
    if response.activation_manifest is None:
        return None, response
    shard_chunks = []
    for raw in response_chunk_iter:
        msg = parse_message(raw)
        assert isinstance(msg, ActivationChunk)
        shard_chunks.append(ShardChunk(
            shard_id=response.activation_manifest.shard_id,
            sequence=msg.sequence,
            data=msg.data,
            chunk_sha256=msg.chunk_sha256,
        ))
    chunked = ChunkedActivation(
        manifest=response.activation_manifest,
        chunks=shard_chunks,
        shape=response.activation_shape,
        dtype_str=response.activation_dtype,
    )
    arr = reassemble_chunked(chunked, chunks=shard_chunks)
    return arr, response


# ──────────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────────


class TestStreamedShardedHappyPath:
    def test_prefill_chunked_input_roundtrips_through_runner(self):
        # Non-tail: server response carries next_token_id=None +
        # chunked output activation.
        server, settler, _, fake_model, _ = _build_server()
        manifest_bytes, chunks, _ = _build_streamed_request(
            settler=settler, decode_mode=DecodeMode.PREFILL,
        )
        resp_manifest, resp_chunks = server.handle_streamed(
            manifest_bytes, iter(chunks),
        )
        arr, response = _decode_streamed_response(
            resp_manifest, resp_chunks,
        )
        assert isinstance(response, RunLayerSliceResponse)
        assert arr is not None
        assert arr.dtype == np.float32
        # The fake model returned 10x-tiled activation.
        assert arr.size > 0
        # Non-tail: next_token_id is None.
        assert response.next_token_id is None
        assert response.is_terminal is False
        # Sharded runner WAS called (PREFILL).
        assert len(fake_model.prefill_calls) == 1
        layer_range_called = fake_model.prefill_calls[0][1]
        assert layer_range_called == (0, 3)

    def test_prefill_tail_carries_next_token_id_and_is_terminal(self):
        # Tail-configured server samples + populates next_token_id.
        server, settler, _, _, _ = _build_server(
            is_tail=True, sample_token=77,
        )
        # Use chain_stage_index = total - 1 so server detects tail.
        manifest_bytes, chunks, _ = _build_streamed_request(
            settler=settler, decode_mode=DecodeMode.PREFILL,
            chain_stage_index=1, chain_total_stages=2,
        )
        resp_manifest, resp_chunks = server.handle_streamed(
            manifest_bytes, iter(chunks),
        )
        _, response = _decode_streamed_response(
            resp_manifest, resp_chunks,
        )
        assert isinstance(response, RunLayerSliceResponse)
        assert response.next_token_id == 77
        # max_tokens=4 default in the tail's sampling_defaults; one
        # PREFILL token sampled → tokens_generated=1 < 4 → not
        # terminal yet.
        assert response.is_terminal is False


# ──────────────────────────────────────────────────────────────────────────
# Decode-mode + tier rejections
# ──────────────────────────────────────────────────────────────────────────


class TestStreamedShardedRejections:
    def test_incremental_streamed_rejected(self):
        # Sharded INCREMENTAL is unary-only (single-position
        # activations don't benefit from chunking).
        server, settler, _, _, _ = _build_server()
        manifest_bytes, chunks, _ = _build_streamed_request(
            settler=settler, decode_mode=DecodeMode.INCREMENTAL,
        )
        resp_manifest, _ = server.handle_streamed(
            manifest_bytes, iter(chunks),
        )
        msg = parse_message(resp_manifest)
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.MALFORMED_REQUEST.value
        assert "unary-only" in msg.message

    def test_tier_c_streamed_rejected(self):
        # Tier C structural deny mirrors _dispatch_sharded.
        server, settler, _, _, _ = _build_server()
        manifest_bytes, chunks, _ = _build_streamed_request(
            settler=settler, content_tier=ContentTier.C,
        )
        resp_manifest, _ = server.handle_streamed(
            manifest_bytes, iter(chunks),
        )
        msg = parse_message(resp_manifest)
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.TIER_GATE.value
        assert "Tier C" in msg.message


# ──────────────────────────────────────────────────────────────────────────
# Envelope validation
# ──────────────────────────────────────────────────────────────────────────


class TestStreamedShardedEnvelopeValidation:
    def test_inflated_payload_bytes_rejected_before_chunks_consumed(self):
        # Phase 3.x.7.1 H1+M1 round-1 defence carries forward to
        # the sharded streamed path. A hostile peer that ships an
        # inflated payload_bytes is rejected BEFORE the
        # ShardAssembler buffers anything.
        server, settler, _, _, _ = _build_server()
        manifest_bytes, chunks, _ = _build_streamed_request(
            settler=settler,
            payload_bytes_override=10 * 1024 * 1024 * 1024,  # 10 GiB
        )
        resp_manifest, _ = server.handle_streamed(
            manifest_bytes, iter(chunks),
        )
        msg = parse_message(resp_manifest)
        assert isinstance(msg, StageError)
        # Envelope-validation rejection — could be
        # ACTIVATION_INVALID or MALFORMED_REQUEST depending on
        # the specific assertion (Phase 3.x.7.1 envelope-mismatch
        # path → ACTIVATION_INVALID; payload_bytes-cap path →
        # ACTIVATION_INVALID).
        assert msg.code in (
            StageErrorCode.ACTIVATION_INVALID.value,
            StageErrorCode.MALFORMED_REQUEST.value,
        )


# ──────────────────────────────────────────────────────────────────────────
# Reassembly error mapping
# ──────────────────────────────────────────────────────────────────────────


class TestStreamedShardedReassemblyErrors:
    def test_corrupt_chunk_data_rejected_with_activation_invalid(self):
        # Tamper with chunk data — the chunk's claimed sha256 no
        # longer matches the actual bytes, so reassembly fails.
        server, settler, _, _, _ = _build_server()
        manifest_bytes, chunks, _ = _build_streamed_request(
            settler=settler,
        )
        # Corrupt the FIRST chunk's data field while keeping the
        # claimed chunk_sha256.
        first = parse_message(chunks[0])
        assert isinstance(first, ActivationChunk)
        bad_first = ActivationChunk(
            request_id=first.request_id,
            sequence=first.sequence,
            data=b"\xff" * len(first.data),  # tampered
            chunk_sha256=first.chunk_sha256,  # claim unchanged
        )
        chunks[0] = encode_message(bad_first)
        resp_manifest, _ = server.handle_streamed(
            manifest_bytes, iter(chunks),
        )
        msg = parse_message(resp_manifest)
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.ACTIVATION_INVALID.value


# ──────────────────────────────────────────────────────────────────────────
# Runner exception mapping
# ──────────────────────────────────────────────────────────────────────────


class TestStreamedShardedRunnerExceptions:
    def test_runner_raise_maps_to_internal_error(self):
        # Construct a server whose sharded_runner raises on every
        # forward.
        from prsm.compute.inference.sharded_runner import (
            ShardedAutoregressiveRunner,
        )

        class _RaisingModel:
            def forward_prefill(self, **kwargs):
                raise RuntimeError("synthetic forward failure")

            def forward_incremental(self, **kwargs):
                raise NotImplementedError

        identity = generate_node_identity("stage")
        settler = generate_node_identity("settler")
        anchor = _FakeAnchor()
        anchor.registered[identity.node_id] = identity.public_key_b64
        anchor.registered[settler.node_id] = settler.public_key_b64
        kv_cache = KVCacheManager()
        runner = ShardedAutoregressiveRunner(
            model=_RaisingModel(),
            layer_range=(0, 3),
            kv_cache_manager=kv_cache,
            tee_attestation=b"\x07" * 32,
            tee_type=TEEType.SOFTWARE,
        )
        server = LayerStageServer(
            identity=identity,
            registry=_Registry(),
            runner=_PassthroughRunner(),
            tee_runtime=_TEERuntime(),
            anchor=anchor,
            clock=lambda: 1000.0,
            chunk_bytes=128,
            kv_cache_manager=kv_cache,
            sharded_runner=runner,
        )
        manifest_bytes, chunks, _ = _build_streamed_request(
            settler=settler,
        )
        resp_manifest, _ = server.handle_streamed(
            manifest_bytes, iter(chunks),
        )
        msg = parse_message(resp_manifest)
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.INTERNAL_ERROR.value
        assert "synthetic forward failure" in msg.message


# ──────────────────────────────────────────────────────────────────────────
# Back-compat: non-sharded streamed unchanged
# ──────────────────────────────────────────────────────────────────────────


class TestNonShardedStreamedUnchanged:
    def test_non_sharded_streamed_still_uses_regular_runner(self):
        # Server WITHOUT sharded_runner wired — streamed PREFILL
        # routes to the original _dispatch_streamed (regular
        # runner). Pre-3.x.11.x behavior preserved.
        identity = generate_node_identity("stage")
        settler = generate_node_identity("settler")
        anchor = _FakeAnchor()
        anchor.registered[identity.node_id] = identity.public_key_b64
        anchor.registered[settler.node_id] = settler.public_key_b64
        server = LayerStageServer(
            identity=identity,
            registry=_Registry(),
            runner=_PassthroughRunner(),
            tee_runtime=_TEERuntime(),
            anchor=anchor,
            clock=lambda: 1000.0,
            chunk_bytes=128,
        )
        manifest_bytes, chunks, original = _build_streamed_request(
            settler=settler,
        )
        resp_manifest, resp_chunks = server.handle_streamed(
            manifest_bytes, iter(chunks),
        )
        arr, response = _decode_streamed_response(
            resp_manifest, resp_chunks,
        )
        assert isinstance(response, RunLayerSliceResponse)
        # Regular runner is the passthrough; output equals input.
        np.testing.assert_array_equal(arr, original)
        # No next_token_id (regular runner doesn't sample).
        assert response.next_token_id is None
