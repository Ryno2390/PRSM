"""Phase 3.x.11.y.x Task 6 — LayerStageServer VERIFY routing v1↔v2
backwards-compat tests.

The server's _dispatch_sharded must:
  - Forward proposed_token_probs to the runner when wire field is
    set (v2 stochastic dispatch).
  - OMIT the proposed_token_probs kwarg from the runner call when
    the wire field is None (v1 traffic stays bit-identical against
    pre-3.x.11.y.x runners that don't accept the kwarg).
  - Catch TypeError from a stale runner that rejects
    proposed_token_probs= and surface MALFORMED_REQUEST so the
    executor can fall back or fail loud.

The fakes here intentionally bypass the full ShardedAutoregressiveRunner
and capture the kwargs handed to run_layer_slice_unary directly —
that's the contract surface Task 6 changed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prsm.compute.chain_rpc.activation_codec import encode_activation
from prsm.compute.chain_rpc.kv_cache import KVCacheManager
from prsm.compute.chain_rpc.protocol import (
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
from prsm.compute.inference.models import ContentTier
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Fakes
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
        self.shards = [_Shard((0, 4))]


class _Registry:
    def __init__(self) -> None:
        self._model = _Model()

    def get(self, model_id: str) -> Any:
        return self._model


class _PassthroughRunner(LayerSliceRunner):
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


class _TEERuntime:
    tee_type = TEEType.SOFTWARE

    def get_attestation_bytes(self) -> bytes:
        return b"\x07" * 32


class _RecordingShardedRunner:
    """Captures kwargs handed to run_layer_slice_unary. Returns a
    canned LayerSliceResult so the server completes the dispatch
    end-to-end."""

    def __init__(self, *, accept_probs_kwarg: bool = True) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._accept_probs_kwarg = accept_probs_kwarg

    def run_layer_slice_unary(self, **kwargs):
        if (
            not self._accept_probs_kwarg
            and "proposed_token_probs" in kwargs
        ):
            # Mimic a stale runner without the kwarg in its
            # signature — TypeError mirrors what Python raises on
            # unexpected keyword argument.
            raise TypeError(
                "run_layer_slice_unary() got an unexpected "
                "keyword argument 'proposed_token_probs'"
            )
        self.calls.append(dict(kwargs))
        # Construct a minimal valid LayerSliceResult-like object.
        # The server needs hidden_state + duration + attestation
        # (we route via decode_mode=VERIFY tail so we also need
        # verified_token_ids + accepted_count).
        from prsm.compute.inference.sharded_runner import (
            LayerSliceResult as ShardedResult,
        )
        proposed = kwargs.get("proposed_token_ids") or []
        if kwargs.get("proposed_token_probs") is not None:
            verified = tuple(int(t) for t in proposed[:1]) + (9999,)
            ac = 1
        else:
            verified = tuple(int(t) for t in proposed) + (9999,)
            ac = len(proposed)
        return ShardedResult(
            hidden_state=np.zeros((1,), dtype=np.float32),
            duration_seconds=0.001,
            decode_mode=kwargs["decode_mode"],
            n_layers_run=1,
            next_token_id=int(verified[ac]),
            is_terminal=False,
            verified_token_ids=verified,
            accepted_count=ac,
        )


# ──────────────────────────────────────────────────────────────────────────
# Server build + request build
# ──────────────────────────────────────────────────────────────────────────


def _build_server(*, recording_runner: _RecordingShardedRunner):
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
        kv_cache_manager=KVCacheManager(),
        sharded_runner=recording_runner,
    )
    return server, settler, identity


def _build_verify_request(
    *,
    settler,
    proposed_ids: Tuple[int, ...],
    proposed_probs: Optional[Tuple[float, ...]] = None,
) -> bytes:
    # Tail stage_index = chain_total_stages - 1 = 0 (single-stage).
    token = HandoffToken.sign(
        identity=settler,
        request_id="req-v2",
        chain_stage_index=0,
        chain_total_stages=1,
        deadline_unix=2000.0,
    )
    activation = np.array([1, 2, 3], dtype=np.int64)
    blob, shape, dtype_str = encode_activation(activation)
    request = RunLayerSliceRequest(
        request_id="req-v2",
        model_id="m",
        layer_range=(0, 4),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        activation_blob=blob,
        activation_shape=shape,
        activation_dtype=dtype_str,
        upstream_token=token,
        deadline_unix=2000.0,
        decode_mode=DecodeMode.VERIFY,
        proposed_token_ids=proposed_ids,
        proposed_token_probs=proposed_probs,
        temperature=0.7 if proposed_probs is not None else 0.0,
    )
    return encode_message(request)


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


class TestServerVerifyRouting:
    def test_v1_dispatch_omits_probs_kwarg(self):
        # Wire: proposed_token_probs=None. Server MUST NOT pass
        # proposed_token_probs= to runner — preserves call shape
        # against pre-3.x.11.y.x runners that don't have the
        # kwarg in their signature.
        runner = _RecordingShardedRunner(accept_probs_kwarg=True)
        server, settler, _ = _build_server(recording_runner=runner)
        req_bytes = _build_verify_request(
            settler=settler, proposed_ids=(10, 20),
            proposed_probs=None,
        )
        response_bytes = server.handle(req_bytes)
        msg = parse_message(response_bytes)
        # Must be a successful response, not an error.
        assert isinstance(msg, RunLayerSliceResponse)
        # Runner saw exactly one call.
        assert len(runner.calls) == 1
        kwargs = runner.calls[0]
        # v1 path: proposed_token_ids set, proposed_token_probs
        # MUST NOT appear in kwargs (omit-when-None semantics).
        assert kwargs["proposed_token_ids"] == [10, 20]
        assert "proposed_token_probs" not in kwargs

    def test_v2_dispatch_forwards_probs_kwarg(self):
        # Wire: proposed_token_probs set. Server MUST pass it
        # through to the runner.
        runner = _RecordingShardedRunner(accept_probs_kwarg=True)
        server, settler, _ = _build_server(recording_runner=runner)
        req_bytes = _build_verify_request(
            settler=settler,
            proposed_ids=(10, 20),
            proposed_probs=(0.6, 0.4),
        )
        response_bytes = server.handle(req_bytes)
        msg = parse_message(response_bytes)
        assert isinstance(msg, RunLayerSliceResponse)
        assert len(runner.calls) == 1
        kwargs = runner.calls[0]
        assert kwargs["proposed_token_ids"] == [10, 20]
        assert kwargs["proposed_token_probs"] == [0.6, 0.4]

    def test_v2_dispatch_against_stale_runner_returns_malformed(self):
        # Stale runner doesn't accept proposed_token_probs= →
        # TypeError. Server catches and returns MALFORMED_REQUEST.
        runner = _RecordingShardedRunner(accept_probs_kwarg=False)
        server, settler, _ = _build_server(recording_runner=runner)
        req_bytes = _build_verify_request(
            settler=settler,
            proposed_ids=(10, 20),
            proposed_probs=(0.6, 0.4),
        )
        response_bytes = server.handle(req_bytes)
        msg = parse_message(response_bytes)
        assert isinstance(msg, StageError)
        assert msg.code == StageErrorCode.MALFORMED_REQUEST.value
        assert "v2 stochastic" in msg.message
        assert "proposed_token_probs" in msg.message

    def test_v1_dispatch_against_stale_runner_unaffected(self):
        # Stale runner that rejects proposed_token_probs= still
        # works on v1 traffic — the kwarg is omitted entirely.
        # Backwards-compat regression: pre-3.x.11.y.x deployments
        # must keep functioning.
        runner = _RecordingShardedRunner(accept_probs_kwarg=False)
        server, settler, _ = _build_server(recording_runner=runner)
        req_bytes = _build_verify_request(
            settler=settler, proposed_ids=(10, 20),
            proposed_probs=None,
        )
        response_bytes = server.handle(req_bytes)
        msg = parse_message(response_bytes)
        assert isinstance(msg, RunLayerSliceResponse)
        assert len(runner.calls) == 1
        assert "proposed_token_probs" not in runner.calls[0]
