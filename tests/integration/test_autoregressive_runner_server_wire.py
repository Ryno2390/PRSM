"""Phase 3.x.10 Task 8 round-1 M3+H1 remediation — server-wired
integration tests for ``AutoregressiveStreamingRunner``.

Closes the M3 finding from round-1 review: the original
``test_autoregressive_runner_e2e.py`` drove the runner DIRECTLY
(bypassing ``LayerStageServer.handle_token_stream``) — so the H1
finding (mid-decode exception path violating the server's
joined-text invariant) was undetectable by automated CI.

These tests wire ``AutoregressiveStreamingRunner`` (with mocked HF
model + tokenizer; no real distilgpt2 needed) through a real
``LayerStageServer.handle_token_stream`` and assert:

  1. Happy-path: TokenFrame deltas join to ``StreamFinalFrame.response.activation_blob``
     bytes — the server's joined-text invariant holds end-to-end.
  2. Mid-decode exception (H1 fix): buffered partial pieces emit
     as TokenFrames first, then terminal — joined deltas still
     match terminal's ``full_output_text``, so the server does NOT
     reject with INTERNAL_ERROR. Pre-fix, this test would fail
     with the server emitting a generic StageError instead of
     the expected wire frames + StreamFinalFrame committing to
     the partial.
  3. Sequence-index monotonicity holds across the partial-then-
     terminal sequence.

Mocked HF model + tokenizer keep the tests fast (sub-second).
The real distilgpt2 E2E lives in
``test_autoregressive_runner_e2e.py`` (slow-marked).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pytest

from prsm.compute.chain_rpc.protocol import (
    HandoffToken,
    RunLayerSliceRequest,
    StageError,
    StreamFinalFrame,
    TokenFrame,
    encode_message,
    parse_message,
)
from prsm.compute.chain_rpc.server import (
    LayerSliceResult,
    LayerSliceRunner,
    LayerStageServer,
)
from prsm.compute.inference.autoregressive_runner import (
    AutoregressiveStreamingRunner,
    SamplingDefaults,
)
from prsm.compute.inference.models import ContentTier
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Minimal inlined fakes (mirrors test_chain_rpc_server.py's pattern).
# ──────────────────────────────────────────────────────────────────────────


class _Anchor:
    def __init__(self) -> None:
        self.registered: Dict[str, str] = {}

    def register(self, identity) -> None:
        self.registered[identity.node_id] = identity.public_key_b64

    def lookup(self, node_id: str) -> Optional[str]:
        return self.registered.get(node_id)


@dataclass
class _Shard:
    layer_range: Tuple[int, int]


@dataclass
class _Model:
    model_id: str
    shards: List[_Shard] = field(default_factory=list)

    @classmethod
    def linear_chain(cls, model_id: str) -> "_Model":
        return cls(model_id=model_id, shards=[_Shard(layer_range=(0, 4))])


@dataclass
class _Registry:
    models: Dict[str, _Model] = field(default_factory=dict)

    def get(self, model_id: str) -> _Model:
        if model_id not in self.models:
            raise _ModelNotFoundError(model_id)
        return self.models[model_id]


class _ModelNotFoundError(Exception):
    pass


class _TEERuntime:
    def __init__(self) -> None:
        self.tee_type = TEEType.SOFTWARE


class _UnaryRunner(LayerSliceRunner):
    """Pass-through unary runner (server requires it for non-streaming
    paths even when this test only exercises the streaming path)."""

    def run_layer_range(
        self, *, model, layer_range, activation, privacy_tier, is_final_stage,
    ) -> LayerSliceResult:
        return LayerSliceResult(
            output=activation.copy(),
            duration_seconds=0.001,
            tee_attestation=b"\x09" * 32,
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )


# ──────────────────────────────────────────────────────────────────────────
# HF-shaped fakes (re-uses pattern from test_autoregressive_runner.py).
# ──────────────────────────────────────────────────────────────────────────


class _FakeTokenizer:
    def __init__(
        self, *, id_to_piece: Dict[int, str],
        eos_token_id: Optional[int] = None,
    ) -> None:
        self._id_to_piece = id_to_piece
        self.eos_token_id = eos_token_id
        self._special: set = set()
        if eos_token_id is not None:
            self._special.add(eos_token_id)

    def encode(self, text: str) -> List[int]:
        return [100, 101]

    def decode(self, ids, skip_special_tokens=True) -> str:
        out = []
        for tid in ids:
            if skip_special_tokens and tid in self._special:
                continue
            out.append(self._id_to_piece.get(int(tid), ""))
        return "".join(out)


class _FakeModel:
    def __init__(
        self, *, emit_ids: List[int],
        raise_after: Optional[int] = None,
        raise_exc: Optional[Exception] = None,
    ) -> None:
        self.emit_ids = emit_ids
        self.raise_after = raise_after
        self.raise_exc = raise_exc

    def generate(
        self, *, input_ids, streamer, max_new_tokens, temperature,
        do_sample, top_k, top_p, eos_token_id,
    ):
        if hasattr(input_ids, "tolist"):
            as_list = input_ids.tolist()
            if (
                isinstance(as_list, list)
                and as_list and isinstance(as_list[0], list)
            ):
                as_list = as_list[0]
            input_ids = as_list
        emitted: List[int] = []
        for i, tid in enumerate(self.emit_ids):
            if i >= max_new_tokens:
                break
            if (
                self.raise_after is not None
                and i >= self.raise_after
                and self.raise_exc is not None
            ):
                raise self.raise_exc
            streamer.put(tid)
            emitted.append(tid)
            if eos_token_id is not None and tid == eos_token_id:
                break
        streamer.end()
        return list(input_ids) + emitted


def _build_server(
    *, emit_ids: List[int], id_to_piece: Dict[int, str],
    eos_token_id: Optional[int] = None,
    raise_after: Optional[int] = None,
    raise_exc: Optional[Exception] = None,
) -> Tuple[LayerStageServer, Any, Any]:
    stage_identity = generate_node_identity(display_name="stage")
    settler_identity = generate_node_identity(display_name="settler")
    anchor = _Anchor()
    anchor.register(stage_identity)
    anchor.register(settler_identity)

    registry = _Registry()
    registry.models["test-model"] = _Model.linear_chain("test-model")

    tok = _FakeTokenizer(
        id_to_piece=id_to_piece, eos_token_id=eos_token_id,
    )
    mdl = _FakeModel(
        emit_ids=emit_ids,
        raise_after=raise_after, raise_exc=raise_exc,
    )
    streaming_runner = AutoregressiveStreamingRunner(
        model=mdl,
        tokenizer=tok,
        tee_attestation=b"\x09" * 32,
        tee_type=TEEType.SOFTWARE,
        sampling_defaults=SamplingDefaults(max_tokens=16),
        prompt_provider=lambda lr, act, pt: "fixed-prompt",
    )

    server = LayerStageServer(
        identity=stage_identity,
        registry=registry,
        runner=_UnaryRunner(),
        tee_runtime=_TEERuntime(),
        anchor=anchor,
        clock=lambda: 1000.0,
        streaming_runner=streaming_runner,
    )
    return server, settler_identity, stage_identity


def _make_streaming_request(
    *, settler_identity, deadline: float = 2000.0,
) -> RunLayerSliceRequest:
    activation = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    blob = activation.tobytes()
    token = HandoffToken.sign(
        identity=settler_identity,
        request_id="req-stream-1",
        chain_stage_index=0,
        chain_total_stages=2,
        deadline_unix=deadline,
    )
    return RunLayerSliceRequest(
        request_id="req-stream-1",
        model_id="test-model",
        layer_range=(0, 4),  # tail
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        activation_blob=blob,
        activation_shape=tuple(activation.shape),
        activation_dtype=str(activation.dtype),
        upstream_token=token,
        deadline_unix=deadline,
        streaming=True,
    )


def _decode(
    iter_bytes: Iterable[bytes],
) -> Tuple[List[TokenFrame], Optional[StreamFinalFrame], Optional[StageError]]:
    tokens: List[TokenFrame] = []
    final: Optional[StreamFinalFrame] = None
    err: Optional[StageError] = None
    for raw in iter_bytes:
        msg = parse_message(raw)
        if isinstance(msg, TokenFrame):
            tokens.append(msg)
        elif isinstance(msg, StreamFinalFrame):
            final = msg
        elif isinstance(msg, StageError):
            err = msg
        else:
            raise AssertionError(
                f"unexpected wire type: {type(msg).__name__}"
            )
    return tokens, final, err


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


class TestAutoregressiveRunnerThroughLayerStageServer:
    def test_happy_path_joined_invariant_holds(self):
        # Server consumes AutoregressiveStreamingRunner, emits
        # TokenFrames + StreamFinalFrame. Joined text_deltas equal
        # the StreamFinalFrame.response.activation_blob bytes.
        server, settler_identity, _ = _build_server(
            emit_ids=[1, 2, 3],
            id_to_piece={1: "alpha ", 2: "beta ", 3: "gamma"},
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        tokens, final, err = _decode(
            server.handle_token_stream(encode_message(req))
        )
        assert err is None
        assert final is not None
        assert len(tokens) == 3
        # Joined-text invariant.
        joined = "".join(t.text_delta for t in tokens)
        assert joined == "alpha beta gamma"
        assert final.response.activation_blob == joined.encode("utf-8")
        # Sequence indices strictly increasing 0-based.
        assert [t.sequence_index for t in tokens] == [0, 1, 2]
        # Only the terminal TokenFrame carries finish_reason.
        for t in tokens[:-1]:
            assert t.finish_reason is None
        assert tokens[-1].finish_reason == "max_tokens"

    def test_mid_decode_exception_partial_emits_then_terminal(self):
        # H1 round-1 remediation regression test. Pre-fix: the
        # runner yielded ONE terminal error chunk with text_delta=""
        # but full_output_text=partial — server's joined-text
        # invariant rejected this and emitted a generic StageError,
        # so the receipt never committed to the partial. Post-fix:
        # buffered pieces emit as non-terminal TokenFrames first,
        # then a terminal error chunk; joined matches
        # full_output_text and the wire contract holds.
        server, settler_identity, _ = _build_server(
            emit_ids=[1, 2, 3, 4],
            id_to_piece={1: "aa ", 2: "bb ", 3: "cc ", 4: "dd"},
            raise_after=2,
            raise_exc=RuntimeError("boom"),
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        tokens, final, err = _decode(
            server.handle_token_stream(encode_message(req))
        )
        # Server's terminal-chunk integrity check + joined-text
        # invariant both pass; runner's terminal "error" finish_reason
        # is mapped to a normal StreamFinalFrame committing to the
        # partial.
        assert err is None
        assert final is not None
        # 2 non-terminal pieces + 1 terminal error frame.
        assert len(tokens) == 3
        assert tokens[0].text_delta == "aa "
        assert tokens[0].finish_reason is None
        assert tokens[1].text_delta == "bb "
        assert tokens[1].finish_reason is None
        assert tokens[-1].text_delta == ""
        assert tokens[-1].finish_reason == "error"
        # Joined-text invariant holds.
        joined = "".join(t.text_delta for t in tokens)
        assert joined == "aa bb "
        assert final.response.activation_blob == joined.encode("utf-8")

    def test_mid_decode_exception_sequence_indices_monotonic(self):
        server, settler_identity, _ = _build_server(
            emit_ids=[1, 2, 3, 4, 5],
            id_to_piece={i: f"t{i} " for i in range(1, 6)},
            raise_after=3,
            raise_exc=RuntimeError("boom"),
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        tokens, _, err = _decode(
            server.handle_token_stream(encode_message(req))
        )
        # H1 fix: 3 partial pieces + 1 terminal = 4 frames; indices
        # are 0..3 with no gaps.
        assert err is None
        assert [t.sequence_index for t in tokens] == [0, 1, 2, 3]

    def test_pre_decode_exception_yields_stage_error(self):
        # Crash BEFORE any token emits. The runner yields a single
        # terminal error chunk with empty text + empty
        # full_output_text. Server cannot construct a signable
        # StreamFinalFrame over empty activation_blob (the wire
        # protocol's activation_shape entries must be positive
        # int), so it correctly falls through to a StageError —
        # there's no signed-receipt material to commit to anyway.
        # This is the "empty stream" honest path.
        server, settler_identity, _ = _build_server(
            emit_ids=[1, 2, 3],
            id_to_piece={1: "aa ", 2: "bb ", 3: "cc "},
            raise_after=0,
            raise_exc=RuntimeError("boom"),
        )
        req = _make_streaming_request(settler_identity=settler_identity)
        tokens, final, err = _decode(
            server.handle_token_stream(encode_message(req))
        )
        # Server emits StageError when terminal chunk's
        # full_output_text is empty (can't sign over empty
        # activation_blob).
        assert err is not None
        assert err.code == "INTERNAL_ERROR"
        assert final is None
