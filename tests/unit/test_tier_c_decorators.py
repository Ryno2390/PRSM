"""Phase 3.x.10.y Task 2 — unit tests for ``BatchedTrailingStreamingRunner``
(M2 decorator from the timing-sidechannel memo §5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import pytest

from prsm.compute.inference import (
    BatchedTrailingStreamingRunner,
    StreamingChunk,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


class _StubInner:
    """Minimal StreamingLayerRunner-conforming inner that emits a
    fixed sequence of StreamingChunks. Records its dispatch
    arguments so tests can assert pass-through behavior."""

    def __init__(self, chunks: List[StreamingChunk]) -> None:
        self._chunks = chunks
        self.last_call: dict = {}

    def run_layer_slice_streaming(
        self,
        *,
        model: Any,
        layer_range: Tuple[int, int],
        activation: np.ndarray,
        privacy_tier: PrivacyLevel,
        is_final_stage: bool,
        request: Any = None,
    ) -> Iterator[StreamingChunk]:
        self.last_call = dict(
            model=model, layer_range=layer_range,
            privacy_tier=privacy_tier,
            is_final_stage=is_final_stage, request=request,
        )
        yield from self._chunks


def _drive(
    decorator: BatchedTrailingStreamingRunner,
    *, is_final_stage: bool = True, request: Any = None,
) -> List[StreamingChunk]:
    return list(
        decorator.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 1),
            activation=np.zeros((1, 1), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=is_final_stage,
            request=request,
        )
    )


def _make_three_chunk_stream() -> List[StreamingChunk]:
    """Three-chunk stream — two non-terminal + one terminal with
    aggregate fields populated per Phase 3.x.8 Task 2 contract."""
    return [
        StreamingChunk(
            sequence_index=0, text_delta="aa ", token_id=1,
        ),
        StreamingChunk(
            sequence_index=1, text_delta="bb ", token_id=2,
        ),
        StreamingChunk(
            sequence_index=2, text_delta="cc",
            token_id=3,
            finish_reason="stop",
            full_output_text="aa bb cc",
            duration_seconds=0.123,
            tee_attestation=b"\x05" * 32,
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_run_layer_slice_streaming_returns_iterator(self):
        decorator = BatchedTrailingStreamingRunner(
            _StubInner(_make_three_chunk_stream()),
        )
        gen = decorator.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 1),
            activation=np.zeros((1, 1), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=True,
        )
        assert hasattr(gen, "__iter__")
        assert hasattr(gen, "__next__")

    def test_dispatch_args_pass_through_to_inner(self):
        inner = _StubInner(_make_three_chunk_stream())
        decorator = BatchedTrailingStreamingRunner(inner)
        _drive(decorator)
        assert inner.last_call["layer_range"] == (0, 1)
        assert inner.last_call["privacy_tier"] == PrivacyLevel.NONE
        assert inner.last_call["is_final_stage"] is True


class TestBatchedTrailingHappyPath:
    def test_emits_exactly_one_terminal_chunk(self):
        # Three inner chunks → exactly ONE wire chunk emitted.
        # Per-token timing of the inner stream is unobservable.
        decorator = BatchedTrailingStreamingRunner(
            _StubInner(_make_three_chunk_stream()),
        )
        chunks = _drive(decorator)
        assert len(chunks) == 1

    def test_text_delta_is_joined_inner_text(self):
        decorator = BatchedTrailingStreamingRunner(
            _StubInner(_make_three_chunk_stream()),
        )
        chunks = _drive(decorator)
        # The single emitted chunk's text_delta is the
        # concatenation of all inner chunks' text_deltas.
        assert chunks[0].text_delta == "aa bb cc"

    def test_terminal_aggregate_fields_propagate_from_inner(self):
        decorator = BatchedTrailingStreamingRunner(
            _StubInner(_make_three_chunk_stream()),
        )
        chunks = _drive(decorator)
        terminal = chunks[0]
        # All Phase 3.x.8 Task 2 aggregate fields propagated
        # from inner terminal.
        assert terminal.finish_reason == "stop"
        assert terminal.full_output_text == "aa bb cc"
        assert terminal.duration_seconds == 0.123
        assert terminal.tee_attestation == b"\x05" * 32
        assert terminal.tee_type == TEEType.SOFTWARE
        assert terminal.epsilon_spent == 0.0
        # Single-frame emission means sequence_index=0.
        assert terminal.sequence_index == 0
        # token_id intentionally None — batched output has no
        # single triggering token.
        assert terminal.token_id is None

    def test_server_joined_text_invariant_holds(self):
        # The Phase 3.x.8 server-side invariant
        # (``"".join(text_deltas) == terminal.full_output_text``)
        # MUST hold for the decorated output. Trivially true for
        # a single-frame emission, but pinning explicitly because
        # this is the wire-side correctness contract.
        decorator = BatchedTrailingStreamingRunner(
            _StubInner(_make_three_chunk_stream()),
        )
        chunks = _drive(decorator)
        joined = "".join(c.text_delta for c in chunks)
        assert joined == chunks[-1].full_output_text


class TestBatchedTrailingEdgeCases:
    def test_empty_inner_stream_yields_nothing(self):
        # No chunks from inner → no wire emission. Server
        # interprets as StageError per Phase 3.x.8 contract
        # (no signed receipt material).
        decorator = BatchedTrailingStreamingRunner(_StubInner([]))
        chunks = _drive(decorator)
        assert chunks == []

    def test_single_chunk_inner_stream_passes_through(self):
        # Inner emits a single terminal chunk (zero-token edge case
        # from AutoregressiveStreamingRunner immediate-EOS path).
        # Decorator emits one chunk with the same aggregates.
        inner_chunks = [
            StreamingChunk(
                sequence_index=0, text_delta="",
                finish_reason="stop", full_output_text="",
                duration_seconds=0.001,
                tee_attestation=b"\x06" * 32,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
            ),
        ]
        decorator = BatchedTrailingStreamingRunner(
            _StubInner(inner_chunks),
        )
        chunks = _drive(decorator)
        assert len(chunks) == 1
        assert chunks[0].text_delta == ""
        assert chunks[0].full_output_text == ""
        assert chunks[0].finish_reason == "stop"

    def test_request_kwarg_passes_through_to_inner(self):
        # Sampling shim from Phase 3.x.10.x is forwarded to the
        # inner runner — decorator MUST NOT consume / strip it.
        from prsm.compute.inference import StreamingSamplingShim

        inner = _StubInner(_make_three_chunk_stream())
        decorator = BatchedTrailingStreamingRunner(inner)
        shim = StreamingSamplingShim(max_tokens=4, temperature=0.7)
        _drive(decorator, request=shim)
        assert inner.last_call["request"] is shim

    def test_inner_error_finish_reason_propagates(self):
        # Inner runner's mid-decode exception path (Phase 3.x.10
        # round-1 H1 fix) yields buffered pieces + terminal error
        # chunk. The decorator joins them into a single batched
        # frame with finish_reason="error" carried from inner
        # terminal.
        inner_chunks = [
            StreamingChunk(
                sequence_index=0, text_delta="partial1",
                token_id=1,
            ),
            StreamingChunk(
                sequence_index=1, text_delta="partial2",
                token_id=2,
            ),
            StreamingChunk(
                sequence_index=2, text_delta="",
                finish_reason="error",
                full_output_text="partial1partial2",
                duration_seconds=0.05,
                tee_attestation=b"\x07" * 32,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
            ),
        ]
        decorator = BatchedTrailingStreamingRunner(
            _StubInner(inner_chunks),
        )
        chunks = _drive(decorator)
        assert len(chunks) == 1
        assert chunks[0].text_delta == "partial1partial2"
        assert chunks[0].finish_reason == "error"
        assert chunks[0].full_output_text == "partial1partial2"


class TestBatchedTrailingConstructorValidation:
    def test_rejects_none_inner(self):
        with pytest.raises(RuntimeError, match="inner"):
            BatchedTrailingStreamingRunner(None)  # type: ignore[arg-type]

    def test_rejects_non_runner_inner(self):
        # Object without run_layer_slice_streaming method.
        with pytest.raises(
            RuntimeError, match="run_layer_slice_streaming",
        ):
            BatchedTrailingStreamingRunner(object())


# ──────────────────────────────────────────────────────────────────────────
# Composition with AutoregressiveStreamingRunner — slow-marked
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _hf_model_and_tokenizer_for_decorator_compose():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "distilgpt2",
        )
        model.eval()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"distilgpt2 unavailable: "
            f"{exc.__class__.__name__}: {exc}"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.mark.slow
class TestComposeWithAutoregressiveRunner:
    """Confirms the decorator composes with the real
    AutoregressiveStreamingRunner against distilgpt2 — no per-
    token frames reach the wire, but the receipt invariants
    (signed via streaming wire format with streamed_output=True)
    hold."""

    def test_decorated_real_runner_emits_single_wire_frame(
        self, _hf_model_and_tokenizer_for_decorator_compose,
    ):
        model, tokenizer = _hf_model_and_tokenizer_for_decorator_compose
        from prsm.compute.inference import (
            SamplingDefaults,
            make_autoregressive_streaming_runner,
        )

        inner = make_autoregressive_streaming_runner(
            model=model,
            tokenizer=tokenizer,
            tee_attestation=b"\x07" * 32,
            prompt_provider=lambda lr, act, pt: "The quick brown fox",
            sampling_defaults=SamplingDefaults(max_tokens=4),
        )
        decorator = BatchedTrailingStreamingRunner(inner)
        chunks = _drive(decorator)

        # The whole point: per-token frames are NOT observable on
        # the decorated output. Exactly ONE wire frame.
        assert len(chunks) == 1
        terminal = chunks[0]
        # 3.x.10.y Task 1 prompt-skip: "The quick brown fox" must
        # not appear in the decorated output.
        assert "The quick brown fox" not in terminal.text_delta
        # Joined-text invariant: the single emitted chunk's
        # text_delta equals the inner runner's joined output.
        assert terminal.text_delta == terminal.full_output_text
        # Cap-hit invariant from 3.x.10.x.
        assert terminal.finish_reason == "max_tokens"
