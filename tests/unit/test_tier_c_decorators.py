"""Phase 3.x.10.y Task 2-3 — unit tests for the Tier-C
constant-time padding decorators (``BatchedTrailingStreamingRunner``
M2 + ``FixedRateStreamingRunner`` M1) from the timing-sidechannel
memo §5.

NOTE: ``conftest.py`` autouse-mocks ``time.sleep`` globally to keep
the test suite fast. The FixedRate tests below specifically
validate real-time cadence behavior, so this file overrides the
mock with a same-named fixture that does NOT patch sleep —
``time.sleep`` runs as real wall-clock sleep for every test in
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_time_sleep():
    """Override ``conftest.py``'s global ``time.sleep`` mock.

    Cadence-mask tests in this file MUST observe real wall-clock
    timing — mocking sleep makes the FixedRate decorator's
    interval pacing collapse to instant emission, which would
    silently pass the timing-mask invariant assertions even
    when the production code is broken.
    """
    yield

import time

from prsm.compute.inference import (
    BatchedTrailingStreamingRunner,
    FixedRateStreamingRunner,
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


# ──────────────────────────────────────────────────────────────────────────
# FixedRateStreamingRunner (M1 — cadence-driven emission)
# ──────────────────────────────────────────────────────────────────────────


class _SlowInner:
    """Inner runner that sleeps before yielding each chunk —
    simulates per-token decode latency variance the M1
    decorator is designed to mask."""

    def __init__(
        self,
        chunks: List[StreamingChunk],
        sleep_per_chunk: float = 0.0,
    ) -> None:
        self._chunks = chunks
        self._sleep = sleep_per_chunk

    def run_layer_slice_streaming(
        self, *, model, layer_range, activation, privacy_tier,
        is_final_stage, request=None,
    ) -> Iterator[StreamingChunk]:
        for chunk in self._chunks:
            if self._sleep:
                time.sleep(self._sleep)
            yield chunk


class _RaisingInner:
    """Inner runner that raises mid-iteration."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def run_layer_slice_streaming(
        self, *, model, layer_range, activation, privacy_tier,
        is_final_stage, request=None,
    ) -> Iterator[StreamingChunk]:
        raise self._exc
        yield  # pragma: no cover (required for generator)


def _drive_fixed_rate(
    decorator: FixedRateStreamingRunner,
) -> List[StreamingChunk]:
    return list(
        decorator.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 1),
            activation=np.zeros((1, 1), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=True,
        )
    )


class TestFixedRateProtocolConformance:
    def test_returns_iterator(self):
        decorator = FixedRateStreamingRunner(
            _SlowInner(_make_three_chunk_stream()),
            cadence_seconds=0.01,
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
        # Drain so the producer thread completes cleanly.
        list(gen)


class TestFixedRateConstructorValidation:
    def test_rejects_none_inner(self):
        with pytest.raises(RuntimeError, match="inner"):
            FixedRateStreamingRunner(None)  # type: ignore[arg-type]

    def test_rejects_non_runner_inner(self):
        with pytest.raises(
            RuntimeError, match="run_layer_slice_streaming",
        ):
            FixedRateStreamingRunner(object())

    def test_rejects_non_numeric_cadence(self):
        with pytest.raises(RuntimeError, match="cadence_seconds"):
            FixedRateStreamingRunner(
                _SlowInner([]), cadence_seconds="0.05",  # type: ignore[arg-type]
            )

    def test_rejects_zero_cadence(self):
        with pytest.raises(RuntimeError, match="positive"):
            FixedRateStreamingRunner(
                _SlowInner([]), cadence_seconds=0,
            )

    def test_rejects_negative_cadence(self):
        with pytest.raises(RuntimeError, match="positive"):
            FixedRateStreamingRunner(
                _SlowInner([]), cadence_seconds=-0.01,
            )

    def test_rejects_bool_cadence(self):
        with pytest.raises(RuntimeError, match="cadence_seconds"):
            FixedRateStreamingRunner(
                _SlowInner([]), cadence_seconds=True,  # type: ignore[arg-type]
            )


class TestFixedRateCadenceTimingMasking:
    def test_inter_frame_intervals_match_cadence(self):
        # Inner produces 3 chunks instantly; decorator emits at
        # 30ms cadence. Inter-frame deltas MUST be ≥ cadence.
        cadence = 0.030
        inner = _SlowInner(_make_three_chunk_stream())
        decorator = FixedRateStreamingRunner(
            inner, cadence_seconds=cadence,
        )
        timestamps: List[float] = []
        for _ in decorator.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 1),
            activation=np.zeros((1, 1), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=True,
        ):
            timestamps.append(time.monotonic())
        # At least 2 frames (first non-terminal + terminal)
        # before any backlog draining.
        assert len(timestamps) >= 2
        # Each consecutive pair must be at least one cadence
        # apart (with small tolerance for thread scheduling).
        tolerance = 0.005
        for i, (prev, cur) in enumerate(
            zip(timestamps, timestamps[1:])
        ):
            assert (cur - prev) >= (cadence - tolerance), (
                f"inter-frame interval [{i}→{i+1}] "
                f"{(cur - prev) * 1000:.4f}ms < "
                f"cadence {cadence * 1000:.0f}ms — timing mask "
                f"violated. All deltas (ms): "
                f"{[(t1-t0)*1000 for t0, t1 in zip(timestamps, timestamps[1:])]}"
            )

    def test_no_op_frames_pad_slow_inner(self):
        # Inner is slow (50ms per chunk), cadence is fast (10ms).
        # Decorator MUST emit no-op pad frames between real chunks.
        cadence = 0.010
        sleep_per_chunk = 0.050
        inner = _SlowInner(
            _make_three_chunk_stream(),
            sleep_per_chunk=sleep_per_chunk,
        )
        decorator = FixedRateStreamingRunner(
            inner, cadence_seconds=cadence,
        )
        chunks = _drive_fixed_rate(decorator)
        # More wire frames than inner chunks → no-op pads
        # exist.
        assert len(chunks) > 3
        # Some chunks have empty text_delta + no
        # finish_reason → those are no-op pads.
        no_op_count = sum(
            1 for c in chunks
            if c.text_delta == "" and c.finish_reason is None
        )
        assert no_op_count >= 1


class TestFixedRateTerminalDrain:
    def test_terminal_aggregate_fields_propagate(self):
        decorator = FixedRateStreamingRunner(
            _SlowInner(_make_three_chunk_stream()),
            cadence_seconds=0.005,
        )
        chunks = _drive_fixed_rate(decorator)
        # Terminal must be the LAST chunk and carry all aggregate
        # fields from the inner terminal.
        terminal = chunks[-1]
        assert terminal.finish_reason == "stop"
        assert terminal.full_output_text == "aa bb cc"
        assert terminal.duration_seconds == 0.123
        assert terminal.tee_attestation == b"\x05" * 32
        assert terminal.tee_type == TEEType.SOFTWARE
        assert terminal.epsilon_spent == 0.0

    def test_joined_text_invariant_holds(self):
        # Server-side invariant: ".join(text_deltas) ==
        # terminal.full_output_text" — no-op frames have empty
        # text_delta so they don't violate this.
        decorator = FixedRateStreamingRunner(
            _SlowInner(_make_three_chunk_stream()),
            cadence_seconds=0.005,
        )
        chunks = _drive_fixed_rate(decorator)
        joined = "".join(c.text_delta for c in chunks)
        assert joined == chunks[-1].full_output_text

    def test_sequence_indices_strictly_increasing(self):
        decorator = FixedRateStreamingRunner(
            _SlowInner(_make_three_chunk_stream()),
            cadence_seconds=0.005,
        )
        chunks = _drive_fixed_rate(decorator)
        idx = [c.sequence_index for c in chunks]
        assert idx == sorted(idx)
        assert idx == list(range(len(idx)))


class TestFixedRateErrorPath:
    def test_inner_raises_yields_terminal_error(self):
        decorator = FixedRateStreamingRunner(
            _RaisingInner(RuntimeError("boom")),
            cadence_seconds=0.005,
        )
        chunks = _drive_fixed_rate(decorator)
        assert len(chunks) >= 1
        terminal = chunks[-1]
        assert terminal.finish_reason == "error"
        # Empty aggregates — server surfaces as StageError per
        # the existing Phase 3.x.8 contract.
        assert terminal.full_output_text == ""
        assert terminal.text_delta == ""

    def test_inner_error_does_not_leak_partial_text_via_pad_frames(self):
        # All wire frames before the terminal MUST have empty
        # text_delta (no-op pads). The decorator's error path
        # MUST NOT leak any partial inner state.
        decorator = FixedRateStreamingRunner(
            _RaisingInner(RuntimeError("secret-state")),
            cadence_seconds=0.005,
        )
        chunks = _drive_fixed_rate(decorator)
        for c in chunks[:-1]:
            assert c.text_delta == ""
        # Terminal also empty (no leak via aggregates).
        assert "secret-state" not in (
            chunks[-1].text_delta or ""
        )
        assert "secret-state" not in (
            chunks[-1].full_output_text or ""
        )


class TestFixedRateEmptyInner:
    def test_empty_inner_yields_nothing(self):
        # Inner with no chunks → decorator emits nothing
        # (matches Phase 3.x.8 server contract for empty
        # streams: StageError surfaces).
        decorator = FixedRateStreamingRunner(
            _SlowInner([]),
            cadence_seconds=0.005,
        )
        chunks = _drive_fixed_rate(decorator)
        assert chunks == []

    def test_request_kwarg_passes_through_to_inner(self):
        from prsm.compute.inference import StreamingSamplingShim

        recorded: list = []

        class _RecordingInner:
            def run_layer_slice_streaming(
                self, *, model, layer_range, activation,
                privacy_tier, is_final_stage, request=None,
            ) -> Iterator[StreamingChunk]:
                recorded.append(request)
                yield from _make_three_chunk_stream()

        shim = StreamingSamplingShim(max_tokens=4, temperature=0.7)
        decorator = FixedRateStreamingRunner(
            _RecordingInner(), cadence_seconds=0.005,
        )
        list(decorator.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 1),
            activation=np.zeros((1, 1), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=True,
            request=shim,
        ))
        assert recorded == [shim]


# ──────────────────────────────────────────────────────────────────────────
# FixedRate compose with AutoregressiveStreamingRunner — slow-marked
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestFixedRateComposeWithAutoregressiveRunner:
    def test_decorated_real_runner_emits_at_cadence(
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
        cadence = 0.030
        decorator = FixedRateStreamingRunner(
            inner, cadence_seconds=cadence,
        )
        timestamps: List[float] = []
        chunks: List[StreamingChunk] = []
        for chunk in decorator.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 1),
            activation=np.zeros((1, 1), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=True,
        ):
            timestamps.append(time.monotonic())
            chunks.append(chunk)

        # Cap-hit invariant from 3.x.10.x still holds through
        # the decorator's terminal forwarding.
        assert chunks[-1].finish_reason == "max_tokens"
        # Prompt-skip from Task 1 still holds — joined text
        # MUST NOT contain the prompt.
        joined = "".join(c.text_delta for c in chunks)
        assert "The quick brown fox" not in joined
        # Cadence honored: every consecutive pair ≥ cadence apart.
        tolerance = 0.005
        for prev, cur in zip(timestamps, timestamps[1:]):
            assert (cur - prev) >= (cadence - tolerance)
