"""Phase 3.x.8 Task 2 — unit tests for the StreamingLayerRunner
Protocol + ``SyntheticStreamingRunner`` adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

from prsm.compute.chain_rpc.server import LayerSliceResult
from prsm.compute.inference.streaming_runner import (
    StreamingChunk,
    SyntheticStreamingRunner,
    split_text_into_deltas,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType


# ──────────────────────────────────────────────────────────────────────────
# split_text_into_deltas
# ──────────────────────────────────────────────────────────────────────────


class TestSplitTextIntoDeltas:
    def test_simple_words_split_with_separators_preserved(self):
        deltas = split_text_into_deltas("hello world foo")
        assert "".join(deltas) == "hello world foo"
        assert deltas == ["hello", " ", "world", " ", "foo"]

    def test_join_invariant_on_arbitrary_strings(self):
        for s in [
            "single",
            "two words",
            "  leading space",
            "trailing space  ",
            "tabs\tand\nnewlines",
            "multiple   internal   spaces",
        ]:
            assert "".join(split_text_into_deltas(s)) == s

    def test_empty_text_returns_single_empty_delta(self):
        # Empty input still produces ONE delta so the stream has a
        # terminal frame to ride on.
        assert split_text_into_deltas("") == [""]

    def test_whitespace_only_text(self):
        deltas = split_text_into_deltas("   ")
        assert "".join(deltas) == "   "
        assert len(deltas) >= 1


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


class _FakeRunner:
    """Minimal one-shot runner exposing the LayerSliceRunner shape."""

    def __init__(
        self,
        *,
        duration: float = 0.05,
        attestation: bytes = b"\x01" * 32,
        tee_type: TEEType = TEEType.SOFTWARE,
        epsilon: float = 0.0,
        raise_on_call: Optional[Exception] = None,
    ):
        self.duration = duration
        self.attestation = attestation
        self.tee_type = tee_type
        self.epsilon = epsilon
        self.raise_on_call = raise_on_call
        self.call_count = 0

    def run_layer_range(
        self,
        *,
        model: Any,
        layer_range: Tuple[int, int],
        activation: np.ndarray,
        privacy_tier: PrivacyLevel,
        is_final_stage: bool,
    ) -> LayerSliceResult:
        self.call_count += 1
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return LayerSliceResult(
            output=activation.copy(),
            duration_seconds=self.duration,
            tee_attestation=self.attestation,
            tee_type=self.tee_type,
            epsilon_spent=self.epsilon,
        )


def _make_decoder(text: str):
    def decoder(activation: np.ndarray) -> str:
        return text
    return decoder


# ──────────────────────────────────────────────────────────────────────────
# SyntheticStreamingRunner construction
# ──────────────────────────────────────────────────────────────────────────


class TestSyntheticStreamingRunnerConstruction:
    def test_rejects_missing_runner(self):
        with pytest.raises(RuntimeError, match="LayerSliceRunner"):
            SyntheticStreamingRunner(
                runner=None,  # type: ignore[arg-type]
                output_decoder=_make_decoder("x"),
            )

    def test_rejects_runner_without_run_layer_range(self):
        class Empty:
            pass

        with pytest.raises(RuntimeError, match="LayerSliceRunner"):
            SyntheticStreamingRunner(
                runner=Empty(),  # type: ignore[arg-type]
                output_decoder=_make_decoder("x"),
            )

    def test_rejects_non_callable_decoder(self):
        with pytest.raises(RuntimeError, match="output_decoder"):
            SyntheticStreamingRunner(
                runner=_FakeRunner(),
                output_decoder="not-callable",  # type: ignore[arg-type]
            )

    def test_accepts_custom_splitter(self):
        # Custom splitter — produces fixed 2-character chunks.
        def splitter(text: str) -> List[str]:
            return [text[i:i + 2] for i in range(0, len(text), 2)] or [""]

        sr = SyntheticStreamingRunner(
            runner=_FakeRunner(),
            output_decoder=_make_decoder("abcdef"),
            splitter=splitter,
        )
        assert sr is not None  # construction success


# ──────────────────────────────────────────────────────────────────────────
# Streaming behavior
# ──────────────────────────────────────────────────────────────────────────


class TestSyntheticStreamingRunnerStreaming:
    def _run(
        self,
        *,
        text: str,
        runner: Optional[_FakeRunner] = None,
    ) -> List[StreamingChunk]:
        underlying = runner or _FakeRunner()
        sr = SyntheticStreamingRunner(
            runner=underlying,
            output_decoder=_make_decoder(text),
        )
        chunks = list(sr.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 4),
            activation=np.zeros((1, 4), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=True,
        ))
        return chunks

    def test_streams_at_least_one_chunk_for_nonempty_text(self):
        chunks = self._run(text="hello world")
        assert len(chunks) >= 1

    def test_joined_text_deltas_equal_full_output(self):
        text = "hello world from synthetic"
        chunks = self._run(text=text)
        joined = "".join(c.text_delta for c in chunks)
        assert joined == text

    def test_terminal_chunk_finish_reason_is_stop(self):
        chunks = self._run(text="hello world")
        assert chunks[-1].finish_reason == "stop"
        # All non-terminal chunks have None finish_reason.
        for c in chunks[:-1]:
            assert c.finish_reason is None

    def test_terminal_chunk_carries_aggregate_fields(self):
        runner = _FakeRunner(
            duration=0.123,
            attestation=b"\xaa" * 16,
            tee_type=TEEType.SGX,
            epsilon=0.5,
        )
        chunks = self._run(text="abc def", runner=runner)
        last = chunks[-1]
        assert last.full_output_text == "abc def"
        assert last.duration_seconds == 0.123
        assert last.tee_attestation == b"\xaa" * 16
        assert last.tee_type == TEEType.SGX
        assert last.epsilon_spent == 0.5

    def test_sequence_indices_are_strictly_increasing(self):
        chunks = self._run(text="a b c d e f")
        assert [c.sequence_index for c in chunks] == list(range(len(chunks)))

    def test_runner_invoked_exactly_once_for_full_forward_pass(self):
        runner = _FakeRunner()
        list(self._run(text="anything", runner=runner))
        # Synthetic runner runs the underlying ONCE — that's the
        # "one-shot wrapped as streaming" contract.
        assert runner.call_count == 1

    def test_underlying_runner_exception_propagates(self):
        runner = _FakeRunner(raise_on_call=RuntimeError("inner failed"))
        sr = SyntheticStreamingRunner(
            runner=runner,
            output_decoder=_make_decoder("x"),
        )
        # The exception propagates out of the generator — the server's
        # handle_token_stream catches it and emits a StageError frame.
        with pytest.raises(RuntimeError, match="inner failed"):
            list(sr.run_layer_slice_streaming(
                model=None,
                layer_range=(0, 4),
                activation=np.zeros((1,), dtype=np.float32),
                privacy_tier=PrivacyLevel.NONE,
                is_final_stage=True,
            ))

    def test_decoder_returning_non_string_raises_type_error(self):
        sr = SyntheticStreamingRunner(
            runner=_FakeRunner(),
            output_decoder=lambda a: 12345,  # type: ignore[arg-type]
        )
        with pytest.raises(TypeError, match="output_decoder must return str"):
            list(sr.run_layer_slice_streaming(
                model=None,
                layer_range=(0, 4),
                activation=np.zeros((1,), dtype=np.float32),
                privacy_tier=PrivacyLevel.NONE,
                is_final_stage=True,
            ))

    def test_empty_text_still_produces_terminal_chunk(self):
        # Decoder returns "" — splitter returns [""] — runner emits
        # ONE terminal chunk with empty text_delta.
        chunks = self._run(text="")
        assert len(chunks) == 1
        assert chunks[0].text_delta == ""
        assert chunks[0].finish_reason == "stop"
        assert chunks[0].full_output_text == ""

    def test_custom_splitter_used(self):
        def splitter(text: str) -> List[str]:
            # Split into single characters.
            return list(text) if text else [""]

        sr = SyntheticStreamingRunner(
            runner=_FakeRunner(),
            output_decoder=_make_decoder("abc"),
            splitter=splitter,
        )
        chunks = list(sr.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 4),
            activation=np.zeros((1,), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=True,
        ))
        assert len(chunks) == 3
        assert "".join(c.text_delta for c in chunks) == "abc"
