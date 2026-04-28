"""Phase 3.x.10 Task 1 — unit tests for ``AutoregressiveStreamingRunner``.

Tests use a hand-rolled mocked HF-shaped model + tokenizer that
exposes the minimum surface the runner consumes:
``model.generate(input_ids=, streamer=, ...)`` calls
``streamer.put(token)`` for a fixed sequence, then
``streamer.end()``, then returns the full output_ids tensor-like.
``tokenizer.encode(text)`` returns a deterministic id list and
``tokenizer.decode(ids, skip_special_tokens=True)`` reconstructs
text per a fixed id→str table.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from prsm.compute.inference.autoregressive_runner import (
    AutoregressiveStreamingRunner,
    SamplingDefaults,
    _coerce_token_ids,
    _last_token_id,
)
from prsm.compute.inference.streaming_runner import StreamingChunk
from prsm.compute.tee.models import PrivacyLevel, TEEType


# ──────────────────────────────────────────────────────────────────────────
# Mocks
# ──────────────────────────────────────────────────────────────────────────


class _FakeTokenizer:
    """Deterministic id↔text mapping. ``decode([1, 2, 3])`` returns
    the concatenated piece strings; unknown ids decode to empty
    string. Special-token ids (e.g., EOS) are skipped when
    ``skip_special_tokens=True``."""

    def __init__(
        self,
        *,
        id_to_piece: Dict[int, str],
        prompt_ids: List[int],
        eos_token_id: Optional[int] = None,
        special_ids: Optional[List[int]] = None,
    ) -> None:
        self._id_to_piece = id_to_piece
        self._prompt_ids = prompt_ids
        self.eos_token_id = eos_token_id
        self._special = set(special_ids or [])
        if eos_token_id is not None:
            self._special.add(eos_token_id)

    def encode(self, text: str) -> List[int]:
        return list(self._prompt_ids)

    def decode(
        self, ids: List[int], skip_special_tokens: bool = True,
    ) -> str:
        out = []
        for tid in ids:
            if skip_special_tokens and tid in self._special:
                continue
            out.append(self._id_to_piece.get(int(tid), ""))
        return "".join(out)


class _FakeModel:
    """``.generate()`` drives the streamer through a fixed token
    sequence. Returns concatenation of prompt_ids + emit_ids as a
    1-d list."""

    def __init__(
        self,
        *,
        emit_ids: List[int],
        raise_after: Optional[int] = None,
        raise_exc: Optional[Exception] = None,
    ) -> None:
        self.emit_ids = emit_ids
        self.raise_after = raise_after
        self.raise_exc = raise_exc
        self.last_call: Dict[str, Any] = {}

    def generate(
        self,
        *,
        input_ids: List[int],
        streamer: Any,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        top_k: int,
        top_p: float,
        eos_token_id: Optional[int],
    ) -> List[int]:
        self.last_call = dict(
            input_ids=list(input_ids),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )
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


def _prompt_provider(
    layer_range: Tuple[int, int],
    activation: np.ndarray,
    privacy_tier: PrivacyLevel,
) -> str:
    return "fixed-prompt"


def _make_runner(
    *,
    emit_ids: List[int],
    id_to_piece: Dict[int, str],
    eos_token_id: Optional[int] = None,
    special_ids: Optional[List[int]] = None,
    raise_after: Optional[int] = None,
    raise_exc: Optional[Exception] = None,
    sampling: Optional[SamplingDefaults] = None,
) -> Tuple[AutoregressiveStreamingRunner, _FakeModel, _FakeTokenizer]:
    tok = _FakeTokenizer(
        id_to_piece=id_to_piece,
        prompt_ids=[100, 101],
        eos_token_id=eos_token_id,
        special_ids=special_ids,
    )
    mdl = _FakeModel(
        emit_ids=emit_ids, raise_after=raise_after, raise_exc=raise_exc,
    )
    runner = AutoregressiveStreamingRunner(
        model=mdl,
        tokenizer=tok,
        tee_attestation=b"\x01" * 32,
        tee_type=TEEType.SOFTWARE,
        sampling_defaults=sampling or SamplingDefaults(max_tokens=16),
        prompt_provider=_prompt_provider,
    )
    return runner, mdl, tok


def _drive(
    runner: AutoregressiveStreamingRunner,
    *,
    is_final_stage: bool = True,
    request: Any = None,
) -> List[StreamingChunk]:
    return list(
        runner.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 4),
            activation=np.zeros((1, 4), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=is_final_stage,
            request=request,
        )
    )


# ──────────────────────────────────────────────────────────────────────────
# coercion helpers
# ──────────────────────────────────────────────────────────────────────────


class TestCoerceTokenIds:
    def test_int(self):
        assert _coerce_token_ids(42) == [42]

    def test_list(self):
        assert _coerce_token_ids([1, 2, 3]) == [1, 2, 3]

    def test_numpy_1d(self):
        assert _coerce_token_ids(np.array([4, 5])) == [4, 5]

    def test_numpy_2d_batched(self):
        assert _coerce_token_ids(np.array([[7, 8, 9]])) == [7, 8, 9]

    def test_numpy_0d_scalar(self):
        assert _coerce_token_ids(np.int64(11)) == [11]

    def test_unsupported_raises(self):
        with pytest.raises(TypeError):
            _coerce_token_ids("not-tokens")


class TestLastTokenIdHelper:
    def test_1d_list(self):
        assert _last_token_id([1, 2, 3]) == 3

    def test_1d_numpy(self):
        assert _last_token_id(np.array([4, 5])) == 5

    def test_2d_batched_numpy(self):
        assert _last_token_id(np.array([[1, 2, 3]])) == 3

    def test_empty_returns_none(self):
        assert _last_token_id([]) is None


# ──────────────────────────────────────────────────────────────────────────
# AutoregressiveStreamingRunner — Task 1 acceptance tests
# ──────────────────────────────────────────────────────────────────────────


class TestRunnerEmitsChunks:
    def test_emits_n_streaming_chunks(self):
        # 3 tokens → 3 chunks. The cumulative-decode pattern
        # produces one piece per token boundary because each piece
        # is whole-character.
        runner, _, _ = _make_runner(
            emit_ids=[1, 2, 3],
            id_to_piece={1: "hel", 2: "lo ", 3: "world"},
        )
        chunks = _drive(runner)
        assert len(chunks) == 3
        # Joined-text invariant.
        joined = "".join(c.text_delta for c in chunks)
        assert joined == "hello world"

    def test_sequence_index_strictly_increasing(self):
        runner, _, _ = _make_runner(
            emit_ids=[1, 2, 3, 4],
            id_to_piece={1: "a", 2: "b", 3: "c", 4: "d"},
        )
        chunks = _drive(runner)
        indices = [c.sequence_index for c in chunks]
        assert indices == sorted(indices)
        assert indices == list(range(len(indices)))

    def test_terminal_chunk_carries_aggregate_fields(self):
        runner, _, _ = _make_runner(
            emit_ids=[1, 2],
            id_to_piece={1: "ok ", 2: "done"},
        )
        chunks = _drive(runner)
        terminal = chunks[-1]
        # Terminal aggregate fields populated.
        assert terminal.finish_reason in {"stop", "max_tokens"}
        assert terminal.full_output_text == "ok done"
        assert terminal.duration_seconds is not None
        assert terminal.duration_seconds >= 0.0
        assert terminal.tee_attestation == b"\x01" * 32
        assert terminal.tee_type == TEEType.SOFTWARE
        assert terminal.epsilon_spent == 0.0
        # Non-terminal chunks have aggregate fields None.
        for c in chunks[:-1]:
            assert c.finish_reason is None
            assert c.full_output_text is None
            assert c.duration_seconds is None
            assert c.tee_attestation is None


class TestMaxTokensCap:
    def test_runner_respects_max_tokens_cap(self):
        # Model would emit 10 tokens, cap is 3 — only 3 reach the
        # adapter, terminal chunk has finish_reason="max_tokens".
        runner, mdl, _ = _make_runner(
            emit_ids=list(range(1, 11)),
            id_to_piece={i: f"t{i} " for i in range(1, 11)},
            sampling=SamplingDefaults(max_tokens=3),
        )
        chunks = _drive(runner)
        assert len(chunks) == 3
        assert chunks[-1].finish_reason == "max_tokens"
        assert mdl.last_call["max_new_tokens"] == 3


class TestEosTriggersStop:
    def test_eos_token_triggers_finish_reason_stop(self):
        # Model emits [1, 2, EOS=99]; runner reports stop.
        runner, _, _ = _make_runner(
            emit_ids=[1, 2, 99],
            id_to_piece={1: "hi ", 2: "there"},
            eos_token_id=99,
        )
        chunks = _drive(runner)
        assert chunks[-1].finish_reason == "stop"
        # EOS itself decodes to empty (special-token skip), so
        # full_output_text excludes it.
        assert chunks[-1].full_output_text == "hi there"


class TestMidGenerateException:
    def test_mid_generate_exception_yields_terminal_error(self):
        # Model raises after 2 tokens. Runner yields a single
        # terminal error chunk (per the implementation: partial
        # text rides as full_output_text on the terminal chunk;
        # text_delta is empty).
        runner, _, _ = _make_runner(
            emit_ids=[1, 2, 3, 4],
            id_to_piece={1: "aa ", 2: "bb ", 3: "cc ", 4: "dd"},
            raise_after=2,
            raise_exc=RuntimeError("boom"),
        )
        chunks = _drive(runner)
        # Implementation yields exactly ONE terminal chunk on
        # exception (even if pieces accumulated before the crash).
        terminal = chunks[-1]
        assert terminal.finish_reason == "error"
        assert terminal.text_delta == ""
        # Partial output is preserved on the terminal chunk.
        assert terminal.full_output_text == "aa bb "


class TestNonTailDispatch:
    def test_non_tail_dispatch_yields_terminal_error_chunk(self):
        # Tail-only contract: is_final_stage=False → exactly one
        # terminal chunk with finish_reason="error", no text.
        runner, mdl, _ = _make_runner(
            emit_ids=[1, 2, 3],
            id_to_piece={1: "a", 2: "b", 3: "c"},
        )
        chunks = _drive(runner, is_final_stage=False)
        assert len(chunks) == 1
        assert chunks[0].finish_reason == "error"
        assert chunks[0].text_delta == ""
        assert chunks[0].full_output_text == ""
        # Model.generate must NOT have been called.
        assert mdl.last_call == {}


class TestRunnerIsIterable:
    def test_runner_returns_iterable_generator(self):
        runner, _, _ = _make_runner(
            emit_ids=[1],
            id_to_piece={1: "x"},
        )
        gen = runner.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 4),
            activation=np.zeros((1, 4), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=True,
        )
        # Generator shape: has __iter__ + __next__.
        assert hasattr(gen, "__iter__")
        assert hasattr(gen, "__next__")
        chunks = list(gen)
        assert len(chunks) >= 1


class TestEmptyEmission:
    def test_immediate_eos_yields_single_terminal_chunk(self):
        # Model emits EOS as the first (and only) token; nothing
        # decodes to text. Runner yields one terminal chunk with
        # empty text and stop reason.
        runner, _, _ = _make_runner(
            emit_ids=[99],
            id_to_piece={},
            eos_token_id=99,
        )
        chunks = _drive(runner)
        assert len(chunks) == 1
        assert chunks[0].finish_reason == "stop"
        assert chunks[0].text_delta == ""
        assert chunks[0].full_output_text == ""


class TestRequestOverridesDefaults:
    def test_request_max_tokens_overrides_default(self):
        class _Req:
            max_tokens = 2
            temperature = None

        runner, mdl, _ = _make_runner(
            emit_ids=[1, 2, 3, 4, 5],
            id_to_piece={i: f"t{i}" for i in range(1, 6)},
            sampling=SamplingDefaults(max_tokens=100),
        )
        chunks = _drive(runner, request=_Req())
        assert mdl.last_call["max_new_tokens"] == 2
        assert len(chunks) == 2

    def test_temperature_zero_triggers_greedy(self):
        class _Req:
            max_tokens = None
            temperature = 0.0

        runner, mdl, _ = _make_runner(
            emit_ids=[1],
            id_to_piece={1: "x"},
        )
        _drive(runner, request=_Req())
        assert mdl.last_call["do_sample"] is False
        assert mdl.last_call["temperature"] == 0.0


class TestProtocolStructural:
    def test_runner_satisfies_streaming_layer_runner_protocol(self):
        # Structural-typing check: the runner exposes the
        # ``run_layer_slice_streaming`` method with the right
        # call signature. Protocol membership is duck-typed; the
        # only functional check is that the call returns an
        # iterator of StreamingChunk.
        runner, _, _ = _make_runner(
            emit_ids=[1],
            id_to_piece={1: "x"},
        )
        out = runner.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 1),
            activation=np.zeros((1, 1), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=True,
        )
        first = next(iter(out))
        assert isinstance(first, StreamingChunk)


class TestConstructorValidation:
    def test_rejects_model_without_generate(self):
        with pytest.raises(RuntimeError, match="model"):
            AutoregressiveStreamingRunner(
                model=object(),
                tokenizer=_FakeTokenizer(
                    id_to_piece={}, prompt_ids=[],
                ),
                tee_attestation=b"\x00",
                tee_type=TEEType.SOFTWARE,
                prompt_provider=_prompt_provider,
            )

    def test_rejects_tokenizer_without_encode_decode(self):
        class _Mdl:
            def generate(self, **kw):
                return []

        with pytest.raises(RuntimeError, match="tokenizer"):
            AutoregressiveStreamingRunner(
                model=_Mdl(),
                tokenizer=object(),
                tee_attestation=b"\x00",
                tee_type=TEEType.SOFTWARE,
                prompt_provider=_prompt_provider,
            )

    def test_rejects_non_bytes_attestation(self):
        class _Mdl:
            def generate(self, **kw):
                return []

        with pytest.raises(RuntimeError, match="tee_attestation"):
            AutoregressiveStreamingRunner(
                model=_Mdl(),
                tokenizer=_FakeTokenizer(
                    id_to_piece={}, prompt_ids=[],
                ),
                tee_attestation="not-bytes",  # type: ignore[arg-type]
                tee_type=TEEType.SOFTWARE,
                prompt_provider=_prompt_provider,
            )

    def test_rejects_non_callable_prompt_provider(self):
        class _Mdl:
            def generate(self, **kw):
                return []

        with pytest.raises(RuntimeError, match="prompt_provider"):
            AutoregressiveStreamingRunner(
                model=_Mdl(),
                tokenizer=_FakeTokenizer(
                    id_to_piece={}, prompt_ids=[],
                ),
                tee_attestation=b"\x00",
                tee_type=TEEType.SOFTWARE,
                prompt_provider=None,  # type: ignore[arg-type]
            )
