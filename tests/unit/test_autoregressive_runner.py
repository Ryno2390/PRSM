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


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.10 Task 2 — Multi-byte UTF-8 handling
#
# Byte-level BPE tokenizers split codepoints across token boundaries.
# A 4-byte emoji like 🎉 (F0 9F 8E 89) may span 2-3 BPE tokens; a
# 3-byte CJK char like 中 (E4 B8 AD) may span 2. The
# ``_HFStreamerAdapter`` cumulative-decode + U+FFFD-suffix detection
# must hold the buffer across these boundaries so that
# ``"".join(text_deltas)`` ALWAYS forms valid UTF-8.
#
# The fake tokenizer below maps token id → raw bytes and uses Python's
# ``bytes.decode("utf-8", errors="replace")`` semantics — which match
# HF's byte-level tokenizers in collapsing incomplete trailing
# sequences into a single ``"�"`` replacement char.
# ──────────────────────────────────────────────────────────────────────────


class _BPEFakeTokenizer:
    """Byte-level BPE-shaped tokenizer. ``decode(ids,
    skip_special_tokens=True)`` concatenates per-token byte payloads
    and decodes via UTF-8 with errors="replace" — incomplete
    trailing multi-byte sequences become a single ``"\\ufffd"``
    suffix, which is the signal the adapter buffers on.
    """

    def __init__(
        self,
        *,
        id_to_bytes: Dict[int, bytes],
        prompt_ids: List[int],
        eos_token_id: Optional[int] = None,
    ) -> None:
        self._id_to_bytes = id_to_bytes
        self._prompt_ids = prompt_ids
        self.eos_token_id = eos_token_id
        self._special: set = set()
        if eos_token_id is not None:
            self._special.add(eos_token_id)

    def encode(self, text: str) -> List[int]:
        return list(self._prompt_ids)

    def decode(
        self, ids: List[int], skip_special_tokens: bool = True,
    ) -> str:
        joined = b"".join(
            self._id_to_bytes.get(int(i), b"")
            for i in ids
            if not (skip_special_tokens and i in self._special)
        )
        return joined.decode("utf-8", errors="replace")


def _make_bpe_runner(
    *,
    emit_ids: List[int],
    id_to_bytes: Dict[int, bytes],
    eos_token_id: Optional[int] = None,
) -> Tuple[AutoregressiveStreamingRunner, _FakeModel, _BPEFakeTokenizer]:
    tok = _BPEFakeTokenizer(
        id_to_bytes=id_to_bytes,
        prompt_ids=[100, 101],
        eos_token_id=eos_token_id,
    )
    mdl = _FakeModel(emit_ids=emit_ids)
    runner = AutoregressiveStreamingRunner(
        model=mdl,
        tokenizer=tok,
        tee_attestation=b"\x01" * 32,
        tee_type=TEEType.SOFTWARE,
        sampling_defaults=SamplingDefaults(max_tokens=32),
        prompt_provider=_prompt_provider,
    )
    return runner, mdl, tok


class TestMultiByteUtf8:
    def test_ascii_passes_through_unchanged(self):
        # Each token is a whole ASCII char — decode never produces
        # U+FFFD, so every token boundary flushes immediately.
        runner, _, _ = _make_bpe_runner(
            emit_ids=[1, 2, 3, 4, 5],
            id_to_bytes={
                1: b"h", 2: b"i", 3: b"!", 4: b" ", 5: b"o",
            },
        )
        chunks = _drive(runner)
        joined = "".join(c.text_delta for c in chunks)
        assert joined == "hi! o"
        # UTF-8 round-trip invariant.
        assert joined.encode("utf-8").decode("utf-8") == joined

    def test_emoji_split_across_two_tokens_buffers_until_complete(self):
        # 🎉 = F0 9F 8E 89 (4 bytes). Split as [F0 9F] [8E 89].
        # After token 1: cumulative decode ends in U+FFFD → adapter
        # holds buffer, emits NOTHING. After token 2: full emoji,
        # adapter emits "🎉" as a single piece.
        emoji = "🎉"
        b = emoji.encode("utf-8")
        assert len(b) == 4
        runner, _, _ = _make_bpe_runner(
            emit_ids=[1, 2],
            id_to_bytes={1: b[:2], 2: b[2:]},
        )
        chunks = _drive(runner)
        # Exactly ONE non-empty text_delta carrying the whole emoji.
        non_empty = [c for c in chunks if c.text_delta]
        assert len(non_empty) == 1
        assert non_empty[0].text_delta == "🎉"
        # Joined output is valid UTF-8.
        joined = "".join(c.text_delta for c in chunks)
        assert joined == "🎉"
        joined.encode("utf-8").decode("utf-8")  # would raise on invalid

    def test_emoji_split_across_three_tokens_buffers_until_complete(self):
        # 🎉 = F0 9F 8E 89, split as [F0] [9F 8E] [89]. All three
        # intermediate cumulative decodes must end in U+FFFD →
        # nothing emits until the final token completes the codepoint.
        b = "🎉".encode("utf-8")
        runner, _, _ = _make_bpe_runner(
            emit_ids=[1, 2, 3],
            id_to_bytes={1: b[:1], 2: b[1:3], 3: b[3:]},
        )
        chunks = _drive(runner)
        non_empty = [c for c in chunks if c.text_delta]
        assert len(non_empty) == 1
        assert non_empty[0].text_delta == "🎉"

    def test_cjk_split_across_two_tokens_buffers_until_complete(self):
        # 中 = E4 B8 AD (3 bytes). Split as [E4 B8] [AD].
        b = "中".encode("utf-8")
        assert len(b) == 3
        runner, _, _ = _make_bpe_runner(
            emit_ids=[1, 2],
            id_to_bytes={1: b[:2], 2: b[2:]},
        )
        chunks = _drive(runner)
        non_empty = [c for c in chunks if c.text_delta]
        assert len(non_empty) == 1
        assert non_empty[0].text_delta == "中"

    def test_mixed_ascii_emoji_cjk_sequence_joined_invariant(self):
        # "ab🎉中c" — interleaved, emoji + CJK split across token
        # boundaries. The joined-deltas-form-valid-UTF-8 invariant
        # MUST hold for every intermediate state too.
        emoji_b = "🎉".encode("utf-8")  # 4 bytes
        cjk_b = "中".encode("utf-8")  # 3 bytes
        runner, _, _ = _make_bpe_runner(
            emit_ids=[1, 2, 3, 4, 5, 6, 7],
            id_to_bytes={
                1: b"a",
                2: b"b",
                3: emoji_b[:2],   # incomplete emoji prefix
                4: emoji_b[2:],   # completes emoji
                5: cjk_b[:1],     # incomplete cjk prefix
                6: cjk_b[1:],     # completes cjk
                7: b"c",
            },
        )
        chunks = _drive(runner)
        joined = "".join(c.text_delta for c in chunks)
        assert joined == "ab🎉中c"
        # Strict UTF-8 round-trip — would raise if any delta
        # contained bare lone surrogates / invalid sequences.
        assert joined.encode("utf-8").decode("utf-8") == joined

    def test_partial_buffer_at_end_of_stream_flushed_via_end(self):
        # Stream truncates mid-multi-byte: only [E4 B8] of 中 emits,
        # then generate() returns. The runner's defensive
        # ``adapter.end()`` flushes the buffered partial as U+FFFD.
        # The point: end() DOES emit something rather than swallowing
        # the buffer, AND the emitted text is still valid UTF-8.
        b = "中".encode("utf-8")
        runner, _, _ = _make_bpe_runner(
            emit_ids=[1],
            id_to_bytes={1: b[:2]},  # truncated mid-codepoint
        )
        chunks = _drive(runner)
        joined = "".join(c.text_delta for c in chunks)
        # Replacement char emitted at end-of-stream rather than
        # silent data loss.
        assert "�" in joined
        # Still valid UTF-8.
        assert joined.encode("utf-8").decode("utf-8") == joined

    def test_replacement_char_detection_holds_buffer_correctly(self):
        # Direct unit test on the adapter: drive put() with
        # incomplete bytes, assert NO callback fires; then complete
        # the codepoint, assert ONE callback fires with the whole
        # piece. Exercises the U+FFFD-suffix branch of
        # ``_maybe_flush`` in isolation.
        from prsm.compute.inference.autoregressive_runner import (
            _HFStreamerAdapter,
        )

        captured: List[Tuple[str, int]] = []

        def on_text(piece: str, tid: int) -> None:
            captured.append((piece, tid))

        b = "🎉".encode("utf-8")
        tok = _BPEFakeTokenizer(
            id_to_bytes={1: b[:2], 2: b[2:]},
            prompt_ids=[],
        )
        adapter = _HFStreamerAdapter(tokenizer=tok, on_text=on_text)
        adapter.put(1)
        # After the incomplete-prefix put, NO emission.
        assert captured == []
        adapter.put(2)
        # After the completing put, ONE emission with the whole emoji.
        assert len(captured) == 1
        assert captured[0][0] == "🎉"
        assert captured[0][1] == 2

    def test_zero_width_joiner_family_emoji_sequence(self):
        # 👨‍👩‍👧 = man + ZWJ + woman + ZWJ + girl = 5 codepoints,
        # 18 UTF-8 bytes. Split across many tokens — every
        # intermediate state must be valid UTF-8 (or buffered).
        family = "👨‍👩‍👧"
        b = family.encode("utf-8")
        # Split into 6 ~3-byte chunks deliberately not aligned with
        # codepoint boundaries.
        chunks_b = [b[i:i + 3] for i in range(0, len(b), 3)]
        emit_ids = list(range(1, len(chunks_b) + 1))
        id_to_bytes = dict(zip(emit_ids, chunks_b))
        runner, _, _ = _make_bpe_runner(
            emit_ids=emit_ids, id_to_bytes=id_to_bytes,
        )
        chunks = _drive(runner)
        joined = "".join(c.text_delta for c in chunks)
        assert joined == family
        assert joined.encode("utf-8").decode("utf-8") == joined


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.10 Task 3 — Sampling parameters + stop conditions
#
# Verifies the request → model wiring of max_tokens / temperature /
# top_k / top_p / do_sample / eos_token_id, AND the stop-condition
# mapping to ``finish_reason ∈ {"stop", "max_tokens", "error"}``.
# Greedy determinism is tested via mock-rerun equivalence; real-HF
# determinism with a torch seed is exercised in Task 5 E2E.
# ──────────────────────────────────────────────────────────────────────────


class TestSamplingAndStopConditions:
    def test_temperature_zero_produces_deterministic_greedy_output(self):
        # Drive runner twice with temperature=0. With the mock
        # model the per-call output is trivially deterministic, but
        # the test point is that the runner consistently passes
        # ``do_sample=False`` so HF's own greedy path is engaged
        # — that's what produces real determinism upstream.
        class _Req:
            max_tokens = None
            temperature = 0.0

        out1_calls = []
        out2_calls = []
        for sink in (out1_calls, out2_calls):
            runner, mdl, _ = _make_runner(
                emit_ids=[1, 2, 3],
                id_to_piece={1: "a", 2: "b", 3: "c"},
            )
            chunks = _drive(runner, request=_Req())
            sink.append((
                "".join(c.text_delta for c in chunks),
                mdl.last_call["do_sample"],
                mdl.last_call["temperature"],
            ))
        # Same output text both runs, do_sample=False both runs.
        assert out1_calls[0] == out2_calls[0]
        assert out1_calls[0][1] is False
        assert out1_calls[0][2] == 0.0

    def test_temperature_positive_passes_sampling_params(self):
        # temperature>0 → do_sample=True, and the SamplingDefaults'
        # top_k=50 / top_p=0.95 reach the model unmodified.
        class _Req:
            max_tokens = None
            temperature = 0.7

        runner, mdl, _ = _make_runner(
            emit_ids=[1],
            id_to_piece={1: "x"},
            sampling=SamplingDefaults(
                max_tokens=8, temperature=1.0, top_k=50, top_p=0.95,
            ),
        )
        _drive(runner, request=_Req())
        assert mdl.last_call["do_sample"] is True
        assert mdl.last_call["temperature"] == 0.7
        assert mdl.last_call["top_k"] == 50
        assert mdl.last_call["top_p"] == 0.95

    def test_max_tokens_two_caps_generation_at_two(self):
        class _Req:
            max_tokens = 2
            temperature = None

        runner, mdl, _ = _make_runner(
            emit_ids=[1, 2, 3, 4, 5],
            id_to_piece={i: f"t{i} " for i in range(1, 6)},
        )
        chunks = _drive(runner, request=_Req())
        # Only 2 chunks reach the wire; cap propagated to model.
        assert len(chunks) == 2
        assert mdl.last_call["max_new_tokens"] == 2

    def test_finish_reason_stop_when_eos_reached(self):
        # Model emits a non-EOS token then EOS. Runner reports stop.
        runner, _, _ = _make_runner(
            emit_ids=[1, 99],
            id_to_piece={1: "hello"},
            eos_token_id=99,
        )
        chunks = _drive(runner)
        assert chunks[-1].finish_reason == "stop"

    def test_finish_reason_max_tokens_when_cap_hit(self):
        # Model would emit 5 tokens, none EOS. Cap=3 → terminal
        # finish_reason="max_tokens" (NOT "stop").
        runner, _, _ = _make_runner(
            emit_ids=[1, 2, 3, 4, 5],
            id_to_piece={i: f"w{i} " for i in range(1, 6)},
            sampling=SamplingDefaults(max_tokens=3),
        )
        chunks = _drive(runner)
        assert chunks[-1].finish_reason == "max_tokens"

    def test_request_temperature_overrides_runner_default(self):
        # Runner default temperature=1.0; request specifies 0.3 →
        # 0.3 wins.
        class _Req:
            max_tokens = None
            temperature = 0.3

        runner, mdl, _ = _make_runner(
            emit_ids=[1],
            id_to_piece={1: "x"},
            sampling=SamplingDefaults(max_tokens=8, temperature=1.0),
        )
        _drive(runner, request=_Req())
        assert mdl.last_call["temperature"] == 0.3

    def test_eos_token_id_wired_into_generate(self):
        # tokenizer.eos_token_id MUST reach model.generate() so HF
        # can stop early. Confirms the runner doesn't drop it.
        runner, mdl, _ = _make_runner(
            emit_ids=[1],
            id_to_piece={1: "x"},
            eos_token_id=42,
        )
        _drive(runner)
        assert mdl.last_call["eos_token_id"] == 42

    def test_default_max_tokens_used_when_request_none(self):
        # request=None path falls back to SamplingDefaults.max_tokens.
        runner, mdl, _ = _make_runner(
            emit_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            id_to_piece={i: "x" for i in range(1, 11)},
            sampling=SamplingDefaults(max_tokens=4),
        )
        chunks = _drive(runner, request=None)
        assert mdl.last_call["max_new_tokens"] == 4
        assert len(chunks) == 4

    def test_no_eos_token_id_passes_none(self):
        # Tokenizer without eos_token_id → runner passes None
        # through. HF's generate() handles this (no early stop).
        runner, mdl, _ = _make_runner(
            emit_ids=[1, 2],
            id_to_piece={1: "a", 2: "b"},
            eos_token_id=None,
        )
        _drive(runner)
        assert mdl.last_call["eos_token_id"] is None
        # Without EOS, terminal reason on natural exhaustion of the
        # mock's emit list: finish_reason="max_tokens" (cap hit
        # because all 2 tokens used; mock's loop ended at i=2 < cap).
        # Actually with cap=16 and only 2 emit_ids, generate()
        # returns having NOT triggered EOS — finish_reason maps to
        # "max_tokens" per impl (not-EOS branch).


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.10 Task 4 — Tail-only contract enforcement
#
# Sharded autoregressive decode is deferred to Phase 3.x.11. The
# v1 runner enforces tail-only by yielding exactly one terminal
# error chunk on non-tail dispatch, without calling model.generate
# or prompt_provider. The server maps this finish_reason="error"
# chunk to ``StageErrorCode.INTERNAL_ERROR`` via the existing
# Phase 3.x.8 Task 2 token-stream handler.
# ──────────────────────────────────────────────────────────────────────────


class _SpyPromptProvider:
    """Records every call so tests can assert non-tail dispatch
    short-circuits BEFORE prompt resolution."""

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(
        self,
        layer_range: Tuple[int, int],
        activation: np.ndarray,
        privacy_tier: PrivacyLevel,
    ) -> str:
        self.call_count += 1
        return "spy-prompt"


def _make_runner_with_spy(
    *,
    spy: _SpyPromptProvider,
    emit_ids: Optional[List[int]] = None,
) -> Tuple[AutoregressiveStreamingRunner, _FakeModel]:
    tok = _FakeTokenizer(
        id_to_piece={1: "a", 2: "b", 3: "c"}, prompt_ids=[100, 101],
    )
    mdl = _FakeModel(emit_ids=emit_ids or [1, 2, 3])
    runner = AutoregressiveStreamingRunner(
        model=mdl,
        tokenizer=tok,
        tee_attestation=b"\x02" * 32,
        tee_type=TEEType.SOFTWARE,
        sampling_defaults=SamplingDefaults(max_tokens=8),
        prompt_provider=spy,
    )
    return runner, mdl


class TestTailOnlyContract:
    def test_non_tail_yields_exactly_one_chunk(self):
        spy = _SpyPromptProvider()
        runner, _ = _make_runner_with_spy(spy=spy)
        chunks = _drive(runner, is_final_stage=False)
        assert len(chunks) == 1

    def test_non_tail_terminal_chunk_is_error_with_empty_text(self):
        spy = _SpyPromptProvider()
        runner, _ = _make_runner_with_spy(spy=spy)
        chunks = _drive(runner, is_final_stage=False)
        only = chunks[0]
        assert only.finish_reason == "error"
        assert only.text_delta == ""
        assert only.full_output_text == ""
        assert only.token_id is None
        # Sequence index of a terminal-only chunk is 0.
        assert only.sequence_index == 0

    def test_non_tail_yields_no_preceding_non_error_chunks(self):
        # Defensive against a future bug where a partial decode
        # leaks before the error chunk: assert NO chunk before the
        # terminal one carries non-error finish_reason or non-empty
        # text_delta.
        spy = _SpyPromptProvider()
        runner, _ = _make_runner_with_spy(spy=spy)
        chunks = _drive(runner, is_final_stage=False)
        for c in chunks[:-1]:
            assert c.finish_reason in (None, "error")
            assert c.text_delta == ""

    def test_non_tail_does_not_call_model_generate(self):
        spy = _SpyPromptProvider()
        runner, mdl = _make_runner_with_spy(spy=spy)
        _drive(runner, is_final_stage=False)
        # last_call empty — generate() never invoked.
        assert mdl.last_call == {}

    def test_non_tail_does_not_call_prompt_provider(self):
        # The prompt_provider may be expensive (DB lookup, MCP
        # roundtrip). Non-tail dispatch MUST short-circuit before
        # invoking it.
        spy = _SpyPromptProvider()
        runner, _ = _make_runner_with_spy(spy=spy)
        _drive(runner, is_final_stage=False)
        assert spy.call_count == 0

    def test_non_tail_chunk_carries_runner_attestation(self):
        # The terminal error chunk MUST still carry the runner's
        # tee_attestation + tee_type so the server's
        # handle_token_stream can build the StageError frame
        # without falling back to a default-attestation path.
        spy = _SpyPromptProvider()
        runner, _ = _make_runner_with_spy(spy=spy)
        chunks = _drive(runner, is_final_stage=False)
        only = chunks[0]
        assert only.tee_attestation == b"\x02" * 32
        assert only.tee_type == TEEType.SOFTWARE
        assert only.epsilon_spent == 0.0

    def test_tail_dispatch_unchanged_after_non_tail_check(self):
        # Same runner, two dispatches in sequence: non-tail first
        # (must error cleanly), tail second (must produce real
        # output). Verifies the non-tail short-circuit doesn't
        # corrupt runner state or the spy's reusability.
        spy = _SpyPromptProvider()
        runner, mdl = _make_runner_with_spy(
            spy=spy, emit_ids=[1, 2, 3],
        )
        non_tail = _drive(runner, is_final_stage=False)
        assert len(non_tail) == 1 and non_tail[0].finish_reason == "error"
        assert mdl.last_call == {}
        tail = _drive(runner, is_final_stage=True)
        # Tail dispatch produces real chunks, with prompt_provider
        # called exactly once during the tail run (NOT during the
        # non-tail run).
        assert len(tail) == 3
        assert spy.call_count == 1
        assert mdl.last_call != {}
        # Joined output reconstructs full text.
        joined = "".join(c.text_delta for c in tail)
        assert joined == "abc"

    def test_docstring_documents_phase_3_x_11_deferral(self):
        # Acceptance criterion from the design plan: "Sharded
        # autoregressive deferred to Phase 3.x.11 is documented in
        # the runner's docstring." Encode that as a real assertion
        # so future doc edits can't silently drop the deferral note.
        cls_doc = (AutoregressiveStreamingRunner.__doc__ or "").lower()
        assert "phase 3.x.11" in cls_doc
        # Tail-only contract is also a stated invariant.
        assert "tail-only" in cls_doc
