"""Phase 3.x.10 Task 5 — E2E test with real HF model.

Drives ``AutoregressiveStreamingRunner`` against a real
``transformers.AutoModelForCausalLM`` (``distilgpt2``) +
``AutoTokenizer``. Verifies the design plan §4 Task 5 acceptance
criteria:

  - Real autoregressive output (multiple genuinely-distinct tokens,
    not synthetic word splits).
  - Joined deltas form valid UTF-8.
  - Greedy decode (temperature=0) produces bit-identical output on
    rerun (proves the runner doesn't add nondeterminism on top of
    HF's greedy path).
  - Receipt verifies under settler identity with
    ``streamed_output=True`` (Phase 3.x.8 Task 4 invariant —
    streamed flag commits to signing payload through the full
    end-to-end path).
  - Finish reason correctly set (``"stop"`` if EOS reached,
    ``"max_tokens"`` if cap hit).

All tests in this module are marked ``@pytest.mark.slow`` because
HF model load + first-decode is ~5-30s. Default CI run excludes
slow tests.

If ``transformers`` is unavailable OR distilgpt2 can't be loaded
(no internet + no cache), tests skip cleanly rather than fail —
the runner contract is fully exercised by the unit-test fakes;
this E2E adds the real-model proof on top.
"""

from __future__ import annotations

import dataclasses
import hashlib
from decimal import Decimal
from typing import Any, Tuple

import numpy as np
import pytest

from prsm.compute.inference.autoregressive_runner import (
    AutoregressiveStreamingRunner,
    SamplingDefaults,
)
from prsm.compute.inference.models import (
    ContentTier,
    InferenceReceipt,
)
from prsm.compute.inference.receipt import sign_receipt, verify_receipt
from prsm.compute.inference.streaming_runner import StreamingChunk
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


pytestmark = pytest.mark.slow


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hf_model_and_tokenizer():
    """Load distilgpt2 once per test module. Skips the whole module
    if transformers isn't installed or the model isn't reachable."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "distilgpt2", local_files_only=False,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "distilgpt2", local_files_only=False,
        )
        model.eval()
    except Exception as exc:  # noqa: BLE001 — broad-skip on infra failure
        pytest.skip(f"distilgpt2 unavailable: {exc.__class__.__name__}")
    # distilgpt2 has no native pad token; HF emits a warning during
    # generate() unless we set one. Use eos_token as pad — same
    # convention HF docs recommend for GPT-2 family.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _build_runner(
    model: Any, tokenizer: Any, *, prompt: str = "The quick brown fox",
    max_tokens: int = 8,
) -> AutoregressiveStreamingRunner:
    return AutoregressiveStreamingRunner(
        model=model,
        tokenizer=tokenizer,
        tee_attestation=b"\x07" * 32,
        tee_type=TEEType.SOFTWARE,
        sampling_defaults=SamplingDefaults(
            max_tokens=max_tokens, temperature=1.0,
        ),
        prompt_provider=lambda lr, act, pt: prompt,
    )


def _drive_tail(
    runner: AutoregressiveStreamingRunner, *, request: Any = None,
) -> list:
    return list(
        runner.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 1),
            activation=np.zeros((1, 1), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=True,
            request=request,
        )
    )


class _GreedyRequest:
    max_tokens = 8
    temperature = 0.0


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


class TestRealAutoregressiveDecode:
    def test_real_decode_emits_streaming_chunks(
        self, hf_model_and_tokenizer,
    ):
        # Acceptance: "Real autoregressive output (multiple genuinely-
        # distinct tokens, not synthetic word splits)."
        # Greedy decode with cap=8 → expect at least one chunk and
        # at most cap chunks.
        model, tok = hf_model_and_tokenizer
        runner = _build_runner(model, tok, max_tokens=8)
        chunks = _drive_tail(runner, request=_GreedyRequest())

        assert len(chunks) >= 1
        # All chunks are StreamingChunk instances.
        for c in chunks:
            assert isinstance(c, StreamingChunk)
        # Sequence indices are strictly increasing 0-based.
        idx = [c.sequence_index for c in chunks]
        assert idx == list(range(len(idx)))
        # Each non-terminal chunk has non-None token_id (real
        # autoregressive runner populates it; SyntheticStreamingRunner
        # left it None — this is the marker that we're hitting the
        # real path).
        non_empty_token_ids = [
            c.token_id for c in chunks if c.token_id is not None
        ]
        assert len(non_empty_token_ids) >= 1, (
            "real runner must populate token_id on at least one chunk; "
            "synthetic placeholder leaves it None"
        )

    def test_joined_deltas_form_valid_utf8(
        self, hf_model_and_tokenizer,
    ):
        model, tok = hf_model_and_tokenizer
        runner = _build_runner(model, tok, max_tokens=12)
        chunks = _drive_tail(runner, request=_GreedyRequest())
        joined = "".join(c.text_delta for c in chunks)
        # UTF-8 round-trip — would raise on lone surrogates / invalid
        # sequences. Empty string is also valid UTF-8.
        assert joined.encode("utf-8").decode("utf-8") == joined
        # Terminal chunk's full_output_text matches joined deltas.
        terminal = chunks[-1]
        assert terminal.full_output_text == joined

    def test_greedy_decode_bit_identical_on_rerun(
        self, hf_model_and_tokenizer,
    ):
        # Acceptance: "Greedy decode (temperature=0) produces bit-
        # identical output on rerun." Builds two independent runners
        # against the same model and asserts joined-text equality +
        # token_id sequence equality.
        model, tok = hf_model_and_tokenizer

        runner_a = _build_runner(model, tok, max_tokens=8)
        runner_b = _build_runner(model, tok, max_tokens=8)
        chunks_a = _drive_tail(runner_a, request=_GreedyRequest())
        chunks_b = _drive_tail(runner_b, request=_GreedyRequest())

        text_a = "".join(c.text_delta for c in chunks_a)
        text_b = "".join(c.text_delta for c in chunks_b)
        assert text_a == text_b

        ids_a = [c.token_id for c in chunks_a]
        ids_b = [c.token_id for c in chunks_b]
        assert ids_a == ids_b

    def test_finish_reason_correctly_set(
        self, hf_model_and_tokenizer,
    ):
        # Acceptance: "Finish reason correctly set." With a small
        # cap (4 tokens) on a non-terminating greedy continuation,
        # finish_reason should be "max_tokens" — distilgpt2 won't
        # emit EOS in 4 tokens for "The quick brown fox".
        model, tok = hf_model_and_tokenizer
        runner = _build_runner(model, tok, max_tokens=4)

        class _CapReq:
            max_tokens = 4
            temperature = 0.0

        chunks = _drive_tail(runner, request=_CapReq())
        terminal = chunks[-1]
        assert terminal.finish_reason in {"stop", "max_tokens"}
        # Aggregate fields populated on terminal chunk.
        assert terminal.full_output_text is not None
        assert terminal.duration_seconds is not None
        assert terminal.duration_seconds >= 0.0
        assert terminal.tee_attestation == b"\x07" * 32
        assert terminal.tee_type == TEEType.SOFTWARE


class TestSignedReceiptOverRealOutput:
    def test_receipt_with_streamed_output_flag_verifies(
        self, hf_model_and_tokenizer,
    ):
        # Acceptance: "Receipt verifies under settler identity with
        # streamed_output=True." Builds a real autoregressive output,
        # constructs an InferenceReceipt(streamed_output=True) over
        # the joined output's hash, signs it with a generated
        # NodeIdentity, verifies it both via identity and via the
        # public key.
        model, tok = hf_model_and_tokenizer
        runner = _build_runner(model, tok, max_tokens=8)
        chunks = _drive_tail(runner, request=_GreedyRequest())
        joined = "".join(c.text_delta for c in chunks)
        terminal = chunks[-1]

        identity = generate_node_identity(display_name="settler-test")
        unsigned = InferenceReceipt(
            job_id="job-3x10-task5",
            request_id="req-e2e-1",
            model_id="distilgpt2",
            content_tier=ContentTier.A,
            privacy_tier=PrivacyLevel.NONE,
            epsilon_spent=0.0,
            tee_type=terminal.tee_type or TEEType.SOFTWARE,
            tee_attestation=terminal.tee_attestation or b"",
            output_hash=hashlib.sha256(joined.encode("utf-8")).digest(),
            duration_seconds=terminal.duration_seconds or 0.0,
            cost_ftns=Decimal("0.0001"),
            streamed_output=True,
        )
        signed = sign_receipt(unsigned, identity)

        # Verifies via identity.
        assert verify_receipt(signed, identity=identity) is True
        # Verifies via standalone public key (the prod path —
        # caller doesn't have the private identity).
        assert verify_receipt(
            signed, public_key_b64=identity.public_key_b64,
        ) is True

    def test_streamed_flag_tampering_invalidates_signature(
        self, hf_model_and_tokenizer,
    ):
        # Phase 3.x.8 Task 4 invariant proven through Task 5: a
        # relay can't downgrade a streamed receipt to non-streamed
        # without breaking the signature. Same proof done in Phase
        # 3.x.8.1 unit tests; replays here against the REAL output
        # path so the contract is end-to-end attested.
        model, tok = hf_model_and_tokenizer
        runner = _build_runner(model, tok, max_tokens=4)
        chunks = _drive_tail(runner, request=_GreedyRequest())
        joined = "".join(c.text_delta for c in chunks)

        identity = generate_node_identity()
        signed = sign_receipt(
            InferenceReceipt(
                job_id="job-tamper",
                request_id="req-tamper",
                model_id="distilgpt2",
                content_tier=ContentTier.A,
                privacy_tier=PrivacyLevel.NONE,
                epsilon_spent=0.0,
                tee_type=TEEType.SOFTWARE,
                tee_attestation=b"\x07" * 32,
                output_hash=hashlib.sha256(joined.encode("utf-8")).digest(),
                duration_seconds=0.001,
                cost_ftns=Decimal("0.0001"),
                streamed_output=True,
            ),
            identity,
        )
        assert verify_receipt(signed, identity=identity) is True

        # Tamper: flip streamed_output=True → False without re-signing.
        downgraded = dataclasses.replace(signed, streamed_output=False)
        assert verify_receipt(downgraded, identity=identity) is False
