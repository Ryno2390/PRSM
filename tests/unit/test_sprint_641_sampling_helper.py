"""Sprint 641 — shared sampling helper determinism.

The whole audit-chain story rests on bit-for-bit reproducibility
between `prsm node infer`'s sampling and `verify-receipts`'s replay.
Sprint 641 collapsed both onto a single helper; these tests pin the
determinism contract so future refactors can't drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from prsm.cli_modules.sampling import (
    format_sampling_mode, sample_token_from_logits,
)


def _fixed_logits(vocab: int = 100, seed: int = 7) -> np.ndarray:
    """Deterministic logits for tests so reruns produce same values."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(vocab).astype(np.float32) * 3.0


# --------------------------------------------------------------------------
# Greedy mode
# --------------------------------------------------------------------------


def test_greedy_returns_argmax():
    logits = _fixed_logits()
    expected = int(logits.argmax())
    assert sample_token_from_logits(logits, temperature=None) == expected
    # temperature=0 also routes to greedy (avoid divide-by-zero)
    assert sample_token_from_logits(logits, temperature=0.0) == expected


def test_greedy_ignores_top_k_and_seed():
    """Greedy path doesn't draw from a distribution; top_k + seed
    are irrelevant. Pin so the helper's branching stays correct.
    """
    logits = _fixed_logits()
    expected = int(logits.argmax())
    assert sample_token_from_logits(
        logits, temperature=None, top_k=5, seed=999,
    ) == expected


# --------------------------------------------------------------------------
# Sampled mode — determinism contract
# --------------------------------------------------------------------------


def test_sampled_determinism_same_seed_same_token():
    """Bit-for-bit reproducibility — the audit-chain anchor."""
    logits = _fixed_logits()
    t = 0.7
    s = 42
    step = 3
    a = sample_token_from_logits(
        logits, temperature=t, top_k=20, seed=s, step=step,
    )
    b = sample_token_from_logits(
        logits, temperature=t, top_k=20, seed=s, step=step,
    )
    assert a == b


def test_sampled_different_step_produces_different_distribution():
    """Same seed + different step → different draws (otherwise every
    token in a generation loop would be the same)."""
    logits = _fixed_logits()
    samples = {
        sample_token_from_logits(
            logits, temperature=1.0, seed=42, step=i,
        ) for i in range(10)
    }
    # Not strictly required but with vocab=100 + temp=1.0, getting
    # 10 distinct samples is overwhelmingly likely
    assert len(samples) > 1, (
        "step parameter must influence the draw"
    )


def test_sampled_top_k_restricts_support():
    """All draws must come from the top-K logits — no token outside
    that set should ever be sampled, regardless of step.
    """
    logits = _fixed_logits()
    k = 5
    top_k_set = set(np.argpartition(logits, -k)[-k:].tolist())
    for step in range(50):
        sampled = sample_token_from_logits(
            logits, temperature=1.0, top_k=k, seed=99, step=step,
        )
        assert sampled in top_k_set, (
            f"step={step}: sampled token {sampled} not in top-{k} "
            f"set {sorted(top_k_set)}"
        )


def test_sampled_extreme_low_temperature_concentrates_near_argmax():
    """Very low temperature (T→0+) should approximately concentrate
    on argmax. With T=0.001 + many trials, > 80% should be the
    argmax token.
    """
    logits = _fixed_logits()
    argmax_id = int(logits.argmax())
    hits = 0
    n = 100
    for step in range(n):
        sampled = sample_token_from_logits(
            logits, temperature=0.001, seed=1234, step=step,
        )
        if sampled == argmax_id:
            hits += 1
    assert hits / n > 0.8, f"T→0+ should concentrate; got {hits}/{n}"


# --------------------------------------------------------------------------
# format_sampling_mode canonical string
# --------------------------------------------------------------------------


def test_format_mode_greedy():
    assert format_sampling_mode(
        temperature=None, top_k=None, seed=None,
    ) == "greedy"
    # top_k + seed are ignored when temperature is None (matches the
    # CLI greedy-ignores-everything semantics)
    assert format_sampling_mode(
        temperature=None, top_k=50, seed=42,
    ) == "greedy"


def test_format_mode_temperature_only():
    assert format_sampling_mode(
        temperature=0.7, top_k=None, seed=None,
    ) == "temperature:0.700"


def test_format_mode_full():
    s = format_sampling_mode(
        temperature=1.0, top_k=40, seed=42,
    )
    assert s == "temperature:1.000,top_k:40,seed:42"


def test_format_mode_skips_zero_top_k():
    """top_k=0 means "no masking" → should be omitted from mode string
    so the receipt format stays canonical.
    """
    assert format_sampling_mode(
        temperature=1.0, top_k=0, seed=None,
    ) == "temperature:1.000"


# --------------------------------------------------------------------------
# Integration: round-trip via receipt_verify replay
# --------------------------------------------------------------------------


def test_round_trip_with_verify_replay():
    """Sample a token, build a receipt, verify via verify_chain_invariants
    seed-replay path. Mirrors the real audit flow.
    """
    import base64 as _b64
    from prsm.cli_modules.receipt_verify import verify_chain_invariants

    vocab = 100
    logits_3d = _fixed_logits(vocab=vocab, seed=11).reshape(1, 1, vocab)
    last_logits = logits_3d[0, -1, :]
    sampled = sample_token_from_logits(
        last_logits, temperature=0.8, top_k=30, seed=77, step=0,
    )
    rec = {
        "step": 0,
        "wall_unix": 1.0,
        "request_id": "r0",
        "settler_node_id": "s",
        "model_id": "m",
        "next_token_id": sampled,
        "next_token_text": "x",
        "activation_blob_b64": _b64.b64encode(
            logits_3d.tobytes()
        ).decode("ascii"),
        "activation_shape": [1, 1, vocab],
        "activation_dtype": "float32",
        "sampling_mode": format_sampling_mode(
            temperature=0.8, top_k=30, seed=77,
        ),
    }
    findings = verify_chain_invariants([rec])
    kinds = [f["kind"] for f in findings]
    assert "TOKEN_ID_ARGMAX_MISMATCH" not in kinds, (
        f"replay must accept the helper's own sample; got {findings}"
    )
