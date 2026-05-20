"""Sprint 641 — single source of truth for token sampling.

Sprint 639 introduced greedy/temperature/top-k sampling in
`prsm node infer`. Sprint 640 added seed-replay verification in
`receipt_verify`. Both implementations duplicated the same
softmax-with-top-K-mask pipeline. Any drift between them silently
INVALIDATES audits — a verifier whose math is 1 ULP different
from the sampler's would always report TOKEN_ID_ARGMAX_MISMATCH
even for honest receipts.

Sprint 641 consolidates: both call `sample_token_from_logits(...)`.
The receipt records the canonical `sampling_mode` string; the
verifier parses it back into kwargs and calls the same function.
Drift becomes impossible by construction.
"""
from __future__ import annotations

from typing import Optional


def sample_token_from_logits(
    last_logits,
    *,
    temperature: Optional[float],
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    step: int = 0,
) -> int:
    """Sample one token id from a 1-D logits vector.

    Args:
      last_logits: numpy ndarray, shape (vocab,) or any broadcast-
        compatible 1-D view. Caller passes the LAST-position
        logits (the sampling position).
      temperature: None or 0.0 → greedy argmax. Positive float →
        sampling temperature. Internally clamped at 1e-8 floor so
        very small T values are safe (zero would divide-by-zero
        in scaled = logits/T).
      top_k: None or 0 → no masking. Positive int → keep only the
        top-K logits before softmax. Effectively a sparsified
        distribution. Ignored when temperature is None/0 (greedy
        doesn't need a distribution).
      seed: None → fresh numpy default_rng (non-deterministic
        across runs). Int → seeded `default_rng(seed + step)`.
        The step offset means each token draws from a different
        point in the seed-derived stream when called in a loop.
      step: Step index in the generation loop; ignored when seed
        is None.

    Returns:
      int token id sampled from the distribution (or argmax in
      greedy mode).

    Determinism contract:
      Given the same (last_logits, temperature, top_k, seed, step)
      tuple, this function produces the same output bit-for-bit.
      This is what makes receipt-replay verification possible.
      Any future change to the math here MUST consider all
      downstream audit files signed against the previous version
      — operators verifying old receipts on new daemon code would
      see false TOKEN_ID_ARGMAX_MISMATCH findings.
    """
    import numpy as np

    last_logits = last_logits.astype(np.float32)
    if temperature is None or temperature == 0.0:
        return int(last_logits.argmax())

    scaled = last_logits / max(float(temperature), 1e-8)
    if top_k is not None and top_k > 0:
        k = min(int(top_k), scaled.shape[-1])
        top_indices = np.argpartition(scaled, -k)[-k:]
        mask = np.full_like(scaled, -np.inf)
        mask[top_indices] = scaled[top_indices]
        scaled = mask
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / probs.sum()
    if seed is not None:
        rng = np.random.default_rng(int(seed) + int(step))
    else:
        rng = np.random.default_rng()
    return int(rng.choice(probs.shape[-1], p=probs))


def format_sampling_mode(
    *,
    temperature: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
) -> str:
    """Canonical sampling_mode string written to receipts."""
    if temperature is None:
        return "greedy"
    parts = [f"temperature:{float(temperature):.3f}"]
    if top_k is not None and top_k > 0:
        parts.append(f"top_k:{int(top_k)}")
    if seed is not None:
        parts.append(f"seed:{int(seed)}")
    return ",".join(parts)
