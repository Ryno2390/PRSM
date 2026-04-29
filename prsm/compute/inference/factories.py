"""Phase 3.x.10.x — Production factories for streaming runners.

Public SDK entry points for wiring an ``AutoregressiveStreamingRunner``
into a ``LayerStageServer`` at operator node startup. Combined with
``prsm.compute.chain_rpc.make_layer_stage_server(streaming_runner=...)``
(extended in Phase 3.x.10.x), these close the M5 dormant-scaffolding
gap from Phase 3.x.10's round-1 review — operators can now opt into
real autoregressive streaming with a single factory call.

Why a factory at all:
  - The runner's constructor takes ~6 kwargs and raises on shape
    misuse. The factory codifies the typical wiring + provides
    operator-friendly validation messages.
  - Mirrors the ``prsm.compute.chain_rpc.factories`` pattern that
    already exists for the unary-server path.

What stays the operator's responsibility:
  - Loading the HF model + tokenizer (typically ``transformers.AutoModelForCausalLM``
    + ``transformers.AutoTokenizer``).
  - Sourcing TEE attestation bytes from their TEE runtime
    (``SoftwareTEERuntime`` for dev; ``SgxTEERuntime`` /
    ``SevSnpTEERuntime`` for production).
  - Wiring a ``prompt_provider`` that resolves the prompt text from
    the request envelope (typically a server-side request-id-keyed
    cache the executor populates before dispatch).

See ``docs/2026-04-28-phase3.x.10.x-production-wiring-design-plan.md``
§3.4 for the design rationale and §6 Task 4 acceptance.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np

from prsm.compute.inference.autoregressive_runner import (
    AutoregressiveStreamingRunner,
    SamplingDefaults,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType


__all__ = [
    "make_autoregressive_streaming_runner",
]


def make_autoregressive_streaming_runner(
    *,
    model: Any,
    tokenizer: Any,
    tee_attestation: bytes,
    prompt_provider: Callable[
        [Tuple[int, int], np.ndarray, PrivacyLevel], str
    ],
    sampling_defaults: Optional[SamplingDefaults] = None,
    tee_type: TEEType = TEEType.SOFTWARE,
) -> AutoregressiveStreamingRunner:
    """Build an ``AutoregressiveStreamingRunner`` for production use.

    Required:
      model              HuggingFace-shaped model handle (typically
                         ``transformers.AutoModelForCausalLM``).
                         Must expose ``.generate(input_ids=...,
                         streamer=..., max_new_tokens=...,
                         temperature=..., do_sample=..., top_k=...,
                         top_p=..., eos_token_id=...)``.
      tokenizer          HuggingFace AutoTokenizer-shaped object
                         exposing ``.encode``, ``.decode``, and
                         ``.eos_token_id``.
      tee_attestation    Bytes the stage signs over (TEE-bound
                         identity). Operator-sourced from the
                         platform's TEE runtime.
      prompt_provider    Callable
                         ``(layer_range, activation, privacy_tier) -> str``.
                         Production wires this to a server-side
                         request-id-keyed registry the executor
                         populates before calling
                         ``handle_token_stream``.

    Optional:
      sampling_defaults  Per-runner defaults (max_tokens=512,
                         temperature=1.0, top_k=50, top_p=0.95).
                         Request-level fields override these on
                         each call.
      tee_type           Defaults to ``TEEType.SOFTWARE`` for dev.
                         Production sets ``TEEType.SGX`` /
                         ``TEEType.SEV_SNP`` per platform.

    Operator misconfig surfaces as ``RuntimeError`` with a clear
    message at construction time — matches the runner's existing
    validation pattern. NEVER raises at dispatch time.
    """
    if model is None:
        raise RuntimeError(
            "make_autoregressive_streaming_runner: model is required "
            "(typically transformers.AutoModelForCausalLM)"
        )
    if tokenizer is None:
        raise RuntimeError(
            "make_autoregressive_streaming_runner: tokenizer is required "
            "(typically transformers.AutoTokenizer)"
        )
    if not isinstance(tee_attestation, (bytes, bytearray)):
        raise RuntimeError(
            "make_autoregressive_streaming_runner: tee_attestation must "
            "be bytes (operator-sourced from TEE runtime)"
        )
    if prompt_provider is None or not callable(prompt_provider):
        raise RuntimeError(
            "make_autoregressive_streaming_runner: prompt_provider must "
            "be callable (layer_range, activation, privacy_tier) -> str"
        )
    return AutoregressiveStreamingRunner(
        model=model,
        tokenizer=tokenizer,
        tee_attestation=bytes(tee_attestation),
        tee_type=tee_type,
        sampling_defaults=sampling_defaults,
        prompt_provider=prompt_provider,
    )
