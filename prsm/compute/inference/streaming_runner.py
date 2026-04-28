"""Phase 3.x.8 Task 2 — StreamingLayerRunner Protocol + synthetic v1 runner.

Sibling to the unary ``LayerSliceRunner`` Protocol in
``prsm.compute.chain_rpc.server``. The streaming runner emits the
tail stage's output incrementally as ``StreamingChunk`` objects; the
``LayerStageServer.handle_token_stream`` path encodes each chunk
into a wire-format ``TokenFrame`` and finalizes the stream with one
``StreamFinalFrame`` that carries the signed
``RunLayerSliceResponse`` over the joined output bytes.

PRSM v1 ships a ``SyntheticStreamingRunner`` adapter that wraps an
existing one-shot ``LayerSliceRunner``: it runs the full forward
pass + decodes the resulting activation to text via an injected
decoder, then chunks the joined text into N synthetic
``StreamingChunk``s. This keeps the streaming wire path + executor
generator API + MCP integration testable end-to-end TODAY, while
the real autoregressive-decode runner is a follow-up replacement
that satisfies the same Protocol — no public-surface change
downstream.

Token-boundary policy for the synthetic runner is intentionally
boring: split on whitespace, preserving the trailing space so
joining with ``"".join(deltas)`` produces the original text.
Production autoregressive runners emit one ``StreamingChunk`` per
real token (or per N tokens to amortize wire overhead) — the
Protocol is the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Protocol, Tuple

import numpy as np

from prsm.compute.chain_rpc.server import LayerSliceRunner
from prsm.compute.tee.models import PrivacyLevel, TEEType


__all__ = [
    "StreamingChunk",
    "StreamingLayerRunner",
    "SyntheticStreamingRunner",
    "split_text_into_deltas",
]


# ──────────────────────────────────────────────────────────────────────────
# StreamingChunk
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StreamingChunk:
    """Per-chunk output yielded by a ``StreamingLayerRunner``.

    Multiple chunks per stream, ordered strictly by ``sequence_index``
    (0-indexed, no gaps). The terminal chunk has ``finish_reason``
    set AND populates the trailing aggregate fields
    (``full_output_text`` + timing/attestation/epsilon) — the
    server uses those to build the signed
    ``RunLayerSliceResponse`` that becomes the
    ``StreamFinalFrame``'s payload.

    Non-terminal chunks leave the aggregate fields as ``None``; the
    server's stream handler only reads them on the terminal chunk.

    Token-id is optional: real autoregressive runners populate it,
    the synthetic runner leaves it ``None`` because it has no
    vocab mapping.
    """

    sequence_index: int
    text_delta: str
    token_id: Optional[int] = None
    finish_reason: Optional[str] = None  # set on the LAST chunk
    # Final-aggregate fields — populated ONLY on the chunk where
    # finish_reason is non-None. Used by the server to build the
    # signed RunLayerSliceResponse for the StreamFinalFrame.
    full_output_text: Optional[str] = None
    duration_seconds: Optional[float] = None
    tee_attestation: Optional[bytes] = None
    tee_type: Optional[TEEType] = None
    epsilon_spent: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────────
# StreamingLayerRunner Protocol
# ──────────────────────────────────────────────────────────────────────────


class StreamingLayerRunner(Protocol):
    """Tail-stage streaming runner. Emits the layer slice's output
    incrementally as a generator of ``StreamingChunk``s.

    The runner is invoked ONLY on the chain's tail stage with
    ``is_final_stage=True`` — the server's
    ``handle_token_stream`` validates this BEFORE invoking the
    runner. Non-tail stages never see a streaming dispatch.

    The terminal chunk MUST set ``finish_reason`` and populate
    ``full_output_text`` + ``duration_seconds`` + ``tee_attestation``
    + ``tee_type`` + ``epsilon_spent``. The server uses those to
    build the signed ``RunLayerSliceResponse`` whose
    ``activation_blob`` is the UTF-8-encoded joined output text.

    Errors propagate. The server wraps the iterator in a
    try/except and maps unexpected exceptions to a terminal
    ``StageError`` frame.
    """

    def run_layer_slice_streaming(
        self,
        *,
        model: Any,
        layer_range: Tuple[int, int],
        activation: np.ndarray,
        privacy_tier: PrivacyLevel,
        is_final_stage: bool,
    ) -> Iterator[StreamingChunk]:
        ...


# ──────────────────────────────────────────────────────────────────────────
# Text-splitting helper (synthetic chunking strategy)
# ──────────────────────────────────────────────────────────────────────────


_WHITESPACE_SPLIT = re.compile(r"(\s+)")


def split_text_into_deltas(text: str) -> List[str]:
    """Split ``text`` into chunks suitable for synthetic streaming.

    Strategy: split on whitespace runs while keeping the whitespace
    as its own delta. ``"".join(split_text_into_deltas(text)) ==
    text`` is invariant — joining the deltas reproduces the
    original text exactly. This is the joined-text invariant the
    StreamFinalFrame's signed ``activation_blob`` commits to.

    Empty input returns a single empty delta so the stream still
    has at least one frame for the terminal chunk to ride on.
    """
    if not text:
        return [""]
    parts = _WHITESPACE_SPLIT.split(text)
    # re.split with a captured group keeps separators as elements.
    # Filter out empty strings that arise at start/end when the
    # original text starts/ends with whitespace, but preserve them
    # if they're the only content.
    parts = [p for p in parts if p != ""]
    return parts if parts else [""]


# ──────────────────────────────────────────────────────────────────────────
# SyntheticStreamingRunner
# ──────────────────────────────────────────────────────────────────────────


class SyntheticStreamingRunner:
    """v1 placeholder ``StreamingLayerRunner``. Wraps an existing
    one-shot ``LayerSliceRunner``: runs the full forward pass,
    decodes the resulting activation to text via the injected
    ``output_decoder`` callable, chunks the text into synthetic
    ``StreamingChunk``s.

    This is NOT real autoregressive decode — it's the streaming
    wire path's smoke-test scaffolding. It exercises the full
    end-to-end protocol surface (server frame emission, executor
    iterator forwarding, MCP progress events, receipt finalization)
    so when the autoregressive decode runner replaces it, the
    public surface is already battle-tested.

    Constructor args:
      runner            ``LayerSliceRunner`` that runs the
                        underlying forward pass.
      output_decoder    Activation → text decoder. Production
                        wraps the model's tokenizer + de-embedding
                        layer; tests inject a deterministic fake.
      splitter          Optional override for the text-splitting
                        strategy. Defaults to whitespace-split via
                        ``split_text_into_deltas``.
    """

    def __init__(
        self,
        *,
        runner: LayerSliceRunner,
        output_decoder: Callable[[np.ndarray], str],
        splitter: Callable[[str], List[str]] = split_text_into_deltas,
    ) -> None:
        if runner is None or not hasattr(runner, "run_layer_range"):
            raise RuntimeError(
                "SyntheticStreamingRunner requires a LayerSliceRunner with "
                ".run_layer_range(...)"
            )
        if output_decoder is None or not callable(output_decoder):
            raise RuntimeError(
                "SyntheticStreamingRunner requires a callable output_decoder"
            )
        if splitter is None or not callable(splitter):
            raise RuntimeError(
                "SyntheticStreamingRunner: splitter must be callable"
            )
        self._runner = runner
        self._output_decoder = output_decoder
        self._splitter = splitter

    def run_layer_slice_streaming(
        self,
        *,
        model: Any,
        layer_range: Tuple[int, int],
        activation: np.ndarray,
        privacy_tier: PrivacyLevel,
        is_final_stage: bool,
    ) -> Iterator[StreamingChunk]:
        # Run the underlying one-shot forward pass + decode.
        result = self._runner.run_layer_range(
            model=model,
            layer_range=layer_range,
            activation=activation,
            privacy_tier=privacy_tier,
            is_final_stage=is_final_stage,
        )
        text = self._output_decoder(result.output)
        if not isinstance(text, str):
            raise TypeError(
                f"output_decoder must return str, got {type(text).__name__}"
            )

        deltas = self._splitter(text)
        if not deltas:
            # Defensive: splitter should always return at least one
            # delta (per split_text_into_deltas contract). If a
            # custom splitter violates that, force a single empty
            # delta so the stream still has a terminal frame.
            deltas = [""]

        last_idx = len(deltas) - 1
        for i, delta in enumerate(deltas):
            if i < last_idx:
                yield StreamingChunk(
                    sequence_index=i,
                    text_delta=delta,
                )
            else:
                # Terminal chunk: set finish_reason + populate the
                # final-aggregate fields the server reads to build
                # the signed RunLayerSliceResponse.
                yield StreamingChunk(
                    sequence_index=i,
                    text_delta=delta,
                    finish_reason="stop",
                    full_output_text=text,
                    duration_seconds=result.duration_seconds,
                    tee_attestation=result.tee_attestation,
                    tee_type=result.tee_type,
                    epsilon_spent=result.epsilon_spent,
                )
