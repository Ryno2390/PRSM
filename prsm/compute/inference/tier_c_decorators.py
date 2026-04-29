"""Phase 3.x.10.y — Tier C constant-time padding decorators.

``StreamingLayerRunner``-Protocol-compatible wrappers that
mask the per-token inter-token-latency side-channel
characterized in the Phase 3.x.10 timing-sidechannel memo §5.
Operators wire one of these via
``make_layer_stage_server(tier_c_streaming_decorator=...)``
to enable streaming for Tier C content; without a decorator,
Tier C streaming requests are rejected at the dispatch layer.

Two candidates from the memo:

  - **M2 — BatchedTrailingStreamingRunner** (this file).
    Consumes the full inner stream and emits ONE terminal
    chunk with the joined text. Zero per-token timing
    observable to a wire observer; sacrifices streaming UX
    for Tier C operators choosing maximum leak elimination.

  - **M1 — FixedRateStreamingRunner** (Task 3).
    Emits chunks at a fixed wall-clock cadence; pads with
    no-op frames between real tokens. Preserves streaming
    UX while masking per-token latency at the cost of total
    stream duration leaking total token count.

Honest scope: leak-masking, not leak-elimination. Even M2
leaks total stream duration (one observation), which under
constant-bandwidth + fixed-overhead generation correlates
weakly with output length. Operators mitigate by capping
``max_tokens`` per Tier C request — leak ceiling becomes
``max_tokens × per-token-decode-time``.

Both decorators conform to ``StreamingLayerRunner`` Protocol
structurally — they compose with any inner runner
(SyntheticStreamingRunner, AutoregressiveStreamingRunner,
future variants). The server's ``handle_token_stream``
consumes them identically.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple

import numpy as np

from prsm.compute.inference.streaming_runner import (
    StreamingChunk,
    StreamingLayerRunner,
)
from prsm.compute.tee.models import PrivacyLevel


__all__ = [
    "BatchedTrailingStreamingRunner",
]


class BatchedTrailingStreamingRunner:
    """M2 decorator — single-frame emission for paranoid Tier C.

    Consumes the inner runner's full stream (blocking until the
    inner terminal chunk arrives), then emits ONE terminal
    ``StreamingChunk`` carrying the joined text and the inner
    terminal's aggregate fields (``finish_reason``,
    ``full_output_text``, ``duration_seconds``,
    ``tee_attestation``, ``tee_type``, ``epsilon_spent``).

    From a wire observer's perspective, only ONE wire frame
    appears. Per-token timing is unobservable — that's the
    point. The trade-off: end users see no streaming UX (output
    appears all at once after the model finishes generating).

    ``StreamingLayerRunner`` Protocol-compatible: composes with
    any inner runner. For ``AutoregressiveStreamingRunner`` the
    end-user visible behavior is identical to the unary
    ``/compute/inference`` path, but the receipt is signed via
    the streaming wire format with ``streamed_output=True`` —
    operators get the streaming-protocol's downgrade-resistance
    invariant (Phase 3.x.8 Task 4) without exposing the timing
    side-channel.

    Empty inner stream (no chunks yielded) → no emission. The
    server's ``handle_token_stream`` interprets this as a
    StageError per the existing Phase 3.x.8 contract.
    """

    def __init__(self, inner: StreamingLayerRunner) -> None:
        if inner is None:
            raise RuntimeError(
                "BatchedTrailingStreamingRunner requires an inner "
                "StreamingLayerRunner"
            )
        if not hasattr(inner, "run_layer_slice_streaming"):
            raise RuntimeError(
                "BatchedTrailingStreamingRunner: inner does not look "
                "like a StreamingLayerRunner (no "
                "run_layer_slice_streaming method)"
            )
        self._inner = inner

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
        # Drain the inner stream fully before emitting anything.
        # Per-token timing of the inner runner becomes
        # unobservable to a wire observer.
        chunks = list(self._inner.run_layer_slice_streaming(
            model=model,
            layer_range=layer_range,
            activation=activation,
            privacy_tier=privacy_tier,
            is_final_stage=is_final_stage,
            request=request,
        ))
        if not chunks:
            # Empty inner stream — yield nothing. Server's
            # handle_token_stream surfaces this as StageError per
            # the existing Phase 3.x.8 contract (no signed
            # receipt material to commit to).
            return

        terminal = chunks[-1]
        # The inner terminal MUST carry aggregate fields per the
        # StreamingLayerRunner Protocol contract (Phase 3.x.8
        # Task 2). If it doesn't, propagate as-is — the server's
        # validation will surface a clean error rather than us
        # silently fabricating fields.
        joined = "".join(c.text_delta for c in chunks)
        yield StreamingChunk(
            sequence_index=0,
            text_delta=joined,
            token_id=None,  # batched output has no single token id
            finish_reason=terminal.finish_reason,
            full_output_text=terminal.full_output_text,
            duration_seconds=terminal.duration_seconds,
            tee_attestation=terminal.tee_attestation,
            tee_type=terminal.tee_type,
            epsilon_spent=terminal.epsilon_spent,
        )
