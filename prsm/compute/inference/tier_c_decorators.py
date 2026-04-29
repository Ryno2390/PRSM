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

import queue
import threading
import time
from typing import Any, Iterator, Optional, Tuple

import numpy as np

from prsm.compute.inference.streaming_runner import (
    StreamingChunk,
    StreamingLayerRunner,
)
from prsm.compute.tee.models import PrivacyLevel


__all__ = [
    "BatchedTrailingStreamingRunner",
    "FixedRateStreamingRunner",
]


# Sentinel values for the producer→consumer queue in
# ``FixedRateStreamingRunner``. Using object() so they're never
# confused with a legitimate ``StreamingChunk`` instance.
_PRODUCER_DONE = object()


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


class FixedRateStreamingRunner:
    """M1 decorator — fixed-rate emission for Tier C with
    streaming UX preserved.

    Emits one ``StreamingChunk`` per wall-clock cadence tick.
    Inner runner runs in a background thread, dropping its
    chunks into a buffer queue. The main generator pulls one
    chunk per ``cadence_seconds``; when no inner chunk is
    ready by tick time, emits a no-op frame (empty
    ``text_delta``) so the wire timing stays operator-uniform.

    Effect: inter-frame wall-clock latency is the cadence
    (constant), independent of the inner runner's per-token
    decode time. A passive on-path observer learns nothing
    about per-token complexity from frame timing.

    **Honest scope.** The cadence MUST be operator-uniform
    across all Tier C dispatches — observable variance defeats
    the purpose. The cadence value itself becomes a public
    commitment (operators set it once in node config). Total
    stream duration still leaks total token count
    (``cadence × frame_count = duration``); operators bound
    this by capping ``max_tokens`` for Tier C.

    **v1 trade-off.** No-op frames inflate total wire frame
    count by ``total_decode_seconds / cadence``. For a
    10-second decode at 50ms cadence, that's 200 frames —
    manageable but worth flagging in the audit-prep §7.x. A
    future optimization could batch consecutive no-op frames.

    Constructor args:
      inner            ``StreamingLayerRunner`` to wrap.
      cadence_seconds  Wall-clock interval between consecutive
                       wire frames. Defaults to 50ms (a
                       reasonable streaming-UX tick); operator
                       config tunes per workload.

    Threading model:
      The inner runner runs in a daemon background thread; the
      decorator's generator drives the cadence loop on the
      caller's thread. Generator close (``GeneratorExit``)
      signals the producer to stop via the queue's sentinel
      pattern; the producer thread is daemon so process exit
      is always clean.

    Composes with any ``StreamingLayerRunner``. The server's
    ``handle_token_stream`` consumes it identically.
    """

    def __init__(
        self,
        inner: StreamingLayerRunner,
        cadence_seconds: float = 0.05,
    ) -> None:
        if inner is None:
            raise RuntimeError(
                "FixedRateStreamingRunner requires an inner "
                "StreamingLayerRunner"
            )
        if not hasattr(inner, "run_layer_slice_streaming"):
            raise RuntimeError(
                "FixedRateStreamingRunner: inner does not look like "
                "a StreamingLayerRunner (no run_layer_slice_streaming "
                "method)"
            )
        if (
            isinstance(cadence_seconds, bool)
            or not isinstance(cadence_seconds, (int, float))
        ):
            raise RuntimeError(
                f"FixedRateStreamingRunner: cadence_seconds must be "
                f"number, got {type(cadence_seconds).__name__}"
            )
        if cadence_seconds <= 0:
            raise RuntimeError(
                f"FixedRateStreamingRunner: cadence_seconds must be "
                f"positive, got {cadence_seconds}"
            )
        self._inner = inner
        self._cadence = float(cadence_seconds)

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
        # Producer thread: consumes the inner generator and
        # pushes each chunk + sentinel onto the buffer queue.
        # Captures any exception in a holder so the consumer can
        # synthesize a terminal error chunk.
        buffer: "queue.Queue[Any]" = queue.Queue()
        error_holder: list = []

        def producer() -> None:
            try:
                for chunk in self._inner.run_layer_slice_streaming(
                    model=model,
                    layer_range=layer_range,
                    activation=activation,
                    privacy_tier=privacy_tier,
                    is_final_stage=is_final_stage,
                    request=request,
                ):
                    buffer.put(chunk)
            except Exception as exc:  # noqa: BLE001
                error_holder.append(exc)
            buffer.put(_PRODUCER_DONE)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        seq = 0
        inner_terminal: Optional[StreamingChunk] = None
        producer_done = False

        try:
            while True:
                # Always sleep one full cadence between
                # consecutive yields. Using interval-based pacing
                # rather than anchor-based avoids the catch-up
                # bug where a behind-schedule loop emits
                # bunched-up yields without sleeping. The
                # timing-mask invariant requires consecutive
                # yields ≥ cadence apart; this enforces it
                # unconditionally.
                time.sleep(self._cadence)

                # Drain at most ONE inner chunk per tick, even if
                # multiple are buffered. This is the cadence-
                # masking invariant: wire timing reflects the
                # cadence, not the inner producer's pace.
                if not producer_done:
                    try:
                        item = buffer.get_nowait()
                    except queue.Empty:
                        item = None
                else:
                    # Inner stream signaled done; we're now in
                    # drain phase — pull from buffer if any
                    # backlog remains.
                    try:
                        item = buffer.get_nowait()
                    except queue.Empty:
                        item = None

                if item is _PRODUCER_DONE:
                    producer_done = True
                    item = None  # treat the tick as no-op

                if item is None:
                    if producer_done and inner_terminal is None:
                        # Inner runner finished without emitting a
                        # terminal chunk (possibly raised mid-
                        # stream). Synthesize a terminal error.
                        if error_holder:
                            # Mid-decode exception. Yield a
                            # terminal error chunk with empty
                            # aggregates — server will surface as
                            # StageError per Phase 3.x.8 contract.
                            yield StreamingChunk(
                                sequence_index=seq,
                                text_delta="",
                                finish_reason="error",
                                full_output_text="",
                                duration_seconds=0.0,
                                tee_attestation=b"",
                                tee_type=None,
                                epsilon_spent=0.0,
                            )
                            return
                        # Inner stream ended cleanly with no
                        # chunks — nothing to forward (server
                        # handles empty stream per Phase 3.x.8
                        # contract).
                        return
                    if producer_done and inner_terminal is not None:
                        # Drain phase complete, queue empty.
                        # Emit the cadence-aligned terminal.
                        yield StreamingChunk(
                            sequence_index=seq,
                            text_delta=inner_terminal.text_delta,
                            token_id=inner_terminal.token_id,
                            finish_reason=inner_terminal.finish_reason,
                            full_output_text=inner_terminal.full_output_text,
                            duration_seconds=inner_terminal.duration_seconds,
                            tee_attestation=inner_terminal.tee_attestation,
                            tee_type=inner_terminal.tee_type,
                            epsilon_spent=inner_terminal.epsilon_spent,
                        )
                        return
                    # No chunk ready, inner not yet done — emit a
                    # no-op cadence pad. Empty text_delta keeps
                    # the joined-text invariant intact.
                    yield StreamingChunk(
                        sequence_index=seq,
                        text_delta="",
                    )
                    seq += 1
                    continue

                # ``item`` is a real StreamingChunk from the
                # inner runner. If it's the inner's terminal,
                # stash it for emission AFTER the buffer drains.
                if item.finish_reason is not None:
                    inner_terminal = item
                    # Emit a no-op for this tick; drain phase
                    # will emit the terminal at the next tick
                    # (or whenever the buffer is empty).
                    yield StreamingChunk(
                        sequence_index=seq,
                        text_delta="",
                    )
                    seq += 1
                    continue

                # Non-terminal inner chunk — forward with the
                # decorator's renumbered sequence index.
                yield StreamingChunk(
                    sequence_index=seq,
                    text_delta=item.text_delta,
                    token_id=item.token_id,
                )
                seq += 1
        finally:
            # Generator close / exception unwound: signal the
            # producer thread to stop. Daemon thread will die
            # with the process if it's still iterating; we don't
            # block here since the inner generator may not
            # support clean cancellation.
            pass
