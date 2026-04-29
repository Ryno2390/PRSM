"""Phase 3.x.11.q — Tier C constant-time sharded-decode executors.

Chain-level decorators wrapping ``RpcChainExecutor`` (or any object
exposing ``execute_chain_streaming``) that mask the per-token wire
timing surface characterized in the Phase 3.x.11 threat-model
addendum §3.1. Operators wire one of these via a routing-layer
gate (Phase 3.x.11.q Task 4) to enable Tier C streaming for
sharded autoregressive decode; without a decorator, Tier C
sharded streaming continues to be structurally denied at the
per-stage ``_dispatch_sharded`` boundary.

Two decorators in this slice (mirrors Phase 3.x.10.y for the
single-host path):

  - **M2 — BatchedTrailingShardedExecutor** (this module).
    Drains the inner executor's full stream, then emits ONE
    terminal ``StreamToken`` carrying the joined text, followed
    by the ``ChainExecutionResult``. Zero per-token timing
    observable on the executor → caller wire; sacrifices
    streaming UX for Tier C operators choosing maximum leak
    elimination.

  - **M1 — FixedRateShardedExecutor** (this module, Task 2).
    Yields per-token ``StreamToken``s at a fixed wall-clock
    cadence; if the inner chain produces faster than cadence,
    the decorator sleeps until the next tick. Preserves
    streaming UX while masking per-token latency at the cost of
    total stream duration leaking total token count.

**Honest scope (carries forward from §3.6 of the threat-model
addendum, Phase 3.x.11.q's §3.7 amendment).** These chain-level
decorators mask the executor → caller wire only. The PER-STAGE
wire (executor → each chain stage) still emits per-token
dispatches at the chain's native rate; an adversary with
visibility into a single stage's transport learns the raw
per-token cadence. Operators wanting full-network masking
compose with per-stage cadence wrappers — Phase 3.x.11.q.x
deferred.

Both decorators implement only ``execute_chain_streaming`` (the
streaming surface is what Tier C routing targets). The synchronous
``execute_chain`` is intentionally omitted; Tier C non-streaming
continues to be denied at the per-stage ``_dispatch_sharded``
TIER_GATE deny.
"""

from __future__ import annotations

import time as _time_module
from typing import Any, Callable, Iterator, Optional, Union

from prsm.compute.chain_rpc.client import StreamToken
from prsm.compute.inference.parallax_executor import ChainExecutionResult


__all__ = [
    "BatchedTrailingShardedExecutor",
    "FixedRateShardedExecutor",
]


class BatchedTrailingShardedExecutor:
    """M2 decorator — single-frame emission for Tier C sharded
    streaming.

    Consumes the inner executor's full stream (blocking until the
    terminal ``ChainExecutionResult`` arrives), then emits ONE
    ``StreamToken`` carrying the joined ``text_delta``, followed
    by the ``ChainExecutionResult`` unchanged. From a wire observer
    on the executor → caller path, exactly two events appear: one
    StreamToken, one ChainExecutionResult — regardless of how many
    tokens the inner chain produced or at what per-token cadence.

    The decorator implements ``execute_chain_streaming`` with the
    same shape as ``RpcChainExecutor.execute_chain_streaming``, so
    the existing HTTP SSE endpoint + MCP plumbing pass it through
    unchanged.

    Empty inner stream (no StreamTokens) — only the
    ChainExecutionResult is forwarded. If the inner stream is also
    empty (no result), nothing is emitted; downstream consumers
    handle the empty-iterator case per their existing contract.

    Note: per Phase 3.x.11.q design plan §3.4, total response size
    still leaks the total joined-text length to a byte counter.
    Operator-configurable padding to a fixed max-length is
    deferred to Phase 3.x.11.q.x.
    """

    def __init__(self, inner: Any) -> None:
        if inner is None:
            raise RuntimeError(
                "BatchedTrailingShardedExecutor requires an inner "
                "executor"
            )
        if not hasattr(inner, "execute_chain_streaming"):
            raise RuntimeError(
                "BatchedTrailingShardedExecutor: inner does not "
                "expose execute_chain_streaming(...) — wrap "
                "RpcChainExecutor or another chain-streaming "
                "executor"
            )
        self._inner = inner

    def execute_chain_streaming(
        self,
        *,
        request: Any,
        chain: Any,
    ) -> Iterator[Union[StreamToken, ChainExecutionResult]]:
        tokens: list = []
        result: Optional[ChainExecutionResult] = None
        for event in self._inner.execute_chain_streaming(
            request=request, chain=chain,
        ):
            if isinstance(event, StreamToken):
                if result is not None:
                    # Round-1 review M1 remediation: tokens emitted
                    # by the inner AFTER the terminal result violate
                    # the streaming contract (the result is supposed
                    # to be the LAST event). Drop them rather than
                    # silently merging into the joined text — that
                    # would re-order content across the terminal
                    # boundary.
                    continue
                tokens.append(event)
            elif isinstance(event, ChainExecutionResult):
                # Eagerly emit the joined token + result on receipt
                # of the terminal. Stop draining the inner — any
                # post-terminal events are protocol violations.
                result = event
                break
            else:
                # Unknown event type — pass through to preserve
                # forward-compat (a future executor variant adds
                # a new event type).
                yield event
        if tokens:
            # Round-1 review M2 remediation: defensive str coerce —
            # if upstream ever ships a non-str text_delta (the
            # StreamToken dataclass types it str, but
            # RpcChainExecutor doesn't enforce at runtime), the
            # join would TypeError mid-generator and the whole
            # Tier C request would crash hard. Coerce defensively.
            joined = "".join(
                str(t.text_delta) if t.text_delta is not None else ""
                for t in tokens
            )
            last = tokens[-1]
            yield StreamToken(
                sequence_index=0,
                text_delta=joined,
                token_id=last.token_id,
                finish_reason=last.finish_reason,
            )
        if result is not None:
            yield result


class FixedRateShardedExecutor:
    """M1 decorator — cadence-driven yield for Tier C sharded
    streaming.

    Each ``StreamToken`` from the inner executor is held until at
    least ``cadence_seconds`` have elapsed since the previous
    yield, then forwarded. ``ChainExecutionResult`` is forwarded
    immediately on receipt — the terminal event isn't part of the
    per-token timing surface.

    The chain runs at native speed; the decorator's ``yield`` is
    what gates emission. From a wire observer on the executor →
    caller path, inter-StreamToken intervals are clamped to ≥
    ``cadence_seconds`` regardless of per-token chain compute
    variance.

    **Cadence calibration.** Cadence MUST be ≥ the chain's native
    per-token latency, otherwise the decorator just yields
    immediately every tick and provides no masking. Operators set
    cadence based on measured chain native rate + buffer (recommended
    starting point: 2× measured native rate). See Phase 3.x.11.q
    audit-prep §7.13 for calibration guidance.

    **Honest scope (per design plan §3.4).** Total stream duration
    leaks total token count under M1 (cadence × frame count =
    duration). Operators mitigate by capping ``max_tokens`` per
    Tier C request; the leak ceiling becomes
    ``max_tokens × cadence``.

    Clock + sleep are injectable for test determinism (mocking).
    Production wires the defaults (``time.monotonic`` + ``time.sleep``).
    """

    def __init__(
        self,
        inner: Any,
        cadence_seconds: float,
        *,
        clock: Callable[[], float] = _time_module.monotonic,
        sleep: Callable[[float], None] = _time_module.sleep,
    ) -> None:
        if inner is None:
            raise RuntimeError(
                "FixedRateShardedExecutor requires an inner executor"
            )
        if not hasattr(inner, "execute_chain_streaming"):
            raise RuntimeError(
                "FixedRateShardedExecutor: inner does not expose "
                "execute_chain_streaming(...)"
            )
        if (
            not isinstance(cadence_seconds, (int, float))
            or cadence_seconds <= 0.0
            or isinstance(cadence_seconds, bool)
        ):
            raise ValueError(
                f"cadence_seconds must be positive float, got "
                f"{cadence_seconds!r}"
            )
        if not callable(clock):
            raise RuntimeError(
                "FixedRateShardedExecutor: clock must be callable"
            )
        if not callable(sleep):
            raise RuntimeError(
                "FixedRateShardedExecutor: sleep must be callable"
            )
        self._inner = inner
        self._cadence = float(cadence_seconds)
        self._clock = clock
        self._sleep = sleep

    def execute_chain_streaming(
        self,
        *,
        request: Any,
        chain: Any,
    ) -> Iterator[Union[StreamToken, ChainExecutionResult]]:
        last_emit: Optional[float] = None
        for event in self._inner.execute_chain_streaming(
            request=request, chain=chain,
        ):
            if isinstance(event, StreamToken):
                if last_emit is not None:
                    target = last_emit + self._cadence
                    now = self._clock()
                    if now < target:
                        self._sleep(target - now)
                yield event
                last_emit = self._clock()
            else:
                # ChainExecutionResult or unknown event — forward
                # immediately. The terminal isn't part of the
                # per-token timing surface; gating it would just
                # delay the receipt without masking anything.
                yield event
