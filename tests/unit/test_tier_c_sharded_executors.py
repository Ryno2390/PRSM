"""Phase 3.x.11.q Tasks 1-2 — chain-level Tier C decorator tests.

Covers ``BatchedTrailingShardedExecutor`` (M2) and
``FixedRateShardedExecutor`` (M1) — the chain-level analog of
Phase 3.x.10.y's single-host decorators.

Both decorators wrap any object exposing
``execute_chain_streaming(request=..., chain=...)``. Tests use
recording fakes so the assertions focus on the decorator's own
logic (event ordering, joined text, cadence-driven yield)
without coupling to RpcChainExecutor's full transport surface.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterator, List, Optional, Union

import pytest

from prsm.compute.chain_rpc.client import StreamToken
from prsm.compute.chain_rpc.tier_c_sharded_executors import (
    BatchedTrailingShardedExecutor,
    FixedRateShardedExecutor,
)
from prsm.compute.inference.parallax_executor import ChainExecutionResult
from prsm.compute.tee.models import TEEType


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


class _FakeChainExecutor:
    """Records (request, chain) → emits a scripted event sequence
    in order. Tests script the inner stream content; the decorator
    wraps this to validate its own behavior."""

    def __init__(
        self, events: List[Union[StreamToken, ChainExecutionResult, Any]],
    ) -> None:
        self._events = list(events)
        self.call_log: List[tuple] = []

    def execute_chain_streaming(
        self, *, request: Any, chain: Any,
    ) -> Iterator[Union[StreamToken, ChainExecutionResult]]:
        self.call_log.append((request, chain))
        for ev in self._events:
            yield ev


def _result(output: str = "test-output") -> ChainExecutionResult:
    return ChainExecutionResult(
        output=output,
        duration_seconds=0.001,
        tee_attestation=b"\x00" * 32,
        tee_type=TEEType.SOFTWARE,
        epsilon_spent=0.0,
    )


def _token(
    seq: int, text: str, *,
    token_id: Optional[int] = None,
    finish_reason: Optional[str] = None,
) -> StreamToken:
    return StreamToken(
        sequence_index=seq,
        text_delta=text,
        token_id=token_id,
        finish_reason=finish_reason,
    )


# ──────────────────────────────────────────────────────────────────────────
# Task 1 — BatchedTrailingShardedExecutor (M2)
# ──────────────────────────────────────────────────────────────────────────


class TestBatchedTrailingShardedExecutor:
    def test_protocol_conformance_execute_chain_streaming(self):
        # Decorator exposes execute_chain_streaming(request=, chain=).
        decorator = BatchedTrailingShardedExecutor(
            inner=_FakeChainExecutor([_result()]),
        )
        assert callable(getattr(decorator, "execute_chain_streaming", None))

    def test_happy_path_joins_tokens_into_one_emit(self):
        inner = _FakeChainExecutor([
            _token(0, "Hello"),
            _token(1, " "),
            _token(2, "world", token_id=42, finish_reason="stop"),
            _result(),
        ])
        decorator = BatchedTrailingShardedExecutor(inner=inner)
        events = list(decorator.execute_chain_streaming(
            request="req", chain="chain",
        ))
        # Expect: 1 StreamToken + 1 ChainExecutionResult.
        tokens = [e for e in events if isinstance(e, StreamToken)]
        results = [e for e in events if isinstance(e, ChainExecutionResult)]
        assert len(tokens) == 1
        assert len(results) == 1
        assert tokens[0].text_delta == "Hello world"
        # finish_reason + token_id from the LAST inner token.
        assert tokens[0].finish_reason == "stop"
        assert tokens[0].token_id == 42

    def test_finish_reason_propagates_from_last_token(self):
        # Multiple inner tokens; only the last one carries
        # finish_reason — the joined token MUST forward it.
        inner = _FakeChainExecutor([
            _token(0, "a"),
            _token(1, "b", finish_reason="max_tokens"),
            _result(),
        ])
        decorator = BatchedTrailingShardedExecutor(inner=inner)
        events = list(decorator.execute_chain_streaming(
            request="req", chain="chain",
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        assert tokens[0].finish_reason == "max_tokens"
        assert tokens[0].text_delta == "ab"

    def test_empty_inner_stream_emits_nothing(self):
        # No tokens, no result → no events.
        inner = _FakeChainExecutor([])
        decorator = BatchedTrailingShardedExecutor(inner=inner)
        events = list(decorator.execute_chain_streaming(
            request="req", chain="chain",
        ))
        assert events == []

    def test_result_only_inner_passes_result_unchanged(self):
        # Inner emits ONLY ChainExecutionResult (no tokens) — the
        # result is forwarded, no synthetic StreamToken is fabricated.
        result = _result(output="result-only")
        inner = _FakeChainExecutor([result])
        decorator = BatchedTrailingShardedExecutor(inner=inner)
        events = list(decorator.execute_chain_streaming(
            request="req", chain="chain",
        ))
        assert events == [result]

    def test_event_ordering_token_before_result(self):
        # Even if inner yields in (result, tokens, ...) order (which
        # would be a bug in the inner, but we shouldn't propagate
        # the bug), the decorator MUST emit the joined token first
        # then the result.
        inner = _FakeChainExecutor([
            _token(0, "x"),
            _result(),
            _token(1, "y"),
        ])
        decorator = BatchedTrailingShardedExecutor(inner=inner)
        events = list(decorator.execute_chain_streaming(
            request="req", chain="chain",
        ))
        # Joined token = "xy"; result follows.
        tokens = [
            (i, e) for i, e in enumerate(events)
            if isinstance(e, StreamToken)
        ]
        results = [
            (i, e) for i, e in enumerate(events)
            if isinstance(e, ChainExecutionResult)
        ]
        assert len(tokens) == 1
        assert len(results) == 1
        assert tokens[0][0] < results[0][0]
        assert tokens[0][1].text_delta == "xy"

    def test_passthrough_of_request_chain_args(self):
        inner = _FakeChainExecutor([_result()])
        decorator = BatchedTrailingShardedExecutor(inner=inner)
        list(decorator.execute_chain_streaming(
            request="my-request", chain="my-chain",
        ))
        assert inner.call_log == [("my-request", "my-chain")]

    def test_construction_rejects_none(self):
        with pytest.raises(RuntimeError, match="requires an inner"):
            BatchedTrailingShardedExecutor(inner=None)

    def test_construction_rejects_non_streaming_inner(self):
        class _Bogus:
            pass
        with pytest.raises(
            RuntimeError, match="execute_chain_streaming",
        ):
            BatchedTrailingShardedExecutor(inner=_Bogus())
