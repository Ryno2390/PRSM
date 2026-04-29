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

    def test_post_result_tokens_dropped_round1_m1(self):
        # Round-1 review M1 remediation. Tokens emitted by the
        # inner AFTER the terminal result violate the streaming
        # contract — the decorator MUST drop them rather than
        # silently merging into the joined text (which would
        # re-order content across the terminal boundary).
        inner = _FakeChainExecutor([
            _token(0, "x"),
            _result(),
            _token(1, "y"),  # post-terminal — DROPPED
        ])
        decorator = BatchedTrailingShardedExecutor(inner=inner)
        events = list(decorator.execute_chain_streaming(
            request="req", chain="chain",
        ))
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
        # Joined text contains ONLY pre-terminal token "x" — "y"
        # was correctly dropped.
        assert tokens[0][1].text_delta == "x"

    def test_non_str_text_delta_coerced_round1_m2(self):
        # Round-1 review M2 remediation. Defensive coerce for
        # non-str text_delta — if upstream ever ships None or a
        # non-string type, the decorator must NOT TypeError mid-
        # generator and crash the entire Tier C request.
        inner = _FakeChainExecutor([
            _token(0, "alpha"),
            StreamToken(
                sequence_index=1,
                text_delta=None,  # type: ignore[arg-type]
                token_id=None,
                finish_reason=None,
            ),
            _token(2, "beta", finish_reason="stop"),
            _result(),
        ])
        decorator = BatchedTrailingShardedExecutor(inner=inner)
        events = list(decorator.execute_chain_streaming(
            request="req", chain="chain",
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        # The None text_delta contributed "" — joined is "alphabeta".
        assert tokens[0].text_delta == "alphabeta"
        assert tokens[0].finish_reason == "stop"

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


# ──────────────────────────────────────────────────────────────────────────
# Task 2 — FixedRateShardedExecutor (M1)
# ──────────────────────────────────────────────────────────────────────────


class _ScriptedClock:
    """Deterministic monotonic clock — tests advance time explicitly
    via the sleep stub. Each ``sleep(dt)`` advances ``now`` by ``dt``
    so the cadence math becomes reproducible without wall-clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)
        self.sleep_calls: List[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(float(seconds))
        self.now += float(seconds)


class TestFixedRateShardedExecutor:
    def test_protocol_conformance_execute_chain_streaming(self):
        decorator = FixedRateShardedExecutor(
            inner=_FakeChainExecutor([_result()]),
            cadence_seconds=0.05,
        )
        assert callable(getattr(decorator, "execute_chain_streaming", None))

    def test_first_token_emitted_without_initial_delay(self):
        # First StreamToken has no prior emission to gate against,
        # so it should pass through immediately (no sleep).
        clk = _ScriptedClock()
        inner = _FakeChainExecutor([
            _token(0, "hello"),
            _result(),
        ])
        decorator = FixedRateShardedExecutor(
            inner=inner, cadence_seconds=0.5,
            clock=clk, sleep=clk.sleep,
        )
        events = list(decorator.execute_chain_streaming(
            request="r", chain="c",
        ))
        assert len(events) == 2
        assert clk.sleep_calls == []  # no initial delay

    def test_subsequent_tokens_paced_at_cadence(self):
        # Inner emits 3 tokens back-to-back (chain runs faster than
        # cadence). Decorator must sleep ≈ cadence between each.
        clk = _ScriptedClock(start=1000.0)
        inner = _FakeChainExecutor([
            _token(0, "a"),
            _token(1, "b"),
            _token(2, "c"),
            _result(),
        ])
        decorator = FixedRateShardedExecutor(
            inner=inner, cadence_seconds=0.5,
            clock=clk, sleep=clk.sleep,
        )
        list(decorator.execute_chain_streaming(
            request="r", chain="c",
        ))
        # First token: no sleep. Tokens 2 + 3: sleep ≈ 0.5 each.
        # Result: no sleep (pass-through).
        assert len(clk.sleep_calls) == 2
        for s in clk.sleep_calls:
            assert s == pytest.approx(0.5, abs=1e-9)

    def test_no_artificial_delay_when_inner_runs_slower_than_cadence(self):
        # Simulate the inner already taking longer than cadence
        # between tokens — decorator should NOT sleep.
        clk = _ScriptedClock(start=1000.0)
        inner_events = [
            _token(0, "a"),
            _token(1, "b"),
            _result(),
        ]
        # Wrap inner so that yielding each event advances the clock
        # past the cadence target.
        original_inner = _FakeChainExecutor(inner_events)
        cadence = 0.05

        class _SlowInner:
            def execute_chain_streaming(self, *, request, chain):
                for ev in original_inner.execute_chain_streaming(
                    request=request, chain=chain,
                ):
                    yield ev
                    clk.now += cadence * 2.0  # advance > cadence

        decorator = FixedRateShardedExecutor(
            inner=_SlowInner(), cadence_seconds=cadence,
            clock=clk, sleep=clk.sleep,
        )
        list(decorator.execute_chain_streaming(
            request="r", chain="c",
        ))
        # No sleep — chain was already slow enough.
        assert clk.sleep_calls == []

    def test_chain_execution_result_forwarded_without_cadence_delay(self):
        # The terminal isn't part of the per-token timing surface;
        # gating it would just delay receipt without masking.
        clk = _ScriptedClock(start=1000.0)
        inner = _FakeChainExecutor([
            _token(0, "a"),
            _result(),
        ])
        decorator = FixedRateShardedExecutor(
            inner=inner, cadence_seconds=10.0,  # generous cadence
            clock=clk, sleep=clk.sleep,
        )
        events = list(decorator.execute_chain_streaming(
            request="r", chain="c",
        ))
        # 2 events. Sleep called only on the inter-token delay (none
        # here since just one token); result passes through.
        assert len(events) == 2
        assert clk.sleep_calls == []

    def test_construction_rejects_non_positive_cadence(self):
        inner = _FakeChainExecutor([_result()])
        for bad in [0.0, -0.1, 0]:
            with pytest.raises(ValueError, match="cadence_seconds"):
                FixedRateShardedExecutor(
                    inner=inner, cadence_seconds=bad,
                )

    def test_construction_rejects_bool_cadence(self):
        # bool is a subclass of int in Python — explicit defense.
        inner = _FakeChainExecutor([_result()])
        with pytest.raises(ValueError, match="cadence_seconds"):
            FixedRateShardedExecutor(
                inner=inner, cadence_seconds=True,  # type: ignore[arg-type]
            )

    def test_construction_rejects_none_inner(self):
        with pytest.raises(RuntimeError, match="requires an inner"):
            FixedRateShardedExecutor(
                inner=None, cadence_seconds=0.05,
            )

    def test_construction_rejects_non_streaming_inner(self):
        class _Bogus:
            pass
        with pytest.raises(
            RuntimeError, match="execute_chain_streaming",
        ):
            FixedRateShardedExecutor(
                inner=_Bogus(), cadence_seconds=0.05,
            )

    def test_passthrough_of_request_chain_args(self):
        clk = _ScriptedClock()
        inner = _FakeChainExecutor([_result()])
        decorator = FixedRateShardedExecutor(
            inner=inner, cadence_seconds=0.05,
            clock=clk, sleep=clk.sleep,
        )
        list(decorator.execute_chain_streaming(
            request="my-request", chain="my-chain",
        ))
        assert inner.call_log == [("my-request", "my-chain")]


# ──────────────────────────────────────────────────────────────────────────
# Task 3 — make_tier_c_sharded_executor factory
# ──────────────────────────────────────────────────────────────────────────


class TestMakeTierCShardedExecutor:
    def test_mode_m2_returns_batched_trailing(self):
        from prsm.compute.chain_rpc import make_tier_c_sharded_executor
        inner = _FakeChainExecutor([_result()])
        decorator = make_tier_c_sharded_executor(inner, mode="m2")
        assert isinstance(decorator, BatchedTrailingShardedExecutor)

    def test_mode_m1_returns_fixed_rate(self):
        from prsm.compute.chain_rpc import make_tier_c_sharded_executor
        inner = _FakeChainExecutor([_result()])
        decorator = make_tier_c_sharded_executor(
            inner, mode="m1", cadence_seconds=0.05,
        )
        assert isinstance(decorator, FixedRateShardedExecutor)

    def test_mode_m1_requires_cadence(self):
        from prsm.compute.chain_rpc import make_tier_c_sharded_executor
        inner = _FakeChainExecutor([_result()])
        with pytest.raises(ValueError, match="cadence_seconds is required"):
            make_tier_c_sharded_executor(inner, mode="m1")

    def test_mode_m2_rejects_cadence(self):
        from prsm.compute.chain_rpc import make_tier_c_sharded_executor
        inner = _FakeChainExecutor([_result()])
        with pytest.raises(
            ValueError, match="not applicable for mode='m2'",
        ):
            make_tier_c_sharded_executor(
                inner, mode="m2", cadence_seconds=0.05,
            )

    def test_unknown_mode_raises(self):
        from prsm.compute.chain_rpc import make_tier_c_sharded_executor
        inner = _FakeChainExecutor([_result()])
        with pytest.raises(ValueError, match="unknown mode"):
            make_tier_c_sharded_executor(inner, mode="bogus")

    def test_top_level_chain_rpc_exports(self):
        # Module-level smoke: the three names are exported under
        # prsm.compute.chain_rpc.
        import prsm.compute.chain_rpc as mod
        assert "BatchedTrailingShardedExecutor" in mod.__all__
        assert "FixedRateShardedExecutor" in mod.__all__
        assert "make_tier_c_sharded_executor" in mod.__all__
