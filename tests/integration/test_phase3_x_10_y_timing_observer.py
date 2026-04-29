"""Phase 3.x.10.y Task 5 — E2E timing-mask observer test.

Drives a fake inner ``StreamingLayerRunner`` that introduces
deliberately VARIABLE per-token latency, then plays a "passive
on-path observer" by measuring inter-frame wall-clock intervals
on the decorated output. Asserts the timing-mask invariants
from the timing-sidechannel memo §5:

  - **Undecorated baseline.** Inter-frame intervals correlate
    with the inner runner's per-token latency variance — i.e.,
    a wire observer learns per-token timing. This is the
    pre-3.x.10.y status quo for Tier A/B and the demonstrated
    leak for Tier C without padding.

  - **M2 (BatchedTrailingStreamingRunner).** Exactly ONE wire
    frame is emitted; per-token timing is structurally
    unobservable.

  - **M1 (FixedRateStreamingRunner).** Inter-frame intervals
    cluster at cadence; variance does NOT correlate with the
    inner runner's variance.

NOTE: ``conftest.py`` autouse-mocks ``time.sleep`` globally.
This file overrides the mock with a same-named fixture that
yields without patching — the timing assertions REQUIRE real
wall-clock sleep.

Slow-marked (wall-clock dependent); excluded from default CI.
"""

from __future__ import annotations

import time
import statistics
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pytest

from prsm.compute.inference import (
    BatchedTrailingStreamingRunner,
    FixedRateStreamingRunner,
    StreamingChunk,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType


pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def mock_time_sleep():
    """Override ``conftest.py``'s global ``time.sleep`` mock.
    Timing-mask assertions REQUIRE real wall-clock behavior."""
    yield


# ──────────────────────────────────────────────────────────────────────────
# Variable-latency inner runner — simulates a real LLM with non-uniform
# per-token decode time.
# ──────────────────────────────────────────────────────────────────────────


class _VariableLatencyInner:
    """Yields ``StreamingChunk``s with per-chunk sleep delays
    matching the supplied ``latencies_seconds`` list.

    Mirrors what a real autoregressive runner exhibits: some
    tokens are fast (cache hits, simple decode), some are slow
    (KV-cache misses, complex token, etc.). The variance is the
    side-channel signal an attacker exploits without padding.
    """

    def __init__(
        self,
        *,
        latencies_seconds: List[float],
        text_per_chunk: List[str],
    ) -> None:
        assert len(latencies_seconds) == len(text_per_chunk)
        self._latencies = latencies_seconds
        self._text = text_per_chunk

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
        n = len(self._latencies)
        joined = "".join(self._text)
        for i, (delay, piece) in enumerate(
            zip(self._latencies, self._text)
        ):
            time.sleep(delay)
            if i < n - 1:
                yield StreamingChunk(
                    sequence_index=i,
                    text_delta=piece,
                    token_id=i + 1,
                )
            else:
                # Terminal chunk: aggregate fields populated.
                yield StreamingChunk(
                    sequence_index=i,
                    text_delta=piece,
                    token_id=i + 1,
                    finish_reason="stop",
                    full_output_text=joined,
                    duration_seconds=sum(self._latencies),
                    tee_attestation=b"\x05" * 32,
                    tee_type=TEEType.SOFTWARE,
                    epsilon_spent=0.0,
                )


def _drive(
    runner: Any,
) -> Tuple[List[StreamingChunk], List[float]]:
    """Drive ``runner`` through one tail dispatch, recording a
    timestamp at each yield. Returns (chunks, timestamps)."""
    chunks: List[StreamingChunk] = []
    timestamps: List[float] = []
    for chunk in runner.run_layer_slice_streaming(
        model=None,
        layer_range=(0, 1),
        activation=np.zeros((1, 1), dtype=np.float32),
        privacy_tier=PrivacyLevel.NONE,
        is_final_stage=True,
    ):
        timestamps.append(time.monotonic())
        chunks.append(chunk)
    return chunks, timestamps


def _inter_frame_deltas_ms(timestamps: List[float]) -> List[float]:
    return [
        (cur - prev) * 1000.0
        for prev, cur in zip(timestamps, timestamps[1:])
    ]


# Latency profile: 4 inner chunks with deliberately HIGH variance
# (5ms, 80ms, 5ms, 80ms). The pattern is observable through the
# undecorated wire; M1 should mask it; M2 collapses to 1 frame.
_LATENCY_PROFILE = [0.005, 0.080, 0.005, 0.080]
_TEXT_PROFILE = ["a", "b", "c", "d"]


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


class TestUndecoratedBaselineLeaksTiming:
    """Pre-3.x.10.y status quo: an undecorated streaming runner
    emits frames as the inner runner produces them. A wire
    observer measuring inter-frame intervals reads off the
    inner's per-token decode timing."""

    def test_undecorated_intervals_track_inner_latency(self):
        inner = _VariableLatencyInner(
            latencies_seconds=_LATENCY_PROFILE,
            text_per_chunk=_TEXT_PROFILE,
        )
        chunks, timestamps = _drive(inner)
        deltas = _inter_frame_deltas_ms(timestamps)

        # 4 inner chunks → 4 wire frames → 3 inter-frame deltas.
        assert len(chunks) == 4
        assert len(deltas) == 3

        # Deltas reflect the inner latencies: roughly
        # [80, 5, 80] ms (latency BEFORE each non-first chunk
        # determines its arrival time). Variance is HIGH —
        # observer learns the alternating pattern.
        # Tolerance: thread-scheduling jitter ±15ms.
        # The HIGH-variance signal is what we want to PROVE
        # exists without padding.
        observed_stdev = statistics.pstdev(deltas)
        # With profile [5, 80, 5, 80]: deltas ≈ [80, 5, 80] ms
        # stdev ≈ 35-40ms. Anything ≥ 15ms confirms the leak.
        assert observed_stdev >= 15, (
            f"baseline did NOT leak per-token timing as expected; "
            f"deltas={deltas} stdev={observed_stdev}ms"
        )


class TestM2BatchedTrailingMasksTiming:
    """M2 (``BatchedTrailingStreamingRunner``) drains the inner
    stream fully and emits ONE wire frame. A wire observer sees
    no per-token timing because there are no per-token frames."""

    def test_m2_emits_single_frame_no_per_token_timing_observable(self):
        inner = _VariableLatencyInner(
            latencies_seconds=_LATENCY_PROFILE,
            text_per_chunk=_TEXT_PROFILE,
        )
        decorator = BatchedTrailingStreamingRunner(inner)
        chunks, timestamps = _drive(decorator)

        # Exactly ONE wire frame — by construction of M2, the
        # observer cannot extract per-token timing because
        # there's no inter-frame delta to measure.
        assert len(chunks) == 1
        # No inter-frame deltas exist (just one timestamp).
        deltas = _inter_frame_deltas_ms(timestamps)
        assert deltas == []

    def test_m2_total_duration_still_includes_inner_latency(self):
        # Honest-scope assertion: M2 hides per-token timing but
        # NOT total duration. Total stream duration ≈
        # sum(latencies) + small overhead, which the observer
        # CAN measure as a single observation. This is documented
        # as residual leakage in the timing-sidechannel memo.
        inner = _VariableLatencyInner(
            latencies_seconds=_LATENCY_PROFILE,
            text_per_chunk=_TEXT_PROFILE,
        )
        decorator = BatchedTrailingStreamingRunner(inner)
        t0 = time.monotonic()
        chunks, _ = _drive(decorator)
        t1 = time.monotonic()
        total_ms = (t1 - t0) * 1000.0
        expected_min_ms = sum(_LATENCY_PROFILE) * 1000.0 - 10
        assert total_ms >= expected_min_ms, (
            f"M2 should still take at least sum(latencies); "
            f"observed {total_ms:.1f}ms < expected ≥ "
            f"{expected_min_ms:.1f}ms"
        )


class TestM1FixedRateMasksTiming:
    """M1 (``FixedRateStreamingRunner``) emits at uniform
    cadence. A wire observer sees only the cadence; the inner
    runner's variable per-token timing is masked."""

    def test_m1_inter_frame_intervals_uniform_at_cadence(self):
        cadence = 0.040  # 40ms
        inner = _VariableLatencyInner(
            latencies_seconds=_LATENCY_PROFILE,
            text_per_chunk=_TEXT_PROFILE,
        )
        decorator = FixedRateStreamingRunner(
            inner, cadence_seconds=cadence,
        )
        chunks, timestamps = _drive(decorator)
        deltas = _inter_frame_deltas_ms(timestamps)

        # All inter-frame intervals cluster at cadence.
        # Variance is LOW — the observer learns only the
        # cadence, not the inner's per-token timing.
        # Allow ±10ms tolerance for thread-scheduling jitter.
        cadence_ms = cadence * 1000.0
        for i, d in enumerate(deltas):
            assert abs(d - cadence_ms) < 15, (
                f"frame {i}→{i+1} delta {d:.1f}ms not at "
                f"cadence {cadence_ms}ms (jitter > 15ms); "
                f"all deltas={deltas}"
            )

        # Variance comparison: M1's stdev MUST be much lower
        # than the baseline's stdev, demonstrating the timing
        # mask works.
        m1_stdev = statistics.pstdev(deltas) if len(deltas) > 1 else 0
        # M1 stdev should be < 10ms (just thread jitter); the
        # baseline test asserted stdev ≥ 15ms (real per-token
        # timing leak). The gap proves the mask.
        assert m1_stdev < 10, (
            f"M1 stdev {m1_stdev}ms unexpectedly high — "
            f"timing mask not working as designed; deltas={deltas}"
        )

    def test_m1_emits_at_least_inner_chunk_count_frames(self):
        # M1 emits real chunks plus no-op pads. With cadence=40ms
        # and total inner duration ≈170ms, expect roughly 4-6
        # frames (4 real + possibly some pads at end).
        cadence = 0.040
        inner = _VariableLatencyInner(
            latencies_seconds=_LATENCY_PROFILE,
            text_per_chunk=_TEXT_PROFILE,
        )
        decorator = FixedRateStreamingRunner(
            inner, cadence_seconds=cadence,
        )
        chunks, _ = _drive(decorator)
        # At least 4 frames: pads + real chunks. Upper bound
        # depends on how much the inner outpaces the cadence.
        assert len(chunks) >= 4
        # Joined-text invariant: dropped no-op pads + real
        # text → equals "abcd".
        joined = "".join(c.text_delta for c in chunks)
        assert joined == "abcd"


class TestTimingMaskComparison:
    """Cross-decorator comparison: M1 stdev << baseline stdev,
    proving the timing-mask invariant holds. M2 emits a single
    frame so it has no inter-frame stdev to compute."""

    def test_m1_stdev_strictly_less_than_baseline_stdev(self):
        # Run baseline + M1 back-to-back with the same inner
        # latency profile, compute stdev for each, assert M1's
        # stdev is at least 3x smaller. This is the load-bearing
        # invariant for the timing-mask claim.
        inner_baseline = _VariableLatencyInner(
            latencies_seconds=_LATENCY_PROFILE,
            text_per_chunk=_TEXT_PROFILE,
        )
        _, ts_baseline = _drive(inner_baseline)
        baseline_stdev = statistics.pstdev(
            _inter_frame_deltas_ms(ts_baseline),
        )

        inner_m1 = _VariableLatencyInner(
            latencies_seconds=_LATENCY_PROFILE,
            text_per_chunk=_TEXT_PROFILE,
        )
        decorator = FixedRateStreamingRunner(
            inner_m1, cadence_seconds=0.040,
        )
        _, ts_m1 = _drive(decorator)
        m1_stdev = statistics.pstdev(
            _inter_frame_deltas_ms(ts_m1),
        )

        # Baseline leaks (high stdev); M1 masks (low stdev).
        # M1 stdev ÷ baseline stdev should be well under 1.0.
        # A 3x margin confirms the mask is operationally effective
        # (not just within measurement noise).
        assert baseline_stdev > 3 * m1_stdev, (
            f"timing-mask invariant violated: "
            f"baseline_stdev={baseline_stdev:.1f}ms vs "
            f"m1_stdev={m1_stdev:.1f}ms — expected ratio > 3x"
        )
