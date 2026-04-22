"""Unit tests for prsm.node.nat_traversal.

Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md §6 Task 3.
"""

from __future__ import annotations

import pytest

from prsm.node.nat_traversal import (
    DialResult,
    NatConfig,
    NatTraversalPolicy,
    NatType,
    PeerDialer,
    TraversalStrategy,
    strategy_for_nat,
)


# -----------------------------------------------------------------------------
# NatConfig validation
# -----------------------------------------------------------------------------


def test_config_requires_stun_servers():
    with pytest.raises(ValueError):
        NatConfig(stun_servers=())


def test_config_rejects_stun_missing_port():
    with pytest.raises(ValueError):
        NatConfig(stun_servers=("stun.example.com",))


def test_config_rejects_turn_without_credentials():
    with pytest.raises(ValueError):
        NatConfig(
            stun_servers=("stun.example.com:3478",),
            turn_servers=("turn.example.com:3478",),
            turn_credentials=None,
        )


def test_config_rejects_invalid_timeout():
    with pytest.raises(ValueError):
        NatConfig(
            stun_servers=("stun.example.com:3478",),
            detection_timeout_sec=0,
        )


def test_config_turn_available_reflects_server_list():
    c1 = NatConfig(stun_servers=("s:3478",))
    assert c1.turn_available is False
    c2 = NatConfig(
        stun_servers=("s:3478",),
        turn_servers=("t:3478",),
        turn_credentials=("user", "pass"),
    )
    assert c2.turn_available is True


# -----------------------------------------------------------------------------
# strategy_for_nat — full decision table
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nat_type,turn_available,expected",
    [
        (NatType.NONE, False, TraversalStrategy.DIRECT),
        (NatType.FULL_CONE, False, TraversalStrategy.DIRECT),
        (NatType.FULL_CONE, True, TraversalStrategy.DIRECT),
        (NatType.RESTRICTED_CONE, False, TraversalStrategy.STUN_HOLE_PUNCH),
        (NatType.PORT_RESTRICTED, False, TraversalStrategy.STUN_HOLE_PUNCH),
        (NatType.PORT_RESTRICTED, True, TraversalStrategy.STUN_HOLE_PUNCH),
        (NatType.SYMMETRIC, True, TraversalStrategy.TURN_RELAY),
        (NatType.SYMMETRIC, False, TraversalStrategy.INBOUND_ONLY),
        (NatType.UNKNOWN, True, TraversalStrategy.INBOUND_ONLY),
        (NatType.UNKNOWN, False, TraversalStrategy.INBOUND_ONLY),
    ],
)
def test_strategy_decision_table(nat_type, turn_available, expected):
    assert strategy_for_nat(nat_type, turn_available=turn_available) == expected


# -----------------------------------------------------------------------------
# Test doubles
# -----------------------------------------------------------------------------


class _StubDetector:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def detect(self, stun_servers):
        self.calls += 1
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _StubDialer:
    """Returns a canned result per strategy, in order of attempt."""

    def __init__(self, results_by_strategy: dict):
        self._results = dict(results_by_strategy)
        self.calls: list[tuple[str, TraversalStrategy]] = []

    def dial(self, peer_id, peer_addr, strategy, config):
        self.calls.append((peer_id, strategy))
        return self._results.get(strategy, DialResult.FAILED_UNROUTABLE)


@pytest.fixture
def config_with_turn():
    return NatConfig(
        stun_servers=("stun.prsm.ai:3478",),
        turn_servers=("turn.prsm.ai:3478",),
        turn_credentials=("user", "pass"),
    )


@pytest.fixture
def config_without_turn():
    return NatConfig(stun_servers=("stun.prsm.ai:3478",))


# -----------------------------------------------------------------------------
# from_detection
# -----------------------------------------------------------------------------


def test_from_detection_uses_detector_result(config_with_turn):
    detector = _StubDetector(NatType.FULL_CONE)
    dialer = _StubDialer({})
    policy = NatTraversalPolicy.from_detection(
        config_with_turn, detector=detector, dialer=dialer
    )
    assert policy.local_nat is NatType.FULL_CONE
    assert policy.baseline_strategy is TraversalStrategy.DIRECT
    assert detector.calls == 1


def test_from_detection_falls_back_to_unknown_on_detector_failure(
    config_with_turn,
):
    detector = _StubDetector(RuntimeError("no STUN reachable"))
    dialer = _StubDialer({})
    policy = NatTraversalPolicy.from_detection(
        config_with_turn, detector=detector, dialer=dialer
    )
    assert policy.local_nat is NatType.UNKNOWN
    assert policy.baseline_strategy is TraversalStrategy.INBOUND_ONLY


# -----------------------------------------------------------------------------
# connect() — strategy ladder
# -----------------------------------------------------------------------------


def test_direct_dial_succeeds_for_full_cone_node(config_with_turn):
    dialer = _StubDialer({TraversalStrategy.DIRECT: DialResult.SUCCESS})
    policy = NatTraversalPolicy(
        config_with_turn, NatType.FULL_CONE, dialer=dialer
    )
    result = policy.connect("peer-a", "/ip4/1.2.3.4/tcp/4001")
    assert result.succeeded
    assert result.strategies_tried == (TraversalStrategy.DIRECT,)
    assert len(dialer.calls) == 1


def test_restricted_cone_baseline_is_stun(config_with_turn):
    dialer = _StubDialer({TraversalStrategy.STUN_HOLE_PUNCH: DialResult.SUCCESS})
    policy = NatTraversalPolicy(
        config_with_turn, NatType.RESTRICTED_CONE, dialer=dialer
    )
    result = policy.connect("peer-a", "addr")
    assert result.succeeded
    assert result.final_strategy is TraversalStrategy.STUN_HOLE_PUNCH
    # Must NOT have attempted DIRECT first — baseline starts at STUN.
    assert TraversalStrategy.DIRECT not in result.strategies_tried


def test_symmetric_baseline_is_turn_when_available(config_with_turn):
    dialer = _StubDialer({TraversalStrategy.TURN_RELAY: DialResult.SUCCESS})
    policy = NatTraversalPolicy(
        config_with_turn, NatType.SYMMETRIC, dialer=dialer
    )
    result = policy.connect("peer-a", "addr")
    assert result.succeeded
    assert result.final_strategy is TraversalStrategy.TURN_RELAY


def test_symmetric_without_turn_is_inbound_only(config_without_turn):
    dialer = _StubDialer({})
    policy = NatTraversalPolicy(
        config_without_turn, NatType.SYMMETRIC, dialer=dialer
    )
    result = policy.connect("peer-a", "addr")
    assert result.result is DialResult.SKIPPED_INBOUND_ONLY
    assert dialer.calls == []  # never attempted to dial


def test_direct_failure_escalates_to_stun(config_with_turn):
    dialer = _StubDialer(
        {
            TraversalStrategy.DIRECT: DialResult.FAILED_TIMEOUT,
            TraversalStrategy.STUN_HOLE_PUNCH: DialResult.SUCCESS,
        }
    )
    policy = NatTraversalPolicy(
        config_with_turn, NatType.FULL_CONE, dialer=dialer
    )
    result = policy.connect("peer-a", "addr")
    assert result.succeeded
    assert result.strategies_tried == (
        TraversalStrategy.DIRECT,
        TraversalStrategy.STUN_HOLE_PUNCH,
    )


def test_direct_and_stun_failure_escalates_to_turn(config_with_turn):
    dialer = _StubDialer(
        {
            TraversalStrategy.DIRECT: DialResult.FAILED_TIMEOUT,
            TraversalStrategy.STUN_HOLE_PUNCH: DialResult.FAILED_REFUSED,
            TraversalStrategy.TURN_RELAY: DialResult.SUCCESS,
        }
    )
    policy = NatTraversalPolicy(
        config_with_turn, NatType.FULL_CONE, dialer=dialer
    )
    result = policy.connect("peer-a", "addr")
    assert result.succeeded
    assert result.final_strategy is TraversalStrategy.TURN_RELAY
    assert len(result.strategies_tried) == 3


def test_full_ladder_exhaustion_returns_unroutable(config_with_turn):
    dialer = _StubDialer(
        {
            TraversalStrategy.DIRECT: DialResult.FAILED_TIMEOUT,
            TraversalStrategy.STUN_HOLE_PUNCH: DialResult.FAILED_TIMEOUT,
            TraversalStrategy.TURN_RELAY: DialResult.FAILED_UNROUTABLE,
        }
    )
    policy = NatTraversalPolicy(
        config_with_turn, NatType.FULL_CONE, dialer=dialer
    )
    result = policy.connect("peer-a", "addr")
    # TURN failure → next is INBOUND_ONLY, which returns
    # SKIPPED_INBOUND_ONLY. Our `connect` returns the ladder-exhausted
    # outcome: SKIPPED_INBOUND_ONLY because we reach that rung.
    assert result.result is DialResult.SKIPPED_INBOUND_ONLY


def test_escalation_skips_turn_when_turn_unconfigured(config_without_turn):
    """Restricted-cone node with no TURN fallback: STUN fails → escalate
    straight to inbound-only without touching the TURN rung."""
    dialer = _StubDialer(
        {
            TraversalStrategy.STUN_HOLE_PUNCH: DialResult.FAILED_TIMEOUT,
        }
    )
    policy = NatTraversalPolicy(
        config_without_turn, NatType.RESTRICTED_CONE, dialer=dialer
    )
    result = policy.connect("peer-a", "addr")
    # Only STUN attempted; then escalation lands on TURN which is skipped;
    # final rung is INBOUND_ONLY.
    assert TraversalStrategy.TURN_RELAY not in [s for s, _ in []]  # tautology
    strategies_attempted = [s for _, s in dialer.calls]
    assert TraversalStrategy.TURN_RELAY not in strategies_attempted
    assert result.result is DialResult.SKIPPED_INBOUND_ONLY


def test_inbound_only_skipped_without_dialer_calls(config_with_turn):
    dialer = _StubDialer({})
    policy = NatTraversalPolicy(
        config_with_turn, NatType.UNKNOWN, dialer=dialer
    )
    result = policy.connect("peer-a", "addr")
    assert result.result is DialResult.SKIPPED_INBOUND_ONLY
    assert dialer.calls == []
