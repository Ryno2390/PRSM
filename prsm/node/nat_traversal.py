"""NAT detection + traversal policy for the PRSM P2P layer.

Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md §3.2 + §6 Task 3.

Plan §3.2's decision tree:

    Full-cone NAT / public  -->  direct dial
    Restricted-cone         -->  ICE with STUN hole-punching
    Port-restricted         -->  ICE with STUN hole-punching
    Symmetric NAT           -->  TURN relay fallback
    (TURN unavailable)      -->  inbound-only (accept dials, never dial out)

This module encodes that decision tree as a pure state machine with
injectable `NatDetector` + `PeerDialer` abstractions, so the strategy
logic is testable independently of libp2p wire code. The Foundation's
libp2p AutoNAT / AutoRelay / circuit-relay-v2 configuration lives
alongside this module and consumes the policy decisions it produces.

Scope boundary — what this module does NOT do:
  * Actual STUN RFC 5389 / RFC 5780 probing (delegated to the detector).
  * Actual ICE / TURN session management (delegated to the dialer).
  * libp2p AutoNAT / AutoRelay wiring (consumed by prsm/node/libp2p_*
    at Foundation ops time).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Tuple


logger = logging.getLogger(__name__)


__all__ = [
    "DialResult",
    "NatConfig",
    "NatDetector",
    "NatTraversalPolicy",
    "NatType",
    "PeerDialer",
    "TraversalAttempt",
    "TraversalStrategy",
    "strategy_for_nat",
]


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class NatType(str, Enum):
    NONE = "none"                    # public IP, no NAT
    FULL_CONE = "full_cone"
    RESTRICTED_CONE = "restricted_cone"
    PORT_RESTRICTED = "port_restricted"
    SYMMETRIC = "symmetric"
    UNKNOWN = "unknown"              # detection failed / inconclusive


class TraversalStrategy(str, Enum):
    DIRECT = "direct"                # plain dial, no intermediation
    STUN_HOLE_PUNCH = "stun_hole_punch"
    TURN_RELAY = "turn_relay"
    INBOUND_ONLY = "inbound_only"    # cannot dial out


class DialResult(str, Enum):
    SUCCESS = "success"
    FAILED_TIMEOUT = "failed_timeout"
    FAILED_REFUSED = "failed_refused"
    FAILED_UNROUTABLE = "failed_unroutable"
    SKIPPED_INBOUND_ONLY = "skipped_inbound_only"


def _is_failure(result: DialResult) -> bool:
    return result in (
        DialResult.FAILED_TIMEOUT,
        DialResult.FAILED_REFUSED,
        DialResult.FAILED_UNROUTABLE,
    )


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class NatConfig:
    """Foundation-operated NAT-traversal endpoints.

    `stun_servers` is a list of `host:port` endpoints; RFC 5389 UDP by default.
    `turn_servers` is a list of TURN `host:port` endpoints; credentials are
    TURN long-term or short-term as appropriate for the deployment.
    """

    stun_servers: Tuple[str, ...] = ()
    turn_servers: Tuple[str, ...] = ()
    turn_credentials: Optional[Tuple[str, str]] = None  # (username, password)
    detection_timeout_sec: float = 5.0

    def __post_init__(self) -> None:
        if not self.stun_servers:
            raise ValueError("at least one STUN server required")
        for s in self.stun_servers:
            if ":" not in s:
                raise ValueError(f"STUN server missing port: {s!r}")
        for t in self.turn_servers:
            if ":" not in t:
                raise ValueError(f"TURN server missing port: {t!r}")
        if self.turn_servers and not self.turn_credentials:
            raise ValueError("TURN servers require credentials")
        if self.detection_timeout_sec <= 0:
            raise ValueError("detection_timeout_sec must be > 0")

    @property
    def turn_available(self) -> bool:
        return bool(self.turn_servers)


# -----------------------------------------------------------------------------
# Pure decision table
# -----------------------------------------------------------------------------


def strategy_for_nat(
    nat_type: NatType, *, turn_available: bool
) -> TraversalStrategy:
    """Plan §3.2 decision table — baseline strategy for a given local NAT.

    `turn_available` is a config property, not a NAT property; symmetric
    NAT without TURN fallback is inbound-only. UNKNOWN is treated as
    inbound-only because dialing blindly risks silent failures — the
    safer default is to surface the detection problem to the operator.
    """
    if nat_type in (NatType.NONE, NatType.FULL_CONE):
        return TraversalStrategy.DIRECT
    if nat_type in (NatType.RESTRICTED_CONE, NatType.PORT_RESTRICTED):
        return TraversalStrategy.STUN_HOLE_PUNCH
    if nat_type == NatType.SYMMETRIC:
        return (
            TraversalStrategy.TURN_RELAY
            if turn_available
            else TraversalStrategy.INBOUND_ONLY
        )
    if nat_type == NatType.UNKNOWN:
        return TraversalStrategy.INBOUND_ONLY
    raise ValueError(f"unhandled NatType: {nat_type!r}")


_ESCALATION_LADDER: Tuple[TraversalStrategy, ...] = (
    TraversalStrategy.DIRECT,
    TraversalStrategy.STUN_HOLE_PUNCH,
    TraversalStrategy.TURN_RELAY,
    TraversalStrategy.INBOUND_ONLY,
)


def _escalate(current: TraversalStrategy) -> Optional[TraversalStrategy]:
    """Return the next strategy in the ladder, or None if at the bottom."""
    idx = _ESCALATION_LADDER.index(current)
    if idx + 1 >= len(_ESCALATION_LADDER):
        return None
    return _ESCALATION_LADDER[idx + 1]


# -----------------------------------------------------------------------------
# Abstractions over wire work
# -----------------------------------------------------------------------------


class NatDetector(Protocol):
    """Detects the local node's NAT type by probing STUN servers.

    Real implementation: RFC 5780 classic STUN / libp2p AutoNAT. Test
    implementation: a canned-response stub.
    """

    def detect(self, stun_servers: Tuple[str, ...]) -> NatType: ...


class PeerDialer(Protocol):
    """Attempts a connection to a peer via a specific strategy."""

    def dial(
        self,
        peer_id: str,
        peer_addr: str,
        strategy: TraversalStrategy,
        config: NatConfig,
    ) -> DialResult: ...


# -----------------------------------------------------------------------------
# Result records
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TraversalAttempt:
    peer_id: str
    strategies_tried: Tuple[TraversalStrategy, ...]
    final_strategy: TraversalStrategy
    result: DialResult

    @property
    def succeeded(self) -> bool:
        return self.result is DialResult.SUCCESS


# -----------------------------------------------------------------------------
# Policy
# -----------------------------------------------------------------------------


class NatTraversalPolicy:
    """Applies the plan §3.2 strategy ladder with per-connect escalation.

    Per-connect semantics:
      * The BASELINE strategy is determined by local NAT + TURN availability.
      * If a connect attempt fails with a transport-level failure (timeout /
        refused / unroutable), escalate one rung on the ladder and retry.
      * Never skip rungs (skipping DIRECT → TURN would pointlessly burn
        TURN bandwidth).
      * Stop on SUCCESS or when INBOUND_ONLY is reached (nothing to
        escalate to).
    """

    def __init__(
        self,
        config: NatConfig,
        local_nat: NatType,
        *,
        dialer: PeerDialer,
    ) -> None:
        self._config = config
        self._local_nat = local_nat
        self._dialer = dialer
        self._baseline = strategy_for_nat(
            local_nat, turn_available=config.turn_available
        )

    @property
    def local_nat(self) -> NatType:
        return self._local_nat

    @property
    def baseline_strategy(self) -> TraversalStrategy:
        return self._baseline

    @classmethod
    def from_detection(
        cls,
        config: NatConfig,
        *,
        detector: NatDetector,
        dialer: PeerDialer,
    ) -> "NatTraversalPolicy":
        """Build a policy by first probing the local NAT type via the
        detector. Falls back to UNKNOWN (→ inbound-only) on detector
        failure, logging but not propagating — operators see the state
        via the resulting baseline_strategy.
        """
        try:
            local_nat = detector.detect(config.stun_servers)
        except Exception:
            logger.exception("NAT detection failed; assuming UNKNOWN")
            local_nat = NatType.UNKNOWN
        return cls(config, local_nat, dialer=dialer)

    def connect(self, peer_id: str, peer_addr: str) -> TraversalAttempt:
        tried: list[TraversalStrategy] = []
        current = self._baseline

        while True:
            if current is TraversalStrategy.INBOUND_ONLY:
                tried.append(current)
                return TraversalAttempt(
                    peer_id=peer_id,
                    strategies_tried=tuple(tried),
                    final_strategy=current,
                    result=DialResult.SKIPPED_INBOUND_ONLY,
                )

            # Skip TURN if it's unavailable — escalate straight to
            # inbound-only. Symmetric-local nodes without TURN already
            # baseline here; this guard protects non-baseline escalations
            # that land on TURN when no TURN servers are configured.
            if (
                current is TraversalStrategy.TURN_RELAY
                and not self._config.turn_available
            ):
                next_strat = _escalate(current)
                if next_strat is None:
                    break
                current = next_strat
                continue

            tried.append(current)
            result = self._dialer.dial(peer_id, peer_addr, current, self._config)
            if not _is_failure(result):
                return TraversalAttempt(
                    peer_id=peer_id,
                    strategies_tried=tuple(tried),
                    final_strategy=current,
                    result=result,
                )

            next_strat = _escalate(current)
            if next_strat is None:
                break
            current = next_strat

        # Ladder exhausted.
        return TraversalAttempt(
            peer_id=peer_id,
            strategies_tried=tuple(tried),
            final_strategy=tried[-1] if tried else self._baseline,
            result=DialResult.FAILED_UNROUTABLE,
        )
