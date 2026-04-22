"""In-process chaos simulation for Phase 6 P2P primitives.

Simulates a small-to-medium cluster (20–100 nodes) driving the real
`LivenessMonitor` + `RateLimiter` code under adversarial inputs:

  * dead-on-arrival peers that never pong — should be evicted in ~3
    intervals.
  * adversarial peers that spam DHT queries above the per-peer cap —
    should be throttled then banned, without impacting legitimate
    traffic targeted at the same victim.
  * churn: legitimate peers join/leave between ticks.

The simulation is fully deterministic: all randomness flows through an
injected `random.Random` instance. A fixed seed produces identical
ChaosReport output across runs / machines / Python versions.
"""

from __future__ import annotations

import random as _random_mod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from prsm.node.liveness import LivenessMonitor
from prsm.node.rate_limit import (
    RateLimit,
    RateLimiter,
    RateLimitResult,
)


__all__ = [
    "ChaosReport",
    "ChaosScenario",
    "SimNetwork",
    "SimPeer",
]


# -----------------------------------------------------------------------------
# Scenario + report data
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ChaosScenario:
    """Parameters for a single chaos run.

    Defaults correspond to a small CI-friendly scenario. The plan's
    full-scale scenario (100 nodes, 30% churn/hour, 1-hour run) is
    obtainable by scaling node_count + ticks while keeping the primitives
    unchanged.
    """

    node_count: int = 20
    adversarial_count: int = 2
    dead_peer_count: int = 3
    churn_fraction_per_tick: float = 0.05
    ticks: int = 10
    ping_interval_sec: float = 30.0
    # Legitimate requests per peer per tick (targeted at a random peer).
    legitimate_requests_per_tick: int = 3
    # Adversarial "spam" requests per adversary per tick (targeted at a
    # random peer). Default is well above the rate-limit cap.
    adversarial_requests_per_tick: int = 20
    # Rate limits for the simulated DHT category.
    dht_max_per_window: int = 10
    dht_window_sec: float = 60.0
    violations_for_ban: int = 3
    throttle_duration_sec: float = 60.0
    ban_duration_sec: float = 3600.0
    seed: int = 0


@dataclass
class ChaosReport:
    scenario: ChaosScenario
    legit_requests_total: int = 0
    legit_requests_allowed: int = 0
    legit_requests_rejected: int = 0
    adversarial_requests_total: int = 0
    adversarial_requests_allowed: int = 0
    adversarial_requests_rejected: int = 0
    dead_peers_evicted: int = 0
    dead_peers_still_alive: int = 0
    adversarial_peers_banned: int = 0
    adversarial_peers_not_banned: int = 0
    # Per-adversary tick-count to first ban (averaged).
    adversary_ticks_to_ban: List[int] = field(default_factory=list)
    # Per-dead-peer tick-count to eviction (averaged).
    dead_peer_ticks_to_eviction: List[int] = field(default_factory=list)

    @property
    def legit_allow_rate(self) -> float:
        if self.legit_requests_total == 0:
            return 1.0
        return self.legit_requests_allowed / self.legit_requests_total

    @property
    def adversarial_reject_rate(self) -> float:
        if self.adversarial_requests_total == 0:
            return 0.0
        return self.adversarial_requests_rejected / self.adversarial_requests_total

    @property
    def dead_peer_eviction_rate(self) -> float:
        total = self.scenario.dead_peer_count
        if total == 0:
            return 1.0
        return self.dead_peers_evicted / total


# -----------------------------------------------------------------------------
# Peer + network
# -----------------------------------------------------------------------------


@dataclass
class SimPeer:
    peer_id: str
    is_adversarial: bool = False
    is_dead: bool = False  # never responds to pings
    # Every peer runs its own liveness tracker over its view of the other
    # peers, and its own rate limiter gating incoming requests.
    liveness: Optional[LivenessMonitor] = None
    rate_limiter: Optional[RateLimiter] = None
    active: bool = True
    # Adversaries concentrate their spam on a fixed set of victims —
    # realistic attacker behaviour and avoids the trivial "uniform spam
    # diluted across all victims never trips any rate limit" scenario.
    target_victims: List[str] = field(default_factory=list)
    # Tick number at which a dead peer was evicted by some observer, or
    # at which an adversarial peer was first banned by some victim. -1 if
    # not yet observed.
    first_eviction_tick: int = -1
    first_ban_tick: int = -1


class SimNetwork:
    def __init__(self, scenario: ChaosScenario) -> None:
        self._scenario = scenario
        self._clock = [0.0]
        self._rng = _random_mod.Random(scenario.seed)
        self._tick_num = 0
        self._peers: Dict[str, SimPeer] = {}

    # ---- setup ------------------------------------------------------------

    def setup(self) -> None:
        s = self._scenario
        clock_fn = lambda: self._clock[0]

        # Create peers with roles assigned first-come-first-served.
        role_counter = {"adversarial": 0, "dead": 0}
        for i in range(s.node_count):
            is_adv = role_counter["adversarial"] < s.adversarial_count
            is_dead = (
                not is_adv
                and role_counter["dead"] < s.dead_peer_count
            )
            if is_adv:
                role_counter["adversarial"] += 1
            elif is_dead:
                role_counter["dead"] += 1

            peer = SimPeer(
                peer_id=f"peer-{i:03d}",
                is_adversarial=is_adv,
                is_dead=is_dead,
                liveness=LivenessMonitor(
                    ping_interval_sec=s.ping_interval_sec,
                    dead_threshold=3,
                    clock=clock_fn,
                ),
                rate_limiter=RateLimiter(
                    limits={
                        "dht": RateLimit(
                            max_per_window=s.dht_max_per_window,
                            window_sec=s.dht_window_sec,
                        ),
                    },
                    throttle_duration_sec=s.throttle_duration_sec,
                    ban_duration_sec=s.ban_duration_sec,
                    violations_for_ban=s.violations_for_ban,
                    clock=clock_fn,
                ),
            )
            self._peers[peer.peer_id] = peer

        # Everyone tracks everyone (minus self).
        for observer in self._peers.values():
            for target in self._peers.values():
                if target.peer_id != observer.peer_id:
                    observer.liveness.register(target.peer_id)

        # Each adversary picks a small victim set at setup and concentrates
        # spam on those peers. Models targeted attack behaviour and makes
        # the rate-limit trip condition reliably hit across a short run.
        non_adversary_ids = [
            p.peer_id for p in self._peers.values() if not p.is_adversarial
        ]
        for adv in self._peers.values():
            if adv.is_adversarial and non_adversary_ids:
                victim_count = min(3, len(non_adversary_ids))
                adv.target_victims = self._rng.sample(
                    non_adversary_ids, victim_count
                )

    # ---- main run loop ----------------------------------------------------

    def run(self) -> ChaosReport:
        report = ChaosReport(scenario=self._scenario)
        s = self._scenario

        for tick in range(s.ticks):
            self._tick_num = tick
            self._advance_clock_one_interval()

            self._tick_liveness(report)
            self._tick_requests(report)
            self._tick_churn()

        self._finalise_report(report)
        return report

    def _advance_clock_one_interval(self) -> None:
        self._clock[0] += self._scenario.ping_interval_sec

    def _tick_liveness(self, report: ChaosReport) -> None:
        """Run one ping-round across all active peers.

        For each active observer: call tick(), send pings to the due list,
        deliver pongs from non-dead targets, note evictions.
        """
        for observer in self._peers.values():
            if not observer.active:
                continue
            result = observer.liveness.tick()

            for target_id in result.due_for_ping:
                target = self._peers.get(target_id)
                if target is None or not target.active:
                    continue
                observer.liveness.record_ping_sent(target_id)
                # Non-dead, active targets pong immediately. Dead peers
                # silently drop the ping.
                if not target.is_dead:
                    observer.liveness.record_pong_received(target_id)

            for evicted_id in result.evicted:
                evicted = self._peers.get(evicted_id)
                if (
                    evicted is not None
                    and evicted.is_dead
                    and evicted.first_eviction_tick == -1
                ):
                    evicted.first_eviction_tick = self._tick_num

    def _tick_requests(self, report: ChaosReport) -> None:
        """Fire one round of simulated DHT requests.

        Legitimate peers fire a few per tick; adversarial peers fire many.
        Each request targets a uniformly-random active peer (not self) and
        is resolved by that peer's rate limiter.
        """
        s = self._scenario
        active_ids = [p.peer_id for p in self._peers.values() if p.active]
        if len(active_ids) < 2:
            return

        for source in list(self._peers.values()):
            if not source.active:
                continue

            if source.is_adversarial:
                # Concentrate on the fixed victim set — any victim that is
                # currently active receives the spam.
                live_victims = [
                    vid for vid in source.target_victims
                    if self._peers[vid].active
                ]
                if not live_victims:
                    continue
                for _ in range(s.adversarial_requests_per_tick):
                    target_id = self._rng.choice(live_victims)
                    target = self._peers[target_id]
                    outcome = target.rate_limiter.check_and_consume(
                        source.peer_id, "dht"
                    )
                    self._record_outcome(source, outcome, report)
            else:
                for _ in range(s.legitimate_requests_per_tick):
                    target_id = self._rng.choice(active_ids)
                    if target_id == source.peer_id:
                        continue
                    target = self._peers[target_id]
                    outcome = target.rate_limiter.check_and_consume(
                        source.peer_id, "dht"
                    )
                    self._record_outcome(source, outcome, report)

    def _record_outcome(
        self,
        source: SimPeer,
        outcome: RateLimitResult,
        report: ChaosReport,
    ) -> None:
        allowed = outcome is RateLimitResult.ALLOWED
        is_reject = outcome in (
            RateLimitResult.OVER_LIMIT,
            RateLimitResult.THROTTLED,
            RateLimitResult.BANNED,
        )

        if source.is_adversarial:
            report.adversarial_requests_total += 1
            if allowed:
                report.adversarial_requests_allowed += 1
            elif is_reject:
                report.adversarial_requests_rejected += 1
            if (
                outcome is RateLimitResult.BANNED
                and source.first_ban_tick == -1
            ):
                source.first_ban_tick = self._tick_num
        else:
            report.legit_requests_total += 1
            if allowed:
                report.legit_requests_allowed += 1
            elif is_reject:
                report.legit_requests_rejected += 1

    def _tick_churn(self) -> None:
        """Randomly deactivate/reactivate a fraction of legitimate
        non-dead peers per tick. Adversarial and dead peers do not churn
        (adversaries keep attacking; dead peers stay dead so the eviction
        check is meaningful)."""
        s = self._scenario
        if s.churn_fraction_per_tick <= 0:
            return
        churnable = [
            p for p in self._peers.values() if not p.is_adversarial and not p.is_dead
        ]
        n_to_flip = max(1, int(len(churnable) * s.churn_fraction_per_tick))
        self._rng.shuffle(churnable)
        for p in churnable[:n_to_flip]:
            p.active = not p.active

    def _finalise_report(self, report: ChaosReport) -> None:
        for p in self._peers.values():
            if p.is_dead:
                if p.first_eviction_tick >= 0:
                    report.dead_peers_evicted += 1
                    report.dead_peer_ticks_to_eviction.append(
                        p.first_eviction_tick
                    )
                else:
                    report.dead_peers_still_alive += 1
            if p.is_adversarial:
                if p.first_ban_tick >= 0:
                    report.adversarial_peers_banned += 1
                    report.adversary_ticks_to_ban.append(p.first_ban_tick)
                else:
                    report.adversarial_peers_not_banned += 1
