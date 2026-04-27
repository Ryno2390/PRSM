"""Phase 3.x.6 Task 5 — trust adapter unit tests.

Covers the four PRSM-original adapters layered on top of the vendored
Parallax scheduler:
  - AnchorVerifyAdapter   (Phase-1 input filter)
  - TierGateAdapter       (Phase-2 pre-route filter)
  - StakeWeightedTrustAdapter (rescales ProfileSnapshot.layer_latency_ms)
  - ConsensusMismatchHook (post-route, samples redundant exec → Phase 7.1)
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Dict, List, Optional

import pytest

from prsm.compute.parallax_scheduling.prsm_request_router import (
    GPUChain,
    InMemoryProfileSource,
    ProfileSnapshot,
    ProfileSource,
)
from prsm.compute.parallax_scheduling.prsm_types import (
    TIER_ATTESTATION_NONE,
    ParallaxGPU,
)
from prsm.compute.parallax_scheduling.trust_adapter import (
    DEFAULT_FULLY_STAKED_THRESHOLD,
    DEFAULT_HARDWARE_TIER_PREFIXES,
    MIN_STAKE_FOR_PARTICIPATION,
    AnchorLookup,
    AnchorVerifyAdapter,
    ChallengeRecord,
    ChallengeSubmitter,
    ConsensusMismatchHook,
    StakeLookup,
    StakeWeightedTrustAdapter,
    TierGateAdapter,
    TierGateRejected,
    TrustStack,
    is_hardware_attestation,
)
from prsm.compute.tee.models import HARDWARE_TEE_TYPES, PrivacyLevel


# ──────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────


def _gpu(
    node_id: str,
    *,
    region: str = "us-east",
    layer_capacity: int = 8,
    stake_amount: int = DEFAULT_FULLY_STAKED_THRESHOLD,
    tier_attestation: str = "tier-sgx",
    tflops_fp16: float = 100.0,
    memory_gb: float = 80.0,
    memory_bandwidth_gbps: float = 2000.0,
) -> ParallaxGPU:
    return ParallaxGPU(
        node_id=node_id,
        region=region,
        layer_capacity=layer_capacity,
        stake_amount=stake_amount,
        tier_attestation=tier_attestation,
        tflops_fp16=tflops_fp16,
        memory_gb=memory_gb,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
    )


class FakeAnchor:
    """In-memory AnchorLookup."""

    def __init__(self, registered: Optional[Dict[str, str]] = None):
        self.registered: Dict[str, str] = dict(registered or {})

    def lookup(self, node_id: str) -> Optional[str]:
        return self.registered.get(node_id)


class FakeStakeLookup:
    """In-memory StakeLookup."""

    def __init__(self, stakes: Optional[Dict[str, int]] = None):
        self.stakes: Dict[str, int] = dict(stakes or {})

    def get_stake(self, node_id: str) -> int:
        return self.stakes.get(node_id, 0)


class RecordingSubmitter:
    """ChallengeSubmitter test-double that records every dispatch."""

    def __init__(self):
        self.records: List[ChallengeRecord] = []
        self.raise_on_call = False

    def __call__(self, record: ChallengeRecord) -> None:
        if self.raise_on_call:
            raise RuntimeError("simulated submitter outage")
        self.records.append(record)


def _chain(stages=("a", "b"), region: str = "us-east") -> GPUChain:
    return GPUChain(
        request_id="req-1",
        region=region,
        stages=stages,
        layer_ranges=tuple((i, i + 1) for i, _ in enumerate(stages)),
        total_latency_ms=10.0,
        stale_profile_count=0,
    )


# ──────────────────────────────────────────────────────────────────────────
# Adapter A — AnchorVerifyAdapter
# ──────────────────────────────────────────────────────────────────────────


class TestAnchorVerifyAdapter:
    def test_construction_rejects_missing_anchor(self):
        with pytest.raises(RuntimeError, match="anchor"):
            AnchorVerifyAdapter(anchor=None)  # type: ignore[arg-type]

    def test_construction_rejects_anchor_without_lookup(self):
        class Bad:
            pass

        with pytest.raises(RuntimeError, match="lookup"):
            AnchorVerifyAdapter(anchor=Bad())  # type: ignore[arg-type]

    def test_filter_keeps_only_registered_gpus(self):
        anchor = FakeAnchor({"alice": "pk-alice", "carol": "pk-carol"})
        adapter = AnchorVerifyAdapter(anchor=anchor)
        gpus = [_gpu("alice"), _gpu("bob"), _gpu("carol")]

        out = adapter.filter(gpus)

        assert {g.node_id for g in out} == {"alice", "carol"}

    def test_filter_excludes_empty_pubkey(self):
        anchor = FakeAnchor({"alice": "", "bob": "pk-bob"})
        adapter = AnchorVerifyAdapter(anchor=anchor)

        out = adapter.filter([_gpu("alice"), _gpu("bob")])

        assert [g.node_id for g in out] == ["bob"]

    def test_filter_returns_empty_when_no_match(self):
        adapter = AnchorVerifyAdapter(anchor=FakeAnchor({}))

        out = adapter.filter([_gpu("alice"), _gpu("bob")])

        assert out == []

    def test_filter_propagates_anchor_exception(self):
        class BoomAnchor:
            def lookup(self, node_id: str) -> Optional[str]:
                raise ConnectionError("anchor RPC down")

        adapter = AnchorVerifyAdapter(anchor=BoomAnchor())

        with pytest.raises(ConnectionError):
            adapter.filter([_gpu("alice")])

    def test_filter_preserves_input_order(self):
        anchor = FakeAnchor({"a": "pk", "b": "pk", "c": "pk"})
        adapter = AnchorVerifyAdapter(anchor=anchor)
        gpus = [_gpu("c"), _gpu("a"), _gpu("b")]

        out = adapter.filter(gpus)

        assert [g.node_id for g in out] == ["c", "a", "b"]


# ──────────────────────────────────────────────────────────────────────────
# Adapter B — TierGateAdapter
# ──────────────────────────────────────────────────────────────────────────


class TestIsHardwareAttestation:
    def test_recognizes_each_hardware_tee_type(self):
        for tee in HARDWARE_TEE_TYPES:
            assert is_hardware_attestation(f"tier-{tee}") is True

    def test_rejects_tier_none(self):
        assert is_hardware_attestation(TIER_ATTESTATION_NONE) is False

    def test_rejects_empty_string(self):
        assert is_hardware_attestation("") is False

    def test_rejects_unknown_tier(self):
        assert is_hardware_attestation("tier-bogus") is False

    def test_custom_prefix_set(self):
        custom = frozenset({"tier-experimental"})
        assert is_hardware_attestation("tier-experimental", hardware_prefixes=custom) is True
        assert is_hardware_attestation("tier-sgx", hardware_prefixes=custom) is False


class TestTierGateAdapter:
    def test_privacy_none_returns_all_unfiltered(self):
        adapter = TierGateAdapter()
        gpus = [
            _gpu("a", tier_attestation="tier-sgx"),
            _gpu("b", tier_attestation=TIER_ATTESTATION_NONE),
        ]

        out = adapter.filter(gpus, PrivacyLevel.NONE)

        assert {g.node_id for g in out} == {"a", "b"}

    @pytest.mark.parametrize(
        "level",
        [PrivacyLevel.STANDARD, PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM],
    )
    def test_non_none_level_filters_to_hardware_only(self, level):
        adapter = TierGateAdapter()
        gpus = [
            _gpu("a", tier_attestation="tier-sgx"),
            _gpu("b", tier_attestation="tier-tdx"),
            _gpu("c", tier_attestation=TIER_ATTESTATION_NONE),
        ]

        out = adapter.filter(gpus, level)

        assert {g.node_id for g in out} == {"a", "b"}

    def test_raises_when_no_hardware_gpu(self):
        adapter = TierGateAdapter()
        gpus = [
            _gpu("a", tier_attestation=TIER_ATTESTATION_NONE),
            _gpu("b", tier_attestation=TIER_ATTESTATION_NONE),
        ]

        with pytest.raises(TierGateRejected, match="hardware-TEE"):
            adapter.filter(gpus, PrivacyLevel.HIGH)

    def test_raises_on_empty_pool_for_non_none_level(self):
        adapter = TierGateAdapter()
        with pytest.raises(TierGateRejected):
            adapter.filter([], PrivacyLevel.STANDARD)

    def test_empty_pool_passes_through_for_privacy_none(self):
        adapter = TierGateAdapter()
        assert adapter.filter([], PrivacyLevel.NONE) == []

    def test_custom_hardware_prefixes_override_defaults(self):
        adapter = TierGateAdapter(hardware_prefixes=frozenset({"tier-custom"}))
        gpus = [
            _gpu("a", tier_attestation="tier-sgx"),  # rejected — not in custom set
            _gpu("b", tier_attestation="tier-custom"),
        ]

        out = adapter.filter(gpus, PrivacyLevel.STANDARD)

        assert [g.node_id for g in out] == ["b"]

    def test_default_prefixes_constant_matches_hardware_tee_types(self):
        expected = {f"tier-{t}" for t in HARDWARE_TEE_TYPES}
        assert set(DEFAULT_HARDWARE_TIER_PREFIXES) == expected


# ──────────────────────────────────────────────────────────────────────────
# Adapter C — StakeWeightedTrustAdapter
# ──────────────────────────────────────────────────────────────────────────


class TestStakeWeightedTrustAdapter:
    def _make(
        self,
        *,
        snapshots: Optional[Dict[str, ProfileSnapshot]] = None,
        stakes: Optional[Dict[str, int]] = None,
        threshold: int = DEFAULT_FULLY_STAKED_THRESHOLD,
        min_stake: int = MIN_STAKE_FOR_PARTICIPATION,
    ):
        inner = InMemoryProfileSource(snapshots=snapshots or {})
        adapter = StakeWeightedTrustAdapter(
            inner=inner,
            stake_lookup=FakeStakeLookup(stakes),
            fully_staked_threshold=threshold,
            min_stake_for_participation=min_stake,
        )
        return inner, adapter

    def test_construction_rejects_zero_threshold(self):
        with pytest.raises(ValueError, match="fully_staked_threshold"):
            StakeWeightedTrustAdapter(
                inner=InMemoryProfileSource(),
                stake_lookup=FakeStakeLookup(),
                fully_staked_threshold=0,
            )

    def test_construction_rejects_negative_min_stake(self):
        with pytest.raises(ValueError, match="min_stake"):
            StakeWeightedTrustAdapter(
                inner=InMemoryProfileSource(),
                stake_lookup=FakeStakeLookup(),
                min_stake_for_participation=-1,
            )

    def test_construction_rejects_missing_stake_lookup(self):
        with pytest.raises(RuntimeError, match="stake_lookup"):
            StakeWeightedTrustAdapter(
                inner=InMemoryProfileSource(),
                stake_lookup=None,  # type: ignore[arg-type]
            )

    def test_zero_stake_returns_none(self):
        snap = ProfileSnapshot("alice", 10.0, {"bob": 1.0}, 100.0)
        _, adapter = self._make(snapshots={"alice": snap}, stakes={"alice": 0})

        assert adapter.get_snapshot("alice") is None

    def test_below_min_stake_returns_none(self):
        snap = ProfileSnapshot("alice", 10.0, {"bob": 1.0}, 100.0)
        _, adapter = self._make(
            snapshots={"alice": snap},
            stakes={"alice": 4},
            min_stake=5,
        )

        assert adapter.get_snapshot("alice") is None

    def test_full_stake_returns_unmodified_latency(self):
        snap = ProfileSnapshot("alice", 10.0, {"bob": 1.0}, 100.0)
        _, adapter = self._make(
            snapshots={"alice": snap},
            stakes={"alice": DEFAULT_FULLY_STAKED_THRESHOLD},
        )

        out = adapter.get_snapshot("alice")

        assert out is not None
        assert out.layer_latency_ms == pytest.approx(10.0)
        assert out.rtt_to_peers == {"bob": 1.0}
        assert out.timestamp_unix == 100.0

    def test_above_full_stake_caps_at_one_x(self):
        snap = ProfileSnapshot("alice", 10.0, {}, 100.0)
        _, adapter = self._make(
            snapshots={"alice": snap},
            stakes={"alice": DEFAULT_FULLY_STAKED_THRESHOLD * 100},
        )

        out = adapter.get_snapshot("alice")
        assert out is not None
        assert out.layer_latency_ms == pytest.approx(10.0)

    def test_half_stake_doubles_latency(self):
        snap = ProfileSnapshot("alice", 10.0, {}, 100.0)
        threshold = 1_000_000
        _, adapter = self._make(
            snapshots={"alice": snap},
            stakes={"alice": 500_000},
            threshold=threshold,
        )

        out = adapter.get_snapshot("alice")
        assert out is not None
        assert out.layer_latency_ms == pytest.approx(20.0)

    def test_quarter_stake_quadruples_latency(self):
        snap = ProfileSnapshot("alice", 10.0, {}, 100.0)
        threshold = 1_000_000
        _, adapter = self._make(
            snapshots={"alice": snap},
            stakes={"alice": 250_000},
            threshold=threshold,
        )

        out = adapter.get_snapshot("alice")
        assert out is not None
        assert out.layer_latency_ms == pytest.approx(40.0)

    def test_rtt_to_peers_not_rescaled(self):
        snap = ProfileSnapshot("alice", 10.0, {"bob": 5.0, "carol": 7.0}, 100.0)
        threshold = 1_000_000
        _, adapter = self._make(
            snapshots={"alice": snap},
            stakes={"alice": 500_000},
            threshold=threshold,
        )

        out = adapter.get_snapshot("alice")
        assert out is not None
        # Latency rescaled, RTT preserved.
        assert out.layer_latency_ms == pytest.approx(20.0)
        assert out.rtt_to_peers == {"bob": 5.0, "carol": 7.0}

    def test_inner_returns_none_propagates(self):
        _, adapter = self._make(
            snapshots={},
            stakes={"alice": DEFAULT_FULLY_STAKED_THRESHOLD},
        )
        assert adapter.get_snapshot("alice") is None

    def test_unknown_node_returns_none(self):
        _, adapter = self._make(snapshots={}, stakes={})
        assert adapter.get_snapshot("ghost") is None

    def test_returned_snapshot_is_independent_copy(self):
        snap = ProfileSnapshot("alice", 10.0, {"bob": 1.0}, 100.0)
        _, adapter = self._make(
            snapshots={"alice": snap},
            stakes={"alice": DEFAULT_FULLY_STAKED_THRESHOLD},
        )

        out = adapter.get_snapshot("alice")
        assert out is not None
        # Mutating the returned RTT dict must not affect the inner source.
        out.rtt_to_peers["mallory"] = 999.0
        again = adapter.get_snapshot("alice")
        assert again is not None
        assert "mallory" not in again.rtt_to_peers


# ──────────────────────────────────────────────────────────────────────────
# Adapter D — ConsensusMismatchHook
# ──────────────────────────────────────────────────────────────────────────


class TestConsensusMismatchHookSampling:
    def test_rejects_invalid_sample_rate(self):
        sub = RecordingSubmitter()
        with pytest.raises(ValueError, match="sample_rate"):
            ConsensusMismatchHook(submitter=sub, sample_rate=1.5)
        with pytest.raises(ValueError, match="sample_rate"):
            ConsensusMismatchHook(submitter=sub, sample_rate=-0.1)

    def test_rejects_missing_submitter(self):
        with pytest.raises(RuntimeError, match="submitter"):
            ConsensusMismatchHook(submitter=None)  # type: ignore[arg-type]

    def test_sample_rate_zero_never_samples(self):
        hook = ConsensusMismatchHook(submitter=RecordingSubmitter(), sample_rate=0.0)
        for i in range(100):
            assert hook.should_sample_redundant(f"req-{i}") is False

    def test_sample_rate_one_always_samples(self):
        hook = ConsensusMismatchHook(submitter=RecordingSubmitter(), sample_rate=1.0)
        for i in range(100):
            assert hook.should_sample_redundant(f"req-{i}") is True

    def test_sample_rate_is_deterministic(self):
        hook = ConsensusMismatchHook(submitter=RecordingSubmitter(), sample_rate=0.5)
        decisions = [hook.should_sample_redundant("req-fixed") for _ in range(20)]
        assert all(d == decisions[0] for d in decisions)

    def test_sample_rate_distribution_close_to_target(self):
        """At 10% sampling on 1000 requests, expect ~100 ± reasonable slack."""
        hook = ConsensusMismatchHook(submitter=RecordingSubmitter(), sample_rate=0.1)
        sampled = sum(
            1 for i in range(1000) if hook.should_sample_redundant(f"req-{i}")
        )
        # sha256 hash mod 1e6 — distribution is uniform; allow generous slack.
        assert 50 <= sampled <= 200


class TestConsensusMismatchHookCompare:
    def test_match_returns_none_no_dispatch(self):
        sub = RecordingSubmitter()
        hook = ConsensusMismatchHook(submitter=sub, sample_rate=1.0)
        out = hook.compare_and_challenge(
            request_id="req-1",
            primary_output=b"hello",
            secondary_output=b"hello",
            primary_chain=_chain(("a", "b")),
            secondary_chain=_chain(("c", "d")),
        )
        assert out is None
        assert sub.records == []

    def test_mismatch_dispatches_record(self):
        sub = RecordingSubmitter()
        hook = ConsensusMismatchHook(submitter=sub, sample_rate=1.0)
        out = hook.compare_and_challenge(
            request_id="req-1",
            primary_output=b"answer-A",
            secondary_output=b"answer-B",
            primary_chain=_chain(("a", "b")),
            secondary_chain=_chain(("c", "d")),
        )
        assert out is not None
        assert len(sub.records) == 1
        rec = sub.records[0]
        assert rec.request_id == "req-1"
        assert rec.primary_chain_stages == ("a", "b")
        assert rec.secondary_chain_stages == ("c", "d")
        assert rec.primary_output_hash == hashlib.sha256(b"answer-A").hexdigest()
        assert rec.secondary_output_hash == hashlib.sha256(b"answer-B").hexdigest()

    def test_mismatch_returns_record_even_if_submitter_raises(self):
        """Submitter outage must not propagate — the record itself is the
        audit trail and ops will replay later."""
        sub = RecordingSubmitter()
        sub.raise_on_call = True
        hook = ConsensusMismatchHook(submitter=sub, sample_rate=1.0)

        out = hook.compare_and_challenge(
            request_id="req-1",
            primary_output=b"x",
            secondary_output=b"y",
            primary_chain=_chain(("a",)),
            secondary_chain=_chain(("b",)),
        )

        assert out is not None
        assert out.request_id == "req-1"
        # Submitter raised, so .records stays empty — but no exception leaked.

    def test_chain_stages_preserved_in_record(self):
        sub = RecordingSubmitter()
        hook = ConsensusMismatchHook(submitter=sub, sample_rate=1.0)
        primary = _chain(("alice", "bob", "carol"))
        secondary = _chain(("dan", "eve"))

        rec = hook.compare_and_challenge(
            request_id="req-7",
            primary_output=b"a",
            secondary_output=b"b",
            primary_chain=primary,
            secondary_chain=secondary,
        )

        assert rec is not None
        assert rec.primary_chain_stages == ("alice", "bob", "carol")
        assert rec.secondary_chain_stages == ("dan", "eve")


# ──────────────────────────────────────────────────────────────────────────
# TrustStack composition
# ──────────────────────────────────────────────────────────────────────────


class TestTrustStack:
    def _stack(
        self,
        *,
        registered: Dict[str, str],
        stakes: Dict[str, int],
        snapshots: Optional[Dict[str, ProfileSnapshot]] = None,
        sample_rate: float = 0.0,
    ) -> TrustStack:
        inner = InMemoryProfileSource(snapshots=snapshots or {})
        return TrustStack(
            anchor_verify=AnchorVerifyAdapter(anchor=FakeAnchor(registered)),
            tier_gate=TierGateAdapter(),
            profile_source=StakeWeightedTrustAdapter(
                inner=inner,
                stake_lookup=FakeStakeLookup(stakes),
            ),
            consensus_hook=ConsensusMismatchHook(
                submitter=RecordingSubmitter(),
                sample_rate=sample_rate,
            ),
        )

    def test_filter_pool_applies_anchor_and_stake_admission(self):
        """filter_pool enforces BOTH anchor verification AND stake
        eligibility. The latter prevents the upstream router's
        roofline fallback from routing to a zero-stake GPU based on
        advertised hardware specs alone (caught by Phase 3.x.6 Task 7
        E2E)."""
        stack = self._stack(
            registered={"alice": "pk", "bob": "pk"},
            stakes={"alice": 1_000_000, "bob": 0},
        )
        gpus = [_gpu("alice"), _gpu("bob"), _gpu("ghost")]

        out = stack.filter_pool(gpus)

        # ghost is unregistered → excluded by anchor.
        # bob has zero stake → excluded by stake-admission.
        # Only alice survives both filters.
        assert {g.node_id for g in out} == {"alice"}

    def test_filter_pool_passes_through_when_profile_lacks_is_eligible(self):
        """If the wired profile_source doesn't expose ``is_eligible``,
        only anchor filtering applies (back-compat with custom
        ProfileSource implementations)."""
        # Build a TrustStack whose profile_source is a bare
        # InMemoryProfileSource (no is_eligible method).
        stack = TrustStack(
            anchor_verify=AnchorVerifyAdapter(
                anchor=FakeAnchor({"alice": "pk", "bob": "pk"})
            ),
            tier_gate=TierGateAdapter(),
            profile_source=InMemoryProfileSource(),
            consensus_hook=ConsensusMismatchHook(
                submitter=RecordingSubmitter(), sample_rate=0.0
            ),
        )
        out = stack.filter_pool([_gpu("alice"), _gpu("bob"), _gpu("ghost")])
        # ghost excluded by anchor; alice + bob both pass (no stake gate).
        assert {g.node_id for g in out} == {"alice", "bob"}

    def test_filter_for_request_applies_tier_gate(self):
        stack = self._stack(registered={}, stakes={})
        gpus = [
            _gpu("a", tier_attestation="tier-sgx"),
            _gpu("b", tier_attestation=TIER_ATTESTATION_NONE),
        ]

        out = stack.filter_for_request(gpus, PrivacyLevel.STANDARD)

        assert [g.node_id for g in out] == ["a"]

    def test_filter_for_request_raises_on_no_hardware(self):
        stack = self._stack(registered={}, stakes={})
        gpus = [_gpu("a", tier_attestation=TIER_ATTESTATION_NONE)]

        with pytest.raises(TierGateRejected):
            stack.filter_for_request(gpus, PrivacyLevel.HIGH)

    def test_profile_source_is_stake_weighted(self):
        snap = ProfileSnapshot("alice", 10.0, {}, 100.0)
        stack = self._stack(
            registered={"alice": "pk"},
            stakes={"alice": DEFAULT_FULLY_STAKED_THRESHOLD // 2},
            snapshots={"alice": snap},
        )

        out = stack.profile_source.get_snapshot("alice")
        assert out is not None
        assert out.layer_latency_ms == pytest.approx(20.0)

    def test_full_pipeline_anchor_then_tier_then_profile(self):
        """Compose all three filtering adapters end-to-end."""
        snap_alice = ProfileSnapshot("alice", 10.0, {}, 100.0)
        snap_bob = ProfileSnapshot("bob", 10.0, {}, 100.0)
        stack = self._stack(
            registered={"alice": "pk", "bob": "pk"},  # ghost excluded
            stakes={
                "alice": DEFAULT_FULLY_STAKED_THRESHOLD,
                "bob": 0,  # bob's snapshot will return None
            },
            snapshots={"alice": snap_alice, "bob": snap_bob},
        )
        pool = [
            _gpu("alice", tier_attestation="tier-sgx"),
            _gpu("bob", tier_attestation="tier-tdx"),
            _gpu("ghost", tier_attestation="tier-sgx"),
        ]

        # Step 1: pool-level filter → ghost excluded by anchor; bob
        # excluded by stake-admission (zero stake). Only alice remains.
        post_anchor = stack.filter_pool(pool)
        assert {g.node_id for g in post_anchor} == {"alice"}

        # Step 2: per-request tier filter (HIGH privacy → all hardware).
        post_tier = stack.filter_for_request(post_anchor, PrivacyLevel.HIGH)
        assert {g.node_id for g in post_tier} == {"alice"}

        # Step 3: profile lookups — alice gets full speed, bob (zero
        # stake) returns None even if queried directly.
        assert stack.profile_source.get_snapshot("alice") is not None
        assert stack.profile_source.get_snapshot("bob") is None
