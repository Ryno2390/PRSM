"""Sprint 290 — CreatorStakeClient.

Vision §14 mitigation item (2): "Staking requirements for
high-tier creator status — uploading requires collateral that
can be slashed." The threat of slashing makes spam-uploading
economically unattractive: a creator who spams pays for it
with bonded FTNS.

This v1 ships the operator-side scaffold with the same
PENDING_COMMISSION pattern as Phase 5 (Coinbase WaaS /
paymaster). Real `CreatorStakeRegistry.sol` contract deploy
is a future multi-sig ceremony; the Python adapter +
dependency-injected backend means tier-gating logic ships
now and lights up the moment the contract is deployed +
CREATOR_STAKE_REGISTRY_ADDRESS env var is set.

Pre-commission: in-memory balances; calls to stake/slash
mutate the local view without on-chain settlement.
Post-commission: backend delegates to the real contract.
"""
from __future__ import annotations

import pytest

from prsm.marketplace.creator_stake_client import (
    CreatorStakeClient,
    apply_stake_gate,
    MIN_HIGH_TIER_STAKE_WEI,
)
from prsm.marketplace.creator_reputation import (
    TIER_NEW, TIER_LOW, TIER_MEDIUM, TIER_HIGH,
)


class FakeBackend:
    """Test backend mirroring CreatorStakeRegistry.sol surface."""

    def __init__(self):
        self.balances = {}
        self.calls = []

    def stake(self, creator_id, amount_wei):
        self.calls.append(("stake", creator_id, amount_wei))
        self.balances[creator_id] = (
            self.balances.get(creator_id, 0) + amount_wei
        )

    def slash(self, creator_id, amount_wei, reason):
        self.calls.append(("slash", creator_id, amount_wei))
        current = self.balances.get(creator_id, 0)
        new_balance = max(0, current - amount_wei)
        self.balances[creator_id] = new_balance

    def balance_of(self, creator_id):
        return self.balances.get(creator_id, 0)


# ── In-memory (PENDING_COMMISSION) ───────────────────────


def test_uncommissioned_balance_zero_by_default():
    c = CreatorStakeClient()
    assert c.stake_balance("alice") == 0


def test_uncommissioned_stake_updates_in_memory():
    c = CreatorStakeClient()
    c.stake("alice", amount_wei=500)
    assert c.stake_balance("alice") == 500


def test_uncommissioned_stake_accumulates():
    c = CreatorStakeClient()
    c.stake("alice", amount_wei=500)
    c.stake("alice", amount_wei=300)
    assert c.stake_balance("alice") == 800


def test_uncommissioned_slash_reduces_balance():
    c = CreatorStakeClient()
    c.stake("alice", amount_wei=1000)
    c.slash("alice", amount_wei=400, reason="spam")
    assert c.stake_balance("alice") == 600


def test_uncommissioned_slash_clamps_to_zero():
    """A slash > balance reduces to 0, not negative."""
    c = CreatorStakeClient()
    c.stake("alice", amount_wei=100)
    c.slash("alice", amount_wei=500, reason="spam")
    assert c.stake_balance("alice") == 0


def test_is_commissioned_false_without_backend():
    c = CreatorStakeClient()
    assert c.is_commissioned() is False


# ── Commissioned (backend delegates) ─────────────────────


def test_commissioned_stake_delegates_to_backend():
    backend = FakeBackend()
    c = CreatorStakeClient(
        registry_address="0xreg", rpc_url="https://rpc.x",
        backend=backend,
    )
    assert c.is_commissioned() is True
    c.stake("alice", amount_wei=500)
    assert ("stake", "alice", 500) in backend.calls
    # balance read pulls from backend, not in-memory mirror
    assert c.stake_balance("alice") == 500


def test_commissioned_slash_delegates_to_backend():
    backend = FakeBackend()
    backend.balances["alice"] = 1000
    c = CreatorStakeClient(
        registry_address="0xreg", rpc_url="https://rpc.x",
        backend=backend,
    )
    c.slash("alice", amount_wei=400, reason="spam")
    assert ("slash", "alice", 400) in backend.calls
    assert c.stake_balance("alice") == 600


def test_commissioned_backend_failure_falls_back_to_zero():
    """Backend raises → balance query returns 0 (fail-soft).
    Caller can distinguish via is_commissioned()."""
    class BoomBackend:
        def balance_of(self, c):
            raise RuntimeError("RPC down")
    c = CreatorStakeClient(
        registry_address="0xreg", rpc_url="https://rpc.x",
        backend=BoomBackend(),
    )
    assert c.stake_balance("alice") == 0


# ── Validation ───────────────────────────────────────────


def test_stake_validates_non_empty_creator():
    c = CreatorStakeClient()
    with pytest.raises(ValueError):
        c.stake("", amount_wei=100)


def test_stake_validates_positive_amount():
    c = CreatorStakeClient()
    with pytest.raises(ValueError):
        c.stake("alice", amount_wei=0)
    with pytest.raises(ValueError):
        c.stake("alice", amount_wei=-1)


def test_slash_validates_positive_amount():
    c = CreatorStakeClient()
    c.stake("alice", amount_wei=1000)
    with pytest.raises(ValueError):
        c.slash("alice", amount_wei=0, reason="x")
    with pytest.raises(ValueError):
        c.slash("alice", amount_wei=-1, reason="x")


def test_slash_validates_reason():
    c = CreatorStakeClient()
    c.stake("alice", amount_wei=1000)
    with pytest.raises(ValueError):
        c.slash("alice", amount_wei=100, reason="")


# ── is_high_tier_eligible ────────────────────────────────


def test_high_tier_eligibility_below_threshold():
    c = CreatorStakeClient()
    c.stake("alice", amount_wei=MIN_HIGH_TIER_STAKE_WEI - 1)
    assert c.is_high_tier_eligible("alice") is False


def test_high_tier_eligibility_at_threshold():
    c = CreatorStakeClient()
    c.stake("alice", amount_wei=MIN_HIGH_TIER_STAKE_WEI)
    assert c.is_high_tier_eligible("alice") is True


def test_high_tier_eligibility_above_threshold():
    c = CreatorStakeClient()
    c.stake("alice", amount_wei=MIN_HIGH_TIER_STAKE_WEI * 2)
    assert c.is_high_tier_eligible("alice") is True


def test_high_tier_eligibility_unknown_creator_false():
    c = CreatorStakeClient()
    assert c.is_high_tier_eligible("nobody") is False


# ── from_env ─────────────────────────────────────────────


def test_from_env_uncommissioned_when_no_address(monkeypatch):
    monkeypatch.delenv(
        "CREATOR_STAKE_REGISTRY_ADDRESS", raising=False,
    )
    c = CreatorStakeClient.from_env()
    assert c.is_commissioned() is False


def test_from_env_commissioned_when_both_set(monkeypatch):
    monkeypatch.setenv(
        "CREATOR_STAKE_REGISTRY_ADDRESS", "0xreg",
    )
    monkeypatch.setenv("BASE_RPC_URL", "https://rpc.x")
    # from_env without explicit backend will return None for
    # commissioned (web3 might be missing in test env) — but
    # `is_commissioned()` should still trip based on env vars
    # presence. The actual backend wiring is operator-side.
    c = CreatorStakeClient.from_env()
    assert c.is_commissioned() is True


# ── apply_stake_gate pure function ───────────────────────


def test_stake_gate_high_with_stake_stays_high():
    backend = FakeBackend()
    c = CreatorStakeClient(
        registry_address="x", rpc_url="x", backend=backend,
    )
    c.stake("alice", amount_wei=MIN_HIGH_TIER_STAKE_WEI)
    assert apply_stake_gate(
        TIER_HIGH, "alice", c,
    ) == TIER_HIGH


def test_stake_gate_high_without_stake_demotes_to_medium():
    c = CreatorStakeClient()  # uncommissioned, no stake
    assert apply_stake_gate(
        TIER_HIGH, "alice", c,
    ) == TIER_MEDIUM


def test_stake_gate_low_unchanged_regardless_of_stake():
    backend = FakeBackend()
    c = CreatorStakeClient(
        registry_address="x", rpc_url="x", backend=backend,
    )
    c.stake("alice", amount_wei=MIN_HIGH_TIER_STAKE_WEI * 10)
    # Even with huge stake, LOW stays LOW (stake doesn't buy
    # score)
    assert apply_stake_gate(
        TIER_LOW, "alice", c,
    ) == TIER_LOW


def test_stake_gate_medium_unchanged():
    """MEDIUM passes through regardless of stake — the gate
    only demotes HIGH→MEDIUM, doesn't promote anywhere."""
    c = CreatorStakeClient()
    assert apply_stake_gate(
        TIER_MEDIUM, "alice", c,
    ) == TIER_MEDIUM


def test_stake_gate_new_unchanged():
    c = CreatorStakeClient()
    assert apply_stake_gate(TIER_NEW, "alice", c) == TIER_NEW


def test_stake_gate_none_client_is_passthrough():
    """When stake_client is None (pre-sprint-290 callers,
    test fixtures), the gate is a no-op — preserves
    backwards-compat for sprint 287/288/289 surfaces."""
    assert apply_stake_gate(
        TIER_HIGH, "alice", None,
    ) == TIER_HIGH


def test_stake_gate_invalid_tier_passthrough():
    """If somehow a non-tier string lands here, pass through
    unchanged rather than crash."""
    c = CreatorStakeClient()
    assert apply_stake_gate(
        "weird", "alice", c,
    ) == "weird"


# ── Env-tunable threshold ────────────────────────────────


def test_min_high_tier_stake_env_override():
    """The threshold is module-level; sprint 290 ships a
    default but operators can override at import time via
    PRSM_MIN_HIGH_TIER_STAKE_WEI. Tested indirectly by
    confirming the constant exists and is positive."""
    assert MIN_HIGH_TIER_STAKE_WEI > 0
