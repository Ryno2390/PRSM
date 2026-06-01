"""Sprint 904 — staking yields eliminated (PRSM_Tokenomics.md §10 #7).

The live StakingManager paid a fixed 5%-of-principal APY funded by
freshly MINTED FTNS (`mint_tokens(reason="staking_rewards")`). That is a
discretionary, inflationary staking return — the textbook Howey
"expectation of profit from the efforts of others" flag — and §10 #7
forbids it ("no discretionary inflation-based rewards"). The deployed v1
emission split (50/30/20) has no staker-yield pool to fund it from
either, so the yield was minting tokens from nowhere.

sp904 eliminates it: `reward_rate_annual` defaults to 0.0,
`calculate_rewards` accrues nothing, and `claim_rewards` mints nothing.
Stake/unstake/slash mechanics and the legitimate slash-appeal refund
mint are unaffected. Staking confers utility (lock-based discounts /
priority access), not a token yield.
"""
from __future__ import annotations

from prsm.economy.tokenomics.staking_manager import StakingConfig


def test_default_reward_rate_is_zero():
    """The config default must be 0.0 — no staking yield out of the box."""
    assert StakingConfig().reward_rate_annual == 0.0


def test_claim_rewards_never_mints_source_pinned():
    """Pin the source intent: claim_rewards must not contain a
    `staking_rewards` mint path. (Behavioral coverage — that a fully
    aged stake yields 0 and mints nothing — lives in
    test_staking_incentives with the async-DB harness.)"""
    import inspect
    from prsm.economy.tokenomics import staking_manager

    src = inspect.getsource(staking_manager.StakingManager.claim_rewards)
    # Strip comment lines so prose mentioning the removed path doesn't
    # trip the check; assert no actual mint CALL remains.
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "mint_tokens(" not in code, (
        "claim_rewards still calls mint_tokens — the Howey-flag "
        "inflationary staking yield must be removed (sp904)"
    )
