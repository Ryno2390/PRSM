"""Sprint 299 — InsuranceFundTracker + recovery transfer
composer.

Vision §14 mitigation item 2: "Foundation reserves at least
5% of treasury value as a dedicated insurance fund earmarked
for exploit recovery. Public, on-chain verification. Sized to
ensure that even successful exploits do not result in
unrecoverable user losses."

Scope this sprint:
  InsuranceFundTracker — reads on-chain FTNS balance of a
    designated insurance-fund address; compares against
    Foundation Safe total to compute reserve ratio. Public
    read surface (matches §14's "public, on-chain
    verification" requirement).

  compose_recovery_transfer_tx — Safe-uploadable ERC-20
    transfer payload that moves insurance funds to a
    recovery wallet. Composer-only — Foundation Safe 2-of-3
    multisig approval gates execution (same pattern as
    sprint 298 emergency pause).

  PRSM_INSURANCE_FUND_ADDRESS / PRSM_INSURANCE_FUND_TARGET_BPS
    env vars + from_env() factory.

Mirrors sprint 298's safety-first design: every composed tx
carries explorer URL + numbered Safe-UI instructions +
explicit WARNING string.

ERC-20 transfer selector: 0xa9059cbb (transfer(address,uint256)).
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.insurance_fund_tracker import (
    ERC20_TRANSFER_SELECTOR,
    InsuranceFundStatus,
    InsuranceFundTracker,
    encode_erc20_transfer_calldata,
)


# ── ERC-20 transfer calldata encoding ────────────────────


def test_transfer_selector_constant():
    """transfer(address,uint256) selector. Pinned to detect
    spec drift."""
    assert ERC20_TRANSFER_SELECTOR == "0xa9059cbb"


def test_encode_erc20_transfer_calldata():
    """transfer(address recipient, uint256 amount) →
    0xa9059cbb + 32-byte left-padded recipient + 32-byte
    big-endian amount."""
    recipient = "0x" + "ab" * 20  # 40-hex-char address
    amount = 1234
    data = encode_erc20_transfer_calldata(recipient, amount)
    # Starts with selector
    assert data.startswith(ERC20_TRANSFER_SELECTOR)
    # 4 bytes selector + 32 bytes address + 32 bytes amount
    # = 68 bytes = 136 hex chars + 2 for '0x' = 138 chars
    assert len(data) == 138
    # Address right-aligned in 32-byte slot — last 40 hex
    # chars of address field should match recipient
    address_field = data[10:74]  # bytes 4-36 in hex
    assert address_field.endswith("ab" * 20)
    # Amount field is the last 64 hex chars; value=1234=0x4d2
    amount_field = data[74:138]
    assert int(amount_field, 16) == 1234


def test_encode_erc20_transfer_validates_recipient():
    with pytest.raises(ValueError):
        encode_erc20_transfer_calldata("", 100)
    with pytest.raises(ValueError):
        encode_erc20_transfer_calldata("not-an-address", 100)


def test_encode_erc20_transfer_validates_amount():
    with pytest.raises(ValueError):
        encode_erc20_transfer_calldata("0x" + "ab" * 20, 0)
    with pytest.raises(ValueError):
        encode_erc20_transfer_calldata("0x" + "ab" * 20, -1)


# ── InsuranceFundTracker: backend & status ───────────────


class FakeBalanceBackend:
    """Test backend returning canned FTNS balances per
    address."""

    def __init__(self, balances=None):
        self.balances = balances or {}
        self.calls = []

    def balance_of(self, address):
        self.calls.append(address)
        return self.balances.get(address, 0)


def test_tracker_status_default_5pct_target():
    """Default target is 500 bps (5%). 5M FTNS in insurance
    fund / 100M FTNS treasury → 5.0% → target met."""
    backend = FakeBalanceBackend(
        balances={
            "0xfund": 5_000_000 * (10 ** 18),
            "0xtreasury": 100_000_000 * (10 ** 18),
        },
    )
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        backend=backend,
    )
    status = tracker.status()
    assert status.fund_balance_wei == 5_000_000 * (10 ** 18)
    assert status.treasury_balance_wei == (
        100_000_000 * (10 ** 18)
    )
    assert status.reserve_ratio_bps == 500  # exactly 5%
    assert status.target_bps == 500
    assert status.target_met is True


def test_tracker_target_not_met_when_below():
    """Insurance fund holds 2% of treasury → below 5%
    target → target_met=False."""
    backend = FakeBalanceBackend(
        balances={
            "0xfund": 2_000_000 * (10 ** 18),
            "0xtreasury": 100_000_000 * (10 ** 18),
        },
    )
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        backend=backend,
    )
    status = tracker.status()
    assert status.reserve_ratio_bps == 200  # 2%
    assert status.target_met is False


def test_tracker_target_met_when_over():
    """Insurance fund holds 10% — over target."""
    backend = FakeBalanceBackend(
        balances={
            "0xfund": 10_000_000 * (10 ** 18),
            "0xtreasury": 100_000_000 * (10 ** 18),
        },
    )
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        backend=backend,
    )
    status = tracker.status()
    assert status.reserve_ratio_bps == 1000  # 10%
    assert status.target_met is True


def test_tracker_zero_treasury_ratio_zero():
    """Avoid divide-by-zero: empty treasury → ratio=0,
    target_met=False, surfaces a diagnostic."""
    backend = FakeBalanceBackend(balances={})
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        backend=backend,
    )
    status = tracker.status()
    assert status.fund_balance_wei == 0
    assert status.treasury_balance_wei == 0
    assert status.reserve_ratio_bps == 0
    assert status.target_met is False


def test_tracker_custom_target_bps():
    """Operator can configure target to e.g. 10% instead of
    default 5%."""
    backend = FakeBalanceBackend(
        balances={
            "0xfund": 5_000_000 * (10 ** 18),
            "0xtreasury": 100_000_000 * (10 ** 18),
        },
    )
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        target_bps=1000,  # 10%
        backend=backend,
    )
    status = tracker.status()
    # 5% reserve vs 10% target → not met
    assert status.target_met is False


def test_tracker_default_target_500_bps():
    """Default target is 5% (500 bps)."""
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        backend=FakeBalanceBackend(),
    )
    assert tracker.target_bps == 500


# ── Fail-soft on backend exceptions ──────────────────────


def test_status_fail_soft_on_balance_exception():
    """RPC error → status carries error string; balance
    fields are None (callers distinguish 'zero' from 'unknown')."""
    class BoomBackend:
        def balance_of(self, addr):
            raise RuntimeError("RPC down")
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        backend=BoomBackend(),
    )
    status = tracker.status()
    assert status.fund_balance_wei is None
    assert status.treasury_balance_wei is None
    assert status.reserve_ratio_bps is None
    assert status.target_met is False  # unknown != met
    assert status.error is not None
    assert "rpc down" in status.error.lower()


def test_status_partial_failure_one_address():
    """One address read fails → that field is None; other
    succeeds."""
    class PartialBackend:
        def balance_of(self, addr):
            if addr == "0xfund":
                raise RuntimeError("fund read failed")
            return 100_000_000 * (10 ** 18)
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        backend=PartialBackend(),
    )
    status = tracker.status()
    assert status.fund_balance_wei is None
    assert status.treasury_balance_wei == (
        100_000_000 * (10 ** 18)
    )


# ── PENDING_COMMISSION: uncommissioned tracker ───────────


def test_uncommissioned_status_returns_not_configured():
    """No fund_address + no treasury_address → status with
    `commissioned=False`."""
    tracker = InsuranceFundTracker(
        fund_address=None,
        treasury_address=None,
        backend=FakeBalanceBackend(),
    )
    status = tracker.status()
    assert status.commissioned is False


def test_uncommissioned_compose_raises():
    """Composing a recovery transfer requires fund address.
    Missing config → raise."""
    tracker = InsuranceFundTracker(
        fund_address=None,
        treasury_address="0xtreasury",
        backend=FakeBalanceBackend(),
    )
    with pytest.raises(ValueError, match="not configured"):
        tracker.compose_recovery_transfer_tx(
            recipient="0x" + "ab" * 20,
            amount_wei=100,
            reason="exploit recovery",
        )


# ── compose_recovery_transfer_tx ─────────────────────────


def test_compose_recovery_happy_path():
    recipient = "0x" + "ab" * 20
    amount = 1_000_000 * (10 ** 18)
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        ftns_token_address="0x" + "11" * 20,
        chain_id=8453,
        backend=FakeBalanceBackend(),
    )
    tx = tracker.compose_recovery_transfer_tx(
        recipient=recipient,
        amount_wei=amount,
        reason="exploit recovery — 2026-05-12 BSR exploit "
               "post-mortem",
    )
    # to = FTNS token contract (transfer is on the token)
    assert tx["to"] == "0x" + "11" * 20
    assert tx["data"].startswith(ERC20_TRANSFER_SELECTOR)
    assert tx["value"] == "0"
    assert tx["action"] == "recovery_transfer"
    assert "exploit recovery" in tx["reason"].lower()
    # Operator-safety surfaces (mirror sprint 298)
    assert "warning" in tx
    assert "basescan" in tx["explorer_url"].lower()
    assert tx["chain_id"] == 8453


def test_compose_recovery_validates_amount():
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        ftns_token_address="0x" + "11" * 20,
        backend=FakeBalanceBackend(),
    )
    with pytest.raises(ValueError):
        tracker.compose_recovery_transfer_tx(
            recipient="0x" + "ab" * 20,
            amount_wei=0,
            reason="x",
        )
    with pytest.raises(ValueError):
        tracker.compose_recovery_transfer_tx(
            recipient="0x" + "ab" * 20,
            amount_wei=-100,
            reason="x",
        )


def test_compose_recovery_validates_recipient():
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        ftns_token_address="0x" + "11" * 20,
        backend=FakeBalanceBackend(),
    )
    with pytest.raises(ValueError):
        tracker.compose_recovery_transfer_tx(
            recipient="",
            amount_wei=100,
            reason="x",
        )


def test_compose_recovery_requires_reason():
    """Audit trail: recovery transfers MUST carry a reason
    for post-incident review."""
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        ftns_token_address="0x" + "11" * 20,
        backend=FakeBalanceBackend(),
    )
    with pytest.raises(ValueError, match="reason"):
        tracker.compose_recovery_transfer_tx(
            recipient="0x" + "ab" * 20,
            amount_wei=100,
            reason="",
        )


def test_compose_recovery_requires_token_address():
    """Need to know the FTNS token contract to transfer
    from."""
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        ftns_token_address=None,
        backend=FakeBalanceBackend(),
    )
    with pytest.raises(ValueError, match="ftns_token"):
        tracker.compose_recovery_transfer_tx(
            recipient="0x" + "ab" * 20,
            amount_wei=100,
            reason="x",
        )


# ── from_env factory ─────────────────────────────────────


def test_from_env_defaults_to_mainnet_safe_as_treasury(
    monkeypatch,
):
    """Without PRSM_INSURANCE_FUND_ADDRESS, tracker still
    builds with Foundation Safe from networks.py as the
    treasury. Fund address is None → uncommissioned for
    insurance-specific status."""
    monkeypatch.delenv(
        "PRSM_INSURANCE_FUND_ADDRESS", raising=False,
    )
    monkeypatch.delenv("PRSM_NETWORK", raising=False)
    tracker = InsuranceFundTracker.from_env()
    # Treasury defaults to Foundation Safe mainnet address
    assert (
        tracker.treasury_address
        == "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
    )
    # FTNS token from networks.py
    assert tracker.ftns_token_address is not None
    # Fund address None → uncommissioned
    assert tracker.fund_address is None


def test_from_env_with_fund_address(monkeypatch):
    monkeypatch.setenv(
        "PRSM_INSURANCE_FUND_ADDRESS",
        "0x" + "ab" * 20,
    )
    tracker = InsuranceFundTracker.from_env()
    assert tracker.fund_address == "0x" + "ab" * 20


def test_from_env_with_custom_target_bps(monkeypatch):
    monkeypatch.setenv(
        "PRSM_INSURANCE_FUND_TARGET_BPS", "1000",
    )
    tracker = InsuranceFundTracker.from_env()
    assert tracker.target_bps == 1000


def test_from_env_invalid_target_bps_falls_back(monkeypatch):
    """Non-numeric or out-of-range PRSM_INSURANCE_FUND_TARGET
    falls back to default 500. Defense against bad
    deployment config."""
    monkeypatch.setenv(
        "PRSM_INSURANCE_FUND_TARGET_BPS", "not-a-number",
    )
    tracker = InsuranceFundTracker.from_env()
    assert tracker.target_bps == 500


def test_from_env_negative_target_bps_falls_back(monkeypatch):
    monkeypatch.setenv(
        "PRSM_INSURANCE_FUND_TARGET_BPS", "-100",
    )
    tracker = InsuranceFundTracker.from_env()
    assert tracker.target_bps == 500


# ── InsuranceFundStatus.to_dict ──────────────────────────


def test_status_to_dict_full():
    backend = FakeBalanceBackend(
        balances={
            "0xfund": 5_000_000 * (10 ** 18),
            "0xtreasury": 100_000_000 * (10 ** 18),
        },
    )
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        ftns_token_address="0xttt",
        chain_id=8453,
        backend=backend,
    )
    status = tracker.status()
    d = status.to_dict()
    for key in [
        "fund_balance_wei", "treasury_balance_wei",
        "reserve_ratio_bps", "target_bps", "target_met",
        "commissioned", "error", "fund_address",
        "treasury_address",
    ]:
        assert key in d
    assert d["reserve_ratio_bps"] == 500
    assert d["target_met"] is True
