"""Unit tests for StakeManagerClient — Phase 7 Task 4.

These mock Web3 entirely — they validate the client wraps the StakeBond
ABI correctly and handles common error cases. End-to-end happy-path
against a real Hardhat node lives in the Phase 7 Task 7 integration test.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.stake_manager import (
    SLASH_RATE_CRITICAL_BPS,
    SLASH_RATE_PREMIUM_BPS,
    SLASH_RATE_STANDARD_BPS,
    StakeManagerClient,
    StakeRecord,
    StakeStatus,
    TIER_CRITICAL_MIN_WEI,
    TIER_PREMIUM_MIN_WEI,
    TIER_STANDARD_MIN_WEI,
    TransferStatus,
)


@pytest.fixture
def mock_web3():
    with patch("prsm.economy.web3.stake_manager.Web3") as MockWeb3:
        w3_instance = MagicMock()
        MockWeb3.return_value = w3_instance
        MockWeb3.HTTPProvider.return_value = MagicMock()
        MockWeb3.to_checksum_address.side_effect = lambda x: x
        yield w3_instance, MockWeb3


def _make_client(mock_web3, deploy_address="0xStakeBond"):
    w3_instance, _ = mock_web3
    contract = MagicMock()
    w3_instance.eth.contract.return_value = contract

    account = MagicMock()
    account.address = "0xProvider"
    account.key = b"\x22" * 32
    with patch(
        "prsm.economy.web3.stake_manager.Account.from_key",
        return_value=account,
    ):
        client = StakeManagerClient(
            rpc_url="http://localhost:8545",
            contract_address=deploy_address,
            private_key="0x" + "22" * 32,
        )
    return client, contract, w3_instance


def _stub_send(w3, contract, fn_name: str):
    """Wire up the full build → sign → send → receipt path as a happy path."""
    getattr(contract.functions, fn_name).return_value.build_transaction.return_value = {
        "to": "0xStakeBond",
        "data": "0x",
        "gas": 100000,
        "gasPrice": 1,
        "nonce": 0,
        "chainId": 8453,
    }
    w3.eth.get_transaction_count.return_value = 0
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453
    signed = MagicMock()
    signed.raw_transaction = b"raw"
    w3.eth.account.sign_transaction.return_value = signed
    w3.eth.send_raw_transaction.return_value = b"\xcc" * 32
    receipt = MagicMock()
    receipt.status = 1
    w3.eth.wait_for_transaction_receipt.return_value = receipt


# ── bond ─────────────────────────────────────────────────────────────────


def test_bond_builds_and_sends_tx(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    _stub_send(w3, contract, "bond")

    amount = 25_000 * 10**18
    tx_hash, status = client.bond(amount, SLASH_RATE_PREMIUM_BPS)

    contract.functions.bond.assert_called_once_with(
        amount, SLASH_RATE_PREMIUM_BPS
    )
    assert tx_hash.startswith("0x")
    assert status == TransferStatus.CONFIRMED


def test_bond_rejects_zero_amount(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    with pytest.raises(ValueError, match="amount_wei"):
        client.bond(0, SLASH_RATE_STANDARD_BPS)


def test_bond_rejects_negative_amount(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    with pytest.raises(ValueError, match="amount_wei"):
        client.bond(-1, SLASH_RATE_STANDARD_BPS)


def test_bond_rejects_overflow_amount(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    with pytest.raises(ValueError, match="uint128"):
        client.bond(2**128, SLASH_RATE_STANDARD_BPS)


def test_bond_accepts_uint128_max_minus_one(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    _stub_send(w3, contract, "bond")
    client.bond(2**128 - 1, SLASH_RATE_STANDARD_BPS)  # should not raise


def test_bond_rejects_rate_above_10000_bps(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    with pytest.raises(ValueError, match="tier_slash_rate_bps"):
        client.bond(1_000 * 10**18, 10_001)


def test_bond_rejects_negative_rate(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    with pytest.raises(ValueError, match="tier_slash_rate_bps"):
        client.bond(1_000 * 10**18, -1)


def test_bond_accepts_boundary_rate_10000(mock_web3):
    """Boundary: exactly 10000 bps (100%) is valid — matches contract."""
    client, contract, w3 = _make_client(mock_web3)
    _stub_send(w3, contract, "bond")
    client.bond(TIER_CRITICAL_MIN_WEI, 10_000)  # should not raise


def test_bond_accepts_zero_rate(mock_web3):
    """Boundary: 0 bps ("open" tier) is valid."""
    client, contract, w3 = _make_client(mock_web3)
    _stub_send(w3, contract, "bond")
    client.bond(1_000 * 10**18, 0)  # should not raise


def test_bond_requires_private_key(mock_web3):
    w3_instance, _ = mock_web3
    contract = MagicMock()
    w3_instance.eth.contract.return_value = contract
    # No private_key — read-only client.
    client = StakeManagerClient(
        rpc_url="http://localhost:8545",
        contract_address="0xStakeBond",
    )
    with pytest.raises(RuntimeError, match="private_key"):
        client.bond(1_000 * 10**18, SLASH_RATE_STANDARD_BPS)


# ── request_unbond ──────────────────────────────────────────────────────


def test_request_unbond_builds_and_sends_tx(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    _stub_send(w3, contract, "requestUnbond")
    tx_hash, status = client.request_unbond()
    contract.functions.requestUnbond.assert_called_once_with()
    assert tx_hash.startswith("0x")
    assert status == TransferStatus.CONFIRMED


def test_request_unbond_requires_private_key(mock_web3):
    w3_instance, _ = mock_web3
    w3_instance.eth.contract.return_value = MagicMock()
    client = StakeManagerClient(
        rpc_url="http://localhost:8545", contract_address="0xStakeBond"
    )
    with pytest.raises(RuntimeError, match="private_key"):
        client.request_unbond()


# ── withdraw ─────────────────────────────────────────────────────────────


def test_withdraw_builds_and_sends_tx(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    _stub_send(w3, contract, "withdraw")
    tx_hash, status = client.withdraw()
    contract.functions.withdraw.assert_called_once_with()
    assert tx_hash.startswith("0x")
    assert status == TransferStatus.CONFIRMED


# ── claim_bounty ─────────────────────────────────────────────────────────


def test_claim_bounty_builds_and_sends_tx(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    _stub_send(w3, contract, "claimBounty")
    tx_hash, status = client.claim_bounty()
    contract.functions.claimBounty.assert_called_once_with()
    assert tx_hash.startswith("0x")
    assert status == TransferStatus.CONFIRMED


# ── stake_of ─────────────────────────────────────────────────────────────


def test_stake_of_returns_bonded_record(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    amount = 25_000 * 10**18
    contract.functions.stakeOf.return_value.call.return_value = (
        amount, 1_700_000_000, 0, 1, 10_000,  # status=BONDED, rate=100%
    )
    rec = client.stake_of("0xProvider")
    assert isinstance(rec, StakeRecord)
    assert rec.amount_wei == amount
    assert rec.bonded_at_unix == 1_700_000_000
    assert rec.unbond_eligible_at == 0
    assert rec.status == StakeStatus.BONDED
    assert rec.tier_slash_rate_bps == 10_000
    assert rec.is_bonded is True
    assert rec.is_unbonding is False


def test_stake_of_returns_unbonding_record(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    amount = 5_000 * 10**18
    contract.functions.stakeOf.return_value.call.return_value = (
        amount, 1_700_000_000, 1_700_604_800, 2, 5_000,  # status=UNBONDING
    )
    rec = client.stake_of("0xProvider")
    assert rec.status == StakeStatus.UNBONDING
    assert rec.is_bonded is False
    assert rec.is_unbonding is True
    assert rec.unbond_eligible_at == 1_700_604_800


def test_stake_of_returns_none_status_for_nonexistent(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.stakeOf.return_value.call.return_value = (
        0, 0, 0, 0, 0,  # status=NONE
    )
    rec = client.stake_of("0xNewProvider")
    assert rec.amount_wei == 0
    assert rec.status == StakeStatus.NONE
    assert rec.is_bonded is False


def test_stake_of_returns_withdrawn_record(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.stakeOf.return_value.call.return_value = (
        0, 1_700_000_000, 1_700_604_800, 3, 5_000,  # status=WITHDRAWN
    )
    rec = client.stake_of("0xProvider")
    assert rec.status == StakeStatus.WITHDRAWN
    assert rec.amount_wei == 0


# ── effective_tier ───────────────────────────────────────────────────────


@pytest.mark.parametrize("tier", ["open", "standard", "premium", "critical"])
def test_effective_tier_returns_string(mock_web3, tier):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.effectiveTier.return_value.call.return_value = tier
    assert client.effective_tier("0xProvider") == tier


# ── bounty + reserve + delay reads ──────────────────────────────────────


def test_slashed_bounty_payable_returns_int(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.slashedBountyPayable.return_value.call.return_value = 17_500 * 10**18
    assert client.slashed_bounty_payable("0xChallenger") == 17_500 * 10**18


def test_foundation_reserve_balance_returns_int(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.foundationReserveBalance.return_value.call.return_value = 7_500 * 10**18
    assert client.foundation_reserve_balance() == 7_500 * 10**18


def test_unbond_delay_seconds_returns_int(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.unbondDelaySeconds.return_value.call.return_value = 7 * 24 * 3600
    assert client.unbond_delay_seconds() == 604_800


# ── tier threshold + slash rate constants ───────────────────────────────


def test_tier_threshold_constants_match_contract():
    """These MUST mirror StakeBond.sol:effectiveTier. If the contract
    thresholds change, these constants must update in lock-step."""
    assert TIER_STANDARD_MIN_WEI == 5_000 * 10**18
    assert TIER_PREMIUM_MIN_WEI == 25_000 * 10**18
    assert TIER_CRITICAL_MIN_WEI == 50_000 * 10**18


def test_slash_rate_constants():
    assert SLASH_RATE_STANDARD_BPS == 5_000
    assert SLASH_RATE_PREMIUM_BPS == 10_000
    assert SLASH_RATE_CRITICAL_BPS == 10_000


# ── nonce race protection ───────────────────────────────────────────────


def test_concurrent_bonds_use_distinct_nonces(mock_web3):
    """Two concurrent writes on the same client must serialize via the
    per-client lock and see distinct pending nonces. Prevents collisions
    when an orchestrator fires several staking txs in parallel."""
    import threading
    import time as _t

    client, contract, w3 = _make_client(mock_web3)

    def fake_build(overrides):
        return {**overrides, "to": "0xStakeBond", "data": "0x", "gas": 100000}

    contract.functions.bond.return_value.build_transaction.side_effect = (
        fake_build
    )

    nonce_counter = [0]
    counter_lock = threading.Lock()

    def fake_get_count(addr, *args, **kwargs):
        _t.sleep(0.01)  # realistic RPC latency widens the race window
        with counter_lock:
            n = nonce_counter[0]
            nonce_counter[0] += 1
        return n

    w3.eth.get_transaction_count.side_effect = fake_get_count
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453

    nonces_seen = []
    seen_lock = threading.Lock()

    def fake_sign(tx, key):
        with seen_lock:
            nonces_seen.append(tx["nonce"])
        signed = MagicMock()
        signed.raw_transaction = b"raw"
        return signed

    w3.eth.account.sign_transaction.side_effect = fake_sign
    w3.eth.send_raw_transaction.return_value = b"\xcc" * 32
    receipt = MagicMock()
    receipt.status = 1
    w3.eth.wait_for_transaction_receipt.return_value = receipt

    errors = []

    def call(idx):
        try:
            client.bond(1_000 * 10**18, SLASH_RATE_STANDARD_BPS)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(nonces_seen) == 4
    assert len(set(nonces_seen)) == 4, f"nonce collision: {nonces_seen}"


def test_pending_nonce_strategy_used(mock_web3):
    """_tx_overrides must request 'pending' so back-to-back writes from
    the same client see each other's pending state."""
    client, contract, w3 = _make_client(mock_web3)
    _stub_send(w3, contract, "bond")
    client.bond(1_000 * 10**18, SLASH_RATE_STANDARD_BPS)

    call_args = w3.eth.get_transaction_count.call_args
    assert call_args is not None
    assert "pending" in call_args.args, f"expected 'pending' in {call_args}"


# ── broadcast / revert error paths ──────────────────────────────────────


def test_bond_surfaces_broadcast_failure_as_safe_to_retry(mock_web3):
    """When the broadcast itself fails (e.g., RPC error), the client
    raises BroadcastFailedError — callers may retry safely."""
    from prsm.economy.web3.stake_manager import BroadcastFailedError

    client, contract, w3 = _make_client(mock_web3)
    contract.functions.bond.return_value.build_transaction.return_value = {
        "to": "0xStakeBond", "data": "0x", "gas": 100000,
        "gasPrice": 1, "nonce": 0, "chainId": 8453,
    }
    w3.eth.get_transaction_count.return_value = 0
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453
    signed = MagicMock()
    signed.raw_transaction = b"raw"
    w3.eth.account.sign_transaction.return_value = signed
    w3.eth.send_raw_transaction.side_effect = Exception("RPC down")

    with pytest.raises(BroadcastFailedError):
        client.bond(1_000 * 10**18, SLASH_RATE_STANDARD_BPS)


def test_bond_surfaces_reverted_tx(mock_web3):
    """When a receipt confirms with status=0, raise OnChainRevertedError."""
    from prsm.economy.web3.stake_manager import OnChainRevertedError

    client, contract, w3 = _make_client(mock_web3)
    _stub_send(w3, contract, "bond")
    # Override the receipt to be a revert.
    receipt = MagicMock()
    receipt.status = 0
    w3.eth.wait_for_transaction_receipt.return_value = receipt

    with pytest.raises(OnChainRevertedError):
        client.bond(1_000 * 10**18, SLASH_RATE_STANDARD_BPS)


def test_bond_surfaces_pending_when_receipt_unavailable(mock_web3):
    """If the broadcast succeeded but receipt polling fails, raise
    OnChainPendingError — callers MUST NOT fall back (double-spend risk)."""
    from prsm.economy.web3.stake_manager import OnChainPendingError

    client, contract, w3 = _make_client(mock_web3)
    contract.functions.bond.return_value.build_transaction.return_value = {
        "to": "0xStakeBond", "data": "0x", "gas": 100000,
        "gasPrice": 1, "nonce": 0, "chainId": 8453,
    }
    w3.eth.get_transaction_count.return_value = 0
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453
    signed = MagicMock()
    signed.raw_transaction = b"raw"
    w3.eth.account.sign_transaction.return_value = signed
    w3.eth.send_raw_transaction.return_value = b"\xcc" * 32
    w3.eth.wait_for_transaction_receipt.side_effect = Exception("timeout")

    with pytest.raises(OnChainPendingError) as excinfo:
        client.bond(1_000 * 10**18, SLASH_RATE_STANDARD_BPS)
    # tx_hash exposed for manual reconciliation.
    assert excinfo.value.tx_hash.startswith("0x")
