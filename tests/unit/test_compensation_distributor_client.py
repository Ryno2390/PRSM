"""CompensationDistributorClient — Python client for the on-chain
``CompensationDistributor.sol`` contract (mainnet-deployed 2026-05-07
as part of Phase 8).

Closes the readiness gap surfaced in
`docs/security/EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md` §6.2: the
contract has been live since 2026-05-07 but no Python client shipped,
so operator-side `pullAndDistribute` invocation + `Distributed` event
monitoring was unavailable.

Operator-facing surface (admin functions like updateWeights /
setPoolAddresses are intentionally NOT in this client — those go
through Foundation Safe direct calls):

  - pull_and_distribute()            permissionless write
  - current_weights()                view
  - last_distribution_timestamp()    view
  - creator_pool() / operator_pool() / grant_pool()  views
  - has_scheduled_weights() / scheduled_at()         views
  - DistributedEvent.from_decoded_args(args)         event decode

Tests use a stub Web3 + contract surface to verify contract-call
shape, error mapping, and event-decode without requiring a live chain.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.compensation_distributor import (
    CompensationDistributorClient,
    DistributedEvent,
    PoolWeights,
)
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainRevertedError,
    TransferStatus,
)


# ──────────────────────────────────────────────────────────────────────
# Stub Web3 surface (mirrors test_key_distribution_client.py)
# ──────────────────────────────────────────────────────────────────────


class _FakeContractFunction:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        # Default view return for tests that don't override.
        self._view_return = None

    def build_transaction(self, overrides):
        return {"to": "0xcontract", "data": "0xdeadbeef", **overrides}

    def call(self):
        return self._view_return


class _FakeContractFunctions:
    def __init__(self, parent):
        self._parent = parent

    def pullAndDistribute(self, *args):
        self._parent._calls.append(("pullAndDistribute", args))
        return _FakeContractFunction(*args)

    def currentWeights(self, *args):
        fn = _FakeContractFunction(*args)
        # Return as a tuple matching the Solidity struct order:
        # (creatorPoolBps, operatorPoolBps, grantPoolBps).
        fn._view_return = (4500, 3500, 2000)
        return fn

    def lastDistributionTimestamp(self, *args):
        fn = _FakeContractFunction(*args)
        fn._view_return = 1_700_000_000
        return fn

    def creatorPool(self, *args):
        fn = _FakeContractFunction(*args)
        fn._view_return = "0x" + "11" * 20
        return fn

    def operatorPool(self, *args):
        fn = _FakeContractFunction(*args)
        fn._view_return = "0x" + "22" * 20
        return fn

    def grantPool(self, *args):
        fn = _FakeContractFunction(*args)
        fn._view_return = "0x" + "33" * 20
        return fn

    def hasScheduledWeights(self, *args):
        fn = _FakeContractFunction(*args)
        fn._view_return = False
        return fn

    def scheduledAt(self, *args):
        fn = _FakeContractFunction(*args)
        fn._view_return = 0
        return fn


class _FakeContract:
    def __init__(self):
        self._calls = []
        self.functions = _FakeContractFunctions(self)
        self.events = MagicMock()


class _FakeReceipt:
    def __init__(self, status: int = 1):
        self.status = status


class _FakeAccount:
    address = "0x" + "11" * 20
    key = b"\x01" * 32


class _FakeEth:
    def __init__(self, *, chain_id=8453, gas_price=10**9, send_ok=True, receipt_status=1):
        self.chain_id = chain_id
        self.gas_price = gas_price
        self._send_ok = send_ok
        self._receipt_status = receipt_status
        self.account = MagicMock()
        signed_mock = MagicMock()
        signed_mock.raw_transaction = b"\xff" * 32
        self.account.sign_transaction.return_value = signed_mock

    def get_transaction_count(self, addr, *_):
        return 7

    def contract(self, address, abi):
        return _FakeContract()

    def send_raw_transaction(self, raw):
        if not self._send_ok:
            raise RuntimeError("network error")
        return b"\xab" * 32

    def wait_for_transaction_receipt(self, tx_hash, timeout=120):
        return _FakeReceipt(status=self._receipt_status)


_TEST_WEB3_CONFIG = {"send_ok": True, "receipt_status": 1}


class _FakeWeb3:
    def __init__(self, *args, **kwargs):
        self.eth = _FakeEth(
            chain_id=8453,
            send_ok=_TEST_WEB3_CONFIG["send_ok"],
            receipt_status=_TEST_WEB3_CONFIG["receipt_status"],
        )

    @staticmethod
    def to_checksum_address(addr):
        return addr

    @staticmethod
    def HTTPProvider(url):
        return object()


def _make_client(*, send_ok=True, receipt_status=1, with_key=True):
    _TEST_WEB3_CONFIG["send_ok"] = send_ok
    _TEST_WEB3_CONFIG["receipt_status"] = receipt_status
    with patch(
        "prsm.economy.web3.compensation_distributor.Web3",
        _FakeWeb3,
    ), patch(
        "prsm.economy.web3.compensation_distributor.Account",
    ) as mock_account:
        mock_account.from_key.return_value = _FakeAccount()
        client = CompensationDistributorClient(
            rpc_url="http://test",
            contract_address="0x" + "ab" * 20,
            private_key="0x" + "01" * 32 if with_key else None,
        )
    return client


# ──────────────────────────────────────────────────────────────────────
# PoolWeights dataclass
# ──────────────────────────────────────────────────────────────────────


class TestPoolWeights:
    def test_construction_validates_sum_to_10000(self):
        # Mirror contract InvalidWeights revert client-side.
        with pytest.raises(ValueError, match="must sum to 10000"):
            PoolWeights(creator_bps=5000, operator_bps=3000, grant_bps=1000)

    def test_construction_accepts_valid_sum(self):
        w = PoolWeights(creator_bps=4500, operator_bps=3500, grant_bps=2000)
        assert w.creator_bps == 4500

    def test_each_field_must_fit_uint16(self):
        with pytest.raises(ValueError, match="bps"):
            PoolWeights(creator_bps=70000, operator_bps=0, grant_bps=0)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="bps"):
            PoolWeights(creator_bps=-1, operator_bps=5000, grant_bps=5001)


# ──────────────────────────────────────────────────────────────────────
# DistributedEvent dataclass
# ──────────────────────────────────────────────────────────────────────


class TestDistributedEvent:
    def test_from_decoded_args_happy_path(self):
        event = DistributedEvent.from_decoded_args({
            "toCreator": 4500 * 10**14,
            "toOperator": 3500 * 10**14,
            "toGrant": 2000 * 10**14,
        })
        assert event.to_creator == 4500 * 10**14
        assert event.to_operator == 3500 * 10**14
        assert event.to_grant == 2000 * 10**14

    def test_negative_amounts_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            DistributedEvent(to_creator=-1, to_operator=0, to_grant=0)

    def test_frozen(self):
        event = DistributedEvent(to_creator=1, to_operator=2, to_grant=3)
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            event.to_creator = 999  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_construction_with_key(self):
        client = _make_client()
        assert client.address == "0x" + "11" * 20

    def test_construction_without_key(self):
        client = _make_client(with_key=False)
        assert client.address is None

    def test_write_call_without_key_raises(self):
        client = _make_client(with_key=False)
        with pytest.raises(RuntimeError, match="private_key required"):
            client.pull_and_distribute()


# ──────────────────────────────────────────────────────────────────────
# pull_and_distribute
# ──────────────────────────────────────────────────────────────────────


class TestPullAndDistribute:
    def test_happy_path(self):
        client = _make_client()
        tx_hash, status = client.pull_and_distribute()
        assert tx_hash.startswith("0x")
        assert status == TransferStatus.CONFIRMED

    def test_broadcast_failure_raises_safe_fallback(self):
        client = _make_client(send_ok=False)
        with pytest.raises(BroadcastFailedError):
            client.pull_and_distribute()

    def test_revert_raises_OnChainRevertedError(self):
        # Contract reverts TransferFailed if FTNS transfer fails.
        client = _make_client(receipt_status=0)
        with pytest.raises(OnChainRevertedError):
            client.pull_and_distribute()


# ──────────────────────────────────────────────────────────────────────
# View reads
# ──────────────────────────────────────────────────────────────────────


class TestViews:
    def test_current_weights_returns_PoolWeights(self):
        client = _make_client(with_key=False)
        weights = client.current_weights()
        assert isinstance(weights, PoolWeights)
        assert weights.creator_bps == 4500
        assert weights.operator_bps == 3500
        assert weights.grant_bps == 2000

    def test_last_distribution_timestamp(self):
        client = _make_client(with_key=False)
        ts = client.last_distribution_timestamp()
        assert ts == 1_700_000_000

    def test_pool_addresses(self):
        client = _make_client(with_key=False)
        assert client.creator_pool() == "0x" + "11" * 20
        assert client.operator_pool() == "0x" + "22" * 20
        assert client.grant_pool() == "0x" + "33" * 20

    def test_scheduled_state(self):
        client = _make_client(with_key=False)
        assert client.has_scheduled_weights() is False
        assert client.scheduled_at() == 0
