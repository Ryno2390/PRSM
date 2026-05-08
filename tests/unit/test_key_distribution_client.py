"""KeyDistributionClient — Python client for the on-chain
``KeyDistribution.sol`` contract.

Closes the §13 item #2 honest-scope item: contract is mainnet-live,
Python client to coordinate Tier C key deposit + release-on-payment
is what was missing.

Tests use a stub Web3 + contract surface to verify the client's
contract-call shape, error mapping, and event-decode without
requiring a live chain.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.key_distribution import (
    KeyAlreadyDepositedError,
    KeyDistributionClient,
    KeyNotFoundError,
    KeyReleasedEvent,
    PaymentNotVerifiedError,
)
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainRevertedError,
    TransferStatus,
)


# ──────────────────────────────────────────────────────────────────────
# Stub Web3 surface
# ──────────────────────────────────────────────────────────────────────


class _FakeContractFunction:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def build_transaction(self, overrides):
        return {"to": "0xcontract", "data": "0xdeadbeef", **overrides}


class _FakeContractFunctions:
    def __init__(self, parent):
        self._parent = parent

    def depositKey(self, *args):
        self._parent._calls.append(("depositKey", args))
        return _FakeContractFunction(*args)

    def release(self, *args):
        self._parent._calls.append(("release", args))
        return _FakeContractFunction(*args)

    def deauthorize(self, *args):
        self._parent._calls.append(("deauthorize", args))
        return _FakeContractFunction(*args)


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
        # Mock signing — return an object with raw_transaction.
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


# Module-level test config so the lambda factory can read it.
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
        "prsm.economy.web3.key_distribution.Web3",
        _FakeWeb3,
    ), patch(
        "prsm.economy.web3.key_distribution.Account",
    ) as mock_account:
        mock_account.from_key.return_value = _FakeAccount()
        client = KeyDistributionClient(
            rpc_url="http://test",
            contract_address="0x" + "ab" * 20,
            private_key="0x" + "01" * 32 if with_key else None,
        )
    return client


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
            client.deposit_key(
                content_hash=b"\x00" * 32,
                encrypted_key=b"\x01" * 16,
                royalty_address="0x" + "22" * 20,
                release_fee_ftns_wei=10**18,
            )


# ──────────────────────────────────────────────────────────────────────
# deposit_key
# ──────────────────────────────────────────────────────────────────────


class TestDepositKey:
    def test_happy_path_returns_confirmed_status(self):
        client = _make_client()
        tx_hash, status = client.deposit_key(
            content_hash=b"\xaa" * 32,
            encrypted_key=b"ciphertext-bytes",
            royalty_address="0x" + "22" * 20,
            release_fee_ftns_wei=10**18,
        )
        assert tx_hash.startswith("0x")
        assert status == TransferStatus.CONFIRMED

    def test_content_hash_must_be_32_bytes(self):
        client = _make_client()
        with pytest.raises(ValueError, match="content_hash"):
            client.deposit_key(
                content_hash=b"\x00" * 16,
                encrypted_key=b"ct",
                royalty_address="0x" + "22" * 20,
                release_fee_ftns_wei=10**18,
            )

    def test_empty_encrypted_key_rejected(self):
        # Mirror the contract's EmptyEncryptedKey revert client-side.
        client = _make_client()
        with pytest.raises(ValueError, match="encrypted_key"):
            client.deposit_key(
                content_hash=b"\x00" * 32,
                encrypted_key=b"",
                royalty_address="0x" + "22" * 20,
                release_fee_ftns_wei=10**18,
            )

    def test_zero_fee_rejected(self):
        # Contract reverts ZeroFee — fail-fast client-side.
        client = _make_client()
        with pytest.raises(ValueError, match="release_fee"):
            client.deposit_key(
                content_hash=b"\x00" * 32,
                encrypted_key=b"ct",
                royalty_address="0x" + "22" * 20,
                release_fee_ftns_wei=0,
            )

    def test_zero_address_royalty_rejected(self):
        client = _make_client()
        with pytest.raises(ValueError, match="royalty_address"):
            client.deposit_key(
                content_hash=b"\x00" * 32,
                encrypted_key=b"ct",
                royalty_address="0x" + "00" * 20,
                release_fee_ftns_wei=10**18,
            )

    def test_broadcast_failure_raises_safe_fallback(self):
        client = _make_client(send_ok=False)
        with pytest.raises(BroadcastFailedError):
            client.deposit_key(
                content_hash=b"\x00" * 32,
                encrypted_key=b"ct",
                royalty_address="0x" + "22" * 20,
                release_fee_ftns_wei=10**18,
            )

    def test_revert_raises_OnChainRevertedError(self):
        client = _make_client(receipt_status=0)
        with pytest.raises(OnChainRevertedError):
            client.deposit_key(
                content_hash=b"\x00" * 32,
                encrypted_key=b"ct",
                royalty_address="0x" + "22" * 20,
                release_fee_ftns_wei=10**18,
            )


# ──────────────────────────────────────────────────────────────────────
# release
# ──────────────────────────────────────────────────────────────────────


class TestRelease:
    def test_happy_path(self):
        client = _make_client()
        tx_hash, status = client.release(
            content_hash=b"\xaa" * 32,
            recipient="0x" + "33" * 20,
        )
        assert status == TransferStatus.CONFIRMED

    def test_zero_recipient_rejected(self):
        client = _make_client()
        with pytest.raises(ValueError, match="recipient"):
            client.release(
                content_hash=b"\xaa" * 32,
                recipient="0x" + "00" * 20,
            )

    def test_content_hash_must_be_32_bytes(self):
        client = _make_client()
        with pytest.raises(ValueError, match="content_hash"):
            client.release(
                content_hash=b"\x00" * 16,
                recipient="0x" + "33" * 20,
            )


# ──────────────────────────────────────────────────────────────────────
# deauthorize
# ──────────────────────────────────────────────────────────────────────


class TestDeauthorize:
    def test_happy_path(self):
        client = _make_client()
        tx_hash, status = client.deauthorize(content_hash=b"\xaa" * 32)
        assert status == TransferStatus.CONFIRMED

    def test_content_hash_must_be_32_bytes(self):
        client = _make_client()
        with pytest.raises(ValueError, match="content_hash"):
            client.deauthorize(content_hash=b"\x00" * 16)


# ──────────────────────────────────────────────────────────────────────
# KeyReleasedEvent decode
# ──────────────────────────────────────────────────────────────────────


class TestKeyReleasedEvent:
    def test_dataclass_equality_by_value(self):
        a = KeyReleasedEvent(
            content_hash=b"\xaa" * 32,
            recipient="0x" + "33" * 20,
            encrypted_key=b"ciphertext",
        )
        b = KeyReleasedEvent(
            content_hash=b"\xaa" * 32,
            recipient="0x" + "33" * 20,
            encrypted_key=b"ciphertext",
        )
        assert a == b

    def test_content_hash_must_be_32_bytes(self):
        with pytest.raises(ValueError, match="content_hash"):
            KeyReleasedEvent(
                content_hash=b"\x00" * 16,
                recipient="0x" + "33" * 20,
                encrypted_key=b"ct",
            )

    def test_empty_encrypted_key_rejected(self):
        with pytest.raises(ValueError, match="encrypted_key"):
            KeyReleasedEvent(
                content_hash=b"\xaa" * 32,
                recipient="0x" + "33" * 20,
                encrypted_key=b"",
            )

    def test_from_log_decodes_indexed_topics(self):
        # Stub web3 log entry shape: topics + data.
        # KeyReleased(bytes32 indexed contentHash, address indexed recipient, bytes encryptedKey)
        content_hash = b"\xaa" * 32
        recipient_addr = "0x" + "33" * 20
        # Minimal log shape — the from_log helper just needs the
        # decoded event args, so test against the contract event
        # decoder via dict (bypassing the actual Web3 decode).
        decoded_args = {
            "contentHash": content_hash,
            "recipient": recipient_addr,
            "encryptedKey": b"ciphertext",
        }
        event = KeyReleasedEvent.from_decoded_args(decoded_args)
        assert event.content_hash == content_hash
        assert event.recipient == recipient_addr
        assert event.encrypted_key == b"ciphertext"
