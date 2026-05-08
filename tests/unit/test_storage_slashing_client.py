"""StorageSlashingClient — Python client for the on-chain
``StorageSlashing.sol`` contract (mainnet-deployed 2026-05-07 as
part of Phase 7-storage).

Closes the readiness gap surfaced in
`docs/security/EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md` §6.2.

Three operator roles share this client surface:

  - Provider role: ``record_heartbeat()`` — every storage provider
    calls regularly to demonstrate liveness
  - Verifier role: ``submit_proof_failure(...)`` — authorized
    verifier-only (Foundation-operated service today)
  - Permissionless role: ``slash_for_missing_heartbeat(provider)`` —
    anyone can call after the grace window has elapsed; caller
    receives the challenger bounty

Plus three event-decoders + four view methods.

Tests use a stub Web3 + contract surface — same pattern as
``test_key_distribution_client.py``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.storage_slashing import (
    AlreadySlashedError,
    HeartbeatMissingSlashedEvent,
    HeartbeatNotExpiredError,
    HeartbeatNotRecordedError,
    HeartbeatRecordedEvent,
    NotAuthorizedVerifierError,
    ProofFailureSlashedEvent,
    StorageSlashingClient,
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
        self._view_return = None

    def build_transaction(self, overrides):
        return {"to": "0xcontract", "data": "0xdeadbeef", **overrides}

    def call(self):
        return self._view_return


class _FakeContractFunctions:
    def __init__(self, parent):
        self._parent = parent

    def recordHeartbeat(self, *args):
        self._parent._calls.append(("recordHeartbeat", args))
        return _FakeContractFunction(*args)

    def submitProofFailure(self, *args):
        self._parent._calls.append(("submitProofFailure", args))
        return _FakeContractFunction(*args)

    def slashForMissingHeartbeat(self, *args):
        self._parent._calls.append(("slashForMissingHeartbeat", args))
        return _FakeContractFunction(*args)

    def lastHeartbeat(self, *args):
        self._parent._calls.append(("lastHeartbeat", args))
        fn = _FakeContractFunction(*args)
        fn._view_return = 1_700_000_000
        return fn

    def heartbeatGraceSeconds(self, *args):
        fn = _FakeContractFunction(*args)
        fn._view_return = 86400  # 24h default
        return fn

    def authorizedVerifier(self, *args):
        fn = _FakeContractFunction(*args)
        fn._view_return = "0x" + "44" * 20
        return fn

    def slashRecorded(self, *args):
        self._parent._calls.append(("slashRecorded", args))
        fn = _FakeContractFunction(*args)
        fn._view_return = False
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
        "prsm.economy.web3.storage_slashing.Web3",
        _FakeWeb3,
    ), patch(
        "prsm.economy.web3.storage_slashing.Account",
    ) as mock_account:
        mock_account.from_key.return_value = _FakeAccount()
        client = StorageSlashingClient(
            rpc_url="http://test",
            contract_address="0x" + "ab" * 20,
            private_key="0x" + "01" * 32 if with_key else None,
        )
    return client


# ──────────────────────────────────────────────────────────────────────
# Typed errors
# ──────────────────────────────────────────────────────────────────────


class TestTypedErrors:
    def test_already_slashed_carries_slash_id(self):
        err = AlreadySlashedError(slash_id=b"\xaa" * 32)
        assert err.slash_id == b"\xaa" * 32

    def test_heartbeat_not_expired_carries_now_and_expiry(self):
        err = HeartbeatNotExpiredError(now_ts=100, expiry_ts=200)
        assert err.now_ts == 100
        assert err.expiry_ts == 200

    def test_heartbeat_not_recorded_is_runtime_error(self):
        err = HeartbeatNotRecordedError("provider has no heartbeat")
        assert isinstance(err, RuntimeError)

    def test_not_authorized_verifier_is_runtime_error(self):
        err = NotAuthorizedVerifierError("caller is not verifier")
        assert isinstance(err, RuntimeError)


# ──────────────────────────────────────────────────────────────────────
# Event dataclasses
# ──────────────────────────────────────────────────────────────────────


class TestHeartbeatRecordedEvent:
    def test_from_decoded_args_happy_path(self):
        event = HeartbeatRecordedEvent.from_decoded_args({
            "provider": "0x" + "11" * 20,
            "timestamp": 1_700_000_000,
        })
        assert event.provider == "0x" + "11" * 20
        assert event.timestamp == 1_700_000_000

    def test_negative_timestamp_rejected(self):
        with pytest.raises(ValueError):
            HeartbeatRecordedEvent(provider="0x" + "11" * 20, timestamp=-1)


class TestProofFailureSlashedEvent:
    def test_from_decoded_args_happy_path(self):
        event = ProofFailureSlashedEvent.from_decoded_args({
            "provider": "0x" + "11" * 20,
            "challenger": "0x" + "22" * 20,
            "shardId": b"\x33" * 32,
            "evidenceHash": b"\x44" * 32,
            "slashId": b"\x55" * 32,
        })
        assert event.provider == "0x" + "11" * 20
        assert event.challenger == "0x" + "22" * 20
        assert event.shard_id == b"\x33" * 32
        assert event.evidence_hash == b"\x44" * 32
        assert event.slash_id == b"\x55" * 32

    def test_validates_bytes32_lengths(self):
        with pytest.raises(ValueError, match="32 bytes"):
            ProofFailureSlashedEvent(
                provider="0x" + "11" * 20,
                challenger="0x" + "22" * 20,
                shard_id=b"\x00" * 16,  # wrong length
                evidence_hash=b"\x00" * 32,
                slash_id=b"\x00" * 32,
            )


class TestHeartbeatMissingSlashedEvent:
    def test_from_decoded_args_happy_path(self):
        event = HeartbeatMissingSlashedEvent.from_decoded_args({
            "provider": "0x" + "11" * 20,
            "challenger": "0x" + "22" * 20,
            "lastHeartbeatAt": 1_700_000_000,
            "slashId": b"\x55" * 32,
        })
        assert event.last_heartbeat_at == 1_700_000_000
        assert event.slash_id == b"\x55" * 32


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_with_key(self):
        client = _make_client()
        assert client.address == "0x" + "11" * 20

    def test_without_key(self):
        client = _make_client(with_key=False)
        assert client.address is None

    def test_write_without_key_raises(self):
        client = _make_client(with_key=False)
        with pytest.raises(RuntimeError, match="private_key required"):
            client.record_heartbeat()


# ──────────────────────────────────────────────────────────────────────
# record_heartbeat (provider role)
# ──────────────────────────────────────────────────────────────────────


class TestRecordHeartbeat:
    def test_happy_path(self):
        client = _make_client()
        tx_hash, status = client.record_heartbeat()
        assert tx_hash.startswith("0x")
        assert status == TransferStatus.CONFIRMED

    def test_broadcast_failure(self):
        client = _make_client(send_ok=False)
        with pytest.raises(BroadcastFailedError):
            client.record_heartbeat()


# ──────────────────────────────────────────────────────────────────────
# submit_proof_failure (verifier role)
# ──────────────────────────────────────────────────────────────────────


class TestSubmitProofFailure:
    def test_happy_path(self):
        client = _make_client()
        tx_hash, status = client.submit_proof_failure(
            provider="0x" + "11" * 20,
            shard_id=b"\x33" * 32,
            evidence_hash=b"\x44" * 32,
            challenger="0x" + "22" * 20,
        )
        assert tx_hash.startswith("0x")
        assert status == TransferStatus.CONFIRMED

    def test_rejects_zero_provider_address(self):
        client = _make_client()
        with pytest.raises(ValueError, match="provider"):
            client.submit_proof_failure(
                provider="0x" + "00" * 20,
                shard_id=b"\x33" * 32,
                evidence_hash=b"\x44" * 32,
                challenger="0x" + "22" * 20,
            )

    def test_rejects_wrong_length_shard_id(self):
        client = _make_client()
        with pytest.raises(ValueError, match="shard_id"):
            client.submit_proof_failure(
                provider="0x" + "11" * 20,
                shard_id=b"\x00" * 16,
                evidence_hash=b"\x44" * 32,
                challenger="0x" + "22" * 20,
            )

    def test_rejects_wrong_length_evidence_hash(self):
        client = _make_client()
        with pytest.raises(ValueError, match="evidence_hash"):
            client.submit_proof_failure(
                provider="0x" + "11" * 20,
                shard_id=b"\x33" * 32,
                evidence_hash=b"\x00" * 16,
                challenger="0x" + "22" * 20,
            )

    def test_revert_safe_fallback(self):
        # Simulate AlreadySlashed / NotAuthorizedVerifier — surface as
        # OnChainRevertedError; client lifts to typed error only via
        # explicit error-data decode (not implemented for receipt-only
        # tx flow today).
        client = _make_client(receipt_status=0)
        with pytest.raises(OnChainRevertedError):
            client.submit_proof_failure(
                provider="0x" + "11" * 20,
                shard_id=b"\x33" * 32,
                evidence_hash=b"\x44" * 32,
                challenger="0x" + "22" * 20,
            )


# ──────────────────────────────────────────────────────────────────────
# slash_for_missing_heartbeat (permissionless)
# ──────────────────────────────────────────────────────────────────────


class TestSlashForMissingHeartbeat:
    def test_happy_path(self):
        client = _make_client()
        tx_hash, status = client.slash_for_missing_heartbeat(
            provider="0x" + "11" * 20,
        )
        assert tx_hash.startswith("0x")
        assert status == TransferStatus.CONFIRMED

    def test_rejects_zero_provider_address(self):
        client = _make_client()
        with pytest.raises(ValueError, match="provider"):
            client.slash_for_missing_heartbeat(provider="0x" + "00" * 20)


# ──────────────────────────────────────────────────────────────────────
# Views
# ──────────────────────────────────────────────────────────────────────


class TestViews:
    def test_last_heartbeat(self):
        client = _make_client(with_key=False)
        ts = client.last_heartbeat("0x" + "11" * 20)
        assert ts == 1_700_000_000

    def test_heartbeat_grace_seconds(self):
        client = _make_client(with_key=False)
        assert client.heartbeat_grace_seconds() == 86400

    def test_authorized_verifier(self):
        client = _make_client(with_key=False)
        assert client.authorized_verifier() == "0x" + "44" * 20

    def test_slash_recorded(self):
        client = _make_client(with_key=False)
        assert client.slash_recorded(b"\xaa" * 32) is False

    def test_slash_recorded_validates_length(self):
        client = _make_client(with_key=False)
        with pytest.raises(ValueError, match="32 bytes"):
            client.slash_recorded(b"\x00" * 16)
