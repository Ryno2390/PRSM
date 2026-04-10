"""Unit tests for ProvenanceRegistryClient.

These mock Web3 entirely — they validate the client wraps the contract
ABI correctly and handles common error cases. End-to-end happy-path
testing against a real Hardhat node lives in
tests/integration/test_onchain_provenance_e2e.py.
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.provenance_registry import (
    ProvenanceRegistryClient,
    ContentRecord,
)


def _hash(s: str) -> bytes:
    return hashlib.sha3_256(s.encode()).digest()


@pytest.fixture
def mock_web3():
    with patch("prsm.economy.web3.provenance_registry.Web3") as MockWeb3:
        w3_instance = MagicMock()
        MockWeb3.return_value = w3_instance
        MockWeb3.HTTPProvider.return_value = MagicMock()
        MockWeb3.to_checksum_address.side_effect = lambda x: x
        yield w3_instance, MockWeb3


def _make_client(mock_web3, deploy_address="0xRegistry"):
    w3_instance, _ = mock_web3
    contract = MagicMock()
    w3_instance.eth.contract.return_value = contract

    account = MagicMock()
    account.address = "0xCreator"
    account.key = b"\x11" * 32
    with patch(
        "prsm.economy.web3.provenance_registry.Account.from_key",
        return_value=account,
    ):
        client = ProvenanceRegistryClient(
            rpc_url="http://localhost:8545",
            contract_address=deploy_address,
            private_key="0x" + "11" * 32,
        )
    return client, contract, w3_instance


def test_register_content_builds_and_sends_tx(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.registerContent.return_value.build_transaction.return_value = {
        "to": "0xRegistry",
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
    w3.eth.send_raw_transaction.return_value = b"\xab" * 32
    receipt = MagicMock()
    receipt.status = 1
    w3.eth.wait_for_transaction_receipt.return_value = receipt

    content_hash = _hash("hello")
    tx_hash = client.register_content(
        content_hash, royalty_rate_bps=800, metadata_uri="ipfs://X"
    )

    contract.functions.registerContent.assert_called_once_with(
        content_hash, 800, "ipfs://X"
    )
    assert tx_hash.startswith("0x")


def test_register_content_rejects_invalid_rate(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    with pytest.raises(ValueError, match="royalty_rate_bps"):
        client.register_content(
            _hash("x"), royalty_rate_bps=10001, metadata_uri="ipfs://X"
        )


def test_register_content_rejects_wrong_hash_length(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    with pytest.raises(ValueError, match="32 bytes"):
        client.register_content(
            b"short", royalty_rate_bps=800, metadata_uri="ipfs://X"
        )


def test_get_content_returns_record(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.contents.return_value.call.return_value = (
        "0xCreator",
        800,
        1700000000,
        "ipfs://X",
    )
    rec = client.get_content(_hash("hello"))
    assert isinstance(rec, ContentRecord)
    assert rec.creator == "0xCreator"
    assert rec.royalty_rate_bps == 800
    assert rec.registered_at == 1700000000
    assert rec.metadata_uri == "ipfs://X"


def test_get_content_returns_none_for_unregistered(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.contents.return_value.call.return_value = (
        "0x0000000000000000000000000000000000000000",
        0,
        0,
        "",
    )
    rec = client.get_content(_hash("missing"))
    assert rec is None


def test_is_registered_returns_bool(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.isRegistered.return_value.call.return_value = True
    assert client.is_registered(_hash("yes")) is True
    contract.functions.isRegistered.return_value.call.return_value = False
    assert client.is_registered(_hash("no")) is False
