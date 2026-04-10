"""Unit tests for RoyaltyDistributorClient (mocked Web3)."""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.royalty_distributor import (
    RoyaltyDistributorClient,
    SplitPreview,
)


def _hash(s: str) -> bytes:
    return hashlib.sha3_256(s.encode()).digest()


@pytest.fixture
def mock_web3():
    with patch("prsm.economy.web3.royalty_distributor.Web3") as MockWeb3:
        w3 = MagicMock()
        MockWeb3.return_value = w3
        MockWeb3.HTTPProvider.return_value = MagicMock()
        MockWeb3.to_checksum_address.side_effect = lambda x: x
        yield w3, MockWeb3


def _make_client(mock_web3):
    w3, _ = mock_web3
    distributor_contract = MagicMock()
    token_contract = MagicMock()
    # eth.contract is called twice — first for distributor, then for token
    w3.eth.contract.side_effect = [distributor_contract, token_contract]

    account = MagicMock()
    account.address = "0xPayer"
    account.key = b"\x22" * 32
    with patch(
        "prsm.economy.web3.royalty_distributor.Account.from_key",
        return_value=account,
    ):
        client = RoyaltyDistributorClient(
            rpc_url="http://localhost:8545",
            distributor_address="0xDistributor",
            ftns_token_address="0xFTNS",
            private_key="0x" + "22" * 32,
        )
    return client, distributor_contract, token_contract, w3


def test_preview_returns_split(mock_web3):
    client, distributor, token, w3 = _make_client(mock_web3)
    distributor.functions.preview.return_value.call.return_value = (8, 2, 90)
    preview = client.preview_split(_hash("x"), 100)
    assert isinstance(preview, SplitPreview)
    assert preview.creator_amount == 8
    assert preview.network_amount == 2
    assert preview.serving_node_amount == 90


def test_distribute_royalty_approves_then_distributes(mock_web3):
    client, distributor, token, w3 = _make_client(mock_web3)

    # current allowance is 0 → approval needed
    token.functions.allowance.return_value.call.return_value = 0
    token.functions.approve.return_value.build_transaction.return_value = {
        "to": "0xFTNS",
        "data": "0x",
        "gas": 60000,
        "gasPrice": 1,
        "nonce": 0,
        "chainId": 8453,
    }
    distributor.functions.distributeRoyalty.return_value.build_transaction.return_value = {
        "to": "0xDistributor",
        "data": "0x",
        "gas": 200000,
        "gasPrice": 1,
        "nonce": 1,
        "chainId": 8453,
    }

    w3.eth.get_transaction_count.side_effect = [0, 1]
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453

    signed = MagicMock()
    signed.raw_transaction = b"raw"
    w3.eth.account.sign_transaction.return_value = signed
    w3.eth.send_raw_transaction.side_effect = [b"\xa1" * 32, b"\xa2" * 32]
    receipt = MagicMock()
    receipt.status = 1
    w3.eth.wait_for_transaction_receipt.return_value = receipt

    tx_hash = client.distribute_royalty(_hash("c"), "0xNode", gross=100)

    token.functions.approve.assert_called_once()
    distributor.functions.distributeRoyalty.assert_called_once_with(
        _hash("c"), "0xNode", 100
    )
    assert tx_hash.startswith("0x")


def test_distribute_royalty_skips_approval_when_allowance_sufficient(mock_web3):
    client, distributor, token, w3 = _make_client(mock_web3)
    token.functions.allowance.return_value.call.return_value = 10**30  # huge
    distributor.functions.distributeRoyalty.return_value.build_transaction.return_value = {
        "to": "0xDistributor",
        "data": "0x",
        "gas": 200000,
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
    w3.eth.send_raw_transaction.return_value = b"\xb1" * 32
    receipt = MagicMock()
    receipt.status = 1
    w3.eth.wait_for_transaction_receipt.return_value = receipt

    client.distribute_royalty(_hash("c"), "0xNode", gross=100)

    token.functions.approve.assert_not_called()


def test_distribute_royalty_rejects_zero_gross(mock_web3):
    client, distributor, token, w3 = _make_client(mock_web3)
    with pytest.raises(ValueError, match="gross"):
        client.distribute_royalty(_hash("c"), "0xNode", gross=0)
