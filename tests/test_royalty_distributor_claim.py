"""T6.2 (2026-05-05) — unit tests for RoyaltyDistributorClient.claim
and .claimable methods.

These methods support the `prsm wallet` CLI's read-only balance display
and the user-initiated claim flow. Tests use Web3 mocks; no real chain.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.royalty_distributor import RoyaltyDistributorClient


@pytest.fixture
def patched_web3():
    """Patch Web3 + Account so RoyaltyDistributorClient can construct
    without a live RPC."""
    with patch(
        "prsm.economy.web3.royalty_distributor.Web3"
    ) as mock_web3_cls, patch(
        "prsm.economy.web3.royalty_distributor.Account"
    ) as mock_account_cls:
        # mock provider
        mock_web3 = MagicMock()
        mock_web3_cls.return_value = mock_web3
        mock_web3_cls.HTTPProvider = MagicMock()
        mock_web3_cls.to_checksum_address.side_effect = lambda x: x

        # mock account
        mock_acct = MagicMock()
        mock_acct.address = "0xabc1234567890123456789012345678901234567"
        mock_account_cls.from_key.return_value = mock_acct

        # mock contracts
        mock_distributor = MagicMock()
        mock_token = MagicMock()
        mock_web3.eth.contract.side_effect = [mock_distributor, mock_token]

        yield {
            "web3": mock_web3,
            "distributor": mock_distributor,
            "token": mock_token,
            "account": mock_acct,
        }


def _make_client(patched_web3, with_pk=True):
    return RoyaltyDistributorClient(
        rpc_url="http://x",
        distributor_address="0xdead0000000000000000000000000000000000ff",
        ftns_token_address="0xbeef000000000000000000000000000000000077",
        private_key="0x" + "ff" * 32 if with_pk else None,
    )


def test_claimable_returns_int(patched_web3):
    """claimable(addr) returns the int from the contract call."""
    client = _make_client(patched_web3)
    patched_web3["distributor"].functions.claimable.return_value.call.return_value = 12345 * 10**18

    result = client.claimable("0x1111111111111111111111111111111111111111")
    assert result == 12345 * 10**18
    patched_web3["distributor"].functions.claimable.assert_called_once_with(
        "0x1111111111111111111111111111111111111111"
    )


def test_claimable_defaults_to_signer_address(patched_web3):
    """Without an address arg, claimable() uses the configured signer."""
    client = _make_client(patched_web3)
    patched_web3["distributor"].functions.claimable.return_value.call.return_value = 0

    client.claimable()
    # Signer address comes from mocked Account.from_key
    patched_web3["distributor"].functions.claimable.assert_called_once_with(
        "0xabc1234567890123456789012345678901234567"
    )


def test_claimable_no_signer_no_address_raises(patched_web3):
    """claimable() without signer AND without address arg → raises."""
    client = _make_client(patched_web3, with_pk=False)
    with pytest.raises(RuntimeError, match="address required"):
        client.claimable()


def test_claim_requires_signer(patched_web3):
    """claim() without private key → raises (no key to sign tx)."""
    client = _make_client(patched_web3, with_pk=False)
    with pytest.raises(RuntimeError, match="private_key required"):
        client.claim()


def test_claim_builds_and_sends_tx(patched_web3):
    """claim() builds claim() tx + calls _sign_and_send."""
    client = _make_client(patched_web3)
    # Mock the build_transaction chain
    mock_tx = {"to": "0xdead", "data": "0x4e71d92d"}  # claim() selector
    patched_web3["distributor"].functions.claim.return_value.build_transaction.return_value = mock_tx
    # Mock _sign_and_send
    client._sign_and_send = MagicMock(return_value=("0xabc123", "confirmed"))
    # Mock _tx_overrides (depends on web3.eth.* state)
    client._tx_overrides = MagicMock(return_value={})

    tx_hash, status = client.claim()

    assert tx_hash == "0xabc123"
    assert status == "confirmed"
    patched_web3["distributor"].functions.claim.assert_called_once()
    client._sign_and_send.assert_called_once_with(mock_tx)
