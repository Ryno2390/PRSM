"""node.py builder helper for RoyaltyDistributorClient.

Closes part of the audit-prep §7.23 honest-scope deferred item
(aggregate-source quoting): aggregating claimable royalties into
prsm_balance_check requires the Node to expose a constructed
RoyaltyDistributorClient. This builder follows the established
dual-gate pattern (address env + private key) used by the Phase
7-storage + Phase 8 client builders.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from prsm.node.node import (
    _build_royalty_distributor_client_or_none,
)


class TestBuildRoyaltyDistributorClient:
    def test_returns_none_when_address_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_ROYALTY_DISTRIBUTOR_ADDRESS", None)
            os.environ.pop("PRSM_NETWORK", None)
            assert _build_royalty_distributor_client_or_none() is None

    def test_falls_back_to_canonical_when_network_set(self):
        """Sprint 144 — operator who declared `PRSM_NETWORK=mainnet`
        shouldn't ALSO have to paste the canonical RoyaltyDistributor
        address into a second env var. networks.py is canonical
        source of truth; the builder should consult resolve_endpoints()
        before giving up.
        """
        canonical = "0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e"  # v2 mainnet
        with patch(
            "prsm.economy.web3.royalty_distributor.RoyaltyDistributorClient"
        ) as MockClient, patch.dict(os.environ, {
            "PRSM_NETWORK": "mainnet",
        }, clear=False):
            os.environ.pop("PRSM_ROYALTY_DISTRIBUTOR_ADDRESS", None)
            MockClient.return_value = MagicMock()
            client = _build_royalty_distributor_client_or_none()
            assert client is not None
            kwargs = MockClient.call_args.kwargs
            assert kwargs["distributor_address"] == canonical

    def test_returns_none_when_network_set_but_canonical_missing(self):
        """Network resolves but the canonical addr field is None
        (e.g., a future testnet that hasn't been deployed yet).
        Builder still returns None rather than crashing.
        """
        with patch(
            "prsm.node.node._resolve_endpoints"
        ) as MockResolve, patch.dict(os.environ, {
            "PRSM_NETWORK": "mainnet",
        }, clear=False):
            os.environ.pop("PRSM_ROYALTY_DISTRIBUTOR_ADDRESS", None)
            MockResolve.return_value = MagicMock(royalty_distributor=None)
            assert _build_royalty_distributor_client_or_none() is None

    def test_returns_none_when_private_key_unset(self):
        with patch.dict(os.environ, {
            "PRSM_ROYALTY_DISTRIBUTOR_ADDRESS": "0x" + "ab" * 20,
        }, clear=False):
            os.environ.pop("FTNS_WALLET_PRIVATE_KEY", None)
            # RoyaltyDistributorClient supports read-only mode
            # (claimable() doesn't need a signer). Builder should
            # construct read-only when key absent — operators
            # running aggregate-source quoting without intent to
            # claim still benefit from claimable() reads.
            client = _build_royalty_distributor_client_or_none()
            # Returns a client (read-only) or None depending on
            # design choice. Test pins the design choice: read-only
            # is supported, so a client is returned.
            assert client is not None

    def test_returns_client_when_both_set(self):
        with patch(
            "prsm.economy.web3.royalty_distributor.RoyaltyDistributorClient"
        ) as MockClient, patch.dict(os.environ, {
            "PRSM_ROYALTY_DISTRIBUTOR_ADDRESS": "0x" + "ab" * 20,
            "FTNS_WALLET_PRIVATE_KEY": "0x" + "01" * 32,
            "FTNS_TOKEN_ADDRESS": "0x" + "cd" * 20,
        }, clear=False):
            MockClient.return_value = MagicMock()
            client = _build_royalty_distributor_client_or_none()
            assert client is not None
            MockClient.assert_called_once()
            kwargs = MockClient.call_args.kwargs
            assert kwargs["distributor_address"] == "0x" + "ab" * 20
            assert kwargs["ftns_token_address"] == "0x" + "cd" * 20
            assert kwargs["private_key"] == "0x" + "01" * 32

    def test_returns_none_when_construction_raises(self):
        with patch(
            "prsm.economy.web3.royalty_distributor.RoyaltyDistributorClient",
            side_effect=RuntimeError("rpc unreachable"),
        ), patch.dict(os.environ, {
            "PRSM_ROYALTY_DISTRIBUTOR_ADDRESS": "0x" + "ab" * 20,
            "FTNS_WALLET_PRIVATE_KEY": "0x" + "01" * 32,
        }, clear=False):
            assert _build_royalty_distributor_client_or_none() is None

    def test_uses_default_rpc_url_when_unset(self):
        with patch(
            "prsm.economy.web3.royalty_distributor.RoyaltyDistributorClient"
        ) as MockClient, patch.dict(os.environ, {
            "PRSM_ROYALTY_DISTRIBUTOR_ADDRESS": "0x" + "ab" * 20,
            "FTNS_WALLET_PRIVATE_KEY": "0x" + "01" * 32,
        }, clear=False):
            os.environ.pop("PRSM_BASE_RPC_URL", None)
            MockClient.return_value = MagicMock()
            _build_royalty_distributor_client_or_none()
            kwargs = MockClient.call_args.kwargs
            assert kwargs["rpc_url"] == "https://mainnet.base.org"
