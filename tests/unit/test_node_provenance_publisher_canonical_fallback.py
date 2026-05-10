"""Sprint 146 — canonical-address fallback for ProvenanceRegistry +
PublisherKeyAnchor builders.

Mirrors the sprint 144 work (royalty / phase 7-storage / phase 8
clients) by extending the same canonical-fallback semantics to the
remaining two address-resolving builder helpers in node.py:

  - _build_provenance_client_or_none           (PRSM-PROV-1 Item 4 + 6)
  - _build_publisher_key_anchor_client_or_none (T3c)

Operators who declare PRSM_NETWORK should not have to ALSO paste the
canonical contract address into a per-contract env var. networks.py
is canonical source of truth.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from prsm.node.node import (
    _build_provenance_client_or_none,
    _build_publisher_key_anchor_client_or_none,
)


# ──────────────────────────────────────────────────────────────────────
# ProvenanceRegistry builder
# ──────────────────────────────────────────────────────────────────────


class TestBuildProvenanceClientCanonicalFallback:
    def test_returns_none_when_address_unset_no_network(self):
        """Pre-existing behavior preserved: opt-in flag + no addr +
        no PRSM_NETWORK → None."""
        with patch.dict(os.environ, {
            "PRSM_ONCHAIN_PROVENANCE": "1",
            "FTNS_WALLET_PRIVATE_KEY": "0x" + "01" * 32,
        }, clear=False):
            os.environ.pop("PRSM_PROVENANCE_REGISTRY_ADDRESS", None)
            os.environ.pop("PRSM_NETWORK", None)
            assert _build_provenance_client_or_none() is None

    def test_falls_back_to_canonical_when_network_set(self):
        """Sprint 146 — operator declared PRSM_NETWORK=mainnet +
        opted into on-chain provenance → builder pulls canonical
        ProvenanceRegistry address from networks.py.

        Mainnet `provenance_registry` field still pins V1 for legacy
        Item 7 callers; V2 is surfaced separately via the
        `provenance_registry_v2` field. Match networks.py exactly.
        """
        canonical_v1 = "0xdF470BFa9eF310B196801D5105468515d0069915"
        with patch(
            "prsm.economy.web3.provenance_registry.ProvenanceRegistryClient"
        ) as MockClient, patch.dict(os.environ, {
            "PRSM_ONCHAIN_PROVENANCE": "1",
            "PRSM_NETWORK": "mainnet",
            "FTNS_WALLET_PRIVATE_KEY": "0x" + "01" * 32,
        }, clear=False):
            os.environ.pop("PRSM_PROVENANCE_REGISTRY_ADDRESS", None)
            MockClient.return_value = MagicMock()
            client = _build_provenance_client_or_none()
            assert client is not None
            kwargs = MockClient.call_args.kwargs
            assert kwargs["contract_address"] == canonical_v1

    def test_returns_none_when_opt_in_flag_unset(self):
        """Sprint 146 invariant: even with PRSM_NETWORK + canonical
        addr resolvable, no PRSM_ONCHAIN_PROVENANCE=1 means no
        on-chain registration — operator has to opt in."""
        with patch.dict(os.environ, {
            "PRSM_NETWORK": "mainnet",
            "FTNS_WALLET_PRIVATE_KEY": "0x" + "01" * 32,
        }, clear=False):
            os.environ.pop("PRSM_PROVENANCE_REGISTRY_ADDRESS", None)
            os.environ.pop("PRSM_ONCHAIN_PROVENANCE", None)
            assert _build_provenance_client_or_none() is None


# ──────────────────────────────────────────────────────────────────────
# PublisherKeyAnchor builder
# ──────────────────────────────────────────────────────────────────────


class TestBuildPublisherKeyAnchorCanonicalFallback:
    def test_returns_none_when_address_unset_no_network(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
            os.environ.pop("PRSM_NETWORK", None)
            assert _build_publisher_key_anchor_client_or_none() is None

    def test_falls_back_to_canonical_when_network_set(self):
        """Sprint 146 — when networks.py exposes a canonical anchor
        address for the resolved network, the builder picks it up.

        PublisherKeyAnchor is `None` for both mainnet and testnet in
        networks.py today (Phase 3.x.3 deploy is on Ethereum Sepolia,
        not Base). Mock _resolve_endpoints to simulate a future
        Base-deployment so the fallback path is exercised.
        """
        canonical = "0x" + "11" * 20
        with patch(
            "prsm.node.node._resolve_endpoints"
        ) as MockResolve, patch(
            "prsm.security.publisher_key_anchor.client.PublisherKeyAnchorClient"
        ) as MockClient, patch.dict(os.environ, {
            "PRSM_NETWORK": "mainnet",
        }, clear=False):
            os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
            MockResolve.return_value = MagicMock(
                publisher_key_anchor=canonical,
            )
            MockClient.return_value = MagicMock()
            client = _build_publisher_key_anchor_client_or_none()
            assert client is not None
            kwargs = MockClient.call_args.kwargs
            assert kwargs["contract_address"] == canonical

    def test_returns_none_when_network_resolves_to_no_canonical(self):
        """Network resolves but anchor field is None (current state for
        both mainnet and testnet in networks.py) → None."""
        with patch(
            "prsm.node.node._resolve_endpoints"
        ) as MockResolve, patch.dict(os.environ, {
            "PRSM_NETWORK": "mainnet",
        }, clear=False):
            os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
            MockResolve.return_value = MagicMock(publisher_key_anchor=None)
            assert _build_publisher_key_anchor_client_or_none() is None
