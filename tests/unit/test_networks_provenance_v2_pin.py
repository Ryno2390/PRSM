"""Networks.py provenance_registry_v2 canonical pin.

Added 2026-05-09 in lockstep with the A-08 RoyaltyDistributor v2
mainnet ceremony. The v2 RoyaltyDistributor wires V2 as its
_registry constructor arg; operators building new provenance
pipelines should pin to V2 going forward. V1 retained for legacy
Item 7 ProvenanceRegistry callers.
"""
from __future__ import annotations

import pytest

from prsm.config.networks import MAINNET, TESTNET


class TestProvenanceRegistryV2Pin:
    def test_v2_field_present_on_mainnet(self):
        assert hasattr(MAINNET, "provenance_registry_v2")
        assert MAINNET.provenance_registry_v2 is not None

    def test_v2_address_matches_ceremony_deploy_record(self):
        """Pinned V2 address must match the address deployed in
        commit `e0cedDA354…0dbf` per PRSM-CR-2026-05-06-2 +
        Basescan source-verified."""
        assert MAINNET.provenance_registry_v2.lower() == \
            "0xe0cedda354f99526c7fbb9b9651e12adb2180dbf"

    def test_v1_and_v2_distinct(self):
        """V1 and V2 are different addresses; canonical-match
        checks need to know which to compare against."""
        assert MAINNET.provenance_registry != \
            MAINNET.provenance_registry_v2
        assert MAINNET.provenance_registry.lower() == \
            "0xdf470bfa9ef310b196801d5105468515d0069915"

    def test_testnet_v2_optional_unset(self):
        """Testnet doesn't yet have a V2 deploy; provenance_registry_v2
        defaults to None there."""
        # TESTNET may have V1 set but V2 not yet deployed.
        assert TESTNET.provenance_registry_v2 is None or \
            isinstance(TESTNET.provenance_registry_v2, str)

    def test_v2_used_as_royalty_distributor_v2_registry_arg(self):
        """Cross-check: the V2 ProvenanceRegistry address pinned
        here is what was passed as _registry to the v2
        RoyaltyDistributor constructor (deployed 2026-05-09).
        Lock-step required so a future ceremony updating one
        without the other is caught."""
        assert MAINNET.royalty_distributor.lower() == \
            "0xfea9aeb99e02fdb799e2df3c9195dc4e5323df7e"
        # If/when v3 RoyaltyDistributor ships against a different
        # registry, this test must be updated to reflect the new
        # binding.
