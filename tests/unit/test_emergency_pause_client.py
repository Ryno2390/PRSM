"""Sprint 298 — EmergencyPauseClient + Foundation Safe
pause-tx composer.

Vision §14 "Smart-contract exploit risk" names emergency pause
as the only mitigation that limits blast radius DURING an
active exploit. Foundation Safe is the 2-of-3 hardware
multisig that holds pause authority on every Phase 1.3 +
audit-bundle + Phase 7/8 contract (sole-owned via
acceptOwnership ceremonies 2026-05-04/06/07).

PRSM never executes pause directly. This client COMPOSES the
pause transaction that operators upload to the Safe UI for
multi-sig signing. The composer surface itself doesn't grant
authority — it only encodes the well-known OZ Pausable
selectors against the canonical contract addresses.

Function selectors (well-known OZ Pausable standard):
  pause()    0x8456cb59
  unpause()  0x3f4ba83a
  paused()   0x5c975abb

This sprint ships:
  PauseEligibleContract — dataclass for the registry
  PAUSE_ELIGIBLE_CONTRACTS — static registry of pausable
                              contracts (from networks.py)
  EmergencyPauseClient — composer + on-chain status reader
  ContractPauseStatus — per-contract status result
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.emergency_pause_client import (
    PAUSE_ELIGIBLE_CONTRACTS,
    PAUSE_SELECTOR,
    UNPAUSE_SELECTOR,
    PAUSED_SELECTOR,
    ContractPauseStatus,
    EmergencyPauseClient,
    PauseEligibleContract,
)


# ── PauseEligibleContract dataclass ──────────────────────


def test_contract_dataclass_to_dict():
    c = PauseEligibleContract(
        name="ftns_token",
        address="0xFFFF",
        kind="ERC20Pausable",
        description="FTNS ERC20 token transfers",
    )
    d = c.to_dict()
    assert d["name"] == "ftns_token"
    assert d["address"] == "0xFFFF"


# ── Registry sanity ──────────────────────────────────────


def test_registry_contains_expected_contracts():
    """Audit-bundle + Phase 1.3 + Phase 7/8 contracts that
    are owned by Foundation Safe with OZ Pausable should be
    in the registry."""
    names = {c.name for c in PAUSE_ELIGIBLE_CONTRACTS}
    assert "ftns_token" in names
    assert "royalty_distributor" in names
    # Audit-bundle Ownable2Step contracts
    assert "escrow_pool" in names
    assert "stake_bond" in names
    # Phase 7/8
    assert "storage_slashing" in names
    assert "compensation_distributor" in names


def test_registry_no_duplicate_names():
    names = [c.name for c in PAUSE_ELIGIBLE_CONTRACTS]
    assert len(names) == len(set(names))


def test_pause_selector_constants():
    """OZ Pausable selectors. Pin to detect any future
    drift in the spec."""
    assert PAUSE_SELECTOR == "0x8456cb59"
    assert UNPAUSE_SELECTOR == "0x3f4ba83a"
    assert PAUSED_SELECTOR == "0x5c975abb"


# ── ContractPauseStatus dataclass ────────────────────────


def test_pause_status_to_dict():
    s = ContractPauseStatus(
        name="ftns_token",
        address="0xabc",
        paused=False,
        commissioned=True,
        error=None,
    )
    d = s.to_dict()
    assert d["paused"] is False
    assert d["commissioned"] is True


# ── EmergencyPauseClient: PENDING_COMMISSION ─────────────


def test_uncommissioned_client_status_returns_not_configured():
    """When no contract addresses are wired (e.g., bare
    operator with no on-chain config), status_all returns
    NOT_CONFIGURED entries — operator sees the truth."""
    c = EmergencyPauseClient(
        contract_addresses={}, rpc_url=None,
    )
    statuses = c.status_all()
    assert len(statuses) == len(PAUSE_ELIGIBLE_CONTRACTS)
    for s in statuses.values():
        assert s.commissioned is False


def test_uncommissioned_compose_raises():
    """Composing a pause tx requires the target address.
    Missing config → raise so operators see the
    misconfiguration before trying to upload to the Safe."""
    c = EmergencyPauseClient(
        contract_addresses={}, rpc_url=None,
    )
    with pytest.raises(ValueError, match="not configured"):
        c.compose_pause_tx("ftns_token")


# ── EmergencyPauseClient: with addresses ─────────────────


class FakeChainBackend:
    """Test backend mirroring eth_call surface. Returns
    canned paused states."""

    def __init__(self, paused_map=None):
        self.paused_map = paused_map or {}
        self.calls = []

    def call(self, to_address, data):
        self.calls.append((to_address, data))
        # OZ Pausable.paused() returns ABI-encoded bool
        # (32 bytes big-endian; last byte = 1 if paused)
        is_paused = self.paused_map.get(to_address, False)
        return b"\x00" * 31 + (b"\x01" if is_paused else b"\x00")


def _client_with_addresses(paused_map=None):
    return EmergencyPauseClient(
        contract_addresses={
            "ftns_token": "0xFFFF" + "0" * 36,
            "royalty_distributor": "0xAAAA" + "0" * 36,
            "escrow_pool": "0xBBBB" + "0" * 36,
            "stake_bond": "0xCCCC" + "0" * 36,
            "compensation_distributor": "0xDDDD" + "0" * 36,
            "storage_slashing": "0xEEEE" + "0" * 36,
            "settlement_registry": "0x1111" + "0" * 36,
            "signature_verifier": "0x2222" + "0" * 36,
            "emission_controller": "0x3333" + "0" * 36,
            "key_distribution": "0x4444" + "0" * 36,
        },
        rpc_url="https://rpc.example",
        backend=FakeChainBackend(paused_map),
    )


def test_compose_pause_tx_happy_path():
    c = _client_with_addresses()
    tx = c.compose_pause_tx("ftns_token")
    # Returns a Safe-uploadable dict
    assert tx["to"] == "0xFFFF" + "0" * 36
    assert tx["data"] == PAUSE_SELECTOR
    assert tx["value"] == "0"
    assert tx["action"] == "pause"
    assert "ftns_token" in tx["description"].lower()
    # Operator-facing warning surfaces consequences
    assert "warning" in tx or "consequences" in tx


def test_compose_unpause_tx_happy_path():
    c = _client_with_addresses()
    tx = c.compose_unpause_tx("ftns_token")
    assert tx["data"] == UNPAUSE_SELECTOR
    assert tx["action"] == "unpause"


def test_compose_unknown_contract_raises():
    c = _client_with_addresses()
    with pytest.raises(ValueError, match="not in registry"):
        c.compose_pause_tx("not_a_contract")


def test_compose_returns_chain_id_when_set():
    c = EmergencyPauseClient(
        contract_addresses={"ftns_token": "0xFFFF" + "0" * 36},
        rpc_url="https://rpc.example",
        chain_id=8453,
    )
    tx = c.compose_pause_tx("ftns_token")
    assert tx["chain_id"] == 8453


# ── is_paused / status_all ───────────────────────────────


def test_is_paused_true_when_backend_reports_paused():
    c = _client_with_addresses(
        paused_map={"0xFFFF" + "0" * 36: True},
    )
    assert c.is_paused("ftns_token") is True


def test_is_paused_false_default():
    c = _client_with_addresses()
    assert c.is_paused("ftns_token") is False


def test_is_paused_unknown_contract_raises():
    c = _client_with_addresses()
    with pytest.raises(ValueError):
        c.is_paused("not_a_contract")


def test_is_paused_uncommissioned_returns_none():
    """No address configured → returns None (callers
    distinguish 'unpaused' from 'unknown')."""
    c = EmergencyPauseClient(
        contract_addresses={}, rpc_url=None,
    )
    assert c.is_paused("ftns_token") is None


def test_is_paused_backend_exception_fail_soft():
    """RPC error → returns None + status record carries
    error string. Never raises out of is_paused — operators
    monitoring pause state should not get exceptions on
    transient RPC outages."""
    class BoomBackend:
        def call(self, to, data):
            raise RuntimeError("RPC down")
    c = EmergencyPauseClient(
        contract_addresses={
            "ftns_token": "0xFFFF" + "0" * 36,
        },
        rpc_url="https://rpc.example",
        backend=BoomBackend(),
    )
    assert c.is_paused("ftns_token") is None


def test_status_all_bulk_query():
    c = _client_with_addresses(
        paused_map={"0xFFFF" + "0" * 36: True},
    )
    statuses = c.status_all()
    # ftns_token is paused
    assert statuses["ftns_token"].paused is True
    # Others are not paused
    assert statuses["royalty_distributor"].paused is False
    # All commissioned (have address)
    for s in statuses.values():
        assert s.commissioned is True


def test_status_all_mixes_commissioned_and_uncommissioned():
    """A registry where some contracts have addresses but
    others don't. The uncommissioned entries surface as
    commissioned=False + paused=None."""
    c = EmergencyPauseClient(
        contract_addresses={
            "ftns_token": "0xFFFF" + "0" * 36,
            # other contracts deliberately absent
        },
        rpc_url="https://rpc.example",
        backend=FakeChainBackend(),
    )
    statuses = c.status_all()
    assert statuses["ftns_token"].commissioned is True
    assert statuses["royalty_distributor"].commissioned is False
    assert statuses["royalty_distributor"].paused is None


# ── from_env factory ─────────────────────────────────────


def test_from_env_uses_mainnet_addresses_by_default(
    monkeypatch,
):
    """from_env without PRSM_NETWORK env defaults to mainnet
    and pulls addresses from networks.py registry."""
    monkeypatch.delenv("PRSM_NETWORK", raising=False)
    monkeypatch.delenv("BASE_RPC_URL", raising=False)
    c = EmergencyPauseClient.from_env()
    # All major audit-bundle contracts should have addresses
    # populated from MAINNET
    statuses = c.status_all()
    assert (
        statuses["ftns_token"].address
        == "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
    )


def test_from_env_respects_testnet_selection(monkeypatch):
    monkeypatch.setenv("PRSM_NETWORK", "testnet")
    c = EmergencyPauseClient.from_env()
    statuses = c.status_all()
    # Testnet ftns_token address
    assert (
        statuses["ftns_token"].address
        == "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a"
    )


# ── tx-payload determinism ───────────────────────────────


def test_compose_pause_deterministic():
    """Same client + same contract → same tx (modulo
    nonce/gas which the Safe UI fills in). The data + to +
    value MUST be byte-identical."""
    c = _client_with_addresses()
    tx1 = c.compose_pause_tx("ftns_token")
    tx2 = c.compose_pause_tx("ftns_token")
    assert tx1["to"] == tx2["to"]
    assert tx1["data"] == tx2["data"]
    assert tx1["value"] == tx2["value"]


# ── Operator-facing safety guards ────────────────────────


def test_tx_payload_includes_explorer_url_when_chain_id_set():
    """For mainnet (chain_id=8453), the composed tx should
    surface a basescan URL so operators can verify the
    target contract before signing."""
    c = EmergencyPauseClient(
        contract_addresses={
            "ftns_token": "0xFFFF" + "0" * 36,
        },
        rpc_url="https://rpc.example",
        chain_id=8453,
    )
    tx = c.compose_pause_tx("ftns_token")
    assert "explorer_url" in tx
    assert "basescan" in tx["explorer_url"].lower()


def test_tx_payload_warning_is_explicit():
    """Pause is destructive (halts user transfers). The tx
    payload MUST carry an unmissable warning string for
    operators."""
    c = _client_with_addresses()
    tx = c.compose_pause_tx("ftns_token")
    warning = tx.get("warning", "")
    assert "destructive" in warning.lower() or "halt" in warning.lower()
    assert "multisig" in warning.lower() or "safe" in warning.lower()


# ── PauseEligibleContract resolution ─────────────────────


def test_get_contract_by_name():
    c = _client_with_addresses()
    contract = c.get_contract("ftns_token")
    assert contract.name == "ftns_token"


def test_get_unknown_contract_raises():
    c = _client_with_addresses()
    with pytest.raises(ValueError):
        c.get_contract("not_a_contract")
