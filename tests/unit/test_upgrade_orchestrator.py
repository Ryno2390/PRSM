"""Sprint 303 — UUPS upgrade orchestrator (Vision §14 item 7).

Vision §14 item 7: "UUPS upgrade pattern for non-immutable
contracts permits patching if vulnerability is discovered
post-deployment."

This sprint ships the engineering layer: an UpgradeProposal
record with a strict workflow + composer-only Safe payloads
for `upgradeToAndCall(newImpl, data)`. Each proposal pins
the rationale + reviewer assignments and supports a
PRE-COMMITTED rollback path — when an upgrade ships, the
prior implementation address is captured so the operator
can compose a rollback tx without needing to look up the
previous version under incident pressure.

R-2026-05-08-1 composer-only invariant preserved: PRSM
never executes upgrades; the Foundation Safe 2-of-3
hardware multisig is the gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prsm.economy.web3.upgrade_orchestrator import (
    UPGRADE_TO_AND_CALL_SELECTOR,
    UpgradeOrchestrator,
    UpgradeProposal,
    UpgradeSeverity,
    UpgradeStatus,
    compose_upgrade_tx,
    compose_rollback_tx,
    encode_upgrade_to_and_call_calldata,
    _TERMINAL_STATUSES,
)


# ── Enums ────────────────────────────────────────────


def test_severity_values():
    assert UpgradeSeverity.EMERGENCY.value == "emergency"
    assert UpgradeSeverity.PLANNED.value == "planned"
    assert UpgradeSeverity.MAINTENANCE.value == "maintenance"


def test_status_values():
    assert UpgradeStatus.PROPOSED.value == "proposed"
    assert UpgradeStatus.REVIEWED.value == "reviewed"
    assert UpgradeStatus.SAFE_UPLOADED.value == "safe_uploaded"
    assert UpgradeStatus.EXECUTED.value == "executed"
    assert UpgradeStatus.ROLLED_BACK.value == "rolled_back"
    assert UpgradeStatus.REJECTED.value == "rejected"


def test_terminal_statuses():
    # EXECUTED is NOT terminal — operators must retain the
    # ability to ROLL_BACK from EXECUTED (the whole point of
    # the pre-committed rollback escape).
    assert UpgradeStatus.EXECUTED not in _TERMINAL_STATUSES
    assert UpgradeStatus.ROLLED_BACK in _TERMINAL_STATUSES
    assert UpgradeStatus.REJECTED in _TERMINAL_STATUSES
    assert UpgradeStatus.PROPOSED not in _TERMINAL_STATUSES


# ── Calldata encoding ────────────────────────────────


def test_uups_selector_pinned():
    # upgradeToAndCall(address,bytes) — first 4 bytes of
    # keccak256("upgradeToAndCall(address,bytes)")
    assert UPGRADE_TO_AND_CALL_SELECTOR == "0x4f1ef286"


def test_encode_calldata_no_init_data():
    new_impl = "0x" + "ab" * 20
    data = encode_upgrade_to_and_call_calldata(
        new_impl, b"",
    )
    assert data.startswith(UPGRADE_TO_AND_CALL_SELECTOR)
    # selector(4) + address(32) + offset(32) + length(32)
    # = 100 bytes = 200 hex chars + 0x prefix = 202 chars
    assert len(data) == 202


def test_encode_calldata_with_init_data():
    new_impl = "0x" + "ab" * 20
    init = b"\xde\xad\xbe\xef"
    data = encode_upgrade_to_and_call_calldata(
        new_impl, init,
    )
    assert data.startswith(UPGRADE_TO_AND_CALL_SELECTOR)
    # Init data padded to 32-byte boundary
    assert "deadbeef" in data


def test_encode_calldata_invalid_address():
    with pytest.raises(ValueError, match="address"):
        encode_upgrade_to_and_call_calldata("notaddr", b"")


# ── UpgradeProposal ──────────────────────────────────


def test_proposal_round_trip():
    p = UpgradeProposal(
        proposal_id="up-1",
        opened_ts=100.0,
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.EMERGENCY,
        rationale="reentrancy fix",
        status=UpgradeStatus.PROPOSED,
        init_calldata_hex="0x",
        reviewer_assignments=["alice", "bob"],
        safe_tx_hash=None,
    )
    d = p.to_dict()
    restored = UpgradeProposal.from_dict(d)
    assert restored == p


# ── Orchestrator mutations ───────────────────────────


def test_propose_assigns_id():
    o = UpgradeOrchestrator()
    p = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.PLANNED,
        rationale="x",
    )
    assert p.proposal_id
    assert p.status == UpgradeStatus.PROPOSED


def test_propose_rejects_empty_rationale():
    o = UpgradeOrchestrator()
    with pytest.raises(ValueError, match="rationale"):
        o.propose(
            target_proxy="0x" + "aa" * 20,
            new_implementation="0x" + "bb" * 20,
            previous_implementation="0x" + "cc" * 20,
            severity=UpgradeSeverity.PLANNED,
            rationale="",
        )


def test_propose_rejects_non_enum_severity():
    o = UpgradeOrchestrator()
    with pytest.raises(ValueError, match="severity"):
        o.propose(
            target_proxy="0x" + "aa" * 20,
            new_implementation="0x" + "bb" * 20,
            previous_implementation="0x" + "cc" * 20,
            severity="urgent",  # type: ignore
            rationale="x",
        )


def test_propose_rejects_invalid_address():
    o = UpgradeOrchestrator()
    with pytest.raises(ValueError, match="address"):
        o.propose(
            target_proxy="not-an-addr",
            new_implementation="0x" + "bb" * 20,
            previous_implementation="0x" + "cc" * 20,
            severity=UpgradeSeverity.PLANNED,
            rationale="x",
        )


def test_propose_rejects_same_old_new_impl():
    o = UpgradeOrchestrator()
    with pytest.raises(ValueError, match="same"):
        o.propose(
            target_proxy="0x" + "aa" * 20,
            new_implementation="0x" + "bb" * 20,
            previous_implementation="0x" + "bb" * 20,
            severity=UpgradeSeverity.PLANNED,
            rationale="x",
        )


def test_update_status_workflow_forward():
    o = UpgradeOrchestrator()
    p = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.PLANNED,
        rationale="x",
    )
    o.update_status(p.proposal_id, UpgradeStatus.REVIEWED)
    o.update_status(
        p.proposal_id, UpgradeStatus.SAFE_UPLOADED,
    )
    o.update_status(
        p.proposal_id, UpgradeStatus.EXECUTED,
        safe_tx_hash="0xdeadbeef",
    )
    final = o.get(p.proposal_id)
    assert final.status == UpgradeStatus.EXECUTED
    assert final.safe_tx_hash == "0xdeadbeef"


def test_update_status_rejects_terminal_mutation():
    o = UpgradeOrchestrator()
    p = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.PLANNED,
        rationale="x",
    )
    o.update_status(p.proposal_id, UpgradeStatus.REJECTED)
    with pytest.raises(ValueError, match="terminal"):
        o.update_status(
            p.proposal_id, UpgradeStatus.REVIEWED,
        )


def test_update_status_rejects_back_to_proposed():
    o = UpgradeOrchestrator()
    p = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.PLANNED,
        rationale="x",
    )
    o.update_status(p.proposal_id, UpgradeStatus.REVIEWED)
    with pytest.raises(ValueError, match="back to"):
        o.update_status(
            p.proposal_id, UpgradeStatus.PROPOSED,
        )


def test_update_status_unknown_id():
    o = UpgradeOrchestrator()
    with pytest.raises(ValueError, match="not found"):
        o.update_status(
            "no-such", UpgradeStatus.REVIEWED,
        )


# ── Queries ──────────────────────────────────────────


def test_get_returns_none_unknown():
    assert UpgradeOrchestrator().get("nope") is None


def test_list_filter_by_status():
    o = UpgradeOrchestrator()
    a = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.PLANNED,
        rationale="a",
    )
    o.propose(
        target_proxy="0x" + "11" * 20,
        new_implementation="0x" + "22" * 20,
        previous_implementation="0x" + "33" * 20,
        severity=UpgradeSeverity.PLANNED,
        rationale="b",
    )
    o.update_status(a.proposal_id, UpgradeStatus.REVIEWED)
    out = o.list(status=UpgradeStatus.REVIEWED)
    assert len(out) == 1
    assert out[0].rationale == "a"


# ── compose_upgrade_tx ───────────────────────────────


def test_compose_upgrade_happy_path():
    o = UpgradeOrchestrator()
    p = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.EMERGENCY,
        rationale="reentrancy",
    )
    o.update_status(p.proposal_id, UpgradeStatus.REVIEWED)
    tx = compose_upgrade_tx(
        orchestrator=o,
        proposal_id=p.proposal_id,
        chain_id=8453,
    )
    assert tx["to"] == "0x" + "aa" * 20
    assert tx["data"].startswith(
        UPGRADE_TO_AND_CALL_SELECTOR,
    )
    assert tx["value"] == "0"
    assert tx["action"] == "upgrade"
    assert "WARNING" in tx["warning"].upper() or (
        "destructive" in tx["warning"].lower()
    )


def test_compose_upgrade_requires_reviewed_or_above():
    o = UpgradeOrchestrator()
    p = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.PLANNED,
        rationale="x",
    )
    # Still PROPOSED — must be REVIEWED before composing
    with pytest.raises(ValueError, match="reviewed"):
        compose_upgrade_tx(
            orchestrator=o, proposal_id=p.proposal_id,
        )


def test_compose_upgrade_rejects_terminal():
    o = UpgradeOrchestrator()
    p = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.PLANNED,
        rationale="x",
    )
    o.update_status(p.proposal_id, UpgradeStatus.REJECTED)
    with pytest.raises(ValueError, match="terminal"):
        compose_upgrade_tx(
            orchestrator=o, proposal_id=p.proposal_id,
        )


def test_compose_upgrade_unknown_id():
    o = UpgradeOrchestrator()
    with pytest.raises(ValueError, match="not found"):
        compose_upgrade_tx(
            orchestrator=o, proposal_id="nope",
        )


# ── compose_rollback_tx ──────────────────────────────


def test_compose_rollback_happy_path():
    """The pre-committed rollback escape — operator can
    compose a rollback to previous_implementation WITHOUT
    needing to dig up the prior version under pressure."""
    o = UpgradeOrchestrator()
    p = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.EMERGENCY,
        rationale="rollback test",
    )
    # Walk to EXECUTED — rollback is only valid after the
    # upgrade actually shipped
    o.update_status(p.proposal_id, UpgradeStatus.REVIEWED)
    o.update_status(
        p.proposal_id, UpgradeStatus.SAFE_UPLOADED,
    )
    o.update_status(
        p.proposal_id, UpgradeStatus.EXECUTED,
        safe_tx_hash="0xdeadbeef",
    )
    tx = compose_rollback_tx(
        orchestrator=o,
        proposal_id=p.proposal_id,
        chain_id=8453,
    )
    assert tx["to"] == "0x" + "aa" * 20
    assert tx["action"] == "rollback"
    # Rollback should target the PREVIOUS impl, encoded into
    # the calldata
    assert "cc" * 20 in tx["data"]


def test_compose_rollback_requires_executed():
    """Can't roll back what hasn't shipped."""
    o = UpgradeOrchestrator()
    p = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.PLANNED,
        rationale="x",
    )
    o.update_status(p.proposal_id, UpgradeStatus.REVIEWED)
    with pytest.raises(ValueError, match="executed"):
        compose_rollback_tx(
            orchestrator=o, proposal_id=p.proposal_id,
        )


def test_compose_rollback_rejects_already_rolled_back():
    o = UpgradeOrchestrator()
    p = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.EMERGENCY,
        rationale="x",
    )
    o.update_status(p.proposal_id, UpgradeStatus.REVIEWED)
    o.update_status(
        p.proposal_id, UpgradeStatus.SAFE_UPLOADED,
    )
    o.update_status(
        p.proposal_id, UpgradeStatus.EXECUTED,
        safe_tx_hash="0x1",
    )
    o.update_status(
        p.proposal_id, UpgradeStatus.ROLLED_BACK,
    )
    with pytest.raises(ValueError, match="rolled"):
        compose_rollback_tx(
            orchestrator=o, proposal_id=p.proposal_id,
        )


# ── Persistence ──────────────────────────────────────


def test_persist_round_trip(tmp_path: Path):
    o = UpgradeOrchestrator(persist_dir=tmp_path)
    p = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.PLANNED,
        rationale="x",
    )
    o.update_status(p.proposal_id, UpgradeStatus.REVIEWED)
    o2 = UpgradeOrchestrator(persist_dir=tmp_path)
    loaded = o2.get(p.proposal_id)
    assert loaded is not None
    assert loaded.status == UpgradeStatus.REVIEWED


def test_persist_corrupt_file_failsoft(tmp_path: Path):
    (tmp_path / "broken.json").write_text("{not json")
    o = UpgradeOrchestrator(persist_dir=tmp_path)
    assert o.count() == 0


def test_from_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(
        "PRSM_UPGRADE_ORCHESTRATOR_DIR", str(tmp_path),
    )
    o = UpgradeOrchestrator.from_env()
    p = o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.PLANNED,
        rationale="x",
    )
    files = list(tmp_path.glob("*.json"))
    assert files
    on_disk = json.loads(files[0].read_text())
    assert on_disk["proposal_id"] == p.proposal_id
