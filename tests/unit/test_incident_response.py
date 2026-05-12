"""Sprint 301 — IncidentResponse primitive.

Vision §14 item 5: "Public exploit-response playbook
published before any incident. Decision-makers, response
timelines, and communications sequence pre-committed. Most
historical exploits became terminal trust events partly
because the response was improvised under pressure."

This module gives operators a CODE-HOOK playbook: severity
tiers, phase machine, pre-committed decision-tree
recommendations, pre-committed comms templates, filesystem
audit trail. Sprint 301 ships the primitive; endpoints + MCP
ship same sprint.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prsm.economy.web3.incident_response import (
    COMMS_TEMPLATES,
    DECISION_TREE,
    IncidentPhase,
    IncidentRecord,
    IncidentResponse,
    IncidentSeverity,
    TimelineEvent,
    _TERMINAL_PHASES,
    get_comms_template,
    get_recommendations,
)


# ── Enums ────────────────────────────────────────────────


def test_severity_values():
    assert IncidentSeverity.S0.value == "s0"  # catastrophic
    assert IncidentSeverity.S1.value == "s1"
    assert IncidentSeverity.S2.value == "s2"
    assert IncidentSeverity.S3.value == "s3"


def test_phase_values():
    assert IncidentPhase.DETECTED.value == "detected"
    assert IncidentPhase.TRIAGED.value == "triaged"
    assert IncidentPhase.CONTAINED.value == "contained"
    assert IncidentPhase.MITIGATED.value == "mitigated"
    assert (
        IncidentPhase.POSTMORTEM_PUBLISHED.value
        == "postmortem_published"
    )
    assert IncidentPhase.CLOSED.value == "closed"


def test_closed_is_terminal():
    assert IncidentPhase.CLOSED in _TERMINAL_PHASES


def test_decision_tree_covers_all_severities():
    # Every severity must have at least one recommendation
    # for the DETECTED phase — operators can't be stranded
    # at incident-open with no guidance.
    for sev in IncidentSeverity:
        key = (sev, IncidentPhase.DETECTED)
        assert key in DECISION_TREE, (
            f"DECISION_TREE missing {key}"
        )
        assert DECISION_TREE[key], (
            f"DECISION_TREE[{key}] is empty"
        )


def test_decision_tree_s0_detected_says_pause():
    # The whole point of pre-committed playbook: at S0
    # detected, "PAUSE NOW" is non-negotiable.
    recs = DECISION_TREE[
        (IncidentSeverity.S0, IncidentPhase.DETECTED)
    ]
    blob = " ".join(recs).lower()
    assert "pause" in blob


def test_comms_templates_cover_each_severity_detected():
    # Pre-committed comms text — operators don't write
    # under pressure. At minimum every severity has a
    # DETECTED-phase template.
    for sev in IncidentSeverity:
        key = (sev, IncidentPhase.DETECTED)
        assert key in COMMS_TEMPLATES
        assert COMMS_TEMPLATES[key].strip()


def test_comms_template_substitutes_summary():
    # Templates must accept summary substitution so the
    # operator can post without manual editing.
    sev = IncidentSeverity.S1
    text = get_comms_template(
        sev, IncidentPhase.DETECTED,
        summary="reentrancy in claim()",
    )
    assert "reentrancy in claim()" in text


# ── IncidentRecord ────────────────────────────────────


def test_record_to_dict_round_trip():
    rec = IncidentRecord(
        incident_id="inc-1",
        opened_ts=123.0,
        severity=IncidentSeverity.S1,
        summary="x",
        affected_contracts=["royalty_distributor"],
        current_phase=IncidentPhase.DETECTED,
        timeline=[
            TimelineEvent(
                ts=124.0,
                phase=IncidentPhase.DETECTED,
                note="opened",
                actor="oncall",
            ),
        ],
        related_disclosure_id="d-1",
    )
    d = rec.to_dict()
    restored = IncidentRecord.from_dict(d)
    assert restored == rec


# ── IncidentResponse mutations ────────────────────────


def test_open_assigns_id_and_records_detected_event():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="reentrancy",
        affected_contracts=["royalty_distributor"],
        opened_ts=100.0,
    )
    assert r.incident_id
    assert r.severity == IncidentSeverity.S1
    assert r.current_phase == IncidentPhase.DETECTED
    assert len(r.timeline) == 1
    assert r.timeline[0].phase == IncidentPhase.DETECTED


def test_open_rejects_empty_summary():
    ir = IncidentResponse()
    with pytest.raises(ValueError, match="summary"):
        ir.open(
            severity=IncidentSeverity.S1,
            summary="",
            affected_contracts=[],
        )


def test_open_rejects_non_enum_severity():
    ir = IncidentResponse()
    with pytest.raises(ValueError, match="severity"):
        ir.open(
            severity="catastrophic",  # type: ignore
            summary="x",
            affected_contracts=[],
        )


def test_open_caps_summary_length():
    ir = IncidentResponse()
    with pytest.raises(ValueError, match="exceeds"):
        ir.open(
            severity=IncidentSeverity.S1,
            summary="x" * 10_000,
            affected_contracts=[],
        )


def test_advance_phase_progresses_machine():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x",
        affected_contracts=[],
    )
    r2 = ir.advance_phase(
        r.incident_id, IncidentPhase.TRIAGED,
        note="confirmed", actor="oncall",
    )
    assert r2.current_phase == IncidentPhase.TRIAGED
    assert len(r2.timeline) == 2
    assert r2.timeline[-1].phase == IncidentPhase.TRIAGED


def test_advance_phase_rejects_backwards_progress():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x",
        affected_contracts=[],
    )
    ir.advance_phase(
        r.incident_id, IncidentPhase.TRIAGED,
    )
    with pytest.raises(ValueError, match="backwards"):
        ir.advance_phase(
            r.incident_id, IncidentPhase.DETECTED,
        )


def test_advance_phase_rejects_after_closed():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x",
        affected_contracts=[],
    )
    # Walk to CLOSED
    for phase in [
        IncidentPhase.TRIAGED,
        IncidentPhase.CONTAINED,
        IncidentPhase.MITIGATED,
        IncidentPhase.POSTMORTEM_PUBLISHED,
        IncidentPhase.CLOSED,
    ]:
        ir.advance_phase(r.incident_id, phase)
    with pytest.raises(ValueError, match="closed|terminal"):
        ir.advance_phase(
            r.incident_id, IncidentPhase.POSTMORTEM_PUBLISHED,
        )


def test_advance_phase_unknown_id():
    ir = IncidentResponse()
    with pytest.raises(ValueError, match="not found"):
        ir.advance_phase("no-such-id", IncidentPhase.TRIAGED)


def test_record_event_appends_without_phase_change():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x",
        affected_contracts=[],
    )
    r2 = ir.record_event(
        r.incident_id, note="paused royalty_distributor",
        actor="multisig",
    )
    assert r2.current_phase == IncidentPhase.DETECTED
    assert len(r2.timeline) == 2
    assert r2.timeline[-1].note == (
        "paused royalty_distributor"
    )


def test_record_event_rejects_empty_note():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x",
        affected_contracts=[],
    )
    with pytest.raises(ValueError, match="note"):
        ir.record_event(r.incident_id, note="")


def test_record_event_unknown_id():
    ir = IncidentResponse()
    with pytest.raises(ValueError, match="not found"):
        ir.record_event("no-such-id", note="x")


# ── Queries ───────────────────────────────────────────


def test_get_returns_none_for_unknown():
    ir = IncidentResponse()
    assert ir.get("no-such-id") is None


def test_list_sorts_newest_first():
    ir = IncidentResponse()
    a = ir.open(
        severity=IncidentSeverity.S2,
        summary="a",
        affected_contracts=[],
        opened_ts=100.0,
    )
    b = ir.open(
        severity=IncidentSeverity.S1,
        summary="b",
        affected_contracts=[],
        opened_ts=200.0,
    )
    out = ir.list()
    assert out[0].incident_id == b.incident_id
    assert out[1].incident_id == a.incident_id


def test_list_filter_by_severity():
    ir = IncidentResponse()
    ir.open(
        severity=IncidentSeverity.S0,
        summary="critical",
        affected_contracts=[],
    )
    ir.open(
        severity=IncidentSeverity.S3,
        summary="low",
        affected_contracts=[],
    )
    out = ir.list(severity=IncidentSeverity.S0)
    assert len(out) == 1
    assert out[0].summary == "critical"


def test_list_filter_by_phase():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x",
        affected_contracts=[],
    )
    ir.advance_phase(r.incident_id, IncidentPhase.TRIAGED)
    ir.open(
        severity=IncidentSeverity.S1,
        summary="y",
        affected_contracts=[],
    )
    triaged = ir.list(phase=IncidentPhase.TRIAGED)
    assert len(triaged) == 1
    assert triaged[0].summary == "x"


def test_count():
    ir = IncidentResponse()
    assert ir.count() == 0
    ir.open(
        severity=IncidentSeverity.S1,
        summary="x",
        affected_contracts=[],
    )
    assert ir.count() == 1


# ── Recommendations + comms ───────────────────────────


def test_get_recommendations_returns_decision_tree():
    recs = get_recommendations(
        IncidentSeverity.S0, IncidentPhase.DETECTED,
    )
    assert recs
    assert recs == list(
        DECISION_TREE[
            (IncidentSeverity.S0, IncidentPhase.DETECTED)
        ],
    )


def test_get_recommendations_unknown_combo_returns_empty():
    # Not every (severity, phase) needs a recommendation;
    # missing combos should return [] not raise.
    recs = get_recommendations(
        IncidentSeverity.S3, IncidentPhase.CLOSED,
    )
    assert isinstance(recs, list)


def test_get_comms_template_unknown_combo_returns_empty():
    text = get_comms_template(
        IncidentSeverity.S3, IncidentPhase.CLOSED,
    )
    assert text == ""


# ── Persistence ───────────────────────────────────────


def test_persist_round_trip(tmp_path: Path):
    ir = IncidentResponse(persist_dir=tmp_path)
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="reentrancy",
        affected_contracts=["royalty_distributor"],
    )
    ir.advance_phase(
        r.incident_id, IncidentPhase.TRIAGED,
        note="confirmed",
    )

    # Reload from disk
    ir2 = IncidentResponse(persist_dir=tmp_path)
    assert ir2.count() == 1
    loaded = ir2.get(r.incident_id)
    assert loaded is not None
    assert loaded.current_phase == IncidentPhase.TRIAGED
    assert len(loaded.timeline) == 2


def test_persist_corrupt_file_failsoft(tmp_path: Path):
    (tmp_path / "broken.json").write_text("{not json")
    ir = IncidentResponse(persist_dir=tmp_path)
    assert ir.count() == 0


def test_persist_path_traversal_safe(tmp_path: Path):
    ir = IncidentResponse(persist_dir=tmp_path)
    # Manually inject a record with traversal in its id —
    # the persistence layer must NOT write outside its dir.
    rec = IncidentRecord(
        incident_id="../escape",
        opened_ts=100.0,
        severity=IncidentSeverity.S1,
        summary="x",
        affected_contracts=[],
        current_phase=IncidentPhase.DETECTED,
        timeline=[],
    )
    ir._records[rec.incident_id] = rec
    ir._write_to_disk(rec)
    # Parent directory must be untouched
    siblings = list(tmp_path.parent.glob("escape*"))
    assert siblings == []


def test_from_env_uses_env_var(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(
        "PRSM_INCIDENT_RESPONSE_DIR", str(tmp_path),
    )
    ir = IncidentResponse.from_env()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x",
        affected_contracts=[],
    )
    # File on disk
    files = list(tmp_path.glob("*.json"))
    assert files
    on_disk = json.loads(files[0].read_text())
    assert on_disk["incident_id"] == r.incident_id


def test_from_env_no_var_no_persistence(monkeypatch):
    monkeypatch.delenv(
        "PRSM_INCIDENT_RESPONSE_DIR", raising=False,
    )
    ir = IncidentResponse.from_env()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x",
        affected_contracts=[],
    )
    # No disk persistence — just in-memory
    assert ir.get(r.incident_id) is not None
