"""Sprint 301 — IncidentResponse + pre-committed playbook.

Vision §14 mitigation item 5: "Public exploit-response
playbook published before any incident. Decision-makers,
response timelines, and communications sequence
pre-committed. Most historical exploits became terminal
trust events partly because the response was improvised
under pressure."

This module gives operators a CODE-HOOK playbook:

  IncidentSeverity — S0 (catastrophic, paused mainnet
                     unrecoverable) / S1 (critical) /
                     S2 (high) / S3 (low or near-miss)
  IncidentPhase    — detected → triaged → contained →
                     mitigated → postmortem_published →
                     closed (one-way phase machine)
  DECISION_TREE    — pre-committed (severity, phase) →
                     [recommendation strings] map
  COMMS_TEMPLATES  — pre-committed (severity, phase) →
                     markdown template (supports
                     {summary} substitution)
  IncidentRecord   — persistent record + timeline
  IncidentResponse — open / advance_phase / record_event /
                     get / list / count
                     filesystem-persisted same pattern as
                     sprint 300 DisclosureIntake

The playbook is PUBLIC by design — anyone may read
`COMMS_TEMPLATES` + `DECISION_TREE` to verify what PRSM
has pre-committed to in an incident. That's the §14
promise.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IncidentSeverity(str, Enum):
    S0 = "s0"  # catastrophic, active drain in progress
    S1 = "s1"  # critical, confirmed exploitable
    S2 = "s2"  # high, plausible but not confirmed
    S3 = "s3"  # low / near-miss / informational


class IncidentPhase(str, Enum):
    DETECTED = "detected"
    TRIAGED = "triaged"
    CONTAINED = "contained"
    MITIGATED = "mitigated"
    POSTMORTEM_PUBLISHED = "postmortem_published"
    CLOSED = "closed"


# Strict one-way order — index in this tuple defines
# forward direction. advance_phase rejects backwards moves.
_PHASE_ORDER: Tuple[IncidentPhase, ...] = (
    IncidentPhase.DETECTED,
    IncidentPhase.TRIAGED,
    IncidentPhase.CONTAINED,
    IncidentPhase.MITIGATED,
    IncidentPhase.POSTMORTEM_PUBLISHED,
    IncidentPhase.CLOSED,
)

_TERMINAL_PHASES = {IncidentPhase.CLOSED}

# Size caps (DoS / spam-floor defense).
_MAX_SUMMARY_LEN = 4_000
_MAX_NOTE_LEN = 4_000
_MAX_AFFECTED_CONTRACTS = 50


# ── Pre-committed playbook ────────────────────────────


# DECISION_TREE — what operators should do at each
# (severity, phase). Strings are operator-facing
# imperatives. Public per §14 promise.
DECISION_TREE: Dict[
    Tuple[IncidentSeverity, IncidentPhase], List[str],
] = {
    # ── S0: catastrophic — active drain ──
    (IncidentSeverity.S0, IncidentPhase.DETECTED): [
        "PAUSE NOW: invoke prsm_emergency_pause "
        "compose_pause on every affected contract; "
        "Foundation Safe signs IMMEDIATELY (target: <15min).",
        "Notify multisig signers via out-of-band channel "
        "(Signal/phone). Do NOT wait for normal Slack chain.",
        "Open public S0 incident page within 30 minutes — "
        "silence amplifies trust damage.",
        "Engage outside counsel (securities + breach "
        "notification obligations vary by jurisdiction).",
    ],
    (IncidentSeverity.S0, IncidentPhase.TRIAGED): [
        "Confirm scope: which addresses lost funds, total "
        "value drained, attacker address(es).",
        "Engage on-chain forensics partner (Chainalysis / "
        "TRM Labs).",
        "Initiate insurance fund recovery composer "
        "(prsm_insurance_fund compose_recovery) if "
        "reimbursement scope known.",
    ],
    (IncidentSeverity.S0, IncidentPhase.CONTAINED): [
        "All affected contracts paused. Verify on-chain "
        "via prsm_emergency_pause status.",
        "Public update: PAUSED state + funds-safe / "
        "funds-at-risk breakdown.",
    ],
    (IncidentSeverity.S0, IncidentPhase.MITIGATED): [
        "Patch deployed via UUPS upgrade through 2-of-3 "
        "Foundation Safe; record upgrade tx hash in "
        "incident timeline.",
        "Resume protocol via prsm_emergency_pause "
        "compose_unpause — only after independent review "
        "of the patch.",
    ],
    (IncidentSeverity.S0,
     IncidentPhase.POSTMORTEM_PUBLISHED): [
        "Postmortem covers: root cause, timeline, funds "
        "impact, recovery plan, prevention.",
        "Compensation distribution via Safe if applicable.",
    ],

    # ── S1: critical, confirmed exploitable ──
    (IncidentSeverity.S1, IncidentPhase.DETECTED): [
        "PAUSE affected contracts within 1 hour. Foundation "
        "Safe signs.",
        "Internal triage within 2 hours — confirm "
        "exploitability + impact estimate.",
        "Notify security@ subscribers + relevant exchanges.",
    ],
    (IncidentSeverity.S1, IncidentPhase.TRIAGED): [
        "Decide: pause + patch (recommended) vs "
        "watch-and-warn (only if exploit complexity "
        "extremely high and impact bounded).",
        "Cross-check against responsible-disclosure intake "
        "(prsm_disclosure list) for related reports.",
    ],
    (IncidentSeverity.S1, IncidentPhase.CONTAINED): [
        "Status update on public incident page; explain "
        "scope + ETA.",
    ],
    (IncidentSeverity.S1, IncidentPhase.MITIGATED): [
        "Independent review of patch before unpause.",
    ],
    (IncidentSeverity.S1,
     IncidentPhase.POSTMORTEM_PUBLISHED): [
        "Public postmortem within 14 days.",
    ],

    # ── S2: high, plausible but not confirmed ──
    (IncidentSeverity.S2, IncidentPhase.DETECTED): [
        "Internal triage within 8 hours; do NOT pause yet.",
        "Watch on-chain activity on affected contracts "
        "(prsm_emergency_pause status; treasury monitor).",
    ],
    (IncidentSeverity.S2, IncidentPhase.TRIAGED): [
        "If exploitability confirmed, escalate to S1 and "
        "open new incident; do NOT mutate this record's "
        "severity.",
    ],
    (IncidentSeverity.S2, IncidentPhase.CONTAINED): [
        "Document mitigation; no pause required if scope "
        "stayed S2.",
    ],
    (IncidentSeverity.S2, IncidentPhase.MITIGATED): [
        "Patch via standard upgrade cadence.",
    ],
    (IncidentSeverity.S2,
     IncidentPhase.POSTMORTEM_PUBLISHED): [
        "Postmortem within 30 days; brief format OK.",
    ],

    # ── S3: low / informational ──
    (IncidentSeverity.S3, IncidentPhase.DETECTED): [
        "Log + monitor. No public action required unless "
        "scope escalates.",
    ],
    (IncidentSeverity.S3, IncidentPhase.TRIAGED): [
        "Close-out if confirmed informational.",
    ],
    (IncidentSeverity.S3, IncidentPhase.CONTAINED): [
        "No further action.",
    ],
    (IncidentSeverity.S3, IncidentPhase.MITIGATED): [
        "Roll fix into next scheduled release.",
    ],
    (IncidentSeverity.S3,
     IncidentPhase.POSTMORTEM_PUBLISHED): [
        "Internal note; public postmortem optional.",
    ],
}


# COMMS_TEMPLATES — pre-committed markdown text per
# (severity, phase). {summary} interpolation supported.
COMMS_TEMPLATES: Dict[
    Tuple[IncidentSeverity, IncidentPhase], str,
] = {
    (IncidentSeverity.S0, IncidentPhase.DETECTED): (
        "# ⚠ PRSM Active Incident — S0 Catastrophic\n\n"
        "We have detected an active exploit affecting "
        "PRSM smart contracts.\n\n"
        "**Summary:** {summary}\n\n"
        "**Current status:** Foundation Safe is executing "
        "the emergency pause sequence right now. Affected "
        "contracts will be paused within minutes.\n\n"
        "**What users should do:** Do not interact with "
        "PRSM contracts until further notice. We will "
        "post the next update within 30 minutes.\n\n"
        "Updates will appear here and on our security "
        "channel. We are committed to full transparency.\n"
    ),
    (IncidentSeverity.S0, IncidentPhase.TRIAGED): (
        "# PRSM Incident Update — Triage Complete\n\n"
        "**Summary:** {summary}\n\n"
        "Triage is complete. Scope, attacker addresses, "
        "and affected user balances are now known. The "
        "Foundation insurance fund is being prepared for "
        "any required user reimbursement.\n"
    ),
    (IncidentSeverity.S0, IncidentPhase.CONTAINED): (
        "# PRSM Incident Update — Contained\n\n"
        "**Summary:** {summary}\n\n"
        "All affected contracts are paused. No further "
        "funds at risk. Patch development underway.\n"
    ),
    (IncidentSeverity.S0, IncidentPhase.MITIGATED): (
        "# PRSM Incident Update — Patch Deployed\n\n"
        "**Summary:** {summary}\n\n"
        "Patch deployed and independently reviewed. "
        "Foundation Safe will unpause affected contracts "
        "after a 24-hour observation window. Postmortem "
        "to follow within 14 days.\n"
    ),
    (IncidentSeverity.S0,
     IncidentPhase.POSTMORTEM_PUBLISHED): (
        "# PRSM Postmortem — S0 Incident\n\n"
        "**Summary:** {summary}\n\n"
        "## Root cause\n\n_TBD by incident commander._\n\n"
        "## Timeline\n\n_See incident timeline._\n\n"
        "## Funds impact\n\n_TBD._\n\n"
        "## Recovery plan\n\n_TBD._\n\n"
        "## Prevention measures\n\n_TBD._\n"
    ),

    (IncidentSeverity.S1, IncidentPhase.DETECTED): (
        "# PRSM Security Notice — S1 Incident\n\n"
        "**Summary:** {summary}\n\n"
        "A critical security issue has been detected. "
        "Foundation Safe is preparing the pause sequence "
        "for affected contracts. Users should pause new "
        "interactions until further notice.\n"
    ),
    (IncidentSeverity.S1, IncidentPhase.TRIAGED): (
        "# PRSM Incident Update — Triaged\n\n"
        "**Summary:** {summary}\n\n"
        "Triage complete; mitigation path selected. "
        "Updates to follow.\n"
    ),
    (IncidentSeverity.S1, IncidentPhase.CONTAINED): (
        "# PRSM Incident Update — Contained\n\n"
        "**Summary:** {summary}\n\n"
        "Affected contracts paused or otherwise contained. "
        "Patch in progress.\n"
    ),
    (IncidentSeverity.S1, IncidentPhase.MITIGATED): (
        "# PRSM Incident Update — Patched\n\n"
        "**Summary:** {summary}\n\n"
        "Patch deployed and reviewed. Resuming normal "
        "operation after observation window.\n"
    ),
    (IncidentSeverity.S1,
     IncidentPhase.POSTMORTEM_PUBLISHED): (
        "# PRSM Postmortem — S1 Incident\n\n"
        "**Summary:** {summary}\n\n"
        "Public postmortem within 14 days of mitigation.\n"
    ),

    (IncidentSeverity.S2, IncidentPhase.DETECTED): (
        "# PRSM Internal Note — S2 Suspected Issue\n\n"
        "**Summary:** {summary}\n\n"
        "Issue is plausible but not confirmed "
        "exploitable. Internal triage underway. No "
        "public action yet.\n"
    ),
    (IncidentSeverity.S2, IncidentPhase.TRIAGED): (
        "# PRSM Internal Update — Triaged\n\n"
        "**Summary:** {summary}\n\n"
        "Triage complete. If confirmed, will escalate to "
        "S1.\n"
    ),
    (IncidentSeverity.S2, IncidentPhase.CONTAINED): (
        "# PRSM Internal Note — Contained\n\n"
        "**Summary:** {summary}\n"
    ),
    (IncidentSeverity.S2, IncidentPhase.MITIGATED): (
        "# PRSM Internal Note — Patched\n\n"
        "**Summary:** {summary}\n"
    ),
    (IncidentSeverity.S2,
     IncidentPhase.POSTMORTEM_PUBLISHED): (
        "# PRSM Postmortem — S2\n\n"
        "**Summary:** {summary}\n"
    ),

    (IncidentSeverity.S3, IncidentPhase.DETECTED): (
        "# PRSM Internal Log — S3 Informational\n\n"
        "**Summary:** {summary}\n"
    ),
    (IncidentSeverity.S3, IncidentPhase.TRIAGED): (
        "# PRSM Internal Log — S3 Triaged\n\n"
        "**Summary:** {summary}\n"
    ),
    (IncidentSeverity.S3, IncidentPhase.CONTAINED): (
        "# PRSM Internal Log — S3 Contained\n\n"
        "**Summary:** {summary}\n"
    ),
    (IncidentSeverity.S3, IncidentPhase.MITIGATED): (
        "# PRSM Internal Log — S3 Mitigated\n\n"
        "**Summary:** {summary}\n"
    ),
    (IncidentSeverity.S3,
     IncidentPhase.POSTMORTEM_PUBLISHED): (
        "# PRSM Internal Log — S3 Closed\n\n"
        "**Summary:** {summary}\n"
    ),
}


def get_recommendations(
    severity: IncidentSeverity, phase: IncidentPhase,
) -> List[str]:
    return list(DECISION_TREE.get((severity, phase), []))


def get_comms_template(
    severity: IncidentSeverity,
    phase: IncidentPhase,
    *,
    summary: str = "",
) -> str:
    text = COMMS_TEMPLATES.get((severity, phase), "")
    if text and "{summary}" in text:
        text = text.replace("{summary}", summary)
    return text


# ── Records ───────────────────────────────────────────


@dataclass
class TimelineEvent:
    ts: float
    phase: IncidentPhase
    note: str
    actor: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "phase": self.phase.value,
            "note": self.note,
            "actor": self.actor,
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "TimelineEvent":
        return cls(
            ts=float(d.get("ts", 0.0)),
            phase=IncidentPhase(d["phase"]),
            note=d.get("note", ""),
            actor=d.get("actor", ""),
        )


@dataclass
class IncidentRecord:
    incident_id: str
    opened_ts: float
    severity: IncidentSeverity
    summary: str
    affected_contracts: List[str]
    current_phase: IncidentPhase
    timeline: List[TimelineEvent] = field(default_factory=list)
    related_disclosure_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "opened_ts": self.opened_ts,
            "severity": self.severity.value,
            "summary": self.summary,
            "affected_contracts": list(
                self.affected_contracts,
            ),
            "current_phase": self.current_phase.value,
            "timeline": [e.to_dict() for e in self.timeline],
            "related_disclosure_id": (
                self.related_disclosure_id
            ),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "IncidentRecord":
        return cls(
            incident_id=d["incident_id"],
            opened_ts=float(d.get("opened_ts", 0.0)),
            severity=IncidentSeverity(d["severity"]),
            summary=d.get("summary", ""),
            affected_contracts=list(
                d.get("affected_contracts") or [],
            ),
            current_phase=IncidentPhase(d["current_phase"]),
            timeline=[
                TimelineEvent.from_dict(e)
                for e in (d.get("timeline") or [])
            ],
            related_disclosure_id=d.get(
                "related_disclosure_id",
            ),
        )


# ── IncidentResponse ──────────────────────────────────


class IncidentResponse:
    def __init__(
        self,
        *,
        persist_dir: Optional[Path] = None,
    ) -> None:
        self._records: Dict[str, IncidentRecord] = {}
        self._persist_dir: Optional[Path] = (
            Path(persist_dir)
            if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(
                parents=True, exist_ok=True,
            )
            self._load_from_disk()

    @classmethod
    def from_env(cls) -> "IncidentResponse":
        raw = os.environ.get("PRSM_INCIDENT_RESPONSE_DIR")
        persist_dir = Path(raw) if raw else None
        return cls(persist_dir=persist_dir)

    # ── Mutations ─────────────────────────────────────

    def open(
        self,
        *,
        severity: IncidentSeverity,
        summary: str,
        affected_contracts: List[str],
        opened_ts: Optional[float] = None,
        related_disclosure_id: Optional[str] = None,
        actor: str = "",
    ) -> IncidentRecord:
        if not isinstance(severity, IncidentSeverity):
            raise ValueError(
                f"severity must be an IncidentSeverity, "
                f"got {severity!r}"
            )
        if not summary or not isinstance(summary, str):
            raise ValueError("summary must be non-empty")
        if len(summary) > _MAX_SUMMARY_LEN:
            raise ValueError(
                f"summary exceeds {_MAX_SUMMARY_LEN}-char cap"
            )
        if len(affected_contracts) > _MAX_AFFECTED_CONTRACTS:
            raise ValueError(
                f"affected_contracts exceeds "
                f"{_MAX_AFFECTED_CONTRACTS}-entry cap"
            )
        ts = (
            opened_ts if opened_ts is not None
            else time.time()
        )
        record = IncidentRecord(
            incident_id=str(uuid.uuid4()),
            opened_ts=ts,
            severity=severity,
            summary=summary,
            affected_contracts=list(affected_contracts),
            current_phase=IncidentPhase.DETECTED,
            timeline=[
                TimelineEvent(
                    ts=ts,
                    phase=IncidentPhase.DETECTED,
                    note=f"incident opened: {summary}",
                    actor=actor,
                ),
            ],
            related_disclosure_id=related_disclosure_id,
        )
        self._records[record.incident_id] = record
        self._write_to_disk(record)
        return record

    def advance_phase(
        self,
        incident_id: str,
        new_phase: IncidentPhase,
        *,
        ts: Optional[float] = None,
        note: str = "",
        actor: str = "",
    ) -> IncidentRecord:
        existing = self._records.get(incident_id)
        if existing is None:
            raise ValueError(
                f"incident {incident_id!r} not found"
            )
        if existing.current_phase in _TERMINAL_PHASES:
            raise ValueError(
                f"incident {incident_id!r} is closed "
                f"(terminal phase); cannot advance further"
            )
        if not isinstance(new_phase, IncidentPhase):
            raise ValueError(
                f"new_phase must be an IncidentPhase, got "
                f"{new_phase!r}"
            )
        cur_idx = _PHASE_ORDER.index(existing.current_phase)
        new_idx = _PHASE_ORDER.index(new_phase)
        if new_idx <= cur_idx:
            raise ValueError(
                f"phase machine is one-way; cannot move "
                f"backwards from {existing.current_phase.value!r} "
                f"to {new_phase.value!r}"
            )
        when = ts if ts is not None else time.time()
        event = TimelineEvent(
            ts=when,
            phase=new_phase,
            note=note or f"phase → {new_phase.value}",
            actor=actor,
        )
        updated = IncidentRecord(
            incident_id=existing.incident_id,
            opened_ts=existing.opened_ts,
            severity=existing.severity,
            summary=existing.summary,
            affected_contracts=existing.affected_contracts,
            current_phase=new_phase,
            timeline=existing.timeline + [event],
            related_disclosure_id=existing.related_disclosure_id,
        )
        self._records[incident_id] = updated
        self._write_to_disk(updated)
        return updated

    def record_event(
        self,
        incident_id: str,
        *,
        note: str,
        ts: Optional[float] = None,
        actor: str = "",
    ) -> IncidentRecord:
        existing = self._records.get(incident_id)
        if existing is None:
            raise ValueError(
                f"incident {incident_id!r} not found"
            )
        if not note:
            raise ValueError("note must be non-empty")
        if len(note) > _MAX_NOTE_LEN:
            raise ValueError(
                f"note exceeds {_MAX_NOTE_LEN}-char cap"
            )
        when = ts if ts is not None else time.time()
        event = TimelineEvent(
            ts=when,
            phase=existing.current_phase,
            note=note,
            actor=actor,
        )
        updated = IncidentRecord(
            incident_id=existing.incident_id,
            opened_ts=existing.opened_ts,
            severity=existing.severity,
            summary=existing.summary,
            affected_contracts=existing.affected_contracts,
            current_phase=existing.current_phase,
            timeline=existing.timeline + [event],
            related_disclosure_id=existing.related_disclosure_id,
        )
        self._records[incident_id] = updated
        self._write_to_disk(updated)
        return updated

    # ── Queries ───────────────────────────────────────

    def get(
        self, incident_id: str,
    ) -> Optional[IncidentRecord]:
        return self._records.get(incident_id)

    def list(
        self,
        *,
        severity: Optional[IncidentSeverity] = None,
        phase: Optional[IncidentPhase] = None,
    ) -> List[IncidentRecord]:
        out = list(self._records.values())
        out.sort(key=lambda r: r.opened_ts, reverse=True)
        if severity is not None:
            out = [
                r for r in out if r.severity == severity
            ]
        if phase is not None:
            out = [
                r for r in out if r.current_phase == phase
            ]
        return out

    def count(self) -> int:
        return len(self._records)

    # ── Persistence ───────────────────────────────────

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        for path in self._persist_dir.glob("*.json"):
            try:
                d = json.loads(path.read_text())
                record = IncidentRecord.from_dict(d)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "IncidentResponse: skipping corrupt "
                    "%s: %s", path, exc,
                )
                continue
            self._records[record.incident_id] = record

    def _write_to_disk(
        self, record: IncidentRecord,
    ) -> None:
        if self._persist_dir is None:
            return
        safe = (
            record.incident_id
            .replace("/", "_")
            .replace("\\", "_")
            .replace("..", "_")
        )
        path = self._persist_dir / f"{safe}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(record.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "IncidentResponse: disk write failed for "
                "%s: %s",
                record.incident_id, exc,
            )
