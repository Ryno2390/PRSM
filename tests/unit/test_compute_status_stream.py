"""GET /compute/status/{job_id}/stream — SSE-based job status push.

Closes the last B8 deferred sub-item: event-driven status updates
so MCP clients (and any other streaming-capable consumer) can
render IN_PROGRESS → COMPLETED transitions live without
client-side polling overhead.

Polling-based v1: server polls JobHistoryStore + PaymentEscrow at
a short interval, emits an SSE event whenever the snapshot
changes, and closes the connection on terminal status (history
COMPLETED/FAILED/CANCELLED OR escrow RELEASED/REFUNDED).

Wire format (text/event-stream):
    event: status
    data: {"job_id": "...", "history": {...}, "escrow": {...}}

    event: terminal
    data: {"job_id": "...", "reason": "completed"}

`event: terminal` is the only terminal event — server closes the
connection after emitting it.
"""
from __future__ import annotations

import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.job_history import (
    JobHistoryRecord,
    JobHistoryStore,
    JobStatus,
)
from prsm.node.payment_escrow import EscrowEntry, EscrowStatus


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _entry(*, job_id="forge-abc", amount=5.0,
           status=EscrowStatus.PENDING):
    return EscrowEntry(
        escrow_id=f"esc-{job_id}",
        job_id=job_id,
        requester_id="test-requester",
        amount=amount,
        status=status,
    )


def _record(*, job_id="forge-abc", status=JobStatus.IN_PROGRESS):
    return JobHistoryRecord(
        job_id=job_id,
        query=f"q-{job_id}",
        status=status,
        started_at=time.time(),
    )


def _node(*, history=None, escrow=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._job_history = history
    node._payment_escrow = escrow
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _parse_sse(text: str):
    """Parse SSE text into a list of (event_type, data_dict) tuples."""
    events = []
    blocks = text.strip().split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        event_type = None
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if event_type and data_lines:
            try:
                payload = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                payload = {"_raw": "\n".join(data_lines)}
            events.append((event_type, payload))
    return events


# ──────────────────────────────────────────────────────────────────────
# Service availability
# ──────────────────────────────────────────────────────────────────────


class TestStreamServiceAvailability:
    def test_503_when_neither_subsystem_wired(self):
        node = _node()
        with _client(node).stream(
            "GET", "/compute/status/forge-abc/stream",
        ) as resp:
            assert resp.status_code == 503

    def test_404_when_neither_has_the_job(self):
        history = JobHistoryStore()
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=None)
        node = _node(history=history, escrow=escrow_mock)
        with _client(node).stream(
            "GET", "/compute/status/forge-unknown/stream",
        ) as resp:
            assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────────────
# Already-terminal: emit one event + close immediately
# ──────────────────────────────────────────────────────────────────────


class TestStreamAlreadyTerminal:
    def test_completed_history_emits_terminal_and_closes(self):
        """When the job is already COMPLETED, the stream emits one
        status event (the final state) + a terminal event, then
        closes. Total events: 2."""
        history = JobHistoryStore()
        history.put(_record(status=JobStatus.COMPLETED))
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=None)
        node = _node(history=history, escrow=escrow_mock)
        with patch.dict(
            os.environ, {"PRSM_STATUS_STREAM_POLL_SEC": "0.01"},
        ):
            response = _client(node).get(
                "/compute/status/forge-abc/stream",
            )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )
        events = _parse_sse(response.text)
        # At least one status + one terminal.
        status_events = [e for e in events if e[0] == "status"]
        terminal_events = [e for e in events if e[0] == "terminal"]
        assert len(status_events) >= 1
        assert len(terminal_events) == 1
        assert terminal_events[0][1]["job_id"] == "forge-abc"
        assert terminal_events[0][1]["reason"] in {
            "completed", "history_terminal",
        }

    def test_refunded_escrow_emits_terminal_and_closes(self):
        """An already-REFUNDED escrow (e.g., job was cancelled)
        is terminal even without a history record."""
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=_entry(
            status=EscrowStatus.REFUNDED,
        ))
        node = _node(escrow=escrow_mock)
        with patch.dict(
            os.environ, {"PRSM_STATUS_STREAM_POLL_SEC": "0.01"},
        ):
            response = _client(node).get(
                "/compute/status/forge-abc/stream",
            )
        assert response.status_code == 200
        events = _parse_sse(response.text)
        terminal = [e for e in events if e[0] == "terminal"]
        assert len(terminal) == 1


# ──────────────────────────────────────────────────────────────────────
# Status transitions: IN_PROGRESS → COMPLETED
# ──────────────────────────────────────────────────────────────────────


class TestStreamTransitions:
    def test_in_progress_to_completed_emits_two_status_events(self):
        """JobHistoryStore returns IN_PROGRESS first, then
        COMPLETED on subsequent get(). Stream must emit a status
        event for each distinct snapshot + a terminal."""
        # Custom history that flips status on second get().
        states = [
            _record(status=JobStatus.IN_PROGRESS),
            _record(status=JobStatus.COMPLETED),
        ]
        idx = {"i": 0}

        class _MockHistory:
            def get(self, job_id):
                rec = states[min(idx["i"], len(states) - 1)]
                idx["i"] += 1
                return rec

        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=None)
        node = _node(history=_MockHistory(), escrow=escrow_mock)
        with patch.dict(
            os.environ, {"PRSM_STATUS_STREAM_POLL_SEC": "0.01"},
        ):
            response = _client(node).get(
                "/compute/status/forge-abc/stream",
            )
        assert response.status_code == 200
        events = _parse_sse(response.text)
        status_events = [e for e in events if e[0] == "status"]
        terminal_events = [e for e in events if e[0] == "terminal"]
        # Both snapshots emitted (de-duplication leaves the
        # transition visible).
        assert len(status_events) >= 2
        # Status payload includes history.status field.
        assert any(
            e[1].get("history", {}).get("status") == "in_progress"
            for e in status_events
        )
        assert any(
            e[1].get("history", {}).get("status") == "completed"
            for e in status_events
        )
        assert len(terminal_events) == 1

    def test_repeat_in_progress_polls_dedupe_to_one_event(self):
        """If the snapshot doesn't change between polls, only one
        status event should be emitted (de-duplication). Otherwise
        clients drown in identical noise."""
        # Always returns same IN_PROGRESS record → never terminal,
        # would loop forever. Use a brief stream-timeout to test.
        history = JobHistoryStore()
        history.put(_record(status=JobStatus.IN_PROGRESS))
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=None)
        node = _node(history=history, escrow=escrow_mock)
        # Force a short stream-timeout so the test doesn't hang.
        with patch.dict(os.environ, {
            "PRSM_STATUS_STREAM_POLL_SEC": "0.01",
            "PRSM_STATUS_STREAM_TIMEOUT_SEC": "0.1",
        }):
            response = _client(node).get(
                "/compute/status/forge-abc/stream",
            )
        events = _parse_sse(response.text)
        status_events = [e for e in events if e[0] == "status"]
        # At most one status event since the snapshot never changed.
        assert len(status_events) == 1


# ──────────────────────────────────────────────────────────────────────
# Timeout
# ──────────────────────────────────────────────────────────────────────


class TestStreamTimeout:
    def test_timeout_emits_terminal_with_timeout_reason(self):
        """If the job stays IN_PROGRESS past the timeout, the
        stream emits a terminal event with reason: timeout and
        closes. Caller can re-subscribe if they want to keep
        watching."""
        history = JobHistoryStore()
        history.put(_record(status=JobStatus.IN_PROGRESS))
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=None)
        node = _node(history=history, escrow=escrow_mock)
        with patch.dict(os.environ, {
            "PRSM_STATUS_STREAM_POLL_SEC": "0.01",
            "PRSM_STATUS_STREAM_TIMEOUT_SEC": "0.05",
        }):
            response = _client(node).get(
                "/compute/status/forge-abc/stream",
            )
        events = _parse_sse(response.text)
        terminal = [e for e in events if e[0] == "terminal"]
        assert len(terminal) == 1
        assert terminal[0][1]["reason"] == "timeout"
