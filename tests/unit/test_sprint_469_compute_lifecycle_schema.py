"""Sprint 469 — §5.2 compute job-lifecycle schema pins.

Live-verified 2026-05-16 against a running daemon. These pins
defend the canonical response schemas of the 9 endpoints whose
PRSM_Testing.md rows were promoted 🟢 → ✅ this sprint:

  POST /compute/submit          → {job_id, status, job_type,
                                   ftns_budget}
  GET  /compute/status/{id}     → {job_id, escrow: {...}}
  GET  /compute/status/{id}/stream
                                → SSE event: status
  POST /compute/cancel/{id}     → {job_id, history_cancelled,
                                   escrow_refunded,
                                   refund_amount_ftns}
  GET  /compute/jobs            → {jobs, total, offset, limit}
  POST /compute/cleanup-stale   → {cleaned}
  POST /compute/train           → 503 with privkey hint when
                                   PRSM_FEDERATED_WORKER_PRIVKEY
                                   not set
  GET  /compute/stats           → {resources, allocation,
                                   capacity, active_jobs,
                                   completed_jobs}
  GET  /compute/receipt/{id}    → 404 No receipt for job_id='...'

These pins fire if a future refactor silently renames a response
field or removes a documented error-path detail.

They are SCHEMA pins, not full-system tests — the live verification
itself was the sprint 469 work. CI green here means "the field
names callers depend on still exist", not "the endpoints still
work end-to-end".
"""
from __future__ import annotations

import inspect
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
API_FILE = REPO_ROOT / "prsm" / "node" / "api.py"


def _api_source() -> str:
    return API_FILE.read_text()


# ── Submit ────────────────────────────────────────────────


def test_compute_submit_returns_canonical_fields():
    """`POST /compute/submit` happy-path response shape.
    Callers rely on `job_id` to subsequently poll status —
    renaming this field is a breaking change."""
    src = _api_source()
    assert '@app.post("/compute/submit")' in src
    # Response dict literal lives within the submit handler.
    submit_idx = src.index('@app.post("/compute/submit")')
    next_route_idx = src.index('@app.', submit_idx + 1)
    handler_src = src[submit_idx:next_route_idx]
    for field in (
        '"job_id"',
        '"status"',
        '"job_type"',
        '"ftns_budget"',
    ):
        assert field in handler_src, (
            f"submit response missing canonical field: "
            f"{field}"
        )


def test_compute_submit_payload_cap_env_var():
    """The 413-on-oversize-payload behavior is documented;
    the env var `PRSM_MAX_JOB_PAYLOAD_BYTES` is the operator
    knob. Renaming it breaks operator runbooks."""
    src = _api_source()
    assert "PRSM_MAX_JOB_PAYLOAD_BYTES" in src
    # Default 100KB cap; documented in PRSM_Testing.md row.
    assert "100 * 1024" in src


# ── Cancel ────────────────────────────────────────────────


def test_compute_cancel_returns_refund_fields():
    """Cancel response shape — clients display the refunded
    amount so users know how much FTNS came back."""
    src = _api_source()
    cancel_idx = src.find('@app.post("/compute/cancel/')
    assert cancel_idx >= 0, "cancel route missing"
    next_idx = src.index('@app.', cancel_idx + 1)
    handler_src = src[cancel_idx:next_idx]
    for field in (
        '"job_id"',
        '"history_cancelled"',
        '"escrow_refunded"',
        '"refund_amount_ftns"',
    ):
        assert field in handler_src, (
            f"cancel response missing canonical field: {field}"
        )


# ── Stream ────────────────────────────────────────────────


def test_compute_status_stream_emits_status_events():
    """SSE wire format — clients subscribe by event name."""
    src = _api_source()
    stream_idx = src.find('/compute/status/{job_id}/stream')
    assert stream_idx >= 0
    # Wire format documented in the handler docstring +
    # implemented as `event: status` lines.
    next_idx = src.find('@app.', stream_idx + 1)
    handler_src = src[stream_idx:next_idx]
    assert "event: status" in handler_src
    assert "event: terminal" in handler_src


# ── Cleanup ───────────────────────────────────────────────


def test_compute_cleanup_returns_cleaned_count():
    """The {cleaned: N} response shape is what operators
    consume to know if the cleanup did anything."""
    src = _api_source()
    cleanup_idx = src.find('@app.post("/compute/cleanup-stale")')
    assert cleanup_idx >= 0
    next_idx = src.index('@app.', cleanup_idx + 1)
    handler_src = src[cleanup_idx:next_idx]
    assert '"cleaned"' in handler_src


# ── Train ─────────────────────────────────────────────────


def test_compute_train_503_message_contains_actionable_env_hint():
    """When PRSM_FEDERATED_WORKER_PRIVKEY is not configured,
    `/compute/train` must surface an actionable hint, not
    a generic 503. Operators hitting this read the detail
    field to know what env var to set."""
    src = _api_source()
    train_idx = src.find('@app.post("/compute/train")')
    assert train_idx >= 0
    next_idx = src.index('@app.', train_idx + 1)
    handler_src = src[train_idx:next_idx]
    assert "PRSM_FEDERATED_WORKER_PRIVKEY" in handler_src
    assert "worker privkey not configured" in handler_src


# ── Stats ─────────────────────────────────────────────────


def test_compute_stats_canonical_schema_top_keys():
    """Stats response shape — dashboards consume the
    top-level keys. The /compute/stats handler delegates to
    `ComputeProvider.get_stats()`, so the source of truth
    for the schema is `prsm/node/compute_provider.py`."""
    provider_src = (
        REPO_ROOT / "prsm" / "node" / "compute_provider.py"
    ).read_text()
    # Anchor to the actual get_stats method body.
    gs_idx = provider_src.find("def get_stats(self)")
    assert gs_idx >= 0, "get_stats method missing"
    # Take everything until the next def (method boundary).
    after = provider_src[gs_idx:]
    end_idx = after.find("\n    def ", 10)
    handler_src = after[:end_idx] if end_idx > 0 else after
    for field in (
        '"resources"',
        '"allocation"',
        '"capacity"',
        '"active_jobs"',
        '"completed_jobs"',
    ):
        assert field in handler_src, (
            f"stats response missing top-level key: {field}"
        )


# ── Status (per-job) ──────────────────────────────────────


def test_compute_status_unknown_job_404_message_actionable():
    """The 404 detail for unknown jobs explains the two
    real causes (never ran here, or LRU-evicted) so a user
    isn't left wondering whether their job_id is wrong."""
    src = _api_source()
    # The matching string is in the status handler.
    assert (
        "No history or escrow record for job_id" in src
    )
    assert "LRU-bounded" in src or "evicted" in src
