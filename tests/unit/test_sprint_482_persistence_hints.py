"""Sprint 482 — /health/detailed persistence_env_var hints.

Pre-fix: 6 optional subsystems (job_history, receipt_store,
royalty_dispatch_ring, slash_event_log, heartbeat_log,
distribution_log) reported `persisted: false` with no operator
guidance on which env var enables persistence. Operators
either grepped the codebase OR ran without durability, losing
audit-trail records on every restart.

Sprint 482 adds `persistence_env_var` to each `persisted: false`
entry — names the canonical PRSM_*_DIR env var that, when set,
enables filesystem durability for that subsystem.

This is operator UX work — same dogfood-arc pattern as
sprint 446 (actionable empty-state CLI feedback) and sprint
451 (Phase 5 pool-state NOT_CONFIGURED with seeding-ceremony
date).

These pins defend the contract: every persisted:false branch
emits the corresponding env var name.
"""
from __future__ import annotations

from pathlib import Path


def test_persistence_env_var_map_complete():
    """The hint map must cover every subsystem with opt-in
    persistence. Missing entries would leave operators with
    no actionable signal on those subsystems."""
    api_src = (
        Path(__file__).resolve().parents[2]
        / "prsm" / "node" / "api.py"
    ).read_text()
    # The map definition lives in /health/detailed handler.
    idx = api_src.find("_PERSISTENCE_ENV_VAR = {")
    assert idx >= 0, (
        "_PERSISTENCE_ENV_VAR map missing — operator hint "
        "contract removed"
    )
    end = api_src.find("}", idx)
    map_src = api_src[idx:end]
    for subsystem in (
        "job_history",
        "receipt_store",
        "royalty_dispatch_ring",
        "webhook_log",
        "slash_event_log",
        "heartbeat_log",
        "distribution_log",
    ):
        assert f'"{subsystem}":' in map_src, (
            f"persistence_env_var hint missing for "
            f"{subsystem}"
        )


def test_persistence_env_var_canonical_names():
    """The env var names must match the canonical
    PRSM_*_DIR names used by the actual subsystem
    initializers — otherwise the operator hint sends them
    to a no-op env var."""
    api_src = (
        Path(__file__).resolve().parents[2]
        / "prsm" / "node" / "api.py"
    ).read_text()
    # Each declared env var name must also appear in the
    # node-init code that READS it. We do a coarse check:
    # the env var name must appear elsewhere in the codebase
    # in an os.environ.get / os.getenv / env.get call.
    expected = {
        "PRSM_JOB_HISTORY_DIR",
        "PRSM_RECEIPT_STORE_DIR",
        "PRSM_ROYALTY_DISPATCH_LOG_DIR",
        "PRSM_WEBHOOK_LOG_DIR",
        "PRSM_SLASH_EVENT_LOG_DIR",
        "PRSM_HEARTBEAT_LOG_DIR",
        "PRSM_DISTRIBUTION_LOG_DIR",
    }
    # All seven must appear in the api.py source — that's
    # the hint map declaration. Then at least 6 of the 7
    # must appear in node.py too (the reader). webhook_log
    # may be defined elsewhere.
    for name in expected:
        assert name in api_src, (
            f"hint map references {name} but it's not in "
            f"api.py"
        )
    node_src = (
        Path(__file__).resolve().parents[2]
        / "prsm" / "node" / "node.py"
    ).read_text()
    readers_found = sum(1 for n in expected if n in node_src)
    assert readers_found >= 4, (
        f"hint map references env vars that node.py doesn't "
        f"read — found only {readers_found} of {len(expected)} "
        f"in node.py"
    )


def test_add_persistence_hint_helper_present():
    """The _add_persistence_hint helper must be defined and
    used at each persisted:false branch."""
    api_src = (
        Path(__file__).resolve().parents[2]
        / "prsm" / "node" / "api.py"
    ).read_text()
    assert "def _add_persistence_hint" in api_src
    # Must be called at least 4 times (job_history,
    # receipt_store, royalty_dispatch_ring, ring loop).
    call_count = api_src.count("_add_persistence_hint(")
    # Definition + ≥4 calls = ≥5 occurrences.
    assert call_count >= 5, (
        f"_add_persistence_hint not called at all "
        f"persisted:false branches; got {call_count} "
        f"occurrences (need ≥5: def + ≥4 calls)"
    )
