"""Sprint 318c — operator runtime health check.

Sprint 318 shipped `bringup status` — STATIC env-var
inventory (are required vars set + well-formed?). This
sprint adds `bringup health` — DYNAMIC runtime check:
do the configured persistence dirs exist + are they
writable? Do the configured keypairs actually load as
valid Ed25519/X25519? Can we round-trip a small
operation against each subsystem the env vars enable?

Status answers "is the deployment WIRED?" — health
answers "does the deployment WORK?". Both must pass
before pointing real customers at the node.

This sprint also ships docs/operator-runbook.md as the
operator-facing reference. The runbook isn't itself
tested for content, but a smoke check confirms the file
exists + mentions every subcommand operators rely on.
"""
from __future__ import annotations

import base64
import os
import stat
from pathlib import Path

import pytest

from prsm.enterprise.bringup_health import (
    CheckResult,
    HealthCheckOutcome,
    check_keypair_format,
    check_persistence_dir_writable,
    check_subsystem_round_trip,
    run_health_checks,
)


# ── CheckResult dataclass ──────────────────────────


def test_check_result_to_dict_round_trip():
    r = CheckResult(
        name="example", ok=True,
        diagnostic="all good",
    )
    d = r.to_dict()
    assert d["name"] == "example"
    assert d["ok"] is True
    assert d["diagnostic"] == "all good"


# ── check_persistence_dir_writable ─────────────────


def test_check_persistence_dir_writable_happy_path(
    tmp_path,
):
    r = check_persistence_dir_writable(
        var_name="TEST_DIR", path=str(tmp_path),
    )
    assert r.ok
    assert "writable" in r.diagnostic.lower()


def test_check_persistence_dir_creates_missing_subdir(
    tmp_path,
):
    """If the parent exists + is writable but the leaf
    dir doesn't, the check creates it. Operators don't
    want a deploy to fail because a fresh mount didn't
    pre-create the subdirs."""
    target = tmp_path / "new-subdir"
    assert not target.exists()
    r = check_persistence_dir_writable(
        var_name="TEST_DIR", path=str(target),
    )
    assert r.ok
    assert target.exists()


def test_check_persistence_dir_fails_on_unwritable(
    tmp_path,
):
    """A read-only path fails the check loud — operator
    needs to fix mount perms before deploying."""
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    # Set 0o500: read+execute owner, no write
    readonly.chmod(0o500)
    try:
        r = check_persistence_dir_writable(
            var_name="TEST_DIR", path=str(readonly),
        )
        assert not r.ok
        assert (
            "writ" in r.diagnostic.lower()
            or "permission" in r.diagnostic.lower()
        )
    finally:
        readonly.chmod(0o700)  # let pytest clean up


def test_check_persistence_dir_handles_none():
    """Unset env var → check passes (path is optional).
    Distinguishes 'wasn't configured' from 'is broken'."""
    r = check_persistence_dir_writable(
        var_name="TEST_DIR", path=None,
    )
    assert r.ok
    assert "not configured" in r.diagnostic.lower()


# ── check_keypair_format ───────────────────────────


def test_check_keypair_valid_ed25519():
    valid = base64.b64encode(b"\x00" * 32).decode()
    r = check_keypair_format(
        var_name="TEST_PRIVKEY", value=valid,
        key_type="ed25519",
    )
    assert r.ok


def test_check_keypair_valid_x25519():
    valid = base64.b64encode(b"\x00" * 32).decode()
    r = check_keypair_format(
        var_name="TEST_PRIVKEY", value=valid,
        key_type="x25519",
    )
    assert r.ok


def test_check_keypair_rejects_wrong_length():
    too_short = base64.b64encode(b"\x00" * 16).decode()
    r = check_keypair_format(
        var_name="TEST_PRIVKEY", value=too_short,
        key_type="ed25519",
    )
    assert not r.ok
    assert "32" in r.diagnostic


def test_check_keypair_rejects_invalid_base64():
    r = check_keypair_format(
        var_name="TEST_PRIVKEY", value="not-base64!",
        key_type="ed25519",
    )
    assert not r.ok


def test_check_keypair_handles_none():
    r = check_keypair_format(
        var_name="TEST_PRIVKEY", value=None,
        key_type="ed25519",
    )
    assert r.ok  # unset = not-configured; not failed


def test_check_keypair_rejects_unknown_type():
    with pytest.raises(ValueError, match="key_type"):
        check_keypair_format(
            var_name="TEST", value="x",
            key_type="rsa",
        )


# ── check_subsystem_round_trip ─────────────────────


def test_subsystem_round_trip_fl_orchestrator(tmp_path):
    """Wire up an FL orchestrator at the configured path
    + do a small operation (propose+get job). If anything
    in the path/import chain is broken, this surfaces."""
    r = check_subsystem_round_trip(
        subsystem="fl_orchestrator",
        persist_dir=str(tmp_path),
    )
    assert r.ok
    assert "fl" in r.name.lower() or "feder" in (
        r.name.lower()
    )


def test_subsystem_round_trip_pipeline_orchestrator(
    tmp_path,
):
    r = check_subsystem_round_trip(
        subsystem="pipeline_orchestrator",
        persist_dir=str(tmp_path),
    )
    assert r.ok


def test_subsystem_round_trip_handles_unconfigured():
    """If the persist_dir is None (env var unset), the
    check returns ok=True with a 'not configured'
    diagnostic — operator hasn't opted into this
    subsystem yet."""
    r = check_subsystem_round_trip(
        subsystem="fl_orchestrator", persist_dir=None,
    )
    assert r.ok
    assert "not configured" in r.diagnostic.lower()


def test_subsystem_round_trip_unknown_subsystem():
    with pytest.raises(ValueError, match="subsystem"):
        check_subsystem_round_trip(
            subsystem="unknown_thing",
            persist_dir="/tmp/x",
        )


# ── run_health_checks (full sweep) ─────────────────


def test_run_health_checks_returns_outcome(tmp_path):
    """run_health_checks aggregates every individual
    check against the current env. Returns a
    HealthCheckOutcome with all results."""
    outcome = run_health_checks()
    assert isinstance(outcome, HealthCheckOutcome)
    assert len(outcome.checks) > 0


def test_run_health_checks_overall_ok_when_all_pass(
    tmp_path, monkeypatch,
):
    """With all required env vars set to valid values +
    fresh persistence dirs, the overall outcome is ok."""
    valid_key = base64.b64encode(b"\x00" * 32).decode()
    monkeypatch.setenv(
        "PRSM_FEDERATED_WORKER_PRIVKEY", valid_key,
    )
    monkeypatch.setenv(
        "PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY", valid_key,
    )
    monkeypatch.setenv(
        "PRSM_FEDERATED_LEARNING_DIR",
        str(tmp_path / "fl"),
    )
    monkeypatch.setenv(
        "PRSM_PIPELINE_ORCHESTRATOR_DIR",
        str(tmp_path / "pipeline"),
    )
    outcome = run_health_checks()
    assert outcome.ok, (
        "all-required-set should pass; failed:\n"
        + "\n".join(
            f"  {c.name}: {c.diagnostic}"
            for c in outcome.checks if not c.ok
        )
    )


def test_run_health_checks_overall_fail_on_bad_keypair(
    tmp_path, monkeypatch,
):
    """A malformed keypair env var should make the
    overall outcome fail with the specific check named."""
    valid_key = base64.b64encode(b"\x00" * 32).decode()
    monkeypatch.setenv(
        "PRSM_FEDERATED_WORKER_PRIVKEY", "not-base64!",
    )
    monkeypatch.setenv(
        "PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY", valid_key,
    )
    monkeypatch.setenv(
        "PRSM_FEDERATED_LEARNING_DIR",
        str(tmp_path / "fl"),
    )
    monkeypatch.setenv(
        "PRSM_PIPELINE_ORCHESTRATOR_DIR",
        str(tmp_path / "pipeline"),
    )
    outcome = run_health_checks()
    assert not outcome.ok
    # The failing check names the bad var
    failed = [c for c in outcome.checks if not c.ok]
    assert any(
        "PRSM_FEDERATED_WORKER_PRIVKEY" in c.name
        for c in failed
    )


def test_run_health_checks_summary_formatting(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv(
        "PRSM_FEDERATED_LEARNING_DIR",
        str(tmp_path / "fl"),
    )
    outcome = run_health_checks()
    summary = outcome.summary()
    # Summary lists every check with status
    assert isinstance(summary, str)
    for check in outcome.checks:
        assert check.name in summary


# ── CLI ────────────────────────────────────────────


def test_cli_health_rc_zero_when_all_pass(
    tmp_path, monkeypatch, capsys,
):
    valid_key = base64.b64encode(b"\x00" * 32).decode()
    monkeypatch.setenv(
        "PRSM_FEDERATED_WORKER_PRIVKEY", valid_key,
    )
    monkeypatch.setenv(
        "PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY", valid_key,
    )
    monkeypatch.setenv(
        "PRSM_FEDERATED_LEARNING_DIR",
        str(tmp_path / "fl"),
    )
    monkeypatch.setenv(
        "PRSM_PIPELINE_ORCHESTRATOR_DIR",
        str(tmp_path / "pipeline"),
    )
    from prsm.enterprise.bringup_cli import main
    rc = main(["health"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ok" in out.lower() or "✓" in out


def test_cli_health_rc_nonzero_on_failure(
    monkeypatch, capsys,
):
    """Bad config → rc=1. CI / deploy scripts rely on
    this to fail-fast before sending real traffic."""
    monkeypatch.setenv(
        "PRSM_FEDERATED_WORKER_PRIVKEY", "not-base64!",
    )
    from prsm.enterprise.bringup_cli import main
    rc = main(["health"])
    out = capsys.readouterr().out
    assert rc == 1
    assert (
        "fail" in out.lower()
        or "✗" in out
        or "missing" in out.lower()
    )


# ── Operator runbook (smoke check) ─────────────────


_RUNBOOK = (
    Path(__file__).resolve().parents[2]
    / "docs" / "operator-runbook.md"
)


def test_operator_runbook_exists():
    assert _RUNBOOK.exists(), (
        f"operator runbook missing at {_RUNBOOK}"
    )


def test_operator_runbook_documents_subcommands():
    content = _RUNBOOK.read_text()
    for subcmd in ("generate", "status", "demo", "health"):
        assert subcmd in content, (
            f"runbook should document `{subcmd}` "
            f"subcommand"
        )


def test_operator_runbook_documents_docker_workflow():
    content = _RUNBOOK.read_text()
    assert "docker" in content.lower()
    assert (
        "docker compose" in content.lower()
        or "docker-compose" in content.lower()
        or "docker build" in content.lower()
    )


def test_operator_runbook_explains_persistence_volume():
    """Operators MUST know where state lives + how to
    persist it across container restarts."""
    content = _RUNBOOK.read_text()
    assert "/var/lib/prsm" in content
    assert "volume" in content.lower()
