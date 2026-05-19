"""Sprint 579 — `prsm node trust-stack` CLI observability.

After sprints 558-562 (production trust stack) + 576/577/578
(Phase 1 plumbing for profile_source / consensus_submitter /
chain_executor), the trust stack has FOUR independently
env-controllable components:

  PRSM_PARALLAX_TRUST_STACK_KIND          (mock | production)
  PRSM_PARALLAX_PROFILE_SOURCE_KIND       (in_memory | dht)
  PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND  (logging | onchain)
  PRSM_PARALLAX_CHAIN_EXECUTOR_KIND       (stub | rpc)

For each component, the effective state is one of:
  REAL          — Phase 2 wiring landed + env opted in
  LOGGING       — falls back to logging-only / stub safe variant
  PLACEHOLDER   — fully no-op (e.g., InMemoryProfileSource w/ {})
  UNKNOWN       — operator typo, defaulted to placeholder

Sprint 579 = `prsm node trust-stack` CLI surfaces this for
operator inspection without spelunking startup logs.

Output schema invariants:
- Lists ALL four env vars with their CURRENT value (or "<unset>")
- Reports the effective KIND that the daemon would use on (re)start
- Honors --format json for agent consumption
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner


def _run(args, env=None):
    """Invoke `prsm node` CLI with optional env overlay."""
    from prsm.cli import node
    runner = CliRunner()
    if env:
        with patch.dict(os.environ, env, clear=False):
            return runner.invoke(node, args)
    return runner.invoke(node, args)


def test_trust_stack_shows_default_kinds_when_unset():
    """All four envs unset → reports defaults (mock, in_memory,
    logging, stub).
    """
    # Strip relevant envs to ensure deterministic test
    keys = [
        "PRSM_PARALLAX_TRUST_STACK_KIND",
        "PRSM_PARALLAX_PROFILE_SOURCE_KIND",
        "PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND",
        "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        result = _run(["trust-stack"])
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    assert result.exit_code == 0, result.output
    # All four env names appear in output
    for k in keys:
        assert k in result.output, f"{k} missing from output"


def test_trust_stack_shows_env_values_when_set():
    result = _run(
        ["trust-stack"],
        env={
            "PRSM_PARALLAX_TRUST_STACK_KIND": "production",
            "PRSM_PARALLAX_PROFILE_SOURCE_KIND": "dht",
            "PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND": "onchain",
            "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND": "rpc",
        },
    )
    assert result.exit_code == 0, result.output
    # Each operator-set value visible in output
    for val in ["production", "dht", "onchain", "rpc"]:
        assert val in result.output, f"set value {val!r} missing"


def test_trust_stack_json_format():
    """--format json returns parseable JSON with the 4 components."""
    import json
    keys = [
        "PRSM_PARALLAX_TRUST_STACK_KIND",
        "PRSM_PARALLAX_PROFILE_SOURCE_KIND",
        "PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND",
        "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        result = _run(["trust-stack", "--format", "json"])
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    # Schema: 4 components keyed by env var
    for k in keys:
        assert k in data, f"missing component {k}"
        entry = data[k]
        assert "kind" in entry
        assert "status" in entry
        assert "env_value" in entry


def test_trust_stack_unknown_kind_reports_unknown_status():
    """Operator typo (`bogus`) reflected as status=unknown_falls_back."""
    result = _run(
        ["trust-stack", "--format", "json"],
        env={"PRSM_PARALLAX_PROFILE_SOURCE_KIND": "bogus_xyz"},
    )
    assert result.exit_code == 0, result.output
    import json
    data = json.loads(result.output)
    assert data["PRSM_PARALLAX_PROFILE_SOURCE_KIND"]["env_value"] == "bogus_xyz"
    # Status surfaces the typo somehow
    status = data["PRSM_PARALLAX_PROFILE_SOURCE_KIND"]["status"].lower()
    assert "unknown" in status or "fallback" in status or "invalid" in status
