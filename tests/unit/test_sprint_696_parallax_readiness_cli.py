"""Sprint 696 — operator-UX consolidation: prsm node parallax-readiness.

Across sprints 558-695 the Parallax wiring accumulated 22 PRSM_PARALLAX_*
env vars. Operators have no consolidated way to verify their systemd
unit's env block before starting a daemon. This sprint ships a single
read-only CLI command that reports each env's status, validates values
against known-good enumerations, and surfaces missing required vars.

Exit 0 only when all required vars are set with valid values (CI-gating
friendly).
"""
from __future__ import annotations

import os

import pytest
from click.testing import CliRunner


def test_readiness_command_registered():
    """Pin: `prsm node parallax-readiness` is a registered CLI cmd."""
    from prsm.cli import node as node_group
    cmd_names = [c for c in node_group.commands.keys()]
    assert "parallax-readiness" in cmd_names, (
        f"parallax-readiness must be registered on the node command "
        f"group; available: {cmd_names}"
    )


def test_readiness_exits_nonzero_with_no_env(monkeypatch):
    """Empty env → required vars are missing → exit code 1."""
    # Clear all PRSM_* env vars for a clean baseline
    for key in list(os.environ.keys()):
        if key.startswith("PRSM_"):
            monkeypatch.delenv(key, raising=False)
    from prsm.cli import parallax_readiness_cli
    runner = CliRunner()
    result = runner.invoke(parallax_readiness_cli, ["--format", "json"])
    assert result.exit_code == 1
    import json
    payload = json.loads(result.output)
    assert payload["overall"] == "not_ready"
    assert "PRSM_INFERENCE_EXECUTOR" in payload["missing_required"]
    assert "PRSM_PARALLAX_GPU_POOL_KIND" in payload["missing_required"]


def test_readiness_exits_zero_when_all_required_set(monkeypatch):
    """All required vars set with valid values → exit 0."""
    for key in list(os.environ.keys()):
        if key.startswith("PRSM_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PRSM_INFERENCE_EXECUTOR", "parallax")
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_KIND", "dht-backed")
    monkeypatch.setenv("PRSM_PARALLAX_TRUST_STACK_KIND", "production")
    monkeypatch.setenv(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE",
        "/opt/prsm/config/parallax/model_catalog.json",
    )
    from prsm.cli import parallax_readiness_cli
    runner = CliRunner()
    result = runner.invoke(parallax_readiness_cli, ["--format", "json"])
    assert result.exit_code == 0
    import json
    payload = json.loads(result.output)
    assert payload["overall"] == "ready"
    assert payload["missing_required"] == []
    assert payload["bad_values"] == []


def test_readiness_flags_invalid_value(monkeypatch):
    """Invalid value (not in enum) → bad_values entry + exit 1."""
    for key in list(os.environ.keys()):
        if key.startswith("PRSM_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PRSM_INFERENCE_EXECUTOR", "garbage")  # bad
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_KIND", "dht-backed")
    monkeypatch.setenv("PRSM_PARALLAX_TRUST_STACK_KIND", "production")
    monkeypatch.setenv("PRSM_PARALLAX_MODEL_CATALOG_FILE", "/x")
    from prsm.cli import parallax_readiness_cli
    runner = CliRunner()
    result = runner.invoke(parallax_readiness_cli, ["--format", "json"])
    assert result.exit_code == 1
    import json
    payload = json.loads(result.output)
    bad = [b["env"] for b in payload["bad_values"]]
    assert "PRSM_INFERENCE_EXECUTOR" in bad


def test_readiness_lists_all_22_env_vars():
    """Pin against the env-var sprawl actually covered. If the
    list shrinks, an operator-impactful var stopped being checked."""
    from prsm.cli import _PARALLAX_ENV_REGISTRY
    assert len(_PARALLAX_ENV_REGISTRY) >= 22, (
        f"sprint 696 registered {len(_PARALLAX_ENV_REGISTRY)} env vars; "
        f"expected >=22 (sprints 558-695 accumulated 22+)"
    )
    # Spot-check key sprint-introduced vars
    names = {row[0] for row in _PARALLAX_ENV_REGISTRY}
    for required_var in (
        "PRSM_INFERENCE_EXECUTOR",
        "PRSM_PARALLAX_GPU_POOL_KIND",
        "PRSM_PARALLAX_TRUST_STACK_KIND",
        "PRSM_PARALLAX_MODEL_CATALOG_FILE",
        "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS",
        "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND",
        "PRSM_PARALLAX_STAGE_EXECUTOR_KIND",
        "PRSM_PARALLAX_STREAMING_RUNNER_KIND",
        "PRSM_PARALLAX_MEMORY_GB_OVERRIDE",
        "PRSM_PARALLAX_DEFAULT_RTT_MS",
        "PRSM_PARALLAX_TFLOPS_FP16_OVERRIDE",
        "PRSM_OPERATOR_ADDRESS",
        "PRSM_PARALLAX_STAKE_ELIGIBILITY",
    ):
        assert required_var in names, (
            f"sprint 696 registry must list {required_var}"
        )
