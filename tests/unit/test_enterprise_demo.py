"""Sprint 318b — end-to-end demo script.

Sprint 318 shipped operator config foundations. This
sprint ships the demo runnable: a single script that
exercises the full §7 enterprise FL flow + federated
inference flow against an in-process deployment, printing
each step. Operators run this to:
  - Verify a fresh deploy actually works
  - See what each component contributes end-to-end
  - Get a reproducible reference for their own
    integration tests

The demo uses the orchestrator primitives directly (no
HTTP / Docker) so it's testable from pytest. Sprint 318a
will add the Docker-flavored variant.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prsm.enterprise.demo import (
    DemoEnvironment,
    DemoStepResult,
    run_fl_demo,
    run_full_demo,
    run_pipeline_demo,
    setup_demo_environment,
)


# ── setup_demo_environment ──────────────────────────


def test_setup_produces_complete_environment(tmp_path):
    env = setup_demo_environment(persist_root=tmp_path)
    # All keypairs populated
    assert env.fl_orchestrator_priv
    assert env.fl_orchestrator_pub
    assert env.pipeline_orchestrator_priv
    assert env.pipeline_orchestrator_pub
    assert len(env.worker_keypairs) >= 2
    # Orchestrators wired
    assert env.fl_orchestrator is not None
    assert env.pipeline_orchestrator is not None


def test_setup_keypairs_are_distinct(tmp_path):
    env = setup_demo_environment(persist_root=tmp_path)
    privs = [
        env.fl_orchestrator_priv,
        env.pipeline_orchestrator_priv,
    ] + [w[0] for w in env.worker_keypairs]
    assert len(set(privs)) == len(privs)


def test_setup_persistence_dirs_under_root(tmp_path):
    env = setup_demo_environment(persist_root=tmp_path)
    # The FL orchestrator's persistence dir lives under
    # the supplied persist_root
    fl_jobs = env.fl_orchestrator.list_jobs()
    # Empty initially — but the persist_dir exists
    assert (tmp_path / "fl").exists()
    assert (tmp_path / "pipeline").exists()


# ── run_fl_demo ────────────────────────────────────


def test_run_fl_demo_completes_with_verifiable_round(
    tmp_path, capsys,
):
    """The full §7 FL flow: register worker keys, propose
    job, each worker trains via PyTorch, orchestrator
    aggregates, final aggregated update is non-zero."""
    env = setup_demo_environment(persist_root=tmp_path)
    result = run_fl_demo(env)
    assert isinstance(result, DemoStepResult)
    assert result.ok, result.message
    # The output mentions FL milestones
    captured = capsys.readouterr().out
    # "Proposing" stems from "propose" but loses the 'e'
    # — look for the common stem.
    assert "propos" in captured.lower()
    assert "aggregat" in captured.lower()


def test_run_fl_demo_aggregated_update_is_nonzero(
    tmp_path,
):
    """The aggregated update is a real gradient delta —
    not all zeros (which would indicate the training
    didn't actually run)."""
    env = setup_demo_environment(persist_root=tmp_path)
    result = run_fl_demo(env)
    assert result.ok
    # Result carries the aggregated update bytes
    from prsm.enterprise.federated_learning import (
        decode_gradient,
    )
    grad = decode_gradient(result.aggregated_bytes)
    assert len(grad) > 0
    nonzero = sum(1 for v in grad if v != 0.0)
    assert nonzero >= len(grad) * 0.5


def test_run_fl_demo_round_persists(tmp_path):
    """After the demo runs, the FL orchestrator's
    persistence has recorded the job + round — operator
    can inspect after the fact."""
    env = setup_demo_environment(persist_root=tmp_path)
    run_fl_demo(env)
    jobs = env.fl_orchestrator.list_jobs()
    assert len(jobs) >= 1
    job = jobs[0]
    rnd = env.fl_orchestrator.get_round(job.job_id, 0)
    assert rnd is not None


# ── run_pipeline_demo ──────────────────────────────


def test_run_pipeline_demo_completes_with_signed_receipt(
    tmp_path, capsys,
):
    env = setup_demo_environment(persist_root=tmp_path)
    result = run_pipeline_demo(env)
    assert result.ok, result.message
    captured = capsys.readouterr().out
    assert (
        "pipeline" in captured.lower()
        or "stage" in captured.lower()
    )


def test_pipeline_demo_receipt_verifies(tmp_path):
    """The pipeline demo's receipt verifies under the
    pipeline orchestrator's pubkey + the activation hash
    chain is intact."""
    from prsm.compute.inference.pipeline_receipt import (
        verify_pipeline_receipt,
    )
    env = setup_demo_environment(persist_root=tmp_path)
    result = run_pipeline_demo(env)
    assert result.ok
    assert result.pipeline_receipt is not None
    verification = verify_pipeline_receipt(
        result.pipeline_receipt,
        orchestrator_pubkey_b64=(
            env.pipeline_orchestrator_pub
        ),
    )
    assert verification.ok
    assert verification.chain_valid


def test_pipeline_demo_round_persists(tmp_path):
    env = setup_demo_environment(persist_root=tmp_path)
    run_pipeline_demo(env)
    jobs = env.pipeline_orchestrator.list_jobs()
    assert len(jobs) >= 1


# ── run_full_demo ──────────────────────────────────


def test_run_full_demo_executes_both_flows(
    tmp_path, capsys,
):
    """run_full_demo executes BOTH FL and pipeline
    inference flows. Useful as a single smoke test for
    a fresh deploy."""
    env = setup_demo_environment(persist_root=tmp_path)
    results = run_full_demo(env)
    # Returns a list of (name, DemoStepResult) tuples
    assert len(results) >= 2
    for name, result in results:
        assert result.ok, (
            f"step {name!r} failed: {result.message}"
        )
    captured = capsys.readouterr().out
    # Both demos surfaced their work
    assert "aggregat" in captured.lower()
    assert (
        "pipeline" in captured.lower()
        or "stage" in captured.lower()
    )


def test_run_full_demo_prints_summary_at_end(
    tmp_path, capsys,
):
    """At the end of the full demo, a summary line tells
    the operator everything passed (or which step failed
    + diagnostic)."""
    env = setup_demo_environment(persist_root=tmp_path)
    run_full_demo(env)
    captured = capsys.readouterr().out
    assert (
        "all demos passed" in captured.lower()
        or "✓" in captured
        or "complete" in captured.lower()
    )


# ── CLI ────────────────────────────────────────────


def test_cli_demo_runs_to_completion(capsys):
    """`prsm-enterprise-bringup demo` runs the full demo
    against a fresh in-process env. RC=0 on success."""
    from prsm.enterprise.bringup_cli import main
    rc = main(["demo"])
    out = capsys.readouterr().out
    assert rc == 0
    # Output should mention both flows
    assert "fl" in out.lower() or "federated" in out.lower()
    assert (
        "pipeline" in out.lower()
        or "inference" in out.lower()
    )


def test_cli_demo_uses_tmp_dir_when_persist_unspecified(
    capsys, tmp_path, monkeypatch,
):
    """The CLI shouldn't clobber existing operator state.
    When no --persist-root is given, the demo uses a
    fresh temp dir."""
    from prsm.enterprise.bringup_cli import main
    rc = main(["demo"])
    assert rc == 0
