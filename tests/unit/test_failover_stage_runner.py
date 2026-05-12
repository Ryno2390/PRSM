"""Sprint 315 — per-stage failover + retry.

Sprint 313 introduced cross-node HTTP transport; a stage
worker can now fail mid-pipeline (node crash, network
partition, transient 502). Sprint 312's orchestrator
marked the whole round FAILED on first stage exception
with no retry path. This sprint ships a wrapper around
the StageRunner Protocol so operators can compose:

  - Multiple candidate runners per stage (primary +
    backups) — the wrapper tries them in order until one
    succeeds
  - Optional N retries per runner on transient
    HTTPStageRunnerError before moving to the next
    candidate

The wrapper conforms to the existing StageRunner Protocol;
the orchestrator API doesn't change. v1 retry/failover is
PER-STAGE — round-level retry (rerun the whole pipeline
from scratch) is just calling execute() again on the same
job_id, which the sprint 312 orchestrator already
supports.
"""
from __future__ import annotations

import pytest

from prsm.compute.inference.failover_stage_runner import (
    AllRunnersFailedError,
    FailoverStageRunner,
)
from prsm.compute.inference.http_stage_runner import (
    HTTPStageRunnerError,
)
from prsm.compute.inference.pipeline_stage import (
    deterministic_stub_stage_runner,
)


def _intermittent_runner(failures_before_success: int):
    """Build a stage runner that raises
    HTTPStageRunnerError the first N times it's called,
    then delegates to the deterministic stub on attempt
    N+1+. Useful for retry semantics tests."""
    state = {"calls": 0}
    stub = deterministic_stub_stage_runner()

    def _runner(
        *,
        input_activations, stage_id, layer_indices,
    ):
        state["calls"] += 1
        if state["calls"] <= failures_before_success:
            raise HTTPStageRunnerError(
                f"simulated transient failure "
                f"(call {state['calls']})"
            )
        return stub(
            input_activations=input_activations,
            stage_id=stage_id,
            layer_indices=layer_indices,
        )

    return _runner, state


def _always_failing_runner():
    def _runner(*, input_activations, stage_id, layer_indices):
        raise HTTPStageRunnerError(
            "simulated permanent failure"
        )
    return _runner


# ── Construction ───────────────────────────────────


def test_failover_requires_at_least_one_runner():
    with pytest.raises(ValueError, match="runner"):
        FailoverStageRunner(runners=[])


def test_failover_rejects_negative_retries():
    with pytest.raises(ValueError, match="retries"):
        FailoverStageRunner(
            runners=[deterministic_stub_stage_runner()],
            max_retries_per_runner=-1,
        )


# ── Primary succeeds (no failover invoked) ─────────


def test_primary_succeeds_returns_first_runner_output():
    primary = deterministic_stub_stage_runner()
    failover = FailoverStageRunner(
        runners=[primary, _always_failing_runner()],
    )
    out = failover(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    # Output matches what the primary stub would produce
    expected = primary(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    assert out == expected


# ── Failover on transient failure ──────────────────


def test_failover_moves_to_backup_on_primary_failure():
    backup = deterministic_stub_stage_runner()
    failover = FailoverStageRunner(
        runners=[_always_failing_runner(), backup],
    )
    out = failover(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    expected = backup(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    assert out == expected


def test_failover_traverses_multiple_backups():
    final = deterministic_stub_stage_runner()
    failover = FailoverStageRunner(
        runners=[
            _always_failing_runner(),
            _always_failing_runner(),
            _always_failing_runner(),
            final,
        ],
    )
    out = failover(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    expected = final(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    assert out == expected


def test_failover_all_runners_fail_raises():
    failover = FailoverStageRunner(
        runners=[
            _always_failing_runner(),
            _always_failing_runner(),
            _always_failing_runner(),
        ],
    )
    with pytest.raises(AllRunnersFailedError):
        failover(
            input_activations=b"x",
            stage_id=0, layer_indices=[0],
        )


def test_failover_raises_capture_all_attempt_errors():
    """When everything fails, the raised
    AllRunnersFailedError should record each attempt's
    underlying error message so operators can diagnose."""
    failover = FailoverStageRunner(
        runners=[
            _always_failing_runner(),
            _always_failing_runner(),
        ],
    )
    try:
        failover(
            input_activations=b"x",
            stage_id=0, layer_indices=[0],
        )
    except AllRunnersFailedError as exc:
        assert len(exc.attempt_errors) == 2
        assert all(
            "permanent failure" in e
            for e in exc.attempt_errors
        )
    else:
        pytest.fail("expected AllRunnersFailedError")


# ── Retry semantics ─────────────────────────────────


def test_retries_a_runner_before_failing_over():
    """A flaky runner that fails twice then succeeds, with
    max_retries_per_runner=2 — the wrapper retries the
    primary and gets a successful result on the third
    attempt, without falling over to the backup."""
    flaky, state = _intermittent_runner(
        failures_before_success=2,
    )
    backup = _always_failing_runner()
    failover = FailoverStageRunner(
        runners=[flaky, backup],
        max_retries_per_runner=2,
    )
    out = failover(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    # Primary was called 3 times (2 fails + 1 success);
    # backup never invoked
    assert state["calls"] == 3
    expected = deterministic_stub_stage_runner()(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    assert out == expected


def test_retries_then_fallover_when_retries_exhausted():
    """A primary that fails 3 times, with retries=2, falls
    over to the backup on the 3rd consecutive failure."""
    flaky, state = _intermittent_runner(
        failures_before_success=10,  # never succeeds
    )
    backup = deterministic_stub_stage_runner()
    failover = FailoverStageRunner(
        runners=[flaky, backup],
        max_retries_per_runner=2,
    )
    out = failover(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    # Primary was tried 3 times (1 initial + 2 retries)
    assert state["calls"] == 3
    expected = backup(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    assert out == expected


# ── Non-retryable error doesn't trigger retry/fallover ──


def test_non_retryable_error_propagates_immediately():
    """A ValueError (operator error, e.g., wrong
    layer_indices) is not transient and should NOT be
    retried or failed over. It propagates immediately so
    the operator sees the real bug."""
    def bad_runner(
        *, input_activations, stage_id, layer_indices,
    ):
        raise ValueError("operator bug, not transient")

    failover = FailoverStageRunner(
        runners=[bad_runner, deterministic_stub_stage_runner()],
        max_retries_per_runner=3,
    )
    with pytest.raises(ValueError, match="operator bug"):
        failover(
            input_activations=b"x",
            stage_id=0, layer_indices=[0],
        )


# ── Integration with orchestrator ──────────────────


def test_integrates_with_orchestrator_survives_stage_failure():
    """End-to-end: orchestrator + FailoverStageRunner ->
    transient stage failures don't fail the round; round
    completes + receipt verifies."""
    from prsm.compute.inference.pipeline_orchestrator import (
        PipelineInferenceOrchestrator,
        PipelineRoundStatus,
    )
    from prsm.compute.inference.pipeline_partition import (
        even_layer_partition,
    )
    from prsm.compute.inference.pipeline_receipt import (
        verify_pipeline_receipt,
    )
    from prsm.enterprise.federated_learning import (
        generate_worker_keypair,
    )

    priv, pub = generate_worker_keypair()
    orch = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
    )
    partition = even_layer_partition(
        total_layers=4, node_ids=["n0", "n1"],
    )
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )

    # Stage 0: a flaky primary + a backup stub
    flaky, _ = _intermittent_runner(
        failures_before_success=2,
    )
    stage0_failover = FailoverStageRunner(
        runners=[flaky, deterministic_stub_stage_runner()],
        max_retries_per_runner=2,
    )
    # Stage 1: simple stub
    stage1_runner = deterministic_stub_stage_runner()

    rnd = orch.execute(
        job.job_id, prompt=b"end-to-end",
        stage_runners=[stage0_failover, stage1_runner],
    )
    assert rnd.status == PipelineRoundStatus.COMPLETED
    result = verify_pipeline_receipt(
        rnd.receipt, orchestrator_pubkey_b64=pub,
    )
    assert result.ok


def test_orchestrator_marks_round_failed_when_failover_exhausted():
    """If every runner in a FailoverStageRunner fails, the
    round is marked FAILED with the wrapper's
    AllRunnersFailedError surfaced as the cause."""
    from prsm.compute.inference.pipeline_orchestrator import (
        PipelineInferenceOrchestrator,
        PipelineRoundStatus,
    )
    from prsm.compute.inference.pipeline_partition import (
        even_layer_partition,
    )
    from prsm.enterprise.federated_learning import (
        generate_worker_keypair,
    )

    priv, _ = generate_worker_keypair()
    orch = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
    )
    partition = even_layer_partition(
        total_layers=2, node_ids=["n0", "n1"],
    )
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )
    fully_dead = FailoverStageRunner(
        runners=[
            _always_failing_runner(),
            _always_failing_runner(),
        ],
    )
    with pytest.raises(AllRunnersFailedError):
        orch.execute(
            job.job_id, prompt=b"x",
            stage_runners=[
                fully_dead,
                deterministic_stub_stage_runner(),
            ],
        )
    rnd = orch.get_round(job.job_id)
    assert rnd.status == PipelineRoundStatus.FAILED


# ── Round-level retry (job-level re-execute) ──────


def test_round_level_retry_by_re_executing():
    """Sprint 315 round-level retry IS just calling
    execute() again on the same job. After a FAILED
    round, execute() can be called again (the orchestrator
    overwrites the round). Successful re-execution
    produces a verifiable receipt."""
    from prsm.compute.inference.pipeline_orchestrator import (
        PipelineInferenceOrchestrator,
        PipelineRoundStatus,
    )
    from prsm.compute.inference.pipeline_partition import (
        even_layer_partition,
    )
    from prsm.compute.inference.pipeline_receipt import (
        verify_pipeline_receipt,
    )
    from prsm.enterprise.federated_learning import (
        generate_worker_keypair,
    )

    priv, pub = generate_worker_keypair()
    orch = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
    )
    partition = even_layer_partition(
        total_layers=2, node_ids=["n0", "n1"],
    )
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )

    # First attempt: fails
    with pytest.raises(HTTPStageRunnerError):
        orch.execute(
            job.job_id, prompt=b"x",
            stage_runners=[
                _always_failing_runner(),
                deterministic_stub_stage_runner(),
            ],
        )
    assert (
        orch.get_round(job.job_id).status
        == PipelineRoundStatus.FAILED
    )

    # Retry: pass good runners
    rnd = orch.execute(
        job.job_id, prompt=b"x",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )
    assert rnd.status == PipelineRoundStatus.COMPLETED
    result = verify_pipeline_receipt(
        rnd.receipt, orchestrator_pubkey_b64=pub,
    )
    assert result.ok
