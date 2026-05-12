"""Sprint 315 — per-stage failover + retry.

Wraps the sprint 312 StageRunner Protocol with multi-
runner failover and per-runner retry semantics. Operators
compose a list of (primary, backup1, backup2, ...) runners
for each stage; the wrapper tries them in order, retrying
each on transient errors before falling over to the next.

The wrapper conforms to the StageRunner Protocol — the
sprint 312 orchestrator API is unchanged. Operators
swap their per-stage runners for FailoverStageRunner
instances and get retry/failover transparently.

Retryable error class: HTTPStageRunnerError (sprint 313)
— transient network / server-side failures. Programmer
errors (ValueError, TypeError, etc.) propagate
immediately and do NOT trigger retry or failover, so the
operator sees real bugs without retry-masking noise.
"""
from __future__ import annotations

from typing import List

from prsm.compute.inference.http_stage_runner import (
    HTTPStageRunnerError,
)
from prsm.compute.inference.pipeline_stage import (
    StageRunner,
)


# Retryable error types. Operators wanting to extend (e.g.,
# treating some custom RPC error as transient) build their
# own wrapper on top.
_RETRYABLE_EXC_TYPES = (HTTPStageRunnerError,)


class AllRunnersFailedError(Exception):
    """Raised when every candidate runner in a
    FailoverStageRunner exhausts retries. The
    attempt_errors attribute lists the underlying
    exception messages from each attempt — operators
    diagnose by inspecting it."""

    def __init__(self, attempt_errors: List[str]) -> None:
        self.attempt_errors = list(attempt_errors)
        super().__init__(
            f"all {len(attempt_errors)} failover attempts "
            f"exhausted: {attempt_errors}"
        )


class FailoverStageRunner:
    """A StageRunner-compatible wrapper that delegates to a
    list of candidate runners with optional retry semantics
    per runner.

    runners: ordered list of candidates. Tries them in
        order; first to succeed wins.
    max_retries_per_runner: how many EXTRA attempts to make
        per runner on transient errors before falling over
        to the next. 0 (default) = no retry; one attempt
        per runner.
    """

    def __init__(
        self,
        *,
        runners: List[StageRunner],
        max_retries_per_runner: int = 0,
    ) -> None:
        if not runners:
            raise ValueError(
                "FailoverStageRunner requires at least "
                "one runner"
            )
        if max_retries_per_runner < 0:
            raise ValueError(
                f"max_retries_per_runner must be >= 0; "
                f"got {max_retries_per_runner}"
            )
        self._runners = list(runners)
        self._max_retries = int(max_retries_per_runner)

    def __call__(
        self,
        *,
        input_activations: bytes,
        stage_id: int,
        layer_indices: List[int],
    ) -> bytes:
        attempt_errors: List[str] = []
        for runner_idx, runner in enumerate(self._runners):
            # Up to (1 + max_retries) attempts per runner
            for attempt in range(self._max_retries + 1):
                try:
                    return runner(
                        input_activations=input_activations,
                        stage_id=stage_id,
                        layer_indices=layer_indices,
                    )
                except _RETRYABLE_EXC_TYPES as exc:
                    attempt_errors.append(
                        f"runner[{runner_idx}] "
                        f"attempt {attempt + 1}: {exc}"
                    )
                    # If we have retries left for this
                    # runner, retry it; otherwise fall
                    # over to next runner
                    continue
                # Any non-retryable exception propagates
                # immediately (and is NOT recorded in
                # attempt_errors — it's a programmer error,
                # not a transient failure)
        raise AllRunnersFailedError(attempt_errors)
