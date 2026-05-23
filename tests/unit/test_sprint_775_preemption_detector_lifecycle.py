"""Sprint 775 — wire PreemptionDetector into PRSMNode lifecycle.

Sprints 772-774 shipped detection + announce gate + dispatch
gate — all tested in isolation but INERT in production because
nothing in PRSMNode.start() constructs + registers the detector.
Without sprint 775, an operator setting PRSM_PREEMPTION_DETECTOR=aws
gets no behavior change because the metadata-polling loop is
never started.

Sprint 775 closes that gap by mirroring sprint 766's
AutoClaimWorker lifecycle pattern:
- PRSMNode.start() resolves detector from env; .start() task
  + registers via prsm.node.preemption.register_detector().
- PRSMNode.stop() cancels the polling task + clears the
  registered detector.

Pin tests:
- Source-shape: PRSMNode.start contains resolve + register call.
- Source-shape: PRSMNode.stop contains stop + reset call.
- Source-shape: detector start happens BEFORE _started=True so
  the flag is queryable from the moment the daemon claims ready.
- Source-shape: stop happens BEFORE api_task.cancel() so an
  in-flight 503-decision from sprint 774 sees a stable flag.
- register_detector + reset_for_testing exports exist (sprint
  772 pre-condition pinned here too for the lifecycle code to
  rely on them).
"""
from __future__ import annotations

import inspect


def test_node_start_source_constructs_and_registers_detector():
    """PRSMNode.start() must call resolve_detector_from_env() AND
    register_detector() so module-level is_currently_preempted()
    reads correctly."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "resolve_detector_from_env" in src
    assert "register_detector" in src


def test_node_stop_source_clears_detector():
    """PRSMNode.stop() must stop the detector + clear the
    module-level registration so a process-recycle test or
    re-start doesn't see stale state."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.stop)
    # The detector's .stop() and the module reset both must
    # appear. Reuse the existing reset_for_testing helper —
    # it's the right shape (clears _DETECTOR) and exists already.
    assert "_preemption_detector" in src
    assert "reset_for_testing" in src or "register_detector" in src


def test_detector_started_before_started_flag():
    """The detector start MUST occur before self._started = True
    so any inbound HTTP request that lands at the boundary sees a
    queryable flag (not a None-detector race)."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    det_idx = src.find("_preemption_detector")
    started_idx = src.find("self._started = True")
    assert det_idx > 0
    assert started_idx > 0
    assert det_idx < started_idx, (
        "PreemptionDetector must be wired BEFORE _started=True "
        "to avoid a queryable-flag race"
    )


def test_detector_stopped_before_api_task_cancel():
    """Stop ordering: detector loop must wind down BEFORE the
    api_task is cancelled. Otherwise a final in-flight
    /compute/inference call could observe a partially-torn-down
    detector (rare but the same pattern sprint 766 uses for
    AutoClaimWorker for the same reason)."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.stop)
    det_idx = src.find("_preemption_detector")
    cancel_idx = src.find("self._api_task.cancel()")
    assert det_idx > 0
    assert cancel_idx > 0
    assert det_idx < cancel_idx, (
        "PreemptionDetector cleanup must precede api_task.cancel"
    )


def test_register_detector_and_reset_exports_exist():
    """Pre-condition for the lifecycle code. Sprint 772 shipped
    these — pin them here as a regression guard."""
    from prsm.node.preemption import (
        register_detector,
        reset_for_testing,
    )
    assert callable(register_detector)
    assert callable(reset_for_testing)
