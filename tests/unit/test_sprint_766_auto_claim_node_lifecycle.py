"""Sprint 766 — wire AutoClaimWorker into PRSMNode.start/stop.

Sprint 765 shipped the standalone AutoClaimWorker. Sprint 766
wires it into the daemon lifecycle so operators setting the
env vars actually get a running background worker.

Constructed only when staking_manager + identity are present
(defensive — staking is optional in some test configs).
worker.start() short-circuits when disabled (threshold=0), so
it's safe to always invoke without behavior change for the
existing operator fleet.

Pin tests verify source-shape: the construction + start + stop
calls are in the right place in node.py.
"""
from __future__ import annotations

import inspect


def test_node_start_constructs_auto_claim_worker():
    """PRSMNode.start() imports AutoClaimWorker + constructs it."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "from prsm.node.auto_claim import AutoClaimWorker" in src
    assert "AutoClaimWorker(" in src


def test_node_start_passes_staking_manager_and_user_id():
    """The construction passes the staking_manager + user_id
    (operator's own node_id). Defends against future refactor
    that drops the user_id binding."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    construction_idx = src.find("AutoClaimWorker(")
    construction_block = src[construction_idx:construction_idx + 300]
    assert "staking_manager=" in construction_block
    assert "user_id=" in construction_block
    assert "self.identity.node_id" in construction_block


def test_node_start_calls_worker_start():
    """After construction the worker is .start()'d so the loop
    schedules. .start() short-circuits when disabled — safe."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "self._auto_claim_worker.start()" in src or (
        "_auto_claim_worker.start()" in src
    )


def test_node_start_only_constructs_when_staking_manager_present():
    """Defensive: skip construction if staking_manager is None
    (some test configs don't init it). Don't crash daemon-start."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    # Look for the guard around the construction
    construction_idx = src.find("AutoClaimWorker(")
    pre_construction = src[max(0, construction_idx - 500):construction_idx]
    assert "self.staking_manager is not None" in pre_construction
    assert "self.identity is not None" in pre_construction


def test_node_start_wraps_construction_in_try_except():
    """Construction wrapped in try/except — auto-claim is opt-in
    optional functionality, shouldn't crash daemon-start."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    # The relevant try should be on the AutoClaimWorker import/
    # construct block.
    construction_idx = src.find("AutoClaimWorker(")
    after_construction = src[construction_idx:construction_idx + 800]
    assert "except Exception" in after_construction


def test_node_stop_stops_auto_claim_worker():
    """PRSMNode.stop() cleanly cancels the worker BEFORE tearing
    down staking_manager it depends on."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.stop)
    assert "_auto_claim_worker" in src
    assert ".stop()" in src
    # The stop call must happen near the top of stop() — pin
    # via "before" check: stop_call_idx < api_task_cancel_idx
    auto_stop_idx = src.find("_auto_claim_worker")
    api_cancel_idx = src.find("self._api_task")
    assert auto_stop_idx > 0
    assert api_cancel_idx > 0
    assert auto_stop_idx < api_cancel_idx, (
        "Auto-claim worker must be stopped BEFORE the API task "
        "to avoid in-flight claim attempts hitting torn-down "
        "subsystems"
    )


def test_node_stop_handles_worker_none_or_never_started():
    """If the worker was never constructed (staking_manager
    missing) OR never started (disabled), .stop() must not
    crash. Defensive `getattr` + try/except."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.stop)
    # Should reference getattr OR explicit None check before .stop()
    assert (
        "getattr(self, " in src
        and "_auto_claim_worker" in src
    ) or "if self._auto_claim_worker is not None" in src
    # try/except around .stop() so a buggy worker doesn't break
    # the rest of node teardown
    auto_idx = src.find("_auto_claim_worker")
    next_400_chars = src[auto_idx:auto_idx + 400]
    assert "except" in next_400_chars
