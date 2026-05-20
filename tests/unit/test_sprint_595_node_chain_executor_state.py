"""Sprint 595 (Phase 2D step 1) — node._loop + _chain_executor_pending.

Phase 2D wires the RPC chain executor into the daemon. Step 1 (this
sprint) initializes the state that subsequent steps rely on:

  node._loop                    — captured at daemon startup; used
                                  by sprint-594 run_async_on_loop
                                  primitive to bridge async transport
                                  calls from the chain-executor
                                  sync code path.

  node._chain_executor_pending  — dict[request_id, Future] awaiting
                                  response bytes. Sprint-597
                                  message-handler resolves these.

Reversible: pure attribute init; if nothing reads these (current
state), no behavior change. Sprint 596 wires the SendMessage
adapter to use them; sprint 597 wires the response routing.
"""
from __future__ import annotations


def test_node_start_captures_loop_reference():
    """node.start() must set self._loop to the running event loop."""
    import inspect
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    # Source-grep: the start() body must include the loop capture.
    # Phase 2D steps 2+ rely on this attribute.
    assert "self._loop" in src
    assert "get_running_loop" in src


def test_node_start_initializes_pending_dict():
    """node.start() must initialize self._chain_executor_pending."""
    import inspect
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "_chain_executor_pending" in src


def test_chain_executor_pending_init_is_empty_dict():
    """The pending dict must be initialized to an empty {} so
    request-id lookups during sprint-597 wiring start clean (no
    stale request futures persisting across daemon restarts).
    """
    import inspect
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    # The line `self._chain_executor_pending = {}` (or similar
    # mutable mapping) must appear.
    assert "self._chain_executor_pending = {}" in src, (
        "Sprint 595: pending dict must be initialized to empty {}"
    )
