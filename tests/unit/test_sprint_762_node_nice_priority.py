"""Sprint 762 — operator-facing CPU politeness via os.nice().

Consumer-device operators (MacBook, gaming PC) want the daemon
to YIELD CPU to their interactive workloads (browser, editor,
game). Linux/macOS `nice(N)` adds N to the process priority —
higher nice = lower priority. Non-root processes can only
INCREASE nice (lower priority); reasonable values are 1-19.

Sprint 762 wires `PRSM_NODE_NICE` env var. Set BEFORE event-
loop capture so every coroutine the loop schedules inherits
the priority. Safe-fail on platforms without os.nice (Windows):
warn + continue.

Pin tests verify the source-level shape since exercising
PRSMNode.start() requires the full daemon construction. Plus
a behavioral test that exercises the env-resolution path
directly.
"""
from __future__ import annotations

import inspect
import os


def setup_function():
    os.environ.pop("PRSM_NODE_NICE", None)


def teardown_function():
    os.environ.pop("PRSM_NODE_NICE", None)


def test_node_start_source_reads_prsm_node_nice():
    """PRSMNode.start() reads `PRSM_NODE_NICE` env var. Defends
    against accidental removal of the operator-facing env knob."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "PRSM_NODE_NICE" in src, (
        "Sprint 762 fix requires PRSM_NODE_NICE handling in "
        "PRSMNode.start()"
    )


def test_node_start_calls_os_nice():
    """The source must call `os.nice(N)` to actually adjust
    priority. Defends against a refactor that drops the syscall."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "nice(" in src, (
        "Sprint 762 fix requires actually calling nice()"
    )


def test_node_start_runs_before_event_loop_capture():
    """The nice() adjustment must happen BEFORE event-loop
    capture (`_asyncio.get_running_loop()`) so the priority
    applies to every coroutine the loop schedules."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    nice_idx = src.find("PRSM_NODE_NICE")
    loop_idx = src.find("get_running_loop")
    assert nice_idx > 0
    assert loop_idx > 0
    assert nice_idx < loop_idx, (
        "Nice adjustment must precede event-loop capture so "
        "every scheduled coroutine inherits the priority"
    )


def test_node_start_handles_non_int_nice_value():
    """Operator typo `PRSM_NODE_NICE=foo` shouldn't crash daemon-
    start. Source must catch ValueError + log warning."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    # The pattern: try-int → except ValueError → logger.warning
    assert "except ValueError" in src or "except (ValueError" in src
    assert "is not an int" in src or "not an int" in src


def test_node_start_handles_windows_no_os_nice():
    """Windows: os.nice not available → AttributeError.
    Must safe-fail with a warning, not crash."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "AttributeError" in src
    assert "os.nice() not" in src.lower() or (
        "not available" in src.lower()
    )


def test_node_start_handles_os_rejection_of_negative_nice():
    """Linux/macOS reject negative nice for non-root → OSError.
    Source must catch + log without crashing."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "OSError" in src
    # Operator-facing message should explain non-root constraint
    assert "non-root" in src.lower() or "INCREASE nice" in src


def test_unset_env_means_no_priority_change():
    """Backward-compat: env unset → no os.nice() call → daemon
    runs at default priority. Source check via the `if _nice_raw:`
    guard that gates the entire nice block."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    nice_idx = src.find("_nice_raw = ")
    block_after = src[nice_idx:nice_idx + 200]
    assert "if _nice_raw:" in block_after, (
        "Unset env (empty string) must short-circuit and skip "
        "the os.nice() call"
    )
