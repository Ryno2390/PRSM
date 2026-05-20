"""Sprint 599 (Phase 2D step 5) — response handler wired to dispatch.

Tests verify source-grep invariants since the wiring is inside
PRSMNode.start (lots of async setup; full integration test
impractical without a real running daemon).

Live attestation deferred to daemon restart with PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=rpc
+ section7-readiness preflight (sprint 585).
"""
from __future__ import annotations

from pathlib import Path

import inspect


def test_node_start_registers_chain_executor_handler():
    """PRSMNode.start must register a MSG_DIRECT handler that calls
    handle_chain_executor_response. Source-grep invariant.
    """
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "handle_chain_executor_response" in src, (
        "Sprint 599: response handler not wired into PRSMNode.start"
    )
    assert "MSG_DIRECT" in src, (
        "Sprint 599: handler must register on MSG_DIRECT type"
    )
    assert "self.transport.on_message" in src, (
        "Sprint 599: registration must use transport.on_message"
    )


def test_response_handler_wiring_is_in_try_except():
    """The wiring is wrapped in try/except so failures don't crash
    daemon startup (operators see WARNING log instead).
    """
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    # Crude check: the chain-executor wiring block must include
    # `except Exception` for safety.
    lines = src.splitlines()
    in_block = False
    block_lines = []
    for line in lines:
        if "Sprint 599" in line:
            in_block = True
        if in_block:
            block_lines.append(line)
            if "Sprint 599 chain-executor response wiring failed" in line:
                break
    block = "\n".join(block_lines)
    assert "except Exception" in block, (
        "Sprint 599: wiring must be in try/except to protect daemon "
        "startup"
    )
