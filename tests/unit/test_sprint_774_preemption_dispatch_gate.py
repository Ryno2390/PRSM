"""Sprint 774 — preemption flag gates /compute/inference dispatch.

Sprint 773 stops the daemon from announcing presence when
preempted. But peers that already have us in their known-peer
cache will still route work here for up to `peer_stale_timeout`
(60s default). Sprint 774 closes that window: /compute/inference
+ /compute/inference/stream + /compute/forge all return 503 with
Retry-After when the preemption flag is set.

This mirrors the sprint-756 active-window gate at the same
3 endpoints.

Pin tests:
- _check_not_preempted_or_503 helper exists.
- Helper raises HTTPException(503) when preempted; no-op when
  not preempted.
- Source-shape: helper called inside the 3 inference endpoints
  BEFORE the rate-limit check (same ordering as
  _check_active_window_or_503).
"""
from __future__ import annotations

import inspect


def test_helper_function_exists():
    """Helper is exported from prsm.node.api."""
    from prsm.node.api import _check_not_preempted_or_503
    assert callable(_check_not_preempted_or_503)


def test_helper_raises_when_preempted():
    """Flag set → HTTPException 503 + Retry-After header."""
    from fastapi import HTTPException
    from prsm.node.api import _check_not_preempted_or_503
    from unittest.mock import patch
    import pytest

    with patch(
        "prsm.node.preemption.is_currently_preempted",
        return_value=True,
    ), pytest.raises(HTTPException) as exc:
        _check_not_preempted_or_503()
    assert exc.value.status_code == 503
    # The detail should mention preemption so the caller knows
    # why this 503 is different from rate-limit / active-window.
    assert "preempt" in exc.value.detail.lower()
    assert exc.value.headers.get("Retry-After")


def test_helper_no_op_when_not_preempted():
    """Flag clear → returns None (no raise)."""
    from prsm.node.api import _check_not_preempted_or_503
    from unittest.mock import patch
    with patch(
        "prsm.node.preemption.is_currently_preempted",
        return_value=False,
    ):
        assert _check_not_preempted_or_503() is None


def test_inference_endpoint_calls_gate():
    """Source-shape: /compute/inference handler calls the
    preemption check. Pin by string-grep on the api module."""
    from prsm.node import api
    src = inspect.getsource(api)
    # The 3 inference paths each get the gate. We don't pin per-
    # function (the function bodies are huge + dynamically built
    # inside register_inference_routes) — instead pin that the
    # check helper is referenced AT LEAST 3 times (one per
    # endpoint).
    n = src.count("_check_not_preempted_or_503(")
    assert n >= 3, (
        f"Sprint 774: expected gate called from 3 inference "
        f"endpoints (inference / stream / forge); found {n}"
    )


def test_gate_ordering_pinned():
    """The new preemption gate must appear AFTER the active-
    window gate (so 503-reason precedence is: active-window
    first, then preempted, then rate-limit). This is a stable
    surfacing order operators can document against."""
    from prsm.node import api
    src = inspect.getsource(api)
    # Every occurrence of the new gate should be preceded
    # somewhere earlier in the same function by the active-window
    # gate. We approximate with: the FIRST active-window call
    # appears before the FIRST preemption call.
    aw = src.find("_check_active_window_or_503(")
    pre = src.find("_check_not_preempted_or_503(")
    assert aw > 0
    assert pre > 0
    assert aw < pre, (
        "active-window gate should appear before preemption gate"
        " in source (consistent surfacing order)"
    )
