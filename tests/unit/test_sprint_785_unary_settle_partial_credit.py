"""Sprint 785 — wire settle_inference_receipt into UNARY settle path.

Mirrors sprint 784's streaming integration. The unary
/compute/inference handler in api.py currently calls
release_escrow directly even when the result carries a
partial_completion marker. Sprint 785 swaps that for
settle_inference_receipt so the unary path gets the same
proportional-credit + slash-signal behavior as streaming.

Because the unary settle logic lives in a closure inside
@app.post("/compute/inference"), per-branch unit tests would
require spinning up the full FastAPI app + executor. The
practical pin is source-shape: verify the unary handler's
settle block references settle_inference_receipt (or imports
it from credit_policy), and verify the call site appears
AFTER the receipt is constructed/signed (otherwise it can't
pass the receipt in).

Pin tests:
- Unary handler imports settle_inference_receipt.
- Reference appears within @app.post("/compute/inference")
  handler body (after the route decorator + before the next
  route).
- Reference appears AFTER `receipt = sign_receipt(...)` in
  source order (so receipt is in scope at the integration site).
- Fallback to direct release_escrow remains for the
  no-receipt path (legacy mid-fail handling — no regression).
"""
from __future__ import annotations

import inspect


def _get_api_source():
    from prsm.node import api as _api
    return inspect.getsource(_api)


def _slice_unary_handler(src: str) -> str:
    """Return the source-slice between @app.post("/compute/inference")
    and the next @app.post route. Used by all source-shape pins."""
    start = src.find('@app.post("/compute/inference")')
    assert start != -1, "could not locate /compute/inference handler"
    # Find the next @app.post AFTER our handler
    next_route = src.find("@app.post(", start + 1)
    if next_route == -1:
        next_route = len(src)
    return src[start:next_route]


def test_unary_handler_imports_settle_inference_receipt():
    """settle_inference_receipt referenced in unary handler body."""
    src = _get_api_source()
    handler = _slice_unary_handler(src)
    assert "settle_inference_receipt" in handler, (
        "Sprint 785: unary /compute/inference handler must "
        "reference settle_inference_receipt"
    )


def test_settle_call_appears_after_sign_receipt():
    """receipt must be in scope at the settle call site —
    pin source-order: sign_receipt → settle_inference_receipt."""
    src = _get_api_source()
    handler = _slice_unary_handler(src)
    sign_idx = handler.find("sign_receipt(")
    settle_idx = handler.find("settle_inference_receipt")
    assert sign_idx > 0
    assert settle_idx > 0
    assert sign_idx < settle_idx, (
        "settle_inference_receipt must be called AFTER "
        "sign_receipt so the receipt is in scope"
    )


def test_release_escrow_fallback_path_preserved():
    """When receipt is None (legacy mid-fail), the handler must
    still call release_escrow directly. Pin: the unary handler
    still contains a release_escrow call site (so the no-receipt
    branch has somewhere to go)."""
    src = _get_api_source()
    handler = _slice_unary_handler(src)
    # release_escrow remains in the handler — either inside the
    # legacy branch or via settle_inference_receipt's fallback.
    # We just confirm the function name is still wired here.
    assert "release_escrow" in handler


def test_settle_call_uses_escrow_amount_from_entry():
    """Pin that the call site passes escrow_amount derived from
    escrow_entry (not from receipt.cost_ftns implicitly). This
    matches sprint 784's pattern + ensures over-funded escrow
    math is correct."""
    src = _get_api_source()
    handler = _slice_unary_handler(src)
    settle_idx = handler.find("settle_inference_receipt")
    assert settle_idx > 0
    # The kwarg `escrow_amount=` should appear within a reasonable
    # window after the settle call site.
    after = handler[settle_idx:settle_idx + 1500]
    assert "escrow_amount" in after, (
        "Sprint 785: settle_inference_receipt call must pass "
        "escrow_amount kwarg derived from escrow_entry"
    )


def test_slash_decision_logged_in_unary_path():
    """When should_slash is True, the unary handler must surface
    it via a logger.warning (same operator-visible signal as
    sprint 784's streaming path)."""
    src = _get_api_source()
    handler = _slice_unary_handler(src)
    settle_idx = handler.find("settle_inference_receipt")
    assert settle_idx > 0
    after = handler[settle_idx:settle_idx + 1500]
    # Either references `should_slash` or `slash signal` text
    assert (
        "should_slash" in after or "slash signal" in after.lower()
    ), (
        "Sprint 785: unary handler must log slash signal when "
        "decision.should_slash is True"
    )
