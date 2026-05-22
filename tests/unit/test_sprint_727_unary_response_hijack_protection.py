"""Sprint 727 F60 — unary response hijack protection (F53 sibling).

Sprint 719 closed F53 on the streaming response handler. Sprint
727 closes the identical fix-class on the UNARY response handler
(`handle_chain_executor_response`).

Pre-727, the unary response handler looked up
`pending.get(request_id)` and resolved the future WITHOUT
verifying msg.sender_id. A peer who learned the request_id
(pre-sprint-724 deterministic derivation, network observation,
protocol leak) could send a forged CHAIN_RESP with that
request_id + attacker-controlled bytes. The victim's future
resolved with attacker bytes BEFORE the legitimate response
arrived — and the genuine response would then be silently dropped
(future.done() returns True).

In production: a malicious aggregator or compromised peer could
inject false inference output bytes that bypass the per-token
signature checks (those run on the inference receipt, not on
the wire-level routing).

Fix:
- `pending[request_id]` now stores `(future, expected_sender)`
  instead of bare future. expected_sender = the stage_address
  the dispatcher targeted.
- `handle_chain_executor_response` unpacks tuple + verifies
  msg.sender_id matches before resolving. Forged responses
  return False (silent drop); the genuine response can still
  resolve the future when it arrives.
- Defensive fallback for legacy bare-future shape so upgrade
  in-flight doesn't break.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock


def test_unary_dispatcher_registers_pending_as_future_and_sender_tuple():
    """Pin: pre-727 `pending[request_id] = future` (bare future);
    post-727 `pending[request_id] = (future, stage_address)`.
    Source-level invariant on build_send_message_adapter."""
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
    )
    src = inspect.getsource(build_send_message_adapter)
    assert "pending[request_id] = (future, stage_address)" in src, (
        "F60 fix requires pending[] to store (future, "
        "expected_sender) tuple so handle_chain_executor_response "
        "can verify sender"
    )


def test_unary_response_handler_drops_forged_response_from_wrong_sender():
    """Behavioral: response handler returns False when
    msg.sender_id does NOT match the dispatcher's expected_sender.
    The future is NOT resolved (set_result / set_exception not
    called) — the genuine response can still arrive later and
    resolve correctly."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY, CHAIN_PAYLOAD_KEY,
    )
    import base64

    node = MagicMock()
    legit_peer = "legit-peer-id-aaaaaaaaaaaaaaaaa"
    future = MagicMock()
    future.done.return_value = False
    node._chain_executor_pending = {
        "req-xyz": (future, legit_peer),
    }
    msg = MagicMock()
    msg.payload = {
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY: "req-xyz",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"forged").decode(),
    }
    msg.sender_id = "attacker-peer-zzzzzzzzzzzzzzzzz"
    handled = handle_chain_executor_response(node, msg)
    assert handled is False, (
        "F60: forged response from wrong sender must be dropped"
    )
    # Future must NOT have been resolved
    future.set_result.assert_not_called()
    future.set_exception.assert_not_called()


def test_unary_response_handler_resolves_with_matching_sender():
    """Legitimate path: msg.sender_id matches expected_sender →
    future.set_result is called with decoded payload bytes."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY, CHAIN_PAYLOAD_KEY,
    )
    import base64

    node = MagicMock()
    legit_peer = "legit-peer-id-aaaaaaaaaaaaaaaaa"
    future = MagicMock()
    future.done.return_value = False
    node._chain_executor_pending = {
        "req-xyz": (future, legit_peer),
    }
    msg = MagicMock()
    msg.payload = {
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY: "req-xyz",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"legit-response").decode(),
    }
    msg.sender_id = legit_peer
    handled = handle_chain_executor_response(node, msg)
    assert handled is True
    future.set_result.assert_called_once_with(b"legit-response")


def test_unary_response_handler_legacy_bare_future_still_works():
    """Defensive: if pending[] still has the legacy bare-future
    shape (pre-727 in-flight request that survived an upgrade),
    the response handler should NOT crash. Falls back to non-
    binding mode."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY, CHAIN_PAYLOAD_KEY,
    )
    import base64

    node = MagicMock()
    future = MagicMock()
    future.done.return_value = False
    # Legacy: bare future, no tuple
    node._chain_executor_pending = {"req-xyz": future}
    msg = MagicMock()
    msg.payload = {
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY: "req-xyz",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"resp").decode(),
    }
    msg.sender_id = "any-peer"
    handled = handle_chain_executor_response(node, msg)
    assert handled is True
    future.set_result.assert_called_once_with(b"resp")


def test_unary_response_handler_drops_when_future_already_done():
    """Existing invariant preserved: if future is already resolved
    (duplicate response arrival), drop silently. The earlier
    sender check must NOT short-circuit this — pre-727 done-check
    happened before sender check; post-727 the order is reversed
    (sender check first, then done check) and that's also fine."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY, CHAIN_PAYLOAD_KEY,
    )
    import base64

    node = MagicMock()
    legit_peer = "legit-peer-id-aaaaaaaaaaaaaaaaa"
    future = MagicMock()
    future.done.return_value = True  # already resolved
    node._chain_executor_pending = {"req-xyz": (future, legit_peer)}
    msg = MagicMock()
    msg.payload = {
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY: "req-xyz",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"dup").decode(),
    }
    msg.sender_id = legit_peer
    handled = handle_chain_executor_response(node, msg)
    assert handled is False
    future.set_result.assert_not_called()
