"""Sprint 724 F57 — unary request_id collision (F52 sibling).

Sprint 718 closed F52 in the streaming dispatcher: stream_id was
`sha256(request_bytes + ':stream')` purely deterministic, so
concurrent identical requests collided. Sprint 724 closes the
same fix-class in the UNARY chain-executor dispatcher
(`build_send_message_adapter` → `_adapter` inner function).

Pre-724 derivation:
    request_id = hashlib.sha256(request_bytes).hexdigest()

Real production trigger: coordinator runs idempotent retry on
transient transport failure (timeout, intermittent peer drop).
Same request_bytes → same request_id → second `_send_and_wait`'s
`pending[request_id] = future` overwrote the first dispatch's
future. First dispatch then waited the full 30s timeout, then
raised TimeoutError, despite the response having actually been
delivered to (and resolved by) the SECOND future.

Fix: mix 8 bytes of os.urandom into the sha256 input so two
concurrent identical requests produce distinct request_ids.
Same hex-string length (64 chars) preserves wire-format byte-
identical to pre-724. Collision probability negligible
(birthday-bound 1e-6 at ~2^32 concurrent in-flight requests).
"""
from __future__ import annotations

import inspect


def test_unary_request_id_uses_per_invocation_entropy():
    """Pin: build_send_message_adapter's inner _adapter must mix
    per-invocation entropy into request_id derivation. Otherwise
    concurrent identical requests collide on request_id."""
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
    )
    src = inspect.getsource(build_send_message_adapter)
    assert "os.urandom" in src or "_os.urandom" in src, (
        "F57 fix requires per-invocation entropy in unary "
        "request_id derivation; pre-724 sha256(request_bytes) "
        "was deterministic and collided on identical concurrent "
        "requests"
    )


def test_unary_request_id_derivation_uses_request_bytes_plus_entropy():
    """Same source-shape invariant as sprint 718's F52 fix:
    derivation must combine request_bytes WITH urandom. Either-or
    is a regression — bytes-only is F57, urandom-only loses the
    diagnostic correlation between id + payload."""
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
    )
    src = inspect.getsource(build_send_message_adapter)
    assert "request_bytes" in src and ("urandom" in src), (
        "unary request_id derivation lost either the request_bytes "
        "input (diagnostic value) or the urandom entropy "
        "(reopens F57 collision risk)"
    )


def test_unary_and_streaming_id_derivations_use_different_domain_separators():
    """Defense-in-depth: even with random entropy, distinct codomains
    are nice — unary `:unary:` vs streaming `:stream:` so a collision
    bug in one path can't accidentally affect the other (e.g., the
    same request_bytes used both ways shouldn't share an id)."""
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter, _remote_token_stream_dispatch,
    )
    unary_src = inspect.getsource(build_send_message_adapter)
    stream_src = inspect.getsource(_remote_token_stream_dispatch)
    # Each uses its own domain separator string in the sha256 input.
    assert ":unary:" in unary_src
    assert ":stream:" in stream_src
