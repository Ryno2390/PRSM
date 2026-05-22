"""Sprint 741 F69 — rate limiter keyed by HTTP requester, not local node.

Pre-741, three endpoints used a "per-requester" rate limiter:

    PRSM_FORGE_MAX_RPS_PER_REQUESTER     (/compute/forge)
    PRSM_INFERENCE_MAX_RPS_PER_REQUESTER (/compute/inference)
    PRSM_INFERENCE_MAX_RPS_PER_REQUESTER (/compute/inference/stream)

All three computed the bucket key as:

    requester = node.identity.node_id if node.identity else "anonymous"

`node.identity.node_id` is the LOCAL daemon's own ed25519-derived
node id — a constant. ALL HTTP callers shared one bucket key,
making the "per requester" rate limit effectively GLOBAL across
all clients. Consequences:

- An attacker spamming from one IP exhausts the bucket → legit
  clients from OTHER IPs get 429 (single-client starvation by
  unrelated attacker).
- The env var name + log messages claim "per requester" but
  semantics were "global" — confusing for operators tuning the
  limit.

Fix: new `_resolve_requester_key(request)` helper resolves a
per-HTTP-requester key with proxy-awareness (X-Forwarded-For
last-hop > X-Real-IP > client.host > "anonymous"). All three
endpoints now use this. Bucket keys are per-IP (or proxy-
inferred per-upstream-IP), matching the env var name.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock


def test_resolve_requester_key_prefers_xff_last_hop():
    """XFF last-hop (sprint 737 pattern) is the most-trusted
    forwarded-client signal. Multi-hop XFF → take rightmost."""
    from prsm.node.api import _resolve_requester_key
    request = MagicMock()
    request.headers = {"x-forwarded-for": "1.2.3.4, 5.6.7.8, 9.9.9.9"}
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    assert _resolve_requester_key(request) == "9.9.9.9"


def test_resolve_requester_key_falls_back_to_x_real_ip():
    """If XFF absent, use X-Real-IP (sprint 738 path parity)."""
    from prsm.node.api import _resolve_requester_key
    request = MagicMock()
    request.headers = {"x-real-ip": "203.0.113.42"}
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    assert _resolve_requester_key(request) == "203.0.113.42"


def test_resolve_requester_key_falls_back_to_client_host():
    """No proxy headers → request.client.host (direct
    connection). Matches sprint 734 immediate-client check."""
    from prsm.node.api import _resolve_requester_key
    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "203.0.113.42"
    assert _resolve_requester_key(request) == "203.0.113.42"


def test_resolve_requester_key_anonymous_fallback():
    """Pathological case (no headers + no client): return
    'anonymous' so rate limiter still works rather than
    crashing on None."""
    from prsm.node.api import _resolve_requester_key
    request = MagicMock()
    request.headers = {}
    request.client = None
    assert _resolve_requester_key(request) == "anonymous"


def test_forge_endpoint_uses_resolve_requester_key():
    """Pin via source inspection: /compute/forge's rate-limit
    block calls _resolve_requester_key(request), NOT the buggy
    pre-741 `node.identity.node_id` pattern."""
    from prsm.node import api as _api
    src = inspect.getsource(_api.create_api_app)
    # Find the forge block by its env var name + check the
    # surrounding code references _resolve_requester_key.
    forge_idx = src.find("PRSM_FORGE_MAX_RPS_PER_REQUESTER")
    assert forge_idx > 0
    forge_block = src[forge_idx:forge_idx + 2000]
    assert "_resolve_requester_key" in forge_block, (
        "/compute/forge must use _resolve_requester_key (F69 fix)"
    )


def test_inference_endpoint_uses_resolve_requester_key():
    """Pin: /compute/inference uses the helper."""
    from prsm.node import api as _api
    src = inspect.getsource(_api.create_api_app)
    inf_idx = src.find(
        '@app.post("/compute/inference")'
    )
    assert inf_idx > 0
    inf_block = src[inf_idx:inf_idx + 4500]
    assert "_resolve_requester_key" in inf_block, (
        "/compute/inference must use _resolve_requester_key"
    )


def test_inference_stream_endpoint_uses_resolve_requester_key():
    """Pin: /compute/inference/stream uses the helper."""
    from prsm.node import api as _api
    src = inspect.getsource(_api.create_api_app)
    inf_idx = src.find(
        '@app.post("/compute/inference/stream")'
    )
    assert inf_idx > 0
    inf_block = src[inf_idx:inf_idx + 4500]
    assert "_resolve_requester_key" in inf_block, (
        "/compute/inference/stream must use _resolve_requester_key"
    )


def test_two_different_clients_get_independent_buckets():
    """Behavioral: two requests with DIFFERENT X-Forwarded-For
    last-hops produce DIFFERENT bucket keys. This proves the F69
    fix actually achieves per-requester isolation rather than
    globalizing again."""
    from prsm.node.api import _resolve_requester_key
    r1 = MagicMock()
    r1.headers = {"x-forwarded-for": "1.1.1.1"}
    r1.client = MagicMock()
    r1.client.host = "127.0.0.1"
    r2 = MagicMock()
    r2.headers = {"x-forwarded-for": "2.2.2.2"}
    r2.client = MagicMock()
    r2.client.host = "127.0.0.1"
    key1 = _resolve_requester_key(r1)
    key2 = _resolve_requester_key(r2)
    assert key1 != key2, (
        "two clients with different XFFs must produce different "
        "bucket keys — otherwise F69 isn't actually fixed"
    )
