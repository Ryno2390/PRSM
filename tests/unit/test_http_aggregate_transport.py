"""B7 — HttpAggregateTransport tests.

TDD coverage for the HTTP/TLS implementation of the
``AggregateTransport`` Protocol from
``prsm.compute.query_orchestrator.aggregator_client_adapter``.

Per `docs/2026-05-08-aggregate-rpc-design.md` §"Client-side flow" —
the transport POSTs an ``AggregateRequest`` (canonical JSON) to
``{aggregator_url}/compute/aggregate`` and returns the parsed
``AggregateResponse``. Errors map to ``AggregateTransportError`` so
the adapter / orchestrator retry-loop can route on them uniformly.

Uses ``httpx.MockTransport`` for the actual HTTP layer — no real
sockets, no extra deps beyond the existing PRSM ``httpx`` dep.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

# The autouse `mock_http_requests` fixture in tests/conftest.py patches
# `httpx.AsyncClient` to a MagicMock, which breaks isinstance checks
# inside the production code. Pull the *real* class straight off the
# module by import-time capture, mirroring what conftest.py does for
# its own `_real_httpx_*` shadows. We re-bind the real classes back
# onto the production module + the local symbol so MockTransport-based
# tests use them, then restore the mock after each test (autouse
# fixture below).
_real_AsyncClient = httpx.AsyncClient
_real_MockTransport = httpx.MockTransport
_real_Response = httpx.Response
_real_TimeoutException = httpx.TimeoutException
_real_ConnectError = httpx.ConnectError


@pytest.fixture(autouse=True)
def _restore_real_httpx_classes(monkeypatch):
    """Override the global ``mock_http_requests`` fixture's
    ``patch('httpx.AsyncClient', ...)`` for this module's tests so
    real-class isinstance checks + ``MockTransport`` round-trips work.

    The conftest fixture runs first, replacing the attribute; this
    fixture runs second (autouse) and re-patches the attribute back to
    the real class for the test body, then yields. ``monkeypatch``
    auto-undoes after the test, but the conftest fixture's
    ``with patch(...)`` context manager exits at session boundary
    anyway — net effect: production module sees real ``httpx`` for
    each test in this file.
    """
    monkeypatch.setattr(httpx, "AsyncClient", _real_AsyncClient)
    monkeypatch.setattr(httpx, "MockTransport", _real_MockTransport)
    monkeypatch.setattr(httpx, "Response", _real_Response)
    monkeypatch.setattr(httpx, "TimeoutException", _real_TimeoutException)
    monkeypatch.setattr(httpx, "ConnectError", _real_ConnectError)
    yield

from prsm.compute.query_orchestrator import (
    AggregationCommit,
    SignedPartial,
)
from prsm.compute.query_orchestrator.aggregate_protocol import (
    AggregateRequest,
    AggregateResponse,
)
from prsm.compute.query_orchestrator.http_aggregate_transport import (
    AggregateTransportError,
    HttpAggregateTransport,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _make_request() -> AggregateRequest:
    """Synthesize a valid AggregateRequest with one signed partial."""
    src_priv = ed25519.Ed25519PrivateKey.generate()
    src_pub = src_priv.public_key().public_bytes_raw()
    partial = SignedPartial(
        shard_cid="prsm:shard-0",
        payload=b"partial-bytes",
        creator_id="creator-1",
        dp_noise_applied=True,
        source_agent_pubkey=src_pub,
        source_agent_signature=b"\x11" * 64,
        privacy_budget_consumed=0.1,
    )
    return AggregateRequest(
        request_id=b"\x01" * 32,
        query_id=b"\x02" * 32,
        manifest_json='{"query":"count","instructions":[{"op":"COUNT"}]}',
        partials=(partial,),
        prompter_pubkey=b"\x03" * 32,
        prompter_node_id="prompter-1",
        beacon_used=b"\x04" * 32,
        aggregator_pubkey_hash=b"\x05" * 32,
        ftns_budget=1000,
        deadline_unix=int(time.time()) + 60,
        prompter_signature=b"\x06" * 64,
    )


def _make_response(*, request: AggregateRequest) -> AggregateResponse:
    """Synthesize a structurally-valid AggregateResponse echoing the
    request's request_id + query_id. Signatures and digest are placeholders
    — the HTTP transport does NOT verify them (the adapter does)."""
    plaintext = b"combined-output"
    commit = AggregationCommit(
        query_id=request.query_id,
        aggregator_pubkey_hash=b"\x05" * 32,
        result_digest=hashlib.sha256(plaintext).digest(),
    )
    return AggregateResponse(
        request_id=request.request_id,
        query_id=request.query_id,
        commit=commit,
        commit_signature=b"\x07" * 64,
        encrypted_plaintext=plaintext,
        nonce=b"\x00" * 24,
        aggregator_pubkey=b"\x08" * 32,
        privacy_budget_consumed=0.5,
        contributing_creators=("creator-1",),
        completed_unix=int(time.time()),
    )


def _resolver(node_id: str) -> str:
    return f"https://aggregator.example/{node_id}"


def _build_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ──────────────────────────────────────────────────────────────────────
# 1. Happy path — round-trip
# ──────────────────────────────────────────────────────────────────────


def test_happy_path_round_trip():
    request = _make_request()
    response = _make_response(request=request)

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response.to_dict())

    client = _build_client(handler)
    transport = HttpAggregateTransport(
        endpoint_resolver=_resolver,
        http_client=client,
    )

    out = asyncio.run(transport.send("agg-1", request, timeout_seconds=10.0))
    asyncio.run(client.aclose())

    assert isinstance(out, AggregateResponse)
    assert out.request_id == request.request_id
    assert out.query_id == request.query_id
    assert out.encrypted_plaintext == response.encrypted_plaintext


# ──────────────────────────────────────────────────────────────────────
# 2. Endpoint resolver called with aggregator_node_id
# ──────────────────────────────────────────────────────────────────────


def test_endpoint_resolver_called_with_node_id():
    request = _make_request()
    response = _make_response(request=request)
    seen_urls: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(http_request.url))
        return httpx.Response(200, json=response.to_dict())

    resolved: list[str] = []

    def tracking_resolver(node_id: str) -> str:
        resolved.append(node_id)
        return f"https://aggregator.example/{node_id}"

    client = _build_client(handler)
    transport = HttpAggregateTransport(
        endpoint_resolver=tracking_resolver,
        http_client=client,
    )

    asyncio.run(transport.send("agg-xyz", request, timeout_seconds=10.0))
    asyncio.run(client.aclose())

    assert resolved == ["agg-xyz"]
    # URL is base + /compute/aggregate
    assert seen_urls == ["https://aggregator.example/agg-xyz/compute/aggregate"]


# ──────────────────────────────────────────────────────────────────────
# 3. Request body is JSON-serialized AggregateRequest.to_dict()
# ──────────────────────────────────────────────────────────────────────


def test_request_body_is_canonical_json():
    request = _make_request()
    response = _make_response(request=request)
    captured: dict = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured["body"] = http_request.content
        captured["content_type"] = http_request.headers.get("content-type", "")
        return httpx.Response(200, json=response.to_dict())

    client = _build_client(handler)
    transport = HttpAggregateTransport(
        endpoint_resolver=_resolver, http_client=client,
    )
    asyncio.run(transport.send("agg-1", request, timeout_seconds=10.0))
    asyncio.run(client.aclose())

    body = json.loads(captured["body"].decode("utf-8"))
    # Round-trip via from_dict must reproduce the request.
    rebuilt = AggregateRequest.from_dict(body)
    assert rebuilt == request
    assert "application/json" in captured["content_type"]


# ──────────────────────────────────────────────────────────────────────
# 4. Response correctly parsed back to AggregateResponse
# ──────────────────────────────────────────────────────────────────────


def test_response_parsed_back_to_dataclass():
    request = _make_request()
    response = _make_response(request=request)

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response.to_dict())

    client = _build_client(handler)
    transport = HttpAggregateTransport(
        endpoint_resolver=_resolver, http_client=client,
    )
    out = asyncio.run(transport.send("agg-1", request, timeout_seconds=10.0))
    asyncio.run(client.aclose())

    # Field-by-field equality (frozen dataclass equality).
    assert out == response


# ──────────────────────────────────────────────────────────────────────
# 5. HTTP 503 → AggregateTransportError with status code preserved
# ──────────────────────────────────────────────────────────────────────


def test_http_503_raises_transport_error_with_status():
    request = _make_request()

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="aggregator overloaded")

    client = _build_client(handler)
    transport = HttpAggregateTransport(
        endpoint_resolver=_resolver, http_client=client,
    )
    with pytest.raises(AggregateTransportError) as exc_info:
        asyncio.run(transport.send("agg-1", request, timeout_seconds=10.0))
    asyncio.run(client.aclose())

    assert exc_info.value.status_code == 503
    assert "503" in str(exc_info.value)


def test_http_400_raises_transport_error_with_status():
    """4xx must also surface status_code."""
    request = _make_request()

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="malformed request")

    client = _build_client(handler)
    transport = HttpAggregateTransport(
        endpoint_resolver=_resolver, http_client=client,
    )
    with pytest.raises(AggregateTransportError) as exc_info:
        asyncio.run(transport.send("agg-1", request, timeout_seconds=10.0))
    asyncio.run(client.aclose())

    assert exc_info.value.status_code == 400


# ──────────────────────────────────────────────────────────────────────
# 6. Timeout → AggregateTransportError("timeout")
# ──────────────────────────────────────────────────────────────────────


def test_timeout_raises_transport_error():
    request = _make_request()

    def handler(http_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("read timed out", request=http_request)

    client = _build_client(handler)
    transport = HttpAggregateTransport(
        endpoint_resolver=_resolver, http_client=client,
    )
    with pytest.raises(AggregateTransportError) as exc_info:
        asyncio.run(transport.send("agg-1", request, timeout_seconds=0.5))
    asyncio.run(client.aclose())

    assert "timeout" in str(exc_info.value).lower()


# ──────────────────────────────────────────────────────────────────────
# 7. Connection error → AggregateTransportError
# ──────────────────────────────────────────────────────────────────────


def test_connection_error_raises_transport_error():
    request = _make_request()

    def handler(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=http_request)

    client = _build_client(handler)
    transport = HttpAggregateTransport(
        endpoint_resolver=_resolver, http_client=client,
    )
    with pytest.raises(AggregateTransportError) as exc_info:
        asyncio.run(transport.send("agg-1", request, timeout_seconds=10.0))
    asyncio.run(client.aclose())

    msg = str(exc_info.value).lower()
    assert "connection" in msg


# ──────────────────────────────────────────────────────────────────────
# 8. Non-aggregate-response type field → AggregateTransportError
# ──────────────────────────────────────────────────────────────────────


def test_wrong_response_type_raises():
    request = _make_request()
    response = _make_response(request=request)
    bad = response.to_dict()
    bad["type"] = "something_else"

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=bad)

    client = _build_client(handler)
    transport = HttpAggregateTransport(
        endpoint_resolver=_resolver, http_client=client,
    )
    with pytest.raises(AggregateTransportError):
        asyncio.run(transport.send("agg-1", request, timeout_seconds=10.0))
    asyncio.run(client.aclose())


def test_malformed_json_response_raises():
    """Garbage body — JSON decode failure surfaces as transport error."""
    request = _make_request()

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json {{{")

    client = _build_client(handler)
    transport = HttpAggregateTransport(
        endpoint_resolver=_resolver, http_client=client,
    )
    with pytest.raises(AggregateTransportError):
        asyncio.run(transport.send("agg-1", request, timeout_seconds=10.0))
    asyncio.run(client.aclose())


# ──────────────────────────────────────────────────────────────────────
# 9. Ephemeral client — http_client=None creates one per call
# ──────────────────────────────────────────────────────────────────────


def test_ephemeral_client_per_call_when_none():
    """When constructed with ``http_client=None`` the transport must
    still complete a request (creating + closing a client per send).
    Verify by spying on ``httpx.AsyncClient`` construction."""
    request = _make_request()
    response = _make_response(request=request)

    captured: list[httpx.Request] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.append(http_request)
        return httpx.Response(200, json=response.to_dict())

    # The transport will create its own client; we patch the default
    # transport via the factory hook so MockTransport is what backs it.
    import prsm.compute.query_orchestrator.http_aggregate_transport as mod

    real_async_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    orig = mod._make_default_client
    mod._make_default_client = factory
    try:
        transport = HttpAggregateTransport(
            endpoint_resolver=_resolver, http_client=None,
        )
        out = asyncio.run(transport.send("agg-1", request, timeout_seconds=10.0))
    finally:
        mod._make_default_client = orig

    assert isinstance(out, AggregateResponse)
    assert len(captured) == 1


# ──────────────────────────────────────────────────────────────────────
# 10. Satisfies AggregateTransport Protocol
# ──────────────────────────────────────────────────────────────────────


def test_satisfies_aggregate_transport_protocol():
    from prsm.compute.query_orchestrator.aggregator_client_adapter import (
        AggregateTransport,
    )

    transport = HttpAggregateTransport(
        endpoint_resolver=_resolver, http_client=None,
    )
    assert isinstance(transport, AggregateTransport)
