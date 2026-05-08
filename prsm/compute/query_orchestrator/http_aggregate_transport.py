"""B7 — HttpAggregateTransport.

HTTP/TLS-backed implementation of the
``AggregateTransport`` Protocol from
``prsm/compute/query_orchestrator/aggregator_client_adapter.py``.
Production wires this with ``AggregatorClientAdapter`` so the
prompter ships an ``AggregateRequest`` to the selected aggregator's
``POST /compute/aggregate`` endpoint and parses the
``AggregateResponse`` reply.

Per `docs/2026-05-08-aggregate-rpc-design.md` §"Client-side flow"
(steps 4-5 — wire-encode + send + receive). Wire-encoding lives in
``aggregate_protocol.py::AggregateRequest.to_dict`` /
``AggregateResponse.from_dict``; this module only owns the HTTP
framing + error mapping.

Pattern-lift: timeout/header/error-mapping shape mirrors
``prsm/compute/chain_rpc/client.py`` — failures map to a single
typed exception (``AggregateTransportError``) so the adapter's
"transport failures bubble up unchanged" contract is satisfied
with one catchable type instead of many.

Constructor:
    endpoint_resolver: Callable[[str], str]
        Resolves an aggregator's ``node_id`` to a base URL. Production
        wires this to ``MarketplaceDirectory.get_listing(...)``; tests
        inject a stub. The transport appends ``/compute/aggregate`` to
        the resolved URL.
    http_client: httpx.AsyncClient | None
        Optional shared client. If None, an ephemeral client is created
        per ``send()`` call and closed before return — fine for tests
        and low-throughput callers, but production should inject a
        long-lived ``AsyncClient`` so connection pooling + TLS handshake
        amortize across queries.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

import httpx

from prsm.compute.query_orchestrator.aggregate_protocol import (
    AggregateRequest,
    AggregateResponse,
)
from prsm.compute.chain_rpc.protocol import (
    ChainRpcMalformedError,
    ChainRpcMessageType,
)


# ──────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────


class AggregateTransportError(RuntimeError):
    """Single typed exception for HTTP transport-level failures.

    Carries an optional ``status_code`` so callers can route 4xx vs 5xx
    differently (e.g. retry-loop policy treats 5xx + timeout as
    retryable, 4xx as terminal). The adapter's contract is that
    transport failures propagate unchanged, so the orchestrator's
    retry policy sees this exception directly.
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ──────────────────────────────────────────────────────────────────────
# Implementation
# ──────────────────────────────────────────────────────────────────────


# Override hook so tests can swap the ephemeral-client factory
# without monkey-patching httpx itself. Production code should NOT
# touch this.
def _make_default_client() -> httpx.AsyncClient:
    return httpx.AsyncClient()


# Excerpt cap when including the response body in error messages.
# Aggregator error bodies can be unbounded; truncate so logs stay
# readable + we don't leak very-large payloads via exception strings.
_BODY_EXCERPT_MAX_CHARS = 256


def _excerpt(body: bytes) -> str:
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        text = repr(body)
    if len(text) > _BODY_EXCERPT_MAX_CHARS:
        return text[: _BODY_EXCERPT_MAX_CHARS] + "..."
    return text


class HttpAggregateTransport:
    """Satisfies ``AggregateTransport`` Protocol via HTTP POST to the
    aggregator's ``/compute/aggregate`` endpoint.

    The ``send`` flow:
      1. Resolve ``aggregator_node_id`` → base URL via the injected
         ``endpoint_resolver``.
      2. Serialize ``request.to_dict()`` as JSON in the request body.
      3. POST to ``{url}/compute/aggregate`` with the call's
         ``timeout_seconds`` as the per-request timeout.
      4. On HTTP 200, decode the JSON body, validate the ``type``
         field == ``aggregate_response``, and return
         ``AggregateResponse.from_dict(body)``.
      5. On any failure (4xx / 5xx / timeout / connect error / decode
         error / wrong type field), raise
         ``AggregateTransportError`` with a descriptive message and
         (where applicable) the HTTP status code preserved.
    """

    # Endpoint path appended to the resolver-provided base URL.
    _AGGREGATE_PATH: str = "/compute/aggregate"

    def __init__(
        self,
        *,
        endpoint_resolver: Callable[[str], str],
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not callable(endpoint_resolver):
            raise TypeError("endpoint_resolver must be callable")
        if http_client is not None and not isinstance(http_client, httpx.AsyncClient):
            raise TypeError(
                "http_client must be httpx.AsyncClient or None, got "
                f"{type(http_client).__name__}"
            )
        self._endpoint_resolver = endpoint_resolver
        self._http_client = http_client

    async def send(
        self,
        aggregator_node_id: str,
        request: AggregateRequest,
        timeout_seconds: float,
    ) -> AggregateResponse:
        # Step 1: resolve node_id → URL.
        try:
            base_url = self._endpoint_resolver(aggregator_node_id)
        except Exception as exc:
            raise AggregateTransportError(
                f"endpoint resolution failed for node_id="
                f"{aggregator_node_id!r}: {exc}"
            ) from exc
        if not isinstance(base_url, str) or not base_url:
            raise AggregateTransportError(
                f"endpoint_resolver returned non-string URL for "
                f"node_id={aggregator_node_id!r}: {base_url!r}"
            )
        url = base_url.rstrip("/") + self._AGGREGATE_PATH

        # Step 2: serialize request.
        body_dict = request.to_dict()

        # Step 3: POST. Use the caller-provided client when available
        # (connection pooling); otherwise create + close an ephemeral
        # client. Either way the same internal helper is called so the
        # error-mapping logic is shared.
        if self._http_client is not None:
            return await self._post(self._http_client, url, body_dict, timeout_seconds)

        client = _make_default_client()
        try:
            return await self._post(client, url, body_dict, timeout_seconds)
        finally:
            await client.aclose()

    async def _post(
        self,
        client: httpx.AsyncClient,
        url: str,
        body_dict: dict,
        timeout_seconds: float,
    ) -> AggregateResponse:
        try:
            http_response = await client.post(
                url,
                json=body_dict,
                timeout=timeout_seconds,
                headers={"content-type": "application/json"},
            )
        except httpx.TimeoutException as exc:
            raise AggregateTransportError(
                f"timeout after {timeout_seconds:.3f}s POSTing to {url}: {exc}"
            ) from exc
        except httpx.ConnectError as exc:
            raise AggregateTransportError(
                f"connection failed to {url}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            # Catch-all for any other httpx-side transport error
            # (RequestError parents ConnectError + ReadError + WriteError
            # etc.; HTTPError is the broader root).
            raise AggregateTransportError(
                f"transport error to {url}: {exc}"
            ) from exc

        # Step 4: status check.
        if http_response.status_code != 200:
            raise AggregateTransportError(
                f"HTTP {http_response.status_code} from {url}: "
                f"{_excerpt(http_response.content)}",
                status_code=http_response.status_code,
            )

        # Step 5: decode body.
        try:
            body: Any = http_response.json()
        except json.JSONDecodeError as exc:
            raise AggregateTransportError(
                f"malformed JSON response from {url}: {exc}; body="
                f"{_excerpt(http_response.content)}"
            ) from exc

        if not isinstance(body, dict):
            raise AggregateTransportError(
                f"response body must be JSON object, got "
                f"{type(body).__name__}"
            )

        # Step 6: validate type field. Catch the "wrong type" case
        # explicitly with a clearer message than from_dict's
        # ``ChainRpcMalformedError`` would produce on its own — this
        # is the most likely server-side bug to surface.
        msg_type = body.get("type")
        if msg_type != ChainRpcMessageType.AGGREGATE_RESPONSE.value:
            raise AggregateTransportError(
                f"expected response type="
                f"{ChainRpcMessageType.AGGREGATE_RESPONSE.value!r}, got "
                f"{msg_type!r} from {url}"
            )

        # Step 7: parse to dataclass. ``from_dict`` enforces the rest of
        # the structural invariants (32-byte fields, partials list, etc.)
        # via ``__post_init__``. Wrap the canonical malformed error so
        # callers see a single typed exception.
        try:
            return AggregateResponse.from_dict(body)
        except ChainRpcMalformedError as exc:
            raise AggregateTransportError(
                f"malformed AggregateResponse from {url}: {exc}"
            ) from exc
