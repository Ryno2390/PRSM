"""R9 Phase 6.2 Task 3 — Bootstrap-list discovery over alternate transports.

R9-SCOPING-1 §5.2 chicken-and-egg problem: a PRSM node in a censoring
jurisdiction can't reach its bootstrap list to discover peers until it
knows which transport works — but it needs the bootstrap list to
configure said transport. Partial answer: ship the transport
configuration out-of-band (operator config file), then route the
bootstrap fetch through that configured transport.

This module wires an existing ``TransportAdapter`` into
``prsm.node.bootstrap``'s pre-existing ``HttpsBootstrapFetcher`` and
``DnsBootstrapFetcher`` dependency-injection points. Those fetchers
already accept callables (``get`` + ``resolve_txt``) for network I/O;
Task 3 just provides transport-routed implementations of those
callables.

HTTPS path
----------
httpx accepts SOCKS5 proxy URLs directly via ``httpx.HTTPTransport
(proxy=...)``. For the DirectAdapter case, no proxy is configured; for
SocksAdapter, we synthesize the SOCKS5 URL from the adapter's
configured proxy host/port. Authenticated proxies are supported.

DNS path
--------
DNS-over-HTTPS (DoH). Operators in censoring jurisdictions typically
cannot rely on direct DNS — the ISP sees and can filter/log queries.
DoH encrypts the query and tunnels it through HTTPS, which in turn
routes through the configured TransportAdapter. Any DoH-compliant
resolver works; operator configures their preferred resolver endpoint.

What this module does NOT do
----------------------------
- **Does not auto-configure transport selection.** Operators explicitly
  pick direct / socks5 / socks4 in their node config. Per R9 §8 the
  Foundation does not ship jurisdiction-specific presets.
- **Does not manage the proxy daemon.** ``tor`` / ``v2ray`` / ``trojan-
  client`` / etc. run separately; this module only consumes a SOCKS
  proxy the operator has already configured and verified.
- **Does not validate the bootstrap-list content.** That's
  ``prsm.node.bootstrap.parse_and_verify`` — this module only handles
  the fetch I/O.

Usage
-----

.. code-block:: python

    from prsm.node.bootstrap import (
        HttpsBootstrapFetcher, DnsBootstrapFetcher,
        discover_bootstrap_peers,
    )
    from prsm.node.bootstrap_transport import (
        make_https_get, make_doh_resolve_txt,
    )
    from prsm.node.transport_adapter import SocksAdapter

    adapter = SocksAdapter("127.0.0.1", 9050)  # operator configures

    primary = HttpsBootstrapFetcher(
        url="https://bootstrap.prsm.example/v1/peers.json",
        get=make_https_get(adapter),
    )
    fallback = DnsBootstrapFetcher(
        domain="_prsm.prsm.example",
        resolve_txt=make_doh_resolve_txt(
            adapter,
            resolver_url="https://1.1.1.1/dns-query",
        ),
    )
    peers = discover_bootstrap_peers(pubkey, primary=primary, fallback=fallback)
"""
from __future__ import annotations

import base64
import logging
from typing import Callable, List, Optional

from prsm.node.transport_adapter import (
    DirectAdapter,
    SocksAdapter,
    TransportAdapter,
    TransportConfigError,
)

logger = logging.getLogger(__name__)


# Default timeouts per R9 §5.2 concerns. Bootstrap is pre-connection;
# operator is typically starting up their node and willing to wait.
_DEFAULT_HTTPS_TIMEOUT = 30.0
_DEFAULT_DNS_TIMEOUT = 15.0


def _socks_proxy_url(adapter: SocksAdapter) -> str:
    """Synthesize a socks5:// or socks4:// URL from a SocksAdapter's config.

    Embeds optional username/password credentials per RFC 3986 userinfo
    syntax, URL-encoded to handle passwords with special characters.
    """
    import urllib.parse as _urlparse

    cfg = adapter.config
    scheme = "socks5h" if cfg.version == 5 and cfg.rdns else f"socks{cfg.version}"
    # socks5h = "SOCKS5 with remote DNS" — honors the adapter's rdns=True
    # by using a scheme httpx recognizes.
    if cfg.username:
        user = _urlparse.quote(cfg.username, safe="")
        if cfg.password:
            pwd = _urlparse.quote(cfg.password, safe="")
            auth = f"{user}:{pwd}@"
        else:
            auth = f"{user}@"
    else:
        auth = ""
    return f"{scheme}://{auth}{cfg.host}:{cfg.port}"


def make_https_get(
    adapter: TransportAdapter,
    *,
    timeout: float = _DEFAULT_HTTPS_TIMEOUT,
    follow_redirects: bool = True,
) -> Callable[[str], str]:
    """Build an HTTPS GET callable that routes through ``adapter``.

    The returned callable is compatible with
    ``HttpsBootstrapFetcher.get``: takes a URL, returns the response
    body as a string, raises on failure (the fetcher catches and
    converts to ``None``).

    :param adapter: TransportAdapter. DirectAdapter (default) uses
        httpx with no proxy; SocksAdapter synthesizes a
        ``socks5h://`` URL from its configured proxy host/port and
        passes it to httpx.
    :param timeout: Full-request wall-clock timeout in seconds.
    :param follow_redirects: Whether to follow HTTP 3xx redirects.
        Default True (matches browser + most HTTP client expectations).
    :returns: Callable[[str], str]
    """
    # Lazy-import httpx so environments that don't route through the
    # transport-adapter path don't pay the httpx-with-socks import cost.
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise TransportConfigError(
            "HTTPS-over-transport requires httpx. "
            "Install with: pip install httpx[socks]"
        ) from exc

    if isinstance(adapter, DirectAdapter):
        proxy = None
    elif isinstance(adapter, SocksAdapter):
        proxy = _socks_proxy_url(adapter)
    else:
        # Unknown adapter: try proxy-less. If the adapter needs special
        # routing it should subclass / extend this helper rather than
        # silently bypassing.
        logger.warning(
            "make_https_get: unknown adapter type %r; falling back to direct httpx",
            adapter.name,
        )
        proxy = None

    def _get(url: str) -> str:
        with httpx.Client(
            proxy=proxy,
            timeout=timeout,
            follow_redirects=follow_redirects,
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text

    return _get


def make_doh_resolve_txt(
    adapter: TransportAdapter,
    *,
    resolver_url: str = "https://cloudflare-dns.com/dns-query",
    timeout: float = _DEFAULT_DNS_TIMEOUT,
) -> Callable[[str], List[str]]:
    """Build a DNS TXT resolver that routes over DNS-over-HTTPS.

    DNS TXT queries that would normally travel in cleartext are wrapped
    in HTTPS and sent to a DoH-compliant resolver, then routed through
    the configured TransportAdapter. This simultaneously defeats
    ISP-level DNS filtering (common in censoring jurisdictions) and
    routes the query through the same circumvention transport as HTTPS
    bootstrap fetches.

    Compatible with ``DnsBootstrapFetcher.resolve_txt``.

    :param adapter: TransportAdapter. Same semantics as make_https_get.
    :param resolver_url: DoH endpoint. Common choices:

        - ``https://cloudflare-dns.com/dns-query`` (default; Cloudflare)
        - ``https://1.1.1.1/dns-query`` (same, IP-addressed)
        - ``https://dns.google/dns-query`` (Google)
        - ``https://dns.quad9.net/dns-query`` (Quad9)

        Foundation does NOT pick a default on behalf of censoring-
        jurisdiction operators (per R9 §8) — operators configure their
        preferred resolver based on their own threat model.
    :param timeout: Query timeout in seconds.
    :returns: Callable[[str], List[str]] — returns list of TXT record
        strings for the given domain, or empty list on any failure
        (DnsBootstrapFetcher interprets empty as "unavailable").
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise TransportConfigError(
            "DoH-over-transport requires httpx. "
            "Install with: pip install httpx[socks]"
        ) from exc

    if isinstance(adapter, DirectAdapter):
        proxy = None
    elif isinstance(adapter, SocksAdapter):
        proxy = _socks_proxy_url(adapter)
    else:
        logger.warning(
            "make_doh_resolve_txt: unknown adapter type %r; fallback direct",
            adapter.name,
        )
        proxy = None

    def _resolve(domain: str) -> List[str]:
        # RFC 8484 DoH GET form: ?dns=<base64url-encoded DNS query>.
        # Construct a minimal DNS query packet for TXT records.
        query = _build_txt_query(domain)
        query_b64 = base64.urlsafe_b64encode(query).rstrip(b"=").decode("ascii")
        try:
            with httpx.Client(
                proxy=proxy,
                timeout=timeout,
                follow_redirects=False,  # DoH resolvers shouldn't redirect
                headers={"Accept": "application/dns-message"},
            ) as client:
                response = client.get(resolver_url, params={"dns": query_b64})
                response.raise_for_status()
                return _parse_txt_response(response.content)
        except Exception as exc:
            logger.warning(
                "DoH TXT query for %s via %s failed: %s", domain, resolver_url, exc
            )
            return []

    return _resolve


# ---- Minimal DNS wire-format helpers ---------------------------------------
# We intentionally don't pull in dnspython for this — the TXT query
# format is simple enough to build by hand, and avoiding the dependency
# keeps the bootstrap-fetch path light.


def _build_txt_query(domain: str) -> bytes:
    """Build a DNS TXT query message per RFC 1035.

    Minimal query: one question, class IN (1), type TXT (16). No
    compression. Random-ish transaction ID.
    """
    import secrets
    import struct

    if not domain:
        raise TransportConfigError("domain must be non-empty")
    if len(domain) > 253:
        raise TransportConfigError(f"domain too long: {len(domain)} bytes")

    # DNS header: id, flags, qdcount, ancount, nscount, arcount.
    txn_id = secrets.randbits(16)
    flags = 0x0100  # standard query, recursion desired
    header = struct.pack("!HHHHHH", txn_id, flags, 1, 0, 0, 0)

    # Question: encoded domain name + qtype (TXT=16) + qclass (IN=1).
    qname = _encode_domain_name(domain)
    question = qname + struct.pack("!HH", 16, 1)

    return header + question


def _encode_domain_name(domain: str) -> bytes:
    """Encode a domain as DNS labels: length-prefixed, null-terminated."""
    out = bytearray()
    for label in domain.strip(".").split("."):
        if not label:
            continue
        label_bytes = label.encode("ascii", errors="strict")
        if len(label_bytes) > 63:
            raise TransportConfigError(
                f"DNS label too long: {len(label_bytes)} bytes in {label!r}"
            )
        out.append(len(label_bytes))
        out.extend(label_bytes)
    out.append(0)  # terminator
    return bytes(out)


def _parse_txt_response(response: bytes) -> List[str]:
    """Parse a DNS response message and extract TXT record strings.

    Returns the concatenation of all TXT rdata strings found in the
    answer section. TXT records contain one or more length-prefixed
    strings per record; this helper concatenates within-a-record
    strings and returns each RR's concatenated text as a separate
    list entry.

    On any malformed input, returns [] — DnsBootstrapFetcher treats
    empty as unavailable.
    """
    import struct

    if len(response) < 12:
        return []

    # Unpack header.
    _txn_id, _flags, qdcount, ancount, _nscount, _arcount = struct.unpack(
        "!HHHHHH", response[:12]
    )
    offset = 12

    # Skip the question section.
    for _ in range(qdcount):
        offset = _skip_name(response, offset)
        offset += 4  # qtype + qclass

    results: List[str] = []
    for _ in range(ancount):
        try:
            offset = _skip_name(response, offset)
            if offset + 10 > len(response):
                return results
            atype, _aclass, _ttl, rdlength = struct.unpack(
                "!HHIH", response[offset:offset + 10]
            )
            offset += 10
            rdata_end = offset + rdlength
            if rdata_end > len(response):
                return results
            if atype == 16:  # TXT
                text = _parse_txt_rdata(response[offset:rdata_end])
                if text:
                    results.append(text)
            offset = rdata_end
        except (struct.error, IndexError):
            return results

    return results


def _skip_name(msg: bytes, offset: int) -> int:
    """Advance past a DNS name (labels + optional pointer) in ``msg``."""
    while offset < len(msg):
        length = msg[offset]
        if length == 0:
            return offset + 1
        if length & 0xC0 == 0xC0:
            # Compression pointer: 2 bytes total.
            return offset + 2
        offset += 1 + length
    return offset


def _parse_txt_rdata(rdata: bytes) -> str:
    """Parse a TXT record's rdata: sequence of length-prefixed strings."""
    parts = []
    offset = 0
    while offset < len(rdata):
        length = rdata[offset]
        offset += 1
        if offset + length > len(rdata):
            break
        try:
            parts.append(rdata[offset:offset + length].decode("utf-8"))
        except UnicodeDecodeError:
            # TXT records are supposed to be ASCII but tolerate raw bytes.
            parts.append(rdata[offset:offset + length].decode("latin-1"))
        offset += length
    return "".join(parts)


__all__ = [
    "make_https_get",
    "make_doh_resolve_txt",
]
