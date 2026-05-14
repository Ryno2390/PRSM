"""Sprint 385 — bootstrap-test probe helper.

Background for `prsm node bootstrap-test` CLI subcommand.
Probes the canonical bootstrap-host list (or operator-
supplied URLs) and reports per-host TCP / TLS / WSS health
+ latency. Operator-side diagnostic for "is my regional
bootstrap up, or is something local broken?"

Distinct from sprint 380's `prsm node bootstrap`, which
reports THIS node's bootstrap-registration state. The
probe here doesn't require the node to be running — it's
a from-anywhere reachability check.
"""
from __future__ import annotations

import asyncio
import socket
import ssl
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from urllib.parse import urlparse


class ProbeStatus(str, Enum):
    OK = "ok"
    DNS_FAIL = "dns_fail"
    TCP_FAIL = "tcp_fail"
    TLS_FAIL = "tls_fail"
    WSS_FAIL = "wss_fail"
    TIMEOUT = "timeout"


@dataclass
class HostProbe:
    """One bootstrap host's probe result."""
    url: str
    host: str
    port: int
    status: ProbeStatus
    tcp_ok: bool = False
    tls_ok: bool = False
    wss_ok: bool = False
    latency_ms: Optional[float] = None
    cert_subject: Optional[str] = None
    cert_issuer: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "tcp_ok": self.tcp_ok,
            "tls_ok": self.tls_ok,
            "wss_ok": self.wss_ok,
            "latency_ms": self.latency_ms,
            "cert_subject": self.cert_subject,
            "cert_issuer": self.cert_issuer,
            "error": self.error,
        }


@dataclass
class FleetProbe:
    """Aggregate probe of all bootstrap hosts."""
    hosts: List[HostProbe] = field(default_factory=list)

    @property
    def healthy_count(self) -> int:
        return sum(
            1 for h in self.hosts if h.status == ProbeStatus.OK
        )

    @property
    def total_count(self) -> int:
        return len(self.hosts)

    @property
    def all_healthy(self) -> bool:
        return (
            self.total_count > 0
            and self.healthy_count == self.total_count
        )

    @property
    def any_healthy(self) -> bool:
        return self.healthy_count > 0

    def to_dict(self) -> dict:
        return {
            "hosts": [h.to_dict() for h in self.hosts],
            "summary": {
                "total": self.total_count,
                "healthy": self.healthy_count,
                "degraded": (
                    self.total_count - self.healthy_count
                ),
                "all_healthy": self.all_healthy,
                "any_healthy": self.any_healthy,
            },
        }


def parse_bootstrap_url(url: str) -> tuple[str, int]:
    """Extract (host, port) from a bootstrap URL.

    Accepts:
      wss://host:port
      ws://host:port
      host:port
      host (defaults to port 8765)
    """
    if "://" in url:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port or (
            443 if parsed.scheme == "wss" else 8765
        )
    elif ":" in url:
        host, _, port_str = url.partition(":")
        port = int(port_str)
    else:
        host = url
        port = 8765
    return host, port


async def probe_host(
    url: str,
    *,
    timeout_seconds: float = 10.0,
) -> HostProbe:
    """Probe one bootstrap host: TCP → TLS → WSS handshake.

    Each layer's success/failure is captured independently
    so the result tells you exactly where the chain broke
    (DNS missing? TCP blocked? TLS cert wrong? WSS upgrade
    fails?).
    """
    host, port = parse_bootstrap_url(url)
    result = HostProbe(
        url=url, host=host, port=port,
        status=ProbeStatus.OK,
    )
    start = time.monotonic()

    # ── DNS + TCP ──
    try:
        # asyncio.open_connection handles DNS + TCP in one
        # shot, with a clean timeout boundary.
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout_seconds,
        )
        result.tcp_ok = True
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass
    except asyncio.TimeoutError:
        # Must catch BEFORE OSError — in Python 3.11+,
        # asyncio.TimeoutError IS a subclass of OSError.
        result.status = ProbeStatus.TIMEOUT
        result.error = (
            f"TCP timeout after {timeout_seconds}s"
        )
        result.latency_ms = timeout_seconds * 1000.0
        return result
    except (socket.gaierror, OSError) as e:
        if isinstance(e, socket.gaierror):
            result.status = ProbeStatus.DNS_FAIL
        else:
            result.status = ProbeStatus.TCP_FAIL
        result.error = f"{type(e).__name__}: {e}"
        result.latency_ms = (
            time.monotonic() - start
        ) * 1000.0
        return result

    # ── TLS handshake ──
    try:
        ssl_ctx = ssl.create_default_context()
        tls_reader, tls_writer = await asyncio.wait_for(
            asyncio.open_connection(
                host, port,
                ssl=ssl_ctx,
                server_hostname=host,
            ),
            timeout=timeout_seconds,
        )
        result.tls_ok = True
        # Capture cert info while connection is open
        ssl_obj = tls_writer.get_extra_info("ssl_object")
        if ssl_obj is not None:
            cert = ssl_obj.getpeercert()
            if cert:
                subject = dict(
                    x[0] for x in cert.get("subject", [])
                )
                issuer = dict(
                    x[0] for x in cert.get("issuer", [])
                )
                result.cert_subject = subject.get(
                    "commonName",
                )
                result.cert_issuer = issuer.get(
                    "organizationName",
                ) or issuer.get("commonName")
        tls_writer.close()
        try:
            await tls_writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass
    except (ssl.SSLError, OSError) as e:
        result.status = ProbeStatus.TLS_FAIL
        result.error = f"{type(e).__name__}: {e}"
        result.latency_ms = (
            time.monotonic() - start
        ) * 1000.0
        return result
    except asyncio.TimeoutError:
        result.status = ProbeStatus.TIMEOUT
        result.error = (
            f"TLS timeout after {timeout_seconds}s"
        )
        result.latency_ms = timeout_seconds * 1000.0
        return result

    # ── WSS handshake ──
    try:
        # Import locally so tests that don't exercise WSS
        # don't pay the websockets import cost.
        import websockets
        wss_url = f"wss://{host}:{port}"
        async with await asyncio.wait_for(
            websockets.connect(wss_url, open_timeout=timeout_seconds),
            timeout=timeout_seconds,
        ) as ws:
            result.wss_ok = True
            # Close immediately; the goal is handshake-only
            # not registration-protocol exercise
    except Exception as e:  # noqa: BLE001
        result.status = ProbeStatus.WSS_FAIL
        result.error = f"{type(e).__name__}: {e}"
        result.latency_ms = (
            time.monotonic() - start
        ) * 1000.0
        return result

    result.latency_ms = (time.monotonic() - start) * 1000.0
    return result


async def probe_fleet(
    urls: List[str],
    *,
    timeout_seconds: float = 10.0,
) -> FleetProbe:
    """Probe all hosts in parallel. Returns FleetProbe with
    per-host results."""
    tasks = [
        probe_host(url, timeout_seconds=timeout_seconds)
        for url in urls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return FleetProbe(hosts=list(results))


def canonical_bootstrap_urls() -> List[str]:
    """Default bootstrap URL list. Honors env-var overrides
    so operators can test their own custom bootstrap fleets
    without modifying canonical defaults."""
    from prsm.node.config import (
        DEFAULT_BOOTSTRAP_NODES,
        FALLBACK_BOOTSTRAP_NODES,
    )
    merged: List[str] = []
    seen: set = set()
    for url in (
        list(DEFAULT_BOOTSTRAP_NODES)
        + list(FALLBACK_BOOTSTRAP_NODES)
    ):
        if url and url not in seen:
            merged.append(url)
            seen.add(url)
    return merged
