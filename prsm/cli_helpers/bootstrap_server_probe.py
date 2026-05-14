"""Sprint 390 — bootstrap-server-side ops probe.

Hits the bootstrap server's HTTP API (`api_port`, default
`8000`) to read its `/health` + `/metrics` JSON surfaces in
a single ergonomic call. Backs the `prsm bootstrap-server
status` CLI.

Distinct from `prsm/cli_helpers/bootstrap_probe.py` (sprint
385), which probes the WSS endpoint of canonical bootstraps
from outside. This module probes the HTTP control surface
of a bootstrap server you are running.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx


class ProbeStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    CONNECT_FAIL = "connect_fail"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    UNKNOWN = "unknown"


@dataclass
class BootstrapServerProbe:
    host: str
    port: int
    status: ProbeStatus
    health: Optional[dict] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "health": self.health,
            "metrics": self.metrics,
            "error": self.error,
        }


async def fetch_server_status(
    host: str,
    port: int,
    *,
    timeout_seconds: float = 5.0,
    scheme: str = "http",
) -> BootstrapServerProbe:
    """Probe a bootstrap server's HTTP control surface.

    Order of operations: try `/health` first (the liveness
    signal). If `/health` fails, the metrics fetch is
    irrelevant — surface CONNECT_FAIL / TIMEOUT / HTTP_ERROR.
    If `/health` succeeds but `/metrics` fails, return
    PARTIAL — operators still get the liveness intel even
    when observability gravy is offline.
    """
    base = f"{scheme}://{host}:{port}"
    timeout = httpx.Timeout(timeout_seconds)
    health_payload: Optional[dict] = None
    metrics_payload: Optional[dict] = None
    error: Optional[str] = None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # /health — load-bearing liveness path
            try:
                resp = await client.get(f"{base}/health")
            except httpx.ConnectError as e:
                return BootstrapServerProbe(
                    host=host, port=port,
                    status=ProbeStatus.CONNECT_FAIL,
                    error=f"connection error: {e}",
                )
            except (httpx.ReadTimeout, httpx.ConnectTimeout, asyncio.TimeoutError) as e:
                return BootstrapServerProbe(
                    host=host, port=port,
                    status=ProbeStatus.TIMEOUT,
                    error=f"timeout: {e}",
                )

            if resp.status_code != 200:
                return BootstrapServerProbe(
                    host=host, port=port,
                    status=ProbeStatus.HTTP_ERROR,
                    error=(
                        f"/health returned {resp.status_code}: "
                        f"{getattr(resp, 'text', '')[:200]}"
                    ),
                )
            health_payload = resp.json()

            # /metrics — fail-soft. Operators care about
            # liveness even when observability is degraded.
            try:
                m_resp = await client.get(f"{base}/metrics")
                if m_resp.status_code == 200:
                    metrics_payload = m_resp.json()
                else:
                    error = (
                        f"/metrics returned {m_resp.status_code}"
                    )
            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                asyncio.TimeoutError,
            ) as e:
                error = f"/metrics fetch failed: {e}"

    except Exception as e:  # noqa: BLE001
        return BootstrapServerProbe(
            host=host, port=port,
            status=ProbeStatus.UNKNOWN,
            error=f"unexpected error: {e}",
        )

    if health_payload is not None and metrics_payload is not None:
        return BootstrapServerProbe(
            host=host, port=port, status=ProbeStatus.OK,
            health=health_payload, metrics=metrics_payload,
        )
    return BootstrapServerProbe(
        host=host, port=port, status=ProbeStatus.PARTIAL,
        health=health_payload, metrics=metrics_payload,
        error=error,
    )
