"""HTTP webhook delivery primitive for operator integrations.

Production feature: operators wire a webhook URL to receive
structured event notifications when critical conditions trip
(daemon crash, escrow leak, etc.). Future commits subscribe
this deliverer to specific event sources; this module ships the
delivery + signing + retry primitives in isolation.

Wire format:
  POST <webhook_url>
    Content-Type: application/json
    X-PRSM-Signature: sha256=<hmac>     (when shared secret set)
    X-PRSM-Event: <event_name>
    User-Agent: prsm-node/<version>
    <body: JSON payload>

HMAC-SHA256 signing uses an operator-shared secret + the raw
JSON body. Receiving systems verify by recomputing the HMAC
over the body and comparing to the X-PRSM-Signature header.
Missing-secret deploys send unsigned payloads (operator opted
out of signing).

Retry behavior:
- max_attempts (default 3) with exponential backoff
  (1s, 2s, 4s, ...)
- per-attempt timeout (default 10s)
- final failure returns DeliveryResult.failed with the last
  exception captured for ops triage
- non-retryable status codes (4xx except 408 / 429): give up
  immediately to avoid retry-storming on misconfigured webhook
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_BASE_BACKOFF_SECONDS = 1.0


@dataclass(frozen=True)
class DeliveryResult:
    success: bool
    status_code: Optional[int]
    attempts: int
    error: Optional[str] = None


def compute_signature(secret: str, body: bytes) -> str:
    """HMAC-SHA256 in the GitHub webhook style:
    `sha256=<hex digest>`. Receiving systems compare via
    constant-time comparison."""
    mac = hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256,
    )
    return f"sha256={mac.hexdigest()}"


def _is_retryable_status(code: int) -> bool:
    """Retry on 408 (timeout), 429 (rate limit), 5xx; never
    retry other 4xx (operator misconfiguration — webhook URL
    rejecting the payload shape)."""
    if code == 408 or code == 429:
        return True
    return 500 <= code < 600


class WebhookDeliverer:
    """Stateless deliverer — same instance can be reused across
    events. v1 uses aiohttp under the hood; replaced by direct
    httpx if the dependency footprint becomes a concern."""

    def __init__(
        self,
        *,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        base_backoff_seconds: float = _DEFAULT_BASE_BACKOFF_SECONDS,
        # Override for tests — sleep-stub.
        sleep_fn=None,
        # Optional log ring — every dispatch attempt records here.
        log_ring=None,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError(
                f"max_attempts must be positive, got {max_attempts}",
            )
        if timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {timeout_seconds}",
            )
        self._max_attempts = max_attempts
        self._timeout_seconds = timeout_seconds
        self._base_backoff_seconds = base_backoff_seconds
        self._sleep = sleep_fn or asyncio.sleep
        self._log_ring = log_ring

    async def deliver(
        self,
        *,
        url: str,
        event: str,
        payload: Dict[str, Any],
        secret: Optional[str] = None,
        # Override for tests — fake transport.
        post_fn=None,
    ) -> DeliveryResult:
        """POST the payload to the webhook URL with retry + signing.

        post_fn (test override): async callable taking
        (url, body_bytes, headers) → (status_code, body_text).
        Default impl uses aiohttp.
        """
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-PRSM-Event": event,
            "User-Agent": "prsm-node/0.24.0",
        }
        if secret:
            headers["X-PRSM-Signature"] = compute_signature(secret, body)

        if post_fn is None:
            post_fn = self._aiohttp_post

        last_error: Optional[str] = None
        last_status: Optional[int] = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                status, _resp_body = await asyncio.wait_for(
                    post_fn(url, body, headers),
                    timeout=self._timeout_seconds,
                )
                last_status = status
                if 200 <= status < 300:
                    success_result = DeliveryResult(
                        success=True,
                        status_code=status,
                        attempts=attempt,
                    )
                    self._record(event, url, success_result)
                    return success_result
                if not _is_retryable_status(status):
                    nonretry = DeliveryResult(
                        success=False,
                        status_code=status,
                        attempts=attempt,
                        error=(
                            f"non-retryable status {status} "
                            f"(operator misconfiguration?)"
                        ),
                    )
                    self._record(event, url, nonretry)
                    return nonretry
                last_error = f"retryable status {status}"
            except asyncio.TimeoutError:
                last_error = (
                    f"timeout after {self._timeout_seconds}s"
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

            if attempt < self._max_attempts:
                # Exponential backoff: 1s, 2s, 4s, ...
                backoff = self._base_backoff_seconds * (2 ** (attempt - 1))
                await self._sleep(backoff)

        result = DeliveryResult(
            success=False,
            status_code=last_status,
            attempts=self._max_attempts,
            error=last_error,
        )
        self._record(event, url, result)
        return result

    def _record(
        self, event: str, url: str, result: DeliveryResult,
    ) -> None:
        """Append the dispatch outcome to the log ring if wired."""
        if self._log_ring is None:
            return
        try:
            self._log_ring.append(
                event=event,
                url=url,
                success=result.success,
                attempts=result.attempts,
                status_code=result.status_code,
                error=result.error,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "WebhookDeliverer: log ring append raised: %s",
                exc,
            )

    async def _aiohttp_post(
        self, url: str, body: bytes, headers: Dict[str, str],
    ):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, data=body, headers=headers,
            ) as resp:
                resp_body = await resp.text()
                return resp.status, resp_body
