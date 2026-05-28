"""Sprint 874 — onramp completion outbound webhook notifier.

When sp857's funnel sweep CONFIRMS an intent + sp871's orchestrator
attaches the swap envelope, this notifier POSTs the completion
event to an operator-configured webhook URL. Downstream systems
(notification services, mobile push pipelines, ledger replicators)
can subscribe to "user's USDC arrived + swap ready" without
polling.

Config (env):
  PRSM_ONRAMP_COMPLETION_WEBHOOK_URL
    Outbound URL. When unset, notifier is a no-op (operator hasn't
    wired it; CONFIRMED transition still happens via sp871 — just
    no outbound side-effect).
  PRSM_ONRAMP_COMPLETION_WEBHOOK_SECRET
    Optional HMAC-SHA256 secret. When set, every outbound POST
    carries `X-PRSM-Signature: t=<unix>,v1=<hex hmac>` over
    `<timestamp>.<body>` (Persona's sp283 format — mirrors the
    well-understood inbound pattern so consumers reuse code).
  PRSM_ONRAMP_COMPLETION_WEBHOOK_LOG_DIR
    Persistent delivery log (one JSON per dispatch attempt). Default
    `~/.prsm/onramp-completion-deliveries/`. Opt out via `:memory:`.

Payload:
  {
    "event": "onramp.completion",
    "intent_id": "...",
    "user_id": "...",
    "destination_address": "0x...",
    "expected_usd": 5.0,
    "usdc_received": 4.92,
    "confirmed_at": 1779993XXX,
    "swap_envelope": {...},  // null if pool not configured yet
  }

Dispatch is fail-soft + best-effort:
  - HTTP errors: log + record failure entry; don't propagate
    upward (sp871's CONFIRMED transition must stand even when
    webhook delivery fails)
  - Timeouts: same — webhook receiver might be slow; downstream
    integrators are expected to be idempotent

Delivery log entries:
  {timestamp, intent_id, url, status_code, success, error,
   signature_attached, attempt_count}
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT_SECONDS = 15.0


@dataclass
class DeliveryRecord:
    timestamp: float
    intent_id: str
    url: str
    status_code: int  # 0 on transport error
    success: bool
    error: Optional[str] = None
    signature_attached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _signature_header(
    secret: str, body_bytes: bytes,
) -> tuple[str, str]:
    """Build the `X-PRSM-Signature: t=<unix>,v1=<hex>` header value.

    Format mirrors Persona's t=<ts>,v1=<hmac> pattern (sp283) so
    downstream consumers reuse the same verification code path.
    """
    ts = str(int(time.time()))
    signed = f"{ts}.".encode("utf-8") + body_bytes
    sig = hmac.new(
        secret.encode("utf-8"), signed, hashlib.sha256,
    ).hexdigest()
    return "X-PRSM-Signature", f"t={ts},v1={sig}"


class OnrampCompletionNotifier:
    """Outbound webhook dispatcher for onramp completion events."""

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        secret: Optional[str] = None,
        log_dir: Optional[Path] = None,
        client: Any = None,
    ) -> None:
        self._url = url
        self._secret = secret
        if log_dir is None:
            raw = os.environ.get(
                "PRSM_ONRAMP_COMPLETION_WEBHOOK_LOG_DIR",
            )
            if raw == ":memory:":
                self._log_dir = None
            elif raw:
                self._log_dir = Path(raw)
            else:
                self._log_dir = (
                    Path.home()
                    / ".prsm" / "onramp-completion-deliveries"
                )
        else:
            self._log_dir = log_dir
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        if client is None:
            import httpx
            self._client = httpx.Client(
                timeout=_DEFAULT_TIMEOUT_SECONDS,
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def is_configured(self) -> bool:
        return bool(self._url)

    def _persist(self, record: DeliveryRecord) -> None:
        if self._log_dir is None:
            return
        fname = (
            f"{int(record.timestamp * 1000)}-"
            f"{record.intent_id}.json"
        )
        path = self._log_dir / fname
        try:
            path.write_text(
                json.dumps(record.to_dict(), indent=2),
            )
        except OSError as exc:
            logger.warning(
                "OnrampCompletionNotifier: persist failed: %s",
                exc,
            )

    def list_deliveries(
        self, *, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return persisted dispatch records, newest first."""
        if self._log_dir is None:
            return []
        records: List[Dict[str, Any]] = []
        for path in sorted(
            self._log_dir.glob("*.json"), reverse=True,
        )[:limit]:
            try:
                records.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return records

    def notify(
        self, *, intent: Any,
    ) -> Optional[DeliveryRecord]:
        """Fire the outbound POST for a CONFIRMED intent.

        Returns the DeliveryRecord (whether success or failure) when
        dispatch was attempted; None when not configured (no-op).
        """
        if not self._url:
            return None
        payload = {
            "event": "onramp.completion",
            "intent_id": intent.intent_id,
            "user_id": intent.user_id,
            "destination_address": intent.destination_address,
            "expected_usd": intent.expected_usd,
            "usdc_received": intent.usdc_received,
            "confirmed_at": intent.confirmed_at,
            "swap_envelope": (
                intent.swap_envelope
                if getattr(intent, "swap_envelope", None) is not None
                else None
            ),
        }
        body_bytes = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "PRSM-OnrampNotifier/sp874",
        }
        signature_attached = False
        if self._secret:
            h_name, h_val = _signature_header(
                self._secret, body_bytes,
            )
            headers[h_name] = h_val
            signature_attached = True

        record_ts = time.time()
        try:
            resp = self._client.post(
                self._url, content=body_bytes, headers=headers,
            )
            record = DeliveryRecord(
                timestamp=record_ts,
                intent_id=intent.intent_id,
                url=self._url,
                status_code=resp.status_code,
                success=(200 <= resp.status_code < 300),
                error=(
                    None if (200 <= resp.status_code < 300)
                    else resp.text[:200]
                ),
                signature_attached=signature_attached,
            )
        except Exception as exc:  # noqa: BLE001
            record = DeliveryRecord(
                timestamp=record_ts,
                intent_id=intent.intent_id,
                url=self._url,
                status_code=0,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                signature_attached=signature_attached,
            )
            logger.warning(
                "OnrampCompletionNotifier: POST failed for "
                "intent %s: %s",
                intent.intent_id, exc,
            )
        self._persist(record)
        return record


def from_env(
    *, client: Any = None,
) -> "OnrampCompletionNotifier":
    """Build from env. Always succeeds — when URL is unset, the
    notifier is a no-op (notify() returns None silently)."""
    url = (
        os.environ.get("PRSM_ONRAMP_COMPLETION_WEBHOOK_URL")
        or None
    )
    secret = (
        os.environ.get("PRSM_ONRAMP_COMPLETION_WEBHOOK_SECRET")
        or None
    )
    return OnrampCompletionNotifier(
        url=url, secret=secret, client=client,
    )
