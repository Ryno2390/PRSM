"""Sprint 284 — webhook replay protection.

Sprint 283 ships HMAC signature verification: only payloads
signed by the vendor's secret pass. But a signed payload stays
signed forever — an attacker who captures one (via a leak, a
MITM that doesn't yield the secret, server log scrape, etc.)
can replay it indefinitely.

Two defenses ship together:

  1. Timestamp window. Persona embeds the signing timestamp
     in its header (``t=<unix>``). If the timestamp is more
     than ``tolerance_sec`` away from "now" (in either
     direction), reject. Default 300s matches Stripe /
     Coinbase Commerce convention.

  2. Signature-hash dedup ring. Vendor-agnostic. The full
     signature value is a perfect replay token — it's
     cryptographically unique per (body, timestamp, secret).
     Bounded in-memory ring tracks recently-seen signatures;
     second occurrence → reject.

Both work for Persona; the dedup ring is the primary defense
for Onfido (whose signature carries no timestamp). Apply both
when available for belt-and-suspenders.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


_DEFAULT_TOLERANCE_SEC = 300  # 5 minutes
_DEFAULT_RING_SIZE = 10_000

# Sp893 — how long a seen token stays re-rejectable ACROSS a restart.
# Bounds on-disk growth by time (the sp887 unbounded-disk concern):
# tokens older than this are dropped when the ring is reloaded. 24h
# is far beyond the 300s Persona freshness window and gives Onfido
# (no vendor timestamp → the ring is its ONLY replay defense) a long
# replay-rejection horizon. Operators can raise it via
# PRSM_KYC_WEBHOOK_REPLAY_RETENTION_SEC for stricter Onfido posture.
_DEFAULT_RETENTION_SEC = 86_400  # 24 hours

_PERSIST_FILENAME = "replay-ring.json"


def is_timestamp_fresh(
    ts_str: str,
    current_time: float,
    tolerance_sec: int = _DEFAULT_TOLERANCE_SEC,
) -> Tuple[bool, str]:
    """Check whether a vendor-supplied timestamp string is
    within the freshness window centered on ``current_time``.

    Returns (ok, reason). reason is "" on success; on failure
    it's a short human-readable explanation for logging
    (NOT for return to the vendor — info leak protection).

    Both stale (too old) and future-skewed timestamps are
    rejected. Symmetric tolerance prevents an attacker from
    exploiting clock skew in either direction.
    """
    if not ts_str:
        return (False, "empty timestamp")
    try:
        ts = float(ts_str)
    except (ValueError, TypeError):
        return (
            False,
            f"could not parse timestamp {ts_str!r}",
        )
    delta = abs(current_time - ts)
    if delta > tolerance_sec:
        return (
            False,
            f"timestamp out of window: "
            f"|now - t| = {delta:.1f}s > "
            f"tolerance {tolerance_sec}s",
        )
    return (True, "")


class WebhookReplayRing:
    """Bounded set of recently-seen webhook replay tokens.

    Eviction is FIFO (oldest seen first). The ring size
    governs how long a captured signature remains
    re-rejectable — for typical webhook volumes (≤ thousands
    per hour) 10K entries spans hours of history, well
    beyond the typical 300s timestamp window.

    Sp893 — optional persistence. With ``persist_dir`` set, each
    seen token is stored on disk (with its first-seen timestamp) so
    the ring survives a daemon RESTART. This closes a real gap:
    Onfido signatures carry no timestamp, so the ring is Onfido's
    ONLY replay defense, and an in-memory-only ring let a captured
    Onfido webhook be replayed across the next restart (deploys,
    crashes, OOM-kills happen routinely). Disk is bounded BOTH by
    the FIFO count cap AND by ``retention_sec`` (expired tokens are
    dropped on reload), addressing the sp887 unbounded-disk concern.
    ``persist_dir=None`` (default) keeps the pure in-memory behavior.

    Thread-safe via a single mutex.
    """

    def __init__(
        self,
        max_entries: int = _DEFAULT_RING_SIZE,
        *,
        persist_dir: Optional[object] = None,
        retention_sec: Optional[int] = None,
        now: Optional[float] = None,
    ) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._retention_sec = (
            retention_sec
            if retention_sec is not None
            else _DEFAULT_RETENTION_SEC
        )
        self._order: Deque[str] = deque(maxlen=max_entries)
        self._set: Set[str] = set()
        # token -> first-seen unix timestamp (for time-based expiry).
        self._ts: Dict[str, float] = {}
        self._lock = threading.Lock()

        self._persist_path: Optional[Path] = None
        if persist_dir is not None:
            pdir = Path(persist_dir)
            try:
                pdir.mkdir(parents=True, exist_ok=True)
                self._persist_path = pdir / _PERSIST_FILENAME
                self._load(now if now is not None else time.time())
            except OSError as exc:
                logger.warning(
                    "WebhookReplayRing: persist dir %s unusable "
                    "(%s) — falling back to in-memory only.",
                    pdir, exc,
                )
                self._persist_path = None

    def _load(self, now: float) -> None:
        """Reload the ring from disk, dropping tokens older than the
        retention window and keeping at most ``max_entries`` (FIFO).
        Fail-soft: a corrupt/unreadable file starts the ring empty
        (matches sp857 OnrampFunnel)."""
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text())
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning(
                "WebhookReplayRing: bad persist file %s (%s) — "
                "starting empty.",
                self._persist_path, exc,
            )
            return
        if not isinstance(raw, list):
            logger.warning(
                "WebhookReplayRing: persist file %s not a list — "
                "starting empty.",
                self._persist_path,
            )
            return
        # Keep only well-formed, unexpired pairs, newest-last.
        fresh: List[Tuple[str, float]] = []
        for item in raw:
            try:
                token, ts = item[0], float(item[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if not isinstance(token, str) or not token:
                continue
            if now - ts > self._retention_sec:
                continue  # expired — drop (bounds disk by time)
            fresh.append((token, ts))
        # Enforce the FIFO count cap on the surviving set.
        if len(fresh) > self._max_entries:
            fresh = fresh[-self._max_entries:]
        for token, ts in fresh:
            self._order.append(token)
            self._set.add(token)
            self._ts[token] = ts

    def _persist_locked(self) -> None:
        """Atomically write the current ring to disk. Caller holds
        the lock. Fail-soft: a write error logs and is swallowed so
        a transient disk problem never drops a live webhook."""
        if self._persist_path is None:
            return
        payload = [[tok, self._ts.get(tok, 0.0)] for tok in self._order]
        tmp = self._persist_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, self._persist_path)
        except OSError as exc:
            logger.warning(
                "WebhookReplayRing: persist write failed (%s) — "
                "replay defense degraded to in-memory for this "
                "process.",
                exc,
            )

    def seen(self, token: str) -> bool:
        """Returns True if the token has been recorded
        within the retention window. Does NOT mutate."""
        with self._lock:
            return token in self._set

    def record(self, token: str, now: Optional[float] = None) -> bool:
        """Record a token. Returns True on first occurrence,
        False on duplicate. Caller treats False as "replay
        detected — reject this webhook"."""
        if not isinstance(token, str) or not token:
            raise ValueError(
                "token must be a non-empty string"
            )
        ts = now if now is not None else time.time()
        with self._lock:
            if token in self._set:
                return False
            # If the deque is at max capacity, the oldest
            # token will be evicted by the append — also
            # purge it from the lookup set + timestamp map.
            if len(self._order) >= self._max_entries:
                evicted = self._order[0]
                self._set.discard(evicted)
                self._ts.pop(evicted, None)
            self._order.append(token)
            self._set.add(token)
            self._ts[token] = ts
            self._persist_locked()
            return True

    def count(self) -> int:
        with self._lock:
            return len(self._set)
