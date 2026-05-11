"""Sprint 272 — TakedownNoticeRing for received DMCA / legal /
content moderation notices.

Vision §14 ("Content moderation") flags this as a core mitigation:
"Foundation operates a takedown process for DMCA and similar
legal notices." Combined with R9-SCOPING-1 §8's invariant ("No
Foundation-shipped curated blocklists"), the resolution is that
the Foundation INTAKES notices (logs them, distributes them as
information), but operators decide voluntarily — via their own
sprint-269 ContentFilterStore — whether to act on a given
notice.

This ring is the intake side. It:
  - Captures each received notice with structured fields
    (sender, jurisdiction, basis citation, target CID).
  - Surfaces notices via /admin/takedown-notices for operators
    to inspect.
  - Does NOT modify any operator's filter. Each operator runs
    their own compliance analysis.

Storage shape mirrors HeartbeatRecordedRing + RoyaltyDispatchRing
(in-memory deque + opt-in filesystem persistence via
``PRSM_TAKEDOWN_NOTICE_LOG_DIR``).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional

logger = logging.getLogger(__name__)


_VALID_STATUSES = {"received", "acknowledged", "disputed", "expired"}


@dataclass(frozen=True)
class TakedownNoticeEntry:
    notice_id: str  # UUID4
    timestamp: float
    target_cid: str
    sender: str  # email / org name / legal entity
    jurisdiction: str  # e.g. "US-DMCA", "EU-DSA", "UK-OSA"
    basis: str  # short citation (e.g. "DMCA §512(c)")
    notice_text: str  # full notice body (capped per ring)
    status: str = "received"

    def to_dict(self) -> dict:
        return {
            "notice_id": self.notice_id,
            "timestamp": self.timestamp,
            "target_cid": self.target_cid,
            "sender": self.sender,
            "jurisdiction": self.jurisdiction,
            "basis": self.basis,
            "notice_text": self.notice_text,
            "status": self.status,
        }


_DEFAULT_MAX_ENTRIES = 1024
_MAX_NOTICE_TEXT_LEN = 8192  # 8KB per notice body cap


class TakedownNoticeRing:
    """Bounded in-memory ring of takedown notice records."""

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        *,
        persist_dir: Optional[Path] = None,
    ) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._entries: Deque[TakedownNoticeEntry] = deque(
            maxlen=max_entries,
        )
        # Secondary index for O(1) lookup by notice_id.
        self._by_id: dict = {}
        self._persist_dir: Optional[Path] = (
            Path(persist_dir) if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        loaded: list = []
        for path in self._persist_dir.glob("*.json"):
            try:
                d = json.loads(path.read_text())
                loaded.append(d)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "TakedownNoticeRing: skipping corrupt %s: %s",
                    path, exc,
                )
        loaded.sort(key=lambda d: d.get("timestamp", 0))
        for d in loaded:
            try:
                entry = TakedownNoticeEntry(
                    notice_id=d["notice_id"],
                    timestamp=d["timestamp"],
                    target_cid=d["target_cid"],
                    sender=d["sender"],
                    jurisdiction=d["jurisdiction"],
                    basis=d["basis"],
                    notice_text=d.get("notice_text", ""),
                    status=d.get("status", "received"),
                )
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "TakedownNoticeRing: malformed entry %r: %s",
                    d, exc,
                )
                continue
            self._entries.append(entry)
            self._by_id[entry.notice_id] = entry

    def _write_to_disk(self, entry: TakedownNoticeEntry) -> None:
        if self._persist_dir is None:
            return
        path = self._persist_dir / f"{entry.notice_id}.json"
        try:
            path.write_text(json.dumps(entry.to_dict()))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TakedownNoticeRing: disk write failed: %s", exc,
            )

    def record(
        self,
        *,
        target_cid: str,
        sender: str,
        jurisdiction: str,
        basis: str,
        notice_text: str = "",
        timestamp: Optional[float] = None,
    ) -> TakedownNoticeEntry:
        """Append a new received notice. Caller-supplied fields are
        validated minimally; the FOUNDATION is the authoritative
        recorder, so heavy validation is intentionally out-of-scope.
        """
        if not target_cid or not isinstance(target_cid, str):
            raise ValueError("target_cid must be a non-empty string")
        if not sender or not isinstance(sender, str):
            raise ValueError("sender must be a non-empty string")
        if not jurisdiction or not isinstance(jurisdiction, str):
            raise ValueError("jurisdiction must be a non-empty string")
        if not basis or not isinstance(basis, str):
            raise ValueError("basis must be a non-empty string")
        text = (notice_text or "")[:_MAX_NOTICE_TEXT_LEN]
        entry = TakedownNoticeEntry(
            notice_id=str(uuid.uuid4()),
            timestamp=(
                timestamp if timestamp is not None else time.time()
            ),
            target_cid=target_cid,
            sender=sender,
            jurisdiction=jurisdiction,
            basis=basis,
            notice_text=text,
        )
        self._entries.append(entry)
        self._by_id[entry.notice_id] = entry
        self._write_to_disk(entry)
        return entry

    def get(self, notice_id: str) -> Optional[TakedownNoticeEntry]:
        return self._by_id.get(notice_id)

    def recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        target_cid: Optional[str] = None,
    ) -> List[TakedownNoticeEntry]:
        if not isinstance(limit, int) or limit <= 0 or limit > 1000:
            raise ValueError(
                f"limit must be in [1, 1000], got {limit}"
            )
        if not isinstance(offset, int) or offset < 0:
            raise ValueError(
                f"offset must be >= 0, got {offset}"
            )
        snap = list(self._entries)
        snap.reverse()
        if status:
            if status not in _VALID_STATUSES:
                raise ValueError(
                    f"status must be one of {sorted(_VALID_STATUSES)}, "
                    f"got {status!r}"
                )
            snap = [e for e in snap if e.status == status]
        if target_cid:
            snap = [
                e for e in snap if e.target_cid == target_cid
            ]
        return snap[offset:offset + limit]

    def count(self) -> int:
        return len(self._entries)
