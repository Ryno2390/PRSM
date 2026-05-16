"""Sprint 269 — operator-side ContentSelfFilter state store.

Wraps the (frozen, immutable) ``ContentSelfFilter`` value type
with a mutable store so operators can edit their blocklist via
HTTP at runtime instead of restarting the node with a new
config. Disk persistence opt-in via ``PRSM_CONTENT_FILTER_DIR``
env var (mirrors the JobHistoryStore + ReceiptStore design).

Per R9-SCOPING-1 §7-8: this is the OPERATOR's own filter, not a
Foundation-curated blocklist. Each operator manages their own
list based on their own legal/compliance analysis. The store
is local-only — never gossiped, never propagated.

Threat-model framing
--------------------

Vision §14 ("Content moderation") flags Foundation-operated
takedown + per-operator opt-in filtering as core mitigations
against legal exposure from unmoderated content. This store
implements the *per-operator* half. The Foundation's takedown
intake is a separate sprint.

Filter state shape (on disk)
----------------------------

``{filter_dir}/filter_state.json``::

    {
      "blocked_content_ids": ["bafy...", "Qm..."],
      "blocked_model_tags":   ["safety-flagged"],
      "blocked_input_patterns": ["^secret-pattern.*"],
      "action_on_match": "refuse",
      "updated_at": 1715000000.0,
      "version": 1
    }

Atomic writes via tmp+rename. Reads tolerate missing /
malformed file (treated as empty filter, defensive default).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from threading import Lock
from typing import Iterable, List, Mapping, Optional

from prsm.node.content_self_filter import (
    ContentSelfFilter, FilterAction,
)

logger = logging.getLogger(__name__)


_VALID_ACTIONS = {a.value for a in FilterAction}


class ContentFilterStore:
    """Mutable store backed by an immutable ContentSelfFilter."""

    def __init__(
        self,
        *,
        persist_dir: Optional[Path] = None,
        initial_cids: Optional[Iterable[str]] = None,
        initial_tags: Optional[Iterable[str]] = None,
        initial_patterns: Optional[Iterable[str]] = None,
        action_on_match: str = "refuse",
    ) -> None:
        self._lock = Lock()
        self._persist_dir = persist_dir
        if persist_dir is not None:
            persist_dir.mkdir(parents=True, exist_ok=True)
        self._cids: set = set(initial_cids or ())
        self._tags: set = set(initial_tags or ())
        self._patterns: List[str] = list(initial_patterns or ())
        self._action = (
            action_on_match if action_on_match in _VALID_ACTIONS
            else "refuse"
        )
        self._updated_at: float = time.time()
        if persist_dir is not None:
            self._load_from_disk()

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None,
    ) -> "ContentFilterStore":
        env = env if env is not None else os.environ
        raw = (env.get("PRSM_CONTENT_FILTER_DIR") or "").strip()
        if not raw:
            return cls()
        return cls(persist_dir=Path(raw))

    # ── Disk I/O ──────────────────────────────────────────

    def _state_path(self) -> Path:
        assert self._persist_dir is not None
        return self._persist_dir / "filter_state.json"

    def _load_from_disk(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "ContentFilterStore: skipping malformed %s (%s); "
                "starting from empty filter", path, exc,
            )
            return
        if not isinstance(data, dict):
            logger.warning(
                "ContentFilterStore: %s wrong shape; empty filter",
                path,
            )
            return
        cids = data.get("blocked_content_ids") or []
        tags = data.get("blocked_model_tags") or []
        patterns = data.get("blocked_input_patterns") or []
        action = data.get("action_on_match") or "refuse"
        if (
            isinstance(cids, list)
            and isinstance(tags, list)
            and isinstance(patterns, list)
        ):
            self._cids = {c for c in cids if isinstance(c, str)}
            self._tags = {t for t in tags if isinstance(t, str)}
            self._patterns = [
                p for p in patterns if isinstance(p, str)
            ]
            if action in _VALID_ACTIONS:
                self._action = action
            updated = data.get("updated_at")
            if isinstance(updated, (int, float)):
                self._updated_at = float(updated)

    def _write_to_disk(self) -> None:
        if self._persist_dir is None:
            return
        path = self._state_path()
        tmp = path.with_suffix(".json.tmp")
        body = json.dumps({
            "blocked_content_ids": sorted(self._cids),
            "blocked_model_tags": sorted(self._tags),
            "blocked_input_patterns": list(self._patterns),
            "action_on_match": self._action,
            "updated_at": self._updated_at,
            "version": 1,
        }, indent=2)
        try:
            tmp.write_text(body, encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:  # noqa: BLE001
            logger.warning(
                "ContentFilterStore: disk write failed: %s", exc,
            )

    # ── State accessors ───────────────────────────────────

    def current(self) -> ContentSelfFilter:
        """Snapshot the current filter as an immutable
        ContentSelfFilter. Callers evaluate this against
        DispatchContext."""
        with self._lock:
            return ContentSelfFilter(
                blocked_content_ids=frozenset(self._cids),
                blocked_model_tags=frozenset(self._tags),
                blocked_input_patterns=[
                    re.compile(p) for p in self._patterns
                ],
                action_on_match=FilterAction(self._action),
            )

    def is_cid_blocked(self, cid: str) -> bool:
        # Sprint 492 (F30 fix) — case-insensitive CID
        # matching. PRSM CIDs are lowercase-hex (SHA-1 BT
        # infohash, SHA-256 content_hash). Pre-fix, an
        # operator who blocked `ABC123` (uppercase) didn't
        # block requests for `abc123` (lowercase) and vice
        # versa — trivial case-evasion attack on the §14
        # content moderation surface. The `add_tags` path
        # at line 240 already normalizes via `.lower()`;
        # CID handling now matches.
        if not isinstance(cid, str):
            return False
        with self._lock:
            return cid.lower() in self._cids

    def count(self) -> int:
        """Total number of filter entries across all categories.
        Used by /health/detailed's subsystem-record-count probe
        (sprint 343). Without this, the probe raises
        AttributeError and the health endpoint flips
        `content_filter_store.status` to `error` + the whole
        daemon to `degraded` — false-positive alert noise.
        Fixed in sprint 473 (F21)."""
        with self._lock:
            return (
                len(self._cids)
                + len(self._tags)
                + len(self._patterns)
            )

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "blocked_content_ids": sorted(self._cids),
                "blocked_model_tags": sorted(self._tags),
                "blocked_input_patterns": list(self._patterns),
                "action_on_match": self._action,
                "updated_at": self._updated_at,
                "count_cids": len(self._cids),
                "count_tags": len(self._tags),
                "count_patterns": len(self._patterns),
            }

    # ── Mutations ─────────────────────────────────────────

    def add_cids(self, cids: Iterable[str]) -> int:
        """Add CIDs to the blocklist. Returns number newly added
        (idempotent — adding existing CIDs is a no-op).

        Sprint 492 (F30 fix) — CIDs are lowercase-normalized
        before storage so the case-insensitive lookup in
        is_cid_blocked finds them regardless of how the
        request URL was cased."""
        clean = [
            c.strip().lower()
            for c in cids
            if isinstance(c, str) and c.strip()
        ]
        added = 0
        with self._lock:
            for c in clean:
                if c not in self._cids:
                    self._cids.add(c)
                    added += 1
            if added:
                self._updated_at = time.time()
                self._write_to_disk()
        return added

    def remove_cid(self, cid: str) -> bool:
        """Remove a CID. Returns True if removed, False if not present.

        Sprint 492 (F30 fix) — lowercase-normalize at removal
        time to match add/check paths."""
        if not isinstance(cid, str):
            return False
        normalized = cid.strip().lower()
        with self._lock:
            if normalized in self._cids:
                self._cids.remove(normalized)
                self._updated_at = time.time()
                self._write_to_disk()
                return True
            return False

    def add_tags(self, tags: Iterable[str]) -> int:
        # Sprint 492 (F31 fix) — strip ALL non-printable
        # ASCII characters before storage. Pre-fix, an
        # operator (or compromised admin client) could store
        # tags with embedded \r\n / NUL / control chars, which
        # then corrupted log output ("log injection") and
        # CLI rendering. The lowercase + .strip() pre-fix
        # only handled leading/trailing whitespace.
        def _sanitize(t: str) -> str:
            # Keep only printable ASCII (0x20-0x7E), lower-case.
            return "".join(
                ch for ch in t.strip().lower()
                if 0x20 <= ord(ch) <= 0x7E
            )
        clean = [
            _sanitize(t) for t in tags
            if isinstance(t, str) and t.strip()
        ]
        clean = [t for t in clean if t]  # drop empties post-sanitize
        added = 0
        with self._lock:
            for t in clean:
                if t not in self._tags:
                    self._tags.add(t)
                    added += 1
            if added:
                self._updated_at = time.time()
                self._write_to_disk()
        return added

    def remove_tag(self, tag: str) -> bool:
        with self._lock:
            tag_l = tag.strip().lower()
            if tag_l in self._tags:
                self._tags.remove(tag_l)
                self._updated_at = time.time()
                self._write_to_disk()
                return True
            return False

    def add_patterns(self, patterns: Iterable[str]) -> int:
        """Add regex patterns. Invalid regex strings are rejected
        (raises ValueError naming the offending pattern)."""
        clean: List[str] = []
        for p in patterns:
            if not isinstance(p, str) or not p:
                continue
            try:
                re.compile(p)
            except re.error as e:
                raise ValueError(
                    f"invalid regex {p!r}: {e}"
                )
            clean.append(p)
        added = 0
        with self._lock:
            for p in clean:
                if p not in self._patterns:
                    self._patterns.append(p)
                    added += 1
            if added:
                self._updated_at = time.time()
                self._write_to_disk()
        return added

    def remove_pattern(self, pattern: str) -> bool:
        with self._lock:
            if pattern in self._patterns:
                self._patterns.remove(pattern)
                self._updated_at = time.time()
                self._write_to_disk()
                return True
            return False

    def set_action(self, action: str) -> None:
        if action not in _VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(_VALID_ACTIONS)}; "
                f"got {action!r}"
            )
        with self._lock:
            self._action = action
            self._updated_at = time.time()
            self._write_to_disk()
