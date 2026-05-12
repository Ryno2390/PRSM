"""Sprint 291 — content fingerprint registry.

Vision §14 mitigation item (3) cryptographic deduplication.
The spam attack: a creator downloads someone else's content
and re-uploads it under their own address to claim royalties.
Defense: SHA-256 content fingerprint with first-creator-wins
semantics. Subsequent uploads of the same fingerprint are
recognized as duplicates and the canonical (first) creator's
address is returned regardless of the would-be claimer.

v1 is per-node, in-memory with opt-in filesystem persistence
via PRSM_FINGERPRINT_REGISTRY_DIR. Federated gossip of
fingerprints is a follow-on (operators eventually need to
agree on global first-uploader status; for now each operator
builds their own per-node view, matching the tracker contract
in sprints 287-290).

API:
  - register(content_hash, creator_eth_address, timestamp?)
    → (canonical_creator, is_new)
    is_new=True means the fingerprint was previously unknown
    (caller is the canonical creator). False means the
    fingerprint was already registered (caller is either the
    canonical creator re-claiming OR a duplicate-attempt).
    Either way, canonical_creator is the authoritative answer.
  - canonical_creator(content_hash) → Optional[str]
  - is_duplicate(content_hash, creator_eth_address) → bool
    True iff a different creator already claimed this hash.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ContentFingerprintEntry:
    content_hash: str
    canonical_creator: str
    first_seen_unix: int
    duplicate_attempt_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "canonical_creator": self.canonical_creator,
            "first_seen_unix": self.first_seen_unix,
            "duplicate_attempt_count": (
                self.duplicate_attempt_count
            ),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "ContentFingerprintEntry":
        return cls(
            content_hash=d["content_hash"],
            canonical_creator=d["canonical_creator"],
            first_seen_unix=int(d.get("first_seen_unix", 0)),
            duplicate_attempt_count=int(
                d.get("duplicate_attempt_count", 0),
            ),
        )


class ContentFingerprintRegistry:
    """In-memory fingerprint → creator ledger with optional
    disk persistence."""

    def __init__(
        self,
        *,
        persist_dir: Optional[Path] = None,
    ) -> None:
        self._entries: Dict[str, ContentFingerprintEntry] = {}
        self._persist_dir: Optional[Path] = (
            Path(persist_dir) if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    @classmethod
    def from_env(cls) -> "ContentFingerprintRegistry":
        raw = os.environ.get("PRSM_FINGERPRINT_REGISTRY_DIR")
        persist_dir = Path(raw) if raw else None
        return cls(persist_dir=persist_dir)

    # ── Mutation ─────────────────────────────────────────

    def register(
        self,
        content_hash: str,
        creator_eth_address: str,
        *,
        timestamp: Optional[float] = None,
    ) -> Tuple[str, bool]:
        if not content_hash:
            raise ValueError("content_hash must be non-empty")
        if not creator_eth_address:
            raise ValueError(
                "creator_eth_address must be non-empty"
            )
        now_int = int(
            timestamp if timestamp is not None else time.time()
        )
        existing = self._entries.get(content_hash)
        if existing is None:
            entry = ContentFingerprintEntry(
                content_hash=content_hash,
                canonical_creator=creator_eth_address,
                first_seen_unix=now_int,
                duplicate_attempt_count=0,
            )
            self._entries[content_hash] = entry
            self._write_to_disk(entry)
            return (creator_eth_address, True)
        # Existing: re-claim by same creator is a no-op; by
        # different creator counts as a duplicate attempt.
        if existing.canonical_creator != creator_eth_address:
            existing.duplicate_attempt_count += 1
            self._write_to_disk(existing)
        return (existing.canonical_creator, False)

    # ── Queries ──────────────────────────────────────────

    def canonical_creator(
        self, content_hash: str,
    ) -> Optional[str]:
        e = self._entries.get(content_hash)
        return e.canonical_creator if e else None

    def is_duplicate(
        self,
        content_hash: str,
        creator_eth_address: str,
    ) -> bool:
        e = self._entries.get(content_hash)
        if e is None:
            return False
        return e.canonical_creator != creator_eth_address

    def get_entry(
        self, content_hash: str,
    ) -> Optional[ContentFingerprintEntry]:
        return self._entries.get(content_hash)

    def count(self) -> int:
        return len(self._entries)

    def recent(
        self, *, limit: int = 50,
    ) -> List[ContentFingerprintEntry]:
        if not isinstance(limit, int) or limit <= 0 or limit > 10000:
            raise ValueError(
                f"limit must be in [1, 10000], got {limit}"
            )
        entries = list(self._entries.values())
        entries.sort(
            key=lambda e: e.first_seen_unix, reverse=True,
        )
        return entries[:limit]

    # ── Persistence ──────────────────────────────────────

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        for path in self._persist_dir.glob("*.json"):
            try:
                d = json.loads(path.read_text())
                entry = ContentFingerprintEntry.from_dict(d)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ContentFingerprintRegistry: skipping "
                    "corrupt %s: %s",
                    path, exc,
                )
                continue
            self._entries[entry.content_hash] = entry

    def _write_to_disk(
        self, entry: ContentFingerprintEntry,
    ) -> None:
        if self._persist_dir is None:
            return
        # Path-traversal-proof filename: hash inputs may
        # contain slashes (e.g., "sha256-/x") in adversarial
        # cases. Sanitize.
        safe = (
            entry.content_hash
            .replace("/", "_")
            .replace("\\", "_")
        )
        path = self._persist_dir / f"{safe}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(entry.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ContentFingerprintRegistry: disk write "
                "failed for %s: %s",
                entry.content_hash, exc,
            )
