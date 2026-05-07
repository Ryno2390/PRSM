"""
PRSM-PROV-1 Item 4 T4.6 — FingerprintIndex.

Per-kind binary-fingerprint dedup index. Sits alongside
``_SemanticIndex`` (which owns the text-vector embedding lane); this
class owns the four binary lanes (image-pHash, audio-Chromaprint,
video-multihash, structural). Each lane is a dict keyed by content_id
(CID) and stores the backend-specific fingerprint payload alongside
the creator_id.

A single ``find_nearest`` call routes to the backend matching the
record's ``kind`` and runs that backend's ``similarity()`` function
against every stored record of the same kind. O(n) per kind — fine
for early-network scale; same scaling profile as ``_SemanticIndex``.

Persistence: optional JSON file. Keeps the index across node restarts
the way ``_SemanticIndex`` does for the text-vector lane.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from prsm.data.fingerprints.base import (
    BinaryFingerprint,
    FingerprintKind,
    FingerprintRecord,
)

logger = logging.getLogger(__name__)


# Default per-kind dedup thresholds. Mirrors
# ``prsm/data/dedup_thresholds.yaml::defaults`` so the index can run
# without the YAML being loaded — Item 6 (ThresholdResolver) plugs in
# when callers want the configurable / per-content-type-aware values.
_DEFAULT_DUPLICATE_THRESHOLDS: Dict[FingerprintKind, float] = {
    FingerprintKind.IMAGE_PHASH: 0.906,        # ≤ 6/64 hamming
    FingerprintKind.AUDIO_CHROMAPRINT: 0.92,
    FingerprintKind.VIDEO_MULTIHASH: 0.875,    # ≥ 7/8 keyframes
    FingerprintKind.STRUCTURAL: 1.0,           # exact-match only
}

_DEFAULT_DERIVATIVE_THRESHOLDS: Dict[FingerprintKind, float] = {
    FingerprintKind.IMAGE_PHASH: 0.906,        # same — pHash has one tier
    FingerprintKind.AUDIO_CHROMAPRINT: 0.75,
    FingerprintKind.VIDEO_MULTIHASH: 0.625,    # ≥ 5/8 keyframes
    FingerprintKind.STRUCTURAL: 1.0,
}


class FingerprintMatch:
    """Result of a successful nearest-fingerprint lookup."""

    __slots__ = ("content_id", "similarity", "creator_id", "kind")

    def __init__(
        self,
        content_id: str,
        similarity: float,
        creator_id: str,
        kind: FingerprintKind,
    ) -> None:
        self.content_id = content_id
        self.similarity = similarity
        self.creator_id = creator_id
        self.kind = kind

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"FingerprintMatch(kind={self.kind.value}, "
            f"content_id={self.content_id[:16]!r}..., "
            f"similarity={self.similarity:.4f})"
        )


class FingerprintIndex:
    """Per-kind binary-fingerprint dedup index.

    Construct with the set of available backends (typically
    ``ImageFingerprint()``, ``AudioFingerprint()`` etc — whichever
    optional deps are installed). The index dispatches by
    ``FingerprintKind`` so a host without (e.g.) PyAV simply has no
    video lane.

    Backends are kept by reference so the same instance services every
    similarity scan — keeps backend internals (lazy-loaded model
    weights, decoded codec tables, etc.) warm across calls.
    """

    # Default upper bound on remote fingerprint pulls per find_nearest()
    # call. Each pull is a network round-trip; this caps the worst-case
    # blocking time on a sub-threshold local search. Mirrors
    # ``_SemanticIndex.DEFAULT_MAX_REMOTE_PULLS_PER_QUERY``.
    DEFAULT_MAX_REMOTE_PULLS_PER_QUERY: int = 32

    def __init__(
        self,
        backends: Optional[Dict[FingerprintKind, BinaryFingerprint]] = None,
        *,
        persist_path: Optional[Path] = None,
        duplicate_thresholds: Optional[Dict[FingerprintKind, float]] = None,
        derivative_thresholds: Optional[Dict[FingerprintKind, float]] = None,
        dht_client: Optional[Any] = None,
        peer_candidates_fn: Optional[
            Callable[[], Iterable[Tuple[str, str]]]
        ] = None,
        max_remote_pulls_per_query: Optional[int] = None,
    ) -> None:
        """Construct a fingerprint index, optionally wired to the
        EmbeddingDHT for cross-node escalation (PRSM-PROV-1 Item 4
        T4.9.next2 — fingerprint-lane analogue of T3.5).

        Args:
            backends: Map ``FingerprintKind -> BinaryFingerprint``.
            persist_path: Optional JSON path the index persists to.
            duplicate_thresholds / derivative_thresholds: per-kind
                similarity tiers; defaults from
                ``dedup_thresholds.yaml::defaults``.
            dht_client: An ``EmbeddingDHTClient`` (duck-typed: any
                object exposing
                ``find_fingerprint_providers(content_hash, kind) -> list``
                and ``fetch_fingerprint(provider, content_hash, kind)
                -> FingerprintResponse``). ``None`` disables escalation.
            peer_candidates_fn: A callable returning an iterable of
                ``(content_id, content_hash)`` pairs known via gossip
                whose fingerprints could be pulled from the DHT.
                ``None`` disables escalation.
            max_remote_pulls_per_query: Cap on remote fingerprint pulls
                per ``find_nearest()`` invocation per kind. Defaults to
                ``DEFAULT_MAX_REMOTE_PULLS_PER_QUERY``.

        Trust model: this code path inherits the
        ``EmbeddingDHTClient`` invariant — it requires a verifier
        (creator-pubkey lookup + Ed25519 verifier). Fetched
        fingerprints that fail signature verification are dropped
        silently with a debug log; we do NOT cache poisoned payloads.
        Failures of any kind degrade to local-only behavior — they
        never break the upload path.
        """
        self._backends: Dict[FingerprintKind, BinaryFingerprint] = backends or {}
        self._persist_path = persist_path
        self._duplicate_thresholds = (
            duplicate_thresholds
            if duplicate_thresholds is not None
            else dict(_DEFAULT_DUPLICATE_THRESHOLDS)
        )
        self._derivative_thresholds = (
            derivative_thresholds
            if derivative_thresholds is not None
            else dict(_DEFAULT_DERIVATIVE_THRESHOLDS)
        )

        # kind -> {content_id: (payload_bytes, creator_id)}
        self._index: Dict[FingerprintKind, Dict[str, Tuple[bytes, str]]] = {}
        for kind in self._backends:
            self._index[kind] = {}

        if persist_path and persist_path.exists():
            self._load()

        self._dht_client = dht_client
        self._peer_candidates_fn = peer_candidates_fn
        self._max_remote_pulls_per_query = (
            max_remote_pulls_per_query
            if max_remote_pulls_per_query is not None
            else self.DEFAULT_MAX_REMOTE_PULLS_PER_QUERY
        )

    # ── observation ────────────────────────────────────────────────

    def has_backend(self, kind: FingerprintKind) -> bool:
        """True iff a concrete backend is registered for *kind*."""
        return kind in self._backends

    def __len__(self) -> int:
        """Total record count across all kinds."""
        return sum(len(per_kind) for per_kind in self._index.values())

    def size(self, kind: FingerprintKind) -> int:
        """Record count for a single kind."""
        return len(self._index.get(kind, {}))

    def duplicate_threshold(self, kind: FingerprintKind) -> float:
        """Effective duplicate threshold for *kind*."""
        return self._duplicate_thresholds.get(kind, 1.0)

    def derivative_threshold(self, kind: FingerprintKind) -> float:
        """Effective derivative threshold for *kind*."""
        return self._derivative_thresholds.get(kind, 1.0)

    @property
    def dht_enabled(self) -> bool:
        """True iff DHT escalation is fully wired (client + candidates)."""
        return (
            self._dht_client is not None
            and self._peer_candidates_fn is not None
        )

    # ── compute / store / find ─────────────────────────────────────

    def compute(
        self,
        content: bytes,
        kind: FingerprintKind,
        *,
        filename: Optional[str] = None,
    ) -> Optional[FingerprintRecord]:
        """Run the registered backend for *kind* against *content*.

        Returns ``None`` if no backend is registered for the kind, the
        backend is missing its optional deps, or the content can't be
        fingerprinted by this backend (e.g. truncated audio).
        """
        backend = self._backends.get(kind)
        if backend is None:
            return None
        return backend.compute(content, filename=filename)

    def store(
        self,
        content_id: str,
        record: FingerprintRecord,
        creator_id: str,
    ) -> None:
        """Persist *record* under *content_id* + *creator_id*."""
        if record.kind not in self._index:
            # Backend wasn't registered when the index was built but a
            # caller is supplying a record of that kind anyway. Make
            # space — this keeps the index forgiving when a node loads
            # an older persisted JSON file that contained kinds the
            # current process doesn't have a backend for.
            self._index[record.kind] = {}
        self._index[record.kind][content_id] = (record.payload, creator_id)
        if self._persist_path:
            self._save()

    def find_nearest(
        self, record: FingerprintRecord,
    ) -> Optional[FingerprintMatch]:
        """Return the closest stored record of the same kind, or None.

        Uses the registered backend's ``similarity()`` function. Returns
        the first record that ties for the highest similarity if there
        are duplicates (deterministic but not load-bearing — callers
        treat the match as "a representative original").

        T4.9.next2: If DHT escalation is wired and the local best
        similarity is below the derivative threshold for this kind,
        pull peer fingerprints into the local cache and re-scan.
        Bounded by ``max_remote_pulls_per_query`` per call.
        """
        backend = self._backends.get(record.kind)
        if backend is None:
            return None

        local_best = self._scan_kind(record.kind, record.payload, backend)

        # DHT escalation: only when the local result is weak (no match
        # or sub-derivative for this kind). A strong local hit needs
        # no remote pull.
        if self.dht_enabled:
            derivative = self.derivative_threshold(record.kind)
            if local_best is None or local_best.similarity < derivative:
                pulled = self._pull_remote_fingerprints(record.kind)
                if pulled > 0:
                    local_best = self._scan_kind(
                        record.kind, record.payload, backend,
                    )

        return local_best

    def _scan_kind(
        self,
        kind: FingerprintKind,
        query_payload: bytes,
        backend: BinaryFingerprint,
    ) -> Optional[FingerprintMatch]:
        """O(n) similarity scan across the records stored for *kind*."""
        per_kind = self._index.get(kind, {})
        if not per_kind:
            return None
        best_content_id: Optional[str] = None
        best_creator: str = ""
        best_sim: float = -1.0
        for content_id, (payload, creator) in per_kind.items():
            sim = backend.similarity(query_payload, payload)
            if sim > best_sim:
                best_sim = sim
                best_content_id = content_id
                best_creator = creator
        if best_content_id is None:
            return None
        return FingerprintMatch(
            content_id=best_content_id,
            similarity=best_sim,
            creator_id=best_creator,
            kind=kind,
        )

    def _pull_remote_fingerprints(self, kind: FingerprintKind) -> int:
        """Walk peer-candidates and pull missing fingerprints of *kind*
        from the DHT into the local index.

        Returns the count of fingerprints actually inserted. Bounded by
        ``self._max_remote_pulls_per_query``. Idempotent — peers whose
        fingerprint of this kind is already cached locally are skipped.

        Failure modes (all logged at debug, all silently skipped — we
        NEVER raise out of this path because find_nearest() runs on
        the upload-critical path):
            - peer_candidates_fn raises
            - find_fingerprint_providers raises
            - no providers available
            - fetch_fingerprint raises (transport / parse / sig error)
            - server returns NOT_FOUND under our (content_hash, kind)
            - payload decode fails (malformed base64)
        """
        if not self.dht_enabled:
            return 0
        try:
            candidates = list(self._peer_candidates_fn())
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                f"peer_candidates_fn raised; skipping fingerprint DHT "
                f"escalation: {exc}"
            )
            return 0

        kind_value = kind.value
        per_kind = self._index.setdefault(kind, {})
        inserted = 0
        for content_id, content_hash in candidates:
            if inserted >= self._max_remote_pulls_per_query:
                break
            if not isinstance(content_id, str) or not content_id:
                continue
            if not isinstance(content_hash, str) or not content_hash:
                continue
            if content_id in per_kind:
                continue

            response = self._fetch_one_fingerprint(content_hash, kind_value)
            if response is None:
                continue

            try:
                payload_bytes = base64.b64decode(
                    response.payload_b64, validate=True,
                )
                if not payload_bytes:
                    raise ValueError("payload_b64 decoded to zero bytes")
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    f"fingerprint payload decode failed for "
                    f"content_hash={content_hash[:18]} kind={kind_value}: "
                    f"{exc}"
                )
                continue

            creator_id = getattr(response, "creator_id", "") or ""
            per_kind[content_id] = (payload_bytes, creator_id)
            inserted += 1

        if inserted:
            if self._persist_path:
                self._save()
            logger.info(
                f"FingerprintDHT: pulled {inserted} peer "
                f"fingerprints (kind={kind_value}) into local index"
            )
        return inserted

    def _fetch_one_fingerprint(
        self, content_hash: str, kind_value: str,
    ) -> Optional[Any]:
        """Find providers + try them in order until one returns a
        signature-verified fingerprint for ``(content_hash, kind)``.

        Returns the verified ``FingerprintResponse`` or None on any
        failure path (no providers, all providers failed, NOT_FOUND,
        signature verification failed for every candidate).
        """
        try:
            providers = self._dht_client.find_fingerprint_providers(
                content_hash, kind_value,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                f"find_fingerprint_providers raised for "
                f"content_hash={content_hash[:18]} kind={kind_value}: {exc}"
            )
            return None

        for provider in providers:
            try:
                return self._dht_client.fetch_fingerprint(
                    provider, content_hash, kind_value,
                )
            except Exception as exc:  # noqa: BLE001
                # EmbeddingNotFoundError, TransportFailureError,
                # SignatureVerificationError all land here. Try the
                # next provider — a malicious peer cannot block
                # fetching from honest peers.
                logger.debug(
                    f"fetch_fingerprint from "
                    f"{getattr(provider, 'node_id', '?')} failed: {exc}"
                )
                continue
        return None

    # ── persistence ────────────────────────────────────────────────

    def _save(self) -> None:
        if self._persist_path is None:
            return
        try:
            payload = {
                "version": 1,
                "kinds": {
                    kind.value: {
                        cid: {
                            "payload_hex": data.hex(),
                            "creator_id": creator,
                        }
                        for cid, (data, creator) in per_kind.items()
                    }
                    for kind, per_kind in self._index.items()
                },
            }
            tmp = self._persist_path.with_suffix(
                self._persist_path.suffix + ".tmp",
            )
            tmp.write_text(json.dumps(payload))
            tmp.replace(self._persist_path)
        except OSError as exc:
            logger.warning(
                f"FingerprintIndex persist failed at {self._persist_path}: {exc}",
            )

    def _load(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                f"FingerprintIndex load failed at {self._persist_path}: {exc}",
            )
            return

        kinds_data = raw.get("kinds", {})
        for kind_str, per_kind in kinds_data.items():
            try:
                kind = FingerprintKind(kind_str)
            except ValueError:
                # Persisted file had a kind we don't recognize. Skip.
                continue
            self._index.setdefault(kind, {})
            for cid, entry in per_kind.items():
                payload_hex = entry.get("payload_hex", "")
                creator = entry.get("creator_id", "")
                try:
                    payload = bytes.fromhex(payload_hex)
                except ValueError:
                    continue
                self._index[kind][cid] = (payload, creator)


__all__ = [
    "FingerprintIndex",
    "FingerprintMatch",
]
