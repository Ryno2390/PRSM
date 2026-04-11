"""
Content Index
=============

In-memory index of content available across the PRSM network.
Populated by GOSSIP_CONTENT_ADVERTISE messages — each node that
uploads or pins content broadcasts an advertisement, and every
receiving node upserts a record here.

Supports keyword search over filenames and metadata, and tracks
which nodes can serve each CID (providers).
"""

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from prsm.node.gossip import (
    GOSSIP_CONTENT_ADVERTISE,
    GOSSIP_PROVENANCE_QUERY,
    GOSSIP_PROVENANCE_REGISTER,
    GOSSIP_PROVENANCE_RESPONSE,
    GossipProtocol,
)

logger = logging.getLogger(__name__)

MAX_INDEXED_CIDS = 10_000


@dataclass
class ContentRecord:
    """A piece of content known to the network."""
    cid: str
    filename: str
    size_bytes: int
    content_hash: str
    creator_id: str
    providers: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    royalty_rate: float = 0.01
    parent_cids: List[str] = field(default_factory=list)
    embedding_id: Optional[str] = None
    near_duplicate_of: Optional[str] = None
    # Phase 1.2: canonical creator-bound provenance hash (0x-prefixed
    # hex). Set from the gossip advertisement so peers discovering
    # content remotely can still route royalties on-chain.
    provenance_hash: Optional[str] = None


class ContentIndex:
    """Network-wide content index built from gossip advertisements.

    Maintains an LRU-evicted map of CID → ContentRecord and a keyword
    index for search.  Thread-safety is not required because the event
    loop is single-threaded.
    """

    def __init__(
        self,
        gossip: GossipProtocol,
        max_indexed_cids: int = MAX_INDEXED_CIDS,
        ledger: Optional[Any] = None,
    ):
        self.gossip = gossip
        self.max_indexed_cids = max_indexed_cids
        self.ledger = ledger  # Optional LocalLedger for durable provenance storage
        # OrderedDict for LRU eviction — most-recently-touched at the end
        self._records: OrderedDict[str, ContentRecord] = OrderedDict()
        # keyword → set of CIDs that match
        self._keyword_index: Dict[str, Set[str]] = {}
        # Pending cross-node provenance lookups: cid → Future[dict]
        self._pending_provenance: Dict[str, asyncio.Future] = {}

    def start(self) -> None:
        """Subscribe to content and provenance gossip."""
        self.gossip.subscribe(GOSSIP_CONTENT_ADVERTISE, self._on_content_advertise)
        self.gossip.subscribe(GOSSIP_PROVENANCE_REGISTER, self._on_provenance_register)
        self.gossip.subscribe(GOSSIP_PROVENANCE_QUERY, self._on_provenance_query)
        self.gossip.subscribe(GOSSIP_PROVENANCE_RESPONSE, self._on_provenance_response)
        logger.info("Content index started — listening for advertisements and provenance")

    # ── Gossip handler ────────────────────────────────────────────

    async def _on_content_advertise(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Upsert a content record from a gossip advertisement."""
        cid = data.get("cid", "")
        if not cid:
            return

        provider_id = data.get("provider_id", origin)

        if cid in self._records:
            # Existing record — add the new provider and backfill any
            # empty/default fields from the new advertisement. Never
            # clobber a populated value.
            record = self._records[cid]
            record.providers.add(provider_id)
            self._backfill_record_from_advertise(record, data, origin)
            self._records.move_to_end(cid)
        else:
            # New record
            record = ContentRecord(
                cid=cid,
                filename=data.get("filename", ""),
                size_bytes=data.get("size_bytes", 0),
                content_hash=data.get("content_hash", ""),
                creator_id=data.get("creator_id", origin),
                providers={provider_id},
                created_at=data.get("created_at", time.time()),
                metadata=data.get("metadata", {}),
                royalty_rate=data.get("royalty_rate", 0.01),
                parent_cids=data.get("parent_cids", []),
                embedding_id=data.get("embedding_id"),
                near_duplicate_of=data.get("near_duplicate_of"),
                provenance_hash=data.get("provenance_hash"),
            )
            self._records[cid] = record
            self._index_keywords(record)
            self._evict_if_needed()

        logger.debug(f"Content index: {cid[:12]}... now has {len(self._records[cid].providers)} provider(s)")

    def _backfill_record_from_advertise(
        self,
        record: ContentRecord,
        data: Dict[str, Any],
        origin: str,
    ) -> None:
        """Backfill empty/default fields on an existing ContentRecord
        from a new advertisement payload. Never clobbers populated
        values.

        "Empty" is field-specific:
         - strings: "" or None → backfill
         - dicts:   {} → backfill
         - lists:   [] → backfill
         - creator_id: "" or equal to origin (the fallback used when
           the advertise payload omits creator_id; see new-record path
           above at ``creator_id=data.get("creator_id", origin)``) →
           backfill when the new payload has a different, non-empty
           value.
         - royalty_rate: 0.01 → backfill when the new payload has a
           non-default, non-0.01 value (best-effort: we can't
           distinguish explicit 0.01 from default 0.01, so the first
           advertisement to set a non-default wins).
         - None for provenance_hash / embedding_id / near_duplicate_of
           → backfill

        Phase 1.3 Task 3g — generalization of the Phase 1.2 Task 9
        provenance_hash-only backfill after codex pass 3 flagged the
        single-field version as P2. Without this, nodes that first
        learn about a CID from a minimal replica advertisement get
        stuck with creator_id = origin-peer-id, empty filename, empty
        parent_cids, and default royalty_rate forever — causing
        Task 3f's serve_on_behalf_of_replica to pay royalties to the
        wrong creator.
        """
        # String fields that backfill on empty string or None.
        for field_name in ("filename", "content_hash"):
            current = getattr(record, field_name)
            if not current:
                incoming = data.get(field_name)
                if incoming:
                    setattr(record, field_name, incoming)

        # creator_id: backfill when the existing value is empty OR
        # looks like the origin-fallback placeholder from a minimal
        # ad. The new-record path uses
        # ``data.get("creator_id", origin)`` — so when the advertise
        # payload omitted creator_id, the record's creator_id is the
        # original gossip origin peer id, which is also the initial
        # entry in ``record.providers`` (via the provider_id
        # fallback). We use "creator_id is a member of providers" as
        # the signal that it was the fallback placeholder rather
        # than a real creator. A replica's advertisement has no
        # creator_id field (storage_provider doesn't know the
        # creator), so its record comes out with creator_id =
        # storage_peer_id — definitely wrong, must be repaired when
        # a later advertisement carries the real creator. Note: this
        # heuristic assumes creator_id (typically a wallet address or
        # opaque identifier) does not collide with a peer_id. If a
        # real creator IS also a provider (self-hosting), the
        # advertisement's explicit creator_id matches the existing
        # record's creator_id and nothing is changed — safe.
        creator_is_fallback = (
            not record.creator_id
            or record.creator_id in record.providers
        )
        if creator_is_fallback:
            incoming_creator = data.get("creator_id")
            if incoming_creator and incoming_creator != record.creator_id:
                record.creator_id = incoming_creator

        # metadata dict: backfill when empty.
        if not record.metadata:
            incoming_metadata = data.get("metadata")
            if incoming_metadata:
                record.metadata = incoming_metadata

        # parent_cids list: backfill when empty.
        if not record.parent_cids:
            incoming_parents = data.get("parent_cids")
            if incoming_parents:
                record.parent_cids = list(incoming_parents)

        # royalty_rate: backfill default 0.01 when the new payload
        # carries a non-default value. Best-effort — we can't
        # distinguish an explicit 0.01 from a default 0.01, so if the
        # first advertisement legitimately set royalty_rate=0.01 a
        # later non-default ad will overwrite it. That is safer than
        # the alternative (replica minimal ads permanently pinning
        # royalty_rate to the 0.01 default).
        if record.royalty_rate == 0.01:
            incoming_rate = data.get("royalty_rate")
            if incoming_rate is not None and incoming_rate != 0.01:
                record.royalty_rate = incoming_rate

        # Optional fields that are None by default.
        for field_name in (
            "embedding_id",
            "near_duplicate_of",
            "provenance_hash",
        ):
            if getattr(record, field_name) is None:
                incoming = data.get(field_name)
                if incoming is not None:
                    setattr(record, field_name, incoming)

    async def _on_provenance_register(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Persist a provenance registration to the local ledger."""
        if not self.ledger or not data.get("cid"):
            return
        try:
            await self.ledger.upsert_provenance(data)
        except Exception as exc:
            logger.warning(f"Failed to persist provenance for {data.get('cid', '?')[:12]}: {exc}")

    async def _on_provenance_query(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Respond to a cross-node provenance query if we have the record locally."""
        cid = data.get("cid", "")
        requester_id = data.get("requester_id", "")
        if not cid or not self.ledger:
            return
        try:
            record = await self.ledger.get_provenance(cid)
            if record:
                await self.gossip.publish(GOSSIP_PROVENANCE_RESPONSE, {
                    "cid": cid,
                    "for_requester": requester_id,
                    "provenance": record,
                })
                logger.debug(f"Answered provenance query for {cid[:12]}...")
        except Exception as exc:
            logger.warning(f"Error handling provenance query for {cid[:12]}: {exc}")

    async def _on_provenance_response(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Handle a provenance response — persist it and resolve any pending query."""
        cid = data.get("cid", "")
        provenance = data.get("provenance", {})
        if not cid or not provenance:
            return
        # Persist to local ledger so future lookups are instant
        if self.ledger:
            try:
                await self.ledger.upsert_provenance(provenance)
            except Exception as exc:
                logger.warning(f"Failed to persist provenance response for {cid[:12]}: {exc}")
        # Resolve any pending async get_provenance() call
        future = self._pending_provenance.get(cid)
        if future and not future.done():
            future.set_result(provenance)

    # ── Public queries ────────────────────────────────────────────

    def lookup(self, cid: str) -> Optional[ContentRecord]:
        """Look up a content record by CID."""
        return self._records.get(cid)

    def search(self, query: str, limit: int = 20) -> List[ContentRecord]:
        """Keyword search over filenames and metadata values.

        Returns records whose filename or metadata contain *all* query
        words (AND semantics).  Results are ordered most-recent first.
        """
        words = self._tokenize(query)
        if not words:
            return []

        # Intersect CID sets for each keyword
        matching_cids: Optional[Set[str]] = None
        for word in words:
            cids = self._keyword_index.get(word, set())
            if matching_cids is None:
                matching_cids = set(cids)
            else:
                matching_cids &= cids

        if not matching_cids:
            return []

        # Collect records, most-recently-advertised first
        results: List[ContentRecord] = []
        for cid in reversed(self._records):
            if cid in matching_cids:
                results.append(self._records[cid])
                if len(results) >= limit:
                    break
        return results

    def get_providers(self, cid: str) -> Set[str]:
        """Return the set of node IDs that can serve this CID."""
        record = self._records.get(cid)
        return record.providers if record else set()

    async def get_provenance(
        self, cid: str, timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """Return the provenance record for a CID.

        Resolution order:
        1. Local SQLite ledger (instant, survives restarts).
        2. Cross-node gossip query: broadcasts GOSSIP_PROVENANCE_QUERY and
           waits up to *timeout* seconds for a GOSSIP_PROVENANCE_RESPONSE.
           The response is persisted locally so subsequent calls are instant.

        Returns None if no record is found within the timeout.
        """
        # 1. Check local ledger first
        if self.ledger:
            try:
                record = await self.ledger.get_provenance(cid)
                if record:
                    return record
            except Exception as exc:
                logger.warning(f"Ledger provenance lookup failed for {cid[:12]}: {exc}")

        # 2. Broadcast a query and wait for a response from any peer
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None  # No event loop — can't do async query

        future: asyncio.Future = loop.create_future()
        self._pending_provenance[cid] = future
        try:
            await self.gossip.publish(GOSSIP_PROVENANCE_QUERY, {
                "cid": cid,
                "requester_id": "",  # origin is set automatically by gossip layer
            })
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug(f"Provenance query timed out for {cid[:12]}...")
            return None
        except Exception as exc:
            logger.warning(f"Provenance query failed for {cid[:12]}: {exc}")
            return None
        finally:
            self._pending_provenance.pop(cid, None)

    def get_stats(self) -> Dict[str, Any]:
        """Index statistics for the status endpoint."""
        unique_providers: Set[str] = set()
        for rec in self._records.values():
            unique_providers |= rec.providers
        return {
            "indexed_cids": len(self._records),
            "unique_providers": len(unique_providers),
            "keyword_entries": len(self._keyword_index),
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _index_keywords(self, record: ContentRecord) -> None:
        """Add keywords from a record's filename and metadata."""
        text_parts = [record.filename]
        for v in record.metadata.values():
            if isinstance(v, str):
                text_parts.append(v)

        cid = record.cid
        for word in self._tokenize(" ".join(text_parts)):
            self._keyword_index.setdefault(word, set()).add(cid)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase split, stripping common punctuation."""
        import re
        return [w for w in re.split(r"[\s_\-./\\]+", text.lower()) if len(w) >= 2]

    def _evict_if_needed(self) -> None:
        """Remove the oldest entries when the index exceeds the cap."""
        while len(self._records) > self.max_indexed_cids:
            evicted_cid, evicted_record = self._records.popitem(last=False)
            # Clean up keyword index references
            for word in self._tokenize(evicted_record.filename):
                kw_set = self._keyword_index.get(word)
                if kw_set:
                    kw_set.discard(evicted_cid)
                    if not kw_set:
                        del self._keyword_index[word]
