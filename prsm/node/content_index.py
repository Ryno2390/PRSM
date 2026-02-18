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

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from prsm.node.gossip import GOSSIP_CONTENT_ADVERTISE, GossipProtocol

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


class ContentIndex:
    """Network-wide content index built from gossip advertisements.

    Maintains an LRU-evicted map of CID → ContentRecord and a keyword
    index for search.  Thread-safety is not required because the event
    loop is single-threaded.
    """

    def __init__(self, gossip: GossipProtocol):
        self.gossip = gossip
        # OrderedDict for LRU eviction — most-recently-touched at the end
        self._records: OrderedDict[str, ContentRecord] = OrderedDict()
        # keyword → set of CIDs that match
        self._keyword_index: Dict[str, Set[str]] = {}

    def start(self) -> None:
        """Subscribe to content advertisements on the gossip layer."""
        self.gossip.subscribe(GOSSIP_CONTENT_ADVERTISE, self._on_content_advertise)
        logger.info("Content index started — listening for advertisements")

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
            # Existing record — add the new provider and refresh LRU
            record = self._records[cid]
            record.providers.add(provider_id)
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
            )
            self._records[cid] = record
            self._index_keywords(record)
            self._evict_if_needed()

        logger.debug(f"Content index: {cid[:12]}... now has {len(self._records[cid].providers)} provider(s)")

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
        while len(self._records) > MAX_INDEXED_CIDS:
            evicted_cid, evicted_record = self._records.popitem(last=False)
            # Clean up keyword index references
            for word in self._tokenize(evicted_record.filename):
                kw_set = self._keyword_index.get(word)
                if kw_set:
                    kw_set.discard(evicted_cid)
                    if not kw_set:
                        del self._keyword_index[word]
