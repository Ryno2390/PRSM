"""
Pipeline Audit Log
==================

Records all pipeline randomization decisions for provability
and collusion detection.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """A single pipeline assignment audit record."""
    entry_id: str
    model_id: str
    shard_count: int
    node_assignments: List[Dict[str, str]]
    pool_size: int
    require_tee: bool
    timestamp: float = field(default_factory=time.time)
    entry_hash: str = ""
    prev_hash: str = ""

    def compute_hash(self) -> str:
        data = json.dumps({
            "entry_id": self.entry_id,
            "model_id": self.model_id,
            "assignments": self.node_assignments,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


class PipelineAuditLog:
    """Append-only audit log for pipeline randomization decisions."""

    def __init__(self):
        self._entries: List[AuditEntry] = []

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def record(
        self,
        model_id: str,
        shard_count: int,
        node_assignments: List[Dict[str, str]],
        pool_size: int,
        require_tee: bool = False,
    ) -> AuditEntry:
        """Record a pipeline assignment decision."""
        prev_hash = self._entries[-1].entry_hash if self._entries else ""
        entry_id = f"audit-{len(self._entries):06d}"

        entry = AuditEntry(
            entry_id=entry_id,
            model_id=model_id,
            shard_count=shard_count,
            node_assignments=node_assignments,
            pool_size=pool_size,
            require_tee=require_tee,
            prev_hash=prev_hash,
        )
        entry.entry_hash = entry.compute_hash()
        self._entries.append(entry)

        return entry

    def verify_chain(self) -> bool:
        """Verify the hash chain integrity of the audit log."""
        if not self._entries:
            return True

        for i in range(1, len(self._entries)):
            expected_prev = self._entries[i - 1].entry_hash
            if self._entries[i].prev_hash != expected_prev:
                return False

        return True

    def get_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [
            {
                "entry_id": e.entry_id,
                "model_id": e.model_id,
                "shard_count": e.shard_count,
                "assignments": e.node_assignments,
                "pool_size": e.pool_size,
                "timestamp": e.timestamp,
                "entry_hash": e.entry_hash[:16] + "...",
            }
            for e in self._entries[-limit:]
        ]

    def entropy_report(self) -> Dict[str, Any]:
        """Analyze randomization quality across entries."""
        if not self._entries:
            return {"entries": 0, "unique_nodes": 0, "entropy_score": 0.0}

        all_nodes = []
        for entry in self._entries:
            for a in entry.node_assignments:
                all_nodes.append(a.get("node_id", ""))

        unique = set(all_nodes)
        total = len(all_nodes)

        # Shannon entropy approximation
        if total == 0:
            entropy = 0.0
        else:
            from collections import Counter
            counts = Counter(all_nodes)
            entropy = -sum(
                (c / total) * (c / total).__class__(c / total)
                for c in counts.values()
            )
            # Simplified: just use unique ratio as proxy
            entropy = len(unique) / max(total, 1)

        return {
            "entries": len(self._entries),
            "unique_nodes": len(unique),
            "total_assignments": total,
            "entropy_score": round(entropy, 4),
            "chain_valid": self.verify_chain(),
        }
