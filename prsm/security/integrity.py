"""
Integrity Verifier
==================

Validates model shard checksums and detects tampering.
"""

import hashlib
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class IntegrityVerifier:
    """Verifies integrity of model shards and data."""

    @staticmethod
    def compute_checksum(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def verify_shard(shard_data: bytes, expected_checksum: str) -> bool:
        actual = hashlib.sha256(shard_data).hexdigest()
        return actual == expected_checksum

    def verify_sharded_model(self, shards: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Verify all shards in a model. Returns (all_valid, errors)."""
        errors = []
        for shard in shards:
            data = shard.get("tensor_data", b"")
            expected = shard.get("checksum", "")
            if not expected:
                errors.append(f"Shard {shard.get('shard_id', '?')}: no checksum")
                continue
            if isinstance(data, str):
                data = data.encode()
            if not self.verify_shard(data, expected):
                errors.append(f"Shard {shard.get('shard_id', '?')}: checksum mismatch")
        return len(errors) == 0, errors

    def verify_hash_chain(self, entries: List[Dict[str, Any]]) -> Tuple[bool, int]:
        """Verify a hash chain (each entry includes hash of previous).

        Returns (valid, break_index). break_index is -1 if valid.
        """
        if not entries:
            return True, -1

        for i in range(1, len(entries)):
            prev_hash = entries[i].get("prev_hash", "")
            expected = hashlib.sha256(
                str(entries[i - 1]).encode()
            ).hexdigest()
            if prev_hash != expected:
                return False, i

        return True, -1
