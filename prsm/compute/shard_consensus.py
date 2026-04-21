"""Phase 7.1 Task 1: ShardConsensus.

Pure-function resolver that takes a list of ShardExecutionReceipts from
k providers who executed the same shard and returns a ConsensusOutcome
partitioning them into majority (agreed on the winning output_hash) and
minority (disagreed) groups.

Design contract (docs/2026-04-21-phase7.1-redundant-execution-design.md §3.4):
  - Hash-exact comparison via ShardExecutionReceipt.output_hash. No
    floating-point tolerance at the shard layer — Phase 2's float64
    bitwise-reproduction invariant is the foundation.
  - MAJORITY: largest agreeing group size >= (k // 2) + 1.
  - UNANIMOUS: all responded receipts agree AND the response count == k.
  - SINGLE: k = 1; any lone receipt is trivially agreed.
  - BYZANTINE: deferred to Phase 7.1x — raises NotImplementedError.

No I/O, no async, no signature verification (caller has already run
ReceiptOnlyVerification per Phase 2 before feeding us). This module
only decides whether the collected receipts agree.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.node.result_consensus import ConsensusMode


@dataclass(frozen=True)
class ConsensusOutcome:
    """Result of ShardConsensus.resolve.

    `agreed` is True iff the mode's threshold was met. When False, the
    caller should treat the dispatch as a consensus failure and surface
    an error to the requester — no output is canonical.

    `majority` and `minority` partition the input receipts:
      - majority: receipts whose output_hash == agreed_output_hash.
      - minority: receipts whose output_hash != agreed_output_hash.
    When `agreed` is False, both lists are empty (partition undefined).
    """
    agreed: bool
    agreed_output_hash: Optional[str]
    majority: List[ShardExecutionReceipt]
    minority: List[ShardExecutionReceipt]


class ShardConsensus:
    """Resolve k-of-n output-hash consensus on collected shard receipts."""

    def __init__(self, mode: ConsensusMode, k: int):
        if not isinstance(mode, ConsensusMode):
            raise TypeError(f"mode must be ConsensusMode, got {type(mode).__name__}")
        if k < 1:
            raise ValueError(f"k must be >= 1 (got {k})")
        if mode == ConsensusMode.BYZANTINE:
            # Stubbed for 7.1x. Surface the missing capability loudly at
            # construction rather than letting callers think it's wired.
            raise NotImplementedError(
                "BYZANTINE consensus mode is deferred to Phase 7.1x — use "
                "MAJORITY or UNANIMOUS for Phase 7.1 MVP"
            )
        self.mode = mode
        self.k = k

    def resolve(
        self, receipts: List[ShardExecutionReceipt],
    ) -> ConsensusOutcome:
        """Partition `receipts` into majority / minority under this mode.

        Caller contract: all receipts must share the same job_id and
        shard_index. Violation is a caller bug and raises — we don't
        silently tolerate a malformed consensus group.
        """
        if not receipts:
            return ConsensusOutcome(
                agreed=False, agreed_output_hash=None,
                majority=[], minority=[],
            )

        # Defensive: ensure the caller built the consensus group from a
        # single shard. Mixing shards into one resolve() call would
        # produce nonsense agreement counts.
        first = receipts[0]
        for r in receipts[1:]:
            if r.job_id != first.job_id or r.shard_index != first.shard_index:
                raise ValueError(
                    f"receipts mix job_id/shard_index: "
                    f"({first.job_id!r}, {first.shard_index}) vs "
                    f"({r.job_id!r}, {r.shard_index})"
                )

        counts = Counter(r.output_hash for r in receipts)
        winning_hash, winning_count = counts.most_common(1)[0]

        # Ambiguous ties at the top → no agreement. This is rare at odd k
        # but possible at even k (e.g., 2-2 at k=4) or under partial
        # response (e.g., 1-1 when only 2 of 3 responded and disagreed).
        top_groups = [h for h, c in counts.items() if c == winning_count]
        if len(top_groups) > 1:
            return ConsensusOutcome(
                agreed=False, agreed_output_hash=None,
                majority=[], minority=[],
            )

        if self.mode == ConsensusMode.SINGLE:
            # k=1 by construction at the policy layer; any lone receipt
            # is trivially agreed. Guard against pathological k>1 SINGLE.
            agreed = len(receipts) >= 1 and winning_count == len(receipts)
        elif self.mode == ConsensusMode.MAJORITY:
            threshold = (self.k // 2) + 1
            agreed = winning_count >= threshold
        elif self.mode == ConsensusMode.UNANIMOUS:
            # All k must have responded AND all responses must agree.
            agreed = len(receipts) == self.k and winning_count == self.k
        else:  # pragma: no cover — BYZANTINE blocked in __init__
            raise RuntimeError(f"unreachable mode: {self.mode}")

        if not agreed:
            return ConsensusOutcome(
                agreed=False, agreed_output_hash=None,
                majority=[], minority=[],
            )

        majority = [r for r in receipts if r.output_hash == winning_hash]
        minority = [r for r in receipts if r.output_hash != winning_hash]
        return ConsensusOutcome(
            agreed=True,
            agreed_output_hash=winning_hash,
            majority=majority,
            minority=minority,
        )
