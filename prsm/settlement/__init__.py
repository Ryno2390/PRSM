"""Phase 3.1 batched-settlement package.

Off-chain half of the Lightning-style settlement layer that replaces
Phase 3's local-ledger-only settlement. Providers accumulate Phase 2
ShardExecutionReceipts via ReceiptAccumulator (Task 4), build Merkle
trees (Task 5), submit commits to the BatchSettlementRegistry contract
(Task 6), and reconcile local PaymentEscrow state on finalization.

See docs/2026-04-21-phase3.1-batch-settlement-design.md for the full
protocol spec.
"""

from prsm.settlement.accumulator import (
    AccumulatorConfig,
    BatchedReceipt,
    PendingBatch,
    ReadyBatch,
    ReceiptAccumulator,
    TriggerReason,
)
from prsm.settlement.client import (
    BatchSettlementClient,
    CommittedBatch,
    FinalizedBatch,
    SettlementContractClient,
)
from prsm.settlement.merkle import (
    MerkleTree,
    ReceiptLeaf,
    batched_receipt_to_leaf,
    build_merkle_proof,
    build_merkle_root,
    build_tree_and_proofs,
    encode_leaf,
    hash_leaf,
    verify_merkle_proof,
)

__all__ = [
    # Accumulator (Task 4)
    "AccumulatorConfig",
    "BatchedReceipt",
    "PendingBatch",
    "ReadyBatch",
    "ReceiptAccumulator",
    "TriggerReason",
    # Merkle + canonical encoding (Task 5)
    "MerkleTree",
    "ReceiptLeaf",
    "batched_receipt_to_leaf",
    "build_merkle_proof",
    "build_merkle_root",
    "build_tree_and_proofs",
    "encode_leaf",
    "hash_leaf",
    "verify_merkle_proof",
    # Client (Task 6)
    "BatchSettlementClient",
    "CommittedBatch",
    "FinalizedBatch",
    "SettlementContractClient",
]
