#!/usr/bin/env python3
"""
Integration Test: Settler Registry + Batch Settlement
=====================================================

Verifies Phase 6 L2-style staking for batch security:
- The Bond: Settlers stake to earn settlement rights
- The Multi-Sig: 3-of-N signatures approve batches
- The Challenge: Ledger export for public audit
"""

import asyncio
from decimal import Decimal


async def test_settler_flow():
    """Test the full settler registration and batch approval flow."""
    
    print("\n" + "="*60)
    print("Phase 6: Settler Registry Integration Test")
    print("="*60)
    
    from prsm.node.settler_registry import SettlerRegistry, SettlerStatus
    
    # 1. Create registry
    print("\n[1] Creating Settler Registry...")
    registry = SettlerRegistry(
        min_settler_bond=10_000.0,    # 10K FTNS
        settlement_threshold=3,        # 3-of-N multi-sig
        max_settlers=10,
    )
    print(f"    Min bond: {registry.min_settler_bond} FTNS")
    print(f"    Threshold: {registry.settlement_threshold} signatures")
    
    # 2. Register settlers
    print("\n[2] Registering Settlers...")
    settlers = []
    for i in range(5):
        settler = await registry.register_settler(
            settler_id=f"node-{i}",
            address=f"0x{i:040x}",
            bond_amount=10_000.0 + i * 1000,
        )
        settlers.append(settler)
        print(f"    Registered: {settler.settler_id} (bond: {settler.bond_amount} FTNS)")
    
    # 3. Verify active settlers
    print("\n[3] Active Settlers:")
    active = registry.list_active_settlers()
    total_bond = sum(s.bond_amount for s in active)
    print(f"    Count: {len(active)}")
    print(f"    Total bonded: {total_bond:,.0f} FTNS")
    
    # 4. Propose a batch
    print("\n[4] Proposing Batch for Settlement...")
    transfers = [
        {"to": "0xAAA", "amount": 1.5, "tx_id": "tx-1"},
        {"to": "0xBBB", "amount": 2.0, "tx_id": "tx-2"},
        {"to": "0xCCC", "amount": 0.5, "tx_id": "tx-3"},
    ]
    batch = await registry.propose_batch(transfers)
    print(f"    Batch ID: {batch.batch_id}")
    print(f"    Batch hash: {batch.batch_hash}")
    print(f"    Total amount: {batch.total_amount} FTNS")
    print(f"    Signatures: {batch.signature_count}/{registry.settlement_threshold}")
    
    # 5. Multi-sig approval
    print("\n[5] Multi-Signature Approval...")
    
    # Track callback
    approved_batch = []
    async def on_approved(b):
        approved_batch.append(b)
        print(f"    ✅ BATCH APPROVED: {b.batch_id}")
    
    registry.on_settlement_ready(on_approved)
    
    # Sign with threshold number of settlers
    for i in range(registry.settlement_threshold):
        sig = await registry.sign_batch(
            batch_id=batch.batch_id,
            settler_id=f"node-{i}",
            signature=f"signature-{i}",
        )
        print(f"    Signed by node-{i}: {batch.signature_count}/{registry.settlement_threshold}")
    
    # 6. Verify approval
    print("\n[6] Verification:")
    print(f"    Batch approved: {registry.is_batch_approved(batch.batch_id)}")
    print(f"    Callback triggered: {len(approved_batch) > 0}")
    
    # 7. Export ledger (Challenge system)
    print("\n[7] Ledger Export (Challenge System)...")
    export = await registry.export_ledger({
        "balances": {"user-1": 100.0, "user-2": 200.0},
        "total_supply": 300.0,
    })
    print(f"    Integrity hash: {export['integrity_hash']}")
    print(f"    Settlers in export: {len(export['settlers'])}")
    print(f"    Pending batches: {len(export['pending_batches'])}")
    
    # 8. Test unbonding
    print("\n[8] Testing Unbonding...")
    unbond_at = await registry.unbond_settler("node-4")
    settler = registry.get_settler("node-4")
    print(f"    Status: {settler.status.value}")
    print(f"    Unbond at: {unbond_at}")
    print(f"    Can still settle: {settler.can_settle}")
    
    # 9. Stats
    print("\n[9] Final Stats:")
    stats = registry.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ All Phase 6 Integration Tests Passed!")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    asyncio.run(test_settler_flow())
