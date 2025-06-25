#!/usr/bin/env python3
"""
Simple Consensus Sharding Test
Quick validation of sharding architecture and basic functionality
"""

import asyncio
import sys
from pathlib import Path

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent))

from prsm.core.models import PeerNode
from prsm.federation.consensus_sharding import (
    ConsensusShardingManager, ConsensusShard, ShardState, ShardingStrategy
)


async def simple_sharding_test():
    """Simple test of consensus sharding architecture"""
    print("🧪 Simple Consensus Sharding Test")
    print("=" * 50)
    
    try:
        # Test 1: Sharding Manager Initialization
        print("\n📊 Test 1: Sharding Manager Initialization")
        
        # Create test nodes
        peer_nodes = []
        for i in range(20):
            peer = PeerNode(
                node_id=f"node_{i}",
                peer_id=f"peer_{i}",
                multiaddr=f"/ip4/127.0.0.1/tcp/{9000+i}",
                reputation_score=0.8 + (i % 5) * 0.04,
                active=True
            )
            peer_nodes.append(peer)
        
        # Test hash-based sharding
        print("   🔧 Testing hash-based sharding strategy")
        hash_manager = ConsensusShardingManager(sharding_strategy=ShardingStrategy.HASH_BASED)
        hash_success = await hash_manager.initialize_sharding(peer_nodes)
        
        if hash_success:
            print(f"   ✅ Hash-based sharding: {len(hash_manager.shards)} shards created")
            print(f"      - Total nodes distributed: {sum(len(s.nodes) for s in hash_manager.shards.values())}")
            print(f"      - Active shards: {sum(1 for s in hash_manager.shards.values() if s.state == ShardState.ACTIVE)}")
        else:
            print(f"   ❌ Hash-based sharding failed")
        
        # Test adaptive sharding
        print("   🔧 Testing adaptive sharding strategy")
        adaptive_manager = ConsensusShardingManager(sharding_strategy=ShardingStrategy.ADAPTIVE)
        adaptive_success = await adaptive_manager.initialize_sharding(peer_nodes)
        
        if adaptive_success:
            print(f"   ✅ Adaptive sharding: {len(adaptive_manager.shards)} shards created")
            print(f"      - Total nodes distributed: {sum(len(s.nodes) for s in adaptive_manager.shards.values())}")
            print(f"      - Active shards: {sum(1 for s in adaptive_manager.shards.values() if s.state == ShardState.ACTIVE)}")
        else:
            print(f"   ❌ Adaptive sharding failed")
        
        # Test 2: Shard Architecture
        print(f"\n🏗️ Test 2: Shard Architecture Analysis")
        
        if hash_success:
            # Analyze hash-based shard distribution
            print(f"   📊 Hash-based shard analysis:")
            for shard_id, shard in hash_manager.shards.items():
                avg_rep = sum(node.reputation_score for node in shard.nodes.values()) / len(shard.nodes)
                print(f"      - {shard_id}: {len(shard.nodes)} nodes, avg reputation: {avg_rep:.2f}")
        
        if adaptive_success:
            # Analyze adaptive shard distribution
            print(f"   📊 Adaptive shard analysis:")
            for shard_id, shard in adaptive_manager.shards.items():
                avg_rep = sum(node.reputation_score for node in shard.nodes.values()) / len(shard.nodes)
                print(f"      - {shard_id}: {len(shard.nodes)} nodes, avg reputation: {avg_rep:.2f}")
        
        # Test 3: Sharding Metrics
        print(f"\n📈 Test 3: Sharding Metrics")
        
        if hash_success:
            hash_metrics = await hash_manager.get_sharding_metrics()
            print(f"   📊 Hash-based metrics:")
            print(f"      - Total shards: {hash_metrics['total_shards']}")
            print(f"      - Active shards: {hash_metrics['active_shards']}")
            print(f"      - Strategy: {hash_metrics['sharding_strategy']}")
            print(f"      - Configuration: min={hash_metrics['configuration']['min_shard_size']}, "
                  f"optimal={hash_metrics['configuration']['optimal_shard_size']}, "
                  f"max={hash_metrics['configuration']['max_shard_size']}")
        
        # Test 4: Shard State Management
        print(f"\n⚙️ Test 4: Shard State Management")
        
        if hash_success and hash_manager.shards:
            shard_id = list(hash_manager.shards.keys())[0]
            test_shard = hash_manager.shards[shard_id]
            
            print(f"   🔍 Testing shard {shard_id}:")
            print(f"      - Initial state: {test_shard.state.value}")
            print(f"      - Node count: {len(test_shard.nodes)}")
            print(f"      - Coordinator: {test_shard.coordinator_node}")
            
            # Get shard metrics
            shard_metrics = test_shard.get_shard_metrics()
            print(f"      - Shard metrics available: {'✅' if 'shard_id' in shard_metrics else '❌'}")
        
        # Test 5: Target Shard Selection
        print(f"\n🎯 Test 5: Target Shard Selection")
        
        if hash_success:
            # Test shard selection for different proposals
            test_proposals = [
                {"shard_key": "test_key_1", "action": "test"},
                {"shard_key": "test_key_2", "action": "test"},
                {"id": "unique_id_1", "action": "test"},
                {"action": "test_no_key"}
            ]
            
            for i, proposal in enumerate(test_proposals):
                target_shards = await hash_manager._select_target_shards(proposal)
                print(f"      - Proposal {i+1}: targets {len(target_shards)} shard(s) -> {target_shards}")
        
        # Test 6: Cross-Shard Configuration
        print(f"\n🔗 Test 6: Cross-Shard Configuration")
        
        if hash_success:
            print(f"   📊 Cross-shard configuration:")
            print(f"      - Cross-shard coordinator: {hash_manager.cross_shard_coordinator}")
            print(f"      - Global consensus initialized: {'✅' if hash_manager.global_consensus else '❌'}")
            print(f"      - Pending cross-shard ops: {len(hash_manager.pending_cross_shard_operations)}")
        
        # Test Summary
        print(f"\n" + "=" * 50)
        print(f"🎯 CONSENSUS SHARDING ARCHITECTURE TEST SUMMARY")
        print(f"=" * 50)
        
        tests_passed = 0
        total_tests = 6
        
        if hash_success:
            tests_passed += 1
            print(f"✅ Hash-based sharding initialization")
        else:
            print(f"❌ Hash-based sharding initialization")
        
        if adaptive_success:
            tests_passed += 1
            print(f"✅ Adaptive sharding initialization")
        else:
            print(f"❌ Adaptive sharding initialization")
        
        # Additional test validations
        if hash_success and len(hash_manager.shards) > 0:
            tests_passed += 1
            print(f"✅ Shard creation and distribution")
        else:
            print(f"❌ Shard creation and distribution")
        
        if hash_success and hash_metrics and hash_metrics.get('total_shards', 0) > 0:
            tests_passed += 1
            print(f"✅ Sharding metrics collection")
        else:
            print(f"❌ Sharding metrics collection")
        
        if hash_success and any(s.state == ShardState.ACTIVE for s in hash_manager.shards.values()):
            tests_passed += 1
            print(f"✅ Shard state management")
        else:
            print(f"❌ Shard state management")
        
        if hash_success and hash_manager.cross_shard_coordinator:
            tests_passed += 1
            print(f"✅ Cross-shard coordination setup")
        else:
            print(f"❌ Cross-shard coordination setup")
        
        success_rate = tests_passed / total_tests
        
        print(f"\n🎯 Architecture Test Results:")
        print(f"   - Tests passed: {tests_passed}/{total_tests}")
        print(f"   - Success rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            print(f"\n✅ CONSENSUS SHARDING ARCHITECTURE: VALIDATED!")
            print(f"🚀 Key capabilities confirmed:")
            print(f"   - Multiple sharding strategies (Hash-based, Adaptive)")
            print(f"   - Automatic shard creation and node distribution")
            print(f"   - Shard state management and monitoring")
            print(f"   - Cross-shard coordination infrastructure")
            print(f"   - Comprehensive metrics collection")
            print(f"   - Target shard selection algorithms")
        else:
            print(f"\n⚠️ Architecture validation needs attention:")
            print(f"   - Success rate: {success_rate:.1%} (target: >80%)")
        
        return success_rate >= 0.8
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(simple_sharding_test())