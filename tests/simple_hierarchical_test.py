#!/usr/bin/env python3
"""
Simple Hierarchical Consensus Test
Quick validation of hierarchical consensus implementation
"""

import asyncio
import sys
from pathlib import Path

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent))

from prsm.core.models import PeerNode
from prsm.compute.federation.hierarchical_consensus import (
    HierarchicalConsensusNetwork, get_hierarchical_consensus
)


async def simple_hierarchical_test():
    """Simple test of hierarchical consensus"""
    print("🧪 Simple Hierarchical Consensus Test")
    print("=" * 40)
    
    try:
        # Test with different network sizes
        test_sizes = [10, 25, 50, 100]
        
        for size in test_sizes:
            print(f"\n🔧 Testing {size} nodes:")
            
            # Create peer nodes
            peer_nodes = []
            for i in range(size):
                peer = PeerNode(
                    node_id=f"node_{i}",
                    peer_id=f"peer_{i}",
                    multiaddr=f"/ip4/127.0.0.1/tcp/{5000+i}",
                    reputation_score=0.8,
                    active=True
                )
                peer_nodes.append(peer)
            
            # Initialize hierarchical network
            hierarchical_network = HierarchicalConsensusNetwork()
            success = await hierarchical_network.initialize_hierarchical_network(peer_nodes)
            
            if not success:
                print(f"   ❌ Failed to initialize network")
                continue
            
            # Get metrics
            metrics = await hierarchical_network.get_hierarchical_metrics()
            topology = await hierarchical_network.get_network_topology()
            
            # Test consensus
            proposal = {
                "action": "test_consensus",
                "network_size": size,
                "data": f"test_data_{size}"
            }
            
            result = await hierarchical_network.achieve_hierarchical_consensus(proposal)
            
            # Calculate improvements
            flat_messages = size * (size - 1)
            hier_messages = metrics['complexity']['hierarchical_message_complexity']
            message_reduction = metrics['complexity']['complexity_reduction']
            
            print(f"   ✅ Results:")
            print(f"      - Consensus: {'✅' if result.consensus_achieved else '❌'}")
            print(f"      - Tiers: {len(topology['tiers'])}")
            print(f"      - Hierarchy depth: {metrics['topology']['hierarchy_depth']}")
            print(f"      - Flat messages: {flat_messages}")
            print(f"      - Hierarchical messages: {hier_messages}")
            print(f"      - Message reduction: {message_reduction:.1%}")
            print(f"      - Improvement factor: {flat_messages/max(1, hier_messages):.1f}x")
        
        print(f"\n✅ HIERARCHICAL CONSENSUS TEST COMPLETE!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(simple_hierarchical_test())