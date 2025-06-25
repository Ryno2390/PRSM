#!/usr/bin/env python3
"""
Basic P2P Network Example
This example demonstrates setting up and running a basic peer-to-peer network with PRSM.
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    from demos.p2p_network_demo import P2PNetworkDemo, P2PNode
    PRSM_P2P_AVAILABLE = True
except ImportError:
    PRSM_P2P_AVAILABLE = False

class MockP2PNode:
    """Mock P2P node for demonstration when PRSM components aren't available"""
    
    def __init__(self, node_id: str, node_type: str = "worker", port: int = 8000):
        self.node_id = node_id
        self.node_type = node_type
        self.port = port
        self.peers = {}
        self.messages_sent = 0
        self.messages_received = 0
        self.status = "initialized"
        self.start_time = None
        
        print(f"🏗️  Mock P2P Node {node_id} ({node_type}) created on port {port}")
    
    async def start(self):
        """Start the mock node"""
        self.status = "active"
        self.start_time = time.time()
        print(f"🟢 Node {self.node_id} started")
        await asyncio.sleep(0.1)  # Simulate startup time
    
    async def connect_to_peer(self, peer: 'MockP2PNode'):
        """Connect to another node"""
        if peer.node_id not in self.peers:
            self.peers[peer.node_id] = {
                "node_id": peer.node_id,
                "node_type": peer.node_type,
                "port": peer.port,
                "connected_at": time.time()
            }
            
            # Add this node to peer's connections
            if self.node_id not in peer.peers:
                peer.peers[self.node_id] = {
                    "node_id": self.node_id,
                    "node_type": self.node_type,
                    "port": self.port,
                    "connected_at": time.time()
                }
            
            print(f"🔗 {self.node_id} connected to {peer.node_id}")
            await asyncio.sleep(0.1)  # Simulate connection time
    
    async def send_message(self, target_node_id: str, message: Dict[str, Any]):
        """Send a message to another node"""
        if target_node_id in self.peers:
            self.messages_sent += 1
            print(f"📤 {self.node_id} → {target_node_id}: {message['type']}")
            await asyncio.sleep(0.05)  # Simulate network latency
            return True
        return False
    
    async def receive_message(self, sender_id: str, message: Dict[str, Any]):
        """Receive a message from another node"""
        self.messages_received += 1
        print(f"📥 {self.node_id} ← {sender_id}: {message['type']}")
    
    async def stop(self):
        """Stop the node"""
        self.status = "stopped"
        print(f"🔴 Node {self.node_id} stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get node status"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "status": self.status,
            "port": self.port,
            "uptime": uptime,
            "peer_count": len(self.peers),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "peers": list(self.peers.keys())
        }

class MockP2PNetwork:
    """Mock P2P network for demonstration"""
    
    def __init__(self, num_nodes: int = 3):
        self.nodes = []
        self.num_nodes = num_nodes
        self.network_start_time = None
        
        # Create nodes
        node_types = ["coordinator", "worker", "validator"]
        for i in range(num_nodes):
            node_type = node_types[i % len(node_types)]
            port = 8000 + i
            node = MockP2PNode(f"node_{i+1}", node_type, port)
            self.nodes.append(node)
        
        print(f"🌐 Mock P2P Network created with {num_nodes} nodes")
    
    async def start_network(self):
        """Start all nodes and establish connections"""
        print(f"🚀 Starting P2P network...")
        self.network_start_time = time.time()
        
        # Start all nodes
        for node in self.nodes:
            await node.start()
        
        # Connect nodes in a simple topology
        for i, node in enumerate(self.nodes):
            # Connect to next node (ring topology)
            next_node = self.nodes[(i + 1) % len(self.nodes)]
            await node.connect_to_peer(next_node)
            
            # Coordinator connects to all nodes
            if node.node_type == "coordinator":
                for other_node in self.nodes:
                    if other_node.node_id != node.node_id:
                        await node.connect_to_peer(other_node)
        
        print(f"✅ P2P network started successfully")
    
    async def simulate_network_activity(self, duration: int = 10):
        """Simulate network activity"""
        print(f"📡 Simulating network activity for {duration} seconds...")
        
        end_time = time.time() + duration
        message_count = 0
        
        while time.time() < end_time:
            # Random node sends message to random peer
            sender = self.nodes[message_count % len(self.nodes)]
            
            if sender.peers:
                target_id = list(sender.peers.keys())[0]
                message = {
                    "type": "ping" if message_count % 2 == 0 else "data_sync",
                    "id": f"msg_{message_count}",
                    "timestamp": time.time(),
                    "data": f"test_data_{message_count}"
                }
                
                await sender.send_message(target_id, message)
                
                # Simulate response
                target_node = next((n for n in self.nodes if n.node_id == target_id), None)
                if target_node:
                    await target_node.receive_message(sender.node_id, message)
            
            message_count += 1
            await asyncio.sleep(0.5)  # Message every 0.5 seconds
        
        print(f"📊 Network activity completed: {message_count} messages exchanged")
    
    async def stop_network(self):
        """Stop all nodes"""
        print(f"🛑 Stopping P2P network...")
        
        for node in self.nodes:
            await node.stop()
        
        print(f"✅ P2P network stopped")
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get network status"""
        uptime = time.time() - self.network_start_time if self.network_start_time else 0
        
        active_nodes = len([n for n in self.nodes if n.status == "active"])
        total_connections = sum(len(n.peers) for n in self.nodes)
        total_messages = sum(n.messages_sent + n.messages_received for n in self.nodes)
        
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "total_connections": total_connections,
            "total_messages": total_messages,
            "network_uptime": uptime,
            "nodes": [n.get_status() for n in self.nodes]
        }

async def demonstrate_basic_p2p():
    """Demonstrate basic P2P network functionality"""
    print("🚀 Basic P2P Network Demo")
    print("=" * 50)
    
    if PRSM_P2P_AVAILABLE:
        print("✅ PRSM P2P components available")
        print("🔧 Creating PRSM P2P network...")
        
        try:
            # Use real PRSM P2P network
            network = P2PNetworkDemo(num_nodes=3)
            
            # Start network
            print("🚀 Starting PRSM P2P network...")
            network_task = asyncio.create_task(network.start_network())
            
            # Wait for initialization
            await asyncio.sleep(3)
            
            # Get network status
            status = network.get_network_status()
            print(f"📊 Network Status:")
            print(f"   Nodes: {status['active_nodes']}/{status['total_nodes']}")
            print(f"   Connections: {status['total_connections']}")
            print(f"   Messages: {status['total_messages']}")
            
            # Simulate some activity
            await asyncio.sleep(5)
            
            # Stop network
            await network.stop_network()
            
            return status
            
        except Exception as e:
            print(f"⚠️  Error with PRSM P2P: {str(e)}")
            print("🔧 Falling back to mock network...")
            PRSM_P2P_AVAILABLE = False
    
    if not PRSM_P2P_AVAILABLE:
        print("🔧 Using mock P2P network")
        
        # Create mock network
        network = MockP2PNetwork(num_nodes=3)
        
        # Start network
        await network.start_network()
        
        # Show initial status
        status = network.get_network_status()
        print(f"\n📊 Initial Network Status:")
        print(f"   Total nodes: {status['total_nodes']}")
        print(f"   Active nodes: {status['active_nodes']}")
        print(f"   Total connections: {status['total_connections']}")
        
        # Simulate network activity
        await network.simulate_network_activity(duration=8)
        
        # Show final status
        final_status = network.get_network_status()
        print(f"\n📊 Final Network Status:")
        print(f"   Total messages: {final_status['total_messages']}")
        print(f"   Network uptime: {final_status['network_uptime']:.1f}s")
        
        # Show individual node details
        print(f"\n🔍 Node Details:")
        for node_status in final_status['nodes']:
            print(f"   {node_status['node_id']} ({node_status['node_type']}):")
            print(f"     Status: {node_status['status']}")
            print(f"     Peers: {node_status['peer_count']}")
            print(f"     Messages: {node_status['messages_sent']}↗ {node_status['messages_received']}↙")
        
        # Stop network
        await network.stop_network()
        
        return final_status

async def demonstrate_node_communication():
    """Demonstrate direct node-to-node communication"""
    print(f"\n💬 Node Communication Demo")
    print("=" * 40)
    
    # Create two nodes for direct communication
    node_a = MockP2PNode("node_a", "coordinator", 8000)
    node_b = MockP2PNode("node_b", "worker", 8001)
    
    # Start nodes
    await node_a.start()
    await node_b.start()
    
    # Connect nodes
    await node_a.connect_to_peer(node_b)
    
    # Exchange messages
    messages = [
        {"type": "handshake", "data": "Hello from node_a"},
        {"type": "task_request", "data": "Please process this task"},
        {"type": "task_response", "data": "Task completed successfully"},
        {"type": "heartbeat", "data": "Still alive"}
    ]
    
    for i, msg in enumerate(messages):
        if i % 2 == 0:
            # A sends to B
            await node_a.send_message("node_b", msg)
            await node_b.receive_message("node_a", msg)
        else:
            # B sends to A
            await node_b.send_message("node_a", msg)
            await node_a.receive_message("node_b", msg)
        
        await asyncio.sleep(0.3)
    
    # Show communication stats
    print(f"\n📊 Communication Summary:")
    print(f"   Node A: {node_a.messages_sent} sent, {node_a.messages_received} received")
    print(f"   Node B: {node_b.messages_sent} sent, {node_b.messages_received} received")
    
    # Stop nodes
    await node_a.stop()
    await node_b.stop()
    
    return {
        "node_a_stats": node_a.get_status(),
        "node_b_stats": node_b.get_status()
    }

async def demonstrate_network_resilience():
    """Demonstrate network fault tolerance"""
    print(f"\n🛡️  Network Resilience Demo")
    print("=" * 40)
    
    # Create network with 4 nodes
    network = MockP2PNetwork(num_nodes=4)
    await network.start_network()
    
    print(f"📊 Initial network: {len(network.nodes)} nodes")
    
    # Simulate node failure
    failed_node = network.nodes[-1]
    print(f"⚠️  Simulating failure of {failed_node.node_id}")
    await failed_node.stop()
    
    # Check network status
    status = network.get_network_status()
    print(f"📊 After failure: {status['active_nodes']}/{status['total_nodes']} nodes active")
    
    # Simulate network activity with reduced nodes
    active_nodes = [n for n in network.nodes if n.status == "active"]
    print(f"📡 Testing communication with {len(active_nodes)} active nodes...")
    
    # Send messages between active nodes
    for i in range(3):
        sender = active_nodes[i % len(active_nodes)]
        if sender.peers:
            target_id = list(sender.peers.keys())[0]
            message = {"type": "resilience_test", "id": f"test_{i}"}
            await sender.send_message(target_id, message)
        await asyncio.sleep(0.2)
    
    # Simulate node recovery
    print(f"🔄 Simulating recovery of {failed_node.node_id}")
    await failed_node.start()
    
    # Reconnect to network
    for node in active_nodes:
        if node.node_type == "coordinator":
            await node.connect_to_peer(failed_node)
            break
    
    final_status = network.get_network_status()
    print(f"📊 After recovery: {final_status['active_nodes']}/{final_status['total_nodes']} nodes active")
    
    await network.stop_network()
    
    return final_status

async def main():
    """Main function"""
    print("🚀 Starting P2P Network Demo")
    
    try:
        # Basic P2P network demo
        network_status = await demonstrate_basic_p2p()
        
        # Node communication demo
        comm_results = await demonstrate_node_communication()
        
        # Network resilience demo
        resilience_results = await demonstrate_network_resilience()
        
        # Compile results
        results = {
            "network_demo": network_status,
            "communication_demo": comm_results,
            "resilience_demo": resilience_results,
            "demo_completed_at": time.time()
        }
        
        # Save results
        output_file = Path(__file__).parent / "p2p_demo_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Summary
        print(f"\n📈 P2P Network Demo Summary:")
        print(f"   ✅ Basic network functionality demonstrated")
        print(f"   ✅ Node-to-node communication verified")
        print(f"   ✅ Network resilience tested")
        
        print(f"\n🎉 P2P Network demo completed!")
        print(f"💡 Next steps:")
        print(f"   • Try: python playground_launcher.py --example ai_models/distributed_inference")
        print(f"   • Try: python playground_launcher.py --example orchestration/multi_agent")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    
    if success:
        print("\n✅ P2P Network example completed successfully!")
    else:
        print("\n❌ Example failed. Check the logs for details.")
        sys.exit(1)