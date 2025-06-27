#!/usr/bin/env python3
"""
PRSM P2P Network Demo
Minimal proof-of-concept demonstrating P2P architecture with:
- Node discovery and registration
- Secure message/file sharing
- Basic consensus simulation
- Real-time monitoring dashboard

This demo simulates 2-3 nodes using async Python with WebSocket-like communication
to demonstrate the core P2P capabilities that will be expanded in production.
"""

import asyncio
import json
import hashlib
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NodeInfo:
    """Information about a P2P node"""
    node_id: str
    node_type: str  # "coordinator", "worker", "validator"
    address: str    # "host:port" 
    public_key: str
    reputation_score: float
    capabilities: List[str]
    status: str  # "active", "inactive", "syncing"
    last_seen: float
    
@dataclass 
class Message:
    """P2P message structure"""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: str  # "discovery", "handshake", "consensus", "data"
    payload: Dict[str, Any]
    timestamp: float
    signature: str
    
@dataclass
class ConsensusProposal:
    """Consensus proposal for validation"""
    proposal_id: str
    proposer_id: str
    proposal_type: str  # "model_hash", "timestamp_sync", "reputation_update"
    data: Dict[str, Any]
    timestamp: float
    required_votes: int
    votes: Dict[str, bool]  # node_id -> vote
    status: str  # "pending", "approved", "rejected"

class P2PNode:
    """
    Simulated P2P node with discovery, messaging, and consensus capabilities
    """
    
    def __init__(self, node_type: str = "worker", port: int = 8000):
        self.node_info = NodeInfo(
            node_id=str(uuid.uuid4())[:8],
            node_type=node_type,
            address=f"localhost:{port}",
            public_key=self._generate_key(),
            reputation_score=50.0,
            capabilities=["data_storage", "model_execution", "consensus_voting"],
            status="inactive",
            last_seen=time.time()
        )
        
        # Network state
        self.known_peers: Dict[str, NodeInfo] = {}
        self.message_history: List[Message] = []
        self.pending_consensus: Dict[str, ConsensusProposal] = {}
        
        # Networking simulation
        self.message_queue = asyncio.Queue()
        self.is_running = False
        
        # Performance metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.consensus_participations = 0
        self.failed_connections = 0
        
        logger.info(f"P2P Node {self.node_info.node_id} ({node_type}) initialized on {self.node_info.address}")
    
    def _generate_key(self) -> str:
        """Generate a simulated public key"""
        return hashlib.sha256(f"{uuid.uuid4()}{time.time()}".encode()).hexdigest()[:32]
    
    def _sign_message(self, message_data: str) -> str:
        """Simulate message signing with private key"""
        return hashlib.sha256(f"{self.node_info.public_key}{message_data}".encode()).hexdigest()[:16]
    
    def _verify_signature(self, message: Message, sender_public_key: str) -> bool:
        """Simulate signature verification"""
        expected_sig = hashlib.sha256(f"{sender_public_key}{json.dumps(message.payload)}".encode()).hexdigest()[:16]
        return message.signature == expected_sig
    
    async def start(self):
        """Start the P2P node"""
        self.node_info.status = "active"
        self.node_info.last_seen = time.time()
        self.is_running = True
        
        logger.info(f"Node {self.node_info.node_id} starting...")
        
        # Start background tasks
        await asyncio.gather(
            self._message_processor(),
            self._heartbeat_sender(),
            self._consensus_monitor()
        )
    
    async def stop(self):
        """Stop the P2P node"""
        self.is_running = False
        self.node_info.status = "inactive"
        logger.info(f"Node {self.node_info.node_id} stopped")
    
    async def discover_peers(self, bootstrap_nodes: List['P2PNode']):
        """Discover and connect to other nodes"""
        logger.info(f"Node {self.node_info.node_id} discovering peers...")
        
        for bootstrap_node in bootstrap_nodes:
            if bootstrap_node.node_info.node_id != self.node_info.node_id:
                await self._send_discovery_message(bootstrap_node)
    
    async def _send_discovery_message(self, target_node: 'P2PNode'):
        """Send discovery message to a potential peer"""
        discovery_payload = {
            "node_info": asdict(self.node_info),
            "request_type": "peer_discovery",
            "known_peers": list(self.known_peers.keys())
        }
        
        message = Message(
            message_id=str(uuid.uuid4())[:8],
            sender_id=self.node_info.node_id,
            receiver_id=target_node.node_info.node_id,
            message_type="discovery",
            payload=discovery_payload,
            timestamp=time.time(),
            signature=self._sign_message(json.dumps(discovery_payload))
        )
        
        await self._deliver_message(message, target_node)
        self.messages_sent += 1
    
    async def _deliver_message(self, message: Message, target_node: 'P2PNode'):
        """Simulate message delivery to another node"""
        # Simulate network delay
        await asyncio.sleep(0.1 + (0.05 * len(self.known_peers)))  # Realistic network delay
        
        # Simulate message loss (5% chance)
        if asyncio.get_event_loop().time() % 20 < 1:  # 5% message loss simulation
            logger.warning(f"Message {message.message_id} lost in transit")
            self.failed_connections += 1
            return
        
        await target_node.message_queue.put(message)
    
    async def _message_processor(self):
        """Process incoming messages"""
        while self.is_running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._handle_message(message)
                self.messages_received += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processing error in {self.node_info.node_id}: {e}")
    
    async def _handle_message(self, message: Message):
        """Handle incoming message based on type"""
        self.message_history.append(message)
        
        if message.message_type == "discovery":
            await self._handle_discovery_message(message)
        elif message.message_type == "handshake":
            await self._handle_handshake_message(message)
        elif message.message_type == "consensus":
            await self._handle_consensus_message(message)
        elif message.message_type == "data":
            await self._handle_data_message(message)
        elif message.message_type == "heartbeat":
            await self._handle_heartbeat_message(message)
        
        logger.debug(f"Node {self.node_info.node_id} processed {message.message_type} message from {message.sender_id}")
    
    async def _handle_discovery_message(self, message: Message):
        """Handle peer discovery message"""
        sender_info = NodeInfo(**message.payload["node_info"])
        
        # Add sender to known peers
        self.known_peers[sender_info.node_id] = sender_info
        
        # Send handshake response
        handshake_payload = {
            "node_info": asdict(self.node_info),
            "response_type": "handshake_response",
            "network_status": {
                "known_peers": len(self.known_peers),
                "uptime": time.time() - self.node_info.last_seen,
                "consensus_active": len(self.pending_consensus)
            }
        }
        
        response = Message(
            message_id=str(uuid.uuid4())[:8],
            sender_id=self.node_info.node_id,
            receiver_id=message.sender_id,
            message_type="handshake",
            payload=handshake_payload,
            timestamp=time.time(),
            signature=self._sign_message(json.dumps(handshake_payload))
        )
        
        # Find sender node and deliver response
        for peer_id, peer_info in self.known_peers.items():
            if peer_id == message.sender_id:
                # In real implementation, would send over network
                logger.info(f"Node {self.node_info.node_id} completed handshake with {peer_id}")
                break
    
    async def _handle_handshake_message(self, message: Message):
        """Handle handshake response"""
        sender_info = NodeInfo(**message.payload["node_info"])
        self.known_peers[sender_info.node_id] = sender_info
        
        logger.info(f"Node {self.node_info.node_id} established connection with {sender_info.node_id}")
    
    async def _handle_consensus_message(self, message: Message):
        """Handle consensus-related message"""
        if message.payload.get("action") == "propose":
            proposal = ConsensusProposal(**message.payload["proposal"])
            self.pending_consensus[proposal.proposal_id] = proposal
            
            # Automatically vote (simplified logic)
            vote = self._evaluate_consensus_proposal(proposal)
            await self._cast_consensus_vote(proposal.proposal_id, vote)
            
        elif message.payload.get("action") == "vote":
            proposal_id = message.payload["proposal_id"]
            vote = message.payload["vote"]
            
            if proposal_id in self.pending_consensus:
                self.pending_consensus[proposal_id].votes[message.sender_id] = vote
                await self._check_consensus_completion(proposal_id)
    
    def _evaluate_consensus_proposal(self, proposal: ConsensusProposal) -> bool:
        """Evaluate consensus proposal and decide vote"""
        # Simplified voting logic
        if proposal.proposal_type == "timestamp_sync":
            # Always approve timestamp sync
            return True
        elif proposal.proposal_type == "model_hash":
            # Check hash validity (simplified)
            return len(proposal.data.get("hash", "")) == 64
        elif proposal.proposal_type == "reputation_update":
            # Approve if reputation change is reasonable
            delta = proposal.data.get("reputation_delta", 0)
            return -10 <= delta <= 10
        
        return False
    
    async def _cast_consensus_vote(self, proposal_id: str, vote: bool):
        """Cast vote on consensus proposal"""
        self.consensus_participations += 1
        logger.info(f"Node {self.node_info.node_id} voted {vote} on proposal {proposal_id}")
    
    async def _check_consensus_completion(self, proposal_id: str):
        """Check if consensus proposal has enough votes"""
        proposal = self.pending_consensus.get(proposal_id)
        if not proposal:
            return
        
        if len(proposal.votes) >= proposal.required_votes:
            # Count votes
            approve_votes = sum(1 for vote in proposal.votes.values() if vote)
            total_votes = len(proposal.votes)
            
            if approve_votes > total_votes / 2:
                proposal.status = "approved"
                logger.info(f"Consensus proposal {proposal_id} APPROVED ({approve_votes}/{total_votes})")
            else:
                proposal.status = "rejected"
                logger.info(f"Consensus proposal {proposal_id} REJECTED ({approve_votes}/{total_votes})")
    
    async def _handle_data_message(self, message: Message):
        """Handle data sharing message"""
        data_type = message.payload.get("data_type")
        data_hash = message.payload.get("hash")
        
        logger.info(f"Node {self.node_info.node_id} received {data_type} data with hash {data_hash[:8]}...")
        
        # Simulate data validation
        is_valid = self._validate_data(message.payload)
        
        if is_valid:
            # Update reputation of sender
            if message.sender_id in self.known_peers:
                self.known_peers[message.sender_id].reputation_score += 0.5
            logger.info(f"Data from {message.sender_id} validated successfully")
        else:
            logger.warning(f"Invalid data received from {message.sender_id}")
    
    def _validate_data(self, payload: Dict[str, Any]) -> bool:
        """Validate received data"""
        # Simplified validation
        required_fields = ["data_type", "hash", "size"]
        return all(field in payload for field in required_fields)
    
    async def _handle_heartbeat_message(self, message: Message):
        """Handle heartbeat message"""
        if message.sender_id in self.known_peers:
            self.known_peers[message.sender_id].last_seen = time.time()
            self.known_peers[message.sender_id].status = "active"
    
    async def _heartbeat_sender(self):
        """Send periodic heartbeat messages"""
        while self.is_running:
            await asyncio.sleep(30)  # 30 second heartbeat
            
            heartbeat_payload = {
                "status": self.node_info.status,
                "metrics": {
                    "messages_sent": self.messages_sent,
                    "messages_received": self.messages_received,
                    "known_peers": len(self.known_peers),
                    "reputation": self.node_info.reputation_score
                }
            }
            
            # Broadcast heartbeat to all known peers
            for peer_id in self.known_peers:
                heartbeat = Message(
                    message_id=str(uuid.uuid4())[:8],
                    sender_id=self.node_info.node_id,
                    receiver_id=peer_id,
                    message_type="heartbeat",
                    payload=heartbeat_payload,
                    timestamp=time.time(),
                    signature=self._sign_message(json.dumps(heartbeat_payload))
                )
                # In real implementation, would send over network
    
    async def _consensus_monitor(self):
        """Monitor and initiate consensus proposals"""
        while self.is_running:
            await asyncio.sleep(60)  # Check every minute
            
            # Propose timestamp synchronization
            if self.node_info.node_type == "coordinator":
                await self._propose_timestamp_sync()
    
    async def _propose_timestamp_sync(self):
        """Propose timestamp synchronization consensus"""
        proposal = ConsensusProposal(
            proposal_id=str(uuid.uuid4())[:8],
            proposer_id=self.node_info.node_id,
            proposal_type="timestamp_sync",
            data={"current_timestamp": time.time()},
            timestamp=time.time(),
            required_votes=min(3, len(self.known_peers) + 1),
            votes={},
            status="pending"
        )
        
        self.pending_consensus[proposal.proposal_id] = proposal
        logger.info(f"Node {self.node_info.node_id} proposed timestamp sync {proposal.proposal_id}")
    
    async def share_file(self, file_data: bytes, file_name: str, target_nodes: List[str]):
        """Share file data with specific nodes"""
        file_hash = hashlib.sha256(file_data).hexdigest()
        
        file_payload = {
            "data_type": "file",
            "file_name": file_name,
            "hash": file_hash,
            "size": len(file_data),
            "chunks": 1,  # Simplified - single chunk
            "data": file_data.hex()  # In real implementation, would use IPFS
        }
        
        for target_id in target_nodes:
            if target_id in self.known_peers:
                message = Message(
                    message_id=str(uuid.uuid4())[:8],
                    sender_id=self.node_info.node_id,
                    receiver_id=target_id,
                    message_type="data",
                    payload=file_payload,
                    timestamp=time.time(),
                    signature=self._sign_message(json.dumps({k: v for k, v in file_payload.items() if k != "data"}))
                )
                
                logger.info(f"Node {self.node_info.node_id} sharing file {file_name} with {target_id}")
                self.messages_sent += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get current node status and metrics"""
        return {
            "node_info": asdict(self.node_info),
            "network_metrics": {
                "known_peers": len(self.known_peers),
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "consensus_participations": self.consensus_participations,
                "failed_connections": self.failed_connections,
                "pending_consensus": len(self.pending_consensus)
            },
            "peers": {peer_id: asdict(peer_info) for peer_id, peer_info in self.known_peers.items()},
            "recent_messages": [
                {
                    "id": msg.message_id,
                    "type": msg.message_type,
                    "sender": msg.sender_id,
                    "timestamp": msg.timestamp
                }
                for msg in self.message_history[-10:]  # Last 10 messages
            ]
        }

class P2PNetworkDemo:
    """
    Demonstration of P2P network with multiple nodes
    """
    
    def __init__(self, num_nodes: int = 3):
        self.nodes: List[P2PNode] = []
        self.network_logs: List[Dict[str, Any]] = []
        self.is_running = False
        
        # Create nodes with different types
        node_types = ["coordinator", "worker", "validator"]
        for i in range(num_nodes):
            node_type = node_types[i % len(node_types)]
            port = 8000 + i
            node = P2PNode(node_type=node_type, port=port)
            self.nodes.append(node)
        
        logger.info(f"P2P Network Demo initialized with {num_nodes} nodes")
    
    async def start_network(self):
        """Start all nodes and establish P2P connections"""
        logger.info("Starting P2P network demo...")
        self.is_running = True
        
        # Start all nodes
        start_tasks = []
        for node in self.nodes:
            start_tasks.append(asyncio.create_task(self._start_node(node)))
        
        # Wait a bit for nodes to initialize
        await asyncio.sleep(1)
        
        # Perform peer discovery
        await self._perform_peer_discovery()
        
        # Start network monitoring
        monitor_task = asyncio.create_task(self._network_monitor())
        
        # Wait for all tasks
        await asyncio.gather(*start_tasks, monitor_task, return_exceptions=True)
    
    async def _start_node(self, node: P2PNode):
        """Start individual node"""
        try:
            await node.start()
        except Exception as e:
            logger.error(f"Error starting node {node.node_info.node_id}: {e}")
    
    async def _perform_peer_discovery(self):
        """Perform peer discovery across all nodes"""
        logger.info("Performing peer discovery...")
        
        # Each node discovers others
        for i, node in enumerate(self.nodes):
            other_nodes = [n for j, n in enumerate(self.nodes) if j != i]
            await node.discover_peers(other_nodes)
            await asyncio.sleep(0.5)  # Stagger discovery
    
    async def _network_monitor(self):
        """Monitor network status and log metrics"""
        while self.is_running:
            await asyncio.sleep(10)  # Monitor every 10 seconds
            
            network_status = self.get_network_status()
            self.network_logs.append({
                "timestamp": time.time(),
                "status": network_status
            })
            
            # Log summary
            active_nodes = sum(1 for node in self.nodes if node.node_info.status == "active")
            total_messages = sum(node.messages_sent + node.messages_received for node in self.nodes)
            
            logger.info(f"Network Status: {active_nodes}/{len(self.nodes)} nodes active, {total_messages} total messages")
    
    async def demonstrate_consensus(self):
        """Demonstrate consensus mechanism"""
        logger.info("Demonstrating consensus mechanism...")
        
        # Find coordinator node
        coordinator = next((node for node in self.nodes if node.node_info.node_type == "coordinator"), None)
        if not coordinator:
            logger.error("No coordinator node found for consensus demo")
            return
        
        # Propose model hash validation
        proposal = ConsensusProposal(
            proposal_id=str(uuid.uuid4())[:8],
            proposer_id=coordinator.node_info.node_id,
            proposal_type="model_hash",
            data={
                "model_name": "demo_model_v1",
                "hash": hashlib.sha256(b"model_data_demo").hexdigest(),
                "size": 1024
            },
            timestamp=time.time(),
            required_votes=len(self.nodes),
            votes={},
            status="pending"
        )
        
        coordinator.pending_consensus[proposal.proposal_id] = proposal
        
        # Broadcast consensus proposal to all nodes
        consensus_payload = {
            "action": "propose",
            "proposal": asdict(proposal)
        }
        
        for node in self.nodes:
            if node.node_info.node_id != coordinator.node_info.node_id:
                message = Message(
                    message_id=str(uuid.uuid4())[:8],
                    sender_id=coordinator.node_info.node_id,
                    receiver_id=node.node_info.node_id,
                    message_type="consensus",
                    payload=consensus_payload,
                    timestamp=time.time(),
                    signature=coordinator._sign_message(json.dumps(consensus_payload))
                )
                
                await coordinator._deliver_message(message, node)
        
        # Wait for consensus completion
        await asyncio.sleep(5)
        
        # Check results
        final_proposal = coordinator.pending_consensus.get(proposal.proposal_id)
        if final_proposal:
            logger.info(f"Consensus result: {final_proposal.status} ({len(final_proposal.votes)} votes)")
    
    async def demonstrate_file_sharing(self):
        """Demonstrate secure file sharing"""
        logger.info("Demonstrating file sharing...")
        
        # Create demo file
        demo_file = b"This is a demo model file for P2P sharing test"
        file_name = "demo_model.json"
        
        # First node shares file with others
        sender = self.nodes[0]
        target_ids = [node.node_info.node_id for node in self.nodes[1:]]
        
        await sender.share_file(demo_file, file_name, target_ids)
        
        # Wait for file sharing to complete
        await asyncio.sleep(2)
        
        logger.info(f"File sharing demo completed: {file_name} shared with {len(target_ids)} nodes")
    
    async def simulate_node_failure(self):
        """Simulate node failure and recovery"""
        logger.info("Simulating node failure and recovery...")
        
        if len(self.nodes) < 2:
            logger.warning("Need at least 2 nodes for failure simulation")
            return
        
        # Stop one node
        failed_node = self.nodes[-1]
        await failed_node.stop()
        logger.info(f"Node {failed_node.node_info.node_id} failed")
        
        # Wait for network to detect failure
        await asyncio.sleep(5)
        
        # Restart node
        failed_node.node_info.status = "active"
        failed_node.is_running = True
        
        # Reconnect to network
        other_nodes = [n for n in self.nodes if n.node_info.node_id != failed_node.node_info.node_id]
        await failed_node.discover_peers(other_nodes)
        
        logger.info(f"Node {failed_node.node_info.node_id} recovered and rejoined network")
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": sum(1 for node in self.nodes if node.node_info.status == "active"),
            "total_connections": sum(len(node.known_peers) for node in self.nodes),
            "total_messages": sum(node.messages_sent + node.messages_received for node in self.nodes),
            "consensus_proposals": sum(len(node.pending_consensus) for node in self.nodes),
            "nodes": [node.get_status() for node in self.nodes]
        }
    
    async def stop_network(self):
        """Stop all nodes and cleanup"""
        logger.info("Stopping P2P network...")
        self.is_running = False
        
        stop_tasks = []
        for node in self.nodes:
            stop_tasks.append(asyncio.create_task(node.stop()))
        
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        logger.info("P2P network stopped")

async def run_p2p_demo():
    """Run complete P2P network demonstration"""
    print("🚀 PRSM P2P Network Demo Starting...")
    print("=" * 50)
    
    # Create network with 3 nodes
    network = P2PNetworkDemo(num_nodes=3)
    
    try:
        # Start network in background
        network_task = asyncio.create_task(network.start_network())
        
        # Wait for initial setup
        await asyncio.sleep(3)
        
        print("\n📊 Initial Network Status:")
        status = network.get_network_status()
        print(f"Nodes: {status['active_nodes']}/{status['total_nodes']} active")
        print(f"Connections: {status['total_connections']}")
        print(f"Messages: {status['total_messages']}")
        
        # Demonstrate consensus
        print("\n🤝 Demonstrating Consensus Mechanism...")
        await network.demonstrate_consensus()
        
        # Demonstrate file sharing
        print("\n📁 Demonstrating File Sharing...")
        await network.demonstrate_file_sharing()
        
        # Simulate node failure
        print("\n⚠️ Simulating Node Failure and Recovery...")
        await network.simulate_node_failure()
        
        # Final status
        await asyncio.sleep(2)
        print("\n📈 Final Network Status:")
        final_status = network.get_network_status()
        print(f"Nodes: {final_status['active_nodes']}/{final_status['total_nodes']} active")
        print(f"Total Connections: {final_status['total_connections']}")
        print(f"Total Messages: {final_status['total_messages']}")
        print(f"Consensus Proposals: {final_status['consensus_proposals']}")
        
        # Show individual node details
        print("\n🔍 Node Details:")
        for i, node_status in enumerate(final_status['nodes']):
            node_info = node_status['node_info']
            metrics = node_status['network_metrics']
            print(f"  Node {i+1} ({node_info['node_id']}): {node_info['node_type']}")
            print(f"    Status: {node_info['status']}")
            print(f"    Peers: {metrics['known_peers']}")
            print(f"    Messages: {metrics['messages_sent']}↗ {metrics['messages_received']}↙")
            print(f"    Reputation: {node_info['reputation_score']:.1f}")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        
    finally:
        # Cleanup
        await network.stop_network()
        print("\n✅ P2P Network Demo Complete!")

if __name__ == "__main__":
    asyncio.run(run_p2p_demo())