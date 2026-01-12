"""
Distributed RLT Network System

Advanced distributed networking framework for RLT (Recursive Learning Technology)
components, enabling decentralized teacher coordination, federated learning,
and distributed knowledge sharing across PRSM network nodes.

Key Features:
- Decentralized RLT teacher coordination
- Federated learning protocols
- Distributed knowledge aggregation
- P2P teacher model sharing
- Network resilience and fault tolerance
- Load balancing across RLT nodes
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from uuid import uuid4
from enum import Enum
import structlog
from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator

logger = structlog.get_logger(__name__)


class NodeStatus(Enum):
    """Status of an RLT network node"""
    ONLINE = "online"
    OFFLINE = "offline"
    SYNCING = "syncing"
    MAINTENANCE = "maintenance"
    OVERLOADED = "overloaded"


class MessageType(Enum):
    """Types of network messages"""
    TEACHER_REQUEST = "teacher_request"
    TEACHER_RESPONSE = "teacher_response"
    KNOWLEDGE_SYNC = "knowledge_sync"
    PERFORMANCE_UPDATE = "performance_update"
    HEALTH_CHECK = "health_check"
    COORDINATION = "coordination"


@dataclass
class RLTNode:
    """Represents a node in the distributed RLT network"""
    node_id: str
    address: str
    port: int
    status: NodeStatus = NodeStatus.OFFLINE
    
    # Capabilities
    available_teachers: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    max_concurrent_requests: int = 10
    
    # Performance metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0
    success_rate: float = 1.0
    
    # Network metrics
    bandwidth: float = 100.0  # Mbps
    latency: float = 10.0     # ms
    reliability: float = 0.99
    
    # Timestamps
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid4())
    
    def is_available(self) -> bool:
        """Check if node is available for requests"""
        return (
            self.status == NodeStatus.ONLINE and
            self.cpu_usage < 90.0 and
            self.memory_usage < 85.0
        )
    
    def calculate_load_score(self) -> float:
        """Calculate node load score for load balancing"""
        return (self.cpu_usage * 0.4 + self.memory_usage * 0.3 + 
                (100 - self.success_rate * 100) * 0.2 + 
                self.response_time * 0.1)


@dataclass
class NetworkMessage:
    """Message structure for network communication"""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl: int = 300  # Time to live in seconds
    visited_nodes: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid4())
        self.visited_nodes.add(self.sender_id)
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > self.ttl


@dataclass
class NetworkTopology:
    """Network topology information"""
    total_nodes: int = 0
    active_nodes: int = 0
    node_types: Dict[str, int] = field(default_factory=dict)
    average_latency: float = 0.0
    network_load: float = 0.0
    fault_tolerance: float = 0.0


class DistributedRLTNetwork:
    """
    Distributed RLT Network Coordination System
    
    Manages a distributed network of RLT nodes, providing:
    - Decentralized teacher discovery and coordination
    - Federated learning orchestration
    - Distributed knowledge aggregation
    - Network resilience and fault tolerance
    - Load balancing and performance optimization
    """
    
    def __init__(self, node_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.node_id = node_id or str(uuid4())
        self.network_id = str(uuid4())
        
        # Network state
        self.nodes: Dict[str, RLTNode] = {}
        self.local_node: Optional[RLTNode] = None
        self.network_topology = NetworkTopology()
        
        # Message handling
        self.message_queue: List[NetworkMessage] = []
        self.message_handlers: Dict[MessageType, callable] = {}
        self.pending_responses: Dict[str, Dict[str, Any]] = {}
        
        # Teacher coordination
        self.teacher_registry: Dict[str, List[str]] = {}  # teacher_type -> [node_ids]
        self.teacher_performance: Dict[str, Dict[str, float]] = {}  # node_id -> {teacher_type: performance}
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        
        # Network configuration
        self.max_message_queue_size = self.config.get('max_message_queue_size', 10000)
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.sync_interval = self.config.get('sync_interval', 60)
        self.max_hops = self.config.get('max_hops', 5)
        
        # Performance tracking
        self.network_metrics: Dict[str, List[float]] = {
            'latency': [],
            'throughput': [],
            'success_rate': [],
            'node_count': []
        }

        # Neuro-Symbolic Orchestrator for verifiable computation
        self.orchestrator = NeuroSymbolicOrchestrator(node_id=self.node_id)
        
        # Initialize message handlers
        self._setup_message_handlers()
        
        logger.info(
            "Distributed RLT Network initialized",
            node_id=self.node_id,
            network_id=self.network_id,
            config=self.config
        )
    
    def _setup_message_handlers(self):
        """Setup message type handlers"""
        self.message_handlers = {
            MessageType.TEACHER_REQUEST: self._handle_teacher_request,
            MessageType.TEACHER_RESPONSE: self._handle_teacher_response,
            MessageType.KNOWLEDGE_SYNC: self._handle_knowledge_sync,
            MessageType.PERFORMANCE_UPDATE: self._handle_performance_update,
            MessageType.HEALTH_CHECK: self._handle_health_check,
            MessageType.COORDINATION: self._handle_coordination
        }
    
    async def initialize_network(
        self,
        local_address: str = "localhost",
        local_port: int = 8080,
        bootstrap_nodes: Optional[List[Tuple[str, int]]] = None
    ):
        """Initialize the local node and join the network"""
        
        # Create local node
        self.local_node = RLTNode(
            node_id=self.node_id,
            address=local_address,
            port=local_port,
            status=NodeStatus.ONLINE
        )
        
        # Register local node
        self.nodes[self.node_id] = self.local_node
        
        # Connect to bootstrap nodes if provided
        if bootstrap_nodes:
            await self._connect_to_bootstrap_nodes(bootstrap_nodes)
        
        # Start background tasks
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._sync_loop())
        asyncio.create_task(self._message_processing_loop())
        
        logger.info(
            "RLT Network node initialized",
            node_id=self.node_id,
            address=local_address,
            port=local_port,
            bootstrap_nodes=len(bootstrap_nodes) if bootstrap_nodes else 0
        )
    
    async def _connect_to_bootstrap_nodes(self, bootstrap_nodes: List[Tuple[str, int]]):
        """Connect to bootstrap nodes to join the network"""
        
        print(f"🌐 Connecting to {len(bootstrap_nodes)} bootstrap nodes...")
        
        for address, port in bootstrap_nodes:
            try:
                # Simulate connection to bootstrap node
                bootstrap_id = f"bootstrap_{address}_{port}"
                bootstrap_node = RLTNode(
                    node_id=bootstrap_id,
                    address=address,
                    port=port,
                    status=NodeStatus.ONLINE
                )
                
                self.nodes[bootstrap_id] = bootstrap_node
                
                # Send coordination message to announce presence
                await self._send_message(NetworkMessage(
                    message_id=str(uuid4()),
                    message_type=MessageType.COORDINATION,
                    sender_id=self.node_id,
                    recipient_id=bootstrap_id,
                    payload={
                        "action": "join_network",
                        "node_info": {
                            "node_id": self.node_id,
                            "address": self.local_node.address,
                            "port": self.local_node.port,
                            "capabilities": self.local_node.available_teachers
                        }
                    }
                ))
                
                print(f"   ✅ Connected to bootstrap node: {address}:{port}")
                
            except Exception as e:
                logger.error(f"Failed to connect to bootstrap node {address}:{port}: {e}")
                print(f"   ❌ Failed to connect to: {address}:{port}")
    
    async def register_teacher(self, teacher_type: str, performance_score: float = 0.8):
        """Register a teacher available on this node"""
        
        if self.local_node:
            if teacher_type not in self.local_node.available_teachers:
                self.local_node.available_teachers.append(teacher_type)
            
            # Update teacher registry
            if teacher_type not in self.teacher_registry:
                self.teacher_registry[teacher_type] = []
            
            if self.node_id not in self.teacher_registry[teacher_type]:
                self.teacher_registry[teacher_type].append(self.node_id)
            
            # Update performance tracking
            if self.node_id not in self.teacher_performance:
                self.teacher_performance[self.node_id] = {}
            self.teacher_performance[self.node_id][teacher_type] = performance_score
            
            # Broadcast teacher availability
            await self._broadcast_message(NetworkMessage(
                message_id=str(uuid4()),
                message_type=MessageType.COORDINATION,
                sender_id=self.node_id,
                recipient_id=None,
                payload={
                    "action": "teacher_registration",
                    "teacher_type": teacher_type,
                    "performance_score": performance_score,
                    "node_id": self.node_id
                }
            ))
            
            logger.info(f"Registered teacher: {teacher_type} with performance {performance_score}")
    
    async def request_teacher(
        self,
        teacher_type: str,
        task_context: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """Request a teacher from the network"""
        
        request_id = str(uuid4())
        
        # Find best available nodes for this teacher type
        candidate_nodes = self._find_teacher_candidates(teacher_type)
        
        if not candidate_nodes:
            logger.warning(f"No nodes available for teacher type: {teacher_type}")
            return None
        
        # Select best node based on load balancing
        selected_node_id = self._select_best_node(candidate_nodes)
        
        # Send teacher request
        request_message = NetworkMessage(
            message_id=request_id,
            message_type=MessageType.TEACHER_REQUEST,
            sender_id=self.node_id,
            recipient_id=selected_node_id,
            payload={
                "teacher_type": teacher_type,
                "task_context": task_context,
                "request_id": request_id
            }
        )
        
        # Track pending request
        self.pending_responses[request_id] = {
            "start_time": time.time(),
            "timeout": timeout,
            "teacher_type": teacher_type,
            "target_node": selected_node_id,
            "task_context": task_context
        }
        
        await self._send_message(request_message)
        
        # Wait for response
        return await self._wait_for_response(request_id, timeout)
    
    def _find_teacher_candidates(self, teacher_type: str) -> List[str]:
        """Find nodes that have the requested teacher type"""
        candidates = []
        
        # Check teacher registry
        if teacher_type in self.teacher_registry:
            for node_id in self.teacher_registry[teacher_type]:
                if node_id in self.nodes and self.nodes[node_id].is_available():
                    candidates.append(node_id)
        
        return candidates
    
    def _select_best_node(self, candidate_nodes: List[str]) -> str:
        """Select the best node based on load balancing"""
        if not candidate_nodes:
            raise ValueError("No candidate nodes available")
        
        # Calculate load scores for each candidate
        node_scores = []
        for node_id in candidate_nodes:
            node = self.nodes[node_id]
            load_score = node.calculate_load_score()
            
            # Consider reliability and latency
            adjusted_score = load_score + (1 - node.reliability) * 50 + node.latency * 0.1
            
            node_scores.append((node_id, adjusted_score))
        
        # Select node with lowest adjusted score (best performance)
        node_scores.sort(key=lambda x: x[1])
        return node_scores[0][0]
    
    async def _wait_for_response(self, request_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        """Wait for response to a request"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if response received
            if request_id in self.pending_responses:
                pending = self.pending_responses[request_id]
                if "response" in pending:
                    response = pending["response"]
                    del self.pending_responses[request_id]
                    return response
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
        
        # Timeout occurred
        if request_id in self.pending_responses:
            del self.pending_responses[request_id]
        
        logger.warning(f"Teacher request {request_id} timed out")
        return None
    
    async def _send_message(self, message: NetworkMessage):
        """Send a message to a specific node or broadcast"""
        
        try:
            # Simulate network transmission
            # In real implementation, this would use actual network protocols
            
            if message.recipient_id and message.recipient_id in self.nodes:
                # Direct message to specific node
                target_node = self.nodes[message.recipient_id]
                
                # Simulate network latency
                await asyncio.sleep(target_node.latency / 1000.0)
                
                # Add to target node's message queue (simulated)
                await self._process_message(message)
                
            elif message.recipient_id is None:
                # Broadcast message
                await self._broadcast_message(message)
            
            logger.debug(f"Sent message {message.message_id} to {message.recipient_id or 'broadcast'}")
            
        except Exception as e:
            logger.error(f"Failed to send message {message.message_id}: {e}")
    
    async def _broadcast_message(self, message: NetworkMessage):
        """Broadcast message to all nodes in network"""
        
        # Mark local node as visited
        message.visited_nodes.add(self.node_id)
        
        for node_id, node in self.nodes.items():
            if node_id != self.node_id and node_id not in message.visited_nodes and node.status == NodeStatus.ONLINE:
                message_copy = NetworkMessage(
                    message_id=message.message_id,
                    message_type=message.message_type,
                    sender_id=message.sender_id,
                    recipient_id=node_id,
                    payload=message.payload.copy(),
                    timestamp=message.timestamp,
                    ttl=message.ttl,
                    visited_nodes=message.visited_nodes.copy()
                )
                await self._send_message(message_copy)
    
    async def _process_message(self, message: NetworkMessage):
        """Process incoming message"""
        
        if message.is_expired():
            logger.debug(f"Dropping expired message {message.message_id}")
            return
        
        # Get handler for message type
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error processing message {message.message_id}: {e}")
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
    
    async def _handle_teacher_request(self, message: NetworkMessage):
        """Handle incoming teacher request"""
        
        payload = message.payload
        teacher_type = payload.get("teacher_type")
        task_context = payload.get("task_context", {})
        request_id = payload.get("request_id")
        
        # Check if we have the requested teacher
        if (self.local_node and 
            teacher_type in self.local_node.available_teachers and
            self.local_node.is_available()):
            
            # Simulate teacher execution
            start_time = time.time()
            
            # Generate simulated response
            teacher_response = await self._execute_teacher_locally(teacher_type, task_context)
            
            execution_time = time.time() - start_time
            
            # Send response back
            response_message = NetworkMessage(
                message_id=str(uuid4()),
                message_type=MessageType.TEACHER_RESPONSE,
                sender_id=self.node_id,
                recipient_id=message.sender_id,
                payload={
                    "request_id": request_id,
                    "teacher_type": teacher_type,
                    "response": teacher_response,
                    "execution_time": execution_time,
                    "node_id": self.node_id
                }
            )
            
            await self._send_message(response_message)
            
            logger.debug(f"Processed teacher request {request_id} for {teacher_type}")
        
        else:
            # Forward request to other nodes if possible
            await self._forward_teacher_request(message)
    
    async def _forward_teacher_request(self, message: NetworkMessage):
        """Forward a teacher request to other nodes (Simplified)"""
        logger.debug(f"Forwarding teacher request {message.message_id}")
        # In a real implementation, this would use a routing table or DHT
        # Mark sender as visited so we don't send it back
        message.visited_nodes.add(message.sender_id)
        await self._broadcast_message(message)

    async def _handle_teacher_response(self, message: NetworkMessage):
        """Handle teacher response with verification"""
        
        payload = message.payload
        request_id = payload.get("request_id")
        
        if request_id in self.pending_responses:
            # VERIFICATION: Solve the Oracle Problem
            response_data = payload.get("response", {})
            
            # Reconstruct metadata for verification if it's missing but we have it locally
            if "metadata" not in response_data:
                response_data["metadata"] = {}
            
            # The validator knows what they asked for
            local_request_info = self.pending_responses[request_id]
            task_context = local_request_info.get("task_context", {})
            
            if "query" not in response_data["metadata"]:
                response_data["metadata"]["query"] = task_context.get("query", "")
            
            # The validator must use the seed reported by the worker for this specific task
            worker_seed = response_data["metadata"].get("seed", self.orchestrator.seed)

            is_valid = await self.orchestrator.verify_remote_node(response_data, worker_seed)
            
            if not is_valid:
                logger.error(
                    "Oracle Verification Failed!", 
                    request_id=request_id, 
                    node_id=message.sender_id
                )
                payload["verified"] = False
            else:
                payload["verified"] = True
                logger.info(f"Verified computation for request {request_id}")

            # Store response
            self.pending_responses[request_id]["response"] = payload
            logger.debug(f"Received response for request {request_id}")
    
    async def _handle_knowledge_sync(self, message: NetworkMessage):
        """Handle knowledge synchronization"""
        payload = message.payload
        # Implement knowledge sync logic
        logger.debug("Processing knowledge sync message")
    
    async def _handle_performance_update(self, message: NetworkMessage):
        """Handle performance update from nodes"""
        payload = message.payload
        node_id = payload.get("node_id")
        
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Update performance metrics
            if "cpu_usage" in payload:
                node.cpu_usage = payload["cpu_usage"]
            if "memory_usage" in payload:
                node.memory_usage = payload["memory_usage"]
            if "response_time" in payload:
                node.response_time = payload["response_time"]
            
            node.last_seen = datetime.now(timezone.utc)
            
            logger.debug(f"Updated performance metrics for node {node_id}")
    
    async def _handle_health_check(self, message: NetworkMessage):
        """Handle health check ping"""
        # Respond to health check
        response = NetworkMessage(
            message_id=str(uuid4()),
            message_type=MessageType.HEALTH_CHECK,
            sender_id=self.node_id,
            recipient_id=message.sender_id,
            payload={
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "load": self.local_node.calculate_load_score() if self.local_node else 0
            }
        )
        await self._send_message(response)
    
    async def _handle_coordination(self, message: NetworkMessage):
        """Handle coordination messages"""
        payload = message.payload
        action = payload.get("action")
        
        if action == "join_network":
            await self._handle_node_join(message)
        elif action == "teacher_registration":
            await self._handle_teacher_registration(message)
        elif action == "node_leave":
            await self._handle_node_leave(message)
    
    async def _handle_node_join(self, message: NetworkMessage):
        """Handle node joining the network"""
        node_info = message.payload.get("node_info", {})
        node_id = node_info.get("node_id")
        
        if node_id and node_id not in self.nodes:
            new_node = RLTNode(
                node_id=node_id,
                address=node_info.get("address", "unknown"),
                port=node_info.get("port", 0),
                status=NodeStatus.ONLINE,
                available_teachers=node_info.get("capabilities", [])
            )
            
            self.nodes[node_id] = new_node
            logger.info(f"Node {node_id} joined the network")
    
    async def _handle_teacher_registration(self, message: NetworkMessage):
        """Handle teacher registration from remote node"""
        payload = message.payload
        teacher_type = payload.get("teacher_type")
        node_id = payload.get("node_id")
        performance_score = payload.get("performance_score", 0.8)
        
        if teacher_type and node_id:
            # Update teacher registry
            if teacher_type not in self.teacher_registry:
                self.teacher_registry[teacher_type] = []
            
            if node_id not in self.teacher_registry[teacher_type]:
                self.teacher_registry[teacher_type].append(node_id)
            
            # Update performance tracking
            if node_id not in self.teacher_performance:
                self.teacher_performance[node_id] = {}
            self.teacher_performance[node_id][teacher_type] = performance_score
            
            logger.debug(f"Registered teacher {teacher_type} from node {node_id}")
    
    async def _execute_teacher_locally(self, teacher_type: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute teacher using Neuro-Symbolic Orchestrator (Verifiable)"""
        
        query = task_context.get("query", f"Task for {teacher_type}")
        context_str = task_context.get("context", "")
        
        # Execute via orchestrator (System 1/2 layered inference)
        solution = await self.orchestrator.solve_task(query, context_str)
        
        return {
            "teacher_type": teacher_type,
            "output": solution["output"],
            "input_hash": solution["input_hash"],
            "verification_hash": solution["verification_hash"],
            "trace": solution["trace"],
            "reward": solution["reward"],
            "metadata": {
                "processing_node": self.node_id,
                "mode": solution["mode"]
            }
        }

    async def _verify_teacher_response(self, response_payload: Dict[str, Any]) -> bool:
        """Verify the validity of a remote teacher's response using the orchestrator"""
        # This handles the Oracle Problem by verifying the Proof of Useful Work
        return await self.orchestrator.verify_remote_node(
            task_data=response_payload,
            seed=self.orchestrator.seed
        )
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _sync_loop(self):
        """Background synchronization loop"""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                await self._synchronize_network_state()
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
    
    async def _message_processing_loop(self):
        """Background message processing loop"""
        while True:
            try:
                # Process queued messages
                if self.message_queue:
                    message = self.message_queue.pop(0)
                    await self._process_message(message)
                else:
                    await asyncio.sleep(0.1)  # Small delay when no messages
            except Exception as e:
                logger.error(f"Message processing loop error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on network nodes"""
        
        current_time = datetime.now(timezone.utc)
        offline_nodes = []
        
        for node_id, node in self.nodes.items():
            if node_id == self.node_id:  # Skip self
                continue
            
            # Check if node has been seen recently
            time_since_seen = (current_time - node.last_seen).total_seconds()
            
            if time_since_seen > self.health_check_interval * 3:  # 3x interval = offline
                if node.status != NodeStatus.OFFLINE:
                    node.status = NodeStatus.OFFLINE
                    offline_nodes.append(node_id)
                    logger.warning(f"Node {node_id} marked as offline")
        
        # Update network topology
        self._update_network_topology()
    
    async def _synchronize_network_state(self):
        """Synchronize network state with other nodes"""
        
        # Send performance update
        if self.local_node:
            perf_message = NetworkMessage(
                message_id=str(uuid4()),
                message_type=MessageType.PERFORMANCE_UPDATE,
                sender_id=self.node_id,
                recipient_id=None,  # Broadcast
                payload={
                    "node_id": self.node_id,
                    "cpu_usage": self.local_node.cpu_usage,
                    "memory_usage": self.local_node.memory_usage,
                    "response_time": self.local_node.response_time,
                    "available_teachers": self.local_node.available_teachers
                }
            )
            await self._broadcast_message(perf_message)
    
    def _update_network_topology(self):
        """Update network topology information"""
        
        total_nodes = len(self.nodes)
        active_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.ONLINE])
        
        # Calculate average latency
        latencies = [n.latency for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        # Calculate network load
        loads = [n.calculate_load_score() for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
        network_load = sum(loads) / len(loads) if loads else 0
        
        # Update topology
        self.network_topology = NetworkTopology(
            total_nodes=total_nodes,
            active_nodes=active_nodes,
            average_latency=avg_latency,
            network_load=network_load,
            fault_tolerance=active_nodes / max(total_nodes, 1)
        )
        
        # Track metrics
        self.network_metrics['node_count'].append(active_nodes)
        self.network_metrics['latency'].append(avg_latency)
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status"""
        
        return {
            "network_id": self.network_id,
            "local_node_id": self.node_id,
            "topology": {
                "total_nodes": self.network_topology.total_nodes,
                "active_nodes": self.network_topology.active_nodes,
                "average_latency": self.network_topology.average_latency,
                "network_load": self.network_topology.network_load,
                "fault_tolerance": self.network_topology.fault_tolerance
            },
            "teacher_registry": {
                teacher_type: len(nodes) 
                for teacher_type, nodes in self.teacher_registry.items()
            },
            "message_queue_size": len(self.message_queue),
            "pending_requests": len(self.pending_responses)
        }


# Factory function for easy instantiation
def create_distributed_rlt_network(
    node_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> DistributedRLTNetwork:
    """Create and return a Distributed RLT Network instance"""
    return DistributedRLTNetwork(node_id, config)


# Default configuration
DEFAULT_NETWORK_CONFIG = {
    "max_message_queue_size": 10000,
    "health_check_interval": 30,
    "sync_interval": 60,
    "max_hops": 5,
    "enable_detailed_logging": True
}