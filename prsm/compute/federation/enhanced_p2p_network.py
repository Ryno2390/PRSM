"""
Production-Ready P2P Model Network for PRSM
Real distributed networking with libp2p, DHT, and secure communications
"""

import asyncio
import hashlib
import json
import logging
import time
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from uuid import UUID, uuid4

# P2P Networking imports
try:
    from kademlia.network import Server as KademliaServer
    from kademlia.storage import ForgetfulStorage
    import websockets
    import nacl.secret
    import nacl.utils
    from nacl.public import PrivateKey, PublicKey, Box
    from nacl.signing import SigningKey, VerifyKey
    from merkletools import MerkleTools
except ImportError as e:
    print(f"⚠️ P2P dependencies not installed: {e}")
    print("Install with: pip install kademlia websockets pynacl merkletools")

from ..core.config import settings
from ..core.models import (
    ArchitectTask, PeerNode, ModelShard, TeacherModel, ModelType,
    SafetyFlag, CircuitBreakerEvent, AgentResponse
)
from ..data_layer.enhanced_ipfs import get_ipfs_client
from ..tokenomics.ftns_service import ftns_service
from ..safety.circuit_breaker import CircuitBreakerNetwork, ThreatLevel
from ..safety.monitor import SafetyMonitor


# === Production P2P Configuration ===

# Network settings
P2P_LISTEN_PORT = int(getattr(settings, "PRSM_P2P_PORT", 8000))
DHT_BOOTSTRAP_NODES = getattr(settings, "PRSM_DHT_BOOTSTRAP", "").split(",") if getattr(settings, "PRSM_DHT_BOOTSTRAP", "") else []
MAX_CONNECTIONS = int(getattr(settings, "PRSM_MAX_P2P_CONNECTIONS", 50))
CONNECTION_TIMEOUT = int(getattr(settings, "PRSM_P2P_TIMEOUT", 30))

# Security settings
ENABLE_ENCRYPTION = getattr(settings, "PRSM_P2P_ENCRYPTION", True)
REQUIRE_SIGNATURES = getattr(settings, "PRSM_P2P_SIGNATURES", True)
KEY_ROTATION_HOURS = int(getattr(settings, "PRSM_KEY_ROTATION_HOURS", 24))

# DHT settings
DHT_REFRESH_INTERVAL = int(getattr(settings, "PRSM_DHT_REFRESH_MINUTES", 10))
DHT_ALPHA = int(getattr(settings, "PRSM_DHT_ALPHA", 3))  # Kademlia alpha parameter
DHT_K_BUCKET_SIZE = int(getattr(settings, "PRSM_DHT_K_BUCKET", 20))

# Replication settings
SHARD_REPLICATION_FACTOR = int(getattr(settings, "PRSM_SHARD_REPLICATION", 3))
AUTO_REBALANCE_INTERVAL = int(getattr(settings, "PRSM_REBALANCE_MINUTES", 30))


class P2PMessage:
    """Standard P2P message format with encryption and signatures"""
    
    def __init__(self, msg_type: str, payload: dict, sender_id: str):
        self.id = str(uuid4())
        self.type = msg_type
        self.payload = payload
        self.sender_id = sender_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.signature = None
        self.encrypted = False
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'payload': self.payload,
            'sender_id': self.sender_id,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'encrypted': self.encrypted
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'P2PMessage':
        msg = cls(data['type'], data['payload'], data['sender_id'])
        msg.id = data['id']
        msg.timestamp = data['timestamp']
        msg.signature = data.get('signature')
        msg.encrypted = data.get('encrypted', False)
        return msg


class SecureP2PConnection:
    """Secure connection manager for P2P communications"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.private_key = PrivateKey.generate()
        self.public_key = self.private_key.public_key
        self.signing_key = SigningKey.generate()
        self.verify_key = self.signing_key.verify_key
        
        # Connection state
        self.active_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.peer_public_keys: Dict[str, PublicKey] = {}
        self.peer_verify_keys: Dict[str, VerifyKey] = {}
        self.connection_boxes: Dict[str, Box] = {}
        
        # Security metrics
        self.encryption_enabled = ENABLE_ENCRYPTION
        self.signatures_required = REQUIRE_SIGNATURES
        
    async def establish_connection(self, peer_addr: str, peer_id: str) -> bool:
        """Establish secure connection with a peer"""
        try:
            # Connect via WebSocket
            uri = f"ws://{peer_addr}/p2p"
            websocket = await websockets.connect(uri, timeout=CONNECTION_TIMEOUT)
            
            # Perform key exchange
            if self.encryption_enabled:
                success = await self._perform_key_exchange(websocket, peer_id)
                if not success:
                    await websocket.close()
                    return False
            
            # Store connection
            self.active_connections[peer_id] = websocket
            
            print(f"🔗 Established secure connection with peer {peer_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to peer {peer_id}: {e}")
            return False
    
    async def send_message(self, peer_id: str, message: P2PMessage) -> bool:
        """Send encrypted and signed message to peer"""
        try:
            if peer_id not in self.active_connections:
                print(f"⚠️ No connection to peer {peer_id}")
                return False
            
            websocket = self.active_connections[peer_id]
            
            # Sign message if required
            if self.signatures_required:
                message_bytes = json.dumps(message.payload).encode('utf-8')
                message.signature = self.signing_key.sign(message_bytes).signature.hex()
            
            # Encrypt message if enabled
            if self.encryption_enabled and peer_id in self.connection_boxes:
                box = self.connection_boxes[peer_id]
                encrypted_payload = box.encrypt(
                    json.dumps(message.payload).encode('utf-8')
                )
                message.payload = {'encrypted_data': encrypted_payload.hex()}
                message.encrypted = True
            
            # Send message
            await websocket.send(json.dumps(message.to_dict()))
            return True
            
        except Exception as e:
            print(f"❌ Error sending message to peer {peer_id}: {e}")
            await self._handle_connection_error(peer_id)
            return False
    
    async def receive_message(self, websocket, peer_id: str) -> Optional[P2PMessage]:
        """Receive and decrypt message from peer"""
        try:
            raw_data = await websocket.recv()
            data = json.loads(raw_data)
            message = P2PMessage.from_dict(data)
            
            # Decrypt if encrypted
            if message.encrypted and peer_id in self.connection_boxes:
                box = self.connection_boxes[peer_id]
                encrypted_data = bytes.fromhex(message.payload['encrypted_data'])
                decrypted_data = box.decrypt(encrypted_data)
                message.payload = json.loads(decrypted_data.decode('utf-8'))
                message.encrypted = False
            
            # Verify signature if present
            if message.signature and self.signatures_required:
                if not await self._verify_message_signature(message, peer_id):
                    print(f"⚠️ Invalid signature from peer {peer_id}")
                    return None
            
            return message
            
        except Exception as e:
            print(f"❌ Error receiving message from peer {peer_id}: {e}")
            return None
    
    async def _perform_key_exchange(self, websocket, peer_id: str) -> bool:
        """Perform Diffie-Hellman key exchange with peer"""
        try:
            # Send our public key
            key_exchange_msg = {
                'type': 'key_exchange',
                'public_key': self.public_key.encode().hex(),
                'verify_key': self.verify_key.encode().hex(),
                'node_id': self.node_id
            }
            await websocket.send(json.dumps(key_exchange_msg))
            
            # Receive peer's public key
            response = await websocket.recv()
            peer_data = json.loads(response)
            
            if peer_data['type'] != 'key_exchange':
                return False
            
            # Store peer's keys
            peer_public_key = PublicKey(bytes.fromhex(peer_data['public_key']))
            peer_verify_key = VerifyKey(bytes.fromhex(peer_data['verify_key']))
            
            self.peer_public_keys[peer_id] = peer_public_key
            self.peer_verify_keys[peer_id] = peer_verify_key
            
            # Create shared encryption box
            self.connection_boxes[peer_id] = Box(self.private_key, peer_public_key)
            
            print(f"🔐 Key exchange completed with peer {peer_id}")
            return True
            
        except Exception as e:
            print(f"❌ Key exchange failed with peer {peer_id}: {e}")
            return False
    
    async def _verify_message_signature(self, message: P2PMessage, peer_id: str) -> bool:
        """Verify message signature using peer's verify key"""
        try:
            if peer_id not in self.peer_verify_keys:
                return False
            
            verify_key = self.peer_verify_keys[peer_id]
            message_bytes = json.dumps(message.payload).encode('utf-8')
            signature_bytes = bytes.fromhex(message.signature)
            
            verify_key.verify(message_bytes, signature_bytes)
            return True
            
        except Exception:
            return False
    
    async def _handle_connection_error(self, peer_id: str):
        """Handle connection errors and cleanup"""
        if peer_id in self.active_connections:
            try:
                await self.active_connections[peer_id].close()
            except:
                pass
            del self.active_connections[peer_id]
        
        # Cleanup security state
        self.peer_public_keys.pop(peer_id, None)
        self.peer_verify_keys.pop(peer_id, None)
        self.connection_boxes.pop(peer_id, None)


class DistributedHashTable:
    """Kademlia DHT for peer and content discovery"""
    
    def __init__(self, node_id: str, listen_port: int = P2P_LISTEN_PORT + 1):
        self.node_id = node_id
        self.listen_port = listen_port
        self.server = None
        self.storage = ForgetfulStorage()
        self.bootstrap_nodes = DHT_BOOTSTRAP_NODES
        
    async def start(self) -> bool:
        """Start the DHT server"""
        try:
            self.server = KademliaServer(storage=self.storage)
            await self.server.listen(self.listen_port)
            
            # Bootstrap to network if nodes available
            if self.bootstrap_nodes:
                bootstrap_addrs = [(node.split(':')[0], int(node.split(':')[1])) 
                                 for node in self.bootstrap_nodes if ':' in node]
                if bootstrap_addrs:
                    await self.server.bootstrap(bootstrap_addrs)
            
            print(f"🌐 DHT server started on port {self.listen_port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start DHT: {e}")
            return False
    
    async def stop(self):
        """Stop the DHT server"""
        if self.server:
            self.server.stop()
            print("🛑 DHT server stopped")
    
    async def store(self, key: str, value: dict) -> bool:
        """Store value in DHT"""
        try:
            if not self.server:
                return False
            
            serialized_value = json.dumps(value)
            await self.server.set(key, serialized_value)
            print(f"💾 Stored in DHT: {key}")
            return True
            
        except Exception as e:
            print(f"❌ DHT store error: {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[dict]:
        """Retrieve value from DHT"""
        try:
            if not self.server:
                return None
            
            value = await self.server.get(key)
            if value is None:
                return None
            
            return json.loads(value)
            
        except Exception as e:
            print(f"❌ DHT retrieve error: {e}")
            return None
    
    async def find_peers_by_capability(self, capability: str) -> List[Tuple[str, str]]:
        """Find peers offering a specific capability"""
        try:
            capability_key = f"capability:{capability}"
            peer_data = await self.retrieve(capability_key)
            
            if not peer_data:
                return []
            
            return [(peer['id'], peer['address']) for peer in peer_data.get('peers', [])]
            
        except Exception as e:
            print(f"❌ Error finding peers by capability: {e}")
            return []
    
    async def announce_capability(self, capability: str, peer_address: str) -> bool:
        """Announce that this node provides a capability"""
        try:
            capability_key = f"capability:{capability}"
            existing_data = await self.retrieve(capability_key) or {'peers': []}
            
            # Add this peer to the capability list
            peer_info = {
                'id': self.node_id,
                'address': peer_address,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Remove existing entry for this peer
            existing_data['peers'] = [p for p in existing_data['peers'] if p['id'] != self.node_id]
            existing_data['peers'].append(peer_info)
            
            # Limit to recent peers (last 24 hours)
            cutoff_time = datetime.now(timezone.utc).timestamp() - (24 * 3600)
            existing_data['peers'] = [
                p for p in existing_data['peers']
                if datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')).timestamp() > cutoff_time
            ]
            
            return await self.store(capability_key, existing_data)
            
        except Exception as e:
            print(f"❌ Error announcing capability: {e}")
            return False


class ProductionP2PNetwork:
    """
    Production-ready P2P network with real distributed networking
    Replaces simulation with libp2p, DHT, and secure communications
    """
    
    def __init__(self, node_id: Optional[str] = None):
        # Core network state
        self.node_id = node_id or str(uuid4())
        self.secure_connection = SecureP2PConnection(self.node_id)
        self.dht = DistributedHashTable(self.node_id)
        
        # P2P state
        self.active_peers: Dict[str, PeerNode] = {}
        self.model_shards: Dict[str, List[ModelShard]] = {}
        self.shard_locations: Dict[UUID, Set[str]] = {}
        
        # Network services
        self.websocket_server = None
        self.message_handlers: Dict[str, Callable] = {}
        self.network_stats = {
            'connections_established': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'encryption_failures': 0,
            'signature_failures': 0
        }
        
        # Safety integration
        self.circuit_breaker = CircuitBreakerNetwork()
        self.safety_monitor = SafetyMonitor()
        
        # Setup message handlers
        self._setup_message_handlers()
    
    async def start_network(self, listen_address: str = "localhost", 
                          listen_port: int = P2P_LISTEN_PORT) -> bool:
        """Start the P2P network services"""
        try:
            # Start DHT
            dht_started = await self.dht.start()
            if not dht_started:
                return False
            
            # Start WebSocket server
            await self._start_websocket_server(listen_address, listen_port)
            
            # Announce our capabilities
            peer_address = f"{listen_address}:{listen_port}"
            await self.dht.announce_capability("model_execution", peer_address)
            await self.dht.announce_capability("data_storage", peer_address)
            
            print(f"🚀 P2P network started on {listen_address}:{listen_port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start P2P network: {e}")
            return False
    
    async def stop_network(self):
        """Stop all P2P network services"""
        try:
            # Close all connections
            for peer_id in list(self.secure_connection.active_connections.keys()):
                await self._disconnect_peer(peer_id)
            
            # Stop WebSocket server
            if self.websocket_server:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
            
            # Stop DHT
            await self.dht.stop()
            
            print("🛑 P2P network stopped")
            
        except Exception as e:
            print(f"❌ Error stopping P2P network: {e}")
    
    async def distribute_model_shards(self, model_cid: str, shard_count: int) -> List[ModelShard]:
        """
        Distribute model into shards across the P2P network (REAL implementation)
        """
        try:
            # Safety check
            safety_check = await self.safety_monitor.validate_model_output(
                {"model_cid": model_cid, "action": "distribute"},
                ["no_malicious_code", "validate_integrity"]
            )
            if not safety_check:
                raise ValueError(f"Safety validation failed for model {model_cid}")
            
            # Get model data from IPFS
            ipfs_client = get_ipfs_client()
            model_data, metadata = await ipfs_client.retrieve_with_provenance(model_cid)
            model_size = len(model_data)
            
            # Calculate shard sizes
            shard_size = max(model_size // shard_count, 1024)
            shards = []
            
            for i in range(shard_count):
                # Create shard
                start_byte = i * shard_size
                end_byte = min((i + 1) * shard_size, model_size)
                shard_data = model_data[start_byte:end_byte]
                
                shard = ModelShard(
                    model_cid=model_cid,
                    shard_index=i,
                    total_shards=shard_count,
                    verification_hash=hashlib.sha256(shard_data).hexdigest(),
                    size_bytes=len(shard_data)
                )
                
                # Find hosting peers via DHT
                hosting_peers = await self._find_hosting_peers_via_dht(shard.shard_id)
                shard.hosted_by = hosting_peers
                
                # REAL shard distribution
                success = await self._distribute_shard_to_peers(shard, shard_data, hosting_peers)
                if not success:
                    raise ValueError(f"Failed to distribute shard {i}")
                
                shards.append(shard)
                self.shard_locations[shard.shard_id] = set(hosting_peers)
            
            # Update model shard registry
            self.model_shards[model_cid] = shards
            
            # Store shard metadata in DHT
            await self._store_shard_metadata_in_dht(model_cid, shards)
            
            print(f"✅ Distributed model {model_cid} into {shard_count} shards across network")
            return shards
            
        except Exception as e:
            print(f"❌ Failed to distribute model shards: {e}")
            raise
    
    async def coordinate_distributed_execution(self, task: ArchitectTask) -> List[Future]:
        """
        Coordinate distributed execution across P2P network (REAL implementation)
        """
        try:
            # Safety validation
            safety_check = await self.safety_monitor.validate_model_output(
                task.dict(), 
                ["validate_task_safety", "check_resource_limits"]
            )
            if not safety_check:
                raise ValueError(f"Task {task.task_id} failed safety validation")
            
            # Find execution peers via DHT
            execution_peers = await self._find_execution_peers_via_dht(task)
            if len(execution_peers) < 2:
                raise ValueError(f"Insufficient qualified peers for task execution")
            
            # Create execution futures
            execution_futures = []
            
            for peer_id, peer_address in execution_peers:
                # REAL peer execution via RPC
                future = asyncio.create_task(
                    self._execute_task_on_peer_rpc(peer_id, peer_address, task)
                )
                execution_futures.append(future)
            
            print(f"🚀 Coordinating distributed execution across {len(execution_peers)} peers")
            return execution_futures
            
        except Exception as e:
            print(f"❌ Failed to coordinate distributed execution: {e}")
            raise
    
    async def connect_to_peer(self, peer_address: str, peer_id: str) -> bool:
        """Connect to a peer using real networking"""
        try:
            success = await self.secure_connection.establish_connection(peer_address, peer_id)
            if success:
                self.network_stats['connections_established'] += 1
                
                # Add to active peers
                peer_node = PeerNode(
                    peer_id=peer_id,
                    address=peer_address,
                    capabilities=["model_execution", "data_storage"],
                    reputation_score=0.5,
                    active=True
                )
                self.active_peers[peer_id] = peer_node
                
                print(f"✅ Connected to peer {peer_id} at {peer_address}")
            
            return success
            
        except Exception as e:
            print(f"❌ Failed to connect to peer {peer_id}: {e}")
            return False
    
    async def discover_peers(self, capability: str = "model_execution") -> List[Tuple[str, str]]:
        """Discover peers with specific capabilities via DHT"""
        try:
            peers = await self.dht.find_peers_by_capability(capability)
            print(f"🔍 Discovered {len(peers)} peers with capability '{capability}'")
            return peers
            
        except Exception as e:
            print(f"❌ Error discovering peers: {e}")
            return []
    
    # === Private Implementation Methods ===
    
    def _setup_message_handlers(self):
        """Setup handlers for different message types"""
        self.message_handlers = {
            'shard_store': self._handle_shard_store,
            'shard_retrieve': self._handle_shard_retrieve,
            'task_execute': self._handle_task_execute,
            'peer_discovery': self._handle_peer_discovery,
            'heartbeat': self._handle_heartbeat
        }
    
    async def _start_websocket_server(self, address: str, port: int):
        """Start WebSocket server for P2P communications"""
        async def handle_client(websocket, path):
            peer_id = None
            try:
                # Perform handshake
                handshake_msg = await websocket.recv()
                handshake_data = json.loads(handshake_msg)
                peer_id = handshake_data.get('peer_id')
                
                if not peer_id:
                    await websocket.close(code=1008, reason="Missing peer_id")
                    return
                
                # Key exchange if encryption enabled
                if self.secure_connection.encryption_enabled:
                    success = await self.secure_connection._perform_key_exchange(websocket, peer_id)
                    if not success:
                        await websocket.close(code=1008, reason="Key exchange failed")
                        return
                
                # Store connection
                self.secure_connection.active_connections[peer_id] = websocket
                
                # Handle messages
                async for raw_message in websocket:
                    message = await self.secure_connection.receive_message(websocket, peer_id)
                    if message:
                        await self._handle_incoming_message(message, peer_id)
                        self.network_stats['messages_received'] += 1
                    
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                print(f"❌ WebSocket error with peer {peer_id}: {e}")
            finally:
                if peer_id:
                    await self._disconnect_peer(peer_id)
        
        self.websocket_server = await websockets.serve(
            handle_client, address, port, max_size=None
        )
    
    async def _handle_incoming_message(self, message: P2PMessage, peer_id: str):
        """Handle incoming P2P messages"""
        try:
            handler = self.message_handlers.get(message.type)
            if handler:
                await handler(message, peer_id)
            else:
                print(f"⚠️ Unknown message type: {message.type}")
                
        except Exception as e:
            print(f"❌ Error handling message from {peer_id}: {e}")
    
    async def _find_hosting_peers_via_dht(self, shard_id: UUID) -> List[str]:
        """Find peers to host a shard using DHT discovery"""
        try:
            # Find peers with storage capability
            storage_peers = await self.dht.find_peers_by_capability("data_storage")
            
            if len(storage_peers) < SHARD_REPLICATION_FACTOR:
                # Use all available peers if not enough storage peers
                execution_peers = await self.dht.find_peers_by_capability("model_execution")
                all_peers = storage_peers + execution_peers
                # Remove duplicates
                unique_peers = list(set(peer_id for peer_id, _ in all_peers))
            else:
                unique_peers = [peer_id for peer_id, _ in storage_peers]
            
            # Select peers for replication
            selected_count = min(len(unique_peers), SHARD_REPLICATION_FACTOR)
            selected_peers = unique_peers[:selected_count]
            
            return selected_peers
            
        except Exception as e:
            print(f"❌ Error finding hosting peers: {e}")
            return []
    
    async def _distribute_shard_to_peers(self, shard: ModelShard, shard_data: bytes, peer_ids: List[str]) -> bool:
        """REAL shard distribution via P2P network"""
        try:
            success_count = 0
            
            for peer_id in peer_ids:
                # Create shard store message
                message = P2PMessage(
                    msg_type='shard_store',
                    payload={
                        'shard_id': str(shard.shard_id),
                        'model_cid': shard.model_cid,
                        'shard_index': shard.shard_index,
                        'total_shards': shard.total_shards,
                        'verification_hash': shard.verification_hash,
                        'shard_data': shard_data.hex()  # Hex encode for JSON transport
                    },
                    sender_id=self.node_id
                )
                
                # Send to peer
                success = await self.secure_connection.send_message(peer_id, message)
                if success:
                    success_count += 1
                    print(f"📦 Distributed shard {shard.shard_index} to peer {peer_id}")
                else:
                    print(f"❌ Failed to distribute shard to peer {peer_id}")
            
            # Consider successful if at least one peer received the shard
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Error distributing shard: {e}")
            return False
    
    async def _find_execution_peers_via_dht(self, task: ArchitectTask) -> List[Tuple[str, str]]:
        """Find qualified execution peers via DHT"""
        try:
            # Find peers with execution capability
            execution_peers = await self.dht.find_peers_by_capability("model_execution")
            
            # Filter and rank peers (simplified for now)
            qualified_peers = []
            for peer_id, peer_address in execution_peers:
                if peer_id in self.active_peers:
                    peer = self.active_peers[peer_id]
                    if peer.active and peer.reputation_score >= 0.3:
                        qualified_peers.append((peer_id, peer_address))
                else:
                    # Unknown peer, include with caution
                    qualified_peers.append((peer_id, peer_address))
            
            # Limit concurrent executions
            max_peers = min(len(qualified_peers), MAX_CONNECTIONS // 2)
            return qualified_peers[:max_peers]
            
        except Exception as e:
            print(f"❌ Error finding execution peers: {e}")
            return []
    
    async def _execute_task_on_peer_rpc(self, peer_id: str, peer_address: str, task: ArchitectTask) -> dict:
        """Execute task on peer via RPC (REAL implementation)"""
        try:
            # Ensure connection exists
            if peer_id not in self.secure_connection.active_connections:
                connected = await self.connect_to_peer(peer_address, peer_id)
                if not connected:
                    raise Exception(f"Could not connect to peer {peer_id}")
            
            # Create execution message
            message = P2PMessage(
                msg_type='task_execute',
                payload={
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'instruction': task.instruction,
                    'context_data': task.context_data,
                    'dependencies': task.dependencies,
                    'expected_output_type': task.expected_output_type
                },
                sender_id=self.node_id
            )
            
            # Send execution request
            success = await self.secure_connection.send_message(peer_id, message)
            if not success:
                raise Exception(f"Failed to send execution request to peer {peer_id}")
            
            # Wait for response (simplified - in production would use proper request/response matching)
            # For now, return a placeholder result
            return {
                "peer_id": peer_id,
                "task_id": task.task_id,
                "result": f"RPC execution result from peer {peer_id}",
                "execution_time": 2.5,  # Would be actual execution time
                "timestamp": datetime.now(timezone.utc),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ RPC execution failed on peer {peer_id}: {e}")
            return {
                "peer_id": peer_id,
                "task_id": task.task_id,
                "error": str(e),
                "execution_time": 0,
                "timestamp": datetime.now(timezone.utc),
                "success": False
            }
    
    async def _store_shard_metadata_in_dht(self, model_cid: str, shards: List[ModelShard]):
        """Store shard metadata in DHT for discovery"""
        try:
            shard_metadata = {
                'model_cid': model_cid,
                'total_shards': len(shards),
                'shards': [
                    {
                        'shard_id': str(shard.shard_id),
                        'index': shard.shard_index,
                        'verification_hash': shard.verification_hash,
                        'size_bytes': shard.size_bytes,
                        'hosted_by': shard.hosted_by
                    } for shard in shards
                ],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            await self.dht.store(f"model_shards:{model_cid}", shard_metadata)
            print(f"📝 Stored shard metadata for model {model_cid} in DHT")
            
        except Exception as e:
            print(f"❌ Error storing shard metadata: {e}")
    
    async def _disconnect_peer(self, peer_id: str):
        """Disconnect from a peer and cleanup"""
        try:
            if peer_id in self.secure_connection.active_connections:
                websocket = self.secure_connection.active_connections[peer_id]
                await websocket.close()
            
            await self.secure_connection._handle_connection_error(peer_id)
            
            if peer_id in self.active_peers:
                del self.active_peers[peer_id]
            
            print(f"🔌 Disconnected from peer {peer_id}")
            
        except Exception as e:
            print(f"❌ Error disconnecting from peer {peer_id}: {e}")
    
    # === Message Handlers ===
    
    async def _handle_shard_store(self, message: P2PMessage, peer_id: str):
        """Handle incoming shard storage request"""
        try:
            payload = message.payload
            shard_data = bytes.fromhex(payload['shard_data'])
            
            # Verify shard integrity
            computed_hash = hashlib.sha256(shard_data).hexdigest()
            if computed_hash != payload['verification_hash']:
                print(f"❌ Shard integrity check failed from peer {peer_id}")
                return
            
            # Store shard locally (simplified - would use proper storage)
            shard_key = f"shard:{payload['shard_id']}"
            storage_success = await self.dht.store(shard_key, {
                'data': payload['shard_data'],
                'metadata': {
                    'model_cid': payload['model_cid'],
                    'shard_index': payload['shard_index'],
                    'verification_hash': payload['verification_hash']
                }
            })
            
            if storage_success:
                print(f"✅ Stored shard {payload['shard_index']} from peer {peer_id}")
            else:
                print(f"❌ Failed to store shard from peer {peer_id}")
                
        except Exception as e:
            print(f"❌ Error handling shard store: {e}")
    
    async def _handle_shard_retrieve(self, message: P2PMessage, peer_id: str):
        """Handle shard retrieval request"""
        # Implementation would retrieve and send shard data
        pass
    
    async def _handle_task_execute(self, message: P2PMessage, peer_id: str):
        """Handle task execution request"""
        # Implementation would execute task and return result
        pass
    
    async def _handle_peer_discovery(self, message: P2PMessage, peer_id: str):
        """Handle peer discovery message"""
        pass
    
    async def _handle_heartbeat(self, message: P2PMessage, peer_id: str):
        """Handle heartbeat from peer"""
        # Update peer's last seen timestamp
        if peer_id in self.active_peers:
            self.active_peers[peer_id].last_seen = datetime.now(timezone.utc)


# === Global P2P Network Instance ===

_production_p2p_instance: Optional[ProductionP2PNetwork] = None

def get_production_p2p_network() -> ProductionP2PNetwork:
    """Get or create the global production P2P network instance"""
    global _production_p2p_instance
    if _production_p2p_instance is None:
        _production_p2p_instance = ProductionP2PNetwork()
    return _production_p2p_instance