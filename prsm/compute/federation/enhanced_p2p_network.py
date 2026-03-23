"""
Production-Ready P2P Model Network for PRSM
Real distributed networking with libp2p, DHT, and secure communications
"""

import asyncio
import hashlib
import json
import logging
import struct
import time
import warnings
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from uuid import UUID, uuid4
from pathlib import Path

# P2P Networking imports
try:
    from kademlia.network import Server as KademliaServer
    from kademlia.storage import ForgetfulStorage
    import websockets
    import nacl.secret
    import nacl.utils
    from nacl.public import PrivateKey, PublicKey, Box
    from nacl.signing import SigningKey, VerifyKey
    from prsm.core.merkle import MerkleTools
except ImportError as e:
    KademliaServer = None
    ForgetfulStorage = None
    websockets = None
    nacl = None
    PrivateKey = None
    PublicKey = None
    Box = None
    SigningKey = None
    VerifyKey = None
    MerkleTools = None
    print(f"⚠️ P2P dependencies not installed: {e}")
    print("Install with: pip install kademlia websockets pynacl")

from prsm.core.config import settings
from prsm.core.models import (
    ArchitectTask, PeerNode, ModelShard, TeacherModel, ModelType,
    SafetyFlag, CircuitBreakerEvent, AgentResponse
)
from prsm.core.ipfs_client import get_ipfs_client
from prsm.economy.tokenomics.ftns_service import get_ftns_service
from prsm.core.safety.circuit_breaker import CircuitBreakerNetwork, ThreatLevel
from prsm.core.safety.monitor import SafetyMonitor

# HTTP client for RPC execution
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False


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
CAPABILITY_RECORD_MAX_AGE_SECONDS = int(getattr(settings, "PRSM_CAPABILITY_MAX_AGE_SECONDS", 24 * 3600))


logger = logging.getLogger(__name__)

_CANONICAL_COLLABORATION_REDIRECT = (
    "Use canonical collaboration dispatch via "
    "prsm.collaboration.CollaborationManager.dispatch_session() "
    "with prsm.node.agent_collaboration.AgentCollaboration bridge."
)


def _emit_collaboration_compatibility_fence(entrypoint: str) -> None:
    """Emit additive compatibility-only fence for collaboration-like federation entrypoints."""
    message = (
        f"Compatibility-only collaboration entrypoint used: {entrypoint}. "
        f"{_CANONICAL_COLLABORATION_REDIRECT}"
    )
    logger.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


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
        if PrivateKey is None or SigningKey is None:
            raise RuntimeError(
                "P2P cryptography dependencies unavailable. Install pynacl and related P2P extras."
            )
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
            except Exception:
                pass  # Connection may already be closed or in error state
            del self.active_connections[peer_id]
        
        # Cleanup security state
        self.peer_public_keys.pop(peer_id, None)
        self.peer_verify_keys.pop(peer_id, None)
        self.connection_boxes.pop(peer_id, None)


class DistributedHashTable:
    """Kademlia DHT for peer and content discovery"""
    
    def __init__(self, node_id: str, listen_port: int = P2P_LISTEN_PORT + 1):
        if ForgetfulStorage is None:
            raise RuntimeError(
                "DHT dependencies unavailable. Install kademlia for production DHT support."
            )
        self.node_id = node_id
        self.listen_port = listen_port
        self.server = None
        self.storage = ForgetfulStorage()
        self.bootstrap_nodes = DHT_BOOTSTRAP_NODES
        self.local_signing_key: Optional[SigningKey] = None
        self.local_verify_key_hex: Optional[str] = None
        self.peer_identity_keys: Dict[str, str] = {}
        self.peer_identity_status: Dict[str, str] = {}
        self.peer_identity_updated_at: Dict[str, str] = {}
        self.peer_key_versions: Dict[str, int] = {}
        self.revoked_capabilities: Dict[str, Set[str]] = {}

    def set_local_identity(self, node_id: str, signing_key: SigningKey, verify_key_hex: str):
        """Bind the local DHT announcer identity for capability signatures."""
        self.node_id = node_id
        self.local_signing_key = signing_key
        self.local_verify_key_hex = verify_key_hex
        self.register_peer_identity(node_id, verify_key_hex)

    def register_peer_identity(self, peer_id: str, verify_key_hex: str):
        """
        Register trusted identity binding for peer capability verification.

        Deterministic semantics:
        - first-time registration succeeds,
        - same-key re-registration is idempotent,
        - different-key updates are rejected and must use explicit rotation flow.
        """
        existing_key = self.peer_identity_keys.get(peer_id)
        if existing_key and existing_key != verify_key_hex:
            return False

        self.peer_identity_keys[peer_id] = verify_key_hex
        self.peer_identity_status[peer_id] = 'active'
        self.peer_identity_updated_at[peer_id] = datetime.now(timezone.utc).isoformat()
        self.peer_key_versions[peer_id] = self.peer_key_versions.get(peer_id, 0) + (0 if existing_key else 1)
        return True

    def rotate_peer_identity_key(
        self,
        peer_id: str,
        previous_verify_key_hex: str,
        new_verify_key_hex: str,
    ) -> bool:
        """Explicit, safe key rotation path for existing identities."""
        current_key = self.peer_identity_keys.get(peer_id)
        if not current_key:
            return False
        if self.peer_identity_status.get(peer_id) == 'revoked':
            return False
        if current_key != previous_verify_key_hex:
            return False
        if new_verify_key_hex == previous_verify_key_hex:
            return False

        self.peer_identity_keys[peer_id] = new_verify_key_hex
        self.peer_identity_status[peer_id] = 'active'
        self.peer_identity_updated_at[peer_id] = datetime.now(timezone.utc).isoformat()
        self.peer_key_versions[peer_id] = self.peer_key_versions.get(peer_id, 1) + 1
        return True

    def revoke_peer_identity(self, peer_id: str) -> bool:
        """Revoke peer identity and fail-closed future routing/trust decisions."""
        if peer_id not in self.peer_identity_keys:
            return False

        self.peer_identity_status[peer_id] = 'revoked'
        self.peer_identity_updated_at[peer_id] = datetime.now(timezone.utc).isoformat()
        return True

    def revoke_peer_capability(self, peer_id: str, capability: str) -> bool:
        """Revoke a specific capability for a peer from routing candidacy."""
        if peer_id not in self.peer_identity_keys:
            return False

        revoked = self.revoked_capabilities.setdefault(peer_id, set())
        revoked.add(capability)
        return True

    def _log_trust_decision(self, peer_id: str, capability: str, accepted: bool, reason: str):
        decision = 'accept' if accepted else 'reject'
        logger.info(
            "capability_trust_decision decision=%s peer_id=%s capability=%s reason=%s",
            decision,
            peer_id,
            capability,
            reason,
        )

    def _canonical_capability_payload(self, peer_record: dict) -> bytes:
        """Canonical payload used for capability signing/verification."""
        canonical = {
            'id': peer_record.get('id'),
            'address': peer_record.get('address'),
            'capability': peer_record.get('capability'),
            'timestamp': peer_record.get('timestamp'),
            'verify_key': peer_record.get('verify_key')
        }
        return json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode('utf-8')

    def _is_capability_record_trusted(self, peer_record: dict, expected_capability: str) -> bool:
        """Verify signed capability record and identity binding, fail-closed."""
        required_fields = {'id', 'address', 'capability', 'timestamp', 'verify_key', 'signature'}
        if not required_fields.issubset(set(peer_record.keys())):
            self._log_trust_decision('unknown', expected_capability, False, 'missing_required_fields')
            return False

        if peer_record.get('capability') != expected_capability:
            self._log_trust_decision(str(peer_record.get('id', 'unknown')), expected_capability, False, 'capability_mismatch')
            return False

        peer_id = peer_record.get('id')
        verify_key_hex = peer_record.get('verify_key')
        signature_hex = peer_record.get('signature')
        if not isinstance(peer_id, str) or not isinstance(verify_key_hex, str) or not isinstance(signature_hex, str):
            self._log_trust_decision(str(peer_record.get('id', 'unknown')), expected_capability, False, 'invalid_field_types')
            return False

        # Enforce explicit trusted identity binding (unknown == reject)
        expected_verify_key = self.peer_identity_keys.get(peer_id)
        if not expected_verify_key:
            self._log_trust_decision(peer_id, expected_capability, False, 'unknown_identity')
            return False

        identity_status = self.peer_identity_status.get(peer_id, 'active')
        if identity_status != 'active':
            self._log_trust_decision(peer_id, expected_capability, False, f'identity_status_{identity_status}')
            return False

        if expected_capability in self.revoked_capabilities.get(peer_id, set()):
            self._log_trust_decision(peer_id, expected_capability, False, 'capability_revoked')
            return False

        if expected_verify_key != verify_key_hex:
            self._log_trust_decision(peer_id, expected_capability, False, 'identity_key_mismatch')
            return False

        # Verify cryptographic signature
        try:
            verify_key = VerifyKey(bytes.fromhex(verify_key_hex))
            signature = bytes.fromhex(signature_hex)
            verify_key.verify(self._canonical_capability_payload(peer_record), signature)
        except Exception:
            self._log_trust_decision(peer_id, expected_capability, False, 'signature_verification_failed')
            return False

        # Reject stale capability records
        try:
            cutoff_time = datetime.now(timezone.utc).timestamp() - CAPABILITY_RECORD_MAX_AGE_SECONDS
            announced_at = datetime.fromisoformat(peer_record['timestamp'].replace('Z', '+00:00')).timestamp()
            is_fresh = announced_at > cutoff_time
            if not is_fresh:
                self._log_trust_decision(peer_id, expected_capability, False, 'stale_capability_record')
                return False
            self._log_trust_decision(peer_id, expected_capability, True, 'trusted_capability_record')
            return True
        except Exception:
            self._log_trust_decision(peer_id, expected_capability, False, 'invalid_timestamp')
            return False
        
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
            
            trusted_peers = []
            for peer_record in peer_data.get('peers', []):
                if self._is_capability_record_trusted(peer_record, capability):
                    trusted_peers.append((peer_record['id'], peer_record['address']))

            return trusted_peers
            
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
                'capability': capability,
                'verify_key': self.local_verify_key_hex,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            # Add signature when identity is available (production path)
            if self.local_signing_key and self.local_verify_key_hex:
                peer_info['signature'] = self.local_signing_key.sign(
                    self._canonical_capability_payload(peer_info)
                ).signature.hex()
            
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
        self.dht.set_local_identity(
            node_id=self.node_id,
            signing_key=self.secure_connection.signing_key,
            verify_key_hex=self.secure_connection.verify_key.encode().hex()
        )
        
        # P2P state
        self.active_peers: Dict[str, PeerNode] = {}
        self.model_shards: Dict[str, List[ModelShard]] = {}
        self.shard_locations: Dict[UUID, Set[str]] = {}
        self.peer_connection_state: Dict[str, Dict[str, Any]] = {}
        self._peer_reconcile_generation: Dict[str, int] = {}
        self.in_flight_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_task_results: Dict[str, Dict[str, Any]] = {}
        
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
        _emit_collaboration_compatibility_fence(
            "prsm.compute.federation.enhanced_p2p_network.ProductionP2PNetwork.coordinate_distributed_execution"
        )
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
            previous_state = self.peer_connection_state.get(peer_id, {}).get('state')
            success = await self.secure_connection.establish_connection(peer_address, peer_id)
            if success:
                self.network_stats['connections_established'] += 1
                
                # Add to active peers
                peer_node = PeerNode(
                    node_id=peer_id,
                    peer_id=peer_id,
                    multiaddr=peer_address,
                    capabilities=["model_execution", "data_storage"],
                    reputation_score=0.5,
                    active=True
                )
                self.active_peers[peer_id] = peer_node
                self._record_peer_transition(peer_id, 'connected')

                if previous_state in {'disconnected', 'partitioned'}:
                    await self.reconcile_peer_state(peer_id)
                
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

    def _record_peer_transition(self, peer_id: str, next_state: str):
        """Record deterministic peer connection transition state."""
        generation = self._peer_reconcile_generation.get(peer_id, 0)
        if next_state == 'connected':
            generation += 1
        self._peer_reconcile_generation[peer_id] = generation
        self.peer_connection_state[peer_id] = {
            'peer_id': peer_id,
            'state': next_state,
            'generation': generation,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }

    def _canonical_rpc_operation_id(self, peer_id: str, task: ArchitectTask) -> str:
        canonical_payload = {
            'peer_id': peer_id,
            'task_id': task.task_id,
            'task_type': task.task_type,
            'instruction': task.instruction,
            'context_data': task.context_data,
            'dependencies': task.dependencies,
            'expected_output_type': task.expected_output_type,
        }
        canonical_json = json.dumps(canonical_payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def _active_eligible_peer_ids(self) -> Set[str]:
        """Deterministic connected/trusted peer view for reconciliation."""
        eligible: Set[str] = set()
        for peer_id in sorted(self.active_peers.keys()):
            peer = self.active_peers[peer_id]
            state = self.peer_connection_state.get(peer_id, {}).get('state')
            if not peer.active:
                continue
            if state != 'connected':
                continue
            if self.dht.peer_identity_status.get(peer_id, 'active') != 'active':
                continue
            eligible.add(peer_id)
        return eligible

    async def reconcile_peer_state(self, peer_id: str) -> Dict[str, Any]:
        """Reconcile shard/task state for a reconnecting peer (fail-closed on unknowns)."""
        state = self.peer_connection_state.get(peer_id, {}).get('state')
        if state != 'connected':
            return {
                'peer_id': peer_id,
                'status': 'skipped',
                'reason': 'peer_not_connected'
            }

        refreshed_shards = await self._reconcile_shard_locations()
        task_updates = self._reconcile_in_flight_tasks(peer_id)
        return {
            'peer_id': peer_id,
            'status': 'reconciled',
            'generation': self.peer_connection_state.get(peer_id, {}).get('generation', 0),
            'refreshed_shards': refreshed_shards,
            'task_updates': task_updates,
        }

    async def _reconcile_shard_locations(self) -> int:
        """Rebuild shard locations deterministically from canonical shard metadata."""
        reconciled_count = 0
        eligible_peers = self._active_eligible_peer_ids()

        for model_cid in sorted(self.model_shards.keys()):
            for shard in sorted(
                self.model_shards[model_cid],
                key=lambda s: (str(s.shard_id), s.shard_index)
            ):
                canonical_hosts = sorted(set(shard.hosted_by or []))
                reconciled_hosts = {peer_id for peer_id in canonical_hosts if peer_id in eligible_peers}
                self.shard_locations[shard.shard_id] = reconciled_hosts
                reconciled_count += 1

        return reconciled_count

    def _reconcile_in_flight_tasks(self, peer_id: str) -> int:
        """
        Reconcile in-flight tasks for peer deterministically.

        Safety rule (fail-closed): unknown/pending task dispatch states are terminally
        marked aborted and are not retried automatically to avoid duplicate side-effects.
        """
        updates = 0
        for operation_id in sorted(self.in_flight_tasks.keys()):
            record = self.in_flight_tasks[operation_id]
            if record.get('peer_id') != peer_id:
                continue

            if record.get('state') in {'pending_dispatch', 'dispatching'}:
                record['state'] = 'aborted'
                record['finalized_by'] = 'reconcile_fail_closed'
                record['updated_at'] = datetime.now(timezone.utc).isoformat()
                updates += 1

        return updates
    
    # === Private Implementation Methods ===
    
    def _setup_message_handlers(self):
        """Setup handlers for different message types"""
        self.message_handlers = {
            'shard_store': self._handle_shard_store,
            'shard_retrieve': self._handle_shard_retrieve,
            'shard_retrieve_request': self._handle_shard_retrieve,
            'task_execute': self._handle_task_execute,
            'task_execute_request': self._handle_task_execute,
            'peer_discovery': self._handle_peer_discovery,
            'peer_discovery_request': self._handle_peer_discovery,
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
                response = await handler(message, peer_id)
                if isinstance(response, dict) and message.type in {
                    'shard_retrieve', 'shard_retrieve_request',
                    'task_execute', 'task_execute_request',
                    'peer_discovery', 'peer_discovery_request'
                }:
                    request_id = message.payload.get('request_id') if isinstance(message.payload, dict) else None
                    response_payload = dict(response)
                    if request_id is not None:
                        response_payload['request_id'] = request_id
                    response_message = P2PMessage(
                        msg_type=f"{message.type.replace('_request', '')}_response",
                        payload=response_payload,
                        sender_id=self.node_id
                    )
                    await self.secure_connection.send_message(peer_id, response_message)
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
                # Defense-in-depth: fail closed if trust state changed during selection
                peer_identity_status = self.dht.peer_identity_status.get(peer_id)
                if peer_identity_status != 'active':
                    continue
                if "model_execution" in self.dht.revoked_capabilities.get(peer_id, set()):
                    continue

                if peer_id in self.active_peers:
                    peer = self.active_peers[peer_id]
                    if peer.active and peer.reputation_score >= 0.3:
                        qualified_peers.append((peer_id, peer_address))
                else:
                    # Identity already trust-vetted by DHT capability gate
                    qualified_peers.append((peer_id, peer_address))
            
            # Limit concurrent executions
            max_peers = min(len(qualified_peers), MAX_CONNECTIONS // 2)
            return qualified_peers[:max_peers]
            
        except Exception as e:
            print(f"❌ Error finding execution peers: {e}")
            return []
    
    async def _execute_task_on_peer_rpc(self, peer_id: str, peer_address: str, task: ArchitectTask) -> dict:
        """Execute task on peer via RPC (REAL implementation with HTTP/TCP transport)"""
        operation_id = None
        start_time = time.time()

        try:
            operation_id = self._canonical_rpc_operation_id(peer_id, task)
            existing_result = self.completed_task_results.get(operation_id)
            if existing_result is not None:
                return dict(existing_result)

            self.in_flight_tasks[operation_id] = {
                'peer_id': peer_id,
                'task_id': task.task_id,
                'state': 'pending_dispatch',
                'updated_at': datetime.now(timezone.utc).isoformat()
            }

            # Get peer info from active_peers or parse address
            peer_info = self.active_peers.get(peer_id)
            if peer_info:
                # Extract address and port from multiaddr
                multiaddr = peer_info.multiaddr
                # Parse multiaddr format like "/ip4/1.2.3.4/tcp/8000"
                parts = multiaddr.split("/")
                peer_host = None
                peer_port = None
                api_port = None
                for i, part in enumerate(parts):
                    if part == "ip4" and i + 1 < len(parts):
                        peer_host = parts[i + 1]
                    elif part == "tcp" and i + 1 < len(parts):
                        peer_port = int(parts[i + 1])
                # Check for API port in capabilities
                capabilities = getattr(peer_info, 'capabilities', []) or []
                if isinstance(capabilities, dict):
                    api_port = capabilities.get('api_port')
            else:
                # Parse peer_address (format: "host:port")
                if ":" in peer_address:
                    peer_host, peer_port_str = peer_address.rsplit(":", 1)
                    peer_port = int(peer_port_str)
                else:
                    peer_host = peer_address
                    peer_port = 8000
                api_port = None

            # Build request payload
            request_payload = {
                "task_id": str(task.task_id),
                "operation": task.task_type,
                "instruction": task.instruction,
                "args": task.context_data or {},
                "timeout": 30
            }

            self.in_flight_tasks[operation_id]['state'] = 'dispatching'
            self.in_flight_tasks[operation_id]['updated_at'] = datetime.now(timezone.utc).isoformat()

            response_data = None
            rpc_error = None

            # Try HTTP first if httpx is available and we have an API port or default HTTP
            if HTTPX_AVAILABLE and httpx is not None:
                http_port = api_port or (peer_port + 1000 if peer_port else 9000)
                http_url = f"http://{peer_host}:{http_port}/rpc/execute"

                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        http_response = await client.post(
                            http_url,
                            json=request_payload,
                            headers={"Content-Type": "application/json", "X-Requester-Id": self.node_id}
                        )

                        if http_response.status_code == 200:
                            response_data = http_response.json()
                        else:
                            rpc_error = f"HTTP {http_response.status_code}: {http_response.text[:200]}"

                except httpx.TimeoutException:
                    rpc_error = "connection_timeout"
                except httpx.ConnectError:
                    rpc_error = "connection_refused"
                except Exception as http_err:
                    rpc_error = str(http_err)

            # Fallback to raw TCP socket if HTTP failed or not available
            if response_data is None and rpc_error is None or rpc_error in ("connection_refused", "connection_timeout"):
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(peer_host, peer_port),
                        timeout=30.0
                    )

                    # Send length-prefixed JSON
                    json_data = json.dumps(request_payload).encode('utf-8')
                    writer.write(struct.pack('>I', len(json_data)))
                    writer.write(json_data)
                    await writer.drain()

                    # Read response
                    length_data = await asyncio.wait_for(reader.read(4), timeout=30.0)
                    if len(length_data) == 4:
                        response_length = struct.unpack('>I', length_data)[0]
                        response_body = await asyncio.wait_for(reader.read(response_length), timeout=30.0)
                        response_data = json.loads(response_body.decode('utf-8'))
                    else:
                        rpc_error = "invalid_response_length"

                    writer.close()
                    await writer.wait_closed()

                except asyncio.TimeoutError:
                    rpc_error = "connection_timeout"
                except ConnectionError:
                    rpc_error = "connection_refused"
                except Exception as tcp_err:
                    rpc_error = str(tcp_err)

            execution_time = time.time() - start_time

            # Build response
            if response_data is not None:
                response = {
                    "peer_id": peer_id,
                    "task_id": str(task.task_id),
                    "result": response_data.get("result"),
                    "execution_time": execution_time,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": response_data.get("success", True),
                    "operation_id": operation_id
                }
                self.in_flight_tasks[operation_id]['state'] = 'committed'
                self.in_flight_tasks[operation_id]['finalized_by'] = 'dispatch_acknowledged'
                logger.info("rpc_execution_success", peer_id=peer_id, task_id=str(task.task_id), execution_time=execution_time)
            else:
                response = {
                    "peer_id": peer_id,
                    "task_id": str(task.task_id),
                    "success": False,
                    "error": rpc_error or "unknown_error",
                    "execution_time": execution_time,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "operation_id": operation_id
                }
                self.in_flight_tasks[operation_id]['state'] = 'aborted'
                self.in_flight_tasks[operation_id]['finalized_by'] = 'rpc_failed'
                logger.error("rpc_execution_failed", peer_id=peer_id, task_id=str(task.task_id), error=rpc_error)

            self.in_flight_tasks[operation_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
            self.completed_task_results[operation_id] = dict(response)

            # Store in federation_messages table
            await self._record_rpc_message(peer_id, str(task.task_id), request_payload, response)

            return response

        except Exception as e:
            logger.error("rpc_execution_exception", peer_id=peer_id, error=str(e))
            fallback_operation_id = locals().get('operation_id')
            if fallback_operation_id and fallback_operation_id in self.in_flight_tasks:
                self.in_flight_tasks[fallback_operation_id]['state'] = 'aborted'
                self.in_flight_tasks[fallback_operation_id]['updated_at'] = datetime.now(timezone.utc).isoformat()

            response = {
                "peer_id": peer_id,
                "task_id": str(task.task_id),
                "error": str(e),
                "success": False,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation_id": fallback_operation_id
            }
            if fallback_operation_id:
                self.completed_task_results[fallback_operation_id] = dict(response)
            return response

    async def _record_rpc_message(self, peer_id: str, task_id: str, request: dict, response: dict) -> None:
        """Record RPC message in federation_messages table."""
        try:
            from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
            from prsm.core.database import FederationMessageModel
            from prsm.core.config import get_settings

            settings = get_settings()
            db_url = getattr(settings, 'database_url', None)
            if not db_url:
                return

            # Convert to async URL
            if db_url.startswith("postgresql://"):
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
            elif db_url.startswith("sqlite:///"):
                db_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///")

            engine = create_async_engine(db_url, echo=False)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)

            current_time = time.time()
            async with session_factory() as session:
                message_id = str(uuid4())
                session.add(FederationMessageModel(
                    message_id=message_id,
                    message_type="rpc_execute",
                    sender_id=self.node_id,
                    recipient_id=peer_id,
                    payload={"request": request, "response": response},
                    sent_at=current_time,
                    received_at=current_time,
                    processed_at=current_time,
                    status="processed" if response.get("success") else "failed"
                ))
                await session.commit()

        except Exception as e:
            logger.error("record_rpc_message_failed", error=str(e))

    async def _handle_rpc_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle incoming RPC request via TCP socket."""
        try:
            # Read length prefix
            length_data = await asyncio.wait_for(reader.read(4), timeout=30.0)
            if len(length_data) != 4:
                return

            request_length = struct.unpack('>I', length_data)[0]
            request_body = await asyncio.wait_for(reader.read(request_length), timeout=30.0)
            request = json.loads(request_body.decode('utf-8'))

            # Route to appropriate handler based on operation
            operation = request.get("operation", "")
            result = None

            if operation == "model_execution" or operation == "execute":
                # Execute model inference
                result = await self._handle_model_execution(request)
            elif operation == "shard_retrieve":
                # Retrieve shard
                result = await self._handle_shard_retrieve_request(request)
            else:
                result = {"success": False, "error": f"unknown_operation: {operation}"}

            # Send response
            response_data = json.dumps(result).encode('utf-8')
            writer.write(struct.pack('>I', len(response_data)))
            writer.write(response_data)
            await writer.drain()

        except asyncio.TimeoutError:
            logger.error("rpc_request_timeout")
        except Exception as e:
            logger.error("handle_rpc_request_failed", error=str(e))
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_model_execution(self, request: dict) -> dict:
        """Handle model execution RPC request."""
        try:
            instruction = request.get("instruction", "")
            args = request.get("args", {})

            # This would integrate with the actual model execution system
            # For now, return a placeholder
            return {
                "success": True,
                "result": f"Executed: {instruction[:100]}...",
                "tokens_used": len(instruction.split())
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_shard_retrieve_request(self, request: dict) -> dict:
        """Handle shard retrieve RPC request."""
        try:
            shard_id = request.get("args", {}).get("shard_id")
            if not shard_id:
                return {"success": False, "error": "missing_shard_id"}

            # Look up shard in local storage
            shard_key = f"shard:{shard_id}"
            shard_record = await self.dht.retrieve(shard_key)
            if not shard_record:
                return {"success": False, "error": "shard_not_found"}

            return {
                "success": True,
                "shard_data": shard_record.get("data"),
                "metadata": shard_record.get("metadata", {})
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
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
                self.active_peers[peer_id].active = False
            self._record_peer_transition(peer_id, 'disconnected')

            await self._reconcile_shard_locations()
            self._reconcile_in_flight_tasks(peer_id)
            
            print(f"🔌 Disconnected from peer {peer_id}")
            
        except Exception as e:
            print(f"❌ Error disconnecting from peer {peer_id}: {e}")

    async def mark_peer_partitioned(self, peer_id: str):
        """Mark a peer as partitioned and reconcile state in fail-closed mode."""
        if peer_id in self.active_peers:
            self.active_peers[peer_id].active = False
        await self.secure_connection._handle_connection_error(peer_id)
        self._record_peer_transition(peer_id, 'partitioned')
        await self._reconcile_shard_locations()
        self._reconcile_in_flight_tasks(peer_id)
    
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
        try:
            payload = message.payload if isinstance(message.payload, dict) else {}
            shard_id = payload.get('shard_id')

            if not shard_id:
                return {
                    'status': 'error',
                    'handler': 'shard_retrieve',
                    'error_code': 'INVALID_REQUEST',
                    'error': 'Missing required field: shard_id',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

            shard_key = f"shard:{shard_id}"
            shard_record = await self.dht.retrieve(shard_key)
            if not shard_record:
                return {
                    'status': 'error',
                    'handler': 'shard_retrieve',
                    'error_code': 'SHARD_NOT_FOUND',
                    'error': f'Shard not found: {shard_id}',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

            shard_data_hex = shard_record.get('data')
            shard_metadata = shard_record.get('metadata', {})
            if not isinstance(shard_data_hex, str):
                return {
                    'status': 'error',
                    'handler': 'shard_retrieve',
                    'error_code': 'INVALID_SHARD_RECORD',
                    'error': f'Invalid shard record for shard_id={shard_id}',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

            try:
                shard_data = bytes.fromhex(shard_data_hex)
            except Exception:
                return {
                    'status': 'error',
                    'handler': 'shard_retrieve',
                    'error_code': 'INVALID_SHARD_ENCODING',
                    'error': f'Corrupt shard encoding for shard_id={shard_id}',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

            expected_hash = shard_metadata.get('verification_hash')
            computed_hash = hashlib.sha256(shard_data).hexdigest()
            if expected_hash and expected_hash != computed_hash:
                return {
                    'status': 'error',
                    'handler': 'shard_retrieve',
                    'error_code': 'INTEGRITY_CHECK_FAILED',
                    'error': f'Shard integrity verification failed for shard_id={shard_id}',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

            return {
                'status': 'success',
                'handler': 'shard_retrieve',
                'peer_id': peer_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': {
                    'shard_id': shard_id,
                    'shard_data': shard_data_hex,
                    'metadata': shard_metadata
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'handler': 'shard_retrieve',
                'error_code': 'INTERNAL_ERROR',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _handle_task_execute(self, message: P2PMessage, peer_id: str):
        """Handle task execution request"""
        try:
            payload = message.payload if isinstance(message.payload, dict) else {}
            missing = [field for field in ('task_id', 'task_type', 'instruction') if not payload.get(field)]
            if missing:
                return {
                    'status': 'error',
                    'handler': 'task_execute',
                    'error_code': 'INVALID_REQUEST',
                    'error': f"Missing required field(s): {', '.join(missing)}",
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

            deterministic_input = json.dumps(
                {
                    'task_id': payload['task_id'],
                    'task_type': payload['task_type'],
                    'instruction': payload['instruction'],
                    'context_data': payload.get('context_data', {}),
                    'dependencies': payload.get('dependencies', []),
                    'expected_output_type': payload.get('expected_output_type')
                },
                sort_keys=True,
                separators=(',', ':')
            )
            execution_digest = hashlib.sha256(deterministic_input.encode('utf-8')).hexdigest()

            return {
                'status': 'success',
                'handler': 'task_execute',
                'peer_id': peer_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': {
                    'task_id': payload['task_id'],
                    'execution_digest': execution_digest,
                    'result': {
                        'accepted': True,
                        'deterministic_result_token': execution_digest[:24],
                        'simulated': True
                    }
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'handler': 'task_execute',
                'error_code': 'INTERNAL_ERROR',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _handle_peer_discovery(self, message: P2PMessage, peer_id: str):
        """Handle peer discovery message"""
        try:
            payload = message.payload if isinstance(message.payload, dict) else {}
            capability = payload.get('capability', 'model_execution')
            limit = payload.get('limit', 20)
            if not isinstance(limit, int) or limit <= 0:
                limit = 20

            discovered_peers = await self.discover_peers(capability)
            candidate_peers = [
                {'peer_id': discovered_peer_id, 'address': discovered_peer_address}
                for discovered_peer_id, discovered_peer_address in discovered_peers[:limit]
            ]

            return {
                'status': 'success',
                'handler': 'peer_discovery',
                'peer_id': peer_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': {
                    'capability': capability,
                    'count': len(candidate_peers),
                    'peers': candidate_peers
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'handler': 'peer_discovery',
                'error_code': 'INTERNAL_ERROR',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
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
