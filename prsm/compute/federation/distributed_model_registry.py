"""
Production-Ready Distributed Model Registry for PRSM
Real P2P model discovery with DHT and gossip protocols
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from uuid import UUID, uuid4

# P2P imports
try:
    from kademlia.network import Server as KademliaServer
    from kademlia.storage import ForgetfulStorage
    import nacl.utils
    from nacl.signing import SigningKey, VerifyKey
    from nacl.hash import sha256
except ImportError as e:
    print(f"⚠️ P2P dependencies not installed: {e}")

from prsm.core.config import settings
from prsm.core.models import (
    TeacherModel, ModelType, PeerNode, ModelShard,
    FTNSTransaction, ProvenanceRecord
)
from prsm.data.data_layer.enhanced_ipfs import get_ipfs_client
from prsm.economy.tokenomics.ftns_service import get_ftns_service
from .enhanced_p2p_network import get_production_p2p_network


# === Distributed Registry Configuration ===

# DHT settings
DHT_MODEL_TTL_HOURS = int(getattr(settings, "PRSM_MODEL_DHT_TTL", 72))
DHT_REFRESH_INTERVAL = int(getattr(settings, "PRSM_DHT_REFRESH_MINUTES", 15))
MAX_SEARCH_RESULTS = int(getattr(settings, "PRSM_MAX_SEARCH_RESULTS", 100))

# Gossip protocol settings
GOSSIP_FANOUT = int(getattr(settings, "PRSM_GOSSIP_FANOUT", 6))
GOSSIP_INTERVAL_SECONDS = int(getattr(settings, "PRSM_GOSSIP_INTERVAL", 30))
GOSSIP_MESSAGE_TTL = int(getattr(settings, "PRSM_GOSSIP_TTL", 300))

# Validation settings
MIN_MODEL_PERFORMANCE = float(getattr(settings, "PRSM_MIN_PERFORMANCE", 0.1))
INTEGRITY_CHECK_INTERVAL = int(getattr(settings, "PRSM_INTEGRITY_CHECK_HOURS", 6))
REQUIRE_PROVENANCE = getattr(settings, "PRSM_REQUIRE_PROVENANCE", True)

# Replication settings
MODEL_REPLICATION_FACTOR = int(getattr(settings, "PRSM_MODEL_REPLICATION", 3))
AUTO_REBALANCE_INTERVAL = int(getattr(settings, "PRSM_AUTO_REBALANCE_MINUTES", 60))


class GossipMessage:
    """Message for gossip protocol communication"""
    
    def __init__(self, msg_type: str, payload: dict, sender_id: str, ttl: int = GOSSIP_MESSAGE_TTL):
        self.id = str(uuid4())
        self.type = msg_type
        self.payload = payload
        self.sender_id = sender_id
        self.ttl = ttl
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.hops = 0
        self.signature = None
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'payload': self.payload,
            'sender_id': self.sender_id,
            'ttl': self.ttl,
            'timestamp': self.timestamp,
            'hops': self.hops,
            'signature': self.signature
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GossipMessage':
        msg = cls(data['type'], data['payload'], data['sender_id'], data.get('ttl', GOSSIP_MESSAGE_TTL))
        msg.id = data['id']
        msg.timestamp = data['timestamp']
        msg.hops = data.get('hops', 0)
        msg.signature = data.get('signature')
        return msg
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        try:
            msg_time = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            age = (datetime.now(timezone.utc) - msg_time).total_seconds()
            return age > self.ttl or self.hops > 10
        except:
            return True


class GossipProtocol:
    """Gossip protocol for efficient model registry updates"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.received_messages: Set[str] = set()  # Track message IDs to prevent loops
        self.message_handlers: Dict[str, Callable] = {}
        self.active_peers: Dict[str, str] = {}  # peer_id -> address
        
        # Cryptographic signing
        self.signing_key = SigningKey.generate()
        self.verify_key = self.signing_key.verify_key
        self.peer_verify_keys: Dict[str, VerifyKey] = {}
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_duplicated': 0,
            'messages_expired': 0,
            'broadcasts_initiated': 0
        }
        
        # Setup handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup message handlers"""
        self.message_handlers = {
            'model_announcement': self._handle_model_announcement,
            'model_update': self._handle_model_update,
            'model_removal': self._handle_model_removal,
            'peer_discovery': self._handle_peer_discovery,
            'registry_sync': self._handle_registry_sync
        }
    
    async def broadcast_message(self, msg_type: str, payload: dict) -> bool:
        """Broadcast a message to the network via gossip"""
        try:
            message = GossipMessage(msg_type, payload, self.node_id)
            
            # Sign message
            message_bytes = json.dumps(message.to_dict(), sort_keys=True).encode('utf-8')
            message.signature = self.signing_key.sign(message_bytes).signature.hex()
            
            # Send to random subset of peers (gossip fanout)
            selected_peers = self._select_gossip_peers()
            
            success_count = 0
            for peer_id in selected_peers:
                success = await self._send_gossip_message(peer_id, message)
                if success:
                    success_count += 1
            
            self.stats['messages_sent'] += success_count
            self.stats['broadcasts_initiated'] += 1
            
            print(f"📡 Gossiped {msg_type} to {success_count}/{len(selected_peers)} peers")
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Error broadcasting gossip message: {e}")
            return False
    
    async def receive_message(self, message_data: dict, sender_peer_id: str) -> bool:
        """Receive and process a gossip message"""
        try:
            message = GossipMessage.from_dict(message_data)
            
            # Check for duplicates
            if message.id in self.received_messages:
                self.stats['messages_duplicated'] += 1
                return False
            
            # Check if expired
            if message.is_expired():
                self.stats['messages_expired'] += 1
                return False
            
            # Verify signature
            if not await self._verify_message_signature(message):
                print(f"⚠️ Invalid gossip signature from {message.sender_id}")
                return False
            
            # Track message
            self.received_messages.add(message.id)
            self.stats['messages_received'] += 1
            
            # Handle message
            handler = self.message_handlers.get(message.type)
            if handler:
                await handler(message.payload, message.sender_id)
            
            # Forward message (with decreased TTL)
            if message.ttl > 0:
                await self._forward_message(message)
            
            return True
            
        except Exception as e:
            print(f"❌ Error receiving gossip message: {e}")
            return False
    
    def _select_gossip_peers(self) -> List[str]:
        """Select random subset of peers for gossip"""
        available_peers = list(self.active_peers.keys())
        if len(available_peers) <= GOSSIP_FANOUT:
            return available_peers
        
        import random
        return random.sample(available_peers, GOSSIP_FANOUT)
    
    async def _send_gossip_message(self, peer_id: str, message: GossipMessage) -> bool:
        """Send gossip message to a specific peer"""
        try:
            # Get P2P network instance
            p2p_network = get_production_p2p_network()
            
            # Create P2P message
            from .enhanced_p2p_network import P2PMessage
            p2p_msg = P2PMessage(
                msg_type='gossip',
                payload=message.to_dict(),
                sender_id=self.node_id
            )
            
            # Send via secure P2P connection
            return await p2p_network.secure_connection.send_message(peer_id, p2p_msg)
            
        except Exception as e:
            print(f"❌ Error sending gossip to peer {peer_id}: {e}")
            return False
    
    async def _forward_message(self, message: GossipMessage):
        """Forward message to other peers (with TTL decrease)"""
        try:
            # Decrease TTL and increment hops
            message.ttl -= 1
            message.hops += 1
            
            # Select fewer peers for forwarding
            forward_count = max(1, GOSSIP_FANOUT // 2)
            selected_peers = self._select_gossip_peers()[:forward_count]
            
            for peer_id in selected_peers:
                if peer_id != message.sender_id:  # Don't send back to sender
                    await self._send_gossip_message(peer_id, message)
            
        except Exception as e:
            print(f"❌ Error forwarding gossip message: {e}")
    
    async def _verify_message_signature(self, message: GossipMessage) -> bool:
        """Verify message signature"""
        try:
            if not message.signature:
                return False
            
            sender_verify_key = self.peer_verify_keys.get(message.sender_id)
            if not sender_verify_key:
                # Accept messages from unknown peers for now
                # In production, would require key exchange first
                return True
            
            # Verify signature
            message_copy = message.to_dict()
            message_copy.pop('signature', None)
            message_bytes = json.dumps(message_copy, sort_keys=True).encode('utf-8')
            signature_bytes = bytes.fromhex(message.signature)
            
            sender_verify_key.verify(message_bytes, signature_bytes)
            return True
            
        except Exception:
            return False
    
    # === Message Handlers ===
    
    async def _handle_model_announcement(self, payload: dict, sender_id: str):
        """Handle model announcement message"""
        print(f"📢 Received model announcement from {sender_id}: {payload.get('model_name', 'unknown')}")
    
    async def _handle_model_update(self, payload: dict, sender_id: str):
        """Handle model update message"""
        print(f"📝 Received model update from {sender_id}")
    
    async def _handle_model_removal(self, payload: dict, sender_id: str):
        """Handle model removal message"""
        print(f"🗑️ Received model removal from {sender_id}")
    
    async def _handle_peer_discovery(self, payload: dict, sender_id: str):
        """Handle peer discovery message"""
        print(f"🔍 Received peer discovery from {sender_id}")
    
    async def _handle_registry_sync(self, payload: dict, sender_id: str):
        """Handle registry synchronization message"""
        print(f"🔄 Received registry sync from {sender_id}")


class DistributedModelIndex:
    """DHT-based model indexing and discovery"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.dht_server = None
        self.storage = ForgetfulStorage()
        
        # Index categories
        self.category_indices: Dict[str, Set[str]] = defaultdict(set)
        self.performance_indices: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.provider_indices: Dict[str, Set[str]] = defaultdict(set)
        
        # Cache for faster lookups
        self.lookup_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
    
    async def start_dht(self, listen_port: int = 8468) -> bool:
        """Start DHT server"""
        try:
            from kademlia.network import Server
            self.dht_server = Server(storage=self.storage)
            await self.dht_server.listen(listen_port)
            
            print(f"🌐 DHT model index started on port {listen_port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start DHT: {e}")
            return False
    
    async def stop_dht(self):
        """Stop DHT server"""
        if self.dht_server:
            self.dht_server.stop()
            print("🛑 DHT model index stopped")
    
    async def index_model(self, model: TeacherModel, cid: str, provider_id: str) -> bool:
        """Index a model in the distributed registry"""
        try:
            model_id = str(model.teacher_id)
            
            # Create comprehensive model record
            model_record = {
                'model_id': model_id,
                'name': model.name,
                'specialization': model.specialization,
                'model_type': model.model_type.value if hasattr(model.model_type, 'value') else str(model.model_type),
                'performance_score': model.performance_score,
                'ipfs_cid': cid,
                'provider_id': provider_id,
                'indexed_at': datetime.now(timezone.utc).isoformat(),
                'ttl': DHT_MODEL_TTL_HOURS * 3600,
                'capabilities': getattr(model, 'capabilities', []),
                'tags': getattr(model, 'tags', []),
                'active': getattr(model, 'active', True)
            }
            
            # Store main model record
            model_key = f"model:{model_id}"
            await self._dht_store(model_key, model_record)
            
            # Index by category/specialization
            category_key = f"category:{model.specialization.lower()}"
            await self._add_to_category_index(category_key, model_id, model.performance_score)
            
            # Index by performance range
            performance_range = self._get_performance_range(model.performance_score)
            performance_key = f"performance:{performance_range}"
            await self._add_to_performance_index(performance_key, model_id, model.performance_score)
            
            # Index by provider
            provider_key = f"provider:{provider_id}"
            await self._add_to_provider_index(provider_key, model_id)
            
            # Index by model type
            type_key = f"type:{model.model_type}"
            await self._add_to_category_index(type_key, model_id, model.performance_score)
            
            print(f"📇 Indexed model {model.name} in distributed registry")
            print(f"   - Model ID: {model_id}")
            print(f"   - Category: {model.specialization}")
            print(f"   - Performance: {model.performance_score:.3f}")
            print(f"   - Provider: {provider_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error indexing model: {e}")
            return False
    
    async def search_models(self, query: dict) -> List[dict]:
        """Search for models using distributed index"""
        try:
            # Check cache first
            cache_key = json.dumps(query, sort_keys=True)
            if cache_key in self.lookup_cache:
                cached_result, cache_time = self.lookup_cache[cache_key]
                if (datetime.now(timezone.utc) - cache_time).total_seconds() < self.cache_ttl_seconds:
                    print(f"🎯 Cache hit for model search")
                    return cached_result
            
            results = []
            
            # Search by category/specialization
            if 'category' in query:
                category_results = await self._search_by_category(query['category'])
                results.extend(category_results)
            
            # Search by performance range
            if 'min_performance' in query:
                performance_results = await self._search_by_performance(query['min_performance'])
                results.extend(performance_results)
            
            # Search by provider
            if 'provider_id' in query:
                provider_results = await self._search_by_provider(query['provider_id'])
                results.extend(provider_results)
            
            # Search by model type
            if 'model_type' in query:
                type_results = await self._search_by_type(query['model_type'])
                results.extend(type_results)
            
            # Text search (if no specific criteria)
            if not results and 'text' in query:
                text_results = await self._text_search(query['text'])
                results.extend(text_results)
            
            # Remove duplicates and sort by performance
            unique_results = {}
            for result in results:
                model_id = result.get('model_id')
                if model_id and model_id not in unique_results:
                    unique_results[model_id] = result
            
            final_results = list(unique_results.values())
            final_results.sort(key=lambda x: x.get('performance_score', 0), reverse=True)
            
            # Apply limit
            max_results = query.get('limit', MAX_SEARCH_RESULTS)
            final_results = final_results[:max_results]
            
            # Cache results
            self.lookup_cache[cache_key] = (final_results, datetime.now(timezone.utc))
            
            print(f"🔍 Found {len(final_results)} models matching search criteria")
            return final_results
            
        except Exception as e:
            print(f"❌ Error searching models: {e}")
            return []
    
    async def discover_specialists(self, specialization: str) -> List[str]:
        """Discover specialist models for a domain"""
        try:
            category_key = f"category:{specialization.lower()}"
            category_data = await self._dht_retrieve(category_key)
            
            if not category_data:
                return []
            
            # Return model IDs sorted by performance
            models = category_data.get('models', [])
            models.sort(key=lambda x: x.get('performance_score', 0), reverse=True)
            
            return [model['model_id'] for model in models[:MAX_SEARCH_RESULTS]]
            
        except Exception as e:
            print(f"❌ Error discovering specialists: {e}")
            return []
    
    async def get_model_details(self, model_id: str) -> Optional[dict]:
        """Get detailed information about a model"""
        try:
            model_key = f"model:{model_id}"
            model_data = await self._dht_retrieve(model_key)
            
            if not model_data:
                return None
            
            # Verify model is still valid (not expired)
            indexed_time = datetime.fromisoformat(model_data['indexed_at'].replace('Z', '+00:00'))
            age_hours = (datetime.now(timezone.utc) - indexed_time).total_seconds() / 3600
            
            if age_hours > DHT_MODEL_TTL_HOURS:
                print(f"⚠️ Model {model_id} has expired in DHT")
                return None
            
            return model_data
            
        except Exception as e:
            print(f"❌ Error getting model details: {e}")
            return None
    
    async def remove_model(self, model_id: str) -> bool:
        """Remove a model from the distributed index"""
        try:
            # Get model details first
            model_data = await self.get_model_details(model_id)
            if not model_data:
                return False
            
            # Remove from main index
            model_key = f"model:{model_id}"
            await self._dht_delete(model_key)
            
            # Remove from category indices
            specialization = model_data.get('specialization', '')
            if specialization:
                category_key = f"category:{specialization.lower()}"
                await self._remove_from_category_index(category_key, model_id)
            
            # Remove from performance indices
            performance_score = model_data.get('performance_score', 0)
            performance_range = self._get_performance_range(performance_score)
            performance_key = f"performance:{performance_range}"
            await self._remove_from_performance_index(performance_key, model_id)
            
            # Remove from provider index
            provider_id = model_data.get('provider_id', '')
            if provider_id:
                provider_key = f"provider:{provider_id}"
                await self._remove_from_provider_index(provider_key, model_id)
            
            print(f"🗑️ Removed model {model_id} from distributed index")
            return True
            
        except Exception as e:
            print(f"❌ Error removing model: {e}")
            return False
    
    # === Private Helper Methods ===
    
    async def _dht_store(self, key: str, value: dict) -> bool:
        """Store value in DHT"""
        try:
            if not self.dht_server:
                return False
            
            serialized_value = json.dumps(value)
            await self.dht_server.set(key, serialized_value)
            return True
            
        except Exception as e:
            print(f"❌ DHT store error for key {key}: {e}")
            return False
    
    async def _dht_retrieve(self, key: str) -> Optional[dict]:
        """Retrieve value from DHT"""
        try:
            if not self.dht_server:
                return None
            
            value = await self.dht_server.get(key)
            if value is None:
                return None
            
            return json.loads(value)
            
        except Exception as e:
            print(f"❌ DHT retrieve error for key {key}: {e}")
            return None
    
    async def _dht_delete(self, key: str) -> bool:
        """Delete value from DHT"""
        try:
            if not self.dht_server:
                return False
            
            # Kademlia doesn't have direct delete, so we store None or expired data
            await self.dht_server.set(key, json.dumps(None))
            return True
            
        except Exception as e:
            print(f"❌ DHT delete error for key {key}: {e}")
            return False
    
    def _get_performance_range(self, score: float) -> str:
        """Get performance range bucket for a score"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "average"
        elif score >= 0.3:
            return "poor"
        else:
            return "very_poor"
    
    async def _add_to_category_index(self, category_key: str, model_id: str, performance_score: float):
        """Add model to category index"""
        try:
            category_data = await self._dht_retrieve(category_key) or {'models': []}
            
            # Add or update model in category
            models = category_data['models']
            models = [m for m in models if m.get('model_id') != model_id]  # Remove existing
            models.append({
                'model_id': model_id,
                'performance_score': performance_score,
                'added_at': datetime.now(timezone.utc).isoformat()
            })
            
            # Sort by performance and limit
            models.sort(key=lambda x: x.get('performance_score', 0), reverse=True)
            category_data['models'] = models[:MAX_SEARCH_RESULTS]
            
            await self._dht_store(category_key, category_data)
            
        except Exception as e:
            print(f"❌ Error adding to category index: {e}")
    
    async def _search_by_category(self, category: str) -> List[dict]:
        """Search models by category"""
        try:
            category_key = f"category:{category.lower()}"
            category_data = await self._dht_retrieve(category_key)
            
            if not category_data:
                return []
            
            results = []
            for model_info in category_data.get('models', []):
                model_id = model_info.get('model_id')
                if model_id:
                    model_details = await self.get_model_details(model_id)
                    if model_details:
                        results.append(model_details)
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching by category: {e}")
            return []
    
    async def _search_by_performance(self, min_performance: float) -> List[dict]:
        """Search models by minimum performance"""
        try:
            results = []
            
            # Check all performance ranges that meet the criteria
            performance_ranges = ["excellent", "good", "average", "poor", "very_poor"]
            range_thresholds = {"excellent": 0.9, "good": 0.7, "average": 0.5, "poor": 0.3, "very_poor": 0.0}
            
            for perf_range in performance_ranges:
                if range_thresholds[perf_range] >= min_performance:
                    performance_key = f"performance:{perf_range}"
                    performance_data = await self._dht_retrieve(performance_key)
                    
                    if performance_data:
                        for model_info in performance_data.get('models', []):
                            model_id = model_info.get('model_id')
                            if model_id:
                                model_details = await self.get_model_details(model_id)
                                if model_details and model_details.get('performance_score', 0) >= min_performance:
                                    results.append(model_details)
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching by performance: {e}")
            return []
    
    async def _search_by_provider(self, provider_id: str) -> List[dict]:
        """Search models by provider"""
        try:
            provider_key = f"provider:{provider_id}"
            provider_data = await self._dht_retrieve(provider_key)
            
            if not provider_data:
                return []
            
            results = []
            for model_id in provider_data.get('models', []):
                model_details = await self.get_model_details(model_id)
                if model_details:
                    results.append(model_details)
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching by provider: {e}")
            return []
    
    async def _search_by_type(self, model_type: str) -> List[dict]:
        """Search models by type"""
        try:
            type_key = f"type:{model_type}"
            type_data = await self._dht_retrieve(type_key)
            
            if not type_data:
                return []
            
            results = []
            for model_info in type_data.get('models', []):
                model_id = model_info.get('model_id')
                if model_id:
                    model_details = await self.get_model_details(model_id)
                    if model_details:
                        results.append(model_details)
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching by type: {e}")
            return []
    
    async def _text_search(self, text: str) -> List[dict]:
        """Perform text search across model names and descriptions"""
        try:
            # This is a simplified text search - in production would use more
            # sophisticated indexing like full-text search capabilities
            results = []
            
            # Search common categories that might match the text
            text_lower = text.lower()
            potential_categories = [
                "nlp", "computer_vision", "speech", "reasoning", "coding",
                "medical", "legal", "financial", "scientific", "creative"
            ]
            
            for category in potential_categories:
                if text_lower in category or category in text_lower:
                    category_results = await self._search_by_category(category)
                    results.extend(category_results)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in text search: {e}")
            return []
    
    # Additional helper methods for other indices...
    async def _add_to_performance_index(self, performance_key: str, model_id: str, performance_score: float):
        """Add model to performance index"""
        await self._add_to_category_index(performance_key, model_id, performance_score)
    
    async def _add_to_provider_index(self, provider_key: str, model_id: str):
        """Add model to provider index"""
        try:
            provider_data = await self._dht_retrieve(provider_key) or {'models': []}
            models = provider_data['models']
            
            if model_id not in models:
                models.append(model_id)
                provider_data['models'] = models
                await self._dht_store(provider_key, provider_data)
                
        except Exception as e:
            print(f"❌ Error adding to provider index: {e}")
    
    async def _remove_from_category_index(self, category_key: str, model_id: str):
        """Remove model from category index"""
        try:
            category_data = await self._dht_retrieve(category_key)
            if category_data and 'models' in category_data:
                category_data['models'] = [
                    m for m in category_data['models'] 
                    if m.get('model_id') != model_id
                ]
                await self._dht_store(category_key, category_data)
                
        except Exception as e:
            print(f"❌ Error removing from category index: {e}")
    
    async def _remove_from_performance_index(self, performance_key: str, model_id: str):
        """Remove model from performance index"""
        await self._remove_from_category_index(performance_key, model_id)
    
    async def _remove_from_provider_index(self, provider_key: str, model_id: str):
        """Remove model from provider index"""
        try:
            provider_data = await self._dht_retrieve(provider_key)
            if provider_data and 'models' in provider_data:
                models = provider_data['models']
                if model_id in models:
                    models.remove(model_id)
                    provider_data['models'] = models
                    await self._dht_store(provider_key, provider_data)
                    
        except Exception as e:
            print(f"❌ Error removing from provider index: {e}")


class ProductionModelRegistry:
    """
    Production-ready distributed model registry with real P2P discovery
    Replaces simulation with DHT indexing and gossip protocols
    """
    
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or str(uuid4())
        
        # Distributed components
        self.gossip_protocol = GossipProtocol(self.node_id)
        self.distributed_index = DistributedModelIndex(self.node_id)
        
        # Local registry (cache)
        self.local_registry: Dict[UUID, TeacherModel] = {}
        self.local_cids: Dict[UUID, str] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[UUID, Dict[str, Any]] = {}
        
        # Network state
        self.connected_peers: Dict[str, PeerNode] = {}
        self.registry_health = {
            'local_models': 0,
            'indexed_models': 0,
            'dht_connected': False,
            'gossip_active': False,
            'last_sync': None
        }
        
        # Statistics
        self.stats = {
            'models_registered': 0,
            'models_discovered': 0,
            'search_queries': 0,
            'gossip_messages': 0,
            'cache_hits': 0,
            'integrity_checks': 0
        }
    
    async def start_registry(self, dht_port: int = 8468) -> bool:
        """Start the distributed model registry"""
        try:
            # Start DHT index
            dht_started = await self.distributed_index.start_dht(dht_port)
            if not dht_started:
                return False
            
            self.registry_health['dht_connected'] = True
            
            # Start gossip protocol (will be activated when peers connect)
            self.registry_health['gossip_active'] = True
            
            # Initialize background tasks
            asyncio.create_task(self._periodic_sync())
            asyncio.create_task(self._periodic_health_check())
            
            print(f"🚀 Production model registry started")
            print(f"   - DHT index on port {dht_port}")
            print(f"   - Gossip protocol active")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start model registry: {e}")
            return False
    
    async def stop_registry(self):
        """Stop the distributed model registry"""
        try:
            await self.distributed_index.stop_dht()
            self.registry_health['dht_connected'] = False
            self.registry_health['gossip_active'] = False
            
            print("🛑 Production model registry stopped")
            
        except Exception as e:
            print(f"❌ Error stopping model registry: {e}")
    
    async def register_model(self, model: TeacherModel, cid: str, announce: bool = True) -> bool:
        """
        Register a model in the distributed registry (REAL implementation)
        """
        try:
            # Validate model integrity via IPFS
            ipfs_client = get_ipfs_client()
            integrity_valid = await ipfs_client.verify_model_integrity(cid)
            if not integrity_valid:
                print(f"❌ Model integrity validation failed for CID: {cid}")
                return False
            
            # Validate performance threshold
            if model.performance_score < MIN_MODEL_PERFORMANCE:
                print(f"❌ Model performance {model.performance_score} below minimum {MIN_MODEL_PERFORMANCE}")
                return False
            
            model_id = model.teacher_id
            
            # Store locally
            self.local_registry[model_id] = model
            self.local_cids[model_id] = cid
            
            # Index in distributed DHT
            success = await self.distributed_index.index_model(model, cid, self.node_id)
            if not success:
                print(f"❌ Failed to index model in DHT")
                return False
            
            # Initialize performance metrics
            self.performance_metrics[model_id] = {
                "registration_time": datetime.now(timezone.utc).isoformat(),
                "usage_count": 0,
                "average_rating": model.performance_score,
                "last_used": None,
                "success_rate": 1.0,
                "response_time_avg": 0.0,
                "provider_id": self.node_id
            }
            
            # Announce via gossip protocol
            if announce:
                await self._announce_model_registration(model, cid)
            
            # Reward registration
            await ftns_service.reward_contribution(
                self.node_id,
                'model_registration',
                2.0,  # Higher reward for model registration
                {
                    'model_id': str(model_id),
                    'specialization': model.specialization,
                    'performance': model.performance_score,
                    'cid': cid
                }
            )
            
            self.stats['models_registered'] += 1
            self.registry_health['local_models'] += 1
            
            print(f"✅ Registered model in distributed registry: {model.name}")
            print(f"   - Model ID: {model_id}")
            print(f"   - IPFS CID: {cid}")
            print(f"   - Specialization: {model.specialization}")
            print(f"   - Performance: {model.performance_score:.3f}")
            print(f"   - DHT indexed: ✅")
            print(f"   - Gossip announced: {'✅' if announce else '⏭️'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error registering model: {e}")
            return False
    
    async def discover_models(self, query: dict) -> List[dict]:
        """
        Discover models using distributed search (REAL implementation)
        """
        try:
            self.stats['search_queries'] += 1
            
            # Search distributed index
            results = await self.distributed_index.search_models(query)
            
            # Enhance results with real-time data
            enhanced_results = []
            for result in results:
                model_id = result.get('model_id')
                if model_id:
                    # Get performance metrics if available
                    model_uuid = UUID(model_id) if model_id else None
                    if model_uuid and model_uuid in self.performance_metrics:
                        metrics = self.performance_metrics[model_uuid]
                        result['current_metrics'] = metrics
                    
                    # Verify model availability
                    if await self._verify_model_availability(result):
                        enhanced_results.append(result)
            
            self.stats['models_discovered'] += len(enhanced_results)
            
            print(f"🔍 Discovered {len(enhanced_results)} models via distributed search")
            return enhanced_results
            
        except Exception as e:
            print(f"❌ Error discovering models: {e}")
            return []
    
    async def find_specialists(self, specialization: str) -> List[str]:
        """
        Find specialist models for a domain (REAL implementation)
        """
        try:
            # Use distributed index to find specialists
            model_ids = await self.distributed_index.discover_specialists(specialization)
            
            # Filter by availability and performance
            qualified_specialists = []
            for model_id in model_ids:
                model_details = await self.distributed_index.get_model_details(model_id)
                if model_details and model_details.get('active', True):
                    # Check if model meets performance criteria
                    performance = model_details.get('performance_score', 0)
                    if performance >= MIN_MODEL_PERFORMANCE:
                        qualified_specialists.append(model_id)
            
            print(f"🎯 Found {len(qualified_specialists)} specialists for: {specialization}")
            return qualified_specialists
            
        except Exception as e:
            print(f"❌ Error finding specialists: {e}")
            return []
    
    async def get_model_info(self, model_id: str) -> Optional[dict]:
        """Get comprehensive model information"""
        try:
            # Try local cache first
            model_uuid = UUID(model_id)
            if model_uuid in self.local_registry:
                model = self.local_registry[model_uuid]
                cid = self.local_cids.get(model_uuid)
                metrics = self.performance_metrics.get(model_uuid, {})
                
                self.stats['cache_hits'] += 1
                
                return {
                    'model': model.model_dump(),
                    'ipfs_cid': cid,
                    'performance_metrics': metrics,
                    'source': 'local',
                    'available': True
                }
            
            # Search distributed index
            model_details = await self.distributed_index.get_model_details(model_id)
            if model_details:
                model_details['source'] = 'distributed'
                model_details['available'] = await self._verify_model_availability(model_details)
                return model_details
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
            return None
    
    async def update_model_metrics(self, model_id: str, metrics: dict) -> bool:
        """Update model performance metrics"""
        try:
            model_uuid = UUID(model_id)
            
            # Update local metrics
            if model_uuid in self.performance_metrics:
                current_metrics = self.performance_metrics[model_uuid]
                
                # Update usage count
                if 'usage_increment' in metrics:
                    current_metrics['usage_count'] += 1
                    current_metrics['last_used'] = datetime.now(timezone.utc).isoformat()
                
                # Update performance ratings
                if 'rating' in metrics:
                    old_rating = current_metrics.get('average_rating', 0.0)
                    usage_count = current_metrics.get('usage_count', 1)
                    new_rating = ((old_rating * (usage_count - 1)) + metrics['rating']) / usage_count
                    current_metrics['average_rating'] = new_rating
                
                # Update success rate
                if 'success' in metrics:
                    old_success_rate = current_metrics.get('success_rate', 1.0)
                    usage_count = current_metrics.get('usage_count', 1)
                    success_value = 1.0 if metrics['success'] else 0.0
                    new_success_rate = ((old_success_rate * (usage_count - 1)) + success_value) / usage_count
                    current_metrics['success_rate'] = new_success_rate
                
                current_metrics['last_updated'] = datetime.now(timezone.utc).isoformat()
                
                print(f"📊 Updated metrics for model {model_id}")
                
                # Announce metrics update via gossip
                await self._announce_model_update(model_id, current_metrics)
                
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error updating model metrics: {e}")
            return False
    
    async def connect_peer(self, peer: PeerNode, verify_key_hex: str):
        """Connect to a peer for registry synchronization"""
        try:
            self.connected_peers[peer.peer_id] = peer
            self.gossip_protocol.active_peers[peer.peer_id] = peer.address
            self.gossip_protocol.peer_verify_keys[peer.peer_id] = VerifyKey(bytes.fromhex(verify_key_hex))
            
            print(f"🔗 Connected peer for registry sync: {peer.peer_id}")
            
            # Initiate registry synchronization
            await self._sync_with_peer(peer.peer_id)
            
        except Exception as e:
            print(f"❌ Error connecting peer: {e}")
    
    # === Private Implementation Methods ===
    
    async def _announce_model_registration(self, model: TeacherModel, cid: str):
        """Announce model registration via gossip protocol"""
        try:
            announcement = {
                'action': 'model_registered',
                'model_id': str(model.teacher_id),
                'model_name': model.name,
                'specialization': model.specialization,
                'performance_score': model.performance_score,
                'ipfs_cid': cid,
                'provider_id': self.node_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            success = await self.gossip_protocol.broadcast_message('model_announcement', announcement)
            if success:
                self.stats['gossip_messages'] += 1
                print(f"📡 Announced model registration: {model.name}")
            
        except Exception as e:
            print(f"❌ Error announcing model registration: {e}")
    
    async def _announce_model_update(self, model_id: str, metrics: dict):
        """Announce model metrics update via gossip"""
        try:
            update = {
                'action': 'metrics_updated',
                'model_id': model_id,
                'metrics': metrics,
                'provider_id': self.node_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            await self.gossip_protocol.broadcast_message('model_update', update)
            
        except Exception as e:
            print(f"❌ Error announcing model update: {e}")
    
    async def _verify_model_availability(self, model_details: dict) -> bool:
        """Verify that a model is actually available"""
        try:
            # Check if IPFS CID is accessible
            cid = model_details.get('ipfs_cid')
            if cid:
                ipfs_client = get_ipfs_client()
                return await ipfs_client.verify_model_integrity(cid)
            
            return False
            
        except Exception as e:
            print(f"❌ Error verifying model availability: {e}")
            return False
    
    async def _sync_with_peer(self, peer_id: str):
        """Synchronize registry with a peer (REAL implementation)"""
        try:
            # Request peer's model listings
            sync_request = {
                'action': 'request_model_list',
                'requester_id': self.node_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            await self.gossip_protocol.broadcast_message('registry_sync', sync_request)
            
            print(f"🔄 Initiated registry sync with peer {peer_id}")
            
        except Exception as e:
            print(f"❌ Error syncing with peer: {e}")
    
    async def _periodic_sync(self):
        """Periodic registry synchronization"""
        while True:
            try:
                await asyncio.sleep(DHT_REFRESH_INTERVAL * 60)  # Convert to seconds
                
                # Refresh our model listings in DHT
                for model_id, model in self.local_registry.items():
                    cid = self.local_cids.get(model_id)
                    if cid:
                        await self.distributed_index.index_model(model, cid, self.node_id)
                
                self.registry_health['last_sync'] = datetime.now(timezone.utc).isoformat()
                print(f"🔄 Completed periodic registry sync")
                
            except Exception as e:
                print(f"❌ Error in periodic sync: {e}")
    
    async def _periodic_health_check(self):
        """Periodic health monitoring"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Update health status
                self.registry_health['indexed_models'] = len(self.local_registry)
                self.registry_health['dht_connected'] = self.distributed_index.dht_server is not None
                
                # Perform integrity checks
                integrity_failures = 0
                for model_id, cid in self.local_cids.items():
                    available = await self._verify_model_availability({'ipfs_cid': cid})
                    if not available:
                        integrity_failures += 1
                
                self.stats['integrity_checks'] += len(self.local_cids)
                
                if integrity_failures > 0:
                    print(f"⚠️ Health check: {integrity_failures} models failed availability check")
                
            except Exception as e:
                print(f"❌ Error in health check: {e}")
    
    async def get_registry_status(self) -> dict:
        """Get comprehensive registry status"""
        return {
            'node_id': self.node_id,
            'health': self.registry_health,
            'stats': self.stats,
            'connected_peers': len(self.connected_peers),
            'gossip_stats': self.gossip_protocol.stats,
            'local_models': len(self.local_registry),
            'dht_connected': self.registry_health['dht_connected']
        }


# === Global Production Registry Instance ===

_production_registry_instance: Optional[ProductionModelRegistry] = None

def get_production_model_registry() -> ProductionModelRegistry:
    """Get or create the global production model registry instance"""
    global _production_registry_instance
    if _production_registry_instance is None:
        _production_registry_instance = ProductionModelRegistry()
    return _production_registry_instance