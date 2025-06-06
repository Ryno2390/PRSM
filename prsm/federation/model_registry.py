"""
Distributed Model Registry for PRSM
P2P model discovery and management system for federated AI models
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID, uuid4

from ..core.config import settings
from ..core.models import (
    TeacherModel, ModelType, PeerNode, ModelShard,
    FTNSTransaction, ProvenanceRecord
)
from ..data_layer.enhanced_ipfs import get_ipfs_client
from ..tokenomics.ftns_service import ftns_service


# === Registry Configuration ===

# Model discovery settings
MAX_SEARCH_RESULTS = int(getattr(settings, "PRSM_MAX_SEARCH_RESULTS", 50))
MODEL_CACHE_TTL_HOURS = int(getattr(settings, "PRSM_MODEL_CACHE_TTL", 24))
ENABLE_PERFORMANCE_TRACKING = getattr(settings, "PRSM_TRACK_PERFORMANCE", True)

# Validation settings
MIN_MODEL_PERFORMANCE = float(getattr(settings, "PRSM_MIN_PERFORMANCE", 0.1))
INTEGRITY_CHECK_INTERVAL = int(getattr(settings, "PRSM_INTEGRITY_CHECK_HOURS", 6))

# Federation settings
ENABLE_P2P_DISCOVERY = getattr(settings, "PRSM_P2P_DISCOVERY", True)
FEDERATION_SYNC_INTERVAL = int(getattr(settings, "PRSM_FEDERATION_SYNC_MINUTES", 15))


class ModelRegistry:
    """
    Distributed model registry for P2P model discovery and management
    Integrates with enhanced IPFS client and FTNS token system
    """
    
    def __init__(self):
        # Core registry storage
        self.registered_models: Dict[UUID, TeacherModel] = {}
        self.model_cids: Dict[UUID, str] = {}  # model_id -> IPFS CID
        self.category_index: Dict[str, Set[UUID]] = {}  # category -> model_ids
        self.performance_metrics: Dict[UUID, Dict[str, Any]] = {}
        
        # P2P federation state
        self.peer_registries: Dict[str, Dict[UUID, TeacherModel]] = {}  # peer_id -> models
        self.federation_nodes: Dict[str, PeerNode] = {}
        
        # Caching and optimization
        self.search_cache: Dict[str, Tuple[List[str], datetime]] = {}
        self.integrity_cache: Dict[str, Tuple[bool, datetime]] = {}
        
        # Statistics
        self.stats = {
            "total_models": 0,
            "total_categories": 0,
            "total_searches": 0,
            "total_validations": 0,
            "cache_hits": 0,
            "federation_syncs": 0
        }
    
    # === Core Registry Functions ===
    
    async def register_teacher_model(self, model: TeacherModel, cid: str) -> bool:
        """
        Register a teacher model in the distributed registry
        
        Args:
            model: TeacherModel instance to register
            cid: IPFS CID where model is stored
            
        Returns:
            True if registration successful
        """
        try:
            # Validate model integrity via IPFS
            if not await self.validate_model_integrity(cid):
                print(f"❌ Model integrity validation failed for CID: {cid}")
                return False
            
            # Validate model performance threshold
            if model.performance_score < MIN_MODEL_PERFORMANCE:
                print(f"❌ Model performance {model.performance_score} below minimum {MIN_MODEL_PERFORMANCE}")
                return False
            
            # Register model
            model_id = model.teacher_id
            self.registered_models[model_id] = model
            self.model_cids[model_id] = cid
            
            # Update category index
            specialization = model.specialization.lower()
            if specialization not in self.category_index:
                self.category_index[specialization] = set()
                self.stats["total_categories"] += 1
            
            self.category_index[specialization].add(model_id)
            
            # Initialize performance metrics
            self.performance_metrics[model_id] = {
                "registration_time": datetime.now(timezone.utc).isoformat(),
                "usage_count": 0,
                "average_rating": model.performance_score,
                "last_used": None,
                "success_rate": 0.0,
                "response_time_avg": 0.0
            }
            
            # Update IPFS CID in model
            model.ipfs_cid = cid
            
            # Reward model registration
            if hasattr(model, 'creator_id'):
                await ftns_service.reward_contribution(
                    model.creator_id if hasattr(model, 'creator_id') else 'unknown',
                    'model',
                    1.0,
                    {
                        'model_id': str(model_id),
                        'specialization': model.specialization,
                        'performance': model.performance_score,
                        'cid': cid
                    }
                )
            
            self.stats["total_models"] += 1
            
            print(f"✅ Registered teacher model: {model.name} ({model.specialization})")
            print(f"   - Model ID: {model_id}")
            print(f"   - IPFS CID: {cid}")
            print(f"   - Performance: {model.performance_score:.3f}")
            
            # Broadcast to federation if enabled
            if ENABLE_P2P_DISCOVERY:
                await self._broadcast_model_registration(model, cid)
            
            return True
            
        except Exception as e:
            print(f"❌ Error registering teacher model: {e}")
            return False
    
    async def discover_specialists(self, task_category: str) -> List[str]:
        """
        Discover specialist models for a given task category
        
        Args:
            task_category: Category/domain to search for (e.g., "data_analysis", "nlp")
            
        Returns:
            List of model IDs that specialize in the given category
        """
        self.stats["total_searches"] += 1
        
        # Check cache first
        cache_key = f"specialists_{task_category.lower()}"
        if cache_key in self.search_cache:
            cached_results, cache_time = self.search_cache[cache_key]
            cache_age = datetime.now(timezone.utc) - cache_time
            if cache_age.total_seconds() < MODEL_CACHE_TTL_HOURS * 3600:
                self.stats["cache_hits"] += 1
                print(f"🎯 Cache hit for specialists: {task_category} ({len(cached_results)} models)")
                return cached_results
        
        try:
            # Search local registry
            local_specialists = await self._search_local_specialists(task_category)
            
            # Search federation if enabled
            federated_specialists = []
            if ENABLE_P2P_DISCOVERY:
                federated_specialists = await self._search_federated_specialists(task_category)
            
            # Combine and deduplicate results
            all_specialists = list(set(local_specialists + federated_specialists))
            
            # Sort by performance score (descending)
            sorted_specialists = await self._sort_specialists_by_performance(all_specialists)
            
            # Limit results
            final_results = sorted_specialists[:MAX_SEARCH_RESULTS]
            
            # Cache results
            self.search_cache[cache_key] = (final_results, datetime.now(timezone.utc))
            
            print(f"🔍 Discovered {len(final_results)} specialists for: {task_category}")
            print(f"   - Local: {len(local_specialists)}, Federated: {len(federated_specialists)}")
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error discovering specialists for {task_category}: {e}")
            return []
    
    async def validate_model_integrity(self, cid: str) -> bool:
        """
        Validate model integrity using IPFS verification
        
        Args:
            cid: IPFS CID of the model to validate
            
        Returns:
            True if model integrity is verified
        """
        self.stats["total_validations"] += 1
        
        # Check cache first
        if cid in self.integrity_cache:
            cached_result, cache_time = self.integrity_cache[cid]
            cache_age = datetime.now(timezone.utc) - cache_time
            if cache_age.total_seconds() < INTEGRITY_CHECK_INTERVAL * 3600:
                return cached_result
        
        try:
            # Validate via enhanced IPFS client
            ipfs_client = get_ipfs_client()
            is_valid = await ipfs_client.verify_model_integrity(cid)
            
            # Cache result
            self.integrity_cache[cid] = (is_valid, datetime.now(timezone.utc))
            
            if is_valid:
                print(f"✅ Model integrity verified for CID: {cid}")
            else:
                print(f"❌ Model integrity check failed for CID: {cid}")
            
            return is_valid
            
        except Exception as e:
            print(f"❌ Error validating model integrity for {cid}: {e}")
            return False
    
    async def update_performance_metrics(self, model_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Update performance metrics for a registered model
        
        Args:
            model_id: ID of the model to update
            metrics: Dictionary containing performance metrics
            
        Returns:
            True if update successful
        """
        try:
            model_uuid = UUID(model_id)
            
            if model_uuid not in self.registered_models:
                print(f"⚠️ Model {model_id} not found in registry")
                return False
            
            if not ENABLE_PERFORMANCE_TRACKING:
                return True
            
            # Get current metrics
            current_metrics = self.performance_metrics.get(model_uuid, {})
            
            # Update metrics
            update_time = datetime.now(timezone.utc).isoformat()
            
            # Update usage count
            if 'usage_increment' in metrics:
                current_metrics['usage_count'] = current_metrics.get('usage_count', 0) + 1
                current_metrics['last_used'] = update_time
            
            # Update performance ratings
            if 'rating' in metrics:
                old_rating = current_metrics.get('average_rating', 0.0)
                usage_count = current_metrics.get('usage_count', 1)
                new_rating = ((old_rating * (usage_count - 1)) + metrics['rating']) / usage_count
                current_metrics['average_rating'] = new_rating
                
                # Update model's performance score
                self.registered_models[model_uuid].performance_score = new_rating
            
            # Update success rate
            if 'success' in metrics:
                old_success_rate = current_metrics.get('success_rate', 0.0)
                usage_count = current_metrics.get('usage_count', 1)
                success_increment = 1.0 if metrics['success'] else 0.0
                new_success_rate = ((old_success_rate * (usage_count - 1)) + success_increment) / usage_count
                current_metrics['success_rate'] = new_success_rate
            
            # Update response time
            if 'response_time' in metrics:
                old_response_time = current_metrics.get('response_time_avg', 0.0)
                usage_count = current_metrics.get('usage_count', 1)
                new_response_time = ((old_response_time * (usage_count - 1)) + metrics['response_time']) / usage_count
                current_metrics['response_time_avg'] = new_response_time
            
            # Store additional custom metrics
            for key, value in metrics.items():
                if key not in ['usage_increment', 'rating', 'success', 'response_time']:
                    current_metrics[key] = value
            
            current_metrics['last_updated'] = update_time
            self.performance_metrics[model_uuid] = current_metrics
            
            print(f"📊 Updated performance metrics for model: {model_id}")
            print(f"   - Usage count: {current_metrics.get('usage_count', 0)}")
            print(f"   - Average rating: {current_metrics.get('average_rating', 0.0):.3f}")
            print(f"   - Success rate: {current_metrics.get('success_rate', 0.0):.3f}")
            
            # Reward model usage
            if 'usage_increment' in metrics and 'user_id' in metrics:
                await ftns_service.charge_context_access(metrics['user_id'], 10)  # Small usage fee
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating performance metrics for {model_id}: {e}")
            return False
    
    # === Advanced Registry Functions ===
    
    async def get_model_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive details about a registered model"""
        try:
            model_uuid = UUID(model_id)
            
            if model_uuid not in self.registered_models:
                return None
            
            model = self.registered_models[model_uuid]
            cid = self.model_cids.get(model_uuid)
            metrics = self.performance_metrics.get(model_uuid, {})
            
            return {
                "model": model.model_dump(),
                "ipfs_cid": cid,
                "performance_metrics": metrics,
                "registration_status": "active",
                "integrity_verified": await self.validate_model_integrity(cid) if cid else False
            }
            
        except Exception as e:
            print(f"❌ Error getting model details for {model_id}: {e}")
            return None
    
    async def search_models(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Advanced model search with filters and ranking
        
        Args:
            query: Search query (model name, specialization, etc.)
            filters: Optional filters (performance_min, model_type, etc.)
            
        Returns:
            List of matching models with details
        """
        try:
            results = []
            query_lower = query.lower()
            filters = filters or {}
            
            for model_id, model in self.registered_models.items():
                # Text matching
                matches_query = (
                    query_lower in model.name.lower() or
                    query_lower in model.specialization.lower()
                )
                
                if not matches_query:
                    continue
                
                # Apply filters
                if 'performance_min' in filters:
                    # Use updated performance score from metrics if available
                    current_performance = model.performance_score
                    if model_id in self.performance_metrics:
                        current_performance = self.performance_metrics[model_id].get('average_rating', model.performance_score)
                    
                    if current_performance < filters['performance_min']:
                        continue
                
                if 'model_type' in filters:
                    if model.model_type != filters['model_type']:
                        continue
                
                if 'active_only' in filters and filters['active_only']:
                    if not model.active:
                        continue
                
                # Get full model details
                details = await self.get_model_details(str(model_id))
                if details:
                    results.append(details)
            
            # Sort by performance score (descending)
            results.sort(key=lambda x: x['model']['performance_score'], reverse=True)
            
            return results[:MAX_SEARCH_RESULTS]
            
        except Exception as e:
            print(f"❌ Error searching models with query '{query}': {e}")
            return []
    
    async def get_top_performers(self, category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing models in a category or overall"""
        try:
            candidates = []
            
            for model_id, model in self.registered_models.items():
                if category and model.specialization.lower() != category.lower():
                    continue
                
                metrics = self.performance_metrics.get(model_id, {})
                performance_score = metrics.get('average_rating', model.performance_score)
                usage_count = metrics.get('usage_count', 0)
                
                # Weighted score considering both performance and usage
                weighted_score = performance_score * (1 + min(usage_count / 100, 1.0))
                
                candidates.append({
                    'model_id': str(model_id),
                    'model': model,
                    'performance_score': performance_score,
                    'usage_count': usage_count,
                    'weighted_score': weighted_score,
                    'metrics': metrics
                })
            
            # Sort by weighted score
            candidates.sort(key=lambda x: x['weighted_score'], reverse=True)
            
            return candidates[:limit]
            
        except Exception as e:
            print(f"❌ Error getting top performers: {e}")
            return []
    
    # === P2P Federation Functions ===
    
    async def join_federation(self, peer_node: PeerNode) -> bool:
        """Join a P2P federation network"""
        try:
            self.federation_nodes[peer_node.node_id] = peer_node
            print(f"🌐 Joined federation network: {peer_node.node_id}")
            
            # Sync models with peer
            if ENABLE_P2P_DISCOVERY:
                await self._sync_with_peer(peer_node.node_id)
            
            return True
            
        except Exception as e:
            print(f"❌ Error joining federation: {e}")
            return False
    
    async def leave_federation(self, peer_id: str) -> bool:
        """Leave a P2P federation network"""
        try:
            if peer_id in self.federation_nodes:
                del self.federation_nodes[peer_id]
                
            if peer_id in self.peer_registries:
                del self.peer_registries[peer_id]
            
            print(f"🌐 Left federation network: {peer_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error leaving federation: {e}")
            return False
    
    # === Private Helper Methods ===
    
    async def _search_local_specialists(self, task_category: str) -> List[str]:
        """Search local registry for specialists"""
        category_lower = task_category.lower()
        
        # Direct category match
        if category_lower in self.category_index:
            return [str(model_id) for model_id in self.category_index[category_lower]]
        
        # Fuzzy matching
        matches = []
        for category, model_ids in self.category_index.items():
            if category_lower in category or category in category_lower:
                matches.extend([str(model_id) for model_id in model_ids])
        
        return matches
    
    async def _search_federated_specialists(self, task_category: str) -> List[str]:
        """Search federated registries for specialists"""
        federated_matches = []
        
        for peer_id, peer_models in self.peer_registries.items():
            for model_id, model in peer_models.items():
                if task_category.lower() in model.specialization.lower():
                    federated_matches.append(f"{peer_id}:{model_id}")
        
        return federated_matches
    
    async def _sort_specialists_by_performance(self, specialist_ids: List[str]) -> List[str]:
        """Sort specialists by performance metrics"""
        scored_specialists = []
        
        for specialist_id in specialist_ids:
            # Handle federated model IDs
            if ':' in specialist_id:
                peer_id, model_id = specialist_id.split(':', 1)
                # For now, use base performance score from federated models
                # In full implementation, would sync performance metrics
                score = 0.5  # Default score for federated models
            else:
                try:
                    model_uuid = UUID(specialist_id)
                    if model_uuid in self.performance_metrics:
                        metrics = self.performance_metrics[model_uuid]
                        score = metrics.get('average_rating', 0.0)
                    else:
                        model = self.registered_models.get(model_uuid)
                        score = model.performance_score if model else 0.0
                except ValueError:
                    score = 0.0
            
            scored_specialists.append((specialist_id, score))
        
        # Sort by score (descending)
        scored_specialists.sort(key=lambda x: x[1], reverse=True)
        
        return [specialist_id for specialist_id, _ in scored_specialists]
    
    async def _broadcast_model_registration(self, model: TeacherModel, cid: str) -> None:
        """Broadcast model registration to federation peers"""
        # Placeholder for P2P broadcasting
        # In full implementation, would send registration to all federation peers
        print(f"📡 Broadcasting model registration: {model.name}")
    
    async def _sync_with_peer(self, peer_id: str) -> None:
        """Sync model registry with a federation peer"""
        # Placeholder for P2P synchronization
        # In full implementation, would exchange model listings with peer
        self.stats["federation_syncs"] += 1
        print(f"🔄 Syncing with federation peer: {peer_id}")
    
    # === Public Status Methods ===
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics"""
        return {
            **self.stats,
            "categories": list(self.category_index.keys()),
            "federation_peers": len(self.federation_nodes),
            "cache_size": len(self.search_cache),
            "integrity_cache_size": len(self.integrity_cache),
            "avg_performance": sum(
                model.performance_score for model in self.registered_models.values()
            ) / max(len(self.registered_models), 1)
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get registry health and operational status"""
        healthy_models = 0
        
        for model_id, cid in self.model_cids.items():
            if await self.validate_model_integrity(cid):
                healthy_models += 1
        
        health_percentage = (healthy_models / max(len(self.model_cids), 1)) * 100
        
        return {
            "status": "healthy" if health_percentage > 90 else "degraded" if health_percentage > 50 else "unhealthy",
            "health_percentage": health_percentage,
            "total_models": len(self.registered_models),
            "healthy_models": healthy_models,
            "federation_connected": len(self.federation_nodes) > 0,
            "ipfs_connected": prsm_ipfs_client.connected,
            "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["total_searches"], 1) * 100
        }


# === Global Registry Instance ===
model_registry = ModelRegistry()