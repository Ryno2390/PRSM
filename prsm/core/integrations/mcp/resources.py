"""
MCP Resource Management
======================

Resource management for MCP servers (external data sources, files, APIs).
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from urllib.parse import urlparse
import hashlib

from .models import ResourceDefinition, MCPResource, MCPRequest, MCPMethod

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manages MCP resources (external data sources)
    
    Features:
    - Resource discovery and caching
    - Content retrieval and validation
    - Access control and permissions
    - Cache management and expiration
    """
    
    def __init__(self):
        self.resources: Dict[str, ResourceDefinition] = {}  # uri -> definition
        self.resource_cache: Dict[str, MCPResource] = {}  # uri -> cached resource
        self.access_log: List[Dict[str, Any]] = []
        
        # Configuration
        self.default_cache_ttl = timedelta(minutes=30)
        self.max_cache_size = 100  # Maximum cached resources
        self.max_resource_size = 10 * 1024 * 1024  # 10MB limit
        
        # Access control
        self.resource_permissions: Dict[str, Dict[str, Any]] = {}  # uri -> permissions
        
        logger.info("Initialized MCP resource manager")
    
    async def register_resources(self, resources: List[ResourceDefinition]) -> int:
        """
        Register multiple resources from an MCP server
        
        Args:
            resources: List of resource definitions
            
        Returns:
            Number of resources successfully registered
        """
        registered_count = 0
        
        for resource in resources:
            try:
                self.resources[resource.uri] = resource
                registered_count += 1
                
                # Set default permissions
                if resource.uri not in self.resource_permissions:
                    self.resource_permissions[resource.uri] = {
                        "read": True,
                        "cache": True,
                        "public": True
                    }
                
                logger.debug(f"Registered resource: {resource.name} ({resource.uri})")
                
            except Exception as e:
                logger.warning(f"Failed to register resource {resource.uri}: {e}")
        
        logger.info(f"Registered {registered_count}/{len(resources)} resources")
        return registered_count
    
    async def get_resource(self, uri: str, user_id: str, 
                         force_refresh: bool = False) -> Optional[MCPResource]:
        """
        Get resource content by URI
        
        Args:
            uri: Resource URI
            user_id: ID of user requesting the resource
            force_refresh: Whether to force refresh from source
            
        Returns:
            Resource with content, or None if not found/accessible
        """
        # Check if resource exists
        if uri not in self.resources:
            logger.warning(f"Resource not found: {uri}")
            return None
        
        # Check permissions
        if not await self._check_access_permission(uri, user_id, "read"):
            logger.warning(f"Access denied to resource {uri} for user {user_id}")
            return None
        
        # Check cache first (unless force refresh)
        if not force_refresh and uri in self.resource_cache:
            cached_resource = self.resource_cache[uri]
            if not self._is_cache_expired(cached_resource):
                logger.debug(f"Returning cached resource: {uri}")
                await self._log_access(uri, user_id, "cache_hit")
                return cached_resource
        
        # Fetch from source
        try:
            resource = await self._fetch_resource(uri, user_id)
            
            # Cache if allowed
            if self._can_cache_resource(uri) and resource:
                await self._cache_resource(resource)
            
            await self._log_access(uri, user_id, "fetch")
            return resource
            
        except Exception as e:
            logger.error(f"Failed to fetch resource {uri}: {e}")
            await self._log_access(uri, user_id, "error", str(e))
            return None
    
    async def list_resources(self, user_id: str, 
                           category: Optional[str] = None,
                           mime_type: Optional[str] = None) -> List[ResourceDefinition]:
        """
        List available resources for a user
        
        Args:
            user_id: ID of user requesting the list
            category: Optional category filter
            mime_type: Optional MIME type filter
            
        Returns:
            List of accessible resource definitions
        """
        accessible_resources = []
        
        for uri, resource in self.resources.items():
            # Check access permission
            if not await self._check_access_permission(uri, user_id, "read"):
                continue
            
            # Apply filters
            if category and resource.annotations.get("category") != category:
                continue
            
            if mime_type and resource.mime_type != mime_type:
                continue
            
            accessible_resources.append(resource)
        
        logger.debug(f"Listed {len(accessible_resources)} resources for user {user_id}")
        return accessible_resources
    
    async def search_resources(self, query: str, user_id: str, 
                             limit: int = 10) -> List[ResourceDefinition]:
        """
        Search resources by name, description, or annotations
        
        Args:
            query: Search query string
            user_id: ID of user performing search
            limit: Maximum number of results
            
        Returns:
            List of matching resource definitions
        """
        query_lower = query.lower()
        matching_resources = []
        
        for uri, resource in self.resources.items():
            # Check access permission
            if not await self._check_access_permission(uri, user_id, "read"):
                continue
            
            score = 0
            
            # Search in name (highest priority)
            if query_lower in resource.name.lower():
                score += 3
            
            # Search in description
            if resource.description and query_lower in resource.description.lower():
                score += 2
            
            # Search in URI
            if query_lower in resource.uri.lower():
                score += 1
            
            # Search in annotations
            for key, value in resource.annotations.items():
                if query_lower in str(value).lower():
                    score += 1
            
            if score > 0:
                matching_resources.append((resource, score))
        
        # Sort by relevance score
        matching_resources.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = [resource for resource, _ in matching_resources[:limit]]
        logger.debug(f"Found {len(results)} matching resources for query '{query}'")
        return results
    
    async def set_resource_permissions(self, uri: str, permissions: Dict[str, Any]):
        """Set permissions for a resource"""
        if uri in self.resources:
            self.resource_permissions[uri] = permissions
            logger.info(f"Updated permissions for resource {uri}")
    
    async def get_resource_statistics(self) -> Dict[str, Any]:
        """Get resource management statistics"""
        now = datetime.utcnow()
        
        # Cache statistics
        cached_count = len(self.resource_cache)
        expired_count = sum(1 for r in self.resource_cache.values() if self._is_cache_expired(r))
        
        # Access statistics
        recent_accesses = [
            log for log in self.access_log
            if log["timestamp"] > now - timedelta(hours=24)
        ]
        
        cache_hits = sum(1 for log in recent_accesses if log["action"] == "cache_hit")
        fetches = sum(1 for log in recent_accesses if log["action"] == "fetch")
        cache_hit_rate = (cache_hits / max(cache_hits + fetches, 1)) * 100
        
        # Resource type distribution
        mime_types = {}
        for resource in self.resources.values():
            mime_type = resource.mime_type or "unknown"
            mime_types[mime_type] = mime_types.get(mime_type, 0) + 1
        
        return {
            "total_resources": len(self.resources),
            "cached_resources": cached_count,
            "expired_cache_entries": expired_count,
            "cache_hit_rate": cache_hit_rate,
            "recent_accesses": len(recent_accesses),
            "mime_type_distribution": mime_types,
            "total_access_logs": len(self.access_log)
        }
    
    async def cleanup_cache(self) -> int:
        """Clean up expired cache entries"""
        expired_uris = [
            uri for uri, resource in self.resource_cache.items()
            if self._is_cache_expired(resource)
        ]
        
        for uri in expired_uris:
            del self.resource_cache[uri]
        
        if expired_uris:
            logger.info(f"Cleaned up {len(expired_uris)} expired cache entries")
        
        return len(expired_uris)
    
    # Private methods
    
    async def _fetch_resource(self, uri: str, user_id: str) -> Optional[MCPResource]:
        """Fetch resource content from MCP server"""
        resource_def = self.resources[uri]
        
        # This would integrate with the MCP client to fetch the actual content
        # For now, we'll create a mock resource
        
        # Simulate fetching content (in real implementation, would use MCP client)
        logger.debug(f"Fetching resource content for {uri}")
        
        # Create resource with mock content
        resource = MCPResource(
            definition=resource_def,
            content=f"Mock content for resource: {resource_def.name}",
            cached_at=datetime.utcnow(),
            cache_expires=datetime.utcnow() + self.default_cache_ttl
        )
        
        return resource
    
    async def _cache_resource(self, resource: MCPResource):
        """Cache a resource"""
        uri = resource.definition.uri
        
        # Check cache size limits
        if len(self.resource_cache) >= self.max_cache_size:
            await self._evict_oldest_cache_entry()
        
        self.resource_cache[uri] = resource
        logger.debug(f"Cached resource: {uri}")
    
    async def _evict_oldest_cache_entry(self):
        """Evict the oldest cache entry"""
        if not self.resource_cache:
            return
        
        oldest_uri = min(
            self.resource_cache.keys(),
            key=lambda uri: self.resource_cache[uri].cached_at or datetime.min
        )
        
        del self.resource_cache[oldest_uri]
        logger.debug(f"Evicted oldest cache entry: {oldest_uri}")
    
    def _is_cache_expired(self, resource: MCPResource) -> bool:
        """Check if cached resource has expired"""
        if not resource.cache_expires:
            return False
        
        return datetime.utcnow() > resource.cache_expires
    
    def _can_cache_resource(self, uri: str) -> bool:
        """Check if resource can be cached"""
        permissions = self.resource_permissions.get(uri, {})
        return permissions.get("cache", True)
    
    async def _check_access_permission(self, uri: str, user_id: str, action: str) -> bool:
        """Check if user has permission to perform action on resource"""
        permissions = self.resource_permissions.get(uri, {})
        
        # Check basic permission
        if not permissions.get(action, True):
            return False
        
        # Check if resource is public or user has specific access
        if permissions.get("public", True):
            return True
        
        # Check user-specific permissions (would integrate with user management)
        user_permissions = permissions.get("users", {}).get(user_id, {})
        return user_permissions.get(action, False)
    
    async def _log_access(self, uri: str, user_id: str, action: str, error: Optional[str] = None):
        """Log resource access"""
        log_entry = {
            "timestamp": datetime.utcnow(),
            "uri": uri,
            "user_id": user_id,
            "action": action,
            "error": error
        }
        
        self.access_log.append(log_entry)
        
        # Keep log size manageable
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-500:]  # Keep last 500 entries
    
    def get_cached_resource_info(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get information about cached resource"""
        if uri not in self.resource_cache:
            return None
        
        resource = self.resource_cache[uri]
        return {
            "uri": uri,
            "cached_at": resource.cached_at.isoformat() if resource.cached_at else None,
            "expires": resource.cache_expires.isoformat() if resource.cache_expires else None,
            "expired": self._is_cache_expired(resource),
            "content_size": len(str(resource.content)) if resource.content else 0,
            "encoding": resource.encoding
        }
    
    async def invalidate_cache(self, uri: Optional[str] = None) -> int:
        """
        Invalidate cache entries
        
        Args:
            uri: Specific URI to invalidate, or None to clear all cache
            
        Returns:
            Number of cache entries invalidated
        """
        if uri:
            if uri in self.resource_cache:
                del self.resource_cache[uri]
                logger.info(f"Invalidated cache for resource: {uri}")
                return 1
            return 0
        else:
            count = len(self.resource_cache)
            self.resource_cache.clear()
            logger.info(f"Cleared all cached resources ({count} entries)")
            return count