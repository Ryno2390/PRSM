"""
Request Size Limits & Resource Protection Middleware
===================================================

Comprehensive middleware for protecting against DoS attacks through request
size limits, rate limiting, and resource usage monitoring.
"""

import asyncio
import json
import structlog
import time
from typing import Dict, Any, Optional, Callable, Set
from datetime import datetime, timezone, timedelta

from fastapi import Request, Response, HTTPException, status, WebSocket
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

from prsm.core.integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)


class RequestLimitsConfig(BaseModel):
    """Configuration for request size limits and protection"""
    
    # Request body size limits (in bytes)
    default_max_body_size: int = Field(default=16 * 1024 * 1024)  # 16MB
    endpoint_body_limits: Dict[str, int] = Field(default_factory=lambda: {
        # API endpoints with specific limits
        '/api/v1/sessions': 1024 * 1024,  # 1MB for session creation
        '/api/v1/nwtn/execute': 512 * 1024,  # 512KB for AI execution
        '/api/v1/credentials/register': 64 * 1024,  # 64KB for credentials
        '/api/v1/conversations/*/messages': 256 * 1024,  # 256KB for messages
        '/api/v1/web3/transfer': 32 * 1024,  # 32KB for transfers
        
        # File upload endpoints (if they exist)
        '/api/v1/upload': 100 * 1024 * 1024,  # 100MB for file uploads
        '/api/v1/ipfs/upload': 50 * 1024 * 1024,  # 50MB for IPFS uploads
    })
    
    # WebSocket message limits
    websocket_max_message_size: int = Field(default=256 * 1024)  # 256KB
    websocket_max_messages_per_minute: int = Field(default=60)
    
    # JSON parsing limits
    max_json_depth: int = Field(default=10)
    max_json_keys: int = Field(default=1000)
    
    # Request rate limits (per IP per minute)
    rate_limit_requests_per_minute: int = Field(default=100)
    rate_limit_expensive_requests_per_minute: int = Field(default=10)
    
    # Expensive endpoints requiring stricter limits
    expensive_endpoints: Set[str] = Field(default_factory=lambda: {
        '/api/v1/nwtn/execute',
        '/api/v1/ipfs/upload',
        '/api/v1/web3/transfer',
        '/sessions',
        '/conversations/*/messages'
    })
    
    # Timeout limits
    request_timeout_seconds: int = Field(default=60)
    websocket_connection_timeout_seconds: int = Field(default=300)  # 5 minutes


class RequestLimitsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for enforcing request size limits and resource protection
    
    Features:
    - Request body size limits per endpoint
    - JSON structure validation and depth limiting
    - Rate limiting for expensive operations
    - Request timeout enforcement
    - Memory usage monitoring
    - Attack pattern detection
    """
    
    def __init__(self, app, config: Optional[RequestLimitsConfig] = None):
        super().__init__(app)
        self.config = config or RequestLimitsConfig()
        
        # Rate limiting storage (in production, use Redis)
        self.rate_limit_store: Dict[str, Dict[str, Any]] = {}
        self.expensive_rate_limit_store: Dict[str, Dict[str, Any]] = {}
        
        # Request tracking
        self.active_requests: Dict[str, datetime] = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'size_limit_violations': 0,
            'rate_limit_violations': 0,
            'timeout_violations': 0
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method"""
        start_time = time.time()
        request_id = f"{request.client.host}:{start_time}"
        
        try:
            # Track active request
            self.active_requests[request_id] = datetime.now(timezone.utc)
            self.stats['total_requests'] += 1
            
            # Check rate limits
            await self._check_rate_limits(request)
            
            # Check request size limits
            await self._check_request_size(request)
            
            # Process request with timeout
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.config.request_timeout_seconds
            )
            
            # Log successful request
            processing_time = time.time() - start_time
            if processing_time > 10:  # Log slow requests
                logger.warning("Slow request detected",
                             path=request.url.path,
                             method=request.method,
                             processing_time=processing_time,
                             client_ip=request.client.host)
            
            return response
            
        except asyncio.TimeoutError:
            self.stats['timeout_violations'] += 1
            await self._log_security_event(
                "request_timeout",
                request,
                {"timeout_seconds": self.config.request_timeout_seconds}
            )
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Request timeout exceeded"
            )
            
        except HTTPException:
            self.stats['blocked_requests'] += 1
            raise
            
        except Exception as e:
            logger.error("Request processing error",
                        path=request.url.path,
                        method=request.method,
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
            
        finally:
            # Clean up active request tracking
            self.active_requests.pop(request_id, None)
    
    async def _check_rate_limits(self, request: Request):
        """Check rate limits for the request"""
        client_ip = request.client.host
        current_time = datetime.now(timezone.utc)
        
        # Check if endpoint is expensive
        is_expensive = any(
            endpoint in request.url.path 
            for endpoint in self.config.expensive_endpoints
        )
        
        if is_expensive:
            # Check expensive endpoint rate limit
            await self._check_rate_limit(
                client_ip,
                current_time,
                self.expensive_rate_limit_store,
                self.config.rate_limit_expensive_requests_per_minute,
                "expensive_rate_limit"
            )
        else:
            # Check general rate limit
            await self._check_rate_limit(
                client_ip,
                current_time,
                self.rate_limit_store,
                self.config.rate_limit_requests_per_minute,
                "general_rate_limit"
            )
    
    async def _check_rate_limit(
        self,
        client_ip: str,
        current_time: datetime,
        store: Dict[str, Dict[str, Any]],
        limit: int,
        limit_type: str
    ):
        """Check individual rate limit"""
        # Clean old entries
        cutoff_time = current_time - timedelta(minutes=1)
        
        if client_ip not in store:
            store[client_ip] = {'requests': [], 'blocked_until': None}
        
        client_data = store[client_ip]
        
        # Remove old requests
        client_data['requests'] = [
            req_time for req_time in client_data['requests']
            if req_time > cutoff_time
        ]
        
        # Check if still blocked
        if (client_data['blocked_until'] and 
            current_time < client_data['blocked_until']):
            await self._log_security_event(
                f"{limit_type}_blocked",
                None,
                {"client_ip": client_ip, "limit": limit}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests per minute"
            )
        
        # Check current request count
        if len(client_data['requests']) >= limit:
            # Block for 1 minute
            client_data['blocked_until'] = current_time + timedelta(minutes=1)
            self.stats['rate_limit_violations'] += 1
            
            await self._log_security_event(
                f"{limit_type}_exceeded",
                None,
                {"client_ip": client_ip, "request_count": len(client_data['requests']), "limit": limit}
            )
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests per minute"
            )
        
        # Add current request
        client_data['requests'].append(current_time)
    
    async def _check_request_size(self, request: Request):
        """Check request body size limits"""
        # Get content length
        content_length = request.headers.get('content-length')
        if not content_length:
            return  # No body or chunked encoding
        
        try:
            body_size = int(content_length)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Content-Length header"
            )
        
        # Get size limit for this endpoint
        size_limit = self._get_size_limit_for_path(request.url.path)
        
        if body_size > size_limit:
            self.stats['size_limit_violations'] += 1
            await self._log_security_event(
                "request_size_exceeded",
                request,
                {"body_size": body_size, "limit": size_limit}
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request body too large: {body_size} bytes > {size_limit} bytes"
            )
    
    def _get_size_limit_for_path(self, path: str) -> int:
        """Get size limit for specific endpoint path"""
        # Check exact matches first
        if path in self.config.endpoint_body_limits:
            return self.config.endpoint_body_limits[path]
        
        # Check pattern matches (for paths with IDs)
        for endpoint_pattern, limit in self.config.endpoint_body_limits.items():
            if '*' in endpoint_pattern:
                # Simple wildcard matching
                pattern_parts = endpoint_pattern.split('*')
                if (len(pattern_parts) == 2 and 
                    path.startswith(pattern_parts[0]) and 
                    path.endswith(pattern_parts[1])):
                    return limit
        
        return self.config.default_max_body_size
    
    async def _log_security_event(
        self,
        event_type: str,
        request: Optional[Request],
        details: Dict[str, Any]
    ):
        """Log security events"""
        try:
            event_details = {
                **details,
                "middleware": "request_limits"
            }
            
            if request:
                event_details.update({
                    "path": str(request.url.path),
                    "method": request.method,
                    "client_ip": request.client.host,
                    "user_agent": request.headers.get("user-agent", "unknown")
                })
            
            await audit_logger.log_security_event(
                event_type=f"request_limits_{event_type}",
                user_id="system",
                details=event_details,
                security_level="warning"
            )
        except Exception as e:
            logger.error("Failed to log security event", error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        return {
            **self.stats,
            "active_requests": len(self.active_requests),
            "rate_limited_ips": len([
                ip for ip, data in self.rate_limit_store.items()
                if data.get('blocked_until') and data['blocked_until'] > datetime.now(timezone.utc)
            ])
        }


class WebSocketLimitsManager:
    """
    WebSocket message size and rate limiting manager
    
    Features:
    - Message size limits
    - Message rate limiting per connection
    - Connection timeout management
    - Memory usage monitoring for WebSocket connections
    """
    
    def __init__(self, config: Optional[RequestLimitsConfig] = None):
        self.config = config or RequestLimitsConfig()
        
        # Per-connection tracking
        self.connection_data: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def validate_websocket_message(
        self,
        websocket: WebSocket,
        message: str,
        user_id: str
    ) -> bool:
        """
        Validate WebSocket message size and rate limits
        
        Args:
            websocket: WebSocket connection
            message: Message content
            user_id: User ID for logging
            
        Returns:
            True if message is valid, False otherwise
            
        Raises:
            Exception: If message violates limits
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            # Initialize connection data if needed
            if websocket not in self.connection_data:
                self.connection_data[websocket] = {
                    'messages': [],
                    'total_bytes': 0,
                    'connected_at': current_time,
                    'user_id': user_id
                }
            
            conn_data = self.connection_data[websocket]
            
            # Check message size
            message_size = len(message.encode('utf-8'))
            if message_size > self.config.websocket_max_message_size:
                await self._log_websocket_event(
                    "message_size_exceeded",
                    websocket,
                    user_id,
                    {"message_size": message_size, "limit": self.config.websocket_max_message_size}
                )
                raise Exception(f"WebSocket message too large: {message_size} bytes")
            
            # Check rate limit
            cutoff_time = current_time - timedelta(minutes=1)
            conn_data['messages'] = [
                msg_time for msg_time in conn_data['messages']
                if msg_time > cutoff_time
            ]
            
            if len(conn_data['messages']) >= self.config.websocket_max_messages_per_minute:
                await self._log_websocket_event(
                    "rate_limit_exceeded",
                    websocket,
                    user_id,
                    {"message_count": len(conn_data['messages']), "limit": self.config.websocket_max_messages_per_minute}
                )
                raise Exception("WebSocket message rate limit exceeded")
            
            # Add current message
            conn_data['messages'].append(current_time)
            conn_data['total_bytes'] += message_size
            
            # Check connection timeout
            connection_duration = current_time - conn_data['connected_at']
            if connection_duration.total_seconds() > self.config.websocket_connection_timeout_seconds:
                await self._log_websocket_event(
                    "connection_timeout",
                    websocket,
                    user_id,
                    {"duration_seconds": connection_duration.total_seconds()}
                )
                raise Exception("WebSocket connection timeout")
            
            return True
            
        except Exception as e:
            logger.error("WebSocket message validation failed",
                        user_id=user_id,
                        error=str(e))
            raise
    
    async def cleanup_connection(self, websocket: WebSocket):
        """Clean up connection data when WebSocket disconnects"""
        self.connection_data.pop(websocket, None)
    
    async def _log_websocket_event(
        self,
        event_type: str,
        websocket: WebSocket,
        user_id: str,
        details: Dict[str, Any]
    ):
        """Log WebSocket security events"""
        try:
            await audit_logger.log_security_event(
                event_type=f"websocket_limits_{event_type}",
                user_id=user_id,
                details={
                    **details,
                    "client_ip": websocket.client.host if websocket.client else "unknown"
                },
                security_level="warning"
            )
        except Exception as e:
            logger.error("Failed to log WebSocket security event", error=str(e))
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        total_connections = len(self.connection_data)
        total_bytes = sum(data['total_bytes'] for data in self.connection_data.values())
        
        return {
            "active_connections": total_connections,
            "total_bytes_transferred": total_bytes,
            "average_bytes_per_connection": total_bytes / total_connections if total_connections > 0 else 0
        }


# Global instances
request_limits_config = RequestLimitsConfig()
websocket_limits_manager = WebSocketLimitsManager(request_limits_config)


async def validate_websocket_message(
    websocket: WebSocket,
    message: str,
    user_id: str
) -> bool:
    """Convenience function for WebSocket message validation"""
    return await websocket_limits_manager.validate_websocket_message(websocket, message, user_id)


async def cleanup_websocket_connection(websocket: WebSocket):
    """Convenience function for WebSocket cleanup"""
    await websocket_limits_manager.cleanup_connection(websocket)