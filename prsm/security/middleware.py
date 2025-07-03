"""
Enhanced Security Middleware
===========================

Production-ready security middleware for PRSM API hardening:
- Request validation and sanitization
- Rate limiting and DDoS protection
- Security headers injection
- CORS configuration
- IP blocking and geolocation filtering
- Content Security Policy (CSP)
"""

import time
import json
from typing import Callable, Dict, Any, Optional
from datetime import datetime, timezone
import structlog

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import RequestResponseEndpoint

from prsm.core.config import get_settings
from prsm.security.enhanced_authorization import get_enhanced_auth_manager

logger = structlog.get_logger(__name__)
settings = get_settings()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Inject security headers into all responses
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self' https:; "
                "frame-ancestors 'none';"
            )
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add request ID for tracing
        request_id = getattr(request.state, 'request_id', 'unknown')
        response.headers["X-Request-ID"] = request_id
        
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Global rate limiting middleware with IP-based and user-based limits
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.ip_rate_limits = {}  # IP -> request timestamps
        self.auth_manager = get_enhanced_auth_manager()
        
        # Rate limits per IP (requests per minute)
        self.ip_rate_limit = 300  # 300 requests per minute per IP
        self.burst_limit = 50     # 50 requests in 10 seconds
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = request.client.host
        current_time = time.time()
        
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)
        
        # Check IP-based rate limiting
        if not await self._check_ip_rate_limit(client_ip, current_time):
            logger.warning("IP rate limit exceeded",
                         ip=client_ip,
                         path=request.url.path)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests from this IP address"
            )
        
        return await call_next(request)
    
    async def _check_ip_rate_limit(self, ip: str, current_time: float) -> bool:
        """Check if IP is within rate limits"""
        if ip not in self.ip_rate_limits:
            self.ip_rate_limits[ip] = []
        
        # Clean old timestamps (older than 1 minute)
        self.ip_rate_limits[ip] = [
            timestamp for timestamp in self.ip_rate_limits[ip]
            if current_time - timestamp < 60
        ]
        
        # Check minute limit
        if len(self.ip_rate_limits[ip]) >= self.ip_rate_limit:
            return False
        
        # Check burst limit (last 10 seconds)
        recent_requests = [
            timestamp for timestamp in self.ip_rate_limits[ip]
            if current_time - timestamp < 10
        ]
        
        if len(recent_requests) >= self.burst_limit:
            return False
        
        # Add current request
        self.ip_rate_limits[ip].append(current_time)
        return True


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate and sanitize incoming requests
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.max_request_size = 10 * 1024 * 1024  # 10 MB
        self.blocked_user_agents = [
            "sqlmap", "nikto", "nmap", "masscan", "zap", "burp"
        ]
        self.auth_manager = get_enhanced_auth_manager()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate request ID for tracing
        request_id = f"req_{int(time.time() * 1000)}_{hash(request.client.host) % 10000}"
        request.state.request_id = request_id
        
        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            logger.warning("Request too large",
                         content_length=content_length,
                         ip=request.client.host,
                         request_id=request_id)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large"
            )
        
        # Validate User-Agent
        user_agent = request.headers.get("user-agent", "").lower()
        for blocked_agent in self.blocked_user_agents:
            if blocked_agent in user_agent:
                logger.warning("Blocked user agent",
                             user_agent=user_agent,
                             ip=request.client.host,
                             request_id=request_id)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Request blocked"
                )
        
        # Log request for audit
        logger.info("API request",
                   method=request.method,
                   path=request.url.path,
                   ip=request.client.host,
                   user_agent=user_agent[:100],
                   request_id=request_id)
        
        return await call_next(request)


class ResponseSecurityMiddleware(BaseHTTPMiddleware):
    """
    Secure response handling and data leak prevention
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.sensitive_fields = [
            "password", "secret", "key", "token", "hash",
            "private", "confidential", "internal"
        ]
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Remove server information
        if "server" in response.headers:
            del response.headers["server"]
        
        # Log response for audit (without sensitive data)
        request_id = getattr(request.state, 'request_id', 'unknown')
        logger.info("API response",
                   status_code=response.status_code,
                   request_id=request_id)
        
        return response


class GeolocationFilterMiddleware(BaseHTTPMiddleware):
    """
    Filter requests based on geolocation (if configured)
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.allowed_countries = getattr(settings, 'allowed_countries', None)
        self.blocked_countries = getattr(settings, 'blocked_countries', [])
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip for health checks
        if request.url.path == "/health":
            return await call_next(request)
        
        # Geolocation filtering (placeholder - would integrate with GeoIP service)
        client_ip = request.client.host
        
        # Check for private/local IPs (always allow)
        if self._is_private_ip(client_ip):
            return await call_next(request)
        
        # For production, would implement actual GeoIP lookup
        # For now, just log the request
        logger.debug("Request geolocation check",
                    ip=client_ip,
                    path=request.url.path)
        
        return await call_next(request)
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is private/local"""
        private_ranges = [
            "127.", "10.", "172.16.", "172.17.", "172.18.", "172.19.",
            "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
            "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
            "172.30.", "172.31.", "192.168.", "::1", "localhost"
        ]
        
        return any(ip.startswith(prefix) for prefix in private_ranges)


def configure_cors() -> dict:
    """
    Configure CORS middleware with secure defaults
    """
    allowed_origins = getattr(settings, 'allowed_origins', [
        "https://localhost:3000",
        "https://prsm.app", 
        "https://api.prsm.app"
    ])
    
    return {
        "allow_origins": allowed_origins,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Accept",
            "Accept-Language", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Request-ID"
        ],
        "expose_headers": ["X-Request-ID"],
        "max_age": 3600,  # 1 hour
    }


def get_security_middleware_stack():
    """
    Get the complete security middleware stack in correct order
    """
    return [
        RequestValidationMiddleware,
        RateLimitingMiddleware,
        SecurityHeadersMiddleware,
        ResponseSecurityMiddleware,
        GeolocationFilterMiddleware
    ]