"""
PRSM Advanced Rate Limiting System
Enterprise-grade rate limiting with multiple tiers and intelligent threat detection
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import hashlib
import ipaddress
from collections import defaultdict, deque
import time
import math

from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as aioredis
import logging

logger = logging.getLogger(__name__)


class UserTier(Enum):
    """User subscription tiers with different rate limits"""
    GUEST = "guest"
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


class RateLimitType(Enum):
    """Types of rate limiting"""
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    CONCURRENT_REQUESTS = "concurrent_requests"
    BANDWIDTH_PER_HOUR = "bandwidth_per_hour"
    FTNS_SPEND_PER_HOUR = "ftns_spend_per_hour"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    limit_type: RateLimitType
    limit: int
    window_seconds: int
    burst_allowance: int = 0  # Allow brief bursts above limit
    cooldown_seconds: int = 0  # Cooldown period after limit hit


@dataclass
class TierLimits:
    """Rate limits for a specific user tier"""
    tier: UserTier
    limits: Dict[RateLimitType, RateLimit] = field(default_factory=dict)
    priority_weight: float = 1.0  # Higher = more priority during congestion
    
    def __post_init__(self):
        """Initialize default limits for each tier"""
        if not self.limits:
            self.limits = self._get_default_limits_for_tier()
    
    def _get_default_limits_for_tier(self) -> Dict[RateLimitType, RateLimit]:
        """Get default rate limits based on tier"""
        
        base_limits = {
            UserTier.GUEST: {
                RateLimitType.REQUESTS_PER_MINUTE: RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 10, 60),
                RateLimitType.REQUESTS_PER_HOUR: RateLimit(RateLimitType.REQUESTS_PER_HOUR, 100, 3600),
                RateLimitType.REQUESTS_PER_DAY: RateLimit(RateLimitType.REQUESTS_PER_DAY, 1000, 86400),
                RateLimitType.CONCURRENT_REQUESTS: RateLimit(RateLimitType.CONCURRENT_REQUESTS, 2, 0),
                RateLimitType.BANDWIDTH_PER_HOUR: RateLimit(RateLimitType.BANDWIDTH_PER_HOUR, 10485760, 3600),  # 10MB
            },
            UserTier.FREE: {
                RateLimitType.REQUESTS_PER_MINUTE: RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 60, 60, burst_allowance=10),
                RateLimitType.REQUESTS_PER_HOUR: RateLimit(RateLimitType.REQUESTS_PER_HOUR, 1000, 3600),
                RateLimitType.REQUESTS_PER_DAY: RateLimit(RateLimitType.REQUESTS_PER_DAY, 10000, 86400),
                RateLimitType.CONCURRENT_REQUESTS: RateLimit(RateLimitType.CONCURRENT_REQUESTS, 5, 0),
                RateLimitType.BANDWIDTH_PER_HOUR: RateLimit(RateLimitType.BANDWIDTH_PER_HOUR, 104857600, 3600),  # 100MB
                RateLimitType.FTNS_SPEND_PER_HOUR: RateLimit(RateLimitType.FTNS_SPEND_PER_HOUR, 100, 3600),
            },
            UserTier.PRO: {
                RateLimitType.REQUESTS_PER_MINUTE: RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 300, 60, burst_allowance=50),
                RateLimitType.REQUESTS_PER_HOUR: RateLimit(RateLimitType.REQUESTS_PER_HOUR, 10000, 3600),
                RateLimitType.REQUESTS_PER_DAY: RateLimit(RateLimitType.REQUESTS_PER_DAY, 100000, 86400),
                RateLimitType.CONCURRENT_REQUESTS: RateLimit(RateLimitType.CONCURRENT_REQUESTS, 20, 0),
                RateLimitType.BANDWIDTH_PER_HOUR: RateLimit(RateLimitType.BANDWIDTH_PER_HOUR, 1073741824, 3600),  # 1GB
                RateLimitType.FTNS_SPEND_PER_HOUR: RateLimit(RateLimitType.FTNS_SPEND_PER_HOUR, 1000, 3600),
            },
            UserTier.ENTERPRISE: {
                RateLimitType.REQUESTS_PER_MINUTE: RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 1000, 60, burst_allowance=200),
                RateLimitType.REQUESTS_PER_HOUR: RateLimit(RateLimitType.REQUESTS_PER_HOUR, 50000, 3600),
                RateLimitType.REQUESTS_PER_DAY: RateLimit(RateLimitType.REQUESTS_PER_DAY, 1000000, 86400),
                RateLimitType.CONCURRENT_REQUESTS: RateLimit(RateLimitType.CONCURRENT_REQUESTS, 100, 0),
                RateLimitType.BANDWIDTH_PER_HOUR: RateLimit(RateLimitType.BANDWIDTH_PER_HOUR, 10737418240, 3600),  # 10GB
                RateLimitType.FTNS_SPEND_PER_HOUR: RateLimit(RateLimitType.FTNS_SPEND_PER_HOUR, 10000, 3600),
            },
            UserTier.ADMIN: {
                RateLimitType.REQUESTS_PER_MINUTE: RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 10000, 60, burst_allowance=1000),
                RateLimitType.REQUESTS_PER_HOUR: RateLimit(RateLimitType.REQUESTS_PER_HOUR, 500000, 3600),
                RateLimitType.REQUESTS_PER_DAY: RateLimit(RateLimitType.REQUESTS_PER_DAY, 10000000, 86400),
                RateLimitType.CONCURRENT_REQUESTS: RateLimit(RateLimitType.CONCURRENT_REQUESTS, 500, 0),
                RateLimitType.BANDWIDTH_PER_HOUR: RateLimit(RateLimitType.BANDWIDTH_PER_HOUR, 107374182400, 3600),  # 100GB
                RateLimitType.FTNS_SPEND_PER_HOUR: RateLimit(RateLimitType.FTNS_SPEND_PER_HOUR, 100000, 3600),
            }
        }
        
        return base_limits.get(self.tier, base_limits[UserTier.FREE])


@dataclass
class RateLimitState:
    """Current rate limit state for a user/IP"""
    user_id: Optional[str]
    ip_address: str
    current_requests: Dict[RateLimitType, int] = field(default_factory=dict)
    window_start: Dict[RateLimitType, datetime] = field(default_factory=dict)
    burst_tokens: Dict[RateLimitType, int] = field(default_factory=dict)
    cooldown_until: Dict[RateLimitType, Optional[datetime]] = field(default_factory=dict)
    concurrent_requests: int = 0
    bandwidth_used: int = 0
    ftns_spent: float = 0.0
    first_request: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_request: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_requests: int = 0


class ThreatDetector:
    """Intelligent threat detection system"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.suspicious_patterns = []
        self.blocked_ips = set()
        self.suspicious_user_agents = {
            'bot', 'crawler', 'spider', 'scraper', 'harvester', 'extractor'
        }
        
    async def analyze_request(self, request: Request, state: RateLimitState) -> Dict[str, Any]:
        """Analyze request for suspicious patterns"""
        
        threat_score = 0
        indicators = []
        
        # Check for suspicious user agents
        user_agent = request.headers.get('user-agent', '').lower()
        if any(pattern in user_agent for pattern in self.suspicious_user_agents):
            threat_score += 30
            indicators.append("suspicious_user_agent")
        
        # Check for rapid requests from same IP
        if state.total_requests > 0:
            time_diff = (state.last_request - state.first_request).total_seconds()
            if time_diff > 0:
                request_rate = state.total_requests / time_diff
                if request_rate > 10:  # More than 10 requests per second
                    threat_score += 40
                    indicators.append("high_request_rate")
        
        # Check for requests without common headers
        expected_headers = ['accept', 'accept-language', 'accept-encoding']
        missing_headers = [h for h in expected_headers if h not in request.headers]
        if len(missing_headers) >= 2:
            threat_score += 20
            indicators.append("missing_common_headers")
        
        # Check for suspicious IP patterns
        ip_threat = await self._analyze_ip_reputation(state.ip_address)
        threat_score += ip_threat["score"]
        indicators.extend(ip_threat["indicators"])
        
        # Check for unusual endpoint access patterns
        endpoint_threat = await self._analyze_endpoint_patterns(state.user_id, request.url.path)
        threat_score += endpoint_threat["score"]
        indicators.extend(endpoint_threat["indicators"])
        
        return {
            "threat_score": threat_score,
            "risk_level": self._get_risk_level(threat_score),
            "indicators": indicators,
            "recommended_action": self._get_recommended_action(threat_score)
        }
    
    async def _analyze_ip_reputation(self, ip_address: str) -> Dict[str, Any]:
        """Analyze IP reputation"""
        
        threat_score = 0
        indicators = []
        
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Check if IP is in known bad ranges
            tor_ranges = []  # Would populate with known Tor exit nodes
            vpn_ranges = []  # Would populate with known VPN ranges
            
            # Check for repeated failures from this IP
            failures_key = f"ip_failures:{ip_address}"
            failures = await self.redis.get(failures_key)
            if failures and int(failures) > 5:
                threat_score += 25
                indicators.append("repeated_failures")
            
            # Check for IP in temporary block list
            if ip_address in self.blocked_ips:
                threat_score += 50
                indicators.append("blocked_ip")
            
        except ValueError:
            threat_score += 30
            indicators.append("invalid_ip")
        
        return {"score": threat_score, "indicators": indicators}
    
    async def _analyze_endpoint_patterns(self, user_id: Optional[str], endpoint: str) -> Dict[str, Any]:
        """Analyze endpoint access patterns"""
        
        threat_score = 0
        indicators = []
        
        if user_id:
            # Check for unusual endpoint access patterns
            pattern_key = f"user_endpoints:{user_id}"
            endpoints = await self.redis.get(pattern_key)
            
            if endpoints:
                endpoint_list = json.loads(endpoints)
                # Check for scanning behavior (accessing many different endpoints)
                if len(set(endpoint_list)) > 20:
                    threat_score += 30
                    indicators.append("endpoint_scanning")
            
            # Update endpoint history
            await self.redis.lpush(pattern_key, endpoint)
            await self.redis.ltrim(pattern_key, 0, 50)  # Keep last 50 endpoints
            await self.redis.expire(pattern_key, 3600)  # Expire after 1 hour
        
        return {"score": threat_score, "indicators": indicators}
    
    def _get_risk_level(self, threat_score: int) -> str:
        """Convert threat score to risk level"""
        if threat_score >= 80:
            return "critical"
        elif threat_score >= 60:
            return "high"
        elif threat_score >= 40:
            return "medium"
        elif threat_score >= 20:
            return "low"
        else:
            return "minimal"
    
    def _get_recommended_action(self, threat_score: int) -> str:
        """Get recommended action based on threat score"""
        if threat_score >= 80:
            return "block_immediately"
        elif threat_score >= 60:
            return "require_captcha"
        elif threat_score >= 40:
            return "increase_monitoring"
        elif threat_score >= 20:
            return "log_and_watch"
        else:
            return "allow"


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load and threat levels"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.threat_detector = ThreatDetector(redis_client)
        self.tier_limits = self._initialize_tier_limits()
        self.system_load_factor = 1.0
        self.active_states: Dict[str, RateLimitState] = {}
        
    def _initialize_tier_limits(self) -> Dict[UserTier, TierLimits]:
        """Initialize rate limits for all tiers"""
        return {
            tier: TierLimits(tier) for tier in UserTier
        }
    
    async def check_rate_limit(self, request: Request, user_id: Optional[str] = None, 
                             user_tier: UserTier = UserTier.FREE) -> Tuple[bool, Dict[str, Any]]:
        """Check if request should be rate limited"""
        
        client_ip = self._get_client_ip(request)
        state_key = f"{user_id}:{client_ip}" if user_id else client_ip
        
        # Get or create rate limit state
        state = await self._get_rate_limit_state(state_key, user_id, client_ip)
        
        # Perform threat analysis
        threat_analysis = await self.threat_detector.analyze_request(request, state)
        
        # Get tier limits (adjusted for threat level and system load)
        tier_limits = self._get_adjusted_limits(user_tier, threat_analysis)
        
        # Check all rate limit types
        limit_results = {}
        overall_allowed = True
        
        for limit_type, rate_limit in tier_limits.limits.items():
            allowed, limit_info = await self._check_individual_limit(
                state, limit_type, rate_limit, request
            )
            
            limit_results[limit_type.value] = limit_info
            
            if not allowed:
                overall_allowed = False
                # Don't break - collect all limit information
        
        # Update request tracking
        if overall_allowed:
            await self._update_request_tracking(state, request)
        
        # Store updated state
        await self._store_rate_limit_state(state_key, state)
        
        # Prepare response info
        response_info = {
            "allowed": overall_allowed,
            "limits": limit_results,
            "threat_analysis": threat_analysis,
            "user_tier": user_tier.value,
            "system_load_factor": self.system_load_factor
        }
        
        return overall_allowed, response_info
    
    async def _get_rate_limit_state(self, state_key: str, user_id: Optional[str], 
                                  ip_address: str) -> RateLimitState:
        """Get rate limit state from Redis or create new one"""
        
        state_data = await self.redis.get(f"rate_limit_state:{state_key}")
        
        if state_data:
            data = json.loads(state_data)
            state = RateLimitState(
                user_id=user_id,
                ip_address=ip_address,
                current_requests={RateLimitType(k): v for k, v in data.get("current_requests", {}).items()},
                window_start={RateLimitType(k): datetime.fromisoformat(v) for k, v in data.get("window_start", {}).items()},
                burst_tokens={RateLimitType(k): v for k, v in data.get("burst_tokens", {}).items()},
                cooldown_until={
                    RateLimitType(k): datetime.fromisoformat(v) if v else None 
                    for k, v in data.get("cooldown_until", {}).items()
                },
                concurrent_requests=data.get("concurrent_requests", 0),
                bandwidth_used=data.get("bandwidth_used", 0),
                ftns_spent=data.get("ftns_spent", 0.0),
                first_request=datetime.fromisoformat(data.get("first_request", datetime.now(timezone.utc).isoformat())),
                last_request=datetime.fromisoformat(data.get("last_request", datetime.now(timezone.utc).isoformat())),
                total_requests=data.get("total_requests", 0)
            )
        else:
            state = RateLimitState(user_id=user_id, ip_address=ip_address)
        
        return state
    
    async def _store_rate_limit_state(self, state_key: str, state: RateLimitState):
        """Store rate limit state to Redis"""
        
        state_data = {
            "current_requests": {k.value: v for k, v in state.current_requests.items()},
            "window_start": {k.value: v.isoformat() for k, v in state.window_start.items()},
            "burst_tokens": {k.value: v for k, v in state.burst_tokens.items()},
            "cooldown_until": {
                k.value: v.isoformat() if v else None 
                for k, v in state.cooldown_until.items()
            },
            "concurrent_requests": state.concurrent_requests,
            "bandwidth_used": state.bandwidth_used,
            "ftns_spent": state.ftns_spent,
            "first_request": state.first_request.isoformat(),
            "last_request": state.last_request.isoformat(),
            "total_requests": state.total_requests
        }
        
        await self.redis.setex(
            f"rate_limit_state:{state_key}",
            3600,  # Expire after 1 hour
            json.dumps(state_data, default=str)
        )
    
    def _get_adjusted_limits(self, user_tier: UserTier, threat_analysis: Dict[str, Any]) -> TierLimits:
        """Get rate limits adjusted for threat level and system load"""
        
        base_limits = self.tier_limits[user_tier]
        
        # Adjust limits based on threat level
        threat_multiplier = {
            "minimal": 1.0,
            "low": 0.9,
            "medium": 0.7,
            "high": 0.4,
            "critical": 0.1
        }.get(threat_analysis["risk_level"], 1.0)
        
        # Adjust limits based on system load
        load_multiplier = max(0.1, 1.0 / self.system_load_factor)
        
        final_multiplier = threat_multiplier * load_multiplier
        
        # Create adjusted limits
        adjusted_limits = TierLimits(user_tier)
        for limit_type, rate_limit in base_limits.limits.items():
            adjusted_limits.limits[limit_type] = RateLimit(
                limit_type=rate_limit.limit_type,
                limit=max(1, int(rate_limit.limit * final_multiplier)),
                window_seconds=rate_limit.window_seconds,
                burst_allowance=max(0, int(rate_limit.burst_allowance * final_multiplier)),
                cooldown_seconds=rate_limit.cooldown_seconds
            )
        
        return adjusted_limits
    
    async def _check_individual_limit(self, state: RateLimitState, limit_type: RateLimitType, 
                                    rate_limit: RateLimit, request: Request) -> Tuple[bool, Dict[str, Any]]:
        """Check an individual rate limit"""
        
        now = datetime.now(timezone.utc)
        
        # Check cooldown period
        if (limit_type in state.cooldown_until and 
            state.cooldown_until[limit_type] and 
            now < state.cooldown_until[limit_type]):
            return False, {
                "limited": True,
                "limit": rate_limit.limit,
                "current": "cooldown",
                "reset_time": state.cooldown_until[limit_type].isoformat(),
                "retry_after": int((state.cooldown_until[limit_type] - now).total_seconds())
            }
        
        # Initialize tracking for this limit type if needed
        if limit_type not in state.current_requests:
            state.current_requests[limit_type] = 0
            state.window_start[limit_type] = now
            state.burst_tokens[limit_type] = rate_limit.burst_allowance
        
        # Check if window has expired
        window_elapsed = (now - state.window_start[limit_type]).total_seconds()
        if window_elapsed >= rate_limit.window_seconds:
            # Reset window
            state.current_requests[limit_type] = 0
            state.window_start[limit_type] = now
            state.burst_tokens[limit_type] = rate_limit.burst_allowance
        
        current_count = state.current_requests[limit_type]
        
        # Handle different limit types
        if limit_type == RateLimitType.CONCURRENT_REQUESTS:
            allowed = state.concurrent_requests < rate_limit.limit
            current_value = state.concurrent_requests
        elif limit_type == RateLimitType.BANDWIDTH_PER_HOUR:
            allowed = state.bandwidth_used < rate_limit.limit
            current_value = state.bandwidth_used
        elif limit_type == RateLimitType.FTNS_SPEND_PER_HOUR:
            allowed = state.ftns_spent < rate_limit.limit
            current_value = state.ftns_spent
        else:
            # Standard request-based limits
            effective_limit = rate_limit.limit + state.burst_tokens[limit_type]
            allowed = current_count < effective_limit
            current_value = current_count
            
            # Consume burst tokens if over base limit
            if current_count >= rate_limit.limit and state.burst_tokens[limit_type] > 0:
                state.burst_tokens[limit_type] -= 1
        
        # Calculate reset time
        reset_time = state.window_start[limit_type] + timedelta(seconds=rate_limit.window_seconds)
        retry_after = int((reset_time - now).total_seconds()) if not allowed else 0
        
        # Set cooldown if limit exceeded and cooldown is configured
        if not allowed and rate_limit.cooldown_seconds > 0:
            state.cooldown_until[limit_type] = now + timedelta(seconds=rate_limit.cooldown_seconds)
        
        return allowed, {
            "limited": not allowed,
            "limit": rate_limit.limit,
            "current": current_value,
            "reset_time": reset_time.isoformat(),
            "retry_after": retry_after,
            "burst_tokens_remaining": state.burst_tokens.get(limit_type, 0)
        }
    
    async def _update_request_tracking(self, state: RateLimitState, request: Request):
        """Update request tracking counters"""
        
        now = datetime.now(timezone.utc)
        state.last_request = now
        state.total_requests += 1
        
        # Update request counters for time-based limits
        for limit_type in [RateLimitType.REQUESTS_PER_MINUTE, RateLimitType.REQUESTS_PER_HOUR, RateLimitType.REQUESTS_PER_DAY]:
            if limit_type in state.current_requests:
                state.current_requests[limit_type] += 1
        
        # Estimate bandwidth usage
        content_length = int(request.headers.get('content-length', 0))
        state.bandwidth_used += content_length
        
        # Concurrent requests are handled separately in middleware
        
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        
        # Check various headers for the real IP
        ip_headers = [
            'x-forwarded-for',
            'x-real-ip',
            'cf-connecting-ip',  # Cloudflare
            'x-cluster-client-ip',
            'x-forwarded',
            'forwarded-for',
            'forwarded'
        ]
        
        for header in ip_headers:
            if header in request.headers:
                ip = request.headers[header].split(',')[0].strip()
                if ip:
                    return ip
        
        # Fallback to client host
        return request.client.host if request.client else "unknown"
    
    async def update_system_load(self, load_factor: float):
        """Update system load factor for adaptive rate limiting"""
        self.system_load_factor = max(0.1, min(10.0, load_factor))
        await self.redis.setex("system_load_factor", 60, str(self.system_load_factor))
    
    async def increment_concurrent_requests(self, state_key: str):
        """Increment concurrent request counter"""
        await self.redis.incr(f"concurrent:{state_key}")
        await self.redis.expire(f"concurrent:{state_key}", 300)  # 5 minute timeout
    
    async def decrement_concurrent_requests(self, state_key: str):
        """Decrement concurrent request counter"""
        current = await self.redis.get(f"concurrent:{state_key}")
        if current and int(current) > 0:
            await self.redis.decr(f"concurrent:{state_key}")


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, rate_limiter: AdaptiveRateLimiter):
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""
        
        # Extract user information
        user_id, user_tier = await self._extract_user_info(request)
        
        # Check rate limits
        allowed, limit_info = await self.rate_limiter.check_rate_limit(
            request, user_id, user_tier
        )
        
        if not allowed:
            # Return rate limit exceeded response
            return self._create_rate_limit_response(limit_info)
        
        # Track concurrent requests
        client_ip = self.rate_limiter._get_client_ip(request)
        state_key = f"{user_id}:{client_ip}" if user_id else client_ip
        
        await self.rate_limiter.increment_concurrent_requests(state_key)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            self._add_rate_limit_headers(response, limit_info)
            
            return response
            
        finally:
            await self.rate_limiter.decrement_concurrent_requests(state_key)
    
    async def _extract_user_info(self, request: Request) -> Tuple[Optional[str], UserTier]:
        """Extract user ID and tier from request"""
        
        # Try to get user info from request state (set by auth middleware)
        user_id = getattr(request.state, 'user_id', None)
        user_tier_str = getattr(request.state, 'user_tier', 'free')
        
        # Convert tier string to enum
        try:
            user_tier = UserTier(user_tier_str.lower())
        except ValueError:
            user_tier = UserTier.FREE
        
        # If no user info, treat as guest
        if not user_id:
            user_tier = UserTier.GUEST
        
        return user_id, user_tier
    
    def _create_rate_limit_response(self, limit_info: Dict[str, Any]) -> Response:
        """Create rate limit exceeded response"""
        
        # Find the most restrictive limit that was exceeded
        exceeded_limits = [
            (limit_type, info) for limit_type, info in limit_info["limits"].items()
            if info["limited"]
        ]
        
        if exceeded_limits:
            limit_type, info = exceeded_limits[0]
            retry_after = info["retry_after"]
        else:
            retry_after = 60  # Default retry after
        
        error_response = {
            "error": "RATE_LIMIT_EXCEEDED",
            "message": "Rate limit exceeded. Please slow down your requests.",
            "details": {
                "limits": limit_info["limits"],
                "threat_analysis": limit_info["threat_analysis"],
                "user_tier": limit_info["user_tier"]
            },
            "retry_after": retry_after,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        from fastapi.responses import JSONResponse
        response = JSONResponse(
            content=error_response,
            status_code=429,
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Exceeded": "true"
            }
        )
        
        self._add_rate_limit_headers(response, limit_info)
        return response
    
    def _add_rate_limit_headers(self, response: Response, limit_info: Dict[str, Any]):
        """Add rate limit headers to response"""
        
        # Add general rate limit headers
        response.headers["X-RateLimit-Tier"] = limit_info["user_tier"]
        response.headers["X-RateLimit-SystemLoad"] = str(limit_info["system_load_factor"])
        
        # Add specific limit headers
        for limit_type, info in limit_info["limits"].items():
            header_prefix = f"X-RateLimit-{limit_type.replace('_', '-').title()}"
            response.headers[f"{header_prefix}-Limit"] = str(info["limit"])
            response.headers[f"{header_prefix}-Current"] = str(info["current"])
            response.headers[f"{header_prefix}-Reset"] = info["reset_time"]
            
            if info.get("burst_tokens_remaining") is not None:
                response.headers[f"{header_prefix}-Burst"] = str(info["burst_tokens_remaining"])
        
        # Add threat analysis headers (for monitoring)
        threat = limit_info["threat_analysis"]
        response.headers["X-Threat-Score"] = str(threat["threat_score"])
        response.headers["X-Threat-Level"] = threat["risk_level"]


# Global rate limiter instance (to be initialized with Redis connection)
rate_limiter: Optional[AdaptiveRateLimiter] = None


async def initialize_rate_limiter(redis_client: aioredis.Redis):
    """Initialize the global rate limiter"""
    global rate_limiter
    rate_limiter = AdaptiveRateLimiter(redis_client)
    logger.info("✅ Advanced rate limiter initialized")


def get_rate_limiter() -> AdaptiveRateLimiter:
    """Get the global rate limiter instance"""
    if rate_limiter is None:
        raise RuntimeError("Rate limiter not initialized. Call initialize_rate_limiter() first.")
    return rate_limiter