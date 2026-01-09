"""
PRSM Enhanced Authentication Security System
Comprehensive authentication with multi-factor auth, session management, and advanced security features
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import hashlib
import hmac
import base64
import pyotp
import qrcode
import io
import redis.asyncio as aioredis
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import json
import ipaddress
from collections import defaultdict, deque
import asyncio

logger = logging.getLogger(__name__)


class AuthenticationMethod(Enum):
    """Authentication methods supported"""
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms" 
    API_KEY = "api_key"
    SESSION_TOKEN = "session_token"
    REFRESH_TOKEN = "refresh_token"


class UserRole(Enum):
    """User roles with different permissions"""
    GUEST = "guest"
    RESEARCHER = "researcher"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class SessionStatus(Enum):
    """Session status types"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPICIOUS = "suspicious"


@dataclass
class AuthenticationAttempt:
    """Record of an authentication attempt"""
    user_id: Optional[str]
    email: Optional[str]
    ip_address: str
    user_agent: str
    method: AuthenticationMethod
    success: bool
    failure_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    location: Optional[str] = None
    device_fingerprint: Optional[str] = None


@dataclass
class UserSession:
    """Active user session"""
    session_id: str
    user_id: str
    user_email: str
    user_role: UserRole
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    device_fingerprint: str
    status: SessionStatus = SessionStatus.ACTIVE
    mfa_verified: bool = False
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIKey:
    """API key for programmatic access"""
    key_id: str
    user_id: str
    name: str
    key_hash: str
    permissions: List[str]
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    usage_count: int = 0
    rate_limit_tier: str = "standard"


@dataclass
class MFASetup:
    """Multi-factor authentication setup"""
    user_id: str
    method: AuthenticationMethod
    secret_key: str
    backup_codes: List[str]
    verified: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EnhancedPasswordValidator:
    """Advanced password validation and security"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.min_length = 12
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digits = True
        self.require_special = True
        self.common_passwords = set()  # Would load from common passwords list
        
    def validate_password_strength(self, password: str, user_info: Dict[str, str] = None) -> Tuple[bool, List[str]]:
        """Validate password strength and security"""
        issues = []
        
        # Length check
        if len(password) < self.min_length:
            issues.append(f"Password must be at least {self.min_length} characters long")
        
        # Character requirements
        if self.require_uppercase and not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not any(c.islower() for c in password):
            issues.append("Password must contain at least one lowercase letter")
        
        if self.require_digits and not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one digit")
        
        if self.require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain at least one special character")
        
        # Common password check
        if password.lower() in self.common_passwords:
            issues.append("Password is too common and easily guessable")
        
        # Personal info check
        if user_info:
            for key, value in user_info.items():
                if value and len(value) > 2 and value.lower() in password.lower():
                    issues.append(f"Password should not contain personal information ({key})")
        
        # Pattern checks
        if self._has_repeated_patterns(password):
            issues.append("Password contains repeated patterns")
        
        if self._has_sequential_characters(password):
            issues.append("Password contains sequential characters")
        
        return len(issues) == 0, issues
    
    def _has_repeated_patterns(self, password: str) -> bool:
        """Check for repeated patterns in password"""
        for i in range(len(password) - 2):
            pattern = password[i:i+3]
            if password.count(pattern) > 1:
                return True
        return False
    
    def _has_sequential_characters(self, password: str) -> bool:
        """Check for sequential characters (abc, 123, etc.)"""
        for i in range(len(password) - 2):
            chars = password[i:i+3]
            if (ord(chars[1]) == ord(chars[0]) + 1 and 
                ord(chars[2]) == ord(chars[1]) + 1):
                return True
        return False
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)


class MFAManager:
    """Multi-factor authentication manager"""
    
    def __init__(self):
        self.issuer_name = "PRSM API"
        
    def generate_totp_secret(self) -> str:
        """Generate TOTP secret key"""
        return pyotp.random_base32()
    
    def generate_qr_code(self, user_email: str, secret: str) -> bytes:
        """Generate QR code for TOTP setup"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=self.issuer_name
        )
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
    
    def verify_totp_token(self, secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=window)
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA"""
        return [secrets.token_hex(4).upper() for _ in range(count)]
    
    def verify_backup_code(self, backup_codes: List[str], code: str) -> Tuple[bool, List[str]]:
        """Verify backup code and remove it from list"""
        code = code.upper().replace(" ", "").replace("-", "")
        
        if code in backup_codes:
            remaining_codes = [c for c in backup_codes if c != code]
            return True, remaining_codes
        
        return False, backup_codes


class SessionManager:
    """Advanced session management"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.session_timeout = timedelta(hours=24)
        self.refresh_token_timeout = timedelta(days=30)
        self.concurrent_session_limit = 5
        
    async def create_session(self, user_id: str, user_email: str, user_role: UserRole, 
                           request: Request, mfa_verified: bool = False) -> UserSession:
        """Create new user session"""
        
        session_id = self._generate_session_id()
        device_fingerprint = self._generate_device_fingerprint(request)
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            user_email=user_email,
            user_role=user_role,
            created_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            ip_address=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent", ""),
            device_fingerprint=device_fingerprint,
            mfa_verified=mfa_verified,
            permissions=self._get_role_permissions(user_role)
        )
        
        # Store session in Redis
        await self._store_session(session)
        
        # Enforce concurrent session limit
        await self._enforce_session_limit(user_id)
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        session_data = await self.redis.get(f"session:{session_id}")
        
        if not session_data:
            return None
        
        data = json.loads(session_data)
        session = UserSession(
            session_id=data["session_id"],
            user_id=data["user_id"],
            user_email=data["user_email"],
            user_role=UserRole(data["user_role"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            ip_address=data["ip_address"],
            user_agent=data["user_agent"],
            device_fingerprint=data["device_fingerprint"],
            status=SessionStatus(data.get("status", "active")),
            mfa_verified=data.get("mfa_verified", False),
            permissions=data.get("permissions", []),
            metadata=data.get("metadata", {})
        )
        
        # Check if session is expired
        if self._is_session_expired(session):
            await self.revoke_session(session_id)
            return None
        
        return session
    
    async def update_session_activity(self, session_id: str):
        """Update session last activity time"""
        session = await self.get_session(session_id)
        if session:
            session.last_activity = datetime.now(timezone.utc)
            await self._store_session(session)
    
    async def revoke_session(self, session_id: str):
        """Revoke a session"""
        await self.redis.delete(f"session:{session_id}")
        logger.info(f"Revoked session {session_id}")
    
    async def revoke_user_sessions(self, user_id: str, except_session: Optional[str] = None):
        """Revoke all sessions for a user"""
        pattern = f"session:*"
        async for key in self.redis.scan_iter(match=pattern):
            session_data = await self.redis.get(key)
            if session_data:
                data = json.loads(session_data)
                if (data["user_id"] == user_id and 
                    (not except_session or data["session_id"] != except_session)):
                    await self.redis.delete(key)
    
    async def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """Get all active sessions for a user"""
        sessions = []
        pattern = f"session:*"
        
        async for key in self.redis.scan_iter(match=pattern):
            session_data = await self.redis.get(key)
            if session_data:
                data = json.loads(session_data)
                if data["user_id"] == user_id:
                    session = UserSession(
                        session_id=data["session_id"],
                        user_id=data["user_id"],
                        user_email=data["user_email"],
                        user_role=UserRole(data["user_role"]),
                        created_at=datetime.fromisoformat(data["created_at"]),
                        last_activity=datetime.fromisoformat(data["last_activity"]),
                        ip_address=data["ip_address"],
                        user_agent=data["user_agent"],
                        device_fingerprint=data["device_fingerprint"],
                        status=SessionStatus(data.get("status", "active")),
                        mfa_verified=data.get("mfa_verified", False),
                        permissions=data.get("permissions", []),
                        metadata=data.get("metadata", {})
                    )
                    
                    if not self._is_session_expired(session):
                        sessions.append(session)
        
        return sessions
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)
    
    def _generate_device_fingerprint(self, request: Request) -> str:
        """Generate device fingerprint from request"""
        fingerprint_data = {
            "user_agent": request.headers.get("user-agent", ""),
            "accept": request.headers.get("accept", ""),
            "accept_language": request.headers.get("accept-language", ""),
            "accept_encoding": request.headers.get("accept-encoding", ""),
        }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        ip_headers = [
            'x-forwarded-for',
            'x-real-ip',
            'cf-connecting-ip',
            'x-cluster-client-ip',
        ]
        
        for header in ip_headers:
            if header in request.headers:
                ip = request.headers[header].split(',')[0].strip()
                if ip:
                    return ip
        
        return request.client.host if request.client else "unknown"
    
    def _get_role_permissions(self, role: UserRole) -> List[str]:
        """Get permissions for user role"""
        permissions_map = {
            UserRole.GUEST: ["read:public"],
            UserRole.RESEARCHER: ["read:public", "read:marketplace", "purchase:resources"],
            UserRole.PREMIUM: ["read:public", "read:marketplace", "purchase:resources", "create:sessions", "read:analytics"],
            UserRole.ENTERPRISE: ["read:public", "read:marketplace", "purchase:resources", "create:sessions", "read:analytics", "manage:team"],
            UserRole.ADMIN: ["read:all", "write:all", "delete:resources", "manage:users"],
            UserRole.SUPER_ADMIN: ["*"]
        }
        
        return permissions_map.get(role, [])
    
    async def _store_session(self, session: UserSession):
        """Store session in Redis"""
        session_data = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "user_email": session.user_email,
            "user_role": session.user_role.value,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "ip_address": session.ip_address,
            "user_agent": session.user_agent,
            "device_fingerprint": session.device_fingerprint,
            "status": session.status.value,
            "mfa_verified": session.mfa_verified,
            "permissions": session.permissions,
            "metadata": session.metadata
        }
        
        await self.redis.setex(
            f"session:{session.session_id}",
            int(self.session_timeout.total_seconds()),
            json.dumps(session_data)
        )
        
        # Also add to user's session list
        await self.redis.sadd(f"user_sessions:{session.user_id}", session.session_id)
    
    async def _enforce_session_limit(self, user_id: str):
        """Enforce concurrent session limit"""
        user_sessions = await self.get_user_sessions(user_id)
        
        if len(user_sessions) > self.concurrent_session_limit:
            # Sort by last activity and remove oldest sessions
            user_sessions.sort(key=lambda s: s.last_activity)
            sessions_to_remove = user_sessions[:-self.concurrent_session_limit]
            
            for session in sessions_to_remove:
                await self.revoke_session(session.session_id)
    
    def _is_session_expired(self, session: UserSession) -> bool:
        """Check if session is expired"""
        return (datetime.now(timezone.utc) - session.last_activity) > self.session_timeout


class APIKeyManager:
    """API key management system"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        
    async def create_api_key(self, user_id: str, name: str, permissions: List[str], 
                           expires_in_days: Optional[int] = None) -> Tuple[str, APIKey]:
        """Create new API key"""
        
        key_id = self._generate_key_id()
        raw_key = self._generate_raw_key()
        key_hash = self._hash_key(raw_key)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            key_id=key_id,
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            permissions=permissions,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at
        )
        
        await self._store_api_key(api_key)
        
        # Return the raw key (only time it's available)
        full_key = f"prsm_{key_id}_{raw_key}"
        return full_key, api_key
    
    async def verify_api_key(self, key: str) -> Optional[APIKey]:
        """Verify and get API key details"""
        
        if not key.startswith("prsm_"):
            return None
        
        try:
            parts = key.split("_", 2)
            if len(parts) != 3:
                return None
            
            key_id = parts[1]
            raw_key = parts[2]
            
            api_key = await self.get_api_key(key_id)
            if not api_key:
                return None
            
            # Verify key hash
            if not self._verify_key(raw_key, api_key.key_hash):
                return None
            
            # Check if expired
            if api_key.expires_at and datetime.now(timezone.utc) > api_key.expires_at:
                return None
            
            # Check if active
            if not api_key.is_active:
                return None
            
            # Update usage
            await self._update_key_usage(api_key)
            
            return api_key
            
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return None
    
    async def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID"""
        key_data = await self.redis.get(f"api_key:{key_id}")
        
        if not key_data:
            return None
        
        data = json.loads(key_data)
        return APIKey(
            key_id=data["key_id"],
            user_id=data["user_id"],
            name=data["name"],
            key_hash=data["key_hash"],
            permissions=data["permissions"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            is_active=data.get("is_active", True),
            usage_count=data.get("usage_count", 0),
            rate_limit_tier=data.get("rate_limit_tier", "standard")
        )
    
    async def revoke_api_key(self, key_id: str):
        """Revoke an API key"""
        api_key = await self.get_api_key(key_id)
        if api_key:
            api_key.is_active = False
            await self._store_api_key(api_key)
    
    async def get_user_api_keys(self, user_id: str) -> List[APIKey]:
        """Get all API keys for a user"""
        keys = []
        pattern = f"api_key:*"
        
        async for key in self.redis.scan_iter(match=pattern):
            key_data = await self.redis.get(key)
            if key_data:
                data = json.loads(key_data)
                if data["user_id"] == user_id:
                    api_key = APIKey(
                        key_id=data["key_id"],
                        user_id=data["user_id"],
                        name=data["name"],
                        key_hash=data["key_hash"],
                        permissions=data["permissions"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
                        expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                        is_active=data.get("is_active", True),
                        usage_count=data.get("usage_count", 0),
                        rate_limit_tier=data.get("rate_limit_tier", "standard")
                    )
                    keys.append(api_key)
        
        return keys
    
    def _generate_key_id(self) -> str:
        """Generate key ID"""
        return secrets.token_hex(8)
    
    def _generate_raw_key(self) -> str:
        """Generate raw key"""
        return secrets.token_urlsafe(32)
    
    def _hash_key(self, raw_key: str) -> str:
        """Hash raw key"""
        return hashlib.sha256(raw_key.encode()).hexdigest()
    
    def _verify_key(self, raw_key: str, key_hash: str) -> bool:
        """Verify raw key against hash"""
        return hmac.compare_digest(self._hash_key(raw_key), key_hash)
    
    async def _store_api_key(self, api_key: APIKey):
        """Store API key in Redis"""
        key_data = {
            "key_id": api_key.key_id,
            "user_id": api_key.user_id,
            "name": api_key.name,
            "key_hash": api_key.key_hash,
            "permissions": api_key.permissions,
            "created_at": api_key.created_at.isoformat(),
            "last_used": api_key.last_used.isoformat() if api_key.last_used else None,
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "is_active": api_key.is_active,
            "usage_count": api_key.usage_count,
            "rate_limit_tier": api_key.rate_limit_tier
        }
        
        await self.redis.set(f"api_key:{api_key.key_id}", json.dumps(key_data))
    
    async def _update_key_usage(self, api_key: APIKey):
        """Update API key usage statistics"""
        api_key.last_used = datetime.now(timezone.utc)
        api_key.usage_count += 1
        await self._store_api_key(api_key)


class EnhancedAuthenticationSystem:
    """Main authentication system orchestrator"""
    
    def __init__(self, redis_client: aioredis.Redis, jwt_secret: str):
        self.redis = redis_client
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = "HS256"
        self.access_token_expire_minutes = 60
        self.refresh_token_expire_days = 30
        
        # Initialize components
        self.password_validator = EnhancedPasswordValidator()
        self.mfa_manager = MFAManager()
        self.session_manager = SessionManager(redis_client)
        self.api_key_manager = APIKeyManager(redis_client)
        
        # Authentication attempt tracking
        self.failed_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.lockout_duration = timedelta(minutes=15)
        self.max_failed_attempts = 5
    
    async def authenticate_user(self, email: str, password: str, request: Request, 
                             mfa_token: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Authenticate user with enhanced security"""
        
        client_ip = self.session_manager._get_client_ip(request)
        
        # Check for account lockout
        if await self._is_account_locked(email, client_ip):
            return False, {
                "error": "ACCOUNT_LOCKED",
                "message": "Account temporarily locked due to too many failed attempts",
                "retry_after": self.lockout_duration.total_seconds()
            }
        
        # Record authentication attempt
        attempt = AuthenticationAttempt(
            user_id=None,
            email=email,
            ip_address=client_ip,
            user_agent=request.headers.get("user-agent", ""),
            method=AuthenticationMethod.PASSWORD,
            success=False
        )
        
        try:
            # Verify credentials (would integrate with user database)
            user_data = await self._verify_user_credentials(email, password)
            if not user_data:
                attempt.failure_reason = "Invalid credentials"
                await self._record_failed_attempt(email, client_ip, attempt)
                return False, {
                    "error": "INVALID_CREDENTIALS",
                    "message": "Invalid email or password"
                }
            
            # Check if MFA is required
            mfa_setup = await self._get_user_mfa_setup(user_data["user_id"])
            if mfa_setup and mfa_setup.verified:
                if not mfa_token:
                    return False, {
                        "error": "MFA_REQUIRED",
                        "message": "Multi-factor authentication required",
                        "mfa_methods": [mfa_setup.method.value]
                    }
                
                # Verify MFA token
                if not await self._verify_mfa_token(mfa_setup, mfa_token):
                    attempt.failure_reason = "Invalid MFA token"
                    await self._record_failed_attempt(email, client_ip, attempt)
                    return False, {
                        "error": "INVALID_MFA_TOKEN",
                        "message": "Invalid multi-factor authentication token"
                    }
            
            # Create session
            session = await self.session_manager.create_session(
                user_id=user_data["user_id"],
                user_email=email,
                user_role=UserRole(user_data["role"]),
                request=request,
                mfa_verified=bool(mfa_setup and mfa_setup.verified)
            )
            
            # Generate tokens
            access_token = self._create_access_token(session)
            refresh_token = self._create_refresh_token(session)
            
            # Record successful attempt
            attempt.success = True
            attempt.user_id = user_data["user_id"]
            await self._record_authentication_attempt(attempt)
            
            # Clear failed attempts
            await self._clear_failed_attempts(email, client_ip)
            
            return True, {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire_minutes * 60,
                "session_id": session.session_id,
                "user": {
                    "id": user_data["user_id"],
                    "email": email,
                    "role": user_data["role"],
                    "mfa_enabled": bool(mfa_setup and mfa_setup.verified)
                }
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            attempt.failure_reason = "System error"
            await self._record_failed_attempt(email, client_ip, attempt)
            return False, {
                "error": "AUTHENTICATION_ERROR",
                "message": "Authentication system error"
            }
    
    async def verify_token(self, token: str) -> Optional[UserSession]:
        """Verify JWT token and return session"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            session_id = payload.get("session_id")
            
            if not session_id:
                return None
            
            session = await self.session_manager.get_session(session_id)
            if session:
                await self.session_manager.update_session_activity(session_id)
            
            return session
            
        except JWTError:
            return None
    
    async def verify_api_key(self, key: str) -> Optional[APIKey]:
        """Verify API key"""
        return await self.api_key_manager.verify_api_key(key)
    
    def _create_access_token(self, session: UserSession) -> str:
        """Create JWT access token"""
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        payload = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "email": session.user_email,
            "role": session.user_role.value,
            "permissions": session.permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _create_refresh_token(self, session: UserSession) -> str:
        """Create JWT refresh token"""
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        payload = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    async def _verify_user_credentials(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials (mock implementation)"""
        # This would integrate with your user database
        # For now, return mock data
        return {
            "user_id": "user_123",
            "email": email,
            "role": "researcher",
            "password_hash": self.password_validator.hash_password(password)
        }
    
    async def _get_user_mfa_setup(self, user_id: str) -> Optional[MFASetup]:
        """Get user's MFA setup"""
        mfa_data = await self.redis.get(f"mfa_setup:{user_id}")
        if not mfa_data:
            return None
        
        data = json.loads(mfa_data)
        return MFASetup(
            user_id=data["user_id"],
            method=AuthenticationMethod(data["method"]),
            secret_key=data["secret_key"],
            backup_codes=data["backup_codes"],
            verified=data["verified"],
            created_at=datetime.fromisoformat(data["created_at"])
        )
    
    async def _verify_mfa_token(self, mfa_setup: MFASetup, token: str) -> bool:
        """Verify MFA token"""
        if mfa_setup.method == AuthenticationMethod.MFA_TOTP:
            return self.mfa_manager.verify_totp_token(mfa_setup.secret_key, token)
        
        # Handle backup codes
        if len(token) == 8:  # Backup code format
            is_valid, remaining_codes = self.mfa_manager.verify_backup_code(
                mfa_setup.backup_codes, token
            )
            if is_valid:
                # Update backup codes
                mfa_setup.backup_codes = remaining_codes
                await self._store_mfa_setup(mfa_setup)
                return True
        
        return False
    
    async def _store_mfa_setup(self, mfa_setup: MFASetup):
        """Store MFA setup"""
        data = {
            "user_id": mfa_setup.user_id,
            "method": mfa_setup.method.value,
            "secret_key": mfa_setup.secret_key,
            "backup_codes": mfa_setup.backup_codes,
            "verified": mfa_setup.verified,
            "created_at": mfa_setup.created_at.isoformat()
        }
        await self.redis.set(f"mfa_setup:{mfa_setup.user_id}", json.dumps(data))
    
    async def _is_account_locked(self, email: str, ip_address: str) -> bool:
        """Check if account is locked due to failed attempts"""
        
        # Check email-based lockout
        email_attempts = await self.redis.get(f"failed_attempts:email:{email}")
        if email_attempts and int(email_attempts) >= self.max_failed_attempts:
            return True
        
        # Check IP-based lockout
        ip_attempts = await self.redis.get(f"failed_attempts:ip:{ip_address}")
        if ip_attempts and int(ip_attempts) >= self.max_failed_attempts:
            return True
        
        return False
    
    async def _record_failed_attempt(self, email: str, ip_address: str, attempt: AuthenticationAttempt):
        """Record failed authentication attempt"""
        
        # Increment counters with expiration
        await self.redis.incr(f"failed_attempts:email:{email}")
        await self.redis.expire(f"failed_attempts:email:{email}", int(self.lockout_duration.total_seconds()))
        
        await self.redis.incr(f"failed_attempts:ip:{ip_address}")
        await self.redis.expire(f"failed_attempts:ip:{ip_address}", int(self.lockout_duration.total_seconds()))
        
        # Record detailed attempt
        await self._record_authentication_attempt(attempt)
    
    async def _clear_failed_attempts(self, email: str, ip_address: str):
        """Clear failed attempt counters after successful login"""
        await self.redis.delete(f"failed_attempts:email:{email}")
        await self.redis.delete(f"failed_attempts:ip:{ip_address}")
    
    async def _record_authentication_attempt(self, attempt: AuthenticationAttempt):
        """Record authentication attempt for monitoring"""
        attempt_data = {
            "user_id": attempt.user_id,
            "email": attempt.email,
            "ip_address": attempt.ip_address,
            "user_agent": attempt.user_agent,
            "method": attempt.method.value,
            "success": attempt.success,
            "failure_reason": attempt.failure_reason,
            "timestamp": attempt.timestamp.isoformat(),
            "location": attempt.location,
            "device_fingerprint": attempt.device_fingerprint
        }
        
        # Store in Redis with TTL for log rotation
        attempt_id = secrets.token_hex(8)
        await self.redis.setex(
            f"auth_attempt:{attempt_id}",
            86400 * 30,  # 30 days retention
            json.dumps(attempt_data)
        )


# Global authentication system instance
auth_system: Optional[EnhancedAuthenticationSystem] = None


async def initialize_auth_system(redis_client: aioredis.Redis, jwt_secret: str):
    """Initialize the global authentication system"""
    global auth_system
    auth_system = EnhancedAuthenticationSystem(redis_client, jwt_secret)
    logger.info("✅ Enhanced authentication system initialized")


def get_auth_system() -> EnhancedAuthenticationSystem:
    """Get the global authentication system instance"""
    if auth_system is None:
        raise RuntimeError("Authentication system not initialized. Call initialize_auth_system() first.")
    return auth_system


# FastAPI dependencies for authentication
security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserSession:
    """FastAPI dependency to get current authenticated user"""
    auth = get_auth_system()
    session = await auth.verify_token(credentials.credentials)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return session


async def get_current_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> APIKey:
    """FastAPI dependency to get current API key"""
    auth = get_auth_system()
    api_key = await auth.verify_api_key(credentials.credentials)
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key


def require_permissions(*required_permissions: str):
    """Decorator to require specific permissions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get user from kwargs (injected by dependency)
            user_session = None
            api_key = None
            
            for value in kwargs.values():
                if isinstance(value, UserSession):
                    user_session = value
                    break
                elif isinstance(value, APIKey):
                    api_key = value
                    break
            
            # Check permissions
            user_permissions = []
            if user_session:
                user_permissions = user_session.permissions
            elif api_key:
                user_permissions = api_key.permissions
            
            # Check if user has required permissions
            if "*" not in user_permissions:  # Super admin bypass
                for permission in required_permissions:
                    if permission not in user_permissions:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Missing required permission: {permission}"
                        )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator