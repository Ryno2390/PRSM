"""
Authentication and Authorization Models
Comprehensive user management and security models for PRSM
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import Column, String, DateTime, Boolean, Enum as SQLEnum, Text, Integer
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from prsm.core.models import TimestampMixin, Base


class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"                    # Full system access
    RESEARCHER = "researcher"          # Full AI model access
    DEVELOPER = "developer"            # Development and testing access
    ANALYST = "analyst"                # Read-only analysis access
    USER = "user"                      # Basic user access
    GUEST = "guest"                    # Limited guest access


class Permission(str, Enum):
    """Fine-grained permissions"""
    # Model Operations
    MODEL_CREATE = "model:create"
    MODEL_READ = "model:read"
    MODEL_UPDATE = "model:update"
    MODEL_DELETE = "model:delete"
    MODEL_EXECUTE = "model:execute"
    
    # Agent Operations
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_EXECUTE = "agent:execute"
    
    # Data Operations
    DATA_CREATE = "data:create"
    DATA_READ = "data:read"
    DATA_UPDATE = "data:update"
    DATA_DELETE = "data:delete"
    
    # System Administration
    SYSTEM_ADMIN = "system:admin"
    USER_MANAGE = "user:manage"
    CONFIG_MANAGE = "config:manage"
    
    # IPFS Operations
    IPFS_UPLOAD = "ipfs:upload"
    IPFS_DOWNLOAD = "ipfs:download"
    IPFS_PIN = "ipfs:pin"
    
    # Token Operations
    TOKEN_TRANSFER = "token:transfer"
    TOKEN_MINT = "token:mint"
    TOKEN_BURN = "token:burn"


# Role-Permission Mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [p for p in Permission],  # All permissions
    UserRole.RESEARCHER: [
        Permission.MODEL_CREATE, Permission.MODEL_READ, Permission.MODEL_UPDATE, Permission.MODEL_EXECUTE,
        Permission.AGENT_CREATE, Permission.AGENT_READ, Permission.AGENT_UPDATE, Permission.AGENT_EXECUTE,
        Permission.DATA_CREATE, Permission.DATA_READ, Permission.DATA_UPDATE,
        Permission.IPFS_UPLOAD, Permission.IPFS_DOWNLOAD, Permission.IPFS_PIN,
        Permission.TOKEN_TRANSFER
    ],
    UserRole.DEVELOPER: [
        Permission.MODEL_READ, Permission.MODEL_EXECUTE,
        Permission.AGENT_READ, Permission.AGENT_EXECUTE,
        Permission.DATA_READ,
        Permission.IPFS_DOWNLOAD,
        Permission.TOKEN_TRANSFER
    ],
    UserRole.ANALYST: [
        Permission.MODEL_READ,
        Permission.AGENT_READ,
        Permission.DATA_READ,
        Permission.IPFS_DOWNLOAD
    ],
    UserRole.USER: [
        Permission.MODEL_READ, Permission.MODEL_EXECUTE,
        Permission.AGENT_READ, Permission.AGENT_EXECUTE,
        Permission.DATA_READ,
        Permission.IPFS_DOWNLOAD,
        Permission.TOKEN_TRANSFER
    ],
    UserRole.GUEST: [
        Permission.MODEL_READ,
        Permission.AGENT_READ,
        Permission.DATA_READ
    ]
}


class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    
    # Role and permissions
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.USER)
    custom_permissions = Column(Text, nullable=True)  # JSON array of additional permissions
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Security fields
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    def get_permissions(self) -> Set[Permission]:
        """Get all permissions for this user"""
        permissions = set(ROLE_PERMISSIONS.get(self.role, []))
        
        # Add custom permissions if any
        if self.custom_permissions:
            try:
                import json
                custom = json.loads(self.custom_permissions)
                permissions.update(Permission(p) for p in custom if p in Permission.__members__.values())
            except (json.JSONDecodeError, ValueError):
                pass
        
        return permissions
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission"""
        if self.is_superuser:
            return True
        return permission in self.get_permissions()
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions"""
        if self.is_superuser:
            return True
        user_permissions = self.get_permissions()
        return any(p in user_permissions for p in permissions)


class AuthToken(Base):
    """Authentication token model for JWT management"""
    __tablename__ = "auth_tokens"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    token_hash = Column(String(255), nullable=False, unique=True, index=True)
    token_type = Column(String(20), nullable=False, default="access")  # access, refresh
    
    # Token metadata
    issued_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_used = Column(DateTime(timezone=True), nullable=True)
    
    # Security
    is_revoked = Column(Boolean, default=False, nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    client_ip = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    
    def is_valid(self) -> bool:
        """Check if token is valid (not expired or revoked)"""
        now = datetime.now(timezone.utc)
        return not self.is_revoked and self.expires_at > now


# Pydantic models for API

class UserResponse(BaseModel):
    """User response model for API"""
    id: UUID
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    role: UserRole
    is_active: bool
    is_verified: bool
    last_login: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """User creation model"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    full_name: Optional[str] = Field(None, max_length=255)
    password: str = Field(..., min_length=8, max_length=100)
    role: UserRole = UserRole.USER


class UserUpdate(BaseModel):
    """User update model"""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    full_name: Optional[str] = Field(None, max_length=255)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None


class PasswordChange(BaseModel):
    """Password change model"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


class LoginRequest(BaseModel):
    """Login request model"""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    remember_me: bool = False


class RegisterRequest(BaseModel):
    """Registration request model"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    full_name: Optional[str] = Field(None, max_length=255)
    password: str = Field(..., min_length=8, max_length=100)
    confirm_password: str = Field(..., min_length=8, max_length=100)
    
    def passwords_match(self) -> bool:
        """Check if passwords match"""
        return self.password == self.confirm_password


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenRefreshRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str


class PermissionCheck(BaseModel):
    """Permission check request"""
    permission: Permission
    resource_id: Optional[str] = None


class RoleAssignment(BaseModel):
    """Role assignment model"""
    user_id: UUID
    role: UserRole
    custom_permissions: Optional[List[Permission]] = None