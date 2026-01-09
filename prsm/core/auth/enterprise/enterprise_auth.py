"""
Enterprise Authentication Framework for PRSM
===========================================

Comprehensive enterprise authentication system that integrates SSO, LDAP,
and MFA capabilities. Provides a unified interface for all enterprise
authentication needs.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import structlog

from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer

from ..auth_manager import AuthManager, auth_manager
from ..models import User, UserRole, TokenResponse
from .sso_provider import EnterpriseSSO, SSOConfig, enterprise_sso
from .ldap_provider import LDAPProvider, LDAPConfig, create_ldap_provider
from .mfa_provider import MFAProvider, MFAConfig, MFAMethod, create_mfa_provider

logger = structlog.get_logger(__name__)


class AuthenticationMethod(Enum):
    """Authentication methods"""
    LOCAL = "local"
    SSO = "sso"
    LDAP = "ldap"


@dataclass
class EnterpriseAuthConfig:
    """Enterprise authentication configuration"""
    # General settings
    enabled: bool = True
    allow_local_auth: bool = True
    require_mfa_for_admin: bool = True
    require_mfa_for_all: bool = False
    
    # SSO settings
    sso_enabled: bool = False
    sso_providers: List[SSOConfig] = None
    sso_auto_provision: bool = True
    
    # LDAP settings
    ldap_enabled: bool = False
    ldap_config: Optional[LDAPConfig] = None
    ldap_auto_sync: bool = True
    ldap_sync_interval_hours: int = 24
    
    # MFA settings
    mfa_enabled: bool = True
    mfa_config: Optional[MFAConfig] = None
    
    # Security settings
    session_timeout_hours: int = 8
    max_concurrent_sessions: int = 5
    enforce_password_policy: bool = True
    audit_login_attempts: bool = True


@dataclass
class AuthenticationResult:
    """Authentication result"""
    success: bool
    user: Optional[User] = None
    token_response: Optional[TokenResponse] = None
    requires_mfa: bool = False
    mfa_challenge_id: Optional[str] = None
    error: Optional[str] = None
    method: Optional[AuthenticationMethod] = None
    provider: Optional[str] = None


class EnterpriseAuthManager:
    """
    Enterprise Authentication Manager
    
    Provides a unified interface for:
    - Local authentication
    - SSO authentication (SAML/OIDC)
    - LDAP authentication
    - Multi-factor authentication
    - User provisioning and synchronization
    """
    
    def __init__(self, config: EnterpriseAuthConfig):
        self.config = config
        self.base_auth = auth_manager
        
        # Enterprise providers
        self.sso_manager: Optional[EnterpriseSSO] = None
        self.ldap_provider: Optional[LDAPProvider] = None
        self.mfa_provider: Optional[MFAProvider] = None
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize enterprise authentication providers"""
        try:
            # Initialize SSO
            if self.config.sso_enabled and self.config.sso_providers:
                self.sso_manager = enterprise_sso
                for sso_config in self.config.sso_providers:
                    self.sso_manager.register_provider(sso_config)
            
            # Initialize LDAP
            if self.config.ldap_enabled and self.config.ldap_config:
                self.ldap_provider = create_ldap_provider(self.config.ldap_config.__dict__)
            
            # Initialize MFA
            if self.config.mfa_enabled and self.config.mfa_config:
                self.mfa_provider = create_mfa_provider(self.config.mfa_config.__dict__)
            
            logger.info("Enterprise auth providers initialized",
                       sso_enabled=self.config.sso_enabled,
                       ldap_enabled=self.config.ldap_enabled,
                       mfa_enabled=self.config.mfa_enabled)
            
        except Exception as e:
            logger.error("Failed to initialize enterprise auth providers", error=str(e))
            raise
    
    async def initialize(self):
        """Initialize the enterprise authentication system"""
        try:
            # Initialize base auth manager
            await self.base_auth.initialize()
            
            # Initialize SSO providers
            if self.sso_manager:
                await self.sso_manager.initialize_providers()
            
            # Test LDAP connection
            if self.ldap_provider:
                connection_test = await self.ldap_provider.test_connection()
                if not connection_test['success']:
                    logger.warning("LDAP connection test failed", error=connection_test.get('error'))
            
            logger.info("Enterprise authentication system initialized")
            
        except Exception as e:
            logger.error("Failed to initialize enterprise auth system", error=str(e))
            raise
    
    async def authenticate(self, 
                          method: AuthenticationMethod,
                          credentials: Dict[str, Any],
                          client_info: Optional[Dict[str, Any]] = None) -> AuthenticationResult:
        """
        Authenticate user using specified method
        
        Args:
            method: Authentication method to use
            credentials: Authentication credentials
            client_info: Client information for audit logging
            
        Returns:
            Authentication result with user and tokens
        """
        try:
            if method == AuthenticationMethod.LOCAL:
                return await self._authenticate_local(credentials, client_info)
            
            elif method == AuthenticationMethod.SSO:
                return await self._authenticate_sso(credentials, client_info)
            
            elif method == AuthenticationMethod.LDAP:
                return await self._authenticate_ldap(credentials, client_info)
            
            else:
                return AuthenticationResult(
                    success=False,
                    error=f"Unsupported authentication method: {method.value}"
                )
                
        except Exception as e:
            logger.error("Authentication error", method=method.value, error=str(e))
            return AuthenticationResult(
                success=False,
                error="Authentication failed"
            )
    
    async def _authenticate_local(self, 
                                 credentials: Dict[str, Any],
                                 client_info: Optional[Dict[str, Any]] = None) -> AuthenticationResult:
        """Authenticate using local credentials"""
        try:
            if not self.config.allow_local_auth:
                return AuthenticationResult(
                    success=False,
                    error="Local authentication disabled"
                )
            
            # Use base auth manager for local authentication
            from ..models import LoginRequest
            login_request = LoginRequest(**credentials)
            
            token_response = await self.base_auth.authenticate_user(login_request, client_info)
            
            # Get user for MFA check
            user = await self.base_auth.get_current_user(token_response.access_token)
            
            # Check if MFA is required
            if self.mfa_provider and self._requires_mfa(user):
                # Initiate MFA challenge
                challenge = await self.mfa_provider.initiate_challenge(str(user.id))
                
                return AuthenticationResult(
                    success=True,
                    user=user,
                    requires_mfa=True,
                    mfa_challenge_id=challenge.id,
                    method=AuthenticationMethod.LOCAL
                )
            
            return AuthenticationResult(
                success=True,
                user=user,
                token_response=token_response,
                method=AuthenticationMethod.LOCAL
            )
            
        except HTTPException as e:
            return AuthenticationResult(
                success=False,
                error=e.detail
            )
        except Exception as e:
            logger.error("Local authentication error", error=str(e))
            return AuthenticationResult(
                success=False,
                error="Local authentication failed"
            )
    
    async def _authenticate_sso(self,
                               credentials: Dict[str, Any],
                               client_info: Optional[Dict[str, Any]] = None) -> AuthenticationResult:
        """Authenticate using SSO"""
        try:
            if not self.sso_manager:
                return AuthenticationResult(
                    success=False,
                    error="SSO not configured"
                )
            
            provider_name = credentials.get('provider')
            if not provider_name:
                return AuthenticationResult(
                    success=False,
                    error="SSO provider not specified"
                )
            
            # Process SSO callback
            sso_response = await self.sso_manager.process_callback(provider_name, **credentials)
            
            if not sso_response.success:
                return AuthenticationResult(
                    success=False,
                    error=sso_response.error,
                    provider=provider_name
                )
            
            # Create or get user from SSO response
            user = await self._provision_sso_user(sso_response)
            
            # Generate tokens
            token_response = await self._generate_tokens(user, client_info)
            
            # Check if MFA is required
            if self.mfa_provider and self._requires_mfa(user):
                challenge = await self.mfa_provider.initiate_challenge(str(user.id))
                
                return AuthenticationResult(
                    success=True,
                    user=user,
                    requires_mfa=True,
                    mfa_challenge_id=challenge.id,
                    method=AuthenticationMethod.SSO,
                    provider=provider_name
                )
            
            return AuthenticationResult(
                success=True,
                user=user,
                token_response=token_response,
                method=AuthenticationMethod.SSO,
                provider=provider_name
            )
            
        except Exception as e:
            logger.error("SSO authentication error", error=str(e))
            return AuthenticationResult(
                success=False,
                error="SSO authentication failed"
            )
    
    async def _authenticate_ldap(self,
                                credentials: Dict[str, Any],
                                client_info: Optional[Dict[str, Any]] = None) -> AuthenticationResult:
        """Authenticate using LDAP"""
        try:
            if not self.ldap_provider:
                return AuthenticationResult(
                    success=False,
                    error="LDAP not configured"
                )
            
            username = credentials.get('username')
            password = credentials.get('password')
            
            if not username or not password:
                return AuthenticationResult(
                    success=False,
                    error="Username and password required"
                )
            
            # Authenticate with LDAP
            ldap_user = await self.ldap_provider.authenticate_user(username, password)
            
            if not ldap_user:
                return AuthenticationResult(
                    success=False,
                    error="Invalid LDAP credentials"
                )
            
            # Create or update user from LDAP
            user = await self._provision_ldap_user(ldap_user)
            
            # Generate tokens
            token_response = await self._generate_tokens(user, client_info)
            
            # Check if MFA is required
            if self.mfa_provider and self._requires_mfa(user):
                challenge = await self.mfa_provider.initiate_challenge(str(user.id))
                
                return AuthenticationResult(
                    success=True,
                    user=user,
                    requires_mfa=True,
                    mfa_challenge_id=challenge.id,
                    method=AuthenticationMethod.LDAP
                )
            
            return AuthenticationResult(
                success=True,
                user=user,
                token_response=token_response,
                method=AuthenticationMethod.LDAP
            )
            
        except Exception as e:
            logger.error("LDAP authentication error", error=str(e))
            return AuthenticationResult(
                success=False,
                error="LDAP authentication failed"
            )
    
    async def complete_mfa_challenge(self,
                                   challenge_id: str,
                                   verification_code: str,
                                   client_info: Optional[Dict[str, Any]] = None) -> AuthenticationResult:
        """Complete MFA challenge"""
        try:
            if not self.mfa_provider:
                return AuthenticationResult(
                    success=False,
                    error="MFA not configured"
                )
            
            # Verify MFA challenge
            is_valid = await self.mfa_provider.verify_challenge(challenge_id, verification_code)
            
            if not is_valid:
                return AuthenticationResult(
                    success=False,
                    error="Invalid MFA code"
                )
            
            # Get challenge to find user
            challenge = self.mfa_provider.active_challenges.get(challenge_id)
            if not challenge:
                return AuthenticationResult(
                    success=False,
                    error="MFA challenge not found"
                )
            
            # Get user
            user = await self.base_auth._get_user_by_id(challenge.user_id)
            if not user:
                return AuthenticationResult(
                    success=False,
                    error="User not found"
                )
            
            # Generate final tokens
            token_response = await self._generate_tokens(user, client_info)
            
            return AuthenticationResult(
                success=True,
                user=user,
                token_response=token_response
            )
            
        except Exception as e:
            logger.error("MFA challenge completion error", error=str(e))
            return AuthenticationResult(
                success=False,
                error="MFA verification failed"
            )
    
    async def _provision_sso_user(self, sso_response) -> User:
        """Provision user from SSO response"""
        # Check if user exists
        email = sso_response.user_attributes.get('email')
        if not email:
            raise ValueError("Email required for user provisioning")
        
        # For now, create user from SSO data
        # In a real implementation, this would check/update the database
        user = self.sso_manager.create_user_from_sso(sso_response)
        
        logger.info("User provisioned from SSO",
                   email=user.email,
                   provider=sso_response.provider)
        
        return user
    
    async def _provision_ldap_user(self, ldap_user) -> User:
        """Provision user from LDAP data"""
        # Check if user exists
        # For now, create user from LDAP data
        # In a real implementation, this would check/update the database
        user = self.ldap_provider.create_user_from_ldap(ldap_user)
        
        logger.info("User provisioned from LDAP",
                   email=user.email,
                   username=user.username)
        
        return user
    
    async def _generate_tokens(self, user: User, client_info: Optional[Dict[str, Any]] = None) -> TokenResponse:
        """Generate authentication tokens for user"""
        from ..jwt_handler import jwt_handler
        
        user_data = {
            "user_id": str(user.id),
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "permissions": [p.value for p in user.get_permissions()],
            "client_info": client_info or {}
        }
        
        access_token, access_token_data = await jwt_handler.create_access_token(user_data)
        refresh_token, refresh_token_data = await jwt_handler.create_refresh_token(user_data)
        
        expires_in = int((access_token_data.expires_at - datetime.now(timezone.utc)).total_seconds())
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=expires_in
        )
    
    def _requires_mfa(self, user: User) -> bool:
        """Check if user requires MFA"""
        if not self.mfa_provider:
            return False
        
        return self.mfa_provider.is_mfa_required(user)
    
    async def get_authentication_methods(self) -> List[Dict[str, Any]]:
        """Get available authentication methods"""
        methods = []
        
        if self.config.allow_local_auth:
            methods.append({
                'method': 'local',
                'name': 'Username/Password',
                'enabled': True
            })
        
        if self.sso_manager:
            sso_providers = self.sso_manager.get_available_providers()
            for provider in sso_providers:
                methods.append({
                    'method': 'sso',
                    'name': f"SSO ({provider['name']})",
                    'provider': provider['name'],
                    'enabled': provider['enabled']
                })
        
        if self.ldap_provider:
            methods.append({
                'method': 'ldap',
                'name': 'LDAP/Active Directory',
                'enabled': self.config.ldap_enabled
            })
        
        return methods
    
    async def initiate_sso_login(self, provider_name: str, return_url: Optional[str] = None) -> Dict[str, str]:
        """Initiate SSO login"""
        if not self.sso_manager:
            raise ValueError("SSO not configured")
        
        return await self.sso_manager.initiate_login(provider_name, return_url)
    
    async def enroll_mfa_device(self, user_id: str, method: MFAMethod, 
                               device_name: str, **kwargs) -> Dict[str, Any]:
        """Enroll MFA device for user"""
        if not self.mfa_provider:
            raise ValueError("MFA not configured")
        
        device = await self.mfa_provider.enroll_device(user_id, method, device_name, **kwargs)
        
        result = {
            'device_id': device.id,
            'method': device.method.value,
            'name': device.name,
            'verified': device.verified
        }
        
        # Add QR code for TOTP
        if method == MFAMethod.TOTP:
            user = await self.base_auth._get_user_by_id(user_id)
            if user:
                qr_code = self.mfa_provider.get_qr_code(device.id, user.email)
                if qr_code:
                    import base64
                    result['qr_code'] = base64.b64encode(qr_code).decode()
        
        return result
    
    async def get_user_mfa_devices(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's MFA devices"""
        if not self.mfa_provider:
            return []
        
        devices = self.mfa_provider.get_user_devices(user_id)
        
        return [
            {
                'device_id': device.id,
                'method': device.method.value,
                'name': device.name,
                'verified': device.verified,
                'created_at': device.created_at.isoformat() if device.created_at else None,
                'last_used': device.last_used.isoformat() if device.last_used else None
            }
            for device in devices
        ]
    
    async def sync_ldap_users(self, force: bool = False) -> Dict[str, Any]:
        """Synchronize users from LDAP"""
        if not self.ldap_provider:
            return {'error': 'LDAP not configured'}
        
        from .ldap_provider import LDAPSync
        sync_manager = LDAPSync(self.ldap_provider)
        
        return await sync_manager.sync_users(force)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get enterprise auth system status"""
        return {
            'enabled': self.config.enabled,
            'local_auth': self.config.allow_local_auth,
            'sso': {
                'enabled': self.config.sso_enabled,
                'providers': len(self.config.sso_providers or [])
            },
            'ldap': {
                'enabled': self.config.ldap_enabled,
                'configured': self.ldap_provider is not None
            },
            'mfa': {
                'enabled': self.config.mfa_enabled,
                'configured': self.mfa_provider is not None,
                'required_for_admin': self.config.require_mfa_for_admin,
                'required_for_all': self.config.require_mfa_for_all
            }
        }


# Factory function for creating enterprise auth manager
def create_enterprise_auth_manager(config: Dict[str, Any]) -> EnterpriseAuthManager:
    """Create enterprise authentication manager from configuration"""
    enterprise_config = EnterpriseAuthConfig(**config)
    return EnterpriseAuthManager(enterprise_config)


# Global enterprise auth manager instance
enterprise_auth_manager: Optional[EnterpriseAuthManager] = None


async def get_enterprise_auth_manager() -> EnterpriseAuthManager:
    """Get global enterprise auth manager instance"""
    global enterprise_auth_manager
    
    if enterprise_auth_manager is None:
        # Load configuration from environment or config file
        config = {
            'enabled': True,
            'allow_local_auth': True,
            'mfa_enabled': True,
            # Add other configuration...
        }
        enterprise_auth_manager = create_enterprise_auth_manager(config)
        await enterprise_auth_manager.initialize()
    
    return enterprise_auth_manager