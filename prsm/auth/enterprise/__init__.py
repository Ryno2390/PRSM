"""
PRSM Enterprise Authentication Framework
=======================================

Comprehensive enterprise authentication system providing:
- Single Sign-On (SSO) with SAML 2.0 and OpenID Connect
- LDAP/Active Directory integration  
- Multi-Factor Authentication (MFA)
- Unified enterprise authentication management
- REST API for all enterprise auth features

Key Components:
- EnterpriseAuthManager: Main orchestrator for all enterprise auth
- SSOProvider: SAML/OIDC SSO authentication
- LDAPProvider: LDAP/AD authentication and user sync
- MFAProvider: Multi-factor authentication with TOTP, SMS, Email
- API: REST endpoints for all enterprise auth features

Example Usage:
    from prsm.auth.enterprise import (
        create_enterprise_auth_manager,
        EnterpriseAuthConfig,
        SSOConfig,
        LDAPConfig, 
        MFAConfig
    )
    
    # Configure enterprise authentication
    config = EnterpriseAuthConfig(
        sso_enabled=True,
        ldap_enabled=True,
        mfa_enabled=True,
        require_mfa_for_admin=True
    )
    
    # Create and initialize manager
    auth_manager = create_enterprise_auth_manager(config)
    await auth_manager.initialize()
    
    # Authenticate user
    result = await auth_manager.authenticate(
        AuthenticationMethod.LDAP,
        {'username': 'user@domain.com', 'password': 'password'}
    )
"""

from .enterprise_auth import (
    EnterpriseAuthManager,
    EnterpriseAuthConfig,
    AuthenticationMethod,
    AuthenticationResult,
    create_enterprise_auth_manager,
    get_enterprise_auth_manager
)

from .sso_provider import (
    EnterpriseSSO,
    SSOConfig,
    SSOResponse,
    SAMLProvider,
    OIDCProvider,
    enterprise_sso
)

from .ldap_provider import (
    LDAPProvider,
    LDAPConfig,
    LDAPUser,
    LDAPSync,
    create_ldap_provider
)

from .mfa_provider import (
    MFAProvider,
    MFAConfig,
    MFAMethod,
    MFADevice,
    MFAChallenge,
    TOTPProvider,
    SMSProvider,
    EmailProvider,
    BackupCodesProvider,
    create_mfa_provider
)

from .api import enterprise_router

__all__ = [
    # Enterprise Auth Manager
    "EnterpriseAuthManager",
    "EnterpriseAuthConfig", 
    "AuthenticationMethod",
    "AuthenticationResult",
    "create_enterprise_auth_manager",
    "get_enterprise_auth_manager",
    
    # SSO Components
    "EnterpriseSSO",
    "SSOConfig",
    "SSOResponse",
    "SAMLProvider", 
    "OIDCProvider",
    "enterprise_sso",
    
    # LDAP Components
    "LDAPProvider",
    "LDAPConfig",
    "LDAPUser",
    "LDAPSync",
    "create_ldap_provider",
    
    # MFA Components
    "MFAProvider",
    "MFAConfig",
    "MFAMethod",
    "MFADevice",
    "MFAChallenge",
    "TOTPProvider",
    "SMSProvider", 
    "EmailProvider",
    "BackupCodesProvider",
    "create_mfa_provider",
    
    # API Router
    "enterprise_router"
]

# Version info
__version__ = "1.0.0"
__author__ = "PRSM Team"
__description__ = "Enterprise authentication framework for PRSM"