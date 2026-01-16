"""
Enterprise SSO Provider for PRSM
===============================

Supports SAML 2.0 and OpenID Connect (OIDC) for enterprise single sign-on.
Provides seamless integration with enterprise identity providers.
"""

import asyncio
import base64
import json
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union
from urllib.parse import urlencode, urlparse, parse_qs
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import aiohttp
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509 import load_pem_x509_certificate
import structlog

from ..models import User, UserRole

logger = structlog.get_logger(__name__)


@dataclass
class SSOConfig:
    """SSO configuration"""
    provider_type: str  # 'saml' or 'oidc'
    provider_name: str
    entity_id: str
    sso_url: str
    slo_url: Optional[str] = None
    certificate: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    scopes: List[str] = None
    attribute_mapping: Dict[str, str] = None
    auto_provision: bool = True
    default_role: UserRole = UserRole.USER
    enabled: bool = True


@dataclass
class SSOResponse:
    """SSO authentication response"""
    success: bool
    user_attributes: Dict[str, Any]
    provider: str
    session_id: Optional[str] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None


class SAMLProvider:
    """SAML 2.0 SSO Provider"""
    
    def __init__(self, config: SSOConfig):
        self.config = config
        self.certificate = None
        if config.certificate:
            self.certificate = load_pem_x509_certificate(config.certificate.encode())
    
    async def generate_auth_request(self, relay_state: Optional[str] = None) -> Dict[str, str]:
        """Generate SAML authentication request"""
        try:
            request_id = f"_id{secrets.token_hex(16)}"
            issue_instant = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            # SAML AuthnRequest XML
            authn_request = f"""<?xml version="1.0" encoding="UTF-8"?>
<samlp:AuthnRequest xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
                    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
                    ID="{request_id}"
                    Version="2.0"
                    IssueInstant="{issue_instant}"
                    Destination="{self.config.sso_url}"
                    AssertionConsumerServiceURL="https://prsm.ai/auth/sso/saml/callback"
                    ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST">
    <saml:Issuer>{self.config.entity_id}</saml:Issuer>
    <samlp:NameIDPolicy Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
                        AllowCreate="true"/>
    <samlp:RequestedAuthnContext Comparison="exact">
        <saml:AuthnContextClassRef>urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport</saml:AuthnContextClassRef>
    </samlp:RequestedAuthnContext>
</samlp:AuthnRequest>"""
            
            # Base64 encode the request
            encoded_request = base64.b64encode(authn_request.encode()).decode()
            
            # Build redirect URL
            params = {'SAMLRequest': encoded_request}
            if relay_state:
                params['RelayState'] = relay_state
            
            redirect_url = f"{self.config.sso_url}?{urlencode(params)}"
            
            return {
                'redirect_url': redirect_url,
                'request_id': request_id,
                'relay_state': relay_state
            }
            
        except Exception as e:
            logger.error("Failed to generate SAML auth request", error=str(e))
            raise
    
    async def process_response(self, saml_response: str, relay_state: Optional[str] = None) -> SSOResponse:
        """Process SAML authentication response"""
        try:
            # Decode the SAML response
            decoded_response = base64.b64decode(saml_response).decode()
            
            # Parse XML securely (preventing XXE attacks)
            # Use defusedxml if available, otherwise configure ET parser for safety
            try:
                from defusedxml import ElementTree as DefusedET
                root = DefusedET.fromstring(decoded_response)
            except ImportError:
                # Fallback: Skip XML processing if defusedxml not available for security
                logger.warning("defusedxml not available - skipping SAML response processing for security")
                raise ValueError("SAML response processing requires defusedxml for security")
            
            # Define namespaces
            namespaces = {
                'samlp': 'urn:oasis:names:tc:SAML:2.0:protocol',
                'saml': 'urn:oasis:names:tc:SAML:2.0:assertion'
            }
            
            # Check response status
            status_code = root.find('.//samlp:StatusCode', namespaces)
            if status_code is None or status_code.attrib.get('Value') != 'urn:oasis:names:tc:SAML:2.0:status:Success':
                return SSOResponse(
                    success=False,
                    user_attributes={},
                    provider=self.config.provider_name,
                    error="SAML authentication failed"
                )
            
            # Extract assertion
            assertion = root.find('.//saml:Assertion', namespaces)
            if assertion is None:
                return SSOResponse(
                    success=False,
                    user_attributes={},
                    provider=self.config.provider_name,
                    error="No assertion found in SAML response"
                )
            
            # Verify assertion signature if certificate is available
            if self.certificate:
                # Signature verification would go here
                pass
            
            # Extract user attributes
            user_attributes = {}
            attribute_statements = assertion.findall('.//saml:AttributeStatement/saml:Attribute', namespaces)
            
            for attr in attribute_statements:
                attr_name = attr.attrib.get('Name', '').lower()
                attr_values = [val.text for val in attr.findall('.//saml:AttributeValue', namespaces)]
                
                if attr_values:
                    # Map SAML attributes to user attributes
                    mapped_name = self._map_attribute(attr_name)
                    if mapped_name:
                        user_attributes[mapped_name] = attr_values[0] if len(attr_values) == 1 else attr_values
            
            # Extract subject NameID as email if available
            name_id = assertion.find('.//saml:Subject/saml:NameID', namespaces)
            if name_id is not None and name_id.text:
                if 'email' not in user_attributes:
                    user_attributes['email'] = name_id.text
            
            # Extract session information
            authn_statement = assertion.find('.//saml:AuthnStatement', namespaces)
            session_id = None
            if authn_statement is not None:
                session_id = authn_statement.attrib.get('SessionIndex')
            
            return SSOResponse(
                success=True,
                user_attributes=user_attributes,
                provider=self.config.provider_name,
                session_id=session_id,
                raw_response=decoded_response
            )
            
        except Exception as e:
            logger.error("Failed to process SAML response", error=str(e))
            return SSOResponse(
                success=False,
                user_attributes={},
                provider=self.config.provider_name,
                error=f"SAML response processing error: {str(e)}"
            )
    
    def _map_attribute(self, saml_attribute: str) -> Optional[str]:
        """Map SAML attribute to user attribute"""
        default_mapping = {
            'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress': 'email',
            'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname': 'first_name',
            'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname': 'last_name',
            'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name': 'full_name',
            'http://schemas.microsoft.com/ws/2008/06/identity/claims/groups': 'groups',
            'email': 'email',
            'firstname': 'first_name',
            'lastname': 'last_name',
            'displayname': 'full_name',
            'groups': 'groups'
        }
        
        # Use custom mapping if available
        if self.config.attribute_mapping:
            return self.config.attribute_mapping.get(saml_attribute)
        
        return default_mapping.get(saml_attribute.lower())


class OIDCProvider:
    """OpenID Connect SSO Provider"""

    def __init__(self, config: SSOConfig):
        self.config = config
        self.discovery_document = None
        self.jwks = None
        self._jwks_client = None  # PyJWT JWKS client for key retrieval
    
    async def initialize(self):
        """Initialize OIDC provider by fetching discovery document"""
        try:
            # Fetch OpenID Connect discovery document
            discovery_url = f"{self.config.sso_url}/.well-known/openid_configuration"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(discovery_url) as response:
                    if response.status == 200:
                        self.discovery_document = await response.json()
                    else:
                        raise Exception(f"Failed to fetch OIDC discovery document: {response.status}")
                
                # Fetch JWKS and create JWT client for signature verification
                if self.discovery_document and 'jwks_uri' in self.discovery_document:
                    jwks_uri = self.discovery_document['jwks_uri']
                    async with session.get(jwks_uri) as jwks_response:
                        if jwks_response.status == 200:
                            self.jwks = await jwks_response.json()

                    # Create PyJWKClient for JWKS-based signature verification
                    try:
                        from jwt import PyJWKClient
                        self._jwks_client = PyJWKClient(jwks_uri, cache_keys=True)
                        logger.info("JWKS client initialized for signature verification",
                                   provider=self.config.provider_name,
                                   jwks_uri=jwks_uri)
                    except Exception as jwks_err:
                        logger.error("Failed to create JWKS client - signature verification disabled",
                                    provider=self.config.provider_name,
                                    error=str(jwks_err))
                        self._jwks_client = None

            logger.info("OIDC provider initialized", provider=self.config.provider_name)
            
        except Exception as e:
            logger.error("Failed to initialize OIDC provider", provider=self.config.provider_name, error=str(e))
            raise
    
    async def generate_auth_request(self, state: Optional[str] = None) -> Dict[str, str]:
        """Generate OIDC authentication request"""
        try:
            if not self.discovery_document:
                await self.initialize()
            
            # Generate state and nonce
            state = state or secrets.token_urlsafe(32)
            nonce = secrets.token_urlsafe(32)
            
            # Build authorization URL
            auth_endpoint = self.discovery_document['authorization_endpoint']
            
            params = {
                'client_id': self.config.client_id,
                'response_type': 'code',
                'scope': ' '.join(self.config.scopes or ['openid', 'email', 'profile']),
                'redirect_uri': 'https://prsm.ai/auth/sso/oidc/callback',
                'state': state,
                'nonce': nonce
            }
            
            auth_url = f"{auth_endpoint}?{urlencode(params)}"
            
            return {
                'auth_url': auth_url,
                'state': state,
                'nonce': nonce
            }
            
        except Exception as e:
            logger.error("Failed to generate OIDC auth request", error=str(e))
            raise
    
    async def exchange_code(self, code: str, state: str) -> SSOResponse:
        """Exchange authorization code for tokens"""
        try:
            if not self.discovery_document:
                await self.initialize()
            
            token_endpoint = self.discovery_document['token_endpoint']
            
            # Exchange code for tokens
            data = {
                'grant_type': 'authorization_code',
                'client_id': self.config.client_id,
                'client_secret': self.config.client_secret,
                'code': code,
                'redirect_uri': 'https://prsm.ai/auth/sso/oidc/callback'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(token_endpoint, data=data) as response:
                    if response.status == 200:
                        token_response = await response.json()
                    else:
                        error_text = await response.text()
                        return SSOResponse(
                            success=False,
                            user_attributes={},
                            provider=self.config.provider_name,
                            error=f"Token exchange failed: {error_text}"
                        )
            
            # Verify and decode ID token
            id_token = token_response.get('id_token')
            if not id_token:
                return SSOResponse(
                    success=False,
                    user_attributes={},
                    provider=self.config.provider_name,
                    error="No ID token received"
                )

            # SECURITY FIX: Properly verify ID token signature using JWKS
            # Previously used verify_signature=False which allowed token forgery
            try:
                payload = await self._verify_id_token(id_token)
            except jwt.InvalidTokenError as e:
                logger.error("ID token verification failed",
                           provider=self.config.provider_name,
                           error=str(e))
                return SSOResponse(
                    success=False,
                    user_attributes={},
                    provider=self.config.provider_name,
                    error=f"Invalid ID token: {str(e)}"
                )
            except Exception as e:
                logger.error("Unexpected error during ID token verification",
                           provider=self.config.provider_name,
                           error=str(e))
                return SSOResponse(
                    success=False,
                    user_attributes={},
                    provider=self.config.provider_name,
                    error=f"Token verification error: {str(e)}"
                )
            
            # Extract user attributes
            user_attributes = {}
            for claim, value in payload.items():
                mapped_name = self._map_claim(claim)
                if mapped_name:
                    user_attributes[mapped_name] = value
            
            return SSOResponse(
                success=True,
                user_attributes=user_attributes,
                provider=self.config.provider_name,
                raw_response=json.dumps(token_response)
            )
            
        except Exception as e:
            logger.error("Failed to exchange OIDC code", error=str(e))
            return SSOResponse(
                success=False,
                user_attributes={},
                provider=self.config.provider_name,
                error=f"Code exchange error: {str(e)}"
            )
    
    async def _verify_id_token(self, id_token: str) -> Dict[str, Any]:
        """
        Verify OIDC ID token signature and claims using JWKS.

        SECURITY: This method properly verifies the JWT signature using the
        identity provider's JWKS public keys. It also validates standard
        OIDC claims including issuer, audience, and expiration.

        Args:
            id_token: The JWT ID token to verify

        Returns:
            Dict containing the verified token payload

        Raises:
            jwt.InvalidTokenError: If token verification fails
            ValueError: If JWKS client is not initialized
        """
        # Ensure JWKS client is available
        if not self._jwks_client:
            # Attempt to reinitialize
            if self.discovery_document and 'jwks_uri' in self.discovery_document:
                try:
                    from jwt import PyJWKClient
                    self._jwks_client = PyJWKClient(
                        self.discovery_document['jwks_uri'],
                        cache_keys=True
                    )
                except Exception as e:
                    logger.error("Failed to initialize JWKS client", error=str(e))
                    raise ValueError(
                        "JWKS client not available - cannot verify token signature. "
                        "This is a CRITICAL security requirement for OIDC authentication."
                    )
            else:
                raise ValueError(
                    "OIDC provider not properly initialized - missing JWKS configuration"
                )

        # Get the signing key from JWKS based on token header's 'kid'
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(id_token)
        except Exception as e:
            logger.error("Failed to get signing key from JWKS",
                        provider=self.config.provider_name,
                        error=str(e))
            raise jwt.InvalidTokenError(f"Unable to find signing key: {str(e)}")

        # Determine expected issuer from discovery document
        expected_issuer = None
        if self.discovery_document:
            expected_issuer = self.discovery_document.get('issuer')

        # Verify and decode the token with full validation
        # CRITICAL: verify_signature MUST be True (default) for security
        decode_options = {
            "verify_signature": True,  # CRITICAL: Must verify signature
            "verify_exp": True,        # Verify expiration
            "verify_iat": True,        # Verify issued-at
            "verify_aud": True,        # Verify audience
            "require": ["exp", "iat", "sub", "aud"]  # Required claims
        }

        # Build verification parameters
        verify_params = {
            "key": signing_key.key,
            "algorithms": ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
            "options": decode_options,
            "audience": self.config.client_id,  # Audience must match our client_id
        }

        # Add issuer verification if available
        if expected_issuer:
            verify_params["issuer"] = expected_issuer

        payload = jwt.decode(id_token, **verify_params)

        logger.info("ID token verified successfully",
                   provider=self.config.provider_name,
                   subject=payload.get('sub', 'unknown'))

        return payload

    def _map_claim(self, claim: str) -> Optional[str]:
        """Map OIDC claim to user attribute"""
        default_mapping = {
            'email': 'email',
            'given_name': 'first_name',
            'family_name': 'last_name',
            'name': 'full_name',
            'preferred_username': 'username',
            'groups': 'groups',
            'roles': 'roles'
        }

        # Use custom mapping if available
        if self.config.attribute_mapping:
            return self.config.attribute_mapping.get(claim)

        return default_mapping.get(claim)


class EnterpriseSSO:
    """Enterprise SSO Manager"""
    
    def __init__(self):
        self.providers: Dict[str, Union[SAMLProvider, OIDCProvider]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def register_provider(self, config: SSOConfig):
        """Register an SSO provider"""
        try:
            if config.provider_type.lower() == 'saml':
                provider = SAMLProvider(config)
            elif config.provider_type.lower() == 'oidc':
                provider = OIDCProvider(config)
            else:
                raise ValueError(f"Unsupported provider type: {config.provider_type}")
            
            self.providers[config.provider_name] = provider
            logger.info("SSO provider registered", 
                       provider=config.provider_name, 
                       type=config.provider_type)
            
        except Exception as e:
            logger.error("Failed to register SSO provider", 
                        provider=config.provider_name, 
                        error=str(e))
            raise
    
    async def initialize_providers(self):
        """Initialize all registered providers"""
        for name, provider in self.providers.items():
            try:
                if hasattr(provider, 'initialize'):
                    await provider.initialize()
                logger.info("SSO provider initialized", provider=name)
            except Exception as e:
                logger.error("Failed to initialize SSO provider", provider=name, error=str(e))
    
    def get_available_providers(self) -> List[Dict[str, str]]:
        """Get list of available SSO providers"""
        return [
            {
                'name': provider.config.provider_name,
                'type': provider.config.provider_type,
                'enabled': provider.config.enabled
            }
            for provider in self.providers.values()
            if provider.config.enabled
        ]
    
    async def initiate_login(self, provider_name: str, return_url: Optional[str] = None) -> Dict[str, str]:
        """Initiate SSO login with a provider"""
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Unknown SSO provider: {provider_name}")
        
        if not provider.config.enabled:
            raise ValueError(f"SSO provider disabled: {provider_name}")
        
        # Generate login request
        if provider.config.provider_type == 'saml':
            return await provider.generate_auth_request(return_url)
        elif provider.config.provider_type == 'oidc':
            return await provider.generate_auth_request(return_url)
        else:
            raise ValueError(f"Unsupported provider type: {provider.config.provider_type}")
    
    async def process_callback(self, provider_name: str, **kwargs) -> SSOResponse:
        """Process SSO callback response"""
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Unknown SSO provider: {provider_name}")
        
        # Process response based on provider type
        if provider.config.provider_type == 'saml':
            saml_response = kwargs.get('SAMLResponse')
            relay_state = kwargs.get('RelayState')
            if not saml_response:
                return SSOResponse(
                    success=False,
                    user_attributes={},
                    provider=provider_name,
                    error="No SAML response received"
                )
            return await provider.process_response(saml_response, relay_state)
        
        elif provider.config.provider_type == 'oidc':
            code = kwargs.get('code')
            state = kwargs.get('state')
            if not code:
                return SSOResponse(
                    success=False,
                    user_attributes={},
                    provider=provider_name,
                    error="No authorization code received"
                )
            return await provider.exchange_code(code, state)
        
        else:
            return SSOResponse(
                success=False,
                user_attributes={},
                provider=provider_name,
                error=f"Unsupported provider type: {provider.config.provider_type}"
            )
    
    def create_user_from_sso(self, sso_response: SSOResponse) -> User:
        """Create or update user from SSO response"""
        provider = self.providers.get(sso_response.provider)
        if not provider:
            raise ValueError(f"Unknown provider: {sso_response.provider}")
        
        attrs = sso_response.user_attributes
        
        # Extract required attributes
        email = attrs.get('email')
        if not email:
            raise ValueError("Email attribute required for user creation")
        
        # Create user object
        user = User(
            email=email,
            username=attrs.get('username', email.split('@')[0]),
            full_name=attrs.get('full_name', f"{attrs.get('first_name', '')} {attrs.get('last_name', '')}").strip(),
            is_active=True,
            is_verified=True,  # SSO users are pre-verified
            role=self._determine_user_role(attrs, provider.config),
            sso_provider=sso_response.provider,
            sso_subject_id=attrs.get('sub', email)
        )
        
        return user
    
    def _determine_user_role(self, attributes: Dict[str, Any], config: SSOConfig) -> UserRole:
        """Determine user role based on SSO attributes"""
        # Check for groups/roles in attributes
        groups = attributes.get('groups', [])
        roles = attributes.get('roles', [])
        
        # Map groups/roles to PRSM roles (customizable)
        admin_groups = ['prsm_admins', 'administrators', 'admin']
        moderator_groups = ['prsm_moderators', 'moderators', 'mod']
        
        if isinstance(groups, str):
            groups = [groups]
        if isinstance(roles, str):
            roles = [roles]
        
        all_groups = (groups or []) + (roles or [])
        
        for group in all_groups:
            if group.lower() in admin_groups:
                return UserRole.ADMIN
            elif group.lower() in moderator_groups:
                return UserRole.MODERATOR
        
        return config.default_role


# Global SSO manager instance
enterprise_sso = EnterpriseSSO()