"""
Enterprise Authentication API Endpoints
======================================

REST API endpoints for enterprise authentication features including
SSO, LDAP, and MFA management.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Depends, Request, Form, Query
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr

from ..auth_manager import get_current_user, require_permission, require_role
from ..models import User, UserRole, Permission
from .enterprise_auth import get_enterprise_auth_manager, AuthenticationMethod
from .mfa_provider import MFAMethod

import structlog

logger = structlog.get_logger(__name__)

# Create router
enterprise_router = APIRouter(prefix="/auth/enterprise", tags=["Enterprise Authentication"])
security = HTTPBearer()


# Request/Response Models

class AuthMethodsResponse(BaseModel):
    """Available authentication methods response"""
    methods: List[Dict[str, Any]]


class SSOInitiateRequest(BaseModel):
    """SSO login initiation request"""
    provider: str
    return_url: Optional[str] = None


class SSOInitiateResponse(BaseModel):
    """SSO login initiation response"""
    auth_url: Optional[str] = None
    redirect_url: Optional[str] = None
    state: Optional[str] = None
    request_id: Optional[str] = None


class LocalAuthRequest(BaseModel):
    """Local authentication request"""
    username: str
    password: str


class LDAPAuthRequest(BaseModel):
    """LDAP authentication request"""
    username: str
    password: str


class AuthenticationResponse(BaseModel):
    """Authentication response"""
    success: bool
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: Optional[str] = None
    expires_in: Optional[int] = None
    user: Optional[Dict[str, Any]] = None
    requires_mfa: bool = False
    mfa_challenge_id: Optional[str] = None
    error: Optional[str] = None
    method: Optional[str] = None
    provider: Optional[str] = None


class MFAChallengeRequest(BaseModel):
    """MFA challenge completion request"""
    challenge_id: str
    verification_code: str


class MFAEnrollRequest(BaseModel):
    """MFA device enrollment request"""
    method: str
    device_name: str
    phone_number: Optional[str] = None
    email: Optional[EmailStr] = None


class MFAVerifyEnrollmentRequest(BaseModel):
    """MFA enrollment verification request"""
    device_id: str
    verification_code: str


class MFADeviceResponse(BaseModel):
    """MFA device response"""
    device_id: str
    method: str
    name: str
    verified: bool
    created_at: Optional[str] = None
    last_used: Optional[str] = None


class LDAPSyncRequest(BaseModel):
    """LDAP synchronization request"""
    force: bool = False


class SystemStatusResponse(BaseModel):
    """System status response"""
    enabled: bool
    local_auth: bool
    sso: Dict[str, Any]
    ldap: Dict[str, Any]
    mfa: Dict[str, Any]


# API Endpoints

@enterprise_router.get("/methods", response_model=AuthMethodsResponse)
async def get_authentication_methods():
    """Get available authentication methods"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        methods = await enterprise_auth.get_authentication_methods()
        
        return AuthMethodsResponse(methods=methods)
        
    except Exception as e:
        logger.error("Failed to get authentication methods", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve authentication methods"
        )


@enterprise_router.post("/login/local", response_model=AuthenticationResponse)
async def authenticate_local(request: LocalAuthRequest, client_request: Request):
    """Authenticate using local credentials"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        
        # Extract client info
        client_info = {
            'ip_address': client_request.client.host,
            'user_agent': client_request.headers.get('user-agent'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Authenticate
        result = await enterprise_auth.authenticate(
            AuthenticationMethod.LOCAL,
            {'username': request.username, 'password': request.password},
            client_info
        )
        
        # Build response
        response_data = {
            'success': result.success,
            'requires_mfa': result.requires_mfa,
            'mfa_challenge_id': result.mfa_challenge_id,
            'error': result.error,
            'method': result.method.value if result.method else None,
            'provider': result.provider
        }
        
        if result.token_response:
            response_data.update({
                'access_token': result.token_response.access_token,
                'refresh_token': result.token_response.refresh_token,
                'token_type': result.token_response.token_type,
                'expires_in': result.token_response.expires_in
            })
        
        if result.user:
            response_data['user'] = {
                'id': str(result.user.id),
                'username': result.user.username,
                'email': result.user.email,
                'full_name': result.user.full_name,
                'role': result.user.role.value
            }
        
        return AuthenticationResponse(**response_data)
        
    except Exception as e:
        logger.error("Local authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@enterprise_router.post("/login/ldap", response_model=AuthenticationResponse)
async def authenticate_ldap(request: LDAPAuthRequest, client_request: Request):
    """Authenticate using LDAP credentials"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        
        client_info = {
            'ip_address': client_request.client.host,
            'user_agent': client_request.headers.get('user-agent'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        result = await enterprise_auth.authenticate(
            AuthenticationMethod.LDAP,
            {'username': request.username, 'password': request.password},
            client_info
        )
        
        response_data = {
            'success': result.success,
            'requires_mfa': result.requires_mfa,
            'mfa_challenge_id': result.mfa_challenge_id,
            'error': result.error,
            'method': result.method.value if result.method else None,
            'provider': result.provider
        }
        
        if result.token_response:
            response_data.update({
                'access_token': result.token_response.access_token,
                'refresh_token': result.token_response.refresh_token,
                'token_type': result.token_response.token_type,
                'expires_in': result.token_response.expires_in
            })
        
        if result.user:
            response_data['user'] = {
                'id': str(result.user.id),
                'username': result.user.username,
                'email': result.user.email,
                'full_name': result.user.full_name,
                'role': result.user.role.value
            }
        
        return AuthenticationResponse(**response_data)
        
    except Exception as e:
        logger.error("LDAP authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LDAP authentication failed"
        )


@enterprise_router.post("/sso/initiate", response_model=SSOInitiateResponse)
async def initiate_sso_login(request: SSOInitiateRequest):
    """Initiate SSO login"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        
        result = await enterprise_auth.initiate_sso_login(
            request.provider,
            request.return_url
        )
        
        return SSOInitiateResponse(**result)
        
    except Exception as e:
        logger.error("SSO initiation failed", provider=request.provider, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initiate SSO login: {str(e)}"
        )


@enterprise_router.post("/sso/callback/{provider}")
async def sso_callback(
    provider: str,
    client_request: Request,
    SAMLResponse: Optional[str] = Form(None),
    RelayState: Optional[str] = Form(None),
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None)
):
    """Handle SSO callback"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        
        client_info = {
            'ip_address': client_request.client.host,
            'user_agent': client_request.headers.get('user-agent'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Build credentials based on SSO type
        credentials = {'provider': provider}
        
        if SAMLResponse:  # SAML response
            credentials.update({
                'SAMLResponse': SAMLResponse,
                'RelayState': RelayState
            })
        elif code:  # OIDC authorization code
            credentials.update({
                'code': code,
                'state': state
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid SSO callback parameters"
            )
        
        result = await enterprise_auth.authenticate(
            AuthenticationMethod.SSO,
            credentials,
            client_info
        )
        
        # For SSO callbacks, typically redirect to frontend
        if result.success and result.token_response:
            # In a real implementation, you might redirect to frontend with tokens
            return JSONResponse({
                'success': True,
                'access_token': result.token_response.access_token,
                'user': {
                    'id': str(result.user.id),
                    'username': result.user.username,
                    'email': result.user.email,
                    'role': result.user.role.value
                } if result.user else None
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.error or "SSO authentication failed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("SSO callback failed", provider=provider, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SSO callback processing failed"
        )


@enterprise_router.post("/mfa/complete", response_model=AuthenticationResponse)
async def complete_mfa_challenge(request: MFAChallengeRequest, client_request: Request):
    """Complete MFA challenge"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        
        client_info = {
            'ip_address': client_request.client.host,
            'user_agent': client_request.headers.get('user-agent'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        result = await enterprise_auth.complete_mfa_challenge(
            request.challenge_id,
            request.verification_code,
            client_info
        )
        
        response_data = {
            'success': result.success,
            'error': result.error
        }
        
        if result.token_response:
            response_data.update({
                'access_token': result.token_response.access_token,
                'refresh_token': result.token_response.refresh_token,
                'token_type': result.token_response.token_type,
                'expires_in': result.token_response.expires_in
            })
        
        if result.user:
            response_data['user'] = {
                'id': str(result.user.id),
                'username': result.user.username,
                'email': result.user.email,
                'full_name': result.user.full_name,
                'role': result.user.role.value
            }
        
        return AuthenticationResponse(**response_data)
        
    except Exception as e:
        logger.error("MFA challenge completion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA verification failed"
        )


@enterprise_router.post("/mfa/enroll")
async def enroll_mfa_device(
    request: MFAEnrollRequest,
    current_user: User = Depends(get_current_user)
):
    """Enroll MFA device for current user"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        
        # Convert method string to enum
        try:
            method = MFAMethod(request.method)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid MFA method: {request.method}"
            )
        
        # Build kwargs based on method
        kwargs = {}
        if method == MFAMethod.SMS and request.phone_number:
            kwargs['phone_number'] = request.phone_number
        elif method == MFAMethod.EMAIL and request.email:
            kwargs['email'] = request.email
        
        result = await enterprise_auth.enroll_mfa_device(
            str(current_user.id),
            method,
            request.device_name,
            **kwargs
        )
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("MFA enrollment failed", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA enrollment failed"
        )


@enterprise_router.post("/mfa/verify-enrollment")
async def verify_mfa_enrollment(
    request: MFAVerifyEnrollmentRequest,
    current_user: User = Depends(get_current_user)
):
    """Verify MFA device enrollment"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        
        if not enterprise_auth.mfa_provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA not configured"
            )
        
        is_verified = await enterprise_auth.mfa_provider.verify_enrollment(
            request.device_id,
            request.verification_code
        )
        
        return JSONResponse({
            'success': is_verified,
            'message': 'Device verified successfully' if is_verified else 'Invalid verification code'
        })
        
    except Exception as e:
        logger.error("MFA enrollment verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA enrollment verification failed"
        )


@enterprise_router.get("/mfa/devices", response_model=List[MFADeviceResponse])
async def get_user_mfa_devices(current_user: User = Depends(get_current_user)):
    """Get current user's MFA devices"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        
        devices = await enterprise_auth.get_user_mfa_devices(str(current_user.id))
        
        return [MFADeviceResponse(**device) for device in devices]
        
    except Exception as e:
        logger.error("Failed to get MFA devices", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve MFA devices"
        )


@enterprise_router.delete("/mfa/devices/{device_id}")
async def remove_mfa_device(
    device_id: str,
    current_user: User = Depends(get_current_user)
):
    """Remove MFA device"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        
        if not enterprise_auth.mfa_provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA not configured"
            )
        
        success = await enterprise_auth.mfa_provider.remove_device(
            str(current_user.id),
            device_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="MFA device not found"
            )
        
        return JSONResponse({'success': True, 'message': 'Device removed successfully'})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to remove MFA device", device_id=device_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove MFA device"
        )


@enterprise_router.post("/ldap/sync")
async def sync_ldap_users(
    request: LDAPSyncRequest,
    current_user: User = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """Synchronize users from LDAP (Admin only)"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        
        result = await enterprise_auth.sync_ldap_users(request.force)
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error("LDAP sync failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LDAP synchronization failed"
        )


@enterprise_router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    current_user: User = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """Get enterprise authentication system status (Admin only)"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        
        status_data = enterprise_auth.get_system_status()
        
        return SystemStatusResponse(**status_data)
        
    except Exception as e:
        logger.error("Failed to get system status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system status"
        )


@enterprise_router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        enterprise_auth = await get_enterprise_auth_manager()
        
        # Perform basic health checks
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'enterprise_auth': enterprise_auth is not None
        }
        
        # Check LDAP connection if configured
        if enterprise_auth.ldap_provider:
            ldap_test = await enterprise_auth.ldap_provider.test_connection()
            health_status['ldap'] = ldap_test
        
        return JSONResponse(health_status)
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            {'status': 'unhealthy', 'error': str(e)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )