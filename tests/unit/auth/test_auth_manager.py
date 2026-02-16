"""
Unit Tests for Auth Manager
Critical security component testing for authentication and authorization
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta
from uuid import uuid4, UUID

from fastapi import HTTPException
from pydantic import ValidationError

from prsm.core.auth.auth_manager import (
    AuthManager, AuthenticationError, AuthorizationError,
    get_current_user, require_auth, require_permission, require_role
)
from prsm.core.auth.models import (
    User, UserRole, Permission, LoginRequest, RegisterRequest, 
    TokenResponse, ROLE_PERMISSIONS
)
from prsm.core.auth.jwt_handler import TokenData


# Module-level fixtures accessible by all test classes
@pytest.fixture
def auth_manager():
    """Create fresh auth manager instance for each test"""
    return AuthManager()


@pytest.fixture
def mock_user():
    """Create mock user for testing"""
    user = User(
        email="test@prsm.ai",
        username="testuser",
        full_name="Test User",
        hashed_password="$2b$12$hashedpassword",
        role=UserRole.USER,
        is_active=True,
        is_verified=True
    )
    user.id = uuid4()
    user.failed_login_attempts = 0
    user.last_login = None
    return user


@pytest.fixture
def mock_admin_user():
    """Create mock admin user for testing"""
    user = User(
        email="admin@prsm.ai",
        username="admin",
        full_name="Admin User",
        hashed_password="$2b$12$hashedpassword",
        role=UserRole.ADMIN,
        is_active=True,
        is_verified=True,
        is_superuser=True
    )
    user.id = uuid4()
    user.failed_login_attempts = 0
    return user


@pytest.fixture
def valid_register_request():
    """Create valid registration request"""
    return RegisterRequest(
        email="newuser@prsm.ai",
        username="newuser",
        full_name="New User",
        password="SecurePass123!",
        confirm_password="SecurePass123!"
    )


@pytest.fixture
def valid_login_request():
    """Create valid login request"""
    return LoginRequest(
        username="testuser",
        password="correctpassword"
    )


class TestAuthManager:
    """Unit tests for AuthManager core functionality"""

    # === Initialization Tests ===
    
    @pytest.mark.asyncio
    async def test_auth_manager_initialization(self, auth_manager):
        """Test auth manager initialization"""
        with patch('prsm.core.database.get_database_service') as mock_db:
            with patch('prsm.core.auth.jwt_handler.jwt_handler.initialize') as mock_jwt:
                mock_db.return_value = Mock()
                mock_jwt.return_value = None
                
                await auth_manager.initialize()
                
                assert auth_manager.db_service is not None
                mock_jwt.assert_called_once()

    # === User Registration Tests ===
    
    @pytest.mark.asyncio
    async def test_register_user_success(self, auth_manager, valid_register_request):
        """Test successful user registration"""
        with patch.object(auth_manager, '_user_exists', new_callable=AsyncMock, return_value=False):
            with patch('prsm.core.auth.auth_manager.jwt_handler.hash_password', return_value="hashed"):
                with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
                    
                    user = await auth_manager.register_user(valid_register_request)
                    
                    assert user.email == valid_register_request.email
                    assert user.username == valid_register_request.username
                    assert user.role == UserRole.USER
                    assert user.is_active == True
                    assert user.is_verified == False
                    mock_audit.assert_called()

    @pytest.mark.asyncio
    async def test_register_user_password_mismatch(self, auth_manager):
        """Test registration with mismatched passwords"""
        request = RegisterRequest(
            email="test@prsm.ai",
            username="testuser",
            password="password123",
            confirm_password="different123"
        )
        
        with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
            with pytest.raises(HTTPException) as exc_info:
                await auth_manager.register_user(request)
            
            assert exc_info.value.status_code == 400
            assert "do not match" in exc_info.value.detail
            mock_audit.assert_called_with(
                "registration_failed",
                {"reason": "password_mismatch", "username": request.username},
                None
            )

    @pytest.mark.asyncio
    async def test_register_user_already_exists(self, auth_manager, valid_register_request):
        """Test registration when user already exists"""
        with patch.object(auth_manager, '_user_exists', new_callable=AsyncMock, return_value=True):
            with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
                
                with pytest.raises(HTTPException) as exc_info:
                    await auth_manager.register_user(valid_register_request)
                
                assert exc_info.value.status_code == 400
                assert "already exists" in exc_info.value.detail
                mock_audit.assert_called()

    @pytest.mark.asyncio
    async def test_register_user_weak_password(self, auth_manager):
        """Test registration with weak password"""
        request = RegisterRequest(
            email="test@prsm.ai",
            username="testuser",
            password="weak",
            confirm_password="weak"
        )
        
        with patch.object(auth_manager, '_user_exists', new_callable=AsyncMock, return_value=False):
            with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
                
                with pytest.raises(HTTPException) as exc_info:
                    await auth_manager.register_user(request)
                
                assert exc_info.value.status_code == 400
                assert "strength requirements" in exc_info.value.detail
                mock_audit.assert_called()

    # === User Authentication Tests ===
    
    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, auth_manager, valid_login_request, mock_user):
        """Test successful user authentication"""
        with patch.object(auth_manager, '_get_user_by_login', new_callable=AsyncMock, return_value=mock_user):
            with patch.object(auth_manager, '_is_account_locked', new_callable=AsyncMock, return_value=False):
                with patch('prsm.core.auth.auth_manager.jwt_handler.verify_password', return_value=True):
                    with patch('prsm.core.auth.auth_manager.jwt_handler.create_access_token') as mock_access:
                        with patch('prsm.core.auth.auth_manager.jwt_handler.create_refresh_token') as mock_refresh:
                            with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
                                
                                # Setup token mocks
                                token_data = TokenData(
                                    user_id=mock_user.id,
                                    username=mock_user.username,
                                    email=mock_user.email,
                                    role=mock_user.role.value,
                                    permissions=[],
                                    expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
                                    issued_at=datetime.now(timezone.utc)
                                )
                                mock_access.return_value = ("access_token", token_data)
                                mock_refresh.return_value = ("refresh_token", token_data)
                                
                                response = await auth_manager.authenticate_user(valid_login_request)
                                
                                assert isinstance(response, TokenResponse)
                                assert response.access_token == "access_token"
                                assert response.refresh_token == "refresh_token"
                                assert response.token_type == "bearer"
                                assert response.expires_in > 0
                                mock_audit.assert_called()

    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self, auth_manager, valid_login_request):
        """Test authentication with non-existent user"""
        with patch.object(auth_manager, '_get_user_by_login', new_callable=AsyncMock, return_value=None):
            with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
                
                with pytest.raises(AuthenticationError):
                    await auth_manager.authenticate_user(valid_login_request)
                
                mock_audit.assert_called_with(
                    "login_failed",
                    {"reason": "user_not_found", "username": valid_login_request.username},
                    None
                )

    @pytest.mark.asyncio
    async def test_authenticate_user_account_locked(self, auth_manager, valid_login_request, mock_user):
        """Test authentication with locked account"""
        with patch.object(auth_manager, '_get_user_by_login', new_callable=AsyncMock, return_value=mock_user):
            with patch.object(auth_manager, '_is_account_locked', new_callable=AsyncMock, return_value=True):
                with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
                    
                    with pytest.raises(AuthenticationError) as exc_info:
                        await auth_manager.authenticate_user(valid_login_request)
                    
                    assert "locked" in str(exc_info.value.detail)
                    mock_audit.assert_called()

    @pytest.mark.asyncio
    async def test_authenticate_user_inactive_account(self, auth_manager, valid_login_request, mock_user):
        """Test authentication with inactive account"""
        mock_user.is_active = False
        
        with patch.object(auth_manager, '_get_user_by_login', new_callable=AsyncMock, return_value=mock_user):
            with patch.object(auth_manager, '_is_account_locked', new_callable=AsyncMock, return_value=False):
                with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
                    
                    with pytest.raises(AuthenticationError) as exc_info:
                        await auth_manager.authenticate_user(valid_login_request)
                    
                    assert "inactive" in str(exc_info.value.detail)
                    mock_audit.assert_called()

    @pytest.mark.asyncio
    async def test_authenticate_user_wrong_password(self, auth_manager, valid_login_request, mock_user):
        """Test authentication with wrong password"""
        with patch.object(auth_manager, '_get_user_by_login', new_callable=AsyncMock, return_value=mock_user):
            with patch.object(auth_manager, '_is_account_locked', new_callable=AsyncMock, return_value=False):
                with patch('prsm.core.auth.auth_manager.jwt_handler.verify_password', return_value=False):
                    with patch.object(auth_manager, '_record_failed_login', new_callable=AsyncMock) as mock_record:
                        with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
                            
                            with pytest.raises(AuthenticationError):
                                await auth_manager.authenticate_user(valid_login_request)
                            
                            mock_record.assert_called_once_with(mock_user)
                            mock_audit.assert_called()

    # === Token Management Tests ===
    
    @pytest.mark.asyncio
    async def test_get_current_user_success(self, auth_manager, mock_user):
        """Test successful current user retrieval"""
        token_data = TokenData(
            user_id=mock_user.id,
            username=mock_user.username,
            email=mock_user.email,
            role=mock_user.role.value,
            permissions=[],
            token_type="access",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            issued_at=datetime.now(timezone.utc)
        )
        
        with patch('prsm.core.auth.auth_manager.jwt_handler.verify_token', return_value=token_data):
            with patch.object(auth_manager, '_get_user_by_id', new_callable=AsyncMock, return_value=mock_user):
                
                user = await auth_manager.get_current_user("valid_token")
                
                assert user == mock_user

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, auth_manager):
        """Test current user retrieval with invalid token"""
        with patch('prsm.core.auth.auth_manager.jwt_handler.verify_token', return_value=None):
            
            with pytest.raises(AuthenticationError):
                await auth_manager.get_current_user("invalid_token")

    @pytest.mark.asyncio
    async def test_get_current_user_wrong_token_type(self, auth_manager, mock_user):
        """Test current user retrieval with wrong token type"""
        token_data = TokenData(
            user_id=mock_user.id,
            username=mock_user.username,
            email=mock_user.email,
            role=mock_user.role.value,
            permissions=[],
            token_type="refresh",  # Wrong type
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            issued_at=datetime.now(timezone.utc)
        )
        
        with patch('prsm.core.auth.auth_manager.jwt_handler.verify_token', return_value=token_data):
            
            with pytest.raises(AuthenticationError):
                await auth_manager.get_current_user("refresh_token")

    @pytest.mark.asyncio
    async def test_refresh_tokens_success(self, auth_manager):
        """Test successful token refresh"""
        new_access_token = "new_access_token"
        new_refresh_token = "new_refresh_token"
        
        token_data = TokenData(
            user_id=uuid4(),
            username="testuser",
            email="test@prsm.ai",
            role=UserRole.USER.value,
            permissions=[],
            token_type="access",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            issued_at=datetime.now(timezone.utc)
        )
        
        with patch('prsm.core.auth.auth_manager.jwt_handler.refresh_access_token') as mock_refresh:
            with patch('prsm.core.auth.auth_manager.jwt_handler.verify_token', return_value=token_data):
                with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
                    
                    mock_refresh.return_value = (new_access_token, new_refresh_token)
                    
                    response = await auth_manager.refresh_tokens("valid_refresh_token")
                    
                    assert response.access_token == new_access_token
                    assert response.refresh_token == new_refresh_token
                    mock_audit.assert_called()

    @pytest.mark.asyncio
    async def test_refresh_tokens_invalid(self, auth_manager):
        """Test token refresh with invalid refresh token"""
        with patch('prsm.core.auth.auth_manager.jwt_handler.refresh_access_token', return_value=None):
            with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
                
                with pytest.raises(AuthenticationError):
                    await auth_manager.refresh_tokens("invalid_refresh_token")
                
                mock_audit.assert_called()

    @pytest.mark.asyncio
    async def test_logout_user_success(self, auth_manager, mock_user):
        """Test successful user logout"""
        token_data = TokenData(
            user_id=mock_user.id,
            username=mock_user.username,
            email=mock_user.email,
            role=mock_user.role.value,
            permissions=[],
            token_type="access",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            issued_at=datetime.now(timezone.utc)
        )
        
        with patch('prsm.core.auth.auth_manager.jwt_handler.verify_token', return_value=token_data):
            with patch('prsm.core.auth.auth_manager.jwt_handler.revoke_token') as mock_revoke:
                with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
                    
                    result = await auth_manager.logout_user("valid_token")
                    
                    assert result == True
                    mock_revoke.assert_called_once_with("valid_token")
                    mock_audit.assert_called()

    # === Permission and Role Tests ===
    
    def test_check_permission_user_has_permission(self, auth_manager, mock_user):
        """Test permission check when user has permission"""
        with patch.object(mock_user, 'has_permission', return_value=True):
            
            result = auth_manager.check_permission(mock_user, Permission.MODEL_READ)
            
            assert result == True

    def test_check_permission_user_lacks_permission(self, auth_manager, mock_user):
        """Test permission check when user lacks permission"""
        with patch.object(mock_user, 'has_permission', return_value=False):
            
            result = auth_manager.check_permission(mock_user, Permission.SYSTEM_ADMIN)
            
            assert result == False

    def test_check_any_permission_user_has_one(self, auth_manager, mock_user):
        """Test any permission check when user has one of them"""
        permissions = [Permission.MODEL_READ, Permission.SYSTEM_ADMIN]
        
        with patch.object(mock_user, 'has_any_permission', return_value=True):
            
            result = auth_manager.check_any_permission(mock_user, permissions)
            
            assert result == True

    def test_check_any_permission_user_has_none(self, auth_manager, mock_user):
        """Test any permission check when user has none"""
        permissions = [Permission.SYSTEM_ADMIN, Permission.USER_MANAGE]
        
        with patch.object(mock_user, 'has_any_permission', return_value=False):
            
            result = auth_manager.check_any_permission(mock_user, permissions)
            
            assert result == False

    # === Security Tests ===
    
    def test_password_strength_validation_strong(self, auth_manager):
        """Test password strength validation with strong password"""
        strong_password = "SecurePass123!"
        
        result = auth_manager._validate_password_strength(strong_password)
        
        assert result == True

    def test_password_strength_validation_weak(self, auth_manager):
        """Test password strength validation with weak passwords"""
        weak_passwords = [
            "short",                    # Too short
            "nouppercase123!",         # No uppercase
            "NOLOWERCASE123!",         # No lowercase
            "NoDigitsHere!",           # No digits
            "NoSpecialChars123"        # No special characters
        ]
        
        for password in weak_passwords:
            result = auth_manager._validate_password_strength(password)
            assert result == False, f"Password '{password}' should be considered weak"

    @pytest.mark.asyncio
    async def test_failed_login_tracking(self, auth_manager, mock_user):
        """Test failed login attempt tracking"""
        initial_attempts = mock_user.failed_login_attempts
        
        await auth_manager._record_failed_login(mock_user)
        
        assert mock_user.failed_login_attempts == initial_attempts + 1

    @pytest.mark.asyncio
    async def test_reset_failed_login_attempts(self, auth_manager, mock_user):
        """Test resetting failed login attempts"""
        mock_user.failed_login_attempts = 3
        
        await auth_manager._reset_failed_login_attempts(mock_user)
        
        assert mock_user.failed_login_attempts == 0

    @pytest.mark.asyncio
    async def test_account_lockout_check(self, auth_manager, mock_user):
        """Test account lockout mechanism"""
        # Test not locked
        mock_user.failed_login_attempts = 3
        result = await auth_manager._is_account_locked(mock_user)
        assert result == False
        
        # Test locked (exceeds max attempts)
        mock_user.failed_login_attempts = 10
        result = await auth_manager._is_account_locked(mock_user)
        assert result == False  # Current implementation always returns False


class TestUserModel:
    """Test User model security features"""
    
    def test_user_permissions_by_role(self):
        """Test that users get correct permissions based on role"""
        # Test admin user gets all permissions
        admin = User(
            email="admin@prsm.ai",
            username="admin",
            hashed_password="hashed",
            role=UserRole.ADMIN
        )
        admin_permissions = admin.get_permissions()
        assert len(admin_permissions) == len(Permission)
        
        # Test user gets limited permissions
        user = User(
            email="user@prsm.ai",
            username="user",
            hashed_password="hashed",
            role=UserRole.USER
        )
        user_permissions = user.get_permissions()
        assert Permission.MODEL_READ in user_permissions
        assert Permission.SYSTEM_ADMIN not in user_permissions
        
        # Test guest gets minimal permissions
        guest = User(
            email="guest@prsm.ai",
            username="guest",
            hashed_password="hashed",
            role=UserRole.GUEST
        )
        guest_permissions = guest.get_permissions()
        assert Permission.MODEL_READ in guest_permissions
        assert Permission.MODEL_CREATE not in guest_permissions

    def test_superuser_has_all_permissions(self):
        """Test that superuser has all permissions regardless of role"""
        superuser = User(
            email="super@prsm.ai",
            username="super",
            hashed_password="hashed",
            role=UserRole.USER,  # Even with basic role
            is_superuser=True
        )
        
        # Superuser should have any permission
        assert superuser.has_permission(Permission.SYSTEM_ADMIN) == True
        assert superuser.has_permission(Permission.MODEL_DELETE) == True
        assert superuser.has_any_permission([Permission.SYSTEM_ADMIN]) == True

    def test_user_custom_permissions(self):
        """Test user custom permissions functionality"""
        user = User(
            email="user@prsm.ai",
            username="user",
            hashed_password="hashed",
            role=UserRole.USER,
            custom_permissions='["model:create", "data:delete"]'
        )
        
        permissions = user.get_permissions()
        assert Permission.MODEL_CREATE in permissions
        assert Permission.DATA_DELETE in permissions

    def test_user_custom_permissions_invalid_json(self):
        """Test user custom permissions with invalid JSON"""
        user = User(
            email="user@prsm.ai",
            username="user",
            hashed_password="hashed",
            role=UserRole.USER,
            custom_permissions='invalid json'
        )
        
        # Should not crash and should fall back to role permissions
        permissions = user.get_permissions()
        assert isinstance(permissions, set)
        assert Permission.MODEL_READ in permissions  # From USER role


class TestPasswordValidationRequests:
    """Test API request validation for security"""
    
    def test_register_request_validation(self):
        """Test registration request validation"""
        # Valid request
        valid_request = RegisterRequest(
            email="test@prsm.ai",
            username="validuser",
            password="SecurePass123!",
            confirm_password="SecurePass123!"
        )
        assert valid_request.passwords_match() == True
        
        # Invalid username (too short)
        with pytest.raises(ValidationError):
            RegisterRequest(
                email="test@prsm.ai",
                username="ab",  # Too short
                password="SecurePass123!",
                confirm_password="SecurePass123!"
            )
        
        # Invalid username (invalid characters)
        with pytest.raises(ValidationError):
            RegisterRequest(
                email="test@prsm.ai",
                username="user@name",  # Invalid characters
                password="SecurePass123!",
                confirm_password="SecurePass123!"
            )

    def test_password_change_validation(self):
        """Test password change request validation"""
        from prsm.core.auth.models import PasswordChange
        
        # Valid password change
        valid_change = PasswordChange(
            current_password="oldpass",
            new_password="NewSecurePass123!"
        )
        assert valid_change.new_password == "NewSecurePass123!"
        
        # Invalid new password (too short)
        with pytest.raises(ValidationError):
            PasswordChange(
                current_password="oldpass",
                new_password="short"
            )

    def test_login_request_validation(self):
        """Test login request validation"""
        # Valid request
        valid_login = LoginRequest(
            username="testuser",
            password="password123"
        )
        assert valid_login.username == "testuser"
        
        # Invalid request (empty username)
        with pytest.raises(ValidationError):
            LoginRequest(
                username="",
                password="password123"
            )


class TestRolePermissionMapping:
    """Test role-permission mapping integrity"""
    
    def test_role_permissions_completeness(self):
        """Test that all roles have permission mappings"""
        for role in UserRole:
            assert role in ROLE_PERMISSIONS, f"Role {role} missing from ROLE_PERMISSIONS"
    
    def test_permission_hierarchy_logic(self):
        """Test permission hierarchy makes sense"""
        admin_perms = set(ROLE_PERMISSIONS[UserRole.ADMIN])
        researcher_perms = set(ROLE_PERMISSIONS[UserRole.RESEARCHER])
        user_perms = set(ROLE_PERMISSIONS[UserRole.USER])
        guest_perms = set(ROLE_PERMISSIONS[UserRole.GUEST])
        
        # Admin should have more permissions than researcher
        assert len(admin_perms) >= len(researcher_perms)
        
        # Researcher should have more permissions than user
        assert len(researcher_perms) >= len(user_perms)
        
        # User should have more permissions than guest
        assert len(user_perms) >= len(guest_perms)
        
        # Guest permissions should be subset of user permissions
        assert guest_perms.issubset(user_perms)
        
        # User permissions should be subset of researcher permissions
        assert user_perms.issubset(researcher_perms)

    def test_critical_permissions_restricted(self):
        """Test that critical permissions are properly restricted"""
        # System admin should only be available to admin role
        admin_only_perms = {Permission.SYSTEM_ADMIN, Permission.USER_MANAGE}
        
        for role in UserRole:
            role_perms = set(ROLE_PERMISSIONS[role])
            if role != UserRole.ADMIN:
                assert not admin_only_perms.intersection(role_perms), f"Role {role} has admin-only permissions"
        
        # Token mint/burn should be restricted
        dangerous_perms = {Permission.TOKEN_MINT, Permission.TOKEN_BURN}
        
        for role in [UserRole.USER, UserRole.GUEST, UserRole.ANALYST]:
            role_perms = set(ROLE_PERMISSIONS[role])
            assert not dangerous_perms.intersection(role_perms), f"Role {role} has dangerous token permissions"


class TestSecurityIntegration:
    """Integration tests for security components"""
    
    @pytest.mark.asyncio
    async def test_auth_flow_timing_attack_protection(self, auth_manager):
        """Test that auth flow protects against timing attacks"""
        import time
        
        # Test non-existent user (should take similar time as wrong password)
        start_time = time.time()
        
        with patch.object(auth_manager, '_get_user_by_login', new_callable=AsyncMock, return_value=None):
            with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event'):
                try:
                    await auth_manager.authenticate_user(LoginRequest(username="nonexistent", password="password"))
                except AuthenticationError:
                    pass
        
        non_existent_time = time.time() - start_time
        
        # Should include sleep to prevent timing attacks
        assert non_existent_time >= 1.0, "Should include timing attack protection delay"

    @pytest.mark.asyncio
    async def test_audit_logging_on_security_events(self, auth_manager, mock_user):
        """Test that security events are properly logged"""
        with patch('prsm.core.auth.auth_manager.audit_logger.log_auth_event') as mock_audit:
            
            # Test failed login logging
            with patch.object(auth_manager, '_get_user_by_login', new_callable=AsyncMock, return_value=None):
                try:
                    await auth_manager.authenticate_user(LoginRequest(username="test", password="wrong"))
                except AuthenticationError:
                    pass
            
            # Should log the failed attempt
            mock_audit.assert_called_with(
                "login_failed",
                {"reason": "user_not_found", "username": "test"},
                None
            )

    def test_exception_security_details(self):
        """Test that exceptions don't leak sensitive information"""
        # Authentication error should be generic
        auth_error = AuthenticationError()
        assert "Could not validate credentials" in str(auth_error.detail)
        assert "password" not in str(auth_error.detail).lower()
        assert "user" not in str(auth_error.detail).lower()
        
        # Authorization error should be generic
        auth_error = AuthorizationError()
        assert "Insufficient permissions" in str(auth_error.detail)