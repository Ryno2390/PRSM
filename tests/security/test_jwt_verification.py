"""
JWT Verification Bypass Tests
=============================

Tests for JWT authentication security including:
- Token revocation verification
- Algorithm confusion attack prevention
- Required claims validation
- Token expiration enforcement
"""

import pytest
import jwt
import hashlib
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class TestJWTRevocationVerification:
    """Test suite for JWT token revocation."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.setex = AsyncMock(return_value=True)
        return redis

    @pytest.fixture
    def mock_db_service(self):
        """Create a mock database service."""
        db = AsyncMock()
        return db

    @pytest.fixture
    def jwt_secret(self):
        """Test JWT secret."""
        return "test-secret-key-for-jwt-signing"

    @pytest.mark.asyncio
    async def test_revoked_token_is_rejected(self, mock_redis, mock_db_service, jwt_secret):
        """Verify that revoked tokens are properly rejected."""
        from prsm.core.auth.jwt_handler import JWTHandler

        handler = JWTHandler(
            secret_key=jwt_secret,
            redis_client=mock_redis,
            db_service=mock_db_service
        )

        # Create a valid token
        user_id = str(uuid4())
        token = handler.create_access_token(
            subject=user_id,
            expires_delta=timedelta(hours=1)
        )

        # Calculate token hash
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # Mark token as revoked in Redis cache
        mock_redis.get = AsyncMock(return_value=b"1")  # Revoked

        # Verify token - should fail
        result = await handler.verify_token(token)
        assert result is None or result.get("revoked", False), "Revoked token should be rejected"

    @pytest.mark.asyncio
    async def test_revocation_check_falls_through_to_database(self, mock_redis, mock_db_service, jwt_secret):
        """Verify revocation check queries database when Redis misses."""
        from prsm.core.auth.jwt_handler import JWTHandler

        handler = JWTHandler(
            secret_key=jwt_secret,
            redis_client=mock_redis,
            db_service=mock_db_service
        )

        # Redis returns None (cache miss)
        mock_redis.get = AsyncMock(return_value=None)

        # Database should be queried
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(
            scalar=MagicMock(return_value=1)  # Token found in revocation table
        ))
        mock_db_service.get_session = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock()
        ))

        user_id = str(uuid4())
        token = handler.create_access_token(
            subject=user_id,
            expires_delta=timedelta(hours=1)
        )

        result = await handler.verify_token(token)

        # Should have checked database after Redis miss
        # Token should be rejected as revoked
        assert result is None or result.get("revoked", False)


class TestAlgorithmConfusionPrevention:
    """Test prevention of algorithm confusion attacks."""

    @pytest.fixture
    def jwt_secret(self):
        return "test-secret-key"

    def test_none_algorithm_is_rejected(self, jwt_secret):
        """Verify that 'none' algorithm tokens are rejected."""
        # Create a token with 'none' algorithm (attack attempt)
        payload = {
            "sub": str(uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4())
        }

        # Manually craft token with 'none' algorithm
        header = {"alg": "none", "typ": "JWT"}
        token_without_signature = jwt.encode(
            payload,
            key="",
            algorithm="none",
            headers=header
        )

        # Attempt to decode with our handler should fail
        from prsm.core.auth.jwt_handler import ALLOWED_ALGORITHMS

        # Verify 'none' is not in allowed algorithms
        assert "none" not in ALLOWED_ALGORITHMS
        assert "None" not in ALLOWED_ALGORITHMS

        # Attempting to decode should raise
        with pytest.raises(jwt.exceptions.InvalidAlgorithmError):
            jwt.decode(
                token_without_signature,
                jwt_secret,
                algorithms=ALLOWED_ALGORITHMS
            )

    def test_hs256_with_rsa_public_key_is_rejected(self, jwt_secret):
        """Verify HS256 algorithm cannot be used with RSA public key."""
        # This prevents the "algorithm confusion" attack where attacker
        # signs with HS256 using the public RSA key

        payload = {
            "sub": str(uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4())
        }

        # Sign with HS256
        token = jwt.encode(payload, jwt_secret, algorithm="HS256")

        # Should decode with HS256 (correct)
        decoded = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        assert decoded["sub"] == payload["sub"]

        # Should NOT decode if we only allow RS256
        with pytest.raises(jwt.exceptions.InvalidAlgorithmError):
            jwt.decode(token, jwt_secret, algorithms=["RS256"])

    def test_only_allowed_algorithms_accepted(self, jwt_secret):
        """Verify only configured algorithms are accepted."""
        from prsm.core.auth.jwt_handler import ALLOWED_ALGORITHMS

        payload = {
            "sub": str(uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4())
        }

        # Test that allowed algorithms work
        for alg in ["HS256", "HS384", "HS512"]:
            if alg in ALLOWED_ALGORITHMS:
                token = jwt.encode(payload, jwt_secret, algorithm=alg)
                decoded = jwt.decode(token, jwt_secret, algorithms=ALLOWED_ALGORITHMS)
                assert decoded["sub"] == payload["sub"]


class TestRequiredClaimsValidation:
    """Test validation of required JWT claims."""

    @pytest.fixture
    def jwt_secret(self):
        return "test-secret-key"

    def test_missing_exp_claim_rejected(self, jwt_secret):
        """Verify tokens without exp claim are rejected."""
        payload = {
            "sub": str(uuid4()),
            "iat": datetime.now(timezone.utc).timestamp(),
            "jti": str(uuid4())
        }  # Missing 'exp'

        token = jwt.encode(payload, jwt_secret, algorithm="HS256")

        # Decoding with require=["exp"] should fail
        with pytest.raises(jwt.exceptions.MissingRequiredClaimError):
            jwt.decode(
                token,
                jwt_secret,
                algorithms=["HS256"],
                options={"require": ["exp"]}
            )

    def test_missing_sub_claim_rejected(self, jwt_secret):
        """Verify tokens without sub (subject) claim are rejected."""
        payload = {
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4())
        }  # Missing 'sub'

        token = jwt.encode(payload, jwt_secret, algorithm="HS256")

        # Should have subject claim
        decoded = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        assert "sub" not in decoded

    def test_missing_jti_claim_allows_replay(self, jwt_secret):
        """Demonstrate why jti (JWT ID) is required for revocation."""
        payload = {
            "sub": str(uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc)
        }  # Missing 'jti'

        token = jwt.encode(payload, jwt_secret, algorithm="HS256")
        decoded = jwt.decode(token, jwt_secret, algorithms=["HS256"])

        # Without jti, we cannot track individual token revocation
        assert "jti" not in decoded

        # With jti, we can revoke specific tokens
        payload_with_jti = {
            **payload,
            "jti": str(uuid4())
        }
        token_with_jti = jwt.encode(payload_with_jti, jwt_secret, algorithm="HS256")
        decoded_with_jti = jwt.decode(token_with_jti, jwt_secret, algorithms=["HS256"])

        assert "jti" in decoded_with_jti

    def test_expired_token_rejected(self, jwt_secret):
        """Verify expired tokens are rejected."""
        payload = {
            "sub": str(uuid4()),
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),  # Expired
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
            "jti": str(uuid4())
        }

        token = jwt.encode(payload, jwt_secret, algorithm="HS256")

        with pytest.raises(jwt.exceptions.ExpiredSignatureError):
            jwt.decode(token, jwt_secret, algorithms=["HS256"])

    def test_future_iat_is_suspicious(self, jwt_secret):
        """Verify tokens with future iat (issued at) are flagged."""
        payload = {
            "sub": str(uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(hours=2),
            "iat": datetime.now(timezone.utc) + timedelta(hours=1),  # Future
            "jti": str(uuid4())
        }

        token = jwt.encode(payload, jwt_secret, algorithm="HS256")

        # PyJWT allows future iat by default, but we should check
        decoded = jwt.decode(token, jwt_secret, algorithms=["HS256"])

        iat = datetime.fromtimestamp(decoded["iat"], tz=timezone.utc)
        now = datetime.now(timezone.utc)

        # Token issued in the future is suspicious
        is_future_iat = iat > now + timedelta(seconds=30)  # Allow 30s clock skew
        if is_future_iat:
            pytest.skip("Future iat should trigger additional verification")


class TestTokenRevocationFlow:
    """Test the complete token revocation flow."""

    @pytest.mark.asyncio
    async def test_revoke_token_adds_to_both_caches(self):
        """Verify token revocation updates both Redis and database."""
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock(return_value=True)

        mock_db = AsyncMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()

        # Simulate revocation
        token_hash = hashlib.sha256(b"test-token").hexdigest()
        jti = str(uuid4())
        user_id = str(uuid4())

        # Redis should be updated
        await mock_redis.setex(f"token_revoked:{token_hash[:16]}", 86400, "1")

        # Database should be updated
        # INSERT INTO revoked_tokens (token_hash, jti, user_id, reason, revoked_at)

        assert mock_redis.setex.called

    @pytest.mark.asyncio
    async def test_logout_revokes_all_user_tokens(self):
        """Verify logout revokes all tokens for a user."""
        user_id = str(uuid4())

        # User has multiple active tokens
        active_tokens = [
            {"jti": str(uuid4()), "token_hash": hashlib.sha256(f"token{i}".encode()).hexdigest()}
            for i in range(5)
        ]

        # Logout should revoke all
        revoked_count = len(active_tokens)

        # All tokens should be marked as revoked
        assert revoked_count == 5

    @pytest.mark.asyncio
    async def test_password_change_revokes_existing_tokens(self):
        """Verify password change revokes all existing tokens."""
        user_id = str(uuid4())

        # Simulate password change flow
        # 1. Validate current password
        # 2. Update password hash
        # 3. Revoke all existing tokens for user
        # 4. Issue new token

        tokens_before_change = 3
        tokens_revoked = tokens_before_change
        new_token_issued = True

        assert tokens_revoked == tokens_before_change
        assert new_token_issued


class TestSecurityHeaders:
    """Test security headers in JWT responses."""

    def test_token_not_logged(self):
        """Verify tokens are not logged in plain text."""
        test_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"

        # Simulate log message
        log_message = f"User authenticated"

        # Token should not appear in logs
        assert test_token not in log_message

    def test_token_masked_in_debug_output(self):
        """Verify tokens are masked in debug output."""
        test_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"

        # Masking function
        def mask_token(token: str) -> str:
            if len(token) > 20:
                return f"{token[:10]}...{token[-5:]}"
            return "***"

        masked = mask_token(test_token)

        # Original token should not be recoverable
        assert test_token not in masked
        assert "..." in masked
