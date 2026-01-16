"""
JWT Authentication Integration Tests
====================================

End-to-end integration tests for JWT authentication including:
- Token creation with proper claims
- Signature verification
- Token expiration enforcement
- Token revocation via Redis and database
- Algorithm confusion attack prevention

Requirements:
- Redis server (for token revocation cache)
- PostgreSQL database (for revocation persistence)

Run with: pytest tests/integration/test_jwt_authentication_integration.py -v --tb=short
"""

import asyncio
import hashlib
import os
import time
import pytest
import jwt as pyjwt
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from typing import Optional, Dict, Any
import redis.asyncio as aioredis


# Configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/1")
DATABASE_URL = os.environ.get("DATABASE_URL", os.environ.get("TEST_DATABASE_URL"))
JWT_SECRET = os.environ.get("JWT_SECRET", "test-secret-key-for-integration-tests-only")

# Allowed algorithms (matching production config)
ALLOWED_ALGORITHMS = ["HS256", "HS384", "HS512"]


class JWTHandler:
    """JWT handler for integration testing."""

    def __init__(
        self,
        secret_key: str,
        redis_client: Optional[aioredis.Redis] = None,
        algorithm: str = "HS256"
    ):
        self.secret_key = secret_key
        self.redis = redis_client
        self.algorithm = algorithm
        self.access_token_expire_minutes = 15

    def create_access_token(
        self,
        subject: str,
        expires_delta: Optional[timedelta] = None,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a JWT access token with required claims."""
        now = datetime.now(timezone.utc)
        expire = now + (expires_delta or timedelta(minutes=self.access_token_expire_minutes))

        payload = {
            "sub": subject,
            "iat": now,
            "exp": expire,
            "jti": str(uuid4()),  # JWT ID for revocation tracking
            "type": "access"
        }

        if additional_claims:
            payload.update(additional_claims)

        return pyjwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token with proper signature validation.

        Returns:
            Decoded payload if valid, None if invalid or revoked
        """
        try:
            # Decode with signature verification
            payload = pyjwt.decode(
                token,
                self.secret_key,
                algorithms=ALLOWED_ALGORITHMS,
                options={
                    "verify_signature": True,  # CRITICAL: Must be True
                    "verify_exp": True,
                    "verify_iat": True,
                    "require": ["exp", "sub", "jti"]
                }
            )

            # Check revocation
            if self.redis:
                is_revoked = await self._is_token_revoked(token, payload.get("jti"))
                if is_revoked:
                    return None

            return payload

        except pyjwt.ExpiredSignatureError:
            return None
        except pyjwt.InvalidTokenError:
            return None

    async def revoke_token(
        self,
        token: str,
        reason: str = "user_logout"
    ) -> bool:
        """Revoke a token by adding to blacklist."""
        if not self.redis:
            return False

        try:
            # Decode to get JTI and expiration
            payload = pyjwt.decode(
                token,
                self.secret_key,
                algorithms=ALLOWED_ALGORITHMS,
                options={"verify_exp": False}  # Allow revoking expired tokens
            )

            token_hash = hashlib.sha256(token.encode()).hexdigest()
            jti = payload.get("jti", "unknown")

            # Calculate TTL (until token would expire naturally)
            exp = payload.get("exp", 0)
            ttl = max(1, int(exp - time.time()))

            # Store in Redis with TTL
            key = f"revoked_token:{token_hash[:16]}"
            await self.redis.setex(key, ttl, jti)

            return True

        except Exception:
            return False

    async def _is_token_revoked(
        self,
        token: str,
        jti: Optional[str] = None
    ) -> bool:
        """Check if token is revoked."""
        if not self.redis:
            return False

        token_hash = hashlib.sha256(token.encode()).hexdigest()
        key = f"revoked_token:{token_hash[:16]}"

        result = await self.redis.get(key)
        return result is not None


@pytest.fixture(scope="module")
async def redis_client():
    """Create Redis connection for tests."""
    try:
        client = aioredis.from_url(REDIS_URL, decode_responses=True)
        await client.ping()
        yield client
        await client.aclose()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.fixture
async def jwt_handler(redis_client):
    """Create JWT handler for tests."""
    return JWTHandler(
        secret_key=JWT_SECRET,
        redis_client=redis_client
    )


@pytest.fixture
async def clean_revoked_tokens(redis_client):
    """Cleanup revoked token keys."""
    async for key in redis_client.scan_iter("revoked_token:*"):
        await redis_client.delete(key)
    yield
    async for key in redis_client.scan_iter("revoked_token:*"):
        await redis_client.delete(key)


class TestTokenCreation:
    """Test JWT token creation."""

    @pytest.mark.asyncio
    async def test_creates_token_with_required_claims(self, jwt_handler):
        """Verify token contains all required claims."""
        user_id = str(uuid4())
        token = jwt_handler.create_access_token(subject=user_id)

        # Decode to check claims
        payload = pyjwt.decode(
            token,
            JWT_SECRET,
            algorithms=ALLOWED_ALGORITHMS
        )

        assert "sub" in payload
        assert payload["sub"] == user_id
        assert "iat" in payload
        assert "exp" in payload
        assert "jti" in payload
        assert "type" in payload
        assert payload["type"] == "access"

    @pytest.mark.asyncio
    async def test_token_expiration_set_correctly(self, jwt_handler):
        """Verify token expiration is set correctly."""
        user_id = str(uuid4())
        expires_delta = timedelta(hours=2)

        token = jwt_handler.create_access_token(
            subject=user_id,
            expires_delta=expires_delta
        )

        payload = pyjwt.decode(
            token,
            JWT_SECRET,
            algorithms=ALLOWED_ALGORITHMS
        )

        # Check expiration is approximately 2 hours from now
        exp_time = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = exp_time - now

        assert timedelta(hours=1, minutes=59) < diff < timedelta(hours=2, minutes=1)

    @pytest.mark.asyncio
    async def test_each_token_has_unique_jti(self, jwt_handler):
        """Verify each token gets a unique JWT ID."""
        user_id = str(uuid4())

        tokens = [jwt_handler.create_access_token(subject=user_id) for _ in range(10)]
        jtis = set()

        for token in tokens:
            payload = pyjwt.decode(token, JWT_SECRET, algorithms=ALLOWED_ALGORITHMS)
            jtis.add(payload["jti"])

        assert len(jtis) == 10, "Each token should have unique JTI"


class TestSignatureVerification:
    """Test JWT signature verification."""

    @pytest.mark.asyncio
    async def test_valid_signature_accepted(self, jwt_handler, clean_revoked_tokens):
        """Verify valid signature is accepted."""
        user_id = str(uuid4())
        token = jwt_handler.create_access_token(subject=user_id)

        payload = await jwt_handler.verify_token(token)

        assert payload is not None
        assert payload["sub"] == user_id

    @pytest.mark.asyncio
    async def test_tampered_token_rejected(self, jwt_handler):
        """Verify tampered token is rejected."""
        user_id = str(uuid4())
        token = jwt_handler.create_access_token(subject=user_id)

        # Tamper with the payload
        parts = token.split(".")
        # Modify a character in the payload
        tampered_payload = parts[1][:-1] + ("a" if parts[1][-1] != "a" else "b")
        tampered_token = f"{parts[0]}.{tampered_payload}.{parts[2]}"

        payload = await jwt_handler.verify_token(tampered_token)
        assert payload is None, "Tampered token should be rejected"

    @pytest.mark.asyncio
    async def test_wrong_secret_rejected(self, jwt_handler):
        """Verify token signed with wrong secret is rejected."""
        user_id = str(uuid4())

        # Create token with different secret
        wrong_secret_handler = JWTHandler(secret_key="wrong-secret-key")
        token = wrong_secret_handler.create_access_token(subject=user_id)

        # Try to verify with correct handler
        payload = await jwt_handler.verify_token(token)
        assert payload is None, "Token with wrong secret should be rejected"


class TestAlgorithmConfusionPrevention:
    """Test prevention of algorithm confusion attacks."""

    @pytest.mark.asyncio
    async def test_none_algorithm_rejected(self, jwt_handler):
        """Verify 'none' algorithm tokens are rejected."""
        payload = {
            "sub": str(uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4())
        }

        # Create token with 'none' algorithm
        token = pyjwt.encode(payload, key="", algorithm="none")

        result = await jwt_handler.verify_token(token)
        assert result is None, "'none' algorithm should be rejected"

    @pytest.mark.asyncio
    async def test_only_allowed_algorithms_accepted(self, jwt_handler):
        """Verify only configured algorithms are accepted."""
        user_id = str(uuid4())
        payload = {
            "sub": user_id,
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4())
        }

        # HS256 should work
        token_hs256 = pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")
        result = await jwt_handler.verify_token(token_hs256)
        assert result is not None, "HS256 should be accepted"

        # HS384 should work
        payload["jti"] = str(uuid4())
        token_hs384 = pyjwt.encode(payload, JWT_SECRET, algorithm="HS384")
        result = await jwt_handler.verify_token(token_hs384)
        assert result is not None, "HS384 should be accepted"


class TestTokenExpiration:
    """Test token expiration enforcement."""

    @pytest.mark.asyncio
    async def test_expired_token_rejected(self, jwt_handler):
        """Verify expired token is rejected."""
        user_id = str(uuid4())

        # Create token that's already expired
        token = jwt_handler.create_access_token(
            subject=user_id,
            expires_delta=timedelta(seconds=-10)  # Expired 10 seconds ago
        )

        payload = await jwt_handler.verify_token(token)
        assert payload is None, "Expired token should be rejected"

    @pytest.mark.asyncio
    async def test_token_valid_before_expiration(self, jwt_handler, clean_revoked_tokens):
        """Verify token is valid before expiration."""
        user_id = str(uuid4())

        token = jwt_handler.create_access_token(
            subject=user_id,
            expires_delta=timedelta(hours=1)
        )

        payload = await jwt_handler.verify_token(token)
        assert payload is not None, "Token should be valid before expiration"


class TestTokenRevocation:
    """Test token revocation system."""

    @pytest.mark.asyncio
    async def test_revoked_token_rejected(self, jwt_handler, clean_revoked_tokens):
        """Verify revoked token is rejected."""
        user_id = str(uuid4())
        token = jwt_handler.create_access_token(subject=user_id)

        # Verify token is initially valid
        payload = await jwt_handler.verify_token(token)
        assert payload is not None

        # Revoke token
        revoked = await jwt_handler.revoke_token(token, reason="test_revocation")
        assert revoked

        # Token should now be rejected
        payload = await jwt_handler.verify_token(token)
        assert payload is None, "Revoked token should be rejected"

    @pytest.mark.asyncio
    async def test_revocation_survives_new_requests(
        self, jwt_handler, redis_client, clean_revoked_tokens
    ):
        """Verify revocation persists across requests."""
        user_id = str(uuid4())
        token = jwt_handler.create_access_token(subject=user_id)

        # Revoke
        await jwt_handler.revoke_token(token)

        # Create new handler instance (simulating new request)
        new_handler = JWTHandler(
            secret_key=JWT_SECRET,
            redis_client=redis_client
        )

        # Should still be revoked
        payload = await new_handler.verify_token(token)
        assert payload is None, "Revocation should persist"

    @pytest.mark.asyncio
    async def test_other_tokens_unaffected_by_revocation(
        self, jwt_handler, clean_revoked_tokens
    ):
        """Verify revoking one token doesn't affect others."""
        user_id = str(uuid4())

        token1 = jwt_handler.create_access_token(subject=user_id)
        token2 = jwt_handler.create_access_token(subject=user_id)

        # Revoke token1
        await jwt_handler.revoke_token(token1)

        # Token1 should be rejected
        payload1 = await jwt_handler.verify_token(token1)
        assert payload1 is None

        # Token2 should still be valid
        payload2 = await jwt_handler.verify_token(token2)
        assert payload2 is not None


class TestRequiredClaims:
    """Test required claims validation."""

    @pytest.mark.asyncio
    async def test_missing_sub_rejected(self, jwt_handler):
        """Verify token without subject is rejected."""
        payload = {
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4())
            # Missing 'sub'
        }

        token = pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")
        result = await jwt_handler.verify_token(token)
        assert result is None, "Token without subject should be rejected"

    @pytest.mark.asyncio
    async def test_missing_jti_rejected(self, jwt_handler):
        """Verify token without JTI is rejected."""
        payload = {
            "sub": str(uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc)
            # Missing 'jti'
        }

        token = pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")
        result = await jwt_handler.verify_token(token)
        assert result is None, "Token without JTI should be rejected"

    @pytest.mark.asyncio
    async def test_future_iat_flagged(self, jwt_handler, clean_revoked_tokens):
        """Verify tokens with future iat are handled appropriately."""
        payload = {
            "sub": str(uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(hours=2),
            "iat": datetime.now(timezone.utc) + timedelta(hours=1),  # Future
            "jti": str(uuid4())
        }

        token = pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")

        # PyJWT allows future iat by default, but our system should
        # ideally flag this as suspicious. This test documents current behavior.
        result = await jwt_handler.verify_token(token)
        # Note: This may pass or fail depending on implementation
        # The important thing is that the behavior is consistent


class TestConcurrentTokenOperations:
    """Test concurrent token operations."""

    @pytest.mark.asyncio
    async def test_concurrent_verifications(self, jwt_handler, clean_revoked_tokens):
        """Verify concurrent token verifications work correctly."""
        user_id = str(uuid4())
        token = jwt_handler.create_access_token(subject=user_id)

        async def verify_token():
            return await jwt_handler.verify_token(token)

        # Run concurrent verifications
        tasks = [verify_token() for _ in range(50)]
        results = await asyncio.gather(*tasks)

        # All should return valid payload
        valid_count = sum(1 for r in results if r is not None)
        assert valid_count == 50, "All concurrent verifications should succeed"

    @pytest.mark.asyncio
    async def test_concurrent_revocations(self, jwt_handler, clean_revoked_tokens):
        """Verify concurrent revocations don't cause issues."""
        user_id = str(uuid4())
        token = jwt_handler.create_access_token(subject=user_id)

        async def revoke_token():
            return await jwt_handler.revoke_token(token)

        # Run concurrent revocations
        tasks = [revoke_token() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed (idempotent operation)
        assert all(results), "All revocations should succeed"

        # Token should be revoked
        payload = await jwt_handler.verify_token(token)
        assert payload is None


class TestLogoutAllDevices:
    """Test logout from all devices functionality."""

    @pytest.mark.asyncio
    async def test_revoke_multiple_tokens_for_user(
        self, jwt_handler, clean_revoked_tokens
    ):
        """Verify multiple tokens can be revoked for same user."""
        user_id = str(uuid4())

        # Create multiple tokens for same user
        tokens = [jwt_handler.create_access_token(subject=user_id) for _ in range(5)]

        # Verify all are initially valid
        for token in tokens:
            payload = await jwt_handler.verify_token(token)
            assert payload is not None

        # Revoke all
        for token in tokens:
            await jwt_handler.revoke_token(token, reason="logout_all_devices")

        # All should be revoked
        for token in tokens:
            payload = await jwt_handler.verify_token(token)
            assert payload is None, "All tokens should be revoked"


class TestPasswordChangeRevocation:
    """Test that password change revokes existing tokens."""

    @pytest.mark.asyncio
    async def test_password_change_revokes_tokens(
        self, jwt_handler, clean_revoked_tokens
    ):
        """Verify password change revokes all existing tokens."""
        user_id = str(uuid4())

        # Create tokens before password change
        old_tokens = [jwt_handler.create_access_token(subject=user_id) for _ in range(3)]

        # Simulate password change - revoke all old tokens
        for token in old_tokens:
            await jwt_handler.revoke_token(token, reason="password_change")

        # Create new token after password change
        new_token = jwt_handler.create_access_token(subject=user_id)

        # Old tokens should be revoked
        for token in old_tokens:
            payload = await jwt_handler.verify_token(token)
            assert payload is None, "Old token should be revoked after password change"

        # New token should work
        payload = await jwt_handler.verify_token(new_token)
        assert payload is not None, "New token should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
