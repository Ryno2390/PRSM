"""dependencies.py — auth-flow regression tests.

Pins:
- get_current_user() returns a canonical ``prsm.core.auth.models.User``
  with the documented anonymous-node fields. (Previous bug: imported
  from the deleted ``prsm.user_content_manager`` module → ImportError
  on every call.)
- get_optional_user() returns None on missing credentials, the same
  anonymous-node user when credentials validate, and None when
  validation raises. (Previous bug: called get_current_user(user_id, db)
  with two args the function does not accept → TypeError.)
- require_admin / require_enterprise accept the anonymous user without
  raise (it's marked is_superuser + role=ADMIN).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.core.auth.models import User, UserRole
from prsm.interface.api import dependencies as deps


@pytest.mark.asyncio
async def test_get_current_user_returns_canonical_user():
    user = await deps.get_current_user()
    assert isinstance(user, User)
    assert user.username == "anonymous-node"
    assert user.is_active is True
    assert user.is_superuser is True
    assert user.role == UserRole.ADMIN


@pytest.mark.asyncio
async def test_get_current_user_permissions_resolvable():
    """The User must be a fully-functional auth model — its
    ``get_permissions()`` method must return the ADMIN role's perms."""
    user = await deps.get_current_user()
    perms = user.get_permissions()
    # ADMIN should have all permissions
    assert len(perms) > 0


@pytest.mark.asyncio
async def test_get_optional_user_no_credentials_returns_none():
    user = await deps.get_optional_user(credentials=None, db=MagicMock())
    assert user is None


@pytest.mark.asyncio
async def test_get_optional_user_empty_credentials_returns_none():
    creds = MagicMock()
    creds.credentials = ""
    user = await deps.get_optional_user(credentials=creds, db=MagicMock())
    assert user is None


@pytest.mark.asyncio
async def test_get_optional_user_valid_credentials_returns_user():
    """Valid creds → verify_api_key passes → return anonymous-node user."""
    creds = MagicMock()
    creds.credentials = "any-jwt"
    with patch.object(deps, "verify_api_key", new=AsyncMock(return_value="uid")):
        user = await deps.get_optional_user(
            credentials=creds, db=MagicMock(),
        )
    assert isinstance(user, User)
    assert user.username == "anonymous-node"


@pytest.mark.asyncio
async def test_get_optional_user_verify_raises_returns_none():
    """verify_api_key raising must yield None, not propagate."""
    creds = MagicMock()
    creds.credentials = "bad-jwt"
    with patch.object(
        deps, "verify_api_key",
        new=AsyncMock(side_effect=RuntimeError("invalid")),
    ):
        user = await deps.get_optional_user(
            credentials=creds, db=MagicMock(),
        )
    assert user is None


@pytest.mark.asyncio
async def test_require_admin_accepts_anonymous_user():
    """The anonymous-node sentinel is is_superuser=True, so admin gates
    must not block it."""
    user = await deps.get_current_user()
    result = await deps.require_admin(user)
    assert result is user


@pytest.mark.asyncio
async def test_require_enterprise_accepts_anonymous_user():
    user = await deps.get_current_user()
    result = await deps.require_enterprise(user)
    assert result is user


def test_no_lingering_user_content_manager_import():
    """Defense-in-depth: the deleted module must not be referenced."""
    import inspect
    src = inspect.getsource(deps)
    assert "user_content_manager" not in src
