"""Sprint 185 — LoginRequest field-length bounds.

Pre-fix the /api/auth/login dashboard endpoint accepted arbitrarily
large password values. A 10MB password returned 200 with the
demo-stub token — wasted memory, and the production auth_manager
path would have crashed (downstream hash functions don't handle
multi-MB input gracefully).

Probed live 2026-05-11:
  curl -d '{"username":"x","password":"<10MB>"}' /api/auth/login
  → 200 with demo-token

Post-fix: Pydantic Field(max_length=...) rejects above-limit
input with the standard 422 response before reaching any handler.
Bounds:
  username: max_length=256
  password: max_length=1024
Both also gain min_length=1 so empty values are rejected (was
silently accepted).
"""
from __future__ import annotations

import pytest


def test_login_request_rejects_long_username():
    from prsm.dashboard.app import LoginRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        LoginRequest(username="x" * 257, password="ok")
    msg = str(exc_info.value).lower()
    assert "username" in msg
    assert "string_too_long" in msg or "at most 256" in msg


def test_login_request_rejects_long_password():
    from prsm.dashboard.app import LoginRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        LoginRequest(username="ok", password="p" * 1025)
    msg = str(exc_info.value).lower()
    assert "password" in msg
    assert "string_too_long" in msg or "at most 1024" in msg


def test_login_request_rejects_empty_username():
    from prsm.dashboard.app import LoginRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LoginRequest(username="", password="ok")


def test_login_request_rejects_empty_password():
    from prsm.dashboard.app import LoginRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LoginRequest(username="ok", password="")


def test_login_request_accepts_boundary_values():
    """Sprint 185 — values at exactly the boundary still accepted."""
    from prsm.dashboard.app import LoginRequest

    req = LoginRequest(username="x" * 256, password="p" * 1024)
    assert len(req.username) == 256
    assert len(req.password) == 1024


def test_login_request_accepts_typical_values():
    """Typical usage continues to work — regression-pin."""
    from prsm.dashboard.app import LoginRequest

    req = LoginRequest(username="alice", password="secret-passphrase-123")
    assert req.username == "alice"
