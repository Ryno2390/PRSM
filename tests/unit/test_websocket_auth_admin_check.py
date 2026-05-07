"""websocket_auth.py admin-check import correctness.

Regression for an earlier broken-import bug at line 441:
    from prsm.core.auth.enhanced_authorization import get_enhanced_auth_manager
The module is actually `prsm.core.security.enhanced_authorization`
(see credential_api.py:19 + middleware.py:22). The import was wrapped
in try/except, so it silent-failed to "no admin permission" — which
made admin-only conversation access dead under WebSocket. Same shape
as the dependencies.py:102 bug fixed earlier today.

Pins:
- The corrected import path resolves at runtime
- The source contains no lingering reference to the wrong submodule
"""
from __future__ import annotations

import importlib
import inspect


def test_get_enhanced_auth_manager_resolves_from_security_path():
    """The canonical home of get_enhanced_auth_manager is
    prsm.core.security.enhanced_authorization. credential_api.py and
    middleware.py have always imported from there; websocket_auth.py
    used to import from the wrong path."""
    mod = importlib.import_module(
        "prsm.core.security.enhanced_authorization",
    )
    assert callable(getattr(mod, "get_enhanced_auth_manager"))


def test_websocket_auth_does_not_reference_wrong_path():
    """Defense-in-depth: scan the source for the bad import string.
    Catches accidental re-introduction on future refactors that move
    files around in prsm.core.*."""
    import prsm.interface.api.websocket_auth as ws
    src = inspect.getsource(ws)
    assert "prsm.core.auth.enhanced_authorization" not in src
    assert "prsm.core.security.enhanced_authorization" in src
