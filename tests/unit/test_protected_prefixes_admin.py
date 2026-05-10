"""PROTECTED_PREFIXES covers /admin/, /wallet/, /content/arbitration/
(sprint 138).

Pre-fix: when an operator set PRSM_NODE_API_KEY (the documented
mechanism for protecting their node), /admin/heartbeat/trigger,
/admin/distribution/trigger, and /wallet/royalty/claim stayed
unprotected. Network-adjacent attackers could spend operator gas.

Post-fix: the prefix list covers all sensitive write endpoints.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from prsm.api.auth_middleware import (
    PROTECTED_PREFIXES, NodeAuthMiddleware, hash_api_key,
)
from starlette.requests import Request


def _is_protected(path: str) -> bool:
    return any(path.startswith(p) for p in PROTECTED_PREFIXES)


class TestAdminProtected:
    def test_heartbeat_trigger_protected(self):
        assert _is_protected("/admin/heartbeat/trigger")

    def test_distribution_trigger_protected(self):
        assert _is_protected("/admin/distribution/trigger")

    def test_admin_history_endpoints_protected(self):
        # /admin/* read endpoints — protected too because
        # operators may not want raw event-log enumeration
        # exposed unauthenticated
        for p in (
            "/admin/webhook-history",
            "/admin/slash-history",
            "/admin/heartbeat-history",
            "/admin/distribution-history",
            "/admin/earnings-summary",
        ):
            assert _is_protected(p), f"{p} should be protected"


class TestWalletProtected:
    def test_royalty_claim_protected(self):
        assert _is_protected("/wallet/royalty/claim")

    def test_offramp_quote_protected(self):
        assert _is_protected("/wallet/offramp/quote")

    def test_balance_endpoints_protected(self):
        # Read-side wallet info is financial-adjacent PII
        assert _is_protected("/wallet/spend")
        assert _is_protected("/wallet/escrows")


class TestArbitrationProtected:
    def test_preview_resolution_protected(self):
        assert _is_protected("/content/arbitration/preview-resolution")


class TestUnchanged:
    def test_public_endpoints_still_pass(self):
        # Sanity: /health, / etc. not in protected list
        for p in ("/", "/health", "/status", "/openapi.json"):
            assert not _is_protected(p), f"{p} should NOT be protected"

    def test_pre_existing_protections_intact(self):
        # The original three guarded endpoints
        assert _is_protected("/settler/register")
        assert _is_protected("/content/upload")
        assert _is_protected("/content/upload/shard")  # via startswith
        assert _is_protected("/compute/forge")
