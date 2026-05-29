"""Sprint 892 — KYC webhook exempt from node-API-key auth.

PRODUCTION BUG surfaced while grounding the sp887 review against the
auth posture. The KYC vendor webhook (`/wallet/kyc/webhook/{vendor}`)
is called BY THE VENDOR (Persona / Onfido / Plaid) — an external
caller that does NOT possess the operator's PRSM_NODE_API_KEY. The
webhook authenticates via its OWN HMAC signature (sp283 + sp888
fail-closed), NOT the node API key.

But the path is under the `/wallet/` PROTECTED_PREFIX, and
PUBLIC_ENDPOINTS is exact-match (doesn't cover the {vendor}
segment). So when an operator sets PRSM_NODE_API_KEY (i.e.
production), the NodeAuthMiddleware 401s the vendor's webhook
BEFORE its signature check runs — breaking the entire
KYC → VERIFIED → auto-provision loop in exactly the deployment
mode operators run in. The sp888 fail-closed signature verification
is unreachable behind the middleware.

Sp892 exempts the webhook prefix from node-API-key auth (it has its
own, stronger, purpose-built signature auth). This does NOT weaken
security: the endpoint still enforces sp888 (no secret → 503; bad
signature → 401 from the endpoint). Other /wallet/* endpoints stay
node-API-key protected.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from prsm.api.auth_middleware import (
    NodeAuthMiddleware,
    hash_api_key,
)


def _app_with_auth(api_key: str):
    app = FastAPI()
    app.add_middleware(
        NodeAuthMiddleware, api_key_hash=hash_api_key(api_key),
    )

    @app.post("/wallet/kyc/webhook/{vendor}")
    def webhook(vendor: str):
        # Reaching here means the middleware let us through; the
        # real endpoint then does signature verification.
        return {"reached": True, "vendor": vendor}

    @app.post("/wallet/onramp/execute")
    def execute():
        return {"reached": True}

    @app.get("/admin/fiat-compliance")
    def admin():
        return {"reached": True}

    return TestClient(app, raise_server_exceptions=False)


# ── The fix: webhook reachable WITHOUT the node API key ──────

def test_kyc_webhook_reachable_without_node_api_key():
    """With PRSM_NODE_API_KEY set (production), the vendor webhook
    must be reachable WITHOUT that key — it has its own signature
    auth. Pre-sp892 this returned 401 from the middleware."""
    client = _app_with_auth("operator-secret-key")
    resp = client.post("/wallet/kyc/webhook/persona", json={})
    assert resp.status_code == 200
    assert resp.json()["reached"] is True


def test_kyc_webhook_reachable_for_all_vendors():
    client = _app_with_auth("operator-secret-key")
    for vendor in ("persona", "onfido", "plaid"):
        resp = client.post(f"/wallet/kyc/webhook/{vendor}", json={})
        assert resp.status_code == 200, vendor


# ── The exemption is SCOPED — other /wallet/* still protected ─

def test_other_wallet_endpoints_still_require_key():
    """The exemption must NOT leak to value-moving endpoints —
    /wallet/onramp/execute still requires the node API key."""
    client = _app_with_auth("operator-secret-key")
    resp = client.post("/wallet/onramp/execute", json={})
    assert resp.status_code == 401


def test_admin_endpoints_still_require_key():
    client = _app_with_auth("operator-secret-key")
    resp = client.get("/admin/fiat-compliance")
    assert resp.status_code == 401


def test_other_wallet_endpoint_passes_with_correct_key():
    client = _app_with_auth("operator-secret-key")
    resp = client.post(
        "/wallet/onramp/execute",
        json={},
        headers={"X-API-Key": "operator-secret-key"},
    )
    assert resp.status_code == 200


# ── Dev mode (no key) unaffected ─────────────────────────────

def test_dev_mode_everything_open():
    """No PRSM_NODE_API_KEY (api_key_hash="") → auth disabled;
    webhook + everything reachable (existing dev behavior)."""
    app = FastAPI()
    app.add_middleware(NodeAuthMiddleware, api_key_hash="")

    @app.post("/wallet/kyc/webhook/{vendor}")
    def webhook(vendor: str):
        return {"reached": True}

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/wallet/kyc/webhook/persona", json={})
    assert resp.status_code == 200


# ── A near-miss path must NOT be exempted ────────────────────

def test_webhook_lookalike_path_not_exempted():
    """Only the exact webhook prefix is exempt — a lookalike like
    /wallet/kyc/status (not a vendor callback) stays protected."""
    app = FastAPI()
    app.add_middleware(
        NodeAuthMiddleware, api_key_hash=hash_api_key("k"),
    )

    @app.get("/wallet/kyc/status")
    def status():
        return {"reached": True}

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/wallet/kyc/status")
    assert resp.status_code == 401
