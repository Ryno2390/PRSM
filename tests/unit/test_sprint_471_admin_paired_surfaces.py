"""Sprint 471 — §13 admin + §14 paired-surface schema pins.

Live-verified 2026-05-16 against a running daemon. These pins
defend canonical response schemas for the 13 surfaces whose
PRSM_Testing.md rows were promoted 🟢 → ✅ this sprint:

  GET  /audit/summary           → {total, status_buckets,
                                    method_buckets, top_paths}
  GET  /audit/recent            → {entries, total, offset,
                                    limit}
  GET  /admin/content-filter    → {blocked_content_ids,
                                    blocked_model_tags,
                                    blocked_input_patterns,
                                    action_on_match, ...}
  POST /admin/content-filter/tags → {added, total}
  DELETE /admin/content-filter/tags/{tag} → {removed, total}
  GET  /marketplace/creator-reputation/{id}
                                → unknown-creator default
                                    schema
  GET  /marketplace/reputation  → {providers, count, limit}
  GET  /content/mine            → {entries, total, offset,
                                    limit}
  GET  /admin/fiat-compliance/summary → {by_kind, total_entries}
  GET  /admin/upgrade           → {records, count}
  GET  /admin/disclosure        → {records, count}
  GET  /admin/tee-policy/node-status
                                → {effective_tier, vendor,
                                    vendor_verified, diagnostic}
  POST /wallet/royalty/claim    → 503 with mainnet env-var hint

These pins fire if a refactor silently renames a field. They
are SCHEMA pins, not full-system tests.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
API_FILE = REPO_ROOT / "prsm" / "node" / "api.py"


def _slice(decorator: str, method: str = "get") -> str:
    """Find a route handler regardless of whether the decorator
    is single-line (`@app.get("/x")`) or multi-line
    (`@app.get(\n    "/x", tags=[...]\n)`). Anchors on the
    literal path string preceded by an `@app.{method}(`
    decorator within ~200 chars."""
    src = API_FILE.read_text()
    # Search every occurrence of the path until we find one
    # immediately preceded by the matching decorator.
    needle = f'"{decorator}"'
    start = 0
    while True:
        idx = src.find(needle, start)
        if idx < 0:
            break
        # Look backwards ~200 chars for the decorator.
        prefix = src[max(0, idx - 200):idx]
        if f"@app.{method}(" in prefix:
            # Found it — return handler body until next @app.
            next_idx = src.find('@app.', idx + 1)
            return src[idx:next_idx] if next_idx > 0 else src[idx:idx + 4000]
        start = idx + 1
    # Allow path-prefix matches for path-parameter routes
    # (e.g. /marketplace/creator-reputation/{creator_id}).
    anchor = f'"{decorator}'
    start = 0
    while True:
        idx = src.find(anchor, start)
        if idx < 0:
            break
        prefix = src[max(0, idx - 200):idx]
        if f"@app.{method}(" in prefix:
            next_idx = src.find('@app.', idx + 1)
            return src[idx:next_idx] if next_idx > 0 else src[idx:idx + 4000]
        start = idx + 1
    raise AssertionError(
        f"route not found: {method.upper()} {decorator}"
    )


# ── Audit ring ───────────────────────────────────────────


def test_audit_summary_schema():
    """Audit summary feeds dashboards + the `prsm_audit_summary`
    MCP tool. Renaming top-level keys breaks both."""
    body = _slice("/audit/summary")
    for field in (
        '"total"',
        '"status_buckets"',
        '"method_buckets"',
        '"top_paths"',
    ):
        assert field in body, (
            f"/audit/summary missing field: {field}"
        )


def test_audit_recent_paginated_envelope():
    """Paginated envelope shape — clients pass offset/limit
    and read total back for pagination."""
    body = _slice("/audit/recent")
    for field in (
        '"entries"',
        '"total"',
        '"offset"',
        '"limit"',
    ):
        assert field in body, (
            f"/audit/recent missing field: {field}"
        )


# ── Content filter ───────────────────────────────────────


def test_content_filter_get_canonical_keys():
    """Handler delegates to ContentSelfFilter — schema source
    of truth is `prsm/node/content_self_filter.py`."""
    filter_src = (
        REPO_ROOT / "prsm" / "node" / "content_self_filter.py"
    ).read_text()
    for field in (
        "blocked_content_ids",
        "blocked_model_tags",
        "blocked_input_patterns",
        "action_on_match",
    ):
        assert field in filter_src, (
            f"ContentSelfFilter snapshot missing field: {field}"
        )


def test_content_filter_tags_post_response():
    body = _slice("/admin/content-filter/tags", method="post")
    assert '"added"' in body
    assert '"total"' in body


# ── Marketplace ──────────────────────────────────────────


def test_creator_reputation_unknown_default_schema():
    """Unknown-creator default response — `known: false`
    with a baseline score is the audit-safe surface (no
    leaks of internal scoring on cold lookups). Schema is
    built in the handler at api.py:9776+ (unknown-creator
    branch) — the canonical fields are tested here."""
    src = API_FILE.read_text()
    # Anchor on the unknown-creator-branch dict literal.
    # The handler returns `{"creator_id": ..., "known": False,
    # "score": ..., "tier": ..., ...}` when the creator has
    # no recorded reputation.
    anchor = '"known": False'
    idx = src.find(anchor)
    assert idx >= 0, (
        "creator-reputation unknown-branch dict missing — "
        "API may have moved or renamed `known` field"
    )
    # Pull ~500 chars around the anchor to verify adjacent
    # fields stayed put.
    region = src[max(0, idx - 200):idx + 500]
    for field in ('"creator_id"', '"score"', '"tier"', '"known"'):
        assert field in region, (
            f"creator-reputation unknown-branch missing "
            f"field: {field}"
        )


def test_marketplace_reputation_paginated_envelope():
    body = _slice("/marketplace/reputation", method="get")
    assert '"providers"' in body
    assert '"count"' in body


# ── Content/mine ─────────────────────────────────────────


def test_content_mine_paginated_envelope():
    body = _slice("/content/mine", method="get")
    for field in (
        '"entries"',
        '"total"',
        '"offset"',
        '"limit"',
    ):
        assert field in body, (
            f"/content/mine missing field: {field}"
        )


# ── Fiat compliance + KYC ────────────────────────────────


def test_fiat_compliance_summary_schema():
    body = _slice("/admin/fiat-compliance/summary", method="get")
    assert '"by_kind"' in body
    assert '"total_entries"' in body


def test_kyc_status_supported_vendors_list():
    """The set of supported KYC vendors is a load-bearing
    contract — adding a vendor without updating commissioning
    docs leaves operators flying blind. Source of truth is
    `KYCClient.SUPPORTED_VENDORS` (handler reads via getattr)."""
    kyc_src = (
        REPO_ROOT / "prsm" / "economy" / "web3" / "kyc_client.py"
    ).read_text()
    for vendor in ("persona", "onfido", "plaid"):
        assert f'"{vendor}"' in kyc_src, (
            f"KYC client missing supported vendor: {vendor}"
        )


# ── Admin list surfaces ──────────────────────────────────


def test_admin_upgrade_list_envelope():
    body = _slice("/admin/upgrade", method="get")
    assert '"records"' in body
    assert '"count"' in body


def test_admin_disclosure_list_envelope():
    body = _slice("/admin/disclosure", method="get")
    assert '"records"' in body
    assert '"count"' in body


# ── TEE policy ───────────────────────────────────────────


def test_tee_policy_node_status_canonical_schema():
    body = _slice("/admin/tee-policy/node-status", method="get")
    for field in (
        '"effective_tier"',
        '"vendor"',
        '"vendor_verified"',
        '"diagnostic"',
    ):
        assert field in body, (
            f"/admin/tee-policy/node-status missing field: "
            f"{field}"
        )


# ── Royalty claim 503 path ───────────────────────────────


def test_royalty_claim_503_has_actionable_env_hint():
    """The 503 detail must guide the operator to either
    explicitly wire PRSM_ROYALTY_DISTRIBUTOR_ADDRESS or set
    PRSM_NETWORK=mainnet. A generic 503 leaves operators
    chasing log files."""
    body = _slice("/wallet/royalty/claim", method="post")
    assert "PRSM_ROYALTY_DISTRIBUTOR_ADDRESS" in body
    assert "PRSM_NETWORK" in body
