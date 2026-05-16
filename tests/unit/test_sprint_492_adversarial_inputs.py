"""Sprint 492 — adversarial input fixes (F30/F31/F32).

Coverage matrix priority #3: adversarial inputs. We schema-
pin canonical fields but had never tried to bypass them.

Sprint 492 found 3 production-relevant bypasses:

  F30 — Content filter CID case-evasion. is_cid_blocked
    used literal `cid in self._cids` set lookup.
    Operator blocking `ABC123` didn't block requests for
    `abc123` (and vice versa). PRSM CIDs are lowercase hex
    by convention — fix lowercase-normalizes both sides.

  F31 — Tag CRLF / control-char injection. add_tags only
    stripped leading/trailing whitespace, leaving embedded
    \\r\\n / NUL / control chars in stored tags. Enabled
    log-injection + CLI display corruption. Fix sanitizes
    to printable ASCII only.

  F32 — Settler register accepted unbacked bond. The
    SettlerRegistry called ftns_service.lock_tokens() but
    didn't check the return value; the adapter's
    lock_tokens swallowed ALL exceptions (including its
    own InsufficientBalance) and silently returned False.
    Adversary could register with 10^12 FTNS bond against
    a 1083-FTNS wallet. Anti-Sybil completely broken.
    Fix: adapter raises on insufficient balance; registry
    checks return value too (defense-in-depth).

These pins defend all 3 fixes.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from prsm.node.content_filter_store import ContentFilterStore


REPO_ROOT = Path(__file__).resolve().parents[2]


# ── F30: Case-insensitive CID matching ──────────────────


def test_f30_blocked_cid_case_insensitive_match():
    """A CID added in UPPERCASE must block requests in any
    case variant (lowercase, mixed). Vision §14 anti-evasion
    invariant."""
    store = ContentFilterStore()
    store.add_cids(["ABCDEF123"])
    assert store.is_cid_blocked("ABCDEF123") is True
    assert store.is_cid_blocked("abcdef123") is True
    assert store.is_cid_blocked("AbCdEf123") is True
    # Unrelated CID is not blocked
    assert store.is_cid_blocked("0123456789") is False


def test_f30_add_normalizes_to_lowercase():
    """Storage is normalized to lowercase so canonical form
    is consistent regardless of operator input."""
    store = ContentFilterStore()
    store.add_cids(["MixedCASE"])
    d = store.to_dict()
    # All stored CIDs must be lowercase
    for cid in d["blocked_content_ids"]:
        assert cid == cid.lower(), (
            f"stored CID not lowercase-normalized: {cid!r}"
        )


def test_f30_remove_case_insensitive():
    """Remove with different case must find + remove the
    canonically-stored entry."""
    store = ContentFilterStore()
    store.add_cids(["LowerCase"])
    assert store.remove_cid("LOWERCASE") is True
    assert store.is_cid_blocked("lowercase") is False


# ── F31: Tag control-char sanitization ──────────────────


def test_f31_crlf_in_tag_stripped():
    """Tag with embedded \\r\\n must have control chars
    removed before storage. Defends against log injection
    + CLI display corruption."""
    store = ContentFilterStore()
    store.add_tags(["benign\r\ninjected"])
    d = store.to_dict()
    # The CRLF must NOT survive in stored tags
    for tag in d["blocked_model_tags"]:
        assert "\r" not in tag
        assert "\n" not in tag


def test_f31_null_byte_in_tag_stripped():
    """NUL byte must be sanitized. Some logging frameworks
    truncate at NUL, hiding subsequent attack content."""
    store = ContentFilterStore()
    store.add_tags(["bad\x00content"])
    d = store.to_dict()
    for tag in d["blocked_model_tags"]:
        assert "\x00" not in tag


def test_f31_non_printable_ascii_stripped():
    """Sanitization is to printable ASCII (0x20–0x7E).
    Any control char or extended-byte must be removed."""
    store = ContentFilterStore()
    store.add_tags(["abc\x01\x02\x1fdef\x7f"])
    d = store.to_dict()
    for tag in d["blocked_model_tags"]:
        for ch in tag:
            assert 0x20 <= ord(ch) <= 0x7E, (
                f"non-printable char {ch!r} in stored tag "
                f"{tag!r}"
            )


# ── F32: Settler register bond-balance check ────────────


def test_f32_staking_ftns_adapter_raises_on_insufficient():
    """_StakingFTNSAdapter.lock_tokens must raise ValueError
    on insufficient available balance — pre-fix it
    swallowed everything and returned False, letting the
    SettlerRegistry create unbacked registrations."""
    from prsm.node.node import _StakingFTNSAdapter

    class _FakeLedger:
        async def get_balance(self, wid):
            return 100.0

    adapter = _StakingFTNSAdapter(_FakeLedger(), "test-node")

    async def attempt():
        # Try to lock 1000 against 100 balance
        return await adapter.lock_tokens(
            "test-node", Decimal("1000"), reason="probe",
        )

    with pytest.raises(ValueError, match="Insufficient"):
        asyncio.run(attempt())


def test_f32_staking_ftns_adapter_succeeds_when_sufficient():
    """Positive case: lock succeeds + balance accounting
    works."""
    from prsm.node.node import _StakingFTNSAdapter

    class _FakeLedger:
        async def get_balance(self, wid):
            return 5000.0

    adapter = _StakingFTNSAdapter(_FakeLedger(), "test-node")

    async def attempt():
        ok = await adapter.lock_tokens(
            "test-node", Decimal("1000"),
        )
        avail_after = await adapter.get_available_balance(
            "test-node",
        )
        return ok, avail_after

    ok, avail = asyncio.run(attempt())
    assert ok is True
    assert avail == Decimal("4000")  # 5000 - 1000 locked


def test_f32_settler_registry_checks_lock_result():
    """Source pin: settler_registry.register_settler must
    check lock_tokens return value AND raise on failure.
    Pre-fix it called the function but discarded the bool."""
    src = (
        REPO_ROOT / "prsm" / "node" / "settler_registry.py"
    ).read_text()
    idx = src.find("async def register_settler")
    assert idx >= 0
    body = src[idx:idx + 3000]
    # Must assign + check the result
    assert "locked = await self.ftns_service.lock_tokens" in body, (
        "F32 regression: register_settler must capture "
        "lock_tokens return value"
    )
    assert "if not locked:" in body, (
        "F32 regression: register_settler must check the "
        "lock_tokens boolean and raise on False"
    )


def test_f32_adapter_no_longer_swallows_exceptions():
    """Source pin: _StakingFTNSAdapter.lock_tokens must NOT
    have the `except Exception: return False` swallow
    pattern. Pre-fix this hid ALL failure modes."""
    src = (REPO_ROOT / "prsm" / "node" / "node.py").read_text()
    idx = src.find("async def lock_tokens(self, user_id")
    assert idx >= 0
    body = src[idx:idx + 1500]
    # Sprint 492 marker must remain
    assert "Sprint 492 (F32 fix)" in body
    # The dangerous broad-except + return-False pattern must
    # NOT appear in this method body
    assert "except Exception:\n            return False" not in body
