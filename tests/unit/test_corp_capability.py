"""Sprint 306 — soulbound $CORP authorization capability.

Vision §7 Enterprise Confidentiality Mode layer 2:
ergonomics + accounting + audit. Not the security gate —
that's the encryption (sprint 304) + TEE policy (305/305a).
This layer is what makes "Confidential Computing" feel
like a corporate credit card to procurement.

The "soulbound" property is enforced by dual-signature
redemption: the capability is signed by the enterprise
issuer's key, but using it ALSO requires a fresh signature
from the subject's device-bound private key. Leaked
capability without leaked device key = useless. Phishing
both is equivalent to existing baseline corporate-account-
takeover risk; no new attack surface.

Quota model: capability declares quota_units; each
redemption declares units_requested; the store tracks
cumulative consumption; over-quota redemptions refuse.
"""
from __future__ import annotations

import base64
import json
import time

import pytest

from prsm.enterprise.corp_capability import (
    CAPABILITY_VERSION,
    CapabilityStatus,
    CorpCapability,
    CorpCapabilityStore,
    CorpIssuer,
    RedemptionRequest,
    canonical_capability_bytes,
    canonical_redemption_bytes,
    evaluate_redemption,
    generate_issuer_keypair,
    generate_subject_keypair,
    sign_capability,
    sign_redemption,
    verify_capability_signature,
    verify_redemption_signature,
)


# ── Keypair generation ───────────────────────────────


def test_issuer_keypair_b64_round_trip():
    priv, pub = generate_issuer_keypair()
    assert isinstance(priv, str)
    assert isinstance(pub, str)
    assert len(base64.b64decode(priv)) == 32
    assert len(base64.b64decode(pub)) == 32


def test_subject_keypair_b64_round_trip():
    priv, pub = generate_subject_keypair()
    assert len(base64.b64decode(priv)) == 32
    assert len(base64.b64decode(pub)) == 32


def test_issuer_and_subject_keypairs_distinct():
    """Both use Ed25519 but live in different rotational
    state; smoke-test that we don't accidentally collide."""
    ipriv, _ = generate_issuer_keypair()
    spriv, _ = generate_subject_keypair()
    assert ipriv != spriv


# ── Canonical encoding stability ─────────────────────


def test_canonical_capability_bytes_stable_under_field_order():
    """The signing bytes must be order-independent so a
    capability re-serialized through any JSON parser still
    verifies."""
    cap_a = CorpCapability(
        capability_id="cap-1",
        issuer_id="acme-corp",
        subject_id="alice@acme",
        subject_pubkey_b64="AAAA",
        scope=["compute.inference"],
        quota_units=1000,
        issued_at=100.0,
        expires_at=200.0,
        nonce="n-1",
        signature_b64="",
        version=CAPABILITY_VERSION,
    )
    b1 = canonical_capability_bytes(cap_a)
    # Re-create with same fields in different argument order
    cap_b = CorpCapability(
        signature_b64="",
        nonce="n-1",
        expires_at=200.0,
        issued_at=100.0,
        quota_units=1000,
        scope=["compute.inference"],
        subject_pubkey_b64="AAAA",
        subject_id="alice@acme",
        issuer_id="acme-corp",
        capability_id="cap-1",
        version=CAPABILITY_VERSION,
    )
    assert canonical_capability_bytes(cap_b) == b1


def test_canonical_excludes_signature_field():
    """The signature MUST NOT be part of what's signed —
    otherwise verification is circular."""
    cap = CorpCapability(
        capability_id="cap-1",
        issuer_id="acme",
        subject_id="alice",
        subject_pubkey_b64="A",
        scope=[],
        quota_units=0,
        issued_at=0.0,
        expires_at=0.0,
        nonce="n",
        signature_b64="signature-A",
    )
    cap2 = CorpCapability(
        capability_id="cap-1",
        issuer_id="acme",
        subject_id="alice",
        subject_pubkey_b64="A",
        scope=[],
        quota_units=0,
        issued_at=0.0,
        expires_at=0.0,
        nonce="n",
        signature_b64="signature-B",
    )
    assert canonical_capability_bytes(cap) == (
        canonical_capability_bytes(cap2)
    )


# ── Capability sign + verify ─────────────────────────


def _issued_capability(
    *, quota_units=100, expires_at=None,
    scope=("compute.inference",),
):
    ipriv, ipub = generate_issuer_keypair()
    spriv, spub = generate_subject_keypair()
    cap = sign_capability(
        issuer_id="acme-corp",
        issuer_privkey_b64=ipriv,
        subject_id="alice@acme",
        subject_pubkey_b64=spub,
        scope=list(scope),
        quota_units=quota_units,
        issued_at=time.time(),
        expires_at=(
            expires_at
            if expires_at is not None
            else time.time() + 3600
        ),
    )
    issuer = CorpIssuer(
        issuer_id="acme-corp",
        signing_pubkey_b64=ipub,
    )
    return cap, issuer, ipriv, spriv


def test_capability_sign_then_verify_passes():
    cap, issuer, _, _ = _issued_capability()
    assert verify_capability_signature(cap, issuer)


def test_capability_signature_tamper_detected():
    cap, issuer, _, _ = _issued_capability()
    # Mutate the quota AFTER signing — invariant: signature
    # binds the full canonical payload
    cap.quota_units = 999_999
    assert not verify_capability_signature(cap, issuer)


def test_capability_wrong_issuer_pubkey_rejected():
    cap, _, _, _ = _issued_capability()
    _, other_pub = generate_issuer_keypair()
    bogus_issuer = CorpIssuer(
        issuer_id="acme-corp",
        signing_pubkey_b64=other_pub,
    )
    assert not verify_capability_signature(
        cap, bogus_issuer,
    )


def test_capability_mismatched_issuer_id_rejected():
    """Caller hands us an issuer object claiming `issuer_id`
    'acme-corp' but the capability says 'evil-corp' — the
    verify must surface this mismatch even when the
    signature happens to verify."""
    cap, _, _, _ = _issued_capability()
    _, other_pub = generate_issuer_keypair()
    wrong_id_issuer = CorpIssuer(
        issuer_id="evil-corp",
        signing_pubkey_b64=other_pub,
    )
    assert not verify_capability_signature(
        cap, wrong_id_issuer,
    )


# ── Redemption sign + verify ─────────────────────────


def _redemption(
    spriv, capability_id="cap-1", action="compute.inference",
    units_requested=10, nonce="n-1", timestamp=None,
):
    return sign_redemption(
        subject_privkey_b64=spriv,
        capability_id=capability_id,
        action=action,
        units_requested=units_requested,
        nonce=nonce,
        timestamp=(
            timestamp if timestamp is not None
            else time.time()
        ),
    )


def test_redemption_sign_then_verify_passes():
    cap, _, _, spriv = _issued_capability()
    req = _redemption(spriv, capability_id=cap.capability_id)
    assert verify_redemption_signature(req, cap)


def test_redemption_signature_tamper_detected():
    cap, _, _, spriv = _issued_capability()
    req = _redemption(spriv, capability_id=cap.capability_id)
    req.units_requested = 999_999
    assert not verify_redemption_signature(req, cap)


def test_redemption_wrong_subject_key_rejected():
    """Phishing scenario: attacker has the capability but
    not the subject's device key. Their fresh redemption
    signature won't verify against the subject pubkey
    embedded in the capability."""
    cap, _, _, _ = _issued_capability()
    attacker_priv, _ = generate_subject_keypair()
    req = _redemption(
        attacker_priv,
        capability_id=cap.capability_id,
    )
    assert not verify_redemption_signature(req, cap)


# ── evaluate_redemption (full chain) ──────────────────


def test_evaluate_pass_within_quota():
    cap, issuer, _, spriv = _issued_capability(
        quota_units=100,
    )
    req = _redemption(
        spriv, capability_id=cap.capability_id,
        units_requested=10,
    )
    result = evaluate_redemption(
        capability=cap, request=req, issuer=issuer,
        consumed_so_far=20,
    )
    assert result.status == CapabilityStatus.PASS
    assert result.remaining_quota == 70  # 100 - 20 - 10


def test_evaluate_fail_over_quota():
    cap, issuer, _, spriv = _issued_capability(
        quota_units=100,
    )
    req = _redemption(
        spriv, capability_id=cap.capability_id,
        units_requested=50,
    )
    result = evaluate_redemption(
        capability=cap, request=req, issuer=issuer,
        consumed_so_far=80,
    )
    assert result.status == CapabilityStatus.FAIL
    assert "quota" in result.diagnostic.lower()


def test_evaluate_fail_expired():
    cap, issuer, _, spriv = _issued_capability(
        expires_at=time.time() - 10,
    )
    req = _redemption(
        spriv, capability_id=cap.capability_id,
        units_requested=1,
    )
    result = evaluate_redemption(
        capability=cap, request=req, issuer=issuer,
        consumed_so_far=0,
    )
    assert result.status == CapabilityStatus.FAIL
    assert "expired" in result.diagnostic.lower()


def test_evaluate_fail_scope_mismatch():
    cap, issuer, _, spriv = _issued_capability(
        scope=("content.upload",),
    )
    req = _redemption(
        spriv, capability_id=cap.capability_id,
        action="compute.inference",
    )
    result = evaluate_redemption(
        capability=cap, request=req, issuer=issuer,
        consumed_so_far=0,
    )
    assert result.status == CapabilityStatus.FAIL
    assert "scope" in result.diagnostic.lower()


def test_evaluate_fail_capability_signature_invalid():
    cap, issuer, _, spriv = _issued_capability()
    cap.quota_units = 999_999  # invalidates signature
    req = _redemption(
        spriv, capability_id=cap.capability_id,
    )
    result = evaluate_redemption(
        capability=cap, request=req, issuer=issuer,
        consumed_so_far=0,
    )
    assert result.status == CapabilityStatus.FAIL
    assert (
        "capability" in result.diagnostic.lower()
        or "signature" in result.diagnostic.lower()
    )


def test_evaluate_fail_redemption_signature_invalid():
    cap, issuer, _, _ = _issued_capability()
    # Attacker without subject's key
    attacker_priv, _ = generate_subject_keypair()
    req = _redemption(
        attacker_priv,
        capability_id=cap.capability_id,
    )
    result = evaluate_redemption(
        capability=cap, request=req, issuer=issuer,
        consumed_so_far=0,
    )
    assert result.status == CapabilityStatus.FAIL
    assert (
        "redemption" in result.diagnostic.lower()
        or "subject" in result.diagnostic.lower()
    )


def test_evaluate_fail_capability_id_mismatch():
    """Redemption refers to a different capability_id than
    the one being presented — refuse."""
    cap, issuer, _, spriv = _issued_capability()
    req = _redemption(
        spriv, capability_id="some-other-cap-id",
    )
    result = evaluate_redemption(
        capability=cap, request=req, issuer=issuer,
        consumed_so_far=0,
    )
    assert result.status == CapabilityStatus.FAIL


def test_evaluate_fail_zero_units_requested():
    """Zero-unit redemption is operator confusion / DoS
    surface — refuse loud."""
    cap, issuer, _, spriv = _issued_capability()
    req = _redemption(
        spriv, capability_id=cap.capability_id,
        units_requested=0,
    )
    result = evaluate_redemption(
        capability=cap, request=req, issuer=issuer,
        consumed_so_far=0,
    )
    assert result.status == CapabilityStatus.FAIL


# ── CorpCapabilityStore ──────────────────────────────


def test_store_register_and_list_issuer():
    store = CorpCapabilityStore()
    _, pub = generate_issuer_keypair()
    issuer = CorpIssuer(
        issuer_id="acme", signing_pubkey_b64=pub,
    )
    store.register_issuer(issuer)
    assert store.get_issuer("acme") == issuer
    assert "acme" in [
        i.issuer_id for i in store.list_issuers()
    ]


def test_store_register_duplicate_issuer_replaces():
    """Re-registering the same issuer_id rotates the
    pubkey. This is the legitimate key-rotation path."""
    store = CorpCapabilityStore()
    _, pub1 = generate_issuer_keypair()
    _, pub2 = generate_issuer_keypair()
    store.register_issuer(CorpIssuer("acme", pub1))
    store.register_issuer(CorpIssuer("acme", pub2))
    assert (
        store.get_issuer("acme").signing_pubkey_b64 == pub2
    )


def test_store_register_invalid_pubkey_rejected():
    store = CorpCapabilityStore()
    with pytest.raises(ValueError):
        store.register_issuer(CorpIssuer(
            issuer_id="acme",
            signing_pubkey_b64="not-base64!",
        ))


def test_store_redeem_records_ledger_and_decrements():
    store = CorpCapabilityStore()
    cap, issuer, _, spriv = _issued_capability(
        quota_units=100,
    )
    store.register_issuer(issuer)
    req = _redemption(
        spriv, capability_id=cap.capability_id,
        units_requested=25,
    )
    r1 = store.redeem(cap, req)
    assert r1.status == CapabilityStatus.PASS
    assert r1.remaining_quota == 75

    # Second redemption sees the prior consumption
    req2 = _redemption(
        spriv, capability_id=cap.capability_id,
        units_requested=30, nonce="n-2",
    )
    r2 = store.redeem(cap, req2)
    assert r2.status == CapabilityStatus.PASS
    assert r2.remaining_quota == 45


def test_store_redeem_unknown_issuer_rejected():
    store = CorpCapabilityStore()
    cap, _, _, spriv = _issued_capability()
    # Issuer NOT registered
    req = _redemption(
        spriv, capability_id=cap.capability_id,
    )
    result = store.redeem(cap, req)
    assert result.status == CapabilityStatus.FAIL
    assert "issuer" in result.diagnostic.lower()


def test_store_redeem_refused_does_not_consume_quota():
    store = CorpCapabilityStore()
    cap, issuer, _, spriv = _issued_capability(
        quota_units=10,
    )
    store.register_issuer(issuer)
    # Over-quota request
    req = _redemption(
        spriv, capability_id=cap.capability_id,
        units_requested=100,
    )
    result = store.redeem(cap, req)
    assert result.status == CapabilityStatus.FAIL
    # Quota must NOT be decremented on refusal
    assert store.get_consumed(cap.capability_id) == 0


def test_store_ledger_records_redemption_metadata():
    store = CorpCapabilityStore()
    cap, issuer, _, spriv = _issued_capability(
        quota_units=100,
    )
    store.register_issuer(issuer)
    req = _redemption(
        spriv, capability_id=cap.capability_id,
        units_requested=10, nonce="audit-1",
    )
    store.redeem(cap, req)
    entries = store.get_ledger(cap.capability_id)
    assert len(entries) == 1
    assert entries[0]["units_requested"] == 10
    assert entries[0]["nonce"] == "audit-1"
    assert entries[0]["action"] == "compute.inference"


def test_store_nonce_replay_refused():
    """Re-presenting the same redemption (same nonce) must
    be refused — otherwise an attacker who intercepts a
    valid redemption can replay it indefinitely."""
    store = CorpCapabilityStore()
    cap, issuer, _, spriv = _issued_capability(
        quota_units=100,
    )
    store.register_issuer(issuer)
    req = _redemption(
        spriv, capability_id=cap.capability_id,
        units_requested=10, nonce="replay-target",
    )
    r1 = store.redeem(cap, req)
    assert r1.status == CapabilityStatus.PASS
    # Replay the IDENTICAL redemption
    r2 = store.redeem(cap, req)
    assert r2.status == CapabilityStatus.FAIL
    assert (
        "replay" in r2.diagnostic.lower()
        or "nonce" in r2.diagnostic.lower()
    )


# ── Persistence ──────────────────────────────────────


def test_store_persist_round_trip(tmp_path):
    store = CorpCapabilityStore(persist_dir=tmp_path)
    cap, issuer, _, spriv = _issued_capability(
        quota_units=100,
    )
    store.register_issuer(issuer)
    req = _redemption(
        spriv, capability_id=cap.capability_id,
        units_requested=20,
    )
    store.redeem(cap, req)

    # Reload
    store2 = CorpCapabilityStore(persist_dir=tmp_path)
    assert store2.get_issuer(issuer.issuer_id) == issuer
    assert store2.get_consumed(cap.capability_id) == 20
    assert len(store2.get_ledger(cap.capability_id)) == 1


def test_store_persist_corrupt_file_failsoft(tmp_path):
    (tmp_path / "issuers.json").write_text("{not json")
    store = CorpCapabilityStore(persist_dir=tmp_path)
    assert store.list_issuers() == []


def test_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PRSM_CORP_CAPABILITY_DIR", str(tmp_path),
    )
    store = CorpCapabilityStore.from_env()
    _, pub = generate_issuer_keypair()
    store.register_issuer(CorpIssuer("acme", pub))
    # Persisted to disk
    assert (tmp_path / "issuers.json").exists()


# ── Serialization ───────────────────────────────────


def test_capability_to_dict_round_trip():
    cap, _, _, _ = _issued_capability()
    restored = CorpCapability.from_dict(cap.to_dict())
    assert restored == cap


def test_capability_from_dict_unknown_version_rejected():
    with pytest.raises(ValueError, match="version"):
        CorpCapability.from_dict({
            "version": "v999",
            "capability_id": "x",
            "issuer_id": "x",
            "subject_id": "x",
            "subject_pubkey_b64": "x",
            "scope": [],
            "quota_units": 0,
            "issued_at": 0.0,
            "expires_at": 0.0,
            "nonce": "x",
            "signature_b64": "x",
        })


def test_redemption_to_dict_round_trip():
    _, _, _, spriv = _issued_capability()
    req = _redemption(spriv)
    restored = RedemptionRequest.from_dict(req.to_dict())
    assert restored == req
