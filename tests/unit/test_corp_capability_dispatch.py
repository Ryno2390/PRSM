"""Sprint 306a — live header-driven $CORP redemption.

When `X-CORP-Capability` + `X-CORP-Redemption` headers
(both base64-JSON) are present on a /compute/inference or
/compute/inference/stream request, the node verifies the
dual-signature capability + redemption BEFORE dispatching.

Refusal: 402 Payment Required with diagnostic (semantically:
"this authorization isn't valid against your quota").
Distinct from 412 (TEE policy, sprint 305a) and 451
(content filter, sprint 271) so operators can tell which
gate refused.

Both headers must be present together — either alone is
operator confusion → 422. Headers absent → no gating
(fully backwards-compatible; opt-in for enterprises).

Successful redemption consumes quota; replays of the same
nonce are refused even within the same request lifecycle.
"""
from __future__ import annotations

import base64
import json
import time
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.enterprise.corp_capability import (
    CorpCapabilityStore, CorpIssuer,
    generate_issuer_keypair, generate_subject_keypair,
    sign_capability, sign_redemption,
)
from prsm.node.api import create_api_app


def _client(store=None, attestation_blob=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._corp_capability_store = store
    node._tee_node_attestation_blob = attestation_blob
    node._content_filter_store = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    node.inference_executor = None  # downstream 5xx
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _b64j(obj) -> str:
    return base64.b64encode(
        json.dumps(obj).encode("utf-8"),
    ).decode("ascii")


def _body(**extra):
    body = {
        "prompt": "hello",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    }
    body.update(extra)
    return body


def _setup(store, *, quota_units=100,
           scope=("compute.inference",), expires_in=3600):
    """Issue a fresh capability + register the issuer.
    Returns (capability_dict, redemption_dict_factory)."""
    ipriv, ipub = generate_issuer_keypair()
    spriv, spub = generate_subject_keypair()
    issuer = CorpIssuer("acme-corp", ipub)
    store.register_issuer(issuer)
    now = time.time()
    cap = sign_capability(
        issuer_id="acme-corp",
        issuer_privkey_b64=ipriv,
        subject_id="alice@acme",
        subject_pubkey_b64=spub,
        scope=list(scope),
        quota_units=quota_units,
        issued_at=now,
        expires_at=now + expires_in,
    )

    def _redeem(
        *, action="compute.inference", units=10,
        nonce="n-1", timestamp=None,
        subject_privkey=spriv,
    ):
        return sign_redemption(
            subject_privkey_b64=subject_privkey,
            capability_id=cap.capability_id,
            action=action,
            units_requested=units,
            nonce=nonce,
            timestamp=(
                timestamp if timestamp is not None else time.time()
            ),
        )
    return cap, _redeem, spriv


# ── No headers → backwards-compat ───────────────────


def test_inference_without_corp_headers_unaffected():
    """No X-CORP-* headers → existing behavior preserved.
    Downstream 5xx (executor unwired), NOT 402."""
    store = CorpCapabilityStore()
    resp = _client(store=store).post(
        "/compute/inference", json=_body(),
    )
    assert resp.status_code != 402


def test_inference_with_only_capability_header_422():
    """Mismatched header pair = operator confusion."""
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(store)
    resp = _client(store=store).post(
        "/compute/inference", json=_body(),
        headers={"X-CORP-Capability": _b64j(cap.to_dict())},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "X-CORP" in detail


def test_inference_with_only_redemption_header_422():
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(store)
    req = redeem_fn()
    resp = _client(store=store).post(
        "/compute/inference", json=_body(),
        headers={
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code == 422


# ── Happy path ───────────────────────────────────────


def test_inference_with_valid_headers_passes_gate():
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(store)
    req = redeem_fn(units=10)
    resp = _client(store=store).post(
        "/compute/inference", json=_body(),
        headers={
            "X-CORP-Capability": _b64j(cap.to_dict()),
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    # NOT 402 — gate passed. Downstream stages reject for
    # different reasons (executor unwired etc.).
    assert resp.status_code != 402
    # Quota was consumed
    assert store.get_consumed(cap.capability_id) == 10


def test_inference_consumes_quota_on_success():
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(store, quota_units=100)
    for i in range(3):
        req = redeem_fn(units=5, nonce=f"n-{i}")
        resp = _client(store=store).post(
            "/compute/inference", json=_body(),
            headers={
                "X-CORP-Capability": _b64j(cap.to_dict()),
                "X-CORP-Redemption": _b64j(req.to_dict()),
            },
        )
        assert resp.status_code != 402
    assert store.get_consumed(cap.capability_id) == 15


# ── Failure modes — 402 ─────────────────────────────


def test_inference_402_when_issuer_not_registered():
    """Capability signed by an issuer this node doesn't
    know about. Different store instance simulates the
    'wrong PRSM node' case."""
    issuing_store = CorpCapabilityStore()
    serving_store = CorpCapabilityStore()  # issuer NOT registered here
    cap, redeem_fn, _ = _setup(issuing_store)
    req = redeem_fn()
    resp = _client(store=serving_store).post(
        "/compute/inference", json=_body(),
        headers={
            "X-CORP-Capability": _b64j(cap.to_dict()),
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code == 402
    detail = resp.json()["detail"]
    assert "issuer" in detail.lower()


def test_inference_402_when_quota_exhausted():
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(store, quota_units=10)
    req = redeem_fn(units=100)  # over quota
    resp = _client(store=store).post(
        "/compute/inference", json=_body(),
        headers={
            "X-CORP-Capability": _b64j(cap.to_dict()),
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code == 402
    assert "quota" in resp.json()["detail"].lower()
    # Quota NOT consumed on refusal
    assert store.get_consumed(cap.capability_id) == 0


def test_inference_402_when_capability_expired():
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(store, expires_in=-10)
    req = redeem_fn()
    resp = _client(store=store).post(
        "/compute/inference", json=_body(),
        headers={
            "X-CORP-Capability": _b64j(cap.to_dict()),
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code == 402
    assert "expired" in resp.json()["detail"].lower()


def test_inference_402_when_capability_tampered():
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(store, quota_units=10)
    req = redeem_fn()
    tampered = cap.to_dict()
    tampered["quota_units"] = 999_999  # invalidates sig
    resp = _client(store=store).post(
        "/compute/inference", json=_body(),
        headers={
            "X-CORP-Capability": _b64j(tampered),
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code == 402


def test_inference_402_when_redemption_from_wrong_subject():
    """Phishing scenario: attacker has the capability but
    not the subject's device key."""
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(store)
    attacker_priv, _ = generate_subject_keypair()
    req = redeem_fn(subject_privkey=attacker_priv)
    resp = _client(store=store).post(
        "/compute/inference", json=_body(),
        headers={
            "X-CORP-Capability": _b64j(cap.to_dict()),
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code == 402


def test_inference_402_when_scope_mismatch():
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(
        store, scope=("content.upload",),
    )
    req = redeem_fn(action="compute.inference")
    resp = _client(store=store).post(
        "/compute/inference", json=_body(),
        headers={
            "X-CORP-Capability": _b64j(cap.to_dict()),
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code == 402
    assert "scope" in resp.json()["detail"].lower()


def test_inference_402_replay_attack_refused():
    """Same redemption used twice — second use refused."""
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(store)
    req = redeem_fn(nonce="replay-target")
    headers = {
        "X-CORP-Capability": _b64j(cap.to_dict()),
        "X-CORP-Redemption": _b64j(req.to_dict()),
    }
    r1 = _client(store=store).post(
        "/compute/inference", json=_body(), headers=headers,
    )
    assert r1.status_code != 402
    r2 = _client(store=store).post(
        "/compute/inference", json=_body(), headers=headers,
    )
    assert r2.status_code == 402
    assert (
        "replay" in r2.json()["detail"].lower()
        or "nonce" in r2.json()["detail"].lower()
    )


# ── Malformed header content ────────────────────────


def test_inference_422_when_capability_not_base64():
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(store)
    req = redeem_fn()
    resp = _client(store=store).post(
        "/compute/inference", json=_body(),
        headers={
            "X-CORP-Capability": "not-base64!",
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code == 422


def test_inference_422_when_capability_not_json():
    store = CorpCapabilityStore()
    _, redeem_fn, _ = _setup(store)
    req = redeem_fn()
    resp = _client(store=store).post(
        "/compute/inference", json=_body(),
        headers={
            "X-CORP-Capability": base64.b64encode(
                b"not json",
            ).decode(),
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code == 422


def test_inference_422_when_capability_malformed_dict():
    store = CorpCapabilityStore()
    _, redeem_fn, _ = _setup(store)
    req = redeem_fn()
    resp = _client(store=store).post(
        "/compute/inference", json=_body(),
        headers={
            "X-CORP-Capability": _b64j(
                {"version": "wat", "junk": True},
            ),
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code == 422


def test_inference_503_when_store_unwired_but_headers_present():
    """If the operator hasn't wired the $CORP store but the
    requester sends headers, refuse explicitly with 503 so
    the requester knows the node can't honor the protocol."""
    cap, redeem_fn, _ = _setup(CorpCapabilityStore())
    req = redeem_fn()
    resp = _client(store=None).post(
        "/compute/inference", json=_body(),
        headers={
            "X-CORP-Capability": _b64j(cap.to_dict()),
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code == 503


# ── Streaming endpoint mirrors the same gate ───────


def test_stream_inference_402_on_capability_failure():
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(store, quota_units=5)
    req = redeem_fn(units=100)
    resp = _client(store=store).post(
        "/compute/inference/stream", json=_body(),
        headers={
            "X-CORP-Capability": _b64j(cap.to_dict()),
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code == 402


def test_stream_inference_without_corp_headers_unaffected():
    store = CorpCapabilityStore()
    resp = _client(store=store).post(
        "/compute/inference/stream", json=_body(),
    )
    assert resp.status_code != 402


def test_stream_inference_valid_headers_pass():
    store = CorpCapabilityStore()
    cap, redeem_fn, _ = _setup(store, quota_units=100)
    req = redeem_fn(units=5)
    resp = _client(store=store).post(
        "/compute/inference/stream", json=_body(),
        headers={
            "X-CORP-Capability": _b64j(cap.to_dict()),
            "X-CORP-Redemption": _b64j(req.to_dict()),
        },
    )
    assert resp.status_code != 402
    assert store.get_consumed(cap.capability_id) == 5
