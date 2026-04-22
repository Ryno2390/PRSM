"""Unit tests for prsm.node.bootstrap.

Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md §6 Task 1.

Covers:
  - sign + verify round-trip
  - tampered body rejection
  - tampered signature rejection
  - wrong pubkey rejection
  - expired list rejection (past expiry + at-expiry boundary)
  - malformed document (missing fields, non-dict, bad version)
  - canonical JSON determinism
  - HTTPS fetch: success, network failure, bad JSON
  - DNS fetch: multi-chunk concatenation, empty, bad JSON
  - discover orchestration: primary wins, fallback on primary fail,
    fallback on primary bad-sig, both unavailable, fallback bad-sig
    re-raises
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from prsm.node.bootstrap import (
    BootstrapExpiredError,
    BootstrapList,
    BootstrapMalformedError,
    BootstrapPeer,
    BootstrapSignatureError,
    BootstrapUnavailableError,
    DnsBootstrapFetcher,
    HttpsBootstrapFetcher,
    _canonical_body_bytes,
    discover_bootstrap_peers,
    sign_bootstrap_list,
    verify_bootstrap_list,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def keypair():
    priv = Ed25519PrivateKey.generate()
    return priv, priv.public_key()


@pytest.fixture
def now():
    return datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def expires_future():
    return "2026-05-22T12:00:00Z"  # 30 days after `now`


@pytest.fixture
def sample_peers():
    return [
        BootstrapPeer(
            peer_id="12D3KooW0",
            multiaddrs=("/ip4/1.2.3.4/tcp/4001",),
            region="us-east",
            operator="PRSM Foundation",
        ),
        BootstrapPeer(
            peer_id="12D3KooW1",
            multiaddrs=("/ip4/5.6.7.8/tcp/4001", "/ip6/2001:db8::1/tcp/4001"),
            region="eu-west",
            operator="PRSM Foundation",
        ),
    ]


# -----------------------------------------------------------------------------
# Sign + verify
# -----------------------------------------------------------------------------


def test_sign_verify_roundtrip(keypair, sample_peers, now, expires_future):
    priv, pub = keypair
    doc = sign_bootstrap_list(sample_peers, expires_future, priv)
    result = verify_bootstrap_list(doc, pub, now=now)
    assert isinstance(result, BootstrapList)
    assert len(result.bootstrap_peers) == 2
    assert result.bootstrap_peers[0].peer_id == "12D3KooW0"


def test_tampered_body_rejected(keypair, sample_peers, now, expires_future):
    priv, pub = keypair
    doc = sign_bootstrap_list(sample_peers, expires_future, priv)
    # Flip a multiaddr — sig was over the original.
    doc["bootstrap_peers"][0]["multiaddrs"].append("/ip4/9.9.9.9/tcp/4001")
    with pytest.raises(BootstrapSignatureError):
        verify_bootstrap_list(doc, pub, now=now)


def test_tampered_signature_rejected(keypair, sample_peers, now, expires_future):
    priv, pub = keypair
    doc = sign_bootstrap_list(sample_peers, expires_future, priv)
    # Flip one base64 byte of the signature.
    bad_sig = list(doc["signature"])
    bad_sig[0] = "A" if bad_sig[0] != "A" else "B"
    doc["signature"] = "".join(bad_sig)
    with pytest.raises((BootstrapSignatureError, BootstrapMalformedError)):
        verify_bootstrap_list(doc, pub, now=now)


def test_wrong_pubkey_rejected(keypair, sample_peers, now, expires_future):
    priv, _ = keypair
    other_pub = Ed25519PrivateKey.generate().public_key()
    doc = sign_bootstrap_list(sample_peers, expires_future, priv)
    with pytest.raises(BootstrapSignatureError):
        verify_bootstrap_list(doc, other_pub, now=now)


def test_expired_list_rejected(keypair, sample_peers):
    priv, pub = keypair
    past = "2026-01-01T00:00:00Z"
    doc = sign_bootstrap_list(sample_peers, past, priv)
    now = datetime(2026, 4, 22, tzinfo=timezone.utc)
    with pytest.raises(BootstrapExpiredError):
        verify_bootstrap_list(doc, pub, now=now)


def test_at_exact_expiry_is_expired(keypair, sample_peers):
    """`now == expires_at` is treated as expired (half-open interval)."""
    priv, pub = keypair
    boundary = datetime(2026, 5, 1, tzinfo=timezone.utc)
    doc = sign_bootstrap_list(sample_peers, "2026-05-01T00:00:00Z", priv)
    with pytest.raises(BootstrapExpiredError):
        verify_bootstrap_list(doc, pub, now=boundary)


def test_malformed_missing_field_rejected(keypair, sample_peers, now, expires_future):
    priv, pub = keypair
    doc = sign_bootstrap_list(sample_peers, expires_future, priv)
    del doc["expires_at"]
    with pytest.raises(BootstrapMalformedError):
        verify_bootstrap_list(doc, pub, now=now)


def test_malformed_not_a_dict_rejected(keypair, now):
    _, pub = keypair
    with pytest.raises(BootstrapMalformedError):
        verify_bootstrap_list("not a dict", pub, now=now)  # type: ignore[arg-type]


def test_unknown_version_rejected(keypair, sample_peers, now, expires_future):
    priv, pub = keypair
    doc = sign_bootstrap_list(sample_peers, expires_future, priv, version=999)
    with pytest.raises(BootstrapMalformedError):
        verify_bootstrap_list(doc, pub, now=now)


# -----------------------------------------------------------------------------
# Canonical JSON determinism
# -----------------------------------------------------------------------------


def test_canonical_body_is_insertion_order_independent():
    a = {"version": 1, "expires_at": "2026-01-01T00:00:00Z", "bootstrap_peers": []}
    b = {"bootstrap_peers": [], "expires_at": "2026-01-01T00:00:00Z", "version": 1}
    assert _canonical_body_bytes(a) == _canonical_body_bytes(b)


# -----------------------------------------------------------------------------
# Fetchers
# -----------------------------------------------------------------------------


def test_https_fetcher_success():
    body = '{"version": 1, "expires_at": "x", "bootstrap_peers": [], "signature": "s"}'
    fetcher = HttpsBootstrapFetcher(url="https://x", get=lambda url: body)
    assert fetcher.fetch() == json.loads(body)


def test_https_fetcher_network_failure_returns_none():
    def raising_get(url):
        raise ConnectionError("no route")

    fetcher = HttpsBootstrapFetcher(url="https://x", get=raising_get)
    assert fetcher.fetch() is None


def test_https_fetcher_bad_json_returns_none():
    fetcher = HttpsBootstrapFetcher(url="https://x", get=lambda url: "{not json")
    assert fetcher.fetch() is None


def test_dns_fetcher_concatenates_multi_chunk_body():
    body = '{"version": 1, "expires_at": "x", "bootstrap_peers": [], "signature": "s"}'
    # Split into 2 chunks simulating 255B TXT boundary.
    mid = len(body) // 2
    chunks = [body[:mid], body[mid:]]
    fetcher = DnsBootstrapFetcher(domain="_prsm", resolve_txt=lambda d: chunks)
    assert fetcher.fetch() == json.loads(body)


def test_dns_fetcher_empty_returns_none():
    fetcher = DnsBootstrapFetcher(domain="_prsm", resolve_txt=lambda d: [])
    assert fetcher.fetch() is None


def test_dns_fetcher_resolve_failure_returns_none():
    def raising(d):
        raise RuntimeError("no DNS")

    fetcher = DnsBootstrapFetcher(domain="_prsm", resolve_txt=raising)
    assert fetcher.fetch() is None


# -----------------------------------------------------------------------------
# Discovery orchestration
# -----------------------------------------------------------------------------


class _StubFetcher:
    def __init__(self, doc=None, fail=False):
        self._doc = doc
        self._fail = fail
        self.calls = 0

    def fetch(self):
        self.calls += 1
        if self._fail:
            return None
        return self._doc


def test_discover_prefers_primary_when_valid(keypair, sample_peers, now, expires_future):
    priv, pub = keypair
    doc = sign_bootstrap_list(sample_peers, expires_future, priv)
    primary = _StubFetcher(doc=doc)
    fallback = _StubFetcher(fail=True)

    result = discover_bootstrap_peers(pub, primary=primary, fallback=fallback, now=now)
    assert len(result.bootstrap_peers) == 2
    assert fallback.calls == 0  # fallback not consulted


def test_discover_falls_back_when_primary_unavailable(
    keypair, sample_peers, now, expires_future
):
    priv, pub = keypair
    doc = sign_bootstrap_list(sample_peers, expires_future, priv)
    primary = _StubFetcher(fail=True)
    fallback = _StubFetcher(doc=doc)

    result = discover_bootstrap_peers(pub, primary=primary, fallback=fallback, now=now)
    assert len(result.bootstrap_peers) == 2
    assert fallback.calls == 1


def test_discover_falls_back_when_primary_signature_fails(
    keypair, sample_peers, now, expires_future
):
    priv, pub = keypair
    # Primary has a valid-looking doc signed by the wrong key.
    wrong_priv = Ed25519PrivateKey.generate()
    bad_doc = sign_bootstrap_list(sample_peers, expires_future, wrong_priv)
    good_doc = sign_bootstrap_list(sample_peers, expires_future, priv)

    primary = _StubFetcher(doc=bad_doc)
    fallback = _StubFetcher(doc=good_doc)

    result = discover_bootstrap_peers(pub, primary=primary, fallback=fallback, now=now)
    assert len(result.bootstrap_peers) == 2
    assert fallback.calls == 1


def test_discover_raises_when_both_unavailable(keypair, now):
    _, pub = keypair
    primary = _StubFetcher(fail=True)
    fallback = _StubFetcher(fail=True)

    with pytest.raises(BootstrapUnavailableError):
        discover_bootstrap_peers(pub, primary=primary, fallback=fallback, now=now)


def test_discover_fallback_bad_signature_propagates(keypair, sample_peers, now, expires_future):
    """A fallback that fetches but fails verification raises rather than
    silently returning — operators must see the key-compromise signal."""
    _, pub = keypair
    wrong_priv = Ed25519PrivateKey.generate()
    bad_doc = sign_bootstrap_list(sample_peers, expires_future, wrong_priv)

    primary = _StubFetcher(fail=True)
    fallback = _StubFetcher(doc=bad_doc)

    with pytest.raises(BootstrapSignatureError):
        discover_bootstrap_peers(pub, primary=primary, fallback=fallback, now=now)
