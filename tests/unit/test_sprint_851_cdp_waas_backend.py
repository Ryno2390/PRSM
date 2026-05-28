"""Sprint 851 — CDP WaaS HTTP backend + CoinbaseWaaSClient auto-wire.

Final adapter_wired closure of the Phase 5 trifecta. JSON-RPC isn't
used here — CDP v2 EVM accounts use a REST endpoint with per-request
Ed25519-signed JWTs.

Pin tests:
  - _load_ed25519_pem rejects placeholders + non-PEM strings cleanly
  - _load_ed25519_pem accepts a freshly-generated Ed25519 PEM
  - _build_jwt envelope has correct alg/kid/aud/uri shape
  - create_wallet POSTs with Authorization: Bearer <jwt>
  - create_wallet returns dict shape (wallet_id, address, network)
  - create_wallet raises when CDP response has no address
  - create_wallet raises on HTTP 4xx
  - from_env returns None when key_name missing
  - from_env returns None when key_priv missing
  - from_env returns None when PEM is a placeholder (honest signal)
  - from_env constructs backend with real PEM
  - CoinbaseWaaSClient.from_env auto-attaches when env wired w/ real PEM
  - CoinbaseWaaSClient.from_env adapter_wired stays False with placeholder
  - CoinbaseWaaSClient.from_env respects explicit backend= override
"""
from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

# Restore real httpx classes
_real_Client = httpx.Client
_real_MockTransport = httpx.MockTransport
_real_Response = httpx.Response
_real_HTTPStatusError = httpx.HTTPStatusError


@pytest.fixture(autouse=True)
def _restore_real_httpx_classes(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _real_Client)
    monkeypatch.setattr(httpx, "MockTransport", _real_MockTransport)
    monkeypatch.setattr(httpx, "Response", _real_Response)
    monkeypatch.setattr(httpx, "HTTPStatusError", _real_HTTPStatusError)
    yield


from prsm.economy.web3.coinbase_waas_cdp_backend import (
    CdpWaaSBackend,
    _load_ed25519_pem,
    from_env as cdp_waas_from_env,
)
from prsm.economy.web3.coinbase_waas_client import CoinbaseWaaSClient


# ── Helpers ──────────────────────────────────────────────────

def _generate_test_pem() -> str:
    """Fresh Ed25519 PEM for tests — never use real CDP key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PrivateFormat, NoEncryption,
    )
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    ).decode("utf-8")
    return pem


def _mock_transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# ── _load_ed25519_pem ────────────────────────────────────────

def test_load_pem_rejects_placeholder():
    with pytest.raises(ValueError) as exc:
        _load_ed25519_pem("REPLACE_WITH_ED25519_PEM_FROM_CDP_DOWNLOAD")
    assert "placeholder" in str(exc.value)


def test_load_pem_rejects_empty():
    with pytest.raises(ValueError):
        _load_ed25519_pem("")


def test_load_pem_rejects_garbage():
    with pytest.raises(ValueError) as exc:
        _load_ed25519_pem("not a real pem at all")
    assert "Ed25519 PEM parse failed" in str(exc.value)


def test_load_pem_accepts_real_ed25519():
    pem = _generate_test_pem()
    key = _load_ed25519_pem(pem)
    # Loaded successfully — no exception
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    assert isinstance(key, Ed25519PrivateKey)


# ── _build_jwt envelope ──────────────────────────────────────

def test_build_jwt_envelope_shape():
    """JWT must have alg=EdDSA, kid=key_name, iss=cdp, sub=key_name,
    aud=[method, host], uri='METHOD host/path'."""
    import jwt as pyjwt
    pem = _generate_test_pem()
    backend = CdpWaaSBackend(
        api_key_name="kid_uuid_test",
        api_key_private_pem=pem,
        client=_mock_transport(lambda r: httpx.Response(200, json={})),
    )
    token = backend._build_jwt("POST", "/platform/v2/evm/accounts")

    # Decode without signature verification just to inspect shape
    headers = pyjwt.get_unverified_header(token)
    payload = pyjwt.decode(
        token, options={"verify_signature": False},
    )
    assert headers["alg"] == "EdDSA"
    assert headers["kid"] == "kid_uuid_test"
    assert headers["typ"] == "JWT"
    assert "nonce" in headers
    assert payload["iss"] == "cdp"
    assert payload["sub"] == "kid_uuid_test"
    assert payload["aud"] == ["POST", "api.cdp.coinbase.com"]
    assert payload["uri"] == (
        "POST api.cdp.coinbase.com/platform/v2/evm/accounts"
    )
    assert payload["exp"] > payload["nbf"]


# ── CdpWaaSBackend.create_wallet ─────────────────────────────

def test_create_wallet_sends_bearer_jwt_and_returns_address():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        captured["path"] = request.url.path
        return httpx.Response(200, json={
            "name": "prsm-alice",
            "address": "0x" + "ab" * 20,
        })

    pem = _generate_test_pem()
    backend = CdpWaaSBackend(
        api_key_name="kid_test",
        api_key_private_pem=pem,
        client=_mock_transport(handler),
    )
    result = backend.create_wallet("alice", "a@x.io")
    assert result["address"] == "0x" + "ab" * 20
    assert result["network"] == "base-mainnet"
    assert result["wallet_id"] == "prsm-alice"
    # Authorization header is a Bearer JWT
    auth = captured["headers"]["authorization"]
    assert auth.startswith("Bearer ey")  # JWT base64 prefix
    # body contains user_id metadata
    assert captured["body"]["metadata"]["user_id"] == "alice"
    assert captured["body"]["metadata"]["email"] == "a@x.io"
    assert captured["path"] == "/platform/v2/evm/accounts"


def test_create_wallet_tolerates_data_envelope():
    """Some CDP responses wrap in {data: {...}}; both shapes work."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "data": {
                "name": "prsm-bob",
                "address": "0x" + "cd" * 20,
            }
        })

    pem = _generate_test_pem()
    backend = CdpWaaSBackend(
        api_key_name="kid",
        api_key_private_pem=pem,
        client=_mock_transport(handler),
    )
    r = backend.create_wallet("bob", "b@x.io")
    assert r["address"] == "0x" + "cd" * 20


def test_create_wallet_raises_when_no_address():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"name": "no-addr"})

    pem = _generate_test_pem()
    backend = CdpWaaSBackend(
        api_key_name="kid", api_key_private_pem=pem,
        client=_mock_transport(handler),
    )
    with pytest.raises(RuntimeError) as exc:
        backend.create_wallet("alice", "a@x.io")
    assert "no address" in str(exc.value).lower()


def test_create_wallet_raises_on_http_401():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad jwt"})

    pem = _generate_test_pem()
    backend = CdpWaaSBackend(
        api_key_name="kid", api_key_private_pem=pem,
        client=_mock_transport(handler),
    )
    with pytest.raises(httpx.HTTPStatusError):
        backend.create_wallet("alice", "a@x.io")


def test_constructor_rejects_empty_key_name():
    pem = _generate_test_pem()
    with pytest.raises(ValueError):
        CdpWaaSBackend(api_key_name="", api_key_private_pem=pem)


# ── cdp_waas_from_env ────────────────────────────────────────

def test_cdp_waas_from_env_returns_none_when_key_name_missing(
    monkeypatch,
):
    monkeypatch.delenv("COINBASE_CDP_API_KEY_NAME", raising=False)
    monkeypatch.setenv("COINBASE_CDP_API_KEY_PRIVATE", "x")
    assert cdp_waas_from_env() is None


def test_cdp_waas_from_env_returns_none_when_key_priv_missing(
    monkeypatch,
):
    monkeypatch.setenv("COINBASE_CDP_API_KEY_NAME", "kid")
    monkeypatch.delenv("COINBASE_CDP_API_KEY_PRIVATE", raising=False)
    assert cdp_waas_from_env() is None


def test_cdp_waas_from_env_returns_none_on_placeholder_pem(
    monkeypatch,
):
    """Honest signal: placeholder PEM → backend not constructed →
    adapter_wired stays False until operator pastes real key."""
    monkeypatch.setenv("COINBASE_CDP_API_KEY_NAME", "kid")
    monkeypatch.setenv(
        "COINBASE_CDP_API_KEY_PRIVATE",
        "REPLACE_WITH_ED25519_PEM_FROM_CDP_DOWNLOAD",
    )
    assert cdp_waas_from_env() is None


def test_cdp_waas_from_env_constructs_with_real_pem(monkeypatch):
    pem = _generate_test_pem()
    monkeypatch.setenv("COINBASE_CDP_API_KEY_NAME", "kid")
    monkeypatch.setenv("COINBASE_CDP_API_KEY_PRIVATE", pem)
    backend = cdp_waas_from_env()
    assert backend is not None
    assert isinstance(backend, CdpWaaSBackend)


# ── CoinbaseWaaSClient.from_env auto-wire ────────────────────

def test_waas_from_env_auto_attaches_when_real_pem(monkeypatch):
    pem = _generate_test_pem()
    monkeypatch.setenv("COINBASE_CDP_API_KEY_NAME", "kid")
    monkeypatch.setenv("COINBASE_CDP_API_KEY_PRIVATE", pem)
    monkeypatch.delenv("PRSM_WAAS_STORE_DIR", raising=False)
    c = CoinbaseWaaSClient.from_env()
    assert c.is_commissioned() is True
    assert c.adapter_wired() is True


def test_waas_from_env_stays_unwired_with_placeholder_pem(
    monkeypatch,
):
    """The load-bearing dogfood UX guarantee: a half-completed
    .env (key_name set, PEM still placeholder) shows
    commissioned=True but adapter_wired=False — the honest signal
    operators need to know what's still missing."""
    monkeypatch.setenv("COINBASE_CDP_API_KEY_NAME", "kid")
    monkeypatch.setenv(
        "COINBASE_CDP_API_KEY_PRIVATE",
        "REPLACE_WITH_ED25519_PEM_FROM_CDP_DOWNLOAD",
    )
    monkeypatch.delenv("PRSM_WAAS_STORE_DIR", raising=False)
    c = CoinbaseWaaSClient.from_env()
    assert c.is_commissioned() is True
    assert c.adapter_wired() is False


def test_waas_from_env_respects_explicit_backend(monkeypatch):
    pem = _generate_test_pem()
    monkeypatch.setenv("COINBASE_CDP_API_KEY_NAME", "kid")
    monkeypatch.setenv("COINBASE_CDP_API_KEY_PRIVATE", pem)
    monkeypatch.delenv("PRSM_WAAS_STORE_DIR", raising=False)

    class _Fake:
        def create_wallet(self, u, e):
            return {
                "wallet_id": "f", "address": "0x" + "00" * 20,
                "network": "base-mainnet",
            }

    fake = _Fake()
    c = CoinbaseWaaSClient.from_env(backend=fake)
    assert c._backend is fake


def test_waas_from_env_graceful_fallback_on_import_failure(
    monkeypatch,
):
    pem = _generate_test_pem()
    monkeypatch.setenv("COINBASE_CDP_API_KEY_NAME", "kid")
    monkeypatch.setenv("COINBASE_CDP_API_KEY_PRIVATE", pem)
    monkeypatch.delenv("PRSM_WAAS_STORE_DIR", raising=False)
    with patch(
        "prsm.economy.web3.coinbase_waas_cdp_backend.from_env",
        side_effect=RuntimeError("boom"),
    ):
        c = CoinbaseWaaSClient.from_env()
    assert c.is_commissioned() is True
    assert c.adapter_wired() is False
