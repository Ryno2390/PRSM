"""Sprint 855 (test backfill) — Coinbase Onramp session-token client.

`CoinbaseOnrampSessionClient` mints Coinbase Pay v2 secure-init session
tokens for the `/wallet/onramp/execute` path (api.py imports `from_env` +
calls `build_buy_url`). The client shipped without tests AND was never
committed — so a fresh checkout silently broke onramp-execute. This
backfills coverage on a fiat-money-path component (injected HTTP client +
a real generated Ed25519 key; no network, no secrets) and the file is now
committed alongside.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.web3.coinbase_onramp_session import (
    CoinbaseOnrampSessionClient,
    from_env,
)


# ── helpers ──────────────────────────────────────────────


def _test_pem() -> str:
    """A real (throwaway) Ed25519 private key in PKCS8 PEM form."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives import serialization
    key = Ed25519PrivateKey.generate()
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


def _fake_client(token="tok-abc", channel_id="chan-1", json_override=None):
    """A stand-in httpx.Client whose .post returns a scripted response."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(
        return_value=json_override
        if json_override is not None
        else {"token": token, "channel_id": channel_id}
    )
    client = MagicMock()
    client.post = MagicMock(return_value=resp)
    return client


def _client():
    return CoinbaseOnrampSessionClient(
        api_key_name="organizations/x/apiKeys/y",
        api_key_private_pem=_test_pem(),
        client=_fake_client(),
    )


# ── from_env ─────────────────────────────────────────────


def test_from_env_none_when_keys_missing(monkeypatch):
    monkeypatch.delenv("COINBASE_CDP_API_KEY_NAME", raising=False)
    monkeypatch.delenv("COINBASE_CDP_API_KEY_PRIVATE", raising=False)
    assert from_env() is None


def test_from_env_constructs_with_explicit_args():
    c = from_env(
        api_key_name="organizations/x/apiKeys/y",
        api_key_private_pem=_test_pem(),
        client=_fake_client(),
    )
    assert isinstance(c, CoinbaseOnrampSessionClient)


def test_init_rejects_empty_api_key_name():
    with pytest.raises(ValueError, match="api_key_name is required"):
        CoinbaseOnrampSessionClient(
            api_key_name="",
            api_key_private_pem=_test_pem(),
            client=_fake_client(),
        )


# ── mint_session_token ───────────────────────────────────


def test_mint_session_token_happy_path():
    c = _client()
    out = c.mint_session_token(address="0x" + "ab" * 20)
    assert out["token"] == "tok-abc"
    assert out["channel_id"] == "chan-1"
    # Body defaults to the Base chain + a Bearer JWT is attached.
    _, kwargs = c._client.post.call_args
    assert kwargs["json"]["addresses"][0]["blockchains"] == ["base"]
    assert kwargs["headers"]["Authorization"].startswith("Bearer ")


def test_mint_session_token_rejects_non_evm_address():
    c = _client()
    with pytest.raises(ValueError, match="0x EVM"):
        c.mint_session_token(address="bc1qnot-evm")


def test_mint_session_token_raises_when_no_token_in_response():
    c = CoinbaseOnrampSessionClient(
        api_key_name="organizations/x/apiKeys/y",
        api_key_private_pem=_test_pem(),
        client=_fake_client(json_override={"channel_id": "c"}),
    )
    with pytest.raises(RuntimeError, match="no token"):
        c.mint_session_token(address="0x" + "ab" * 20)


def test_mint_session_token_includes_client_ip_when_given():
    c = _client()
    c.mint_session_token(address="0x" + "ab" * 20, client_ip="203.0.113.7")
    _, kwargs = c._client.post.call_args
    assert kwargs["json"]["clientIp"] == "203.0.113.7"


# ── build_buy_url ────────────────────────────────────────


def test_build_buy_url_embeds_session_token():
    c = _client()
    out = c.build_buy_url(address="0x" + "cd" * 20)
    assert out["token"] == "tok-abc"
    assert "sessionToken=tok-abc" in out["session_url"]
    assert out["session_url"].startswith("https://pay.coinbase.com/buy")


def test_build_buy_url_threads_preset_amount_and_refs():
    c = _client()
    out = c.build_buy_url(
        address="0x" + "cd" * 20,
        usd_amount=100.0,
        partner_user_ref="user-42",
        redirect_url="https://app.example/done",
    )
    url = out["session_url"]
    assert "presetFiatAmount=100.0" in url
    assert "partnerUserRef=user-42" in url
    assert "redirectUrl=" in url


def test_build_buy_url_rejects_nonpositive_amount():
    c = _client()
    with pytest.raises(ValueError, match="usd_amount must be > 0"):
        c.build_buy_url(address="0x" + "cd" * 20, usd_amount=0)


# ── JWT shape ────────────────────────────────────────────


def test_build_jwt_is_eddsa_with_onramp_audience():
    import jwt as _jwt
    c = _client()
    token = c._build_jwt("POST", "/onramp/v1/token")
    header = _jwt.get_unverified_header(token)
    assert header["alg"] == "EdDSA"
    assert header["kid"] == "organizations/x/apiKeys/y"
    claims = _jwt.decode(token, options={"verify_signature": False})
    assert claims["sub"] == "organizations/x/apiKeys/y"
    assert "api.developer.coinbase.com" in claims["aud"]
    assert claims["uri"] == "POST api.developer.coinbase.com/onramp/v1/token"
