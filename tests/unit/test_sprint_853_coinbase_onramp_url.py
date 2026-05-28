"""Sprint 853 — Coinbase Onramp widget URL generator pin tests.

Defends the URL construction shape + env-driven commissioning gate.
URL contents matter because the Coinbase Pay widget parses query
params strictly — wrong field names = silent failure on the user's
end.

Pin tests:
  - Returns None when COINBASE_PAY_APP_ID unset (PENDING_COMMISSION)
  - Returns URL when env set
  - URL starts with the canonical Coinbase Pay base
  - Includes appId, addresses, assets, defaultAsset, defaultNetwork
    in query params
  - addresses param is JSON: {"<addr>": ["base"]}
  - destinationWallets carries address + blockchains + assets list
  - presetFiatAmount written when usd_amount supplied
  - presetCryptoAmount written when usdc_amount supplied
  - Both amounts together are valid (Coinbase reconciles via rate)
  - Neither amount is valid (user picks on widget)
  - partner_user_id maps to partnerUserId for webhook correlation
  - Rejects empty / non-0x address
  - Rejects negative amounts
  - explicit app_id overrides env
"""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import pytest

from prsm.economy.web3.coinbase_onramp_url import (
    build_onramp_url,
    app_id_commissioned,
)


_TEST_ADDR = "0x" + "1a" * 20


def _parse(url: str) -> dict:
    """Pull query params off a Coinbase Pay URL into a dict."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    # parse_qs returns lists; flatten singletons.
    return {k: v[0] for k, v in qs.items()}


# ── PENDING_COMMISSION gate ──────────────────────────────────

def test_returns_none_when_app_id_missing(monkeypatch):
    monkeypatch.delenv("COINBASE_PAY_APP_ID", raising=False)
    assert build_onramp_url(
        destination_address=_TEST_ADDR, usd_amount=100,
    ) is None


def test_explicit_app_id_overrides_env(monkeypatch):
    monkeypatch.delenv("COINBASE_PAY_APP_ID", raising=False)
    url = build_onramp_url(
        destination_address=_TEST_ADDR, usd_amount=100,
        app_id="explicit-app-id-123",
    )
    assert url is not None
    qs = _parse(url)
    assert qs["appId"] == "explicit-app-id-123"


def test_app_id_commissioned_helper(monkeypatch):
    monkeypatch.delenv("COINBASE_PAY_APP_ID", raising=False)
    assert app_id_commissioned() is False
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "set")
    assert app_id_commissioned() is True


# ── URL shape ────────────────────────────────────────────────

def test_url_uses_canonical_coinbase_pay_base(monkeypatch):
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    url = build_onramp_url(
        destination_address=_TEST_ADDR, usd_amount=100,
    )
    assert url.startswith("https://pay.coinbase.com/buy/select-asset?")


def test_url_includes_required_params(monkeypatch):
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    url = build_onramp_url(
        destination_address=_TEST_ADDR, usd_amount=100,
    )
    qs = _parse(url)
    assert qs["appId"] == "app123"
    assert qs["defaultAsset"] == "USDC"
    assert qs["defaultNetwork"] == "base"
    assert qs["fiatCurrency"] == "USD"
    assert "addresses" in qs
    assert "destinationWallets" in qs


def test_addresses_param_shape(monkeypatch):
    """addresses is a JSON dict: {<addr>: [<network>]}."""
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    url = build_onramp_url(
        destination_address=_TEST_ADDR, usd_amount=100,
    )
    qs = _parse(url)
    parsed = json.loads(qs["addresses"])
    assert parsed == {_TEST_ADDR: ["base"]}


def test_destination_wallets_carries_metadata(monkeypatch):
    """destinationWallets carries the full address + blockchains
    + assets metadata Coinbase's newer flow needs."""
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    url = build_onramp_url(
        destination_address=_TEST_ADDR, usd_amount=100,
    )
    qs = _parse(url)
    wallets = json.loads(qs["destinationWallets"])
    assert wallets == [
        {
            "address": _TEST_ADDR,
            "blockchains": ["base"],
            "assets": ["USDC"],
        }
    ]


def test_assets_param_shape(monkeypatch):
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    url = build_onramp_url(
        destination_address=_TEST_ADDR, usd_amount=100,
    )
    qs = _parse(url)
    assert json.loads(qs["assets"]) == ["USDC"]


# ── Preset amounts ───────────────────────────────────────────

def test_preset_fiat_amount_when_usd_supplied(monkeypatch):
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    url = build_onramp_url(
        destination_address=_TEST_ADDR, usd_amount=250,
    )
    qs = _parse(url)
    assert qs["presetFiatAmount"] == "250"
    assert "presetCryptoAmount" not in qs


def test_preset_crypto_amount_when_usdc_supplied(monkeypatch):
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    url = build_onramp_url(
        destination_address=_TEST_ADDR, usdc_amount=42.5,
    )
    qs = _parse(url)
    assert qs["presetCryptoAmount"] == "42.5"
    assert "presetFiatAmount" not in qs


def test_neither_amount_is_valid(monkeypatch):
    """User picks amount on the widget itself."""
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    url = build_onramp_url(destination_address=_TEST_ADDR)
    qs = _parse(url)
    assert "presetFiatAmount" not in qs
    assert "presetCryptoAmount" not in qs


def test_both_amounts_supplied_both_written(monkeypatch):
    """Coinbase reconciles to live rate; we don't gate on it."""
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    url = build_onramp_url(
        destination_address=_TEST_ADDR,
        usd_amount=100, usdc_amount=99.95,
    )
    qs = _parse(url)
    assert qs["presetFiatAmount"] == "100"
    assert qs["presetCryptoAmount"] == "99.95"


# ── Partner user id ──────────────────────────────────────────

def test_partner_user_id_maps_to_partnerUserId(monkeypatch):
    """Coinbase webhook fires with partnerUserId so we can
    correlate the buy event back to the originating PRSM user."""
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    url = build_onramp_url(
        destination_address=_TEST_ADDR, usd_amount=100,
        partner_user_id="alice_42",
    )
    qs = _parse(url)
    assert qs["partnerUserId"] == "alice_42"


# ── Validation ───────────────────────────────────────────────

def test_rejects_empty_address(monkeypatch):
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    with pytest.raises(ValueError):
        build_onramp_url(destination_address="", usd_amount=100)


def test_rejects_non_0x_address(monkeypatch):
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    with pytest.raises(ValueError) as exc:
        build_onramp_url(
            destination_address="alice.eth", usd_amount=100,
        )
    assert "0x" in str(exc.value)


def test_rejects_zero_usd(monkeypatch):
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    with pytest.raises(ValueError):
        build_onramp_url(
            destination_address=_TEST_ADDR, usd_amount=0,
        )


def test_rejects_negative_usdc(monkeypatch):
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    with pytest.raises(ValueError):
        build_onramp_url(
            destination_address=_TEST_ADDR, usdc_amount=-1,
        )


# ── Override asset/network ───────────────────────────────────

def test_custom_asset_and_network(monkeypatch):
    """Future-proofing: allow USDT on a different L2 if/when
    Coinbase Pay supports it. PRSM defaults to USDC on base."""
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "app123")
    url = build_onramp_url(
        destination_address=_TEST_ADDR, usd_amount=100,
        asset="USDT", network="optimism",
    )
    qs = _parse(url)
    assert qs["defaultAsset"] == "USDT"
    assert qs["defaultNetwork"] == "optimism"
    assert json.loads(qs["assets"]) == ["USDT"]
    assert json.loads(qs["addresses"]) == {
        _TEST_ADDR: ["optimism"],
    }
