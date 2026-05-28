"""Sprint 853 — Coinbase Onramp widget URL generator.

Builds the user-facing Coinbase Pay/Onramp URL that completes
``USD → USDC`` purchase. Returned by ``/wallet/onramp/execute`` so a
user opens the URL, completes payment on Coinbase, and USDC lands
in their WaaS wallet on Base mainnet. From there a separate
swap step (Aerodrome) converts USDC → FTNS — wired in a
sibling sprint when the pool seeding ceremony closes.

URL format (Coinbase Pay v1 public widget):

  https://pay.coinbase.com/buy/select-asset
    ?appId=<COINBASE_PAY_APP_ID>
    &addresses={"<addr>":["base"]}
    &assets=["USDC"]
    &defaultAsset=USDC
    &defaultNetwork=base
    &presetCryptoAmount=<n>           (optional — USDC units)
    &presetFiatAmount=<n>             (optional — USD)
    &fiatCurrency=USD
    &destinationWallets=[{"address":"...", "blockchains":["base"]}]
    &partnerUserId=<reference id>     (optional — webhook correlation)

The ``appId`` is a public-facing identifier, NOT the CDP API secret —
no server-side signing needed. Operators register a Pay project at
https://portal.cdp.coinbase.com/products/onramp and export the App ID
as ``COINBASE_PAY_APP_ID``.

When ``COINBASE_PAY_APP_ID`` is unset, ``build_onramp_url`` returns
None so the calling endpoint can surface PENDING_COMMISSION with
actionable guidance — same honest-signal pattern as the rest of
Phase 5.
"""
from __future__ import annotations

import json
import os
from typing import Optional
from urllib.parse import urlencode


_COINBASE_ONRAMP_BASE = "https://pay.coinbase.com/buy/select-asset"
_DEFAULT_ASSET = "USDC"
_DEFAULT_NETWORK = "base"


def build_onramp_url(
    *,
    destination_address: str,
    usd_amount: Optional[float] = None,
    usdc_amount: Optional[float] = None,
    partner_user_id: Optional[str] = None,
    app_id: Optional[str] = None,
    asset: str = _DEFAULT_ASSET,
    network: str = _DEFAULT_NETWORK,
) -> Optional[str]:
    """Construct a Coinbase Onramp widget URL.

    Returns None when ``COINBASE_PAY_APP_ID`` env var is unset AND no
    explicit ``app_id`` was passed — operator must register a Pay
    project first.

    Either ``usd_amount`` (preset fiat) or ``usdc_amount`` (preset
    crypto) may be supplied; both is allowed (Coinbase will reconcile
    via the live rate). Neither is also valid — the user picks an
    amount themselves on the Coinbase widget.
    """
    if not destination_address:
        raise ValueError("destination_address is required")
    if not destination_address.startswith("0x"):
        raise ValueError(
            f"destination_address must be a 0x EVM address, got "
            f"{destination_address!r}"
        )
    if usd_amount is not None and usd_amount <= 0:
        raise ValueError("usd_amount must be > 0 when supplied")
    if usdc_amount is not None and usdc_amount <= 0:
        raise ValueError("usdc_amount must be > 0 when supplied")

    app_id = app_id or os.environ.get("COINBASE_PAY_APP_ID")
    if not app_id:
        return None

    # Coinbase Pay encodes addresses as a JSON object whose values
    # are arrays of supported blockchain ids per the v1 spec.
    addresses = {destination_address: [network]}
    destination_wallets = [
        {
            "address": destination_address,
            "blockchains": [network],
            "assets": [asset],
        }
    ]

    params = {
        "appId": app_id,
        "addresses": json.dumps(addresses, separators=(",", ":")),
        "assets": json.dumps([asset], separators=(",", ":")),
        "defaultAsset": asset,
        "defaultNetwork": network,
        "destinationWallets": json.dumps(
            destination_wallets, separators=(",", ":"),
        ),
        "fiatCurrency": "USD",
    }
    if usd_amount is not None:
        params["presetFiatAmount"] = str(usd_amount)
    if usdc_amount is not None:
        params["presetCryptoAmount"] = str(usdc_amount)
    if partner_user_id:
        params["partnerUserId"] = partner_user_id

    return f"{_COINBASE_ONRAMP_BASE}?{urlencode(params)}"


def app_id_commissioned() -> bool:
    """True iff COINBASE_PAY_APP_ID is set (operator can build URLs)."""
    return bool(os.environ.get("COINBASE_PAY_APP_ID"))
