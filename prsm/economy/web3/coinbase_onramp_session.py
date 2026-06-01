"""Sprint 855 — Coinbase Onramp secure-init session token client.

Coinbase Pay v2 (the widget URL flow) requires a server-side session
token for projects with secure-init enabled. The legacy v1 widget URL
(with `addresses` / `assets` as public query params) is rejected with
"Missing or invalid parameters / requires sessionToken" once the
project flips on secure init — which is the default for new CDP
projects as of 2026.

Spec: docs.cdp.coinbase.com/onramp/introduction/quickstart

  POST https://api.developer.coinbase.com/onramp/v1/token
  Authorization: Bearer <EdDSA JWT signed with CDP API key>
  Body: {"addresses": [{"address": "...", "blockchains": ["base"]}],
         "clientIp": "<optional>"}
  Response: {"token": "<5-min single-use>", "channel_id": "<corr>"}

  Widget URL becomes:
  https://pay.coinbase.com/buy/select-asset?sessionToken=<token>

Token expiry is 5 minutes + single-use, so we mint a fresh one per
``/wallet/onramp/execute`` call rather than caching.

JWT signing reuses the same Ed25519 API-key plumbing CDP Paymaster
+ WaaS use (operator's COINBASE_CDP_API_KEY_NAME + PRIVATE). No new
operator-side env vars.
"""
from __future__ import annotations

import logging
import os
import secrets as _secrets
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


_CDP_ONRAMP_HOST = "api.developer.coinbase.com"
_CDP_ONRAMP_URL = f"https://{_CDP_ONRAMP_HOST}/onramp/v1/token"
_PAY_WIDGET_BUY = "https://pay.coinbase.com/buy/select-asset"
_PAY_WIDGET_SELL = "https://pay.coinbase.com/v3/sell/input"
_JWT_EXP_SECONDS = 120
_HTTP_TIMEOUT = 30.0


class CoinbaseOnrampSessionClient:
    """Mints sessionTokens for the Coinbase Pay widget.

    Constructed once per node; each `mint_buy_url` / `mint_sell_url`
    call hits CDP to get a fresh 5-minute token + returns the
    user-facing widget URL.
    """

    def __init__(
        self,
        api_key_name: str,
        api_key_private_pem: str,
        *,
        client: Any = None,
    ) -> None:
        if not api_key_name:
            raise ValueError("api_key_name is required")
        # Import here to keep module load cheap when onramp isn't used.
        from prsm.economy.web3.coinbase_waas_cdp_backend import (
            _load_ed25519_pem,
        )
        self._private_key = _load_ed25519_pem(api_key_private_pem)
        self._api_key_name = api_key_name
        if client is None:
            import httpx
            self._client = httpx.Client(timeout=_HTTP_TIMEOUT)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _build_jwt(self, method: str, path: str) -> str:
        """EdDSA Bearer JWT — same shape as paymaster/waas API
        auth, just pointed at the onramp host."""
        import jwt as _jwt
        now = int(time.time())
        headers = {
            "alg": "EdDSA",
            "typ": "JWT",
            "kid": self._api_key_name,
            "nonce": _secrets.token_hex(8),
        }
        payload = {
            "iss": "cdp",
            "sub": self._api_key_name,
            "nbf": now,
            "exp": now + _JWT_EXP_SECONDS,
            "aud": [method, _CDP_ONRAMP_HOST],
            "uri": f"{method} {_CDP_ONRAMP_HOST}{path}",
        }
        return _jwt.encode(
            payload, self._private_key,
            algorithm="EdDSA", headers=headers,
        )

    def mint_session_token(
        self,
        address: str,
        blockchains: Optional[List[str]] = None,
        *,
        client_ip: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST to /onramp/v1/token, return {token, channel_id}.

        ``address`` must be 0x EVM. ``blockchains`` defaults to
        ``["base"]`` since PRSM operates exclusively on Base mainnet.
        """
        if not address or not address.startswith("0x"):
            raise ValueError(
                f"address must be 0x EVM, got {address!r}",
            )
        body: Dict[str, Any] = {
            "addresses": [{
                "address": address,
                "blockchains": blockchains or ["base"],
            }],
        }
        if client_ip:
            body["clientIp"] = client_ip
        token_jwt = self._build_jwt("POST", "/onramp/v1/token")
        resp = self._client.post(
            _CDP_ONRAMP_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {token_jwt}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        token = payload.get("token")
        if not token:
            raise RuntimeError(
                f"CDP /onramp/v1/token returned no token: {payload!r}",
            )
        return {
            "token": token,
            "channel_id": payload.get("channel_id", ""),
        }

    def build_buy_url(
        self,
        address: str,
        *,
        usd_amount: Optional[float] = None,
        partner_user_ref: Optional[str] = None,
        redirect_url: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mint a session token + return the user-facing buy URL.

        Returns {"session_url": str, "token": str, "channel_id": str}.
        The URL is opened by the user in a browser; channel_id is
        useful for webhook correlation.
        """
        if usd_amount is not None and usd_amount <= 0:
            raise ValueError(
                "usd_amount must be > 0 when supplied",
            )
        minted = self.mint_session_token(
            address=address, client_ip=client_ip,
        )
        from urllib.parse import urlencode
        params: Dict[str, str] = {"sessionToken": minted["token"]}
        if usd_amount is not None:
            params["presetFiatAmount"] = str(usd_amount)
        if partner_user_ref:
            params["partnerUserRef"] = partner_user_ref
        if redirect_url:
            params["redirectUrl"] = redirect_url
        return {
            "session_url": f"{_PAY_WIDGET_BUY}?{urlencode(params)}",
            "token": minted["token"],
            "channel_id": minted["channel_id"],
        }


def from_env(
    *,
    api_key_name: Optional[str] = None,
    api_key_private_pem: Optional[str] = None,
    client: Any = None,
) -> Optional["CoinbaseOnrampSessionClient"]:
    """Construct from CDP env or None when keys missing."""
    api_key_name = (
        api_key_name or os.environ.get("COINBASE_CDP_API_KEY_NAME")
    )
    api_key_private_pem = (
        api_key_private_pem
        or os.environ.get("COINBASE_CDP_API_KEY_PRIVATE")
    )
    if not api_key_name or not api_key_private_pem:
        return None
    try:
        return CoinbaseOnrampSessionClient(
            api_key_name=api_key_name,
            api_key_private_pem=api_key_private_pem,
            client=client,
        )
    except ValueError as exc:
        logger.info(
            "CoinbaseOnrampSessionClient not constructed: %s", exc,
        )
        return None
