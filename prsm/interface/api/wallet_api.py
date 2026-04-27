"""Phase 4 Wallet onboarding HTTP API.

Per `docs/2026-04-22-phase4-wallet-sdk-design-plan.md` §5.2 — exposes the
shipped Phase 4 backend modules (SIWE verifier Task 1, wallet binding
Task 2, USD display Task 5) as HTTP endpoints so frontends can drive the
onboarding flow:

    POST /api/v1/auth/wallet/siwe/nonce      — issue a single-use SIWE nonce
    POST /api/v1/auth/wallet/siwe/verify     — verify SIWE message + signature,
                                                resolve (or generate) node_id,
                                                return the binding-attestation
                                                message the wallet should sign next
    POST /api/v1/auth/wallet/bind            — record the wallet ↔ node_id binding
                                                attested by an EIP-191 signature
    GET  /api/v1/auth/wallet/binding         — look up the binding for a wallet
                                                (for re-login flows)
    GET  /api/v1/auth/wallet/balance         — formatted FTNS balance for a
                                                bound wallet (USD or FTNS mode)

Design notes:
  - Services are injected via FastAPI ``Depends`` (overridable by tests).
  - Errors are mapped from the backend exception taxonomy to stable
    string codes so frontends can switch on them without parsing
    free-text messages.
  - The balance endpoint accepts a ``BalanceLookup`` Protocol that
    callers wire to their FTNS state source (oracle, on-chain reader,
    etc.). The default implementation returns zero — operators must
    register a real lookup before the endpoint is useful in production.
  - Binding endpoint is idempotent — re-bind with the same (wallet,
    node_id, signature) returns the existing binding unchanged. This
    matches the underlying ``WalletBindingService.bind`` semantics.

Phase 4 Task 4 (embedded-wallet vendor) is gated on the §6 G1-G4
Foundation operational items (see vendor-decision memo) and does NOT
land here. That task adds the vendor-specific session-token issuance
on top of these endpoints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Protocol

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from prsm.interface.display import (
    DisplayMode,
    StaticPriceSource,
    UsdPriceSource,
    format_balance,
    ftns_to_usd,
)
from prsm.interface.onboarding.siwe import (
    InMemoryNonceStore,
    NonceStore,
    SiweChainIdError,
    SiweDomainError,
    SiweError,
    SiweExpiredError,
    SiweMalformedError,
    SiweNonceError,
    SiweNotYetValidError,
    SiweSignatureError,
)
from prsm.interface.onboarding.siwe import (
    verify as siwe_verify,
)
from prsm.interface.onboarding.wallet_binding import (
    BindingConflictError,
    BindingError,
    BindingSignatureError,
    InMemoryWalletBindingStore,
    WalletBindingService,
    build_binding_message,
)

# ──────────────────────────────────────────────────────────────────────────
# Settings + service container
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WalletApiSettings:
    """Server-side SIWE invariants. Pinned at app construction.

    These MUST match what the frontend constructs into its SIWE message.
    A mismatch is intentional — verify() rejects the message before
    consuming the nonce, so a misconfigured frontend can't burn nonces."""

    expected_domain: str
    expected_chain_id: int
    nonce_ttl_seconds: int = 300


class BalanceLookup(Protocol):
    """Resolves a wallet address → FTNS balance (raw token units as Decimal).

    Production wiring: an FTNS oracle / on-chain reader. Phase 4 ships
    only the contract; actual wiring is operator-side.
    """

    def get_ftns_balance(self, wallet_address: str) -> Decimal: ...


class _ZeroBalanceLookup:
    """Default lookup — returns 0. Safe placeholder until operators
    wire a real source. Logs nothing; returning 0 is documented behavior."""

    def get_ftns_balance(self, wallet_address: str) -> Decimal:  # noqa: ARG002
        # Placeholder ignores wallet_address by design — the Protocol
        # signature requires it, but the zero-balance default doesn't
        # vary by caller. Operators wire a real lookup in production.
        return Decimal("0")


@dataclass
class WalletApiServices:
    """Injectable services for wallet_api routes.

    Tests build their own instance; production builds via
    ``WalletApiServices.default_for_dev`` (or a real production-grade
    builder for prod deployments)."""

    settings: WalletApiSettings
    nonce_store: NonceStore
    binding_service: WalletBindingService
    price_source: UsdPriceSource
    balance_lookup: BalanceLookup

    @classmethod
    def default_for_dev(
        cls,
        *,
        expected_domain: str,
        expected_chain_id: int = 8453,
    ) -> WalletApiServices:
        """In-memory defaults — DEV / TEST ONLY.

        Returns services backed by ``InMemoryNonceStore`` +
        ``InMemoryWalletBindingStore``. These break under multi-worker
        deployments (uvicorn ``--workers > 1`` or gunicorn): each
        worker has its own ``_nonces`` dict, so a nonce issued by one
        worker fails ``consume()`` on another with a confusing
        ``siwe_nonce_invalid_or_consumed`` error.

        Production deployments MUST build their own ``WalletApiServices``
        with shared backends:
          - Redis (or equivalent TTL store) for ``NonceStore``
          - SQLite (single-writer) or Postgres for ``WalletBindingStore``
          - A real oracle price source + balance lookup wired to FTNS
            state.
        Register at startup via ``set_services``.
        """
        return cls(
            settings=WalletApiSettings(
                expected_domain=expected_domain,
                expected_chain_id=expected_chain_id,
            ),
            nonce_store=InMemoryNonceStore(),
            binding_service=WalletBindingService(InMemoryWalletBindingStore()),
            price_source=StaticPriceSource(price_usd=Decimal("0")),
            balance_lookup=_ZeroBalanceLookup(),
        )


# Module-level injection slot. ``set_services`` is the boot-time hook;
# ``get_services`` is the FastAPI Depends source. Tests override via
# ``app.dependency_overrides[get_services] = ...``.
_services: Optional[WalletApiServices] = None


def set_services(services: WalletApiServices) -> None:
    """Bind the wallet_api services. Call ONCE at app startup.

    Re-calling on a non-None ``_services`` raises — the boot-time-only
    contract is enforced explicitly so a re-init script can't race
    with in-flight requests still holding references to the old
    services container. Tests that need to re-configure should call
    ``reset_services_for_tests`` first.
    """
    global _services
    if _services is not None:
        raise RuntimeError(
            "wallet_api services already configured; restart the process "
            "to swap. Tests: call reset_services_for_tests() first."
        )
    _services = services


def reset_services_for_tests() -> None:
    """Clear the services slot. Tests only — do not call from production
    code paths. Production should always build a fresh process for a
    services swap."""
    global _services
    _services = None


def get_services() -> WalletApiServices:
    if _services is None:
        raise RuntimeError(
            "wallet_api services not configured. Call "
            "wallet_api.set_services(...) at app startup, or override "
            "the get_services dependency in tests."
        )
    return _services


# ──────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────


class NonceRequest(BaseModel):
    """Optional client-supplied parameters. None required in v1 — a
    POST with empty body returns a fresh nonce under the server's
    configured chain_id."""

    chain_id: Optional[int] = Field(
        default=None,
        description=(
            "Optional client-stated chain_id (informational only — the "
            "server enforces its configured expected_chain_id at verify "
            "time). Mismatch surfaces as a friendly validation error here."
        ),
    )


class NonceResponse(BaseModel):
    nonce: str
    domain: str
    chain_id: int
    expires_at_unix: int


class SiweVerifyRequest(BaseModel):
    message: str = Field(..., description="EIP-4361 raw message text")
    signature: str = Field(
        ...,
        description="0x-prefixed hex signature from personal_sign",
    )


class SiweVerifyResponse(BaseModel):
    address: str
    node_id_hex: str
    is_new_user: bool
    binding_message: str = Field(
        ...,
        description=(
            "The exact message the wallet must EIP-191-sign next to "
            "complete onboarding. Pass back to /wallet/bind."
        ),
    )
    binding_issued_at: str = Field(
        ...,
        description=(
            "ISO-8601 UTC timestamp baked into the binding_message. "
            "Pass back unchanged to /wallet/bind so the server can "
            "reconstruct the same canonical message."
        ),
    )


class WalletBindRequest(BaseModel):
    wallet_address: str
    node_id_hex: str
    signature: str = Field(
        ...,
        description="0x-prefixed hex signature over the binding_message",
    )
    issued_at: str = Field(
        ...,
        description=(
            "Same ISO-8601 timestamp returned from /siwe/verify. "
            "Server reconstructs the binding_message from "
            "(wallet_address, node_id_hex, issued_at)."
        ),
    )


class WalletBindResponse(BaseModel):
    wallet_address: str
    node_id_hex: str
    bound_at_unix: int
    signing_message_hash: str


class BalanceResponse(BaseModel):
    wallet_address: str
    node_id_hex: str
    ftns: str
    usd: Optional[str]
    formatted: str
    mode: str


# ──────────────────────────────────────────────────────────────────────────
# Error mapping helpers
# ──────────────────────────────────────────────────────────────────────────


# Stable, frontend-switchable error codes. Free-text messages may evolve;
# these strings are part of the API contract.
SIWE_ERROR_CODES = {
    SiweMalformedError: "siwe_malformed",
    SiweSignatureError: "siwe_signature_invalid",
    SiweDomainError: "siwe_domain_mismatch",
    SiweChainIdError: "siwe_chain_id_mismatch",
    SiweExpiredError: "siwe_expired",
    SiweNotYetValidError: "siwe_not_yet_valid",
    SiweNonceError: "siwe_nonce_invalid_or_consumed",
}

BINDING_ERROR_CODES = {
    BindingSignatureError: "binding_signature_invalid",
    BindingConflictError: "binding_conflict",
}


def _siwe_http_error(exc: SiweError) -> HTTPException:
    """SIWE failures → 400 with stable error code. Nonce errors are
    intentionally a 400 (client-fixable: request a fresh nonce) rather
    than 401, so the frontend distinguishes 'bad request' from
    'unauthorized session'."""
    code = SIWE_ERROR_CODES.get(type(exc), "siwe_invalid")
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": code, "message": str(exc)},
    )


def _binding_http_error(exc: BindingError) -> HTTPException:
    if isinstance(exc, BindingConflictError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": BINDING_ERROR_CODES[BindingConflictError],
                "message": str(exc),
            },
        )
    code = BINDING_ERROR_CODES.get(type(exc), "binding_invalid")
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": code, "message": str(exc)},
    )


# ──────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────


router = APIRouter(prefix="/api/v1/auth/wallet", tags=["Wallet Onboarding"])


@router.post("/siwe/nonce", response_model=NonceResponse)
def issue_nonce(
    body: NonceRequest = NonceRequest(),
    services: WalletApiServices = Depends(get_services),
) -> NonceResponse:
    """Issue a single-use SIWE nonce.

    The nonce is bound to the server's configured chain_id at verify
    time; the client builds the EIP-4361 message and submits it to
    /siwe/verify. The nonce is consumed only on a successful verify;
    failed verifies leave the nonce live for re-submission."""
    if (
        body.chain_id is not None
        and body.chain_id != services.settings.expected_chain_id
    ):
        # Surface the mismatch eagerly — saves the client a round-trip.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "siwe_chain_id_mismatch",
                "message": (
                    f"client requested chain_id={body.chain_id}; server "
                    f"expects {services.settings.expected_chain_id}"
                ),
            },
        )
    nonce = services.nonce_store.issue(services.settings.nonce_ttl_seconds)
    return NonceResponse(
        nonce=nonce,
        domain=services.settings.expected_domain,
        chain_id=services.settings.expected_chain_id,
        expires_at_unix=int(time.time()) + services.settings.nonce_ttl_seconds,
    )


@router.post("/siwe/verify", response_model=SiweVerifyResponse)
def verify_siwe(
    body: SiweVerifyRequest,
    services: WalletApiServices = Depends(get_services),
) -> SiweVerifyResponse:
    """Verify an EIP-4361 SIWE message + signature.

    On success: resolve (or freshly generate) the user's node_id and
    return the canonical binding-attestation message the wallet must
    sign next. The frontend captures the returned (binding_message,
    binding_issued_at) and submits the next signature to /wallet/bind.
    """
    try:
        verified = siwe_verify(
            body.message,
            body.signature,
            expected_domain=services.settings.expected_domain,
            expected_chain_id=services.settings.expected_chain_id,
            nonce_store=services.nonce_store,
        )
    except SiweError as exc:
        raise _siwe_http_error(exc) from exc

    node_id_hex, is_new_user = services.binding_service.sign_in(
        verified.address
    )

    issued_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    binding_message = build_binding_message(
        verified.address, node_id_hex, issued_at
    )

    return SiweVerifyResponse(
        address=verified.address,
        node_id_hex=node_id_hex,
        is_new_user=is_new_user,
        binding_message=binding_message,
        binding_issued_at=issued_at,
    )


@router.post("/bind", response_model=WalletBindResponse)
def bind_wallet(
    body: WalletBindRequest,
    services: WalletApiServices = Depends(get_services),
) -> WalletBindResponse:
    """Record a wallet ↔ node_id binding from an EIP-191 attestation.

    Idempotent — re-binding with the same (wallet, node_id) returns
    the existing binding unchanged. A binding-conflict (wallet bound
    to a different node, or node bound to a different wallet) returns
    HTTP 409.
    """
    try:
        binding = services.binding_service.bind(
            wallet_address=body.wallet_address,
            node_id_hex=body.node_id_hex,
            signature=body.signature,
            issued_at_iso=body.issued_at,
        )
    except BindingError as exc:
        raise _binding_http_error(exc) from exc

    return WalletBindResponse(
        wallet_address=binding.wallet_address,
        node_id_hex=binding.node_id_hex,
        bound_at_unix=binding.bound_at_unix,
        signing_message_hash=binding.signing_message_hash,
    )


@router.get("/binding", response_model=Optional[WalletBindResponse])
def get_binding(
    wallet_address: str,
    services: WalletApiServices = Depends(get_services),
) -> Optional[WalletBindResponse]:
    """Look up an existing binding by wallet address. Returns null if
    no binding exists. Used for returning-user re-login flows where
    the frontend wants to skip /siwe/verify if the wallet is already
    known."""
    binding = services.binding_service.get_by_wallet(wallet_address)
    if binding is None:
        return None
    return WalletBindResponse(
        wallet_address=binding.wallet_address,
        node_id_hex=binding.node_id_hex,
        bound_at_unix=binding.bound_at_unix,
        signing_message_hash=binding.signing_message_hash,
    )


@router.get("/balance", response_model=BalanceResponse)
def get_balance(
    wallet_address: str,
    mode: DisplayMode = "usd",
    services: WalletApiServices = Depends(get_services),
) -> BalanceResponse:
    """Formatted FTNS balance for a bound wallet.

    Resolves wallet_address → binding → node_id → balance via the
    injected ``BalanceLookup``. USD value uses the injected price
    source. Returns 404 if the wallet is not bound — frontends should
    drive the user through /siwe/verify + /bind first.
    """
    if mode not in ("usd", "ftns"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_display_mode",
                "message": f"mode must be 'usd' or 'ftns', got {mode!r}",
            },
        )

    binding = services.binding_service.get_by_wallet(wallet_address)
    if binding is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "wallet_not_bound",
                "message": (
                    f"no binding for wallet {wallet_address}; complete "
                    f"/siwe/verify + /bind first"
                ),
            },
        )

    ftns_amount = services.balance_lookup.get_ftns_balance(
        binding.wallet_address
    )
    formatted = format_balance(
        ftns_amount, services.price_source, mode=mode
    )

    if mode == "usd":  # noqa: SIM108  (keep block; the comment is load-bearing)
        # Route through ftns_to_usd so the quantization (ROUND_HALF_EVEN
        # to cents) matches what format_balance computed — keeps the
        # `usd` field byte-equal to the cent value embedded in
        # `formatted` rather than recomputing with truncating ``%.2f``.
        usd_str = str(
            ftns_to_usd(ftns_amount, services.price_source)
        )
    else:
        usd_str = None

    return BalanceResponse(
        wallet_address=binding.wallet_address,
        node_id_hex=binding.node_id_hex,
        ftns=str(ftns_amount),
        usd=usd_str,
        formatted=formatted,
        mode=mode,
    )
