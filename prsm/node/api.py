"""
Node Management API
===================

FastAPI endpoints for monitoring and controlling a running PRSM node.
This is the node-local API (not the main PRSM platform API).

Security Features (Phase 4.2):
- JWT authentication on protected endpoints
- Rate limiting with configurable limits
- WebSocket status updates
- OpenAPI specification
"""

import json
import logging
import os
import time as _time_for_history
import uuid as _uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from prsm.node.api_hardening import (
    APIHardening,
    APISecurityConfig,
    StatusWebSocket,
    require_auth,
)

logger = logging.getLogger(__name__)


class JobSubmission(BaseModel):
    """Request body for submitting a compute job."""
    job_type: str  # inference, embedding, benchmark
    payload: Dict[str, Any] = {}
    ftns_budget: float = 1.0


class ResourceUpdateRequest(BaseModel):
    """Request model for updating node resource settings."""
    cpu_allocation_pct: Optional[int] = Field(default=None, ge=10, le=90, description="CPU allocation percentage (10-90)")
    memory_allocation_pct: Optional[int] = Field(default=None, ge=10, le=90, description="Memory allocation percentage (10-90)")
    storage_gb: Optional[float] = Field(default=None, gt=0, description="Storage pledge in GB")
    max_concurrent_jobs: Optional[int] = Field(default=None, ge=1, description="Maximum concurrent jobs")
    gpu_allocation_pct: Optional[int] = Field(default=None, ge=10, le=100, description="GPU allocation percentage (10-100)")
    upload_mbps_limit: Optional[float] = Field(default=None, ge=0, description="Upload bandwidth limit in Mbps (0=unlimited)")
    download_mbps_limit: Optional[float] = Field(default=None, ge=0, description="Download bandwidth limit in Mbps (0=unlimited)")
    active_hours_start: Optional[int] = Field(default=None, ge=0, le=23, description="Active hours start (0-23)")
    active_hours_end: Optional[int] = Field(default=None, ge=0, le=23, description="Active hours end (0-23)")
    active_days: Optional[List[int]] = Field(default=None, description="Active days (0=Mon...6=Sun, empty=every day)")


class ResourceConfigResponse(BaseModel):
    """Response model for current resource configuration."""
    cpu_allocation_pct: int
    memory_allocation_pct: int
    storage_gb: float
    max_concurrent_jobs: int
    gpu_allocation_pct: int
    upload_mbps_limit: float
    download_mbps_limit: float
    active_hours_start: Optional[int]
    active_hours_end: Optional[int]
    active_days: List[int]
    # Computed values
    effective_cpu_cores: float
    effective_memory_gb: float
    effective_gpu_memory_gb: Optional[float]
    storage_available_gb: float


class ContentUploadRequest(BaseModel):
    """Request body for uploading text content."""
    text: str
    filename: str = "document.txt"
    replicas: int = 3
    royalty_rate: Optional[float] = Field(
        default=None,
        description="FTNS earned per access (0.001–0.1, default 0.01)",
    )
    parent_cids: List[str] = Field(
        default_factory=list,
        description="CIDs of source material this content derives from",
    )



# ──────────────────────────────────────────────────────────────────────
# Phase 3.x.8.1 — SSE helpers for /compute/inference/stream
# ──────────────────────────────────────────────────────────────────────


def _sse_event(event_type: str, data: Any) -> bytes:
    """Encode a single Server-Sent Events frame.

    Per the W3C SSE spec: ``event:`` + ``data:`` lines, terminated by
    a blank line. ``data`` is JSON-serialized; complex objects flow
    through ``json.dumps`` with ``default=str`` as a fallback for any
    Decimal / bytes / dataclass values that slip through the
    pre-encoders.
    """
    import json as _json

    if not isinstance(data, str):
        data = _json.dumps(data, default=str)
    return f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")


def _token_event_to_dict(event: Any) -> Dict[str, Any]:
    """Encode an ``InferenceTokenEvent`` into the SSE ``data`` payload
    shape. Optional fields (token_id / finish_reason) are emitted as
    ``null`` when absent so consumer parsers see a stable schema."""
    return {
        "sequence_index": event.sequence_index,
        "text_delta": event.text_delta,
        "token_id": event.token_id,
        "finish_reason": event.finish_reason,
    }


def _result_to_dict(
    result: Any,
    *,
    job_id: str,
    identity: Optional[Any] = None,
) -> Dict[str, Any]:
    """Encode an ``InferenceResult`` into the SSE ``data`` payload
    shape. Mirrors the unary endpoint's success response with the
    job_id rebound to the API-side id (executor uses an internal
    parallax-stream-job-* id; the API is authoritative for billing
    correlation).

    When ``identity`` is provided, the rebound receipt is re-signed
    under that identity. The ``job_id`` is part of the signed
    payload — rebinding without re-signing would invalidate the
    settler signature. Identity-less callers (tests / dry-run
    encoding helpers) get a receipt with a stale signature; callers
    receiving over the wire MUST pass identity to preserve the
    cryptographic invariant.
    """
    import dataclasses

    receipt = result.receipt
    if receipt is not None:
        receipt = dataclasses.replace(receipt, job_id=job_id)
        if identity is not None:
            try:
                from prsm.compute.inference import sign_receipt
                receipt = sign_receipt(receipt, identity)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Streaming result receipt re-sign failed for "
                    f"job_id={job_id}: {e}"
                )
    return {
        "success": result.success,
        "job_id": job_id,
        "request_id": result.request_id,
        "output": result.output,
        "receipt": receipt.to_dict() if receipt is not None else None,
    }


async def _settle_streaming_escrow(
    node: Any,
    job_id: str,
    escrow_entry: Any,
    request: Any,
    result: Any,
) -> None:
    """Settle escrow + record privacy-budget spend for a successful
    streaming inference.

    Phase 3.x.8.1 round-1 L1 remediation: the receipt re-sign
    previously done here was dead code — the wire-side re-sign in
    ``_result_to_dict(item, job_id=..., identity=node.identity)`` is
    the authoritative path. Re-signing here computed a value that
    was immediately discarded (the local ``rebound`` receipt object
    wasn't propagated anywhere). Removed.

    Privacy budget tracking still fires here on the success path —
    it's the right layer to record DP epsilon spend (the executor
    yielded a complete result with measured ``epsilon_spent``).
    """
    if not escrow_entry or not getattr(node, "_payment_escrow", None):
        return

    # Privacy budget tracking — same as unary path. Only fires on
    # the success path (a complete result with measured
    # epsilon_spent). Mid-stream failures don't produce a receipt,
    # so this code never sees them — they go through the
    # _resolve_post_token_billing settle-without-privacy-budget
    # path instead.
    if (
        hasattr(node, "privacy_budget")
        and node.privacy_budget
        and request.privacy_tier.value != "none"
        and result.receipt is not None
    ):
        try:
            node.privacy_budget.record_spend(
                result.receipt.epsilon_spent, "inference", job_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Streaming privacy budget tracking failed for job_id={job_id}: {e}"
            )

    try:
        await node._payment_escrow.release_escrow(
            job_id=job_id,
            provider_id=node.identity.node_id if node.identity else "self",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Streaming escrow release failed for job_id={job_id}: {e}"
        )


async def _resolve_post_token_billing(
    node: Any,
    job_id: str,
    escrow_entry: Any,
    tokens_emitted: int,
    reason: str,
) -> None:
    """Phase 3.x.8.1 round-1 M1 remediation: design plan §3.4
    settle-on-tokens-emitted policy.

    When the streaming pipeline fails AFTER one or more tokens have
    been emitted on the wire, the requester has consumed the
    network's compute work — the chain executors burned cycles on
    those tokens. Refunding the requester would mean the network
    eats the cost (billing griefing vector: a malicious node could
    emit N tokens then crash, paying nothing despite forcing real
    compute work).

    Policy: tokens emitted > 0 → settle (release escrow at full
    estimate, no privacy-budget recording — we don't have a
    complete receipt with measured epsilon for a partial stream).
    Tokens emitted == 0 → refund (pre-execute failure or
    immediately-failing dispatch — caller paid nothing for nothing).
    """
    if not escrow_entry or not getattr(node, "_payment_escrow", None):
        return
    if tokens_emitted > 0:
        try:
            await node._payment_escrow.release_escrow(
                job_id=job_id,
                provider_id=node.identity.node_id if node.identity else "self",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Streaming post-token escrow release failed for "
                f"job_id={job_id}: {e}"
            )
    else:
        try:
            await node._payment_escrow.refund_escrow(job_id, reason)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Streaming escrow refund failed for job_id={job_id}: {e}"
            )


async def _refund_streaming_escrow(
    node: Any,
    job_id: str,
    escrow_entry: Any,
    reason: str,
) -> None:
    """Refund escrow for a streaming inference that ended without a
    successful result. Swallows refund-time exceptions (logged) —
    they're non-actionable at the wire boundary."""
    if not escrow_entry or not getattr(node, "_payment_escrow", None):
        return
    try:
        await node._payment_escrow.refund_escrow(job_id, reason)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Streaming escrow refund failed for job_id={job_id}: {e}"
        )


def create_api_app(node: Any, enable_security: bool = True) -> FastAPI:
    """
    Create the node management FastAPI app with a reference to the running node.
    
    Args:
        node: The PRSM node instance
        enable_security: Whether to enable security hardening (default: True)
    
    Returns:
        FastAPI application with security hardening applied
    """

    app = FastAPI(
        title="PRSM Node API",
        description="Management API for a PRSM network node",
        version="0.24.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Initialize security hardening
    api_hardening: Optional[APIHardening] = None
    status_websocket: Optional[StatusWebSocket] = None
    
    if enable_security:
        # Create security configuration
        security_config = APISecurityConfig(
            enable_rate_limiting=True,
            enable_jwt_auth=False,
            enable_websocket=True,
            enable_openapi=True,
            rate_limit_requests_per_minute=100,
            rate_limit_requests_per_hour=1000,
        )
        
        # Initialize hardening
        api_hardening = APIHardening(app, security_config)
        
        # Apply middleware (must be done before adding routes)
        # Note: Middleware is applied in reverse order, so we add it after route definition
        # We'll apply it at the end of this function
        
        # Setup OpenAPI schema
        api_hardening.setup_openapi()
        
        # Get WebSocket manager for status updates
        status_websocket = api_hardening.get_status_websocket()

    # Store WebSocket manager in app state for access
    app.state.status_websocket = status_websocket
    app.state.api_hardening = api_hardening

    # ── Public Endpoints (no auth required) ─────────────────────────────────────

    @app.get("/api-info", tags=["status"])
    async def api_info() -> Dict[str, Any]:
        """API information endpoint (dashboard served at root)."""
        return {
            "name": "PRSM Node API",
            "version": "0.2.0",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "websocket": "/ws/status",
        }

    @app.get("/status", tags=["status"])
    async def get_status() -> Dict[str, Any]:
        """Get comprehensive node status."""
        status = await node.get_status()
        
        # Broadcast status update via WebSocket if available
        if app.state.status_websocket:
            await app.state.status_websocket.broadcast_status(status)
        
        return status

    @app.get("/rings/status", tags=["status"])
    async def rings_status() -> Dict[str, Any]:
        """Get Ring 1-10 initialization and health status."""
        from prsm.observability.dashboard_metrics import DashboardMetrics
        metrics = DashboardMetrics(node=node)
        return metrics.get_summary()

    @app.get("/peers")
    async def get_peers() -> Dict[str, Any]:
        """List connected and known peers."""
        connected = []
        if node.transport:
            for pid, peer in node.transport.peers.items():
                connected.append({
                    "peer_id": pid,
                    "address": peer.address,
                    "display_name": peer.display_name,
                    "connected_at": peer.connected_at,
                    "last_seen": peer.last_seen,
                    "outbound": peer.outbound,
                })

        known = []
        if node.discovery:
            for info in node.discovery.get_known_peers():
                known.append({
                    "node_id": info.node_id,
                    "address": info.address,
                    "display_name": info.display_name,
                    "last_seen": info.last_seen,
                })

        return {
            "connected": connected,
            "known": known,
            "connected_count": len(connected),
            "known_count": len(known),
        }

    @app.get("/balance")
    async def get_balance() -> Dict[str, Any]:
        """Get FTNS balance and recent transactions."""
        if not node.ledger or not node.identity:
            raise HTTPException(status_code=503, detail="Node not initialized")

        balance = await node.ledger.get_balance(node.identity.node_id)
        history = await node.ledger.get_transaction_history(node.identity.node_id, limit=20)

        return {
            "wallet_id": node.identity.node_id,
            "balance": balance,
            "recent_transactions": [
                {
                    "tx_id": tx.tx_id,
                    "type": tx.tx_type.value,
                    "from": tx.from_wallet,
                    "to": tx.to_wallet,
                    "amount": tx.amount,
                    "description": tx.description,
                    "timestamp": tx.timestamp,
                }
                for tx in history
            ],
        }

    @app.get("/balance/onchain")
    async def get_balance_onchain(
        address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read on-chain FTNS balance + USD equivalent.

        Backend for the prsm_balance_check MCP tool (v1 scope per
        Vision §13 Phase 5 stand-in closure). Reads via the node's
        already-initialized OnChainFTNSLedger; converts to USD using
        ``PRSM_FTNS_USD_RATE`` env var (default 1.0; placeholder
        until the Aerodrome USDC-FTNS pool is seeded per Vision
        gantt 2026-06-15).

        Query params:
            address: optional override; defaults to the ledger's
                connected address.
        """
        if not getattr(node, "ftns_ledger", None):
            raise HTTPException(
                status_code=503,
                detail=(
                    "On-chain ftns_ledger not initialized; "
                    "set PRSM_ONCHAIN_FTNS=1 + FTNS_TOKEN_ADDRESS "
                    "to enable."
                ),
            )

        target = address or node.ftns_ledger._connected_address
        balance_ftns = await node.ftns_ledger.get_balance(target)

        # USD-rate parsing — graceful fallback to 1.0 on misconfig.
        rate_raw = os.getenv("PRSM_FTNS_USD_RATE", "").strip()
        usd_rate = 1.0
        if rate_raw:
            try:
                parsed = float(rate_raw)
                if parsed > 0:
                    usd_rate = parsed
            except ValueError:
                pass  # keep default

        decimals = getattr(node.ftns_ledger, "_decimals", 18)
        balance_wei = int(balance_ftns * (10 ** decimals))
        usd_equivalent = balance_ftns * usd_rate

        # ── Aggregate-source quoting (audit-prep §7.23 honest-scope) ──
        # Read claimable royalties (RoyaltyDistributor) + escrowed
        # FTNS (PaymentEscrow). Each source independently fail-soft:
        # if reading raises, log + continue + report the source as
        # unavailable. Aggregate doesn't crash on partial sources.

        # Source 2: claimable royalties.
        claimable_ftns = 0.0
        claimable_available = False
        royalty_client = getattr(node, "_royalty_distributor_client", None)
        if royalty_client is not None:
            try:
                claimable_wei = royalty_client.claimable(target)
                claimable_ftns = float(claimable_wei) / (10 ** decimals)
                claimable_available = True
            except Exception as e:
                logger.warning(
                    "balance/onchain: claimable royalties read failed "
                    "(%s); reporting source unavailable", e,
                )

        # Source 3: escrowed FTNS in pending compute jobs.
        escrowed_ftns = 0.0
        escrowed_available = False
        payment_escrow = getattr(node, "_payment_escrow", None)
        if payment_escrow is not None:
            try:
                pending = payment_escrow.list_escrows_by_requester(target)
                escrowed_ftns = sum(e.amount for e in pending)
                escrowed_available = True
            except Exception as e:
                logger.warning(
                    "balance/onchain: escrow listing failed (%s); "
                    "reporting source unavailable", e,
                )

        total_ftns = balance_ftns + claimable_ftns + escrowed_ftns
        total_usd_equivalent = total_ftns * usd_rate

        return {
            # ── v1 fields (preserved bit-identically) ──
            "address": target,
            "balance_wei": balance_wei,
            "balance_ftns": balance_ftns,
            "usd_rate": usd_rate,
            "usd_equivalent": usd_equivalent,
            "source": "onchain",
            # ── v2 aggregate fields (additive) ──
            "claimable_royalties_ftns": claimable_ftns,
            "escrowed_ftns": escrowed_ftns,
            "total_ftns": total_ftns,
            "total_usd_equivalent": total_usd_equivalent,
            "sources": {
                "onchain": {
                    "ftns": balance_ftns,
                    "available": True,
                },
                "claimable_royalties": {
                    "ftns": claimable_ftns,
                    "available": claimable_available,
                },
                "escrowed": {
                    "ftns": escrowed_ftns,
                    "available": escrowed_available,
                },
            },
        }

    class _OfframpQuoteRequest(BaseModel):
        # Validation is in the handler so the 400 vs 422 boundary is
        # explicit (Pydantic's gt=0 returns 422; we want 400 for
        # operator-misconfig-class errors).
        usd_amount: float
        bank_account_alias: str = "primary"

    @app.get("/wallet/spend")
    async def get_wallet_spend(
        days: int = 30,
        address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sum FTNS spent on completed compute jobs over the last
        N days. RELEASED escrows count; REFUNDED + PENDING do not.

        Backs the ``prsm_spend_summary`` MCP tool.
        """
        if days <= 0 or days > 365:
            raise HTTPException(
                status_code=422,
                detail=f"days must be in [1, 365], got {days}",
            )

        escrow_svc = getattr(node, "_payment_escrow", None)
        ftns_ledger = getattr(node, "ftns_ledger", None)
        if escrow_svc is None or ftns_ledger is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "PaymentEscrow or ftns_ledger not initialized."
                ),
            )
        target = address or getattr(
            ftns_ledger, "_connected_address", None,
        )
        if not target:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No connected address; pass ?address=0x... explicitly."
                ),
            )

        try:
            entries = escrow_svc.list_escrows_by_requester(
                target, pending_only=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_escrows_by_requester raised: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"escrow listing failed: {exc}",
            )

        from prsm.node.payment_escrow import EscrowStatus as _ES
        cutoff = _time_for_history.time() - days * 86400.0
        total = 0.0
        count = 0
        for e in entries:
            if e.status != _ES.RELEASED:
                continue
            if e.completed_at is None or e.completed_at < cutoff:
                continue
            total += e.amount
            count += 1

        return {
            "address": target,
            "days": days,
            "total_spent_ftns": total,
            "escrows_count": count,
        }

    @app.get("/wallet/escrows")
    async def get_wallet_escrows(
        address: Optional[str] = None,
        include_terminal: bool = False,
    ) -> Dict[str, Any]:
        """List active escrows for the operator's wallet (or any
        address via override). Backs the ``prsm_escrow_summary``
        MCP tool.

        Returns 503 if PaymentEscrow or ftns_ledger isn't wired,
        200 with `{escrows: [...], total: N, total_locked_ftns: X,
        address: 0x...}`. Default `pending_only=true`; pass
        `?include_terminal=true` for RELEASED + REFUNDED audit
        view.
        """
        escrow_svc = getattr(node, "_payment_escrow", None)
        ftns_ledger = getattr(node, "ftns_ledger", None)
        if escrow_svc is None or ftns_ledger is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "PaymentEscrow or ftns_ledger not initialized "
                    "on this node."
                ),
            )

        target = address or getattr(
            ftns_ledger, "_connected_address", None,
        )
        if not target:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No connected address available; pass "
                    "?address=0x... explicitly."
                ),
            )

        try:
            entries = escrow_svc.list_escrows_by_requester(
                target, pending_only=not include_terminal,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "list_escrows_by_requester raised: %s", exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"escrow listing failed: {exc}",
            )

        from prsm.node.payment_escrow import EscrowStatus as _EscStatus
        escrows_dicts = []
        total_locked = 0.0
        for e in entries:
            escrows_dicts.append({
                "escrow_id": e.escrow_id,
                "job_id": e.job_id,
                "amount_ftns": e.amount,
                "status": e.status.value,
                "provider_winner": e.provider_winner,
                "tx_lock": e.tx_lock,
                "tx_release": e.tx_release,
                "created_at": e.created_at,
                "completed_at": e.completed_at,
            })
            if e.status == _EscStatus.PENDING:
                total_locked += e.amount

        return {
            "address": target,
            "escrows": escrows_dicts,
            "total": len(escrows_dicts),
            "total_locked_ftns": total_locked,
            "include_terminal": include_terminal,
        }

    class _RoyaltyClaimRequest(BaseModel):
        dry_run: bool = True

    @app.post("/wallet/royalty/claim")
    async def post_royalty_claim(
        body: _RoyaltyClaimRequest,
    ) -> Dict[str, Any]:
        """Claim accumulated royalties from RoyaltyDistributor.

        Closes the loop on the offramp-quote claim_required path:
        when /wallet/offramp/quote returns claim_required=True,
        operators authorize the claim via this endpoint (or via
        the prsm_royalty_claim MCP composer that calls it).

        Behavior:
          - dry_run=True (default): read claimable + return artifact
            without on-chain action; status="DRY_RUN"
          - dry_run=False: call client.claim(); status="EXECUTED"
            with tx_hash + amount_claimed
          - claimable=0 + dry_run=False: skip the claim() call
            (avoids on-chain ZeroClaim revert + gas burn);
            status="SKIPPED_ZERO"
        """
        client = getattr(node, "_royalty_distributor_client", None)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "RoyaltyDistributor client not wired on this "
                    "node. Set PRSM_ROYALTY_DISTRIBUTOR_ADDRESS + "
                    "FTNS_TOKEN_ADDRESS to enable."
                ),
            )

        ftns_ledger = getattr(node, "ftns_ledger", None)
        decimals = getattr(ftns_ledger, "_decimals", 18) if ftns_ledger else 18

        try:
            claimable_wei = client.claimable()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RoyaltyDistributorClient.claimable raised: %s", exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"claimable() RPC failed: {exc}",
            )

        claimable_ftns = float(claimable_wei) / (10 ** decimals)

        if body.dry_run:
            return {
                "status": "DRY_RUN",
                "claimable_ftns": claimable_ftns,
                "amount_claimed_ftns": 0.0,
                "tx_hash": None,
            }

        if claimable_wei == 0:
            return {
                "status": "SKIPPED_ZERO",
                "claimable_ftns": 0.0,
                "amount_claimed_ftns": 0.0,
                "tx_hash": None,
                "note": (
                    "Skipped on-chain claim() call to avoid "
                    "ZeroClaim revert + gas burn."
                ),
            }

        try:
            tx_hash, transfer_status = client.claim()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RoyaltyDistributorClient.claim raised: %s", exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"claim() failed: {exc}",
            )

        return {
            "status": "EXECUTED",
            "claimable_ftns": claimable_ftns,
            "amount_claimed_ftns": claimable_ftns,
            "tx_hash": tx_hash,
            "transfer_status": (
                transfer_status.value
                if hasattr(transfer_status, "value")
                else str(transfer_status)
            ),
        }

    @app.post("/wallet/offramp/quote")
    async def post_offramp_quote(
        body: _OfframpQuoteRequest,
        address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pre-flight quote for FTNS → USDC → USD off-ramp via
        Aerodrome + Coinbase CDP.

        V1 scope: returns the transaction-summary artifact described
        in Vision §13 Phase 5 step 2 ("Gemini presents an Artifact in
        your side panel"). Does NOT initiate any on-chain swap or
        Coinbase off-ramp; actual execution gates on CDP commission
        per Vision gantt 2026-06-15. Until then status is
        ``PENDING_COMMISSION``.

        Validation:
          - ``usd_amount`` must be positive (Pydantic gt=0).
          - Source balance must be ≥ requested USD; otherwise 422.
          - ``ftns_ledger`` must be initialized; otherwise 503.
        """
        # Pydantic gt=0 catches usd_amount <= 0 with 422; the test
        # suite expects 400 for explicit negative/zero. Re-validate
        # to surface the simpler 400.
        if body.usd_amount <= 0:
            raise HTTPException(
                status_code=400,
                detail="usd_amount must be positive (> 0)",
            )

        if not getattr(node, "ftns_ledger", None):
            raise HTTPException(
                status_code=503,
                detail=(
                    "On-chain ftns_ledger not initialized; "
                    "cannot quote off-ramp without source balance."
                ),
            )

        target = address or node.ftns_ledger._connected_address
        balance_ftns = await node.ftns_ledger.get_balance(target)

        rate_raw = os.getenv("PRSM_FTNS_USD_RATE", "").strip()
        usd_rate = 1.0
        if rate_raw:
            try:
                parsed = float(rate_raw)
                if parsed > 0:
                    usd_rate = parsed
            except ValueError:
                pass

        balance_usd = balance_ftns * usd_rate

        # Aggregate-source available: on-chain + claimable royalties.
        # Escrowed FTNS does NOT count (locked in pending compute jobs).
        # Each source fail-soft: RPC errors → treat as 0 contribution.
        decimals = getattr(node.ftns_ledger, "_decimals", 18)
        claimable_ftns = 0.0
        royalty_client = getattr(node, "_royalty_distributor_client", None)
        if royalty_client is not None:
            try:
                claimable_wei = royalty_client.claimable(target)
                claimable_ftns = float(claimable_wei) / (10 ** decimals)
            except Exception as e:
                logger.warning(
                    "offramp/quote: royalty claimable() raised — "
                    "treating as 0 (fail-soft): %s", e,
                )

        available_ftns = balance_ftns + claimable_ftns
        available_usd = available_ftns * usd_rate
        ftns_to_swap = body.usd_amount / usd_rate

        if available_usd < body.usd_amount:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Insufficient balance: requested ${body.usd_amount:.2f}, "
                    f"available ${available_usd:.2f} "
                    f"({available_ftns:.6f} FTNS @ {usd_rate} USD/FTNS"
                    f"; on-chain {balance_ftns:.6f} + claimable "
                    f"{claimable_ftns:.6f})"
                ),
            )

        # Claim-required path: on-chain alone insufficient but
        # aggregate covers. Operator must claim royalties before the
        # eventual swap can execute. Surface the claim amount so the
        # composer can chain a claim step.
        claim_required = balance_ftns < ftns_to_swap
        claim_amount_ftns = (
            max(0.0, ftns_to_swap - balance_ftns) if claim_required else 0.0
        )

        return {
            "requested_usd": body.usd_amount,
            "source_address": target,
            "source_balance_ftns": balance_ftns,
            "source_balance_usd": balance_usd,
            # Aggregate-source mirror (additive — clients reading
            # legacy fields keep working).
            "available_ftns": available_ftns,
            "available_usd": available_usd,
            "claimable_royalties_ftns": claimable_ftns,
            "claim_required": claim_required,
            "claim_amount_ftns": claim_amount_ftns,
            "quote": {
                "ftns_to_swap": ftns_to_swap,
                "usdc_received": body.usd_amount,
                "usd_settled": body.usd_amount,
                "swap_route": "aerodrome",
                "offramp_route": "coinbase-cdp",
                "bank_account_alias": body.bank_account_alias,
            },
            "usd_rate": usd_rate,
            "status": "PENDING_COMMISSION",
            "commission_gate_note": (
                "Coinbase CDP commission gates on Aerodrome USDC-FTNS "
                "pool seeding (Vision gantt 2026-06-15). The summary "
                "above shows what the transaction will look like once "
                "execution ships; it does NOT initiate any on-chain "
                "swap or fiat off-ramp."
            ),
        }

    @app.post("/compute/submit")
    async def submit_compute_job(job: JobSubmission) -> Dict[str, Any]:
        """Submit a compute job to the network."""
        if not node.compute_requester:
            raise HTTPException(status_code=503, detail="Compute requester not initialized")

        from prsm.node.compute_provider import JobType
        try:
            job_type = JobType(job.job_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid job type: {job.job_type}. Valid types: inference, embedding, benchmark",
            )

        try:
            submitted = await node.compute_requester.submit_job(
                job_type=job_type,
                payload=job.payload,
                ftns_budget=job.ftns_budget,
            )
            return {
                "job_id": submitted.job_id,
                "status": submitted.status.value,
                "job_type": submitted.job_type.value,
                "ftns_budget": submitted.ftns_budget,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/compute/job/{job_id}")
    async def get_job_status(job_id: str) -> Dict[str, Any]:
        """Get status of a submitted compute job."""
        if not node.compute_requester:
            raise HTTPException(status_code=503, detail="Compute requester not initialized")

        job = node.compute_requester.submitted_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "job_type": job.job_type.value,
            "provider_id": job.provider_id,
            "result": job.result,
            "result_verified": job.result_verified,
            "error": job.error,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }

    @app.post("/compute/query")
    async def compute_query(body: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Submit a compute job and wait for the result (blocking).

        POST body: {"prompt": "...", "model": "nwtn", "timeout": 120}
        Returns: {"job_id", "response", "result"}
        """
        if not node.compute_requester or not node.compute_provider:
            raise HTTPException(status_code=503, detail="Compute not initialized")

        prompt = body.get("prompt", "")
        model = body.get("model", "nwtn")
        timeout = float(body.get("timeout", 120.0))
        budget = float(body.get("budget", 0.0))

        from prsm.node.compute_provider import JobType, ComputeJob

        import uuid as _uuid

        job_id = "compute-" + _uuid.uuid4().hex
        escrow_entry = None

        # --- Phase 1: Lock escrow (no gossip broadcast) ---
        if budget > 0 and hasattr(node, '_payment_escrow') and node._payment_escrow:
            escrow_entry = await node._payment_escrow.create_escrow(
                job_id=job_id,
                amount=budget,
                requester_id=node.identity.node_id,
            )
            if escrow_entry:
                logger.info(
                    f"compute_query: escrow locked {budget:.6f} FTNS "
                    f"for {job_id[:8]}"
                )
            else:
                logger.warning(
                    f"compute_query: escrow creation failed for {job_id[:8]}"
                )

        # --- Phase 2: Route the job ---
        result = None

        # Try Agent Forge first (Rings 1-10 pipeline) if available
        if hasattr(node, 'agent_forge') and node.agent_forge is not None:
            try:
                forge_result = await node.agent_forge.run(
                    query=prompt,
                    budget_ftns=budget,
                )
                if forge_result and forge_result.get("status") == "success":
                    # Extract response from forge result
                    route = forge_result.get("route", "unknown")
                    if route == "direct_llm":
                        result = {"response": forge_result.get("response", ""), "route": route}
                    elif route == "swarm":
                        output = forge_result.get("aggregated_output", {})
                        result = {"response": str(output), "route": route}
                    else:
                        result = {"response": str(forge_result), "route": route}
                    result["forge_result"] = forge_result
            except Exception as e:
                logger.debug(f"compute_query: forge path failed, falling back: {e}")

        # Try gossip-based peers if forge didn't produce a result
        has_peers = (
            node.transport
            and hasattr(node.transport, 'peer_count')
            and node.transport.peer_count > 0
        )
        if has_peers:
            try:
                # Dynamic timeout: allow enough time for remote inference
                # API timeout is the user-facing limit; gossip gets most of it
                gossip_timeout = max(timeout * 0.8, 30.0)

                # Submit via gossip for multi-node federation
                # Pass our job_id so escrow and gossip track the same ID
                submitted = await node.compute_requester.submit_job(
                    job_type=JobType.INFERENCE,
                    payload={"prompt": prompt, "model": model},
                    ftns_budget=budget,
                    use_escrow=False,  # escrow already created above
                    job_id=job_id,     # propagate API job_id
                )
                result = await node.compute_requester.get_result(
                    submitted.job_id, timeout=gossip_timeout
                )
                if result:
                    job_id = submitted.job_id
            except Exception as e:
                logger.warning(f"compute_query: gossip routing failed: {e}")

        # Fallback: direct self-compute (single-node mode)
        if result is None and node.compute_provider.allow_self_compute:
            self_job = ComputeJob(
                job_id=job_id,
                job_type=JobType.INFERENCE,
                payload={"prompt": prompt, "model": model},
                requester_id=node.identity.node_id,
                ftns_budget=budget,
            )
            logger.info(
                f"compute_query: self-compute for {job_id[:8]}"
            )
            await node.compute_provider._execute_job(self_job)
            completed = node.compute_provider.completed_jobs.get(job_id)
            if completed:
                result = completed.result

        if result is None:
            # Refund escrow on failure
            if escrow_entry and node._payment_escrow:
                try:
                    await node._payment_escrow.refund_escrow(job_id)
                except Exception:
                    pass
            raise HTTPException(
                status_code=504, detail="Compute timed out or no provider accepted"
            )

        # --- Phase 3: Release escrow to provider ---
        if budget > 0 and escrow_entry and node._payment_escrow:
            try:
                await node._payment_escrow.release_escrow(
                    job_id=job_id,
                    provider_id=node.identity.node_id,
                    consensus_reached=True,
                )
                logger.info(
                    f"api: escrow released {budget:.6f} FTNS for {job_id[:8]}"
                )
            except Exception as e:
                logger.warning(f"api: escrow release for {job_id[:8]} failed: {e}")

        return {
            "job_id": job_id,
            "response": result.get("response", result.get("text", str(result))),
            "result": result,
        }

    @app.get("/privacy/budget", tags=["privacy"])
    async def get_privacy_budget() -> Dict[str, Any]:
        """Return the current differential-privacy budget audit report.

        Used by the prsm_privacy_status MCP tool. The budget is enforced
        across all forge queries with privacy_level != 'none'.
        """
        if not hasattr(node, "privacy_budget") or node.privacy_budget is None:
            raise HTTPException(
                status_code=503,
                detail="Privacy budget not initialized (Ring 7 unavailable)",
            )
        return node.privacy_budget.get_audit_report()

    @app.post("/compute/forge/quote")
    async def compute_forge_quote(body: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Get a cost quote for a forge query without executing it (free).

        POST body: {
            "query": "...",
            "shard_cids": ["QmA", "QmB"],   // optional
            "shard_count": 3,                // optional, default 3
            "hardware_tier": "t2",           // optional, default t2
            "estimated_pcu_per_shard": 50.0  // optional, default 50.0
        }

        Returns the same CostQuote shape used by the prsm_quote MCP tool and
        the Python SDK's client.quote() helper. This endpoint is what the
        JavaScript and Go SDKs call.
        """
        from prsm.economy.pricing import PricingEngine

        query = body.get("query", "") or ""
        if not query.strip():
            raise HTTPException(status_code=400, detail="Missing 'query' field")

        shard_cids = body.get("shard_cids") or []
        # If caller passed shard_cids, use that count; otherwise honor explicit shard_count
        shard_count = len(shard_cids) if shard_cids else int(body.get("shard_count", 3))
        if shard_count < 1:
            shard_count = 1
        hardware_tier = str(body.get("hardware_tier", "t2"))
        estimated_pcu = float(body.get("estimated_pcu_per_shard", 50.0))

        engine = PricingEngine()
        quote = engine.quote_swarm_job(
            shard_count=shard_count,
            hardware_tier=hardware_tier,
            estimated_pcu_per_shard=estimated_pcu,
        )
        return {
            "query": query,
            "shard_count": shard_count,
            "hardware_tier": hardware_tier,
            **quote.to_dict(),
        }

    @app.post("/compute/forge")
    async def compute_forge(
        body: Dict[str, Any] = {},
        idempotency_key: Optional[str] = Header(
            default=None, alias="Idempotency-Key",
        ),
    ) -> Dict[str, Any]:
        """Submit a query through the full Ring 1-10 Agent Forge pipeline.

        This is the end-to-end sovereign-edge AI path:
        1. AgentForge decomposes the query via LLM
        2. Finds relevant data shards (if any)
        3. Quotes costs via PricingEngine
        4. Routes to appropriate execution path:
           - DIRECT_LLM: simple queries answered by LLM directly
           - SINGLE_AGENT: dispatch WASM agent to one node
           - SWARM: fan out agents across multiple nodes
        5. Collects and aggregates results
        6. Settles FTNS payments
        7. Returns the answer

        POST body: {
            "query": "...",
            "budget_ftns": 10.0,
            "shard_cids": ["QmA", "QmB"],  // optional
            "privacy_level": "standard"     // none, standard, high, maximum
        }
        """
        # Idempotency-Key handling FIRST (before any other checks):
        # if header present + we've seen this key, return the cached
        # job's status directly without locking a new escrow /
        # re-running compute. Retry-safe POST for clients that may
        # double-fire on network blips. Cache hit doesn't require
        # agent_forge to be wired.
        if idempotency_key and getattr(node, "_job_history", None) is not None:
            try:
                prior_job_id = node._job_history.lookup_by_idempotency_key(
                    idempotency_key,
                )
                if prior_job_id:
                    prior_record = node._job_history.get(prior_job_id)
                    if prior_record is not None:
                        return {
                            "status": "idempotent_replay",
                            "job_id": prior_job_id,
                            "history": prior_record.to_dict(),
                            "note": (
                                "Returning cached result from prior request "
                                "with same Idempotency-Key. No new escrow "
                                "locked; no compute re-run."
                            ),
                        }
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Idempotency lookup raised; proceeding with "
                    "fresh forge: %s", exc,
                )

        # PRSM_MAX_FTNS_PER_JOB cap enforcement (cost-control).
        # Default unlimited; non-numeric/zero/negative env values
        # silently disable the cap (log WARN). Operators tune this
        # when worried about misbehaving AI agents draining FTNS
        # via single oversized requests. Fires BEFORE agent_forge
        # availability check so an oversized request cleanly 422s
        # even when the forge subsystem is down.
        _cap_raw = os.getenv("PRSM_MAX_FTNS_PER_JOB", "").strip()
        if _cap_raw:
            try:
                _cap = float(_cap_raw)
                if _cap > 0:
                    requested = float(body.get("budget_ftns", 10.0))
                    if requested > _cap:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"budget_ftns {requested} exceeds "
                                f"PRSM_MAX_FTNS_PER_JOB cap of {_cap}. "
                                f"Either lower the budget or have the "
                                f"operator raise the cap."
                            ),
                        )
            except ValueError:
                logger.warning(
                    "PRSM_MAX_FTNS_PER_JOB=%r not numeric; cap disabled",
                    _cap_raw,
                )

        if not hasattr(node, 'agent_forge') or node.agent_forge is None:
            raise HTTPException(
                status_code=503,
                detail="Agent forge not initialized. Check LLM backend configuration."
            )

        query = body.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' field")

        budget_ftns = float(body.get("budget_ftns", 10.0))
        shard_cids = body.get("shard_cids", None)
        privacy_level_str = body.get("privacy_level", "standard")

        # Enforce minimum budget — PRSM requires FTNS for execution
        if budget_ftns <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "FTNS budget is required for query execution. "
                    "Set budget_ftns to at least 0.01 FTNS. "
                    "Use GET /compute/quote or the prsm_quote MCP tool to estimate costs first."
                ),
            )

        # Lock escrow if budget > 0
        job_id = "forge-" + _uuid.uuid4().hex[:12]
        escrow_entry = None
        if budget_ftns > 0 and hasattr(node, '_payment_escrow') and node._payment_escrow:
            escrow_entry = await node._payment_escrow.create_escrow(
                job_id=job_id,
                amount=budget_ftns,
                requester_id=node.identity.node_id,
            )

        # B8 async-dispatch follow-on: record IN_PROGRESS to
        # JobHistoryStore so /compute/status/{job_id} can surface
        # richer state than the escrow lifecycle alone. Best-effort —
        # store may be None on operators that haven't wired it.
        # When Idempotency-Key header is present, also register the
        # key → job_id mapping so retries hit the cache path above.
        _job_started_at = _time_for_history.time()
        if hasattr(node, "_job_history") and node._job_history is not None:
            try:
                from prsm.node.job_history import (
                    JobHistoryRecord as _JobRec,
                    JobStatus as _JobStat,
                )
                _ip_record = _JobRec(
                    job_id=job_id,
                    query=query,
                    status=_JobStat.IN_PROGRESS,
                    started_at=_job_started_at,
                )
                if idempotency_key:
                    node._job_history.put_with_idempotency(
                        _ip_record, idempotency_key=idempotency_key,
                    )
                else:
                    node._job_history.put(_ip_record)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JobHistoryStore put (IN_PROGRESS) failed for "
                    "job_id=%s: %s",
                    job_id, exc,
                )

        try:
            # Dispatch on agent_forge type. The QueryOrchestrator
            # (Ring 5 replacement, wired via PRSM_QUERY_ORCHESTRATOR_ENABLED)
            # exposes ``dispatch_query`` instead of the legacy
            # ``run`` surface; we route both through this endpoint
            # so MCP clients see a single /compute/forge.
            if hasattr(node.agent_forge, "dispatch_query"):
                # PRSM-PROV-1 / QueryOrchestrator path. Build a 32-byte
                # query_id (binds A6/A9 invariants per the aggregator-
                # selector threat model) and call dispatch_query.
                import os as _os_for_qid
                query_id = _os_for_qid.urandom(32)
                qo_result = await node.agent_forge.dispatch_query(
                    query=query,
                    prompter_node_id=node.identity.node_id,
                    query_id=query_id,
                )
                # Marshal AggregatedResult → /compute/forge response shape.
                # payload is the aggregator's combined output (UTF-8 bytes
                # for COUNT op; opaque for other ops).
                try:
                    response_text = qo_result.payload.decode("utf-8")
                except UnicodeDecodeError:
                    response_text = qo_result.payload.hex()
                result = {
                    "status": "success",
                    "route": "qo_swarm",
                    "aggregator_node_id": qo_result.aggregator_node_id,
                    "contributing_shards": list(qo_result.contributing_shards),
                    "response": response_text,
                    # §4 step 6 settlement attribution. List of
                    # {shard_cid, source_agent_pubkey_hex, creator_id}
                    # — settlement layer below builds the escrow
                    # split from this.
                    "participants": [
                        {
                            "shard_cid": pa.shard_cid,
                            "source_agent_pubkey_hex": pa.source_agent_pubkey.hex(),
                            "creator_id": pa.creator_id,
                        }
                        for pa in qo_result.participants
                    ],
                }
            else:
                # Legacy AgentForge path — preserved for any operator
                # running a non-orchestrator backend (e.g. test fixture).
                result = await node.agent_forge.run(
                    query=query,
                    budget_ftns=budget_ftns,
                    shard_cids=shard_cids,
                )

                if result is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Forge pipeline returned no result",
                    )

            # Track privacy budget if confidential compute is active
            if (
                hasattr(node, 'privacy_budget')
                and node.privacy_budget
                and privacy_level_str != "none"
            ):
                epsilon_map = {"standard": 8.0, "high": 4.0, "maximum": 1.0}
                epsilon = epsilon_map.get(privacy_level_str, 8.0)
                node.privacy_budget.record_spend(epsilon, "forge_query", job_id)

            # Release escrow on success. Two paths:
            #   (a) QO swarm path: §4 step 6 settlement — split the
            #       prompter's compute budget across the N compute
            #       participants (one per shard) + the aggregator
            #       coordination fee. Operator-tunable aggregator
            #       share via PRSM_AGGREGATOR_SHARE_BPS env var
            #       (default 500 = 5%).
            #   (b) Legacy path: single-provider release (prompter's
            #       own node) — preserves pre-QO behavior for
            #       AgentForge backends.
            if escrow_entry and node._payment_escrow and result.get("status") == "success":
                try:
                    qo_participants = result.get("participants") or []
                    if qo_participants:
                        # Build the split: aggregator share +
                        # uniform per-participant compute share.
                        import os as _os_for_share_bps
                        try:
                            agg_share_bps = int(_os_for_share_bps.environ.get(
                                "PRSM_AGGREGATOR_SHARE_BPS", "500",
                            ))
                        except ValueError:
                            agg_share_bps = 500
                        if not (0 <= agg_share_bps <= 10000):
                            agg_share_bps = 500
                        aggregator_share = budget_ftns * (agg_share_bps / 10000.0)
                        compute_share_total = budget_ftns - aggregator_share
                        # Uniform split across compute participants;
                        # PCU-weighted variant deferred to a follow-on
                        # once partials carry per-shard PCU metrics.
                        per_participant = (
                            compute_share_total / len(qo_participants)
                        )
                        splits = [
                            (result["aggregator_node_id"], aggregator_share),
                        ] + [
                            (p["source_agent_pubkey_hex"], per_participant)
                            for p in qo_participants
                        ]
                        await node._payment_escrow.release_escrow_split(
                            job_id=job_id,
                            splits=splits,
                        )
                    else:
                        # Legacy path
                        await node._payment_escrow.release_escrow(
                            job_id=job_id,
                            provider_id=node.identity.node_id,
                        )
                except Exception as e:
                    logger.warning(f"Forge escrow release failed: {e}")

            # Extract response text based on route. The QO path already
            # set "response" + "route" above; legacy paths populate
            # "route" via AgentForge result keys.
            route = result.get("route", "unknown")
            if route == "qo_swarm":
                response_text = result.get("response", str(result))
            elif route == "direct_llm":
                response_text = result.get("response", str(result))
            elif route == "swarm":
                output = result.get("aggregated_output", {})
                response_text = str(output.get("shard_outputs", output))
            elif route == "single_agent":
                agent_result = result.get("result", {})
                response_text = str(agent_result)
            else:
                response_text = str(result)

            traces_count = len(getattr(node.agent_forge, "traces", []))

            # B8 async-dispatch follow-on: record COMPLETED to
            # JobHistoryStore so /compute/status surfaces the full
            # result. Best-effort.
            if hasattr(node, "_job_history") and node._job_history is not None:
                try:
                    from prsm.node.job_history import (
                        JobHistoryRecord as _JobRec,
                        JobStatus as _JobStat,
                    )
                    node._job_history.put(_JobRec(
                        job_id=job_id,
                        query=query,
                        status=_JobStat.COMPLETED,
                        started_at=_job_started_at,
                        completed_at=_time_for_history.time(),
                        route=route,
                        response=response_text,
                        aggregator_node_id=result.get("aggregator_node_id"),
                        contributing_shards=tuple(
                            result.get("contributing_shards") or ()
                        ),
                        participants=tuple(
                            result.get("participants") or ()
                        ),
                        traces_collected=traces_count,
                    ))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "JobHistoryStore put (COMPLETED) failed for "
                        "job_id=%s: %s",
                        job_id, exc,
                    )

            return {
                "job_id": job_id,
                "query": query,
                "route": route,
                "response": response_text,
                "result": result,
                "budget_ftns": budget_ftns,
                "traces_collected": traces_count,
            }

        except HTTPException:
            raise
        except Exception as e:
            # Refund escrow on failure
            if escrow_entry and node._payment_escrow:
                try:
                    await node._payment_escrow.refund_escrow(job_id, str(e))
                except Exception:
                    pass
            # B8 async-dispatch follow-on: record FAILED.
            if hasattr(node, "_job_history") and node._job_history is not None:
                try:
                    from prsm.node.job_history import (
                        JobHistoryRecord as _JobRec,
                        JobStatus as _JobStat,
                    )
                    node._job_history.put(_JobRec(
                        job_id=job_id,
                        query=query,
                        status=_JobStat.FAILED,
                        started_at=_job_started_at,
                        completed_at=_time_for_history.time(),
                        error=str(e),
                    ))
                except Exception as hist_exc:  # noqa: BLE001
                    logger.warning(
                        "JobHistoryStore put (FAILED) failed for "
                        "job_id=%s: %s",
                        job_id, hist_exc,
                    )
            logger.error(f"Forge pipeline error: {e}")
            raise HTTPException(status_code=500, detail=f"Forge pipeline error: {str(e)}")

    @app.get("/content/arbitration/queue")
    async def get_arbitration_queue() -> Dict[str, Any]:
        """List pending content-attribution disputes awaiting
        council adjudication. Closes the operator-UX gap from
        PRSM-PROV-1 Item 6: the FilesystemArbitrationQueue
        persists disputes at ``~/.prsm/arbitration_queue/`` but
        operators previously had no way to see them without
        scanning the directory by hand.

        Backs the ``prsm_arbitration_status`` MCP tool.

        Status:
          503 — arbitration_queue not wired
          502 — list_pending raised (disk error, lock timeout)
          200 — {pending: [...], total: N}
        """
        queue = getattr(node, "_arbitration_queue", None)
        if queue is None:
            raise HTTPException(
                status_code=503,
                detail="ArbitrationQueue not initialized on this node.",
            )
        try:
            records = await queue.list_pending()
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_pending raised: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=str(exc),
            )
        return {
            "pending": [r.to_dict() for r in records],
            "total": len(records),
        }

    @app.post("/compute/cleanup-stale")
    async def post_compute_cleanup_stale() -> Dict[str, Any]:
        """Manually trigger PaymentEscrow.cleanup_expired_escrows.

        Refunds any PENDING escrow whose age exceeds
        ``default_timeout``. Returns ``{cleaned: N}``. Use when
        operators need an immediate sweep without waiting for
        the 10-min periodic loop (e.g., after lowering
        PRSM_ESCROW_TIMEOUT_SEC, draining for maintenance).

        Status:
          503 — PaymentEscrow not wired
          502 — cleanup_expired_escrows raised
          200 — cleaned count returned
        """
        escrow_svc = getattr(node, "_payment_escrow", None)
        if escrow_svc is None:
            raise HTTPException(
                status_code=503,
                detail="PaymentEscrow not initialized on this node.",
            )
        try:
            cleaned = await escrow_svc.cleanup_expired_escrows()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cleanup_expired_escrows raised: %s", exc,
            )
            raise HTTPException(
                status_code=502,
                detail=str(exc),
            )
        return {"cleaned": cleaned}

    @app.get("/compute/jobs")
    async def compute_jobs_list(
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Paginated operator-side job list. Backs the
        ``prsm_jobs_list`` MCP tool.

        Query params:
          - status: optional JobStatus filter (in_progress |
            completed | failed | cancelled).
          - limit: page size (1..100, default 50).
          - offset: pagination offset, default 0.

        Returns 503 if JobHistoryStore not wired, 422 on
        validation errors. 200 returns
        {jobs: [...], total: N, offset: X, limit: Y}.
        """
        from prsm.node.job_history import JobStatus

        history = getattr(node, "_job_history", None)
        if history is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "JobHistoryStore is not initialized on this "
                    "node. Cannot list jobs."
                ),
            )

        # Validate query params.
        if offset < 0:
            raise HTTPException(
                status_code=422,
                detail=f"offset must be >= 0, got {offset}",
            )
        if limit <= 0 or limit > 100:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"limit must be in [1, 100], got {limit}. "
                    f"Use offset to paginate further."
                ),
            )
        status_filter = None
        if status is not None:
            try:
                status_filter = JobStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"invalid status {status!r}. Allowed: "
                        f"{[s.value for s in JobStatus]}"
                    ),
                )

        records = history.list(
            status_filter=status_filter, limit=limit, offset=offset,
        )
        total = history.count(status_filter=status_filter)
        return {
            "jobs": [r.to_dict() for r in records],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    @app.get("/compute/status/{job_id}")
    async def compute_status(job_id: str) -> Dict[str, Any]:
        """Look up the status of a /compute/forge job by its job_id.

        Backs the ``prsm_agent_status`` MCP tool. Two-tier lookup:

        1. **JobHistoryStore** (richer): pipeline state — route,
           response, aggregator, participants, traces, error if
           failed. Always preferred when present.
        2. **PaymentEscrow** (fallback): payment-leg lifecycle
           (pending / released / refunded / disputed + amount +
           timing + provider winner). Used when history doesn't
           know about the job (e.g. node-restart eviction, or
           budget=0 test fixtures that touched the escrow but
           weren't recorded in history).

        The two surfaces compose: a `compute` block (from history)
        and an `escrow` block (from payment_escrow) appear together
        when both are populated. At least one must be available or
        the endpoint returns 404.
        """
        history = getattr(node, "_job_history", None)
        escrow_svc = getattr(node, "_payment_escrow", None)

        if history is None and escrow_svc is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Neither JobHistoryStore nor PaymentEscrow "
                    "is initialized on this node."
                ),
            )

        history_record = None
        if history is not None:
            try:
                history_record = history.get(job_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JobHistoryStore.get raised for job_id=%s: %s",
                    job_id, exc,
                )

        escrow = None
        if escrow_svc is not None:
            escrow = escrow_svc.get_escrow(job_id)

        if history_record is None and escrow is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No history or escrow record for job_id={job_id!r}. "
                    f"Either the job never ran on this node, or its "
                    f"history was evicted (LRU-bounded) and it ran "
                    f"without a locked escrow."
                ),
            )

        body: Dict[str, Any] = {"job_id": job_id}
        if history_record is not None:
            body["compute"] = history_record.to_dict()
        if escrow is not None:
            body["escrow"] = {
                "escrow_id": escrow.escrow_id,
                "requester_id": escrow.requester_id,
                "amount_ftns": escrow.amount,
                "status": escrow.status.value,
                "provider_winner": escrow.provider_winner,
                "tx_lock": escrow.tx_lock,
                "tx_release": escrow.tx_release,
                "created_at": escrow.created_at,
                "completed_at": escrow.completed_at,
                "metadata": dict(escrow.metadata or {}),
            }
        return body

    @app.get("/compute/status/{job_id}/stream")
    async def compute_status_stream(job_id: str):
        """SSE-streaming sibling of /compute/status/{job_id}.

        Closes the last B8 deferred sub-item: clients can render
        IN_PROGRESS → COMPLETED transitions live without polling
        the GET endpoint.

        Polling-based v1: server polls JobHistoryStore +
        PaymentEscrow at PRSM_STATUS_STREAM_POLL_SEC interval
        (default 0.5s), emits an SSE event whenever the snapshot
        changes (de-duplicated by JSON equality), and closes the
        connection on terminal status (history COMPLETED/FAILED/
        CANCELLED OR escrow RELEASED/REFUNDED) OR after
        PRSM_STATUS_STREAM_TIMEOUT_SEC (default 1800s = 30min).

        Wire format:
            event: status
            data: {"job_id": "...", "history": {...},
                   "escrow": {...}}

            event: terminal
            data: {"job_id": "...", "reason": "completed"|
                   "history_terminal"|"escrow_terminal"|"timeout"}

        `event: terminal` is the only terminal event — the server
        closes the connection after emitting it. Clients can
        re-subscribe to keep watching past a timeout.
        """
        from prsm.node.job_history import JobStatus
        from prsm.node.payment_escrow import EscrowStatus

        history = getattr(node, "_job_history", None)
        escrow_svc = getattr(node, "_payment_escrow", None)

        if history is None and escrow_svc is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Neither JobHistoryStore nor PaymentEscrow "
                    "is initialized on this node."
                ),
            )

        # Initial existence check — 404 fast if neither has the job.
        initial_history = history.get(job_id) if history else None
        initial_escrow = (
            escrow_svc.get_escrow(job_id) if escrow_svc else None
        )
        if initial_history is None and initial_escrow is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No history or escrow record for job_id={job_id!r}."
                ),
            )

        poll_raw = os.getenv("PRSM_STATUS_STREAM_POLL_SEC", "0.5").strip()
        try:
            poll_sec = float(poll_raw) if poll_raw else 0.5
            if poll_sec <= 0:
                poll_sec = 0.5
        except ValueError:
            poll_sec = 0.5

        timeout_raw = os.getenv(
            "PRSM_STATUS_STREAM_TIMEOUT_SEC", "1800",
        ).strip()
        try:
            timeout_sec = float(timeout_raw) if timeout_raw else 1800.0
            if timeout_sec <= 0:
                timeout_sec = 1800.0
        except ValueError:
            timeout_sec = 1800.0

        TERMINAL_HISTORY = {
            JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED,
        }
        TERMINAL_ESCROW = {
            EscrowStatus.RELEASED, EscrowStatus.REFUNDED,
        }

        def _snapshot():
            h = history.get(job_id) if history else None
            e = escrow_svc.get_escrow(job_id) if escrow_svc else None
            payload = {"job_id": job_id}
            if h is not None:
                payload["history"] = h.to_dict()
            if e is not None:
                payload["escrow"] = {
                    "escrow_id": e.escrow_id,
                    "status": e.status.value,
                    "amount_ftns": e.amount,
                    "completed_at": e.completed_at,
                }
            return payload, h, e

        def _terminal_reason(h, e):
            if h is not None and h.status in TERMINAL_HISTORY:
                return (
                    h.status.value if h.status == JobStatus.COMPLETED
                    else "history_terminal"
                )
            if e is not None and e.status in TERMINAL_ESCROW:
                return "escrow_terminal"
            return None

        # Pre-built initial payload from the existence-check reads
        # so the first snapshot doesn't double-poll.
        def _build_payload(h, e):
            payload = {"job_id": job_id}
            if h is not None:
                payload["history"] = h.to_dict()
            if e is not None:
                payload["escrow"] = {
                    "escrow_id": e.escrow_id,
                    "status": e.status.value,
                    "amount_ftns": e.amount,
                    "completed_at": e.completed_at,
                }
            return payload

        async def _generate():
            import asyncio as _asyncio
            last_payload_json = None
            start = _time_for_history.time()
            # Use the existence-check reads as the first snapshot
            # to avoid a redundant get() on the loop's first pass.
            h, e = initial_history, initial_escrow
            first_pass = True
            while True:
                if not first_pass:
                    try:
                        _payload, h, e = _snapshot()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "compute_status_stream snapshot raised for "
                            "job_id=%s: %s", job_id, exc,
                        )
                        yield (
                            "event: error\n"
                            f"data: {json.dumps({'error': str(exc)})}\n\n"
                        )
                        return
                payload = _build_payload(h, e)
                payload_json = json.dumps(payload, sort_keys=True)
                if payload_json != last_payload_json:
                    yield f"event: status\ndata: {payload_json}\n\n"
                    last_payload_json = payload_json
                reason = _terminal_reason(h, e)
                if reason is not None:
                    yield (
                        "event: terminal\n"
                        f"data: {json.dumps({'job_id': job_id, 'reason': reason})}\n\n"
                    )
                    return
                if _time_for_history.time() - start >= timeout_sec:
                    yield (
                        "event: terminal\n"
                        f"data: {json.dumps({'job_id': job_id, 'reason': 'timeout'})}\n\n"
                    )
                    return
                first_pass = False
                await _asyncio.sleep(poll_sec)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
        )

    @app.post("/compute/cancel/{job_id}")
    async def compute_cancel(job_id: str) -> Dict[str, Any]:
        """Cancel a /compute/forge job: mark history.status as
        CANCELLED + refund any PENDING escrow.

        v1 caveat: in-flight Python coroutines are NOT interrupted.
        Cancellation marks intent + refunds the budget. If the
        coroutine completes successfully later, its
        release_escrow_split call will race-lose against the
        REFUNDED escrow and raise EscrowAlreadyFinalizedError —
        the correct race-loss outcome.

        Status:
          503 — neither JobHistoryStore nor PaymentEscrow wired.
          404 — neither has the job.
          409 — job already terminal (COMPLETED/FAILED/CANCELLED on
                history side, RELEASED/REFUNDED on escrow side).
          200 — cancelled. Response surfaces history_cancelled,
                escrow_refunded, refund_amount_ftns.
        """
        from prsm.node.job_history import (
            JobHistoryRecord, JobStatus,
        )
        from prsm.node.payment_escrow import (
            EscrowAlreadyFinalizedError, EscrowStatus,
        )

        history = getattr(node, "_job_history", None)
        escrow_svc = getattr(node, "_payment_escrow", None)

        if history is None and escrow_svc is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Neither JobHistoryStore nor PaymentEscrow "
                    "is initialized on this node."
                ),
            )

        # Look up both surfaces.
        history_record = None
        if history is not None:
            try:
                history_record = history.get(job_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JobHistoryStore.get raised for job_id=%s: %s",
                    job_id, exc,
                )

        escrow = None
        if escrow_svc is not None:
            try:
                escrow = escrow_svc.get_escrow(job_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PaymentEscrow.get_escrow raised for job_id=%s: %s",
                    job_id, exc,
                )

        if history_record is None and escrow is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No history or escrow record for job_id={job_id!r}; "
                    f"nothing to cancel."
                ),
            )

        # 409 — already terminal on either surface.
        if history_record is not None and history_record.status in {
            JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED,
        }:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Job {job_id!r} is already in terminal state "
                    f"{history_record.status.value!r}; cannot cancel."
                ),
            )
        if escrow is not None and escrow.status in {
            EscrowStatus.RELEASED, EscrowStatus.REFUNDED,
        }:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Escrow for job {job_id!r} is already "
                    f"{escrow.status.value!r}; cannot cancel."
                ),
            )

        # Mark history CANCELLED.
        history_cancelled = False
        if history_record is not None:
            try:
                cancelled_record = JobHistoryRecord(
                    job_id=history_record.job_id,
                    query=history_record.query,
                    status=JobStatus.CANCELLED,
                    started_at=history_record.started_at,
                    completed_at=_time_for_history.time(),
                    route=history_record.route,
                    response=history_record.response,
                    aggregator_node_id=history_record.aggregator_node_id,
                    contributing_shards=history_record.contributing_shards,
                    participants=history_record.participants,
                    traces_collected=history_record.traces_collected,
                    error=history_record.error,
                )
                history.put(cancelled_record)
                history_cancelled = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JobHistoryStore.put (CANCELLED) failed for "
                    "job_id=%s: %s", job_id, exc,
                )

        # Refund the escrow if PENDING.
        escrow_refunded = False
        refund_amount_ftns = 0.0
        if escrow is not None and escrow.status == EscrowStatus.PENDING:
            try:
                ok = await escrow_svc.refund_escrow(
                    job_id, reason="operator-cancelled via /compute/cancel",
                )
                escrow_refunded = bool(ok)
                if escrow_refunded:
                    refund_amount_ftns = escrow.amount
            except EscrowAlreadyFinalizedError:
                # Race-loss: release_escrow_split landed between
                # our check + our refund. Treat as best-effort
                # cancel; history already CANCELLED.
                logger.info(
                    "Escrow for job_id=%s race-finalized during "
                    "cancel; treating as race-loss.", job_id,
                )
                escrow_refunded = False
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PaymentEscrow.refund_escrow raised for "
                    "job_id=%s: %s", job_id, exc,
                )
                escrow_refunded = False

        return {
            "job_id": job_id,
            "history_cancelled": history_cancelled,
            "escrow_refunded": escrow_refunded,
            "refund_amount_ftns": refund_amount_ftns,
        }

    @app.post("/compute/inference")
    async def compute_inference(body: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Run TEE-attested model inference with verifiable signed receipts.

        Phase 3.x.1 Task 5 — wires the prsm.compute.inference module
        (Tasks 1-2) to the HTTP layer so the prsm_inference MCP tool
        (Task 6) can route through this endpoint.

        Mirrors /compute/forge: validate → lock escrow → run executor →
        sign receipt → track privacy budget → release/refund escrow.

        POST body:
        {
            "prompt": "...",                  // required
            "model_id": "mock-llama-3-8b",    // required
            "budget_ftns": 1.0,
            "privacy_tier": "standard",        // none|standard|high|maximum
            "content_tier": "A",               // A|B|C
            "max_tokens": 256,                 // optional
            "temperature": 0.7,                // optional
        }
        """
        # Lazy imports keep the inference module's dependencies (and any
        # future heavy ones like wasmtime/torch) out of the API import graph
        # for nodes that don't run inference.
        import dataclasses
        from decimal import Decimal

        from prsm.compute.inference import (
            ContentTier,
            InferenceRequest,
            sign_receipt,
        )
        from prsm.compute.tee.models import PrivacyLevel

        if not hasattr(node, 'inference_executor') or node.inference_executor is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference executor not initialized. "
                    "This node does not currently serve inference requests."
                ),
            )

        prompt = body.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing 'prompt' field")

        model_id = body.get("model_id", "")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model_id' field")

        budget_ftns = float(body.get("budget_ftns", 1.0))
        if budget_ftns <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "FTNS budget required for inference. "
                    "Set budget_ftns to at least 0.01 FTNS. "
                    "Use the prsm_quote MCP tool to estimate cost first."
                ),
            )

        # Build the request — coerce enum fields, surface bad values as 400s.
        try:
            request = InferenceRequest(
                prompt=prompt,
                model_id=model_id,
                budget_ftns=Decimal(str(budget_ftns)),
                privacy_tier=PrivacyLevel(body.get("privacy_tier", "standard")),
                content_tier=ContentTier(body.get("content_tier", "A")),
                max_tokens=body.get("max_tokens"),
                temperature=body.get("temperature"),
                requester_node_id=node.identity.node_id if node.identity else None,
            )
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

        # Allocate API-side job_id; this becomes the canonical job_id for
        # both the escrow and the signed receipt.
        job_id = "infer-" + _uuid.uuid4().hex[:12]

        # Lock escrow up-front (pre-pay billing pattern per Phase 3.x.1
        # design plan §6.2).
        escrow_entry = None
        if hasattr(node, '_payment_escrow') and node._payment_escrow:
            try:
                escrow_entry = await node._payment_escrow.create_escrow(
                    job_id=job_id,
                    amount=budget_ftns,
                    requester_id=node.identity.node_id if node.identity else "unknown",
                )
            except Exception as e:
                raise HTTPException(
                    status_code=402,
                    detail=f"Escrow creation failed (insufficient FTNS?): {e}",
                )
            # PaymentEscrow.create_escrow returns None (does NOT raise) when
            # the requester has insufficient FTNS balance. Without this guard
            # the request would proceed unbilled.
            if escrow_entry is None:
                raise HTTPException(
                    status_code=402,
                    detail="Insufficient FTNS balance to lock escrow for inference",
                )

        # Privacy budget pre-flight: reject before executor runs if the
        # cumulative DP epsilon would be exceeded. Post-hoc record_spend
        # below commits the spend; can_spend here gates entry.
        expected_epsilon: Optional[float] = None
        if request.privacy_tier != PrivacyLevel.NONE:
            # Route through PrivacyLevel.config_for_level so any future
            # tier↔ε rebalancing in prsm/compute/tee/models.py stays in
            # one place — duplicating the map here would silently desync
            # the pre-flight gate from the executor's actual ε spend.
            expected_epsilon = PrivacyLevel.config_for_level(
                request.privacy_tier
            ).epsilon
            if (
                hasattr(node, 'privacy_budget')
                and node.privacy_budget
                and not node.privacy_budget.can_spend(expected_epsilon)
            ):
                if escrow_entry and node._payment_escrow:
                    try:
                        await node._payment_escrow.refund_escrow(
                            job_id, "privacy budget exhausted"
                        )
                    except Exception as refund_exc:
                        logger.warning(
                            f"Escrow refund after privacy-budget rejection failed: {refund_exc}"
                        )
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Privacy budget exhausted: ε={expected_epsilon} would "
                        f"exceed remaining budget"
                    ),
                )

        try:
            result = await node.inference_executor.execute(request)

            if not result.success:
                # Inference rejected the request (unknown model, budget too
                # low for this executor's pricing, etc.). Refund the escrow.
                if escrow_entry and node._payment_escrow:
                    try:
                        await node._payment_escrow.refund_escrow(
                            job_id, result.error or "inference failed"
                        )
                    except Exception as refund_exc:
                        logger.warning(f"Inference escrow refund failed: {refund_exc}")
                return {
                    "success": False,
                    "error": result.error or "inference failed",
                    "job_id": job_id,
                    "request_id": request.request_id,
                }

            # Replace the executor's internal job_id with the API-side
            # job_id so the receipt's job_id matches the escrow id +
            # response payload, then sign with the node's identity.
            receipt = result.receipt
            if receipt is not None:
                receipt = dataclasses.replace(receipt, job_id=job_id)
                if node.identity:
                    try:
                        receipt = sign_receipt(receipt, node.identity)
                    except Exception as e:
                        logger.warning(f"Receipt signing failed: {e}")

            # Track DP epsilon spend (privacy budget) for non-none privacy
            # tiers — same accounting flow as /compute/forge.
            if (
                hasattr(node, 'privacy_budget')
                and node.privacy_budget
                and request.privacy_tier != PrivacyLevel.NONE
                and receipt is not None
            ):
                try:
                    node.privacy_budget.record_spend(
                        receipt.epsilon_spent,
                        "inference",
                        job_id,
                    )
                except Exception as e:
                    logger.warning(f"Privacy budget tracking failed: {e}")

            # Release escrow to the serving node identity (us, in the
            # local-execution case).
            if escrow_entry and node._payment_escrow:
                try:
                    await node._payment_escrow.release_escrow(
                        job_id=job_id,
                        provider_id=node.identity.node_id if node.identity else "self",
                    )
                except Exception as e:
                    logger.warning(f"Inference escrow release failed: {e}")

            return {
                "success": True,
                "job_id": job_id,
                "request_id": request.request_id,
                "output": result.output,
                "receipt": receipt.to_dict() if receipt is not None else None,
            }

        except HTTPException:
            raise
        except Exception as e:
            # Refund on unexpected failure (executor crashed, etc.)
            if escrow_entry and node._payment_escrow:
                try:
                    await node._payment_escrow.refund_escrow(job_id, str(e))
                except Exception:
                    pass
            logger.error(f"Inference pipeline error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Inference pipeline error: {str(e)}",
            )

    @app.post("/compute/inference/stream")
    async def compute_inference_stream(body: Dict[str, Any] = {}):
        """SSE-streaming sibling of ``/compute/inference``.

        Phase 3.x.8.1 Task 2 — wires
        ``ParallaxScheduledExecutor.execute_streaming`` to a Server-
        Sent-Events HTTP surface so the ``prsm_inference`` MCP tool
        (and any other streaming-capable client) can render token-
        by-token output as the chain produces it.

        Wire format (text/event-stream):

          event: token
          data: {"sequence_index": N, "text_delta": "...",
                 "token_id": null|int, "finish_reason": null|"stop"|...}

          event: result
          data: {"success": true, "request_id": "...", "output": "...",
                 "receipt": {...}, "job_id": "..."}

          event: error
          data: {"error": "...", "code": "...", "job_id": "..."}

        Same input schema as ``/compute/inference``. Same escrow lock
        + refund/settle semantics: pre-execute failure refunds; ANY
        token emitted = settle at full estimate (caller paid for
        chain dispatch — see design plan §3.4 risk register).

        ``event: result`` and ``event: error`` are terminal — server
        closes the connection after them. ``event: token`` events
        are NEVER terminal.
        """
        # Lazy imports — keep the inference module's dependencies out
        # of the API import graph for nodes that don't run inference.
        import dataclasses
        from decimal import Decimal

        from prsm.compute.inference import (
            ContentTier,
            InferenceRequest,
            sign_receipt,
        )
        from prsm.compute.inference.parallax_executor import (
            InferenceTokenEvent,
        )
        from prsm.compute.inference.models import InferenceResult
        from prsm.compute.tee.models import PrivacyLevel

        if not hasattr(node, 'inference_executor') or node.inference_executor is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference executor not initialized. "
                    "This node does not currently serve inference requests."
                ),
            )
        # Streaming requires execute_streaming on the wired executor.
        # ParallaxScheduledExecutor (Phase 3.x.8.1) implements it; the
        # Phase 3.x.1 TensorParallelInferenceExecutor does not. Surface
        # as 503 — operator misconfig.
        if not hasattr(node.inference_executor, "execute_streaming"):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference executor does not support streaming. "
                    "Wire a ParallaxScheduledExecutor (Phase 3.x.8.1) to "
                    "enable /compute/inference/stream."
                ),
            )

        # Validate input identical to /compute/inference.
        prompt = body.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing 'prompt' field")
        model_id = body.get("model_id", "")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model_id' field")
        budget_ftns = float(body.get("budget_ftns", 1.0))
        if budget_ftns <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "FTNS budget required for inference. "
                    "Set budget_ftns to at least 0.01 FTNS."
                ),
            )
        try:
            request = InferenceRequest(
                prompt=prompt,
                model_id=model_id,
                budget_ftns=Decimal(str(budget_ftns)),
                privacy_tier=PrivacyLevel(body.get("privacy_tier", "standard")),
                content_tier=ContentTier(body.get("content_tier", "A")),
                max_tokens=body.get("max_tokens"),
                temperature=body.get("temperature"),
                requester_node_id=node.identity.node_id if node.identity else None,
            )
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

        job_id = "infer-stream-" + _uuid.uuid4().hex[:12]

        # Lock escrow — same pre-pay billing pattern as the unary
        # endpoint. Failure paths refund; success path settles after
        # the terminal result event.
        escrow_entry = None
        if hasattr(node, '_payment_escrow') and node._payment_escrow:
            try:
                escrow_entry = await node._payment_escrow.create_escrow(
                    job_id=job_id,
                    amount=budget_ftns,
                    requester_id=node.identity.node_id if node.identity else "unknown",
                )
            except Exception as e:
                raise HTTPException(
                    status_code=402,
                    detail=f"Escrow creation failed (insufficient FTNS?): {e}",
                )
            if escrow_entry is None:
                raise HTTPException(
                    status_code=402,
                    detail="Insufficient FTNS balance to lock escrow for inference",
                )

        # Privacy-budget pre-flight identical to /compute/inference.
        if request.privacy_tier != PrivacyLevel.NONE:
            expected_epsilon = PrivacyLevel.config_for_level(
                request.privacy_tier
            ).epsilon
            if (
                hasattr(node, 'privacy_budget')
                and node.privacy_budget
                and not node.privacy_budget.can_spend(expected_epsilon)
            ):
                if escrow_entry and node._payment_escrow:
                    try:
                        await node._payment_escrow.refund_escrow(
                            job_id, "privacy budget exhausted"
                        )
                    except Exception as refund_exc:
                        logger.warning(
                            f"Escrow refund after privacy-budget rejection failed: "
                            f"{refund_exc}"
                        )
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Privacy budget exhausted: ε={expected_epsilon} would "
                        f"exceed remaining budget"
                    ),
                )

        async def _event_generator():
            """Drives the executor's streaming generator and frames
            each yielded item as an SSE event. The terminal item
            (success or failure) drives escrow settle/refund + the
            terminal ``result`` or ``error`` event.

            Raises NEVER cross the SSE boundary — any unhandled
            exception encodes a final ``error`` event + refunds the
            escrow defensively."""
            tokens_emitted = 0
            try:
                async for item in node.inference_executor.execute_streaming(
                    request,
                ):
                    if isinstance(item, InferenceTokenEvent):
                        tokens_emitted += 1
                        yield _sse_event("token", _token_event_to_dict(item))
                    elif isinstance(item, InferenceResult):
                        if item.success:
                            await _settle_streaming_escrow(
                                node, job_id, escrow_entry, request,
                                item,
                            )
                            yield _sse_event(
                                "result",
                                _result_to_dict(
                                    item,
                                    job_id=job_id,
                                    identity=node.identity,
                                ),
                            )
                        else:
                            # Phase 3.x.8.1 round-1 M1: settle on
                            # tokens emitted; refund on
                            # zero-tokens-and-failure (pre-execute
                            # gate trip).
                            await _resolve_post_token_billing(
                                node, job_id, escrow_entry,
                                tokens_emitted,
                                item.error or "inference failed",
                            )
                            yield _sse_event("error", {
                                "error": item.error or "inference failed",
                                "code": "EXECUTION_FAILURE",
                                "job_id": job_id,
                                "request_id": request.request_id,
                            })
                        return
                    else:
                        # Defensive — execute_streaming's contract is
                        # InferenceTokenEvent | InferenceResult.
                        await _resolve_post_token_billing(
                            node, job_id, escrow_entry,
                            tokens_emitted,
                            f"unexpected event type {type(item).__name__}",
                        )
                        yield _sse_event("error", {
                            "error": (
                                f"executor yielded unexpected type "
                                f"{type(item).__name__}"
                            ),
                            "code": "INTERNAL_ERROR",
                            "job_id": job_id,
                        })
                        return
                # Generator exhausted without a terminal — protocol
                # violation by the executor.
                await _resolve_post_token_billing(
                    node, job_id, escrow_entry,
                    tokens_emitted,
                    "executor exhausted without terminal result",
                )
                yield _sse_event("error", {
                    "error": (
                        "executor exhausted without yielding a terminal "
                        "InferenceResult"
                    ),
                    "code": "INTERNAL_ERROR",
                    "job_id": job_id,
                })
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Streaming inference pipeline error for job_id=%r",
                    job_id,
                )
                await _resolve_post_token_billing(
                    node, job_id, escrow_entry,
                    tokens_emitted, str(exc),
                )
                yield _sse_event("error", {
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "code": "INTERNAL_ERROR",
                    "job_id": job_id,
                })

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                # Disable proxy buffering — critical for real-time
                # token delivery through nginx / cloudflare /
                # traefik / etc. SSE without these headers can be
                # buffered into a single chunk.
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/billing/{job_id}")
    async def get_billing_status(job_id: str) -> Dict[str, Any]:
        """Return escrow + billing state for a given job_id.

        Phase 3.x.1 Task 7 — backs the prsm_billing_status MCP tool.
        Queries PaymentEscrow.get_escrow() for the job and returns a
        structured billing snapshot. Returns 404 if no escrow exists for
        the given job_id.
        """
        if not hasattr(node, '_payment_escrow') or node._payment_escrow is None:
            raise HTTPException(
                status_code=503,
                detail="Payment escrow not initialized on this node.",
            )

        entry = node._payment_escrow.get_escrow(job_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"No escrow found for job_id={job_id}",
            )

        return {
            "job_id": entry.job_id,
            "escrow_id": entry.escrow_id,
            "requester_id": entry.requester_id,
            "amount_ftns": entry.amount,
            "status": entry.status.value,
            "provider_winner": entry.provider_winner,
            "tx_lock": entry.tx_lock,
            "tx_release": entry.tx_release,
            "created_at": entry.created_at,
            "completed_at": entry.completed_at,
            "metadata": entry.metadata,
        }

    @app.post("/content/upload")
    async def upload_content(req: ContentUploadRequest) -> Dict[str, Any]:
        """Upload text content to ContentStore with provenance tracking."""
        if not node.content_uploader:
            raise HTTPException(status_code=503, detail="Content uploader not initialized")

        result = await node.content_uploader.upload_text(
            text=req.text,
            filename=req.filename,
            replicas=req.replicas,
            royalty_rate=req.royalty_rate,
            parent_cids=req.parent_cids if req.parent_cids else None,
        )

        if not result:
            raise HTTPException(status_code=502, detail="Upload failed — content store unavailable?")

        return {
            "cid": result.cid,
            "filename": result.filename,
            "size_bytes": result.size_bytes,
            "content_hash": result.content_hash,
            "creator_id": result.creator_id,
            "royalty_rate": result.royalty_rate,
            "parent_cids": result.parent_cids,
        }

    @app.post("/content/upload/shard")
    async def upload_shard_dataset(body: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Upload a dataset with semantic sharding.

        Each shard is published through the proprietary BitTorrent
        layer via ``ContentUploader.upload()`` — the same path the
        regular ``/content/upload`` endpoint uses — so the resulting
        ``cid`` for each shard is a real network-discoverable CID
        (no placeholders).

        POST body: {
            "dataset_id": "nada-nc-2025",
            "title": "NADA NC Vehicle Registrations 2025",
            "content_b64": "<base64-encoded content>",
            "shard_count": 4,
            "royalty_rate": 0.05,
            "base_access_fee": 5.0,
            "per_shard_fee": 0.5,
        }
        """
        import base64
        from prsm.data.shard_models import SemanticShard, SemanticShardManifest

        dataset_id = body.get("dataset_id", "")
        title = body.get("title", dataset_id)
        content_b64 = body.get("content_b64", "")
        shard_count = int(body.get("shard_count", 4))
        royalty_rate = float(body.get("royalty_rate", 0.01))

        if not dataset_id:
            raise HTTPException(status_code=400, detail="Missing dataset_id")
        if shard_count < 1:
            raise HTTPException(
                status_code=400, detail="shard_count must be >= 1",
            )

        # Decode content. Empty payload is rejected — the previous
        # placeholder-CID behavior produced non-discoverable manifests
        # that broke `prsm_search_shards` → fetch → royalty-distribution
        # downstream. Per gap-list delta 2026-05-07.
        try:
            content = base64.b64decode(content_b64) if content_b64 else b""
        except Exception:
            raise HTTPException(
                status_code=400, detail="content_b64 is not valid base64",
            )
        if not content:
            raise HTTPException(
                status_code=400,
                detail=(
                    "content_b64 is empty or missing — refusing to "
                    "create a manifest with no payload"
                ),
            )

        # ContentPublisher path requires a wired ContentUploader,
        # mirroring the regular /content/upload endpoint above.
        if not node.content_uploader:
            raise HTTPException(
                status_code=503, detail="Content uploader not initialized",
            )

        # Slice content into shards + publish each through the real
        # BitTorrent layer. Each upload() call returns an UploadedContent
        # with a network-discoverable CID + provenance record.
        chunk_size = max(len(content) // max(shard_count, 1), 1024)
        shards = []
        for i in range(shard_count):
            start = i * chunk_size
            end = min(start + chunk_size, len(content))
            chunk = content[start:end]
            if not chunk:
                # When shard_count > content size / 1024 the trailing
                # shards would be empty. Skip — we can't publish 0
                # bytes through the BT layer. The manifest reports the
                # actual shard count produced.
                continue

            shard_filename = f"{dataset_id}-shard-{i:04d}"
            uploaded = await node.content_uploader.upload(
                content=chunk,
                filename=shard_filename,
                royalty_rate=royalty_rate,
            )
            if uploaded is None:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Shard {i} upload failed — content publisher "
                        f"unavailable or rejected the payload"
                    ),
                )

            shards.append(SemanticShard(
                shard_id=shard_filename,
                parent_dataset=dataset_id,
                cid=uploaded.cid,
                centroid=[float(i) / max(shard_count, 1)],
                record_count=len(chunk),
                size_bytes=len(chunk),
                keywords=[title, f"shard-{i}"],
            ))

        manifest = SemanticShardManifest(
            dataset_id=dataset_id,
            total_records=len(content),
            total_size_bytes=len(content),
            shards=shards,
        )

        # Register with data listing manager if available
        if hasattr(node, 'data_listing_manager') and node.data_listing_manager:
            from prsm.economy.pricing.data_listing import DataListing
            from decimal import Decimal
            listing = DataListing(
                dataset_id=dataset_id,
                owner_id=node.identity.node_id,
                title=title,
                shard_count=shard_count,
                total_size_bytes=len(content),
                base_access_fee=Decimal(str(body.get("base_access_fee", 0))),
                per_shard_fee=Decimal(str(body.get("per_shard_fee", 0))),
            )
            node.data_listing_manager.publish(listing)

        return {
            "dataset_id": dataset_id,
            "title": title,
            "shard_count": len(shards),
            "total_size_bytes": len(content),
            "manifest": manifest.to_dict() if hasattr(manifest, 'to_dict') else str(manifest),
            "shards": [s.to_dict() for s in shards],
            "listing_registered": hasattr(node, 'data_listing_manager') and node.data_listing_manager is not None,
        }

    @app.get("/content/search")
    async def search_content(q: str = "", limit: int = 20) -> Dict[str, Any]:
        """Search the network content index by keyword."""
        if not node.content_index:
            raise HTTPException(status_code=503, detail="Content index not initialized")

        results = node.content_index.search(q, limit=min(limit, 100))
        return {
            "query": q,
            "results": [
                {
                    "cid": r.cid,
                    "filename": r.filename,
                    "size_bytes": r.size_bytes,
                    "content_hash": r.content_hash,
                    "creator_id": r.creator_id,
                    "providers": list(r.providers),
                    "created_at": r.created_at,
                    "metadata": r.metadata,
                    "royalty_rate": r.royalty_rate,
                    "parent_cids": r.parent_cids,
                }
                for r in results
            ],
            "count": len(results),
        }

    @app.get("/content/index/stats")
    async def content_index_stats() -> Dict[str, Any]:
        """Get content index statistics."""
        if not node.content_index:
            raise HTTPException(status_code=503, detail="Content index not initialized")
        return node.content_index.get_stats()

    @app.get("/content/{cid}")
    async def get_content_record(cid: str) -> Dict[str, Any]:
        """Look up a specific content record by CID."""
        if not node.content_index:
            raise HTTPException(status_code=503, detail="Content index not initialized")

        record = node.content_index.lookup(cid)
        if not record:
            raise HTTPException(status_code=404, detail="Content not found in index")

        return {
            "cid": record.cid,
            "filename": record.filename,
            "size_bytes": record.size_bytes,
            "content_hash": record.content_hash,
            "creator_id": record.creator_id,
            "providers": list(record.providers),
            "created_at": record.created_at,
            "metadata": record.metadata,
            "royalty_rate": record.royalty_rate,
            "parent_cids": record.parent_cids,
        }

    class ContentRetrieveResponse(BaseModel):
        """Response model for content retrieval."""
        cid: str
        status: str = Field(description="Retrieval status: 'success', 'not_found', or 'error'")
        data: Optional[str] = Field(
            default=None,
            description="Base64-encoded content data (only if status is 'success')"
        )
        size_bytes: Optional[int] = Field(default=None, description="Size of retrieved content")
        content_hash: Optional[str] = Field(default=None, description="SHA-256 hash of content")
        filename: Optional[str] = Field(default=None, description="Original filename if available")
        providers_tried: int = Field(default=0, description="Number of providers attempted")
        error: Optional[str] = Field(default=None, description="Error message if status is 'error'")

    @app.get("/content/retrieve/{cid}", response_model=ContentRetrieveResponse)
    async def retrieve_content(cid: str, timeout: float = 30.0, verify_hash: bool = True) -> ContentRetrieveResponse:
        """
        Retrieve content from the network by CID.
        
        This endpoint retrieves content from the P2P network using the ContentProvider.
        It will:
        1. Check if content is available locally
        2. Query the content index for providers
        3. Request content from available providers
        4. Verify content hash if verification is enabled
        
        Args:
            cid: PRSM content identifier
            timeout: Seconds to wait for response (default: 30.0)
            verify_hash: Whether to verify SHA-256 hash (default: True)
        
        Returns:
            ContentRetrieveResponse with status and data (base64-encoded) or error
        """
        import base64
        
        if not node.content_provider:
            raise HTTPException(status_code=503, detail="Content provider not initialized")
        
        # Get provider stats before retrieval to determine providers tried
        stats_before = node.content_provider.get_stats()
        
        try:
            content_bytes = await node.content_provider.request_content(
                cid=cid,
                timeout=timeout,
                verify_hash=verify_hash,
            )
            
            if content_bytes is None:
                # Content not found or retrieval failed
                return ContentRetrieveResponse(
                    cid=cid,
                    status="not_found",
                    error="Content not found on any available provider",
                )
            
            # Get content metadata if available
            content_hash = None
            filename = None
            if node.content_index:
                record = node.content_index.lookup(cid)
                if record:
                    content_hash = record.content_hash
                    filename = record.filename
            
            # Encode content as base64 for JSON response
            data_b64 = base64.b64encode(content_bytes).decode('utf-8')
            
            return ContentRetrieveResponse(
                cid=cid,
                status="success",
                data=data_b64,
                size_bytes=len(content_bytes),
                content_hash=content_hash,
                filename=filename,
            )
            
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Content retrieval timed out after {timeout} seconds"
            )
        except Exception as e:
            logger.error(f"Error retrieving content {cid}: {e}")
            return ContentRetrieveResponse(
                cid=cid,
                status="error",
                error=str(e),
            )

    @app.get("/transactions")
    async def get_transactions(limit: int = 50) -> Dict[str, Any]:
        """Get transaction history."""
        if not node.ledger or not node.identity:
            raise HTTPException(status_code=503, detail="Node not initialized")

        history = await node.ledger.get_transaction_history(
            node.identity.node_id, limit=min(limit, 200)
        )
        return {
            "transactions": [
                {
                    "tx_id": tx.tx_id,
                    "type": tx.tx_type.value,
                    "from": tx.from_wallet,
                    "to": tx.to_wallet,
                    "amount": tx.amount,
                    "description": tx.description,
                    "timestamp": tx.timestamp,
                }
                for tx in history
            ],
            "count": len(history),
        }

    # ── Agent endpoints ─────────────────────────────────────────

    @app.get("/agents")
    async def list_agents(local_only: bool = False) -> Dict[str, Any]:
        """List known agents (local and/or remote)."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        if local_only:
            agents = node.agent_registry.get_local_agents()
        else:
            agents = node.agent_registry.get_all_agents()

        return {
            "agents": [a.to_dict() for a in agents],
            "count": len(agents),
        }

    @app.get("/agents/search")
    async def search_agents(capability: str, limit: int = 20) -> Dict[str, Any]:
        """Search agents by capability."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        results = node.agent_registry.search(capability, limit=min(limit, 100))
        return {
            "capability": capability,
            "agents": [a.to_dict() for a in results],
            "count": len(results),
        }

    @app.get("/agents/spending")
    async def agent_spending() -> Dict[str, Any]:
        """Aggregate spending dashboard across all local agents."""
        if not node.agent_registry or not node.ledger:
            raise HTTPException(status_code=503, detail="Not initialized")

        agents = node.agent_registry.get_local_agents()
        spending = []
        for agent in agents:
            allowance = await node.ledger.get_agent_allowance(agent.agent_id)
            spending.append({
                "agent_id": agent.agent_id,
                "agent_name": agent.agent_name,
                "allowance": allowance,
            })
        return {"agents": spending, "count": len(spending)}

    @app.get("/agents/{agent_id}")
    async def get_agent(agent_id: str) -> Dict[str, Any]:
        """Get agent details, spending, and status."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        record = node.agent_registry.lookup(agent_id)
        if not record:
            raise HTTPException(status_code=404, detail="Agent not found")

        result = record.to_dict()
        if node.ledger:
            result["allowance"] = await node.ledger.get_agent_allowance(agent_id)
        return result

    @app.get("/agents/{agent_id}/conversations")
    async def get_agent_conversations(agent_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get recent conversation threads involving an agent."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        conv_ids = node.agent_registry.get_agent_conversations(agent_id, limit=limit)
        conversations = []
        for conv_id in conv_ids:
            messages = node.agent_registry.get_conversation(conv_id)
            conversations.append({
                "conversation_id": conv_id,
                "message_count": len(messages),
                "messages": messages[-5:],  # Last 5 messages per conversation
            })
        return {"conversations": conversations, "count": len(conversations)}

    @app.post("/agents/{agent_id}/allowance")
    async def set_agent_allowance(agent_id: str, amount: float, epoch_hours: float = 24.0) -> Dict[str, Any]:
        """Set or update an agent's spending allowance."""
        if not node.ledger or not node.identity:
            raise HTTPException(status_code=503, detail="Not initialized")

        if amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be positive")

        await node.ledger.grant_agent_allowance(
            principal_id=node.identity.node_id,
            agent_id=agent_id,
            amount=amount,
            epoch_hours=epoch_hours,
        )
        return await node.ledger.get_agent_allowance(agent_id)

    @app.delete("/agents/{agent_id}/allowance")
    async def revoke_agent_allowance(agent_id: str) -> Dict[str, Any]:
        """Revoke an agent's spending authority."""
        if not node.ledger or not node.identity:
            raise HTTPException(status_code=503, detail="Not initialized")

        revoked = await node.ledger.revoke_agent_allowance(
            principal_id=node.identity.node_id,
            agent_id=agent_id,
        )
        if not revoked:
            raise HTTPException(status_code=404, detail="Agent allowance not found")
        return {"agent_id": agent_id, "revoked": True}

    @app.post("/agents/{agent_id}/pause")
    async def pause_agent(agent_id: str) -> Dict[str, Any]:
        """Temporarily suspend an agent."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        record = node.agent_registry.lookup(agent_id)
        if not record:
            raise HTTPException(status_code=404, detail="Agent not found")

        node.agent_registry.set_agent_status(agent_id, "paused")
        return {"agent_id": agent_id, "status": "paused"}

    @app.post("/agents/{agent_id}/resume")
    async def resume_agent(agent_id: str) -> Dict[str, Any]:
        """Resume a paused agent."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        record = node.agent_registry.lookup(agent_id)
        if not record:
            raise HTTPException(status_code=404, detail="Agent not found")

        node.agent_registry.set_agent_status(agent_id, "online")
        return {"agent_id": agent_id, "status": "online"}

    # ── Ledger endpoints ─────────────────────────────────────────

    @app.get("/ledger/sync/stats")
    async def ledger_sync_stats() -> Dict[str, Any]:
        """Get ledger sync statistics."""
        if not node.ledger_sync:
            raise HTTPException(status_code=503, detail="Ledger sync not initialized")
        return node.ledger_sync.get_stats()

    @app.post("/ledger/transfer")
    async def transfer_ftns(to_wallet: str, amount: float) -> Dict[str, Any]:
        """Transfer FTNS to another node (signed, gossip-broadcast)."""
        if not node.ledger_sync:
            raise HTTPException(status_code=503, detail="Ledger sync not initialized")

        if amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be positive")

        tx = await node.ledger_sync.signed_transfer(
            to_wallet=to_wallet,
            amount=amount,
            description=f"API transfer to {to_wallet[:12]}...",
        )
        if not tx:
            raise HTTPException(status_code=400, detail="Insufficient balance")

        return {
            "tx_id": tx.tx_id,
            "from": tx.from_wallet,
            "to": tx.to_wallet,
            "amount": tx.amount,
            "timestamp": tx.timestamp,
        }

    @app.get("/health")
    async def health() -> Dict[str, str]:
        """Simple health check."""
        return {"status": "ok", "node_id": node.identity.node_id if node.identity else "unknown"}

    @app.get("/health/detailed")
    async def health_detailed() -> Dict[str, Any]:
        """Structured per-subsystem readiness probe for ops
        monitoring. Distinct from /health (which stays minimal
        for load-balancer probes).

        Top-level status:
          - healthy: all wired subsystems operational
          - degraded: optional subsystems unavailable but core
            (FTNS ledger + payment escrow) works
          - unhealthy: core subsystem missing or erroring

        Per-subsystem fields: {available, status, ...details, error?}.
        """
        subsystems: Dict[str, Dict[str, Any]] = {}

        # FTNS ledger (core).
        ftns_ledger = getattr(node, "ftns_ledger", None)
        if ftns_ledger is not None:
            try:
                addr = getattr(ftns_ledger, "_connected_address", None)
                init = getattr(ftns_ledger, "_is_initialized", False)
                subsystems["ftns_ledger"] = {
                    "available": bool(init),
                    "status": "ok" if init else "uninitialized",
                    "connected_address": addr,
                }
            except Exception as exc:  # noqa: BLE001
                subsystems["ftns_ledger"] = {
                    "available": False, "status": "error",
                    "error": str(exc),
                }
        else:
            subsystems["ftns_ledger"] = {
                "available": False, "status": "not_wired",
            }

        # Payment escrow (core).
        escrow_svc = getattr(node, "_payment_escrow", None)
        if escrow_svc is not None:
            try:
                pending_count = sum(
                    1 for e in escrow_svc._escrows.values()
                    if e.status.value == "pending"
                )
                subsystems["payment_escrow"] = {
                    "available": True, "status": "ok",
                    "pending_count": pending_count,
                    "default_timeout_sec": getattr(
                        escrow_svc, "default_timeout", None,
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                subsystems["payment_escrow"] = {
                    "available": False, "status": "error",
                    "error": str(exc),
                }
        else:
            subsystems["payment_escrow"] = {
                "available": False, "status": "not_wired",
            }

        # Job history (optional).
        history = getattr(node, "_job_history", None)
        if history is not None:
            try:
                subsystems["job_history"] = {
                    "available": True, "status": "ok",
                    "count": history.size(),
                    "max_entries": history._max_entries,
                    "persisted": history._persist_dir is not None,
                }
            except Exception as exc:  # noqa: BLE001
                subsystems["job_history"] = {
                    "available": False, "status": "error",
                    "error": str(exc),
                }
        else:
            subsystems["job_history"] = {
                "available": False, "status": "not_wired",
            }

        # Royalty distributor (optional).
        royalty = getattr(node, "_royalty_distributor_client", None)
        if royalty is not None:
            try:
                # Probe via claimable() — a read-only call that
                # exercises RPC connectivity.
                _claimable_wei = royalty.claimable()
                subsystems["royalty_distributor"] = {
                    "available": True, "status": "ok",
                    "claimable_wei": _claimable_wei,
                }
            except Exception as exc:  # noqa: BLE001
                subsystems["royalty_distributor"] = {
                    "available": False, "status": "error",
                    "error": str(exc),
                }
        else:
            subsystems["royalty_distributor"] = {
                "available": False, "status": "not_wired",
            }

        # Aggregate status.
        core = ["ftns_ledger", "payment_escrow"]
        optional = ["job_history", "royalty_distributor"]
        core_ok = all(
            subsystems[s]["available"] for s in core
        )
        all_optional_ok = all(
            subsystems[s]["available"] for s in optional
        )
        if not core_ok:
            top_status = "unhealthy"
        elif not all_optional_ok:
            top_status = "degraded"
        else:
            top_status = "healthy"

        return {
            "status": top_status,
            "node_id": (
                node.identity.node_id if node.identity else "unknown"
            ),
            "subsystems": subsystems,
        }

    # ── Batch Settlement Endpoints ─────────────────────────────────

    @app.get("/settlement/stats")
    async def settlement_stats() -> Dict[str, Any]:
        """Get batch settlement queue stats."""
        if not hasattr(node, '_batch_settlement') or not node._batch_settlement:
            return {"enabled": False}
        stats = node._batch_settlement.get_stats()
        stats["enabled"] = True
        return stats

    @app.get("/settlement/pending")
    async def settlement_pending() -> Dict[str, Any]:
        """List pending (un-settled) on-chain transfers."""
        if not hasattr(node, '_batch_settlement') or not node._batch_settlement:
            return {"pending": [], "count": 0}
        pending = node._batch_settlement.get_pending()
        return {"pending": pending, "count": len(pending)}

    @app.post("/settlement/flush")
    async def settlement_flush() -> Dict[str, Any]:
        """Manually trigger batch settlement (flush all pending transfers)."""
        if not hasattr(node, '_batch_settlement') or not node._batch_settlement:
            raise HTTPException(status_code=503, detail="Batch settlement not initialized")
        result = await node._batch_settlement.flush()
        return {
            "settled_count": result.settled_count,
            "total_amount": result.total_amount,
            "net_transfers": result.net_transfers,
            "tx_hashes": result.tx_hashes,
            "errors": result.errors,
            "duration_seconds": result.duration_seconds,
        }

    @app.get("/settlement/history")
    async def settlement_history(limit: int = 10) -> Dict[str, Any]:
        """Get recent settlement history."""
        if not hasattr(node, '_batch_settlement') or not node._batch_settlement:
            return {"history": [], "count": 0}
        history = node._batch_settlement.get_history(limit=limit)
        return {"history": history, "count": len(history)}


    # ── Staking Endpoints ─────────────────────────────────────────

    class StakeRequest(BaseModel):
        """Request body for staking FTNS tokens."""
        amount: float = Field(..., gt=0, description="Amount of FTNS to stake")
        stake_type: str = Field(default="general", description="Type of staking: governance, validation, compute, storage, liquidity, general")
        metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for the stake")

    class StakeResponse(BaseModel):
        """Response model for a stake operation."""
        stake_id: str
        user_id: str
        amount: float
        stake_type: str
        status: str
        staked_at: str
        rewards_earned: float = 0.0

    class UnstakeRequest(BaseModel):
        """Request body for unstaking FTNS tokens."""
        stake_id: str = Field(..., description="ID of the stake to unstake")
        amount: Optional[float] = Field(default=None, gt=0, description="Amount to unstake (None = full stake)")

    class UnstakeResponse(BaseModel):
        """Response model for an unstake operation."""
        request_id: str
        stake_id: str
        user_id: str
        amount: float
        requested_at: str
        available_at: str
        status: str

    class StakingStatusResponse(BaseModel):
        """Response model for staking status."""
        user_id: str
        total_staked: float
        active_stakes: List[Dict[str, Any]]
        pending_unstake_requests: List[Dict[str, Any]]
        total_rewards_earned: float
        total_rewards_claimed: float

    class ClaimRewardsResponse(BaseModel):
        """Response model for claiming rewards."""
        user_id: str
        total_rewards_claimed: float
        stakes_processed: int

    @app.post("/staking/stake", response_model=StakeResponse, tags=["staking"])
    async def stake_tokens(req: StakeRequest) -> StakeResponse:
        """
        Stake FTNS tokens.
        
        Stakes the specified amount of FTNS tokens for the node's identity.
        The tokens will be locked and start earning rewards based on the
        configured annual reward rate.
        
        Args:
            req: StakeRequest containing amount, stake_type, and optional metadata
            
        Returns:
            StakeResponse with the created stake details
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If stake validation fails
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        # Validate stake_type
        from prsm.economy.tokenomics.staking_manager import StakeType
        try:
            stake_type = StakeType(req.stake_type.lower())
        except ValueError:
            valid_types = [t.value for t in StakeType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stake_type: {req.stake_type}. Valid types: {valid_types}"
            )
        
        try:
            from decimal import Decimal
            stake = await node.staking_manager.stake(
                user_id=node.identity.node_id,
                amount=Decimal(str(req.amount)),
                stake_type=stake_type,
                metadata=req.metadata
            )
            
            return StakeResponse(
                stake_id=stake.stake_id,
                user_id=stake.user_id,
                amount=float(stake.amount),
                stake_type=stake.stake_type.value,
                status=stake.status.value,
                staked_at=stake.staked_at.isoformat(),
                rewards_earned=float(stake.rewards_earned)
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error staking tokens: {e}")
            raise HTTPException(status_code=500, detail=f"Staking failed: {str(e)}")

    @app.post("/staking/unstake", response_model=UnstakeResponse, tags=["staking"])
    async def unstake_tokens(req: UnstakeRequest) -> UnstakeResponse:
        """
        Request to unstake FTNS tokens.
        
        Creates an unstake request that will be available for withdrawal
        after the configured unstaking period (default: 7 days).
        
        Args:
            req: UnstakeRequest containing stake_id and optional amount
            
        Returns:
            UnstakeResponse with the unstake request details
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If unstake validation fails
            HTTPException 404: If stake not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            from decimal import Decimal
            amount = Decimal(str(req.amount)) if req.amount else None
            
            unstake_request = await node.staking_manager.unstake(
                user_id=node.identity.node_id,
                stake_id=req.stake_id,
                amount=amount
            )
            
            return UnstakeResponse(
                request_id=unstake_request.request_id,
                stake_id=unstake_request.stake_id,
                user_id=unstake_request.user_id,
                amount=float(unstake_request.amount),
                requested_at=unstake_request.requested_at.isoformat(),
                available_at=unstake_request.available_at.isoformat(),
                status=unstake_request.status.value
            )
        except ValueError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error unstaking tokens: {e}")
            raise HTTPException(status_code=500, detail=f"Unstake failed: {str(e)}")

    @app.get("/staking/status", response_model=StakingStatusResponse, tags=["staking"])
    async def get_staking_status() -> StakingStatusResponse:
        """
        Get staking status for the current node identity.
        
        Returns information about all active stakes, pending unstake requests,
        and reward totals for the node's identity.
        
        Returns:
            StakingStatusResponse with comprehensive staking information
            
        Raises:
            HTTPException 503: If staking manager not initialized
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        user_id = node.identity.node_id
        
        # Get all stakes for the user
        stakes = await node.staking_manager.get_user_stakes(user_id)
        
        # Get pending unstake requests
        pending_requests = await node.staking_manager.get_pending_unstake_requests(user_id)
        
        # Calculate totals
        total_staked = sum(float(s.amount) for s in stakes if s.status.value == "active")
        total_rewards_earned = sum(float(s.rewards_earned) for s in stakes)
        total_rewards_claimed = sum(float(s.rewards_claimed) for s in stakes)
        
        return StakingStatusResponse(
            user_id=user_id,
            total_staked=total_staked,
            active_stakes=[
                {
                    "stake_id": s.stake_id,
                    "amount": float(s.amount),
                    "stake_type": s.stake_type.value,
                    "status": s.status.value,
                    "staked_at": s.staked_at.isoformat(),
                    "rewards_earned": float(s.rewards_earned),
                    "rewards_claimed": float(s.rewards_claimed)
                }
                for s in stakes
            ],
            pending_unstake_requests=[
                {
                    "request_id": r.request_id,
                    "stake_id": r.stake_id,
                    "amount": float(r.amount),
                    "requested_at": r.requested_at.isoformat(),
                    "available_at": r.available_at.isoformat(),
                    "status": r.status.value
                }
                for r in pending_requests
            ],
            total_rewards_earned=total_rewards_earned,
            total_rewards_claimed=total_rewards_claimed
        )

    @app.post("/staking/claim-rewards", response_model=ClaimRewardsResponse, tags=["staking"])
    async def claim_staking_rewards(stake_id: Optional[str] = None) -> ClaimRewardsResponse:
        """
        Claim accumulated staking rewards.
        
        Claims all pending rewards for the node's stakes. If stake_id is provided,
        only claims rewards for that specific stake.
        
        Args:
            stake_id: Optional specific stake ID to claim rewards from
            
        Returns:
            ClaimRewardsResponse with total rewards claimed and stakes processed
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If claim validation fails
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            total_rewards = await node.staking_manager.claim_rewards(
                user_id=node.identity.node_id,
                stake_id=stake_id
            )
            
            # Count stakes processed
            stakes = await node.staking_manager.get_user_stakes(node.identity.node_id)
            stakes_processed = len(stakes) if not stake_id else 1
            
            return ClaimRewardsResponse(
                user_id=node.identity.node_id,
                total_rewards_claimed=float(total_rewards),
                stakes_processed=stakes_processed
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error claiming rewards: {e}")
            raise HTTPException(status_code=500, detail=f"Claim rewards failed: {str(e)}")

    @app.get("/staking/stakes/{stake_id}", tags=["staking"])
    async def get_stake(stake_id: str) -> Dict[str, Any]:
        """
        Get details of a specific stake.
        
        Args:
            stake_id: The ID of the stake to retrieve
            
        Returns:
            Detailed stake information
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 404: If stake not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        stake = await node.staking_manager.get_stake(stake_id)
        if not stake:
            raise HTTPException(status_code=404, detail="Stake not found")
        
        return {
            "stake_id": stake.stake_id,
            "user_id": stake.user_id,
            "amount": float(stake.amount),
            "stake_type": stake.stake_type.value,
            "status": stake.status.value,
            "staked_at": stake.staked_at.isoformat(),
            "rewards_earned": float(stake.rewards_earned),
            "rewards_claimed": float(stake.rewards_claimed),
            "last_reward_calculation": stake.last_reward_calculation.isoformat() if stake.last_reward_calculation else None,
            "lock_reason": stake.lock_reason,
            "metadata": stake.metadata
        }

    @app.get("/staking/unstake-requests/{request_id}", tags=["staking"])
    async def get_unstake_request(request_id: str) -> Dict[str, Any]:
        """
        Get details of a specific unstake request.
        
        Args:
            request_id: The ID of the unstake request to retrieve
            
        Returns:
            Detailed unstake request information
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 404: If request not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        request = await node.staking_manager.get_unstake_request(request_id)
        if not request:
            raise HTTPException(status_code=404, detail="Unstake request not found")
        
        return {
            "request_id": request.request_id,
            "stake_id": request.stake_id,
            "user_id": request.user_id,
            "amount": float(request.amount),
            "requested_at": request.requested_at.isoformat(),
            "available_at": request.available_at.isoformat(),
            "status": request.status.value,
            "completed_at": request.completed_at.isoformat() if request.completed_at else None,
            "cancellation_reason": request.cancellation_reason,
            "is_available": request.is_available
        }

    @app.post("/staking/withdraw/{request_id}", tags=["staking"])
    async def withdraw_unstaked_tokens(request_id: str) -> Dict[str, Any]:
        """
        Withdraw unstaked tokens after the unstaking period.
        
        Completes an unstake request and returns the tokens to the user's
        available balance.
        
        Args:
            request_id: The ID of the unstake request to withdraw
            
        Returns:
            Withdrawal confirmation with amount
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If withdrawal validation fails
            HTTPException 404: If request not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            success, amount = await node.staking_manager.withdraw(
                user_id=node.identity.node_id,
                request_id=request_id
            )
            
            return {
                "request_id": request_id,
                "success": success,
                "amount_withdrawn": float(amount)
            }
        except ValueError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error withdrawing tokens: {e}")
            raise HTTPException(status_code=500, detail=f"Withdrawal failed: {str(e)}")

    @app.post("/staking/cancel-unstake/{request_id}", tags=["staking"])
    async def cancel_unstake_request(request_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel a pending unstake request.
        
        Cancels an unstake request and restores the tokens to active staking.
        
        Args:
            request_id: The ID of the unstake request to cancel
            reason: Optional reason for cancellation
            
        Returns:
            Cancellation confirmation
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If cancellation validation fails
            HTTPException 404: If request not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            success = await node.staking_manager.cancel_unstake(
                user_id=node.identity.node_id,
                request_id=request_id,
                reason=reason
            )
            
            return {
                "request_id": request_id,
                "cancelled": success,
                "reason": reason
            }
        except ValueError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error cancelling unstake: {e}")
            raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")

    # ── Settler Registry (Phase 6: L2-style staking for batch security) ──

    @app.post("/settler/register", tags=["settler", "phase6"])
    async def register_settler(
        settler_id: str,
        address: str,
        bond_amount: float,
    ) -> Dict[str, Any]:
        """
        Register as a batch settler with staked bond.
        
        Settlers stake FTNS to earn the right to approve batch settlements.
        Requires minimum bond (default: 10K FTNS).
        
        Args:
            settler_id: Unique identifier (e.g., node ID)
            address: Ethereum address for on-chain operations
            bond_amount: FTNS to stake as bond
        """
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        try:
            settler = await node._settler_registry.register_settler(
                settler_id=settler_id,
                address=address,
                bond_amount=bond_amount,
            )
            return {
                "settler_id": settler.settler_id,
                "address": settler.address,
                "bond_amount": settler.bond_amount,
                "status": settler.status.value,
                "staked_at": settler.staked_at.isoformat(),
            }
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/settler/{settler_id}", tags=["settler"])
    async def get_settler(settler_id: str) -> Dict[str, Any]:
        """Get details of a specific settler."""
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        settler = node._settler_registry.get_settler(settler_id)
        if not settler:
            raise HTTPException(404, f"Settler {settler_id} not found")
        
        return {
            "settler_id": settler.settler_id,
            "address": settler.address,
            "bond_amount": settler.bond_amount,
            "status": settler.status.value,
            "can_settle": settler.can_settle,
            "total_settled": settler.total_settled,
            "slashed_amount": settler.slashed_amount,
        }

    @app.get("/settler/list/active", tags=["settler"])
    async def list_active_settlers() -> List[Dict[str, Any]]:
        """List all active settlers."""
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        return [
            {
                "settler_id": s.settler_id,
                "address": s.address,
                "bond_amount": s.bond_amount,
                "total_settled": s.total_settled,
            }
            for s in node._settler_registry.list_active_settlers()
        ]

    @app.post("/settler/unbond", tags=["settler"])
    async def unbond_settler(settler_id: str) -> Dict[str, Any]:
        """Initiate unbonding for a settler (30-day lock period)."""
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        try:
            unbond_at = await node._settler_registry.unbond_settler(settler_id)
            return {
                "settler_id": settler_id,
                "status": "unbonding",
                "unbond_at": unbond_at.isoformat() if unbond_at else None,
            }
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/settler/batch/sign", tags=["settler"])
    async def sign_batch(
        batch_id: str,
        settler_id: str,
        signature: str,
    ) -> Dict[str, Any]:
        """
        Sign a pending batch for multi-sig approval.
        
        When signature threshold (default: 3) is reached,
        the batch is approved for on-chain settlement.
        """
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        try:
            sig = await node._settler_registry.sign_batch(
                batch_id=batch_id,
                settler_id=settler_id,
                signature=signature,
            )
            batch = node._settler_registry.get_pending_batch(batch_id)
            return {
                "batch_id": batch_id,
                "settler_id": settler_id,
                "signed_at": sig.signed_at.isoformat(),
                "signature_count": batch.signature_count if batch else 1,
                "threshold": node._settler_registry.settlement_threshold,
                "approved": node._settler_registry.is_batch_approved(batch_id),
            }
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/settler/batch/pending", tags=["settler"])
    async def list_pending_batches() -> List[Dict[str, Any]]:
        """List all batches awaiting multi-sig approval."""
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        return [
            {
                "batch_id": b.batch_id,
                "batch_hash": b.batch_hash,
                "transfer_count": len(b.transfers),
                "total_amount": b.total_amount,
                "signature_count": b.signature_count,
                "approved": b.signature_count >= node._settler_registry.settlement_threshold,
                "created_at": b.created_at.isoformat(),
            }
            for b in node._settler_registry.list_pending_batches()
            if not b.settled
        ]

    @app.get("/settler/ledger/export", tags=["settler"])
    async def export_ledger() -> Dict[str, Any]:
        """
        Export local ledger state for public audit (Challenge system).
        
        Enables anyone to compare local state against on-chain settlement.
        """
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        # Gather ledger data
        ledger_data = {}
        if hasattr(node, "ledger"):
            ledger_data = {
                "balances": dict(node.ledger._balances) if hasattr(node.ledger, "_balances") else {},
                "total_supply": node.ledger.total_supply if hasattr(node.ledger, "total_supply") else 0,
            }
        
        return await node._settler_registry.export_ledger(ledger_data)

    @app.post("/settler/slash/propose", tags=["settler"])
    async def propose_slash(
        settler_id: str,
        slash_amount: float,
        reason: str,
        proposer_id: str,
    ) -> Dict[str, Any]:
        """
        Propose to slash a settler's bond.
        
        Creates a governance proposal that must be approved before execution.
        """
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        try:
            proposal = await node._settler_registry.propose_slash(
                settler_id=settler_id,
                slash_amount=slash_amount,
                reason=reason,
                evidence={},
                proposer_id=proposer_id,
            )
            return {
                "proposal_id": proposal.proposal_id,
                "settler_id": settler_id,
                "slash_amount": slash_amount,
                "reason": reason,
                "status": "pending_vote",
            }
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/settler/stats", tags=["settler"])
    async def get_settler_stats() -> Dict[str, Any]:
        """Get settler registry statistics."""
        if not hasattr(node, "_settler_registry") or not node._settler_registry:
            raise HTTPException(503, "Settler registry not initialized")
        
        return node._settler_registry.get_stats()


    # ── Storage endpoints ────────────────────────────────────────

    @app.get("/storage/stats")
    async def get_storage_stats() -> Dict[str, Any]:
        """Get storage provider statistics."""
        if not node.storage_provider:
            return {
                "available": False,
                "pledged_gb": 0,
                "used_gb": 0,
                "pinned_count": 0,
                "message": "Storage provider not initialized"
            }
        return node.storage_provider.get_stats()

    # ── Compute endpoints ────────────────────────────────────────

    @app.get("/compute/stats", tags=["compute"])
    async def get_compute_stats() -> Dict[str, Any]:
        """Get compute provider statistics."""
        if not node.compute_provider:
            return {
                "available": False,
                "jobs_completed": 0,
                "jobs_queued": 0,
                "message": "Compute provider not initialized"
            }
        return node.compute_provider.get_stats()

    # ── Resource Configuration Endpoints ─────────────────────────────────

    @app.get("/node/resources", response_model=ResourceConfigResponse, tags=["resources"])
    async def get_resources() -> ResourceConfigResponse:
        """Get current node resource configuration.
        
        Returns the current resource allocation settings including:
        - CPU and memory allocation percentages
        - Storage pledge and availability
        - GPU allocation (if available)
        - Bandwidth limits
        - Active hours/days schedule
        - Computed effective values
        """
        if not node.config:
            raise HTTPException(status_code=503, detail="Node configuration not initialized")
        
        config = node.config
        
        # Get compute provider for effective values
        effective_cpu_cores = 0.0
        effective_memory_gb = 0.0
        effective_gpu_memory_gb = None
        
        if node.compute_provider:
            cp = node.compute_provider
            effective_cpu_cores = round(cp.resources.cpu_count * cp.cpu_allocation_pct / 100, 2)
            effective_memory_gb = round(cp.resources.memory_total_gb * cp.memory_allocation_pct / 100, 2)
            if cp.resources.gpu_available:
                effective_gpu_memory_gb = round(cp.resources.gpu_memory_gb * cp.gpu_allocation_pct / 100, 2)
        
        # Get storage available
        storage_available_gb = config.storage_gb
        if node.storage_provider:
            storage_available_gb = node.storage_provider.available_gb
        
        return ResourceConfigResponse(
            cpu_allocation_pct=config.cpu_allocation_pct,
            memory_allocation_pct=config.memory_allocation_pct,
            storage_gb=config.storage_gb,
            max_concurrent_jobs=config.max_concurrent_jobs,
            gpu_allocation_pct=config.gpu_allocation_pct,
            upload_mbps_limit=config.upload_mbps_limit,
            download_mbps_limit=config.download_mbps_limit,
            active_hours_start=config.active_hours_start,
            active_hours_end=config.active_hours_end,
            active_days=config.active_days,
            effective_cpu_cores=effective_cpu_cores,
            effective_memory_gb=effective_memory_gb,
            effective_gpu_memory_gb=effective_gpu_memory_gb,
            storage_available_gb=storage_available_gb,
        )

    @app.put("/node/resources", response_model=ResourceConfigResponse, tags=["resources"])
    async def update_resources(request: ResourceUpdateRequest) -> ResourceConfigResponse:
        """Update node resource allocation settings at runtime.
        
        Changes take effect immediately for storage and bandwidth limits.
        Compute changes (CPU, memory, jobs) take effect on next job acceptance.
        
        All fields are optional - only provided fields will be updated.
        
        Validation:
        - cpu_allocation_pct: 10-90
        - memory_allocation_pct: 10-90
        - gpu_allocation_pct: 10-100
        - active_hours_start/end: 0-23
        - storage_gb: must be positive
        - max_concurrent_jobs: at least 1
        - bandwidth limits: non-negative (0 = unlimited)
        """
        if not node.config:
            raise HTTPException(status_code=503, detail="Node configuration not initialized")
        
        config = node.config
        updates_applied = []
        
        # Validate active_hours consistency if both provided
        if request.active_hours_start is not None and request.active_hours_end is not None:
            if request.active_hours_start >= request.active_hours_end:
                raise HTTPException(
                    status_code=400,
                    detail=f"active_hours_start ({request.active_hours_start}) must be less than active_hours_end ({request.active_hours_end})"
                )
        
        # Validate active_days
        if request.active_days is not None:
            for day in request.active_days:
                if not 0 <= day <= 6:
                    raise HTTPException(
                        status_code=400,
                        detail=f"active_days must be 0-6 (Mon-Sun), got {day}"
                    )
        
        try:
            # Update config fields
            if request.cpu_allocation_pct is not None:
                config.cpu_allocation_pct = request.cpu_allocation_pct
                updates_applied.append(f"cpu_allocation_pct={request.cpu_allocation_pct}")
            
            if request.memory_allocation_pct is not None:
                config.memory_allocation_pct = request.memory_allocation_pct
                updates_applied.append(f"memory_allocation_pct={request.memory_allocation_pct}")
            
            if request.storage_gb is not None:
                config.storage_gb = request.storage_gb
                updates_applied.append(f"storage_gb={request.storage_gb}")
            
            if request.max_concurrent_jobs is not None:
                config.max_concurrent_jobs = request.max_concurrent_jobs
                updates_applied.append(f"max_concurrent_jobs={request.max_concurrent_jobs}")
            
            if request.gpu_allocation_pct is not None:
                config.gpu_allocation_pct = request.gpu_allocation_pct
                updates_applied.append(f"gpu_allocation_pct={request.gpu_allocation_pct}")
            
            if request.upload_mbps_limit is not None:
                config.upload_mbps_limit = request.upload_mbps_limit
                updates_applied.append(f"upload_mbps_limit={request.upload_mbps_limit}")
            
            if request.download_mbps_limit is not None:
                config.download_mbps_limit = request.download_mbps_limit
                updates_applied.append(f"download_mbps_limit={request.download_mbps_limit}")
            
            if request.active_hours_start is not None:
                config.active_hours_start = request.active_hours_start
                updates_applied.append(f"active_hours_start={request.active_hours_start}")
            
            if request.active_hours_end is not None:
                config.active_hours_end = request.active_hours_end
                updates_applied.append(f"active_hours_end={request.active_hours_end}")
            
            if request.active_days is not None:
                config.active_days = request.active_days
                updates_applied.append(f"active_days={request.active_days}")
            
            # Log schedule changes
            if request.active_hours_start is not None or request.active_hours_end is not None:
                start = config.active_hours_start
                end = config.active_hours_end
                if start is not None and end is not None:
                    logger.info(f"Active hours schedule updated: {start:02d}:00 - {end:02d}:00")
                else:
                    logger.info("Active hours schedule cleared: node is always on")
            
            # Update live provider settings
            if node.compute_provider:
                node.compute_provider.update_allocation(
                    cpu_allocation_pct=request.cpu_allocation_pct,
                    memory_allocation_pct=request.memory_allocation_pct,
                    max_concurrent_jobs=request.max_concurrent_jobs,
                    gpu_allocation_pct=request.gpu_allocation_pct,
                )
            
            if node.storage_provider:
                await node.storage_provider.update_limits(
                    pledged_gb=request.storage_gb,
                    upload_mbps_limit=request.upload_mbps_limit,
                    download_mbps_limit=request.download_mbps_limit,
                )
            
            # Persist config to disk
            config.save()
            
            logger.info(f"Resource configuration updated: {', '.join(updates_applied)}")
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to update resource configuration: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")
        
        # Return updated configuration
        return await get_resources()

    # ── Bridge Endpoints ─────────────────────────────────────────

    class BridgeDepositRequest(BaseModel):
        """Request body for bridge deposit operation."""
        amount: float = Field(..., gt=0, description="Amount of FTNS to deposit (in token units)")
        chain_address: str = Field(..., description="Destination on-chain address")
        destination_chain: int = Field(default=137, description="Destination chain ID (default: Polygon mainnet)")

    class BridgeWithdrawRequest(BaseModel):
        """Request body for bridge withdraw operation."""
        amount: float = Field(..., gt=0, description="Amount of FTNS to withdraw (in token units)")
        chain_address: str = Field(..., description="Source on-chain address")
        source_chain: int = Field(default=137, description="Source chain ID (default: Polygon mainnet)")

    class BridgeTransactionResponse(BaseModel):
        """Response model for bridge transactions."""
        transaction_id: str
        direction: str
        user_id: str
        chain_address: str
        amount: str
        source_chain: int
        destination_chain: int
        status: str
        source_tx_hash: Optional[str]
        destination_tx_hash: Optional[str]
        fee_amount: str
        created_at: str
        updated_at: str
        completed_at: Optional[str]
        error_message: Optional[str]

    @app.post("/bridge/deposit", tags=["bridge"])
    async def bridge_deposit(request: BridgeDepositRequest) -> Dict[str, Any]:
        """
        Deposit FTNS tokens from local balance to external chain.
        
        Burns local FTNS and initiates bridge transfer to mint tokens on the destination chain.
        
        Args:
            request: BridgeDepositRequest with amount, chain_address, and destination_chain
            
        Returns:
            Bridge transaction details including transaction_id and status
            
        Raises:
            HTTPException 503: If bridge not initialized
            HTTPException 400: If validation fails (insufficient balance, invalid address, etc.)
            HTTPException 500: If bridge operation fails
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(status_code=503, detail="FTNS bridge not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            # Convert amount to wei (assuming 18 decimals like ETH)
            amount_wei = int(request.amount * 10**18)
            
            # Execute deposit
            tx = await node.ftns_bridge.deposit_to_chain(
                user_id=node.identity.node_id,
                amount=amount_wei,
                chain_address=request.chain_address,
                destination_chain=request.destination_chain
            )
            
            return {
                "success": True,
                "transaction": tx.to_dict()
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Bridge deposit failed: {error_msg}")
            
            # Map specific errors to HTTP status codes
            if "Insufficient" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            elif "outside limits" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            elif "Invalid" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            else:
                raise HTTPException(status_code=500, detail=f"Bridge deposit failed: {error_msg}")

    @app.post("/bridge/withdraw", tags=["bridge"])
    async def bridge_withdraw(request: BridgeWithdrawRequest) -> Dict[str, Any]:
        """
        Withdraw FTNS tokens from external chain to local balance.
        
        Locks on-chain FTNS and initiates bridge transfer to mint local FTNS.
        
        Args:
            request: BridgeWithdrawRequest with amount, chain_address, and source_chain
            
        Returns:
            Bridge transaction details including transaction_id and status
            
        Raises:
            HTTPException 503: If bridge not initialized
            HTTPException 400: If validation fails (insufficient balance, invalid address, etc.)
            HTTPException 500: If bridge operation fails
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(status_code=503, detail="FTNS bridge not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            # Convert amount to wei (assuming 18 decimals like ETH)
            amount_wei = int(request.amount * 10**18)
            
            # Execute withdraw
            tx = await node.ftns_bridge.withdraw_from_chain(
                chain_address=request.chain_address,
                amount=amount_wei,
                user_id=node.identity.node_id,
                source_chain=request.source_chain
            )
            
            return {
                "success": True,
                "transaction": tx.to_dict()
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Bridge withdraw failed: {error_msg}")
            
            # Map specific errors to HTTP status codes
            if "Insufficient" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            elif "outside limits" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            elif "Invalid" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            else:
                raise HTTPException(status_code=500, detail=f"Bridge withdraw failed: {error_msg}")

    @app.get("/bridge/status", tags=["bridge"])
    async def get_bridge_status() -> Dict[str, Any]:
        """
        Get bridge status and pending operations.
        
        Returns:
            Bridge statistics including:
            - total_deposited: Total FTNS deposited to chain
            - total_withdrawn: Total FTNS withdrawn from chain
            - total_fees_collected: Total fees collected
            - pending_transactions: Number of pending transactions
            - completed_transactions: Number of completed transactions
            - failed_transactions: Number of failed transactions
            - limits: Bridge limits (min/max amounts, fees)
            
        Raises:
            HTTPException 503: If bridge not initialized
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(status_code=503, detail="FTNS bridge not initialized")
        
        try:
            stats = await node.ftns_bridge.get_bridge_stats()
            limits = await node.ftns_bridge.get_bridge_limits()
            pending = await node.ftns_bridge.get_pending_transactions()
            
            return {
                "stats": stats.to_dict(),
                "limits": {
                    "min_amount": str(limits.min_amount) if limits else "0",
                    "max_amount": str(limits.max_amount) if limits else "0",
                    "daily_limit": str(limits.daily_limit) if limits else "0",
                    "fee_bps": limits.fee_bps if limits else 0,
                } if limits else None,
                "pending_transactions": [tx.to_dict() for tx in pending],
                "pending_count": len(pending),
            }
        except Exception as e:
            logger.error(f"Failed to get bridge status: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get bridge status: {str(e)}")

    @app.get("/bridge/transactions/{tx_id}", tags=["bridge"])
    async def get_bridge_transaction(tx_id: str) -> Dict[str, Any]:
        """
        Get status of a specific bridge transaction.
        
        Args:
            tx_id: Transaction ID to look up
            
        Returns:
            Bridge transaction details including status, amounts, and timestamps
            
        Raises:
            HTTPException 503: If bridge not initialized
            HTTPException 404: If transaction not found
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(status_code=503, detail="FTNS bridge not initialized")
        
        tx = await node.ftns_bridge.get_bridge_status(tx_id)
        
        if not tx:
            raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")
        
        return {
            "transaction": tx.to_dict()
        }

    @app.get("/bridge/transactions", tags=["bridge"])
    async def list_bridge_transactions(limit: int = 50) -> Dict[str, Any]:
        """
        List bridge transactions for the current user.
        
        Args:
            limit: Maximum number of transactions to return (default: 50, max: 200)
            
        Returns:
            List of bridge transactions for the current user
            
        Raises:
            HTTPException 503: If bridge not initialized
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(status_code=503, detail="FTNS bridge not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        transactions = await node.ftns_bridge.get_user_transactions(
            user_id=node.identity.node_id,
            limit=min(limit, 200)
        )
        
        return {
            "transactions": [tx.to_dict() for tx in transactions],
            "count": len(transactions),
        }

    # ── WebSocket Endpoints ───────────────────────────────────────

    @app.websocket("/ws/status")
    async def websocket_status(websocket: WebSocket):
        """
        WebSocket endpoint for real-time status updates.
        
        Connect to receive live updates about:
        - Node status changes
        - Peer connections/disconnections
        - Job status updates
        - Transaction notifications
        
        Send JSON messages with type field to interact:
        - {"type": "heartbeat"} - Receive heartbeat acknowledgment
        - {"type": "get_status"} - Request current status
        """
        if not app.state.status_websocket:
            await websocket.close(code=1011, reason="WebSocket not initialized")
            return
        
        status_ws = app.state.status_websocket
        
        try:
            await status_ws.connect(websocket)
            
            while True:
                # Wait for messages from client
                data = await websocket.receive_json()
                
                # Handle heartbeat
                if data.get("type") == "heartbeat":
                    await status_ws.send_personal_status(websocket, {
                        "type": "heartbeat_ack",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                # Handle status request
                elif data.get("type") == "get_status":
                    status = await node.get_status()
                    await status_ws.send_personal_status(websocket, {
                        "type": "status_update",
                        "data": status,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
        except WebSocketDisconnect:
            status_ws.disconnect(websocket)
        except Exception as e:
            status_ws.disconnect(websocket)

    # ── Authentication Endpoints ───────────────────────────────────

    @app.get("/auth/verify", tags=["auth"])
    async def verify_token(request: Request, user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
        """
        Verify JWT token and return user information.
        
        Requires valid JWT token in Authorization header.
        """
        return {
            "valid": True,
            "user_id": user.get("user_id"),
            "username": user.get("username"),
            "role": user.get("role"),
            "permissions": user.get("permissions", []),
        }

    # ── FTNS Faucet (development / testnet) ──────────────────────────────

    @app.post("/ftns/faucet", tags=["ftns"])
    async def ftns_faucet(body: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Request FTNS tokens from the faucet (development/testnet only).

        Limited to 100 FTNS per request, max 1000 FTNS total per node.
        Disabled in production (set PRSM_FAUCET_ENABLED=0).
        """
        import os
        if os.environ.get("PRSM_FAUCET_ENABLED", "1") == "0":
            raise HTTPException(status_code=403, detail="Faucet disabled in production")

        amount = min(float(body.get("amount", 100)), 100)  # Max 100 per request
        wallet_id = body.get("wallet_id", node.identity.node_id)

        try:
            balance = await node.ledger.get_balance(wallet_id)
            if balance >= 1000:
                raise HTTPException(status_code=429, detail=f"Wallet already has {balance:.0f} FTNS (max 1000 from faucet)")

            from prsm.node.local_ledger import TransactionType
            await node.ledger.credit(
                wallet_id=wallet_id,
                amount=amount,
                tx_type=TransactionType.WELCOME_GRANT,
                description=f"Faucet grant: {amount} FTNS",
            )
            new_balance = await node.ledger.get_balance(wallet_id)
            return {
                "granted": amount,
                "new_balance": new_balance,
                "wallet_id": wallet_id,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Web Dashboard (served at /, /static, /api/) ──────────────────────────────

    from pathlib import Path as _Path
    from fastapi.staticfiles import StaticFiles as _StaticFiles
    from fastapi.responses import FileResponse as _FileResponse

    _DASHBOARD_TEMPLATES = _Path(__file__).parent.parent / "dashboard" / "templates"
    _DASHBOARD_STATIC = _Path(__file__).parent.parent / "dashboard" / "static"

    # Serve static assets (JS, CSS) from /static/
    if _DASHBOARD_STATIC.exists():
        app.mount("/static", _StaticFiles(directory=str(_DASHBOARD_STATIC)), name="dashboard-static")

    # Serve the SPA shell at / and /dashboard
    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard():
        html_file = _DASHBOARD_TEMPLATES / "dashboard.html"
        if html_file.exists():
            return _FileResponse(str(html_file))
        return {"message": "Dashboard assets not found. Run from PRSM source tree."}

    # Mount the dashboard's API routes at /api/ (all dashboard.js calls use this prefix)
    try:
        from prsm.dashboard.app import create_dashboard_app as _create_dash_app
        _dash_app = _create_dash_app(node=node)
        app.mount("/api", _dash_app, name="dashboard-api")
        logger.info("Web dashboard mounted at /")
    except Exception as e:
        logger.warning(f"Dashboard not available: {e}")

    # ── Apply Security Hardening ───────────────────────────────────
    
    if enable_security and api_hardening:
        # Initialize the hardening components
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, schedule initialization
                asyncio.create_task(api_hardening.initialize())
            else:
                loop.run_until_complete(api_hardening.initialize())
        except RuntimeError:
            # No event loop, create one
            asyncio.run(api_hardening.initialize())
        
        # Apply middleware
        api_hardening.apply_middleware()

    # API key auth for protected endpoints
    from prsm.api.auth_middleware import get_node_auth_middleware, NodeAuthMiddleware
    auth_mw = get_node_auth_middleware(app)
    if auth_mw:
        app.add_middleware(NodeAuthMiddleware, api_key_hash=auth_mw._api_key_hash)

    return app
