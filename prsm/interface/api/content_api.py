"""
Content API
===========

Authenticated endpoints for uploading content to IPFS with provenance
registration. Distinct from ipfs_api.py (unauthenticated, raw IPFS
passthrough) — this router requires user identity to record the creator
and royalty configuration in the platform database.

Endpoints:
  POST /api/v1/content/upload                 Upload file with provenance
  GET  /api/v1/content/{cid}/provenance        Get provenance record for CID
"""

import hashlib
import time
import structlog
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from prsm.core.auth import get_current_user

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/content", tags=["Content"])

# Royalty rate bounds — matches ContentUploader constants
_MIN_ROYALTY = 0.001
_MAX_ROYALTY = 0.1
_DEFAULT_ROYALTY = 0.01


@router.post("/upload")
async def upload_content_with_provenance(
    file: UploadFile = File(..., description="File to upload to IPFS"),
    description: str = Form("", description="Human-readable description"),
    royalty_rate: float = Form(
        _DEFAULT_ROYALTY,
        description=f"FTNS earned per access ({_MIN_ROYALTY}–{_MAX_ROYALTY})"
    ),
    parent_cids: str = Form(
        "",
        description="Comma-separated CIDs of source material this content derives from"
    ),
    replicas: int = Form(3, ge=1, le=10, description="Replication factor"),
    current_user: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Upload a file to IPFS and register a provenance record.

    The authenticated user becomes the content's creator. When other users
    access this content, they will pay royalties at the configured rate to
    the creator's FTNS account.

    For derivative works (content based on existing IPFS content), supply
    the source CIDs as parent_cids — the 70/25/5 royalty split will apply
    automatically.

    Provenance DB write is non-blocking: if the database is unavailable the
    upload still succeeds and the CID is returned; provenance_registered=false
    in the response signals the gap.
    """
    from prsm.core.ipfs_client import get_ipfs_client
    from prsm.core.database import ProvenanceQueries

    # ── Read file ────────────────────────────────────────────────────────────
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    filename = file.filename or "unnamed"
    size_bytes = len(content)
    content_hash = hashlib.sha256(content).hexdigest()
    royalty_rate = max(_MIN_ROYALTY, min(_MAX_ROYALTY, royalty_rate))
    parent_cid_list: List[str] = [
        c.strip() for c in parent_cids.split(",") if c.strip()
    ]

    # ── Upload to IPFS ───────────────────────────────────────────────────────
    ipfs_client = get_ipfs_client()
    if not ipfs_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="IPFS service not available — ensure an IPFS daemon is running",
        )

    result = await ipfs_client.upload_content(
        content=content,
        filename=filename,
        pin=True,
    )

    if not result.success:
        logger.error("IPFS upload failed", filename=filename, error=result.error)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"IPFS upload failed: {result.error}",
        )

    cid = result.cid
    logger.info("File uploaded to IPFS",
                cid=cid, filename=filename, size=size_bytes, creator=current_user)

    # ── Register provenance ──────────────────────────────────────────────────
    # API-level uploads use the authenticated user_id as creator_id.
    # provenance_signature is empty because the API server doesn't hold a
    # node identity keypair; node-based uploads (ContentUploader) sign with
    # their Ed25519 key. This is a known limitation documented in the record.
    provenance_record = {
        "cid":                      cid,
        "filename":                 filename,
        "size_bytes":               size_bytes,
        "content_hash":             content_hash,
        "creator_id":               current_user,
        "provenance_signature":     "",   # unsigned: API upload, no node key
        "royalty_rate":             royalty_rate,
        "parent_cids":              parent_cid_list,
        "access_count":             0,
        "total_royalties":          0.0,
        "is_sharded":               False,
        "manifest_cid":             None,
        "total_shards":             0,
        "embedding_id":             None,
        "near_duplicate_of":        None,
        "near_duplicate_similarity": None,
        "created_at":               time.time(),
    }

    provenance_registered = False
    try:
        provenance_registered = await ProvenanceQueries.upsert_provenance(
            provenance_record
        )
    except Exception as exc:
        logger.warning("Provenance DB persist failed (upload still succeeded)",
                       cid=cid, error=str(exc))

    return {
        "cid":                    cid,
        "filename":               filename,
        "size_bytes":             size_bytes,
        "content_hash":           content_hash,
        "creator_id":             current_user,
        "royalty_rate":           royalty_rate,
        "parent_cids":            parent_cid_list,
        "replicas_requested":     replicas,
        "provenance_registered":  provenance_registered,
        "access_url":             f"https://ipfs.io/ipfs/{cid}",
    }


@router.get("/{cid}/provenance")
async def get_content_provenance(
    cid: str,
    current_user: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get the provenance record for a content CID.

    Returns creator, royalty configuration, access statistics, and
    derivative lineage (parent_cids) for the given CID. Returns 404 if
    no provenance record exists (content uploaded before Phase 2 Item 1,
    or not uploaded through PRSM).
    """
    from prsm.core.database import ProvenanceQueries

    try:
        record = await ProvenanceQueries.get_provenance(cid)
    except Exception as exc:
        logger.error("Provenance lookup failed", cid=cid, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provenance service temporarily unavailable",
        )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No provenance record found for CID: {cid}",
        )

    return record
