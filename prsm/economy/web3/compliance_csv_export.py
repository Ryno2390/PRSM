"""Sprint 872 — fiat compliance audit ring CSV export.

Production KYB approval (Coinbase full-access, banking-tier
partnerships, etc.) typically requires demonstrating audit-export
capability for AUSTRAC / FinCEN / IRS reporting. PRSM's
sp281+ compliance ring already records every fiat-touching event;
sp872 exposes that data as a canonical CSV.

The export is a single canonical schema (one row per ring entry,
all fields flattened). Operators transform to regulator-specific
formats (AUSTRAC TTR/SMR, FinCEN CTR/SAR, IRS 1099) downstream —
we surface the raw data; we don't presume to file forms.

Filtering supported at export time:
  - `since` (Unix timestamp): include entries with timestamp ≥
  - `until` (Unix timestamp): include entries with timestamp <
  - `user_id`: filter to a single user
  - `kind`: filter to specific entry kinds (onramp_quote, etc.)
  - `min_usd`: filter to entries above a USD threshold (the FinCEN
    $10k CTR threshold is the common one)

Output is RFC 4180-compliant CSV with the header row + UTF-8
encoded. Datetime is ISO 8601 UTC for human + machine readability.
Metadata column is JSON-encoded since regulators accept structured
extension fields.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Set


# Canonical column order — pinned. Changes here are schema-breaking
# for downstream regulator-format transforms.
COMPLIANCE_CSV_HEADER = [
    "entry_id",
    "timestamp_unix",
    "timestamp_iso8601_utc",
    "kind",
    "user_id",
    "usd_amount",
    "ftns_amount",
    "status",
    "kyc_status",
    "tx_hash",
    "vendor_ref",
    "address",
    "jurisdiction",
    "metadata_json",
]


def _iso8601(ts: float) -> str:
    """Unix timestamp → ISO 8601 UTC string."""
    if not ts or ts <= 0:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _entry_passes_filter(
    entry: Any,
    *,
    since: Optional[float] = None,
    until: Optional[float] = None,
    user_id: Optional[str] = None,
    kinds: Optional[Set[str]] = None,
    min_usd: Optional[float] = None,
) -> bool:
    if since is not None and entry.timestamp < since:
        return False
    if until is not None and entry.timestamp >= until:
        return False
    if user_id is not None and entry.user_id != user_id:
        return False
    if kinds is not None and entry.kind not in kinds:
        return False
    if min_usd is not None and (entry.usd_amount or 0) < min_usd:
        return False
    return True


def _entry_to_csv_row(entry: Any) -> List[str]:
    """Canonical row encoding. All None → empty string. Metadata
    → JSON-encoded with stable key ordering."""
    metadata = getattr(entry, "metadata", None) or {}
    metadata_json = (
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        if metadata else ""
    )
    return [
        entry.entry_id,
        str(entry.timestamp),
        _iso8601(entry.timestamp),
        entry.kind,
        entry.user_id or "",
        f"{entry.usd_amount:.6f}" if entry.usd_amount else "0",
        f"{entry.ftns_amount:.6f}" if entry.ftns_amount else "0",
        entry.status or "",
        entry.kyc_status or "",
        entry.tx_hash or "",
        entry.vendor_ref or "",
        entry.address or "",
        entry.jurisdiction or "",
        metadata_json,
    ]


def export_to_csv(
    entries: Iterable[Any],
    *,
    since: Optional[float] = None,
    until: Optional[float] = None,
    user_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
    min_usd: Optional[float] = None,
) -> str:
    """Serialize the filtered ring entries to a CSV string."""
    kind_set = set(kinds) if kinds else None
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COMPLIANCE_CSV_HEADER)
    count = 0
    for entry in entries:
        if not _entry_passes_filter(
            entry, since=since, until=until,
            user_id=user_id, kinds=kind_set, min_usd=min_usd,
        ):
            continue
        writer.writerow(_entry_to_csv_row(entry))
        count += 1
    return buf.getvalue()


def export_summary(
    entries: Iterable[Any],
    *,
    since: Optional[float] = None,
    until: Optional[float] = None,
) -> dict:
    """Compact summary for the CLI / dashboard: counts +
    aggregates per kind. Mirrors the CSV's filter window so
    operators can reconcile."""
    by_kind: dict = {}
    total_usd = 0.0
    total_ftns = 0.0
    total_count = 0
    for entry in entries:
        if since is not None and entry.timestamp < since:
            continue
        if until is not None and entry.timestamp >= until:
            continue
        k = entry.kind
        if k not in by_kind:
            by_kind[k] = {"count": 0, "usd": 0.0, "ftns": 0.0}
        by_kind[k]["count"] += 1
        by_kind[k]["usd"] += entry.usd_amount or 0
        by_kind[k]["ftns"] += entry.ftns_amount or 0
        total_count += 1
        total_usd += entry.usd_amount or 0
        total_ftns += entry.ftns_amount or 0
    return {
        "total_count": total_count,
        "total_usd": total_usd,
        "total_ftns": total_ftns,
        "by_kind": by_kind,
        "filter_since": since,
        "filter_until": until,
    }
