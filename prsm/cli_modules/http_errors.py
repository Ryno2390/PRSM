"""Sprint 647 — shared HTTPException-detail extraction for CLI.

Sprint 646's F25 fix taught us that FastAPI HTTPException response
bodies carry an actionable `detail` field that the default
str(HTTPError) representation swallows ("HTTP Error 503: Service
Unavailable" instead of "...orchestrator not initialized (set
PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY env)"). Sprint 647 generalizes
the fix into a single helper so every CLI command that calls the
local daemon over urllib gets the breadcrumb for free.
"""
from __future__ import annotations

import json
import urllib.error
from typing import Optional


def render_http_error(exc: Exception, group_name: str) -> str:
    """Return the human-readable error message a CLI should print
    when an urllib request raises ``exc``.

    For ``HTTPError`` instances we try to extract the FastAPI
    ``detail`` field from the JSON response body and include it
    in the message. For everything else we fall back to ``str(exc)``.

    ``group_name`` is the operator-facing noun for what the CLI
    was trying to fetch (e.g., "incidents", "TEE status",
    "pipeline jobs"). Lets the rendered message read naturally.
    """
    if isinstance(exc, urllib.error.HTTPError):
        body_msg: Optional[str] = None
        try:
            body_raw = exc.read().decode("utf-8", errors="replace")
            body_json = json.loads(body_raw)
            body_msg = body_json.get("detail")
        except Exception:  # noqa: BLE001
            pass
        if body_msg:
            return (
                f"Failed to fetch {group_name}: "
                f"HTTP {exc.code} — {body_msg}"
            )
        return f"Failed to fetch {group_name}: {exc}"
    return f"Failed to fetch {group_name}: {exc}"
