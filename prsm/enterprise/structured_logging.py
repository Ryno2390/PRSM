"""Sprint 318d — structured JSON logging for ops.

PRSM logs natively use f-strings + plaintext formatters,
which works fine for developer-side `tail -f` but loses
structure when piped to an ops aggregator (Loki / Datadog
/ ELK / OTEL). This module ships a JSON formatter that
emits one parseable line per record with standard ops
fields + any `extra=` kwargs the caller passed.

Operators install the formatter once at process startup:

    from prsm.enterprise.structured_logging import (
        configure_json_logging,
    )
    configure_json_logging()

Sprint 318d ships the helper + formatter. A future
sprint can retrofit existing logger.info() calls to pass
structured fields via `extra=` (job_id, round_index,
node_id, etc.) — the formatter already captures them
when present.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from typing import Any, Dict, Optional


# Standard LogRecord attributes — anything else on a
# record came from `extra=` and should be captured as a
# structured field.
_STANDARD_RECORD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno",
    "pathname", "filename", "module", "exc_info",
    "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread",
    "threadName", "processName", "process",
    "message", "taskName",  # Python 3.12+ adds taskName
})


class JsonLogFormatter(logging.Formatter):
    """Emit one JSON-line per record. Standard fields:
    timestamp (ISO 8601 UTC), level, logger, msg.
    Anything in `record.__dict__` that isn't a standard
    LogRecord attribute is included as a structured
    field — this captures `extra=` kwargs."""

    def format(
        self, record: logging.LogRecord,
    ) -> str:
        payload: Dict[str, Any] = {
            "timestamp": _iso8601_utc(record.created),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Capture extra= fields. record.__dict__ contains
        # every attribute including the standard ones; we
        # exclude the standard ones.
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_ATTRS:
                continue
            if key.startswith("_"):
                continue
            payload[key] = _json_safe(value)
        if record.exc_info:
            payload["exc_info"] = "".join(
                traceback.format_exception(
                    *record.exc_info,
                ),
            )
        return json.dumps(payload, default=str)


def _iso8601_utc(created_unix: float) -> str:
    """Render a float unix timestamp as ISO 8601 UTC
    (e.g., 2026-05-12T15:32:00.123Z). Ops aggregators
    parse this format without configuration."""
    ms = int((created_unix - int(created_unix)) * 1000)
    return (
        time.strftime(
            "%Y-%m-%dT%H:%M:%S",
            time.gmtime(created_unix),
        )
        + f".{ms:03d}Z"
    )


def _json_safe(value: Any) -> Any:
    """If a logger passes a non-JSON-serializable
    object via extra=, render it as a string rather
    than crash the formatter."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def configure_json_logging(
    *,
    level: int = logging.INFO,
    stream=None,
) -> JsonLogFormatter:
    """Install the JsonLogFormatter on the root logger's
    stream handler. Returns the formatter instance for
    test inspection. Idempotent — calling twice doesn't
    stack handlers."""
    formatter = JsonLogFormatter()
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any prsm-installed JsonLogFormatter handlers
    # so subsequent calls are idempotent.
    for handler in list(root.handlers):
        if getattr(handler, "_prsm_json", False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(formatter)
    handler._prsm_json = True  # mark for idempotent re-config
    root.addHandler(handler)
    return formatter
