"""Request-ID contextvar + log filter for cross-request correlation.

Companion to the X-Request-ID middleware in api.py. The middleware
sets the contextvar on entry to each HTTP request handler; this
module exposes the contextvar + a logging.Filter that injects the
current value as a `request_id` attribute on every LogRecord.

Operators wiring a log formatter with `%(request_id)s` see the
in-flight request's ID alongside every log line emitted during
that request's processing — even from deep callees that have no
direct knowledge of the request.

Outside an HTTP request context, the contextvar resolves to "-" so
log formatters get a benign placeholder rather than KeyError or
``<not set>`` noise.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from typing import Optional


REQUEST_ID_VAR: ContextVar[str] = ContextVar(
    "request_id", default="-",
)


def set_request_id(request_id: str) -> Token:
    """Set the contextvar for the current async context. Returns
    a Token that ``clear_request_id`` consumes to revert."""
    return REQUEST_ID_VAR.set(request_id)


def clear_request_id(token: Optional[Token] = None) -> None:
    """Revert the contextvar to its prior value (or default if no
    token supplied — useful in test setup/teardown)."""
    if token is not None:
        REQUEST_ID_VAR.reset(token)
    else:
        # Best-effort: set back to default. Doesn't truly "reset"
        # the var to "no value set" — there's no contextvar API
        # for that without a Token — but for log-formatter
        # purposes "-" is the equivalent.
        REQUEST_ID_VAR.set("-")


class RequestIdLogFilter(logging.Filter):
    """Injects ``request_id`` attribute on every LogRecord pulled
    from the current contextvar. Always returns True — never drops
    records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_VAR.get()
        return True
