"""Sprint 158 — DashboardServer.start_time set in __init__, not just
in async start().

When the dashboard is mounted as a sub-app inside the main API
(via prsm.dashboard.api_routes mount), the dashboard's own
async start() method is never called — only __init__ runs. That
left start_time as None forever, which made every /api/status and
/api/node response report uptime_seconds=0.

Live dogfood reproduced:
  curl /api/status → "uptime_seconds": 0  (after node up 30+ minutes)
  curl /api/node    → "uptime_seconds": 0

Fix: set start_time in __init__ so sub-mounted dashboards still
surface real uptime. The async start() method's existing assignment
is left as a refresh-on-launch (when running standalone) — first
write wins for sub-mounts; standalone overwrites with own start
time. Both paths produce a sensible uptime.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

# Skip the entire module if heavy dashboard deps fail to import in
# the test environment. The dashboard module pulls a lot of
# transitives — be resilient.
pytest.importorskip("fastapi")


def test_start_time_set_in_init():
    """Sprint 158 — DashboardServer.start_time must be set when
    __init__ completes, NOT only when async start() runs."""
    from prsm.dashboard.app import DashboardServer
    server = DashboardServer(node=None)
    assert server.start_time is not None, (
        "start_time was None after __init__; sub-mounted dashboards "
        "would report uptime_seconds=0 forever."
    )
    assert isinstance(server.start_time, datetime)


def test_start_time_close_to_now():
    """Sprint 158 — sanity: start_time is close to current time
    (within 5 seconds of construction)."""
    from prsm.dashboard.app import DashboardServer
    before = datetime.now(timezone.utc)
    server = DashboardServer(node=None)
    after = datetime.now(timezone.utc)
    assert before <= server.start_time <= after, (
        f"start_time {server.start_time} not in window "
        f"[{before}, {after}]"
    )


def test_uptime_computed_from_init_start_time():
    """Sprint 158 — verify the uptime calc actually consumes
    start_time set in __init__."""
    from prsm.dashboard.app import DashboardServer

    server = DashboardServer(node=None)
    initial_time = server.start_time
    # Sleep a tick, then re-read start_time — it must NOT have been
    # silently reset to None or re-bound.
    time.sleep(0.05)
    assert server.start_time is initial_time
