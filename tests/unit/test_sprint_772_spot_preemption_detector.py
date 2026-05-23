"""Sprint 772 — spot-instance preemption detection.

Vision §4.5 known concern: cloud-spot operators get cheaper
compute but can be killed with ~2min warning. The daemon needs
to know it's been signaled for preemption so it can stop
announcing + (in future sprints) refund in-flight escrow.

This sprint ships the foundation:
- A `PreemptionDetector` abstraction with concrete AWS + GCP
  backends.
- A flag (`is_preempted()`) other subsystems can check.
- Background poll loop driven by `PRSM_PREEMPTION_POLL_INTERVAL_S`.
- Safe default: detector DISABLED unless
  `PRSM_PREEMPTION_DETECTOR=aws|gcp`.

Network errors / metadata-server unreachable → flag stays clear
(fail-safe; we'd rather miss a preemption than mark a healthy
non-cloud node as preempted).

Pin tests:
- Module exports.
- AWS backend: 404 → no notice; 200 + JSON → notice.
- GCP backend: "FALSE" → no notice; "TRUE" → notice.
- env resolution: disabled / aws / gcp / unknown.
- Detector class: flag toggles on notice + stays set (preemption
  is monotonic — once signaled, you're going down).
- Fail-safe: network error → flag stays clear.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch


def setup_function():
    for k in (
        "PRSM_PREEMPTION_DETECTOR",
        "PRSM_PREEMPTION_POLL_INTERVAL_S",
    ):
        os.environ.pop(k, None)


def teardown_function():
    for k in (
        "PRSM_PREEMPTION_DETECTOR",
        "PRSM_PREEMPTION_POLL_INTERVAL_S",
    ):
        os.environ.pop(k, None)


# ---- Module exports ---------------------------------------------


def test_module_exports():
    """All public symbols importable."""
    from prsm.node.preemption import (  # noqa: F401
        PreemptionDetector,
        AWSPreemptionBackend,
        GCPPreemptionBackend,
        resolve_detector_from_env,
        is_currently_preempted,
        reset_for_testing,
    )


# ---- AWS backend ------------------------------------------------


def test_aws_backend_404_means_no_notice():
    """AWS spot-instance-action endpoint returns 404 when no
    preemption notice is pending. That's the steady state."""
    from prsm.node.preemption import AWSPreemptionBackend
    backend = AWSPreemptionBackend()
    fake = MagicMock()
    fake.status_code = 404
    with patch("httpx.get", return_value=fake):
        assert asyncio.run(backend.poll()) is False


def test_aws_backend_200_means_preemption():
    """200 + JSON payload with `action` field = AWS told us we
    are going down."""
    from prsm.node.preemption import AWSPreemptionBackend
    backend = AWSPreemptionBackend()
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "action": "terminate",
        "time": "2026-05-23T12:00:00Z",
    }
    with patch("httpx.get", return_value=fake):
        assert asyncio.run(backend.poll()) is True


def test_aws_backend_network_error_fail_safe():
    """Metadata server unreachable (e.g. running on bare metal
    with the env set anyway) → False. Never raise; never claim
    preemption from an error."""
    from prsm.node.preemption import AWSPreemptionBackend
    backend = AWSPreemptionBackend()
    with patch(
        "httpx.get",
        side_effect=ConnectionError("unreachable"),
    ):
        assert asyncio.run(backend.poll()) is False


# ---- GCP backend ------------------------------------------------


def test_gcp_backend_false_means_no_notice():
    """GCP preempted-metadata returns plain string 'FALSE'
    until preemption is signaled, then flips to 'TRUE'."""
    from prsm.node.preemption import GCPPreemptionBackend
    backend = GCPPreemptionBackend()
    fake = MagicMock()
    fake.status_code = 200
    fake.text = "FALSE"
    with patch("httpx.get", return_value=fake):
        assert asyncio.run(backend.poll()) is False


def test_gcp_backend_true_means_preemption():
    from prsm.node.preemption import GCPPreemptionBackend
    backend = GCPPreemptionBackend()
    fake = MagicMock()
    fake.status_code = 200
    fake.text = "TRUE"
    with patch("httpx.get", return_value=fake):
        assert asyncio.run(backend.poll()) is True


def test_gcp_backend_requires_metadata_flavor_header():
    """The GCP metadata server REQUIRES the `Metadata-Flavor:
    Google` header. Without it, the server rejects with 403.
    Pin that our backend sets it."""
    from prsm.node.preemption import GCPPreemptionBackend
    backend = GCPPreemptionBackend()
    fake = MagicMock()
    fake.status_code = 200
    fake.text = "FALSE"
    with patch("httpx.get", return_value=fake) as mg:
        asyncio.run(backend.poll())
    kwargs = mg.call_args.kwargs
    headers = kwargs.get("headers") or {}
    assert headers.get("Metadata-Flavor") == "Google"


# ---- env resolution ---------------------------------------------


def test_resolve_disabled_by_default():
    """Unset env → None (no detector). Safe default."""
    from prsm.node.preemption import resolve_detector_from_env
    assert resolve_detector_from_env() is None


def test_resolve_aws():
    from prsm.node.preemption import (
        resolve_detector_from_env,
        AWSPreemptionBackend,
    )
    os.environ["PRSM_PREEMPTION_DETECTOR"] = "aws"
    det = resolve_detector_from_env()
    assert det is not None
    assert isinstance(det.backend, AWSPreemptionBackend)


def test_resolve_gcp():
    from prsm.node.preemption import (
        resolve_detector_from_env,
        GCPPreemptionBackend,
    )
    os.environ["PRSM_PREEMPTION_DETECTOR"] = "gcp"
    det = resolve_detector_from_env()
    assert det is not None
    assert isinstance(det.backend, GCPPreemptionBackend)


def test_resolve_unknown_returns_none():
    """Typo in env → no detector. Don't crash the daemon."""
    from prsm.node.preemption import resolve_detector_from_env
    os.environ["PRSM_PREEMPTION_DETECTOR"] = "xyz"
    assert resolve_detector_from_env() is None


# ---- PreemptionDetector class -----------------------------------


def test_detector_flag_starts_clear():
    """Fresh detector: not preempted."""
    from prsm.node.preemption import (
        PreemptionDetector,
        AWSPreemptionBackend,
    )
    det = PreemptionDetector(AWSPreemptionBackend())
    assert det.is_preempted() is False


def test_detector_flag_sets_on_notice():
    """One positive poll → flag set."""
    from prsm.node.preemption import PreemptionDetector

    backend = MagicMock()
    backend.poll = AsyncMock(return_value=True)
    det = PreemptionDetector(backend)
    asyncio.run(det._poll_once())
    assert det.is_preempted() is True


def test_detector_flag_monotonic():
    """Once set, the flag stays set even if a subsequent poll
    returns False. Preemption is a death sentence; the metadata
    endpoint can flake but the daemon is still going down."""
    from prsm.node.preemption import PreemptionDetector

    backend = MagicMock()
    # First poll: preempted. Second: backend recovers.
    backend.poll = AsyncMock(side_effect=[True, False])
    det = PreemptionDetector(backend)
    asyncio.run(det._poll_once())
    asyncio.run(det._poll_once())
    assert det.is_preempted() is True


# ---- module-level cache + is_currently_preempted ----------------


def test_is_currently_preempted_no_detector_returns_false():
    """No detector configured → never preempted (cloud agnostic
    default)."""
    from prsm.node.preemption import (
        is_currently_preempted, reset_for_testing,
    )
    reset_for_testing()
    assert is_currently_preempted() is False


def test_is_currently_preempted_after_flag_set():
    """When a detector is registered + flagged, the global helper
    reflects it."""
    from prsm.node.preemption import (
        PreemptionDetector,
        _set_detector_for_testing,
        is_currently_preempted,
        reset_for_testing,
    )
    reset_for_testing()
    backend = MagicMock()
    backend.poll = AsyncMock(return_value=True)
    det = PreemptionDetector(backend)
    asyncio.run(det._poll_once())
    _set_detector_for_testing(det)
    assert is_currently_preempted() is True
    reset_for_testing()
