"""Sprint 772 — spot-instance preemption detection.

Vision §4.5 known concern: cloud-spot operators get cheaper
compute but can be killed with ~2min warning. The daemon needs
to know it's been signaled for preemption so it can stop
announcing + (in future sprints) refund in-flight escrow.

Supported backends:
- AWS EC2 spot: GET http://169.254.169.254/latest/meta-data/spot/instance-action
  → 404 when no notice; 200 + JSON `{"action": "...", "time": "..."}` when signaled.
- GCP Compute Engine preemptible: GET http://metadata.google.internal/computeMetadata/v1/instance/preempted
  with `Metadata-Flavor: Google` header → text "TRUE" or "FALSE".

Other backends (Azure spot, OCI preemptible) can be added later
by implementing the `PreemptionBackend` Protocol's `poll()`.

Env:
    PRSM_PREEMPTION_DETECTOR=aws|gcp     (unset = disabled)
    PRSM_PREEMPTION_POLL_INTERVAL_S=10   (default 10s)

Fail-safe semantics:
    Metadata endpoint unreachable → flag stays clear. We'd rather
    miss a preemption signal than mark a healthy non-cloud node
    as preempted.

Monotonic:
    Once the flag is set, it stays set. Preemption is a death
    sentence — the metadata endpoint can flake but the daemon
    is still going down.

Sprint 772 ships the detector + flag + env wiring. Wire-up into
discovery announce loop + inference dispatch gate is sprint 773.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Optional, Protocol


_AWS_ENDPOINT = (
    "http://169.254.169.254/latest/meta-data/spot/instance-action"
)
_GCP_ENDPOINT = (
    "http://metadata.google.internal/computeMetadata/v1/"
    "instance/preempted"
)
_HTTP_TIMEOUT_S = 2.0  # metadata endpoints are link-local; fast


class PreemptionBackend(Protocol):
    """One per cloud provider. Returns True iff this node has
    been told it is going down."""

    async def poll(self) -> bool: ...


class AWSPreemptionBackend:
    """EC2 spot instance-action endpoint."""

    async def poll(self) -> bool:
        try:
            import httpx
            r = httpx.get(_AWS_ENDPOINT, timeout=_HTTP_TIMEOUT_S)
        except Exception:
            return False
        if r.status_code == 404:
            return False
        if r.status_code == 200:
            try:
                body = r.json()
            except Exception:
                return True  # 200 = signal, even if body weird
            return bool(body.get("action"))
        return False


class GCPPreemptionBackend:
    """GCE preemptible-metadata endpoint."""

    async def poll(self) -> bool:
        try:
            import httpx
            r = httpx.get(
                _GCP_ENDPOINT,
                headers={"Metadata-Flavor": "Google"},
                timeout=_HTTP_TIMEOUT_S,
            )
        except Exception:
            return False
        if r.status_code != 200:
            return False
        return (r.text or "").strip().upper() == "TRUE"


class PreemptionDetector:
    """Polls a backend on an interval; flips an internal flag the
    first time a positive signal comes back. Flag is monotonic.

    Sprint 772 stops here. Sprint 773 wires the flag into the
    discovery announce loop + inference dispatch gate."""

    def __init__(
        self,
        backend: PreemptionBackend,
        poll_interval_s: float = 10.0,
    ):
        self.backend = backend
        self.poll_interval_s = max(1.0, float(poll_interval_s))
        self._preempted = False
        self._task: Optional[asyncio.Task[Any]] = None

    def is_preempted(self) -> bool:
        return self._preempted

    async def _poll_once(self) -> None:
        if self._preempted:
            return  # monotonic — no need to poll once flagged
        try:
            if await self.backend.poll():
                self._preempted = True
        except Exception:
            return  # never raise out of the loop

    async def _run_loop(self) -> None:
        while not self._preempted:
            await self._poll_once()
            try:
                await asyncio.sleep(self.poll_interval_s)
            except asyncio.CancelledError:
                return

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None


def resolve_detector_from_env() -> Optional[PreemptionDetector]:
    """Read env. Returns None when disabled / unknown — caller
    should treat None as 'no preemption detection'."""
    kind = (os.environ.get("PRSM_PREEMPTION_DETECTOR") or "").strip().lower()
    backend: Optional[PreemptionBackend]
    if kind == "aws":
        backend = AWSPreemptionBackend()
    elif kind == "gcp":
        backend = GCPPreemptionBackend()
    else:
        return None

    raw_interval = os.environ.get("PRSM_PREEMPTION_POLL_INTERVAL_S")
    try:
        interval = float(raw_interval) if raw_interval else 10.0
    except (ValueError, TypeError):
        interval = 10.0
    return PreemptionDetector(backend, poll_interval_s=interval)


_DETECTOR: Optional[PreemptionDetector] = None


def is_currently_preempted() -> bool:
    """Module-level convenience: True iff a registered detector
    has flipped its flag. False when no detector is registered
    (cloud-agnostic default)."""
    if _DETECTOR is None:
        return False
    return _DETECTOR.is_preempted()


def _set_detector_for_testing(det: Optional[PreemptionDetector]) -> None:
    global _DETECTOR
    _DETECTOR = det


def register_detector(det: Optional[PreemptionDetector]) -> None:
    """Daemon-startup hook: install the resolved detector so
    `is_currently_preempted()` reads correctly across the
    daemon."""
    global _DETECTOR
    _DETECTOR = det


def reset_for_testing() -> None:
    global _DETECTOR
    _DETECTOR = None
