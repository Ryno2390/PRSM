"""Tests for P2P federation: accept-confirm handshake, target_peers, signatures."""
import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from prsm.node.compute_provider import (
    ComputeProvider,
    ComputeJob,
    JobStatus,
    JobType,
    detect_resources,
)
from prsm.node.compute_requester import ComputeRequester, SubmittedJob
from prsm.node.gossip import (
    GOSSIP_JOB_OFFER,
    GOSSIP_JOB_ACCEPT,
    GOSSIP_JOB_CONFIRM,
    GOSSIP_JOB_CANCEL,
    GOSSIP_JOB_RESULT,
    GOSSIP_PAYMENT_CONFIRM,
)


# ── Fixtures ─────────────────────────────────────────────────────


class FakeIdentity:
    def __init__(self, node_id="node-provider-1"):
        self.node_id = node_id
        self.public_key_b64 = "fake-pub-key-b64"

    def sign(self, data: bytes) -> str:
        return "fake-signature"


class FakeTransport:
    def __init__(self, peer_count=0):
        self.peer_count = peer_count
        # PRSM transport contract surface: subsystems register inbound handlers
        # keyed by message type (MSG_DIRECT, MSG_GOSSIP). See e.g.
        # prsm/node/ledger_sync.py:77, prsm/compute/remote_dispatcher.py:145.
        # The fake just records registrations so tests can drive them directly.
        self._handlers: Dict[str, list] = {}

    def on_message(self, msg_type, handler):
        self._handlers.setdefault(msg_type, []).append(handler)


class FakeGossip:
    def __init__(self):
        self.published: List[tuple] = []
        self._subscribers: Dict[str, list] = {}

    def subscribe(self, subtype: str, handler):
        self._subscribers.setdefault(subtype, []).append(handler)

    async def publish(self, subtype: str, data: dict):
        self.published.append((subtype, data))

    async def deliver(self, subtype: str, data: dict, origin: str = "remote"):
        """Simulate receiving a gossip message."""
        for handler in self._subscribers.get(subtype, []):
            await handler(subtype, data, origin)


class FakeLedger:
    def __init__(self):
        self._balances: Dict[str, float] = {}

    async def get_balance(self, wallet_id: str) -> float:
        return self._balances.get(wallet_id, 100.0)

    async def credit(self, **kwargs):
        return MagicMock(tx_id="tx-credit")

    async def transfer(self, **kwargs):
        return MagicMock(tx_id="tx-transfer")


@pytest.fixture
def provider_identity():
    return FakeIdentity("node-provider-1")


@pytest.fixture
def requester_identity():
    return FakeIdentity("node-requester-1")


@pytest.fixture
def gossip():
    return FakeGossip()


@pytest.fixture
def ledger():
    return FakeLedger()


@pytest.fixture
def transport_no_peers():
    return FakeTransport(peer_count=0)


@pytest.fixture
def transport_with_peers():
    return FakeTransport(peer_count=3)


@pytest.fixture
def provider(provider_identity, transport_no_peers, gossip, ledger):
    p = ComputeProvider(
        identity=provider_identity,
        transport=transport_no_peers,
        gossip=gossip,
        ledger=ledger,
    )
    return p


@pytest.fixture
def provider_multinode(provider_identity, transport_with_peers, gossip, ledger):
    p = ComputeProvider(
        identity=provider_identity,
        transport=transport_with_peers,
        gossip=gossip,
        ledger=ledger,
    )
    return p


def make_job_offer(
    job_id="job-123",
    requester_id="node-requester-1",
    target_peers=None,
    ftns_budget=0.01,
):
    offer = {
        "job_id": job_id,
        "job_type": "inference",
        "requester_id": requester_id,
        "payload": {"prompt": "Hello", "model": "nwtn"},
        "ftns_budget": ftns_budget,
    }
    if target_peers is not None:
        offer["target_peers"] = target_peers
    return offer


# ── Single-node tests (immediate execution) ─────────────────────


@pytest.mark.asyncio
async def test_single_node_immediate_execution(provider, gossip):
    """In single-node mode, provider should execute immediately (no confirm needed)."""
    await provider.start()

    offer = make_job_offer()
    await gossip.deliver(GOSSIP_JOB_OFFER, offer, origin="node-requester-1")

    # Give the async task a moment to run
    await asyncio.sleep(0.2)

    # Job should be in active or completed (executed immediately)
    assert "job-123" in provider.active_jobs or "job-123" in provider.completed_jobs

    # Should have published GOSSIP_JOB_ACCEPT
    accept_msgs = [m for m in gossip.published if m[0] == GOSSIP_JOB_ACCEPT]
    assert len(accept_msgs) == 1
    assert accept_msgs[0][1]["job_id"] == "job-123"


# ── Multi-node tests (accept-confirm handshake) ─────────────────


@pytest.mark.asyncio
async def test_multinode_waits_for_confirm(provider_multinode, gossip):
    """In multi-node mode, provider should NOT execute until confirmed."""
    await provider_multinode.start()

    offer = make_job_offer()
    await gossip.deliver(GOSSIP_JOB_OFFER, offer, origin="node-requester-1")

    # Should NOT be in active_jobs (waiting for confirm)
    assert "job-123" not in provider_multinode.active_jobs
    # Should be in pending confirm
    assert "job-123" in provider_multinode._pending_confirm

    # Should have published GOSSIP_JOB_ACCEPT
    accept_msgs = [m for m in gossip.published if m[0] == GOSSIP_JOB_ACCEPT]
    assert len(accept_msgs) == 1


@pytest.mark.asyncio
async def test_multinode_executes_after_confirm(provider_multinode, gossip):
    """Provider should start execution only after receiving confirm."""
    await provider_multinode.start()

    # Step 1: Offer arrives
    offer = make_job_offer()
    await gossip.deliver(GOSSIP_JOB_OFFER, offer, origin="node-requester-1")

    assert "job-123" in provider_multinode._pending_confirm
    assert "job-123" not in provider_multinode.active_jobs

    # Step 2: Confirm arrives (this provider was chosen)
    await gossip.deliver(GOSSIP_JOB_CONFIRM, {
        "job_id": "job-123",
        "provider_id": "node-provider-1",
        "requester_id": "node-requester-1",
    }, origin="node-requester-1")

    # Should move to active_jobs
    assert "job-123" not in provider_multinode._pending_confirm
    assert "job-123" in provider_multinode.active_jobs


@pytest.mark.asyncio
async def test_multinode_drops_on_other_confirm(provider_multinode, gossip):
    """If another provider is confirmed, we should drop the job."""
    await provider_multinode.start()

    offer = make_job_offer()
    await gossip.deliver(GOSSIP_JOB_OFFER, offer, origin="node-requester-1")
    assert "job-123" in provider_multinode._pending_confirm

    # Confirm goes to a DIFFERENT provider
    await gossip.deliver(GOSSIP_JOB_CONFIRM, {
        "job_id": "job-123",
        "provider_id": "node-provider-OTHER",
        "requester_id": "node-requester-1",
    }, origin="node-requester-1")

    # Should be dropped — not in pending, not in active
    assert "job-123" not in provider_multinode._pending_confirm
    assert "job-123" not in provider_multinode.active_jobs


# ── Target peers enforcement ─────────────────────────────────────


@pytest.mark.asyncio
async def test_target_peers_accepted_when_in_list(provider_multinode, gossip):
    """Provider should accept offers that include us in target_peers."""
    await provider_multinode.start()

    offer = make_job_offer(target_peers=["node-provider-1", "node-provider-2"])
    await gossip.deliver(GOSSIP_JOB_OFFER, offer, origin="node-requester-1")

    # Should have accepted (published accept msg)
    accept_msgs = [m for m in gossip.published if m[0] == GOSSIP_JOB_ACCEPT]
    assert len(accept_msgs) == 1


@pytest.mark.asyncio
async def test_target_peers_rejected_when_not_in_list(provider_multinode, gossip):
    """Provider should reject offers where we're NOT in target_peers."""
    await provider_multinode.start()

    offer = make_job_offer(target_peers=["node-provider-OTHER"])
    await gossip.deliver(GOSSIP_JOB_OFFER, offer, origin="node-requester-1")

    # Should NOT have accepted
    accept_msgs = [m for m in gossip.published if m[0] == GOSSIP_JOB_ACCEPT]
    assert len(accept_msgs) == 0
    assert "job-123" not in provider_multinode._pending_confirm


@pytest.mark.asyncio
async def test_target_peers_empty_accepts_anyone(provider_multinode, gossip):
    """If target_peers is empty/absent, any provider should accept."""
    await provider_multinode.start()

    offer = make_job_offer()  # no target_peers
    await gossip.deliver(GOSSIP_JOB_OFFER, offer, origin="node-requester-1")

    accept_msgs = [m for m in gossip.published if m[0] == GOSSIP_JOB_ACCEPT]
    assert len(accept_msgs) == 1


# ── Job cancellation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_removes_pending(provider_multinode, gossip):
    """Cancel should remove job from pending confirm queue."""
    await provider_multinode.start()

    offer = make_job_offer()
    await gossip.deliver(GOSSIP_JOB_OFFER, offer, origin="node-requester-1")
    assert "job-123" in provider_multinode._pending_confirm

    await gossip.deliver(GOSSIP_JOB_CANCEL, {
        "job_id": "job-123",
        "requester_id": "node-requester-1",
    }, origin="node-requester-1")

    assert "job-123" not in provider_multinode._pending_confirm


# ── Completed jobs cleanup ───────────────────────────────────────


@pytest.mark.asyncio
async def test_completed_jobs_eviction(provider, gossip):
    """Completed jobs dict should not grow beyond _max_completed_jobs."""
    await provider.start()
    provider._max_completed_jobs = 5

    for i in range(10):
        job_id = f"job-{i:03d}"
        offer = make_job_offer(job_id=job_id)
        await gossip.deliver(GOSSIP_JOB_OFFER, offer, origin="node-requester-1")
        await asyncio.sleep(0.3)  # let execution complete

    assert len(provider.completed_jobs) <= 5


# ── No self-credit (double payment fix) ──────────────────────────


@pytest.mark.asyncio
async def test_provider_does_not_self_credit(provider, gossip, ledger):
    """Provider should NOT call ledger.credit on job completion."""
    await provider.start()

    # Spy on ledger.credit
    ledger.credit = AsyncMock(return_value=MagicMock(tx_id="tx-credit"))

    offer = make_job_offer()
    await gossip.deliver(GOSSIP_JOB_OFFER, offer, origin="node-requester-1")
    await asyncio.sleep(0.5)  # let execution complete

    # ledger.credit should NOT have been called
    ledger.credit.assert_not_called()


# ── Requester: confirm on first accept ───────────────────────────


@pytest.mark.asyncio
async def test_requester_publishes_confirm_on_accept(requester_identity, gossip, ledger):
    """Requester should publish GOSSIP_JOB_CONFIRM when first accept arrives."""
    requester = ComputeRequester(
        identity=requester_identity,
        transport=FakeTransport(peer_count=3),
        gossip=gossip,
        ledger=ledger,
    )
    await requester.start()

    # Create a submitted job
    job = SubmittedJob(
        job_id="job-123",
        job_type=JobType.INFERENCE,
        payload={"prompt": "test"},
        ftns_budget=0.01,
    )
    requester.submitted_jobs["job-123"] = job

    # Simulate provider accept
    await gossip.deliver(GOSSIP_JOB_ACCEPT, {
        "job_id": "job-123",
        "provider_id": "node-provider-1",
        "public_key": "fake-key",
    }, origin="node-provider-1")

    # Should have published GOSSIP_JOB_CONFIRM
    confirm_msgs = [m for m in gossip.published if m[0] == GOSSIP_JOB_CONFIRM]
    assert len(confirm_msgs) == 1
    assert confirm_msgs[0][1]["provider_id"] == "node-provider-1"


@pytest.mark.asyncio
async def test_requester_ignores_second_accept(requester_identity, gossip, ledger):
    """Only the first accept should trigger a confirm."""
    requester = ComputeRequester(
        identity=requester_identity,
        transport=FakeTransport(peer_count=3),
        gossip=gossip,
        ledger=ledger,
    )
    await requester.start()

    job = SubmittedJob(
        job_id="job-123",
        job_type=JobType.INFERENCE,
        payload={"prompt": "test"},
        ftns_budget=0.01,
    )
    requester.submitted_jobs["job-123"] = job

    # First accept
    await gossip.deliver(GOSSIP_JOB_ACCEPT, {
        "job_id": "job-123",
        "provider_id": "node-provider-1",
        "public_key": "fake-key",
    }, origin="node-provider-1")

    # Second accept from different provider
    await gossip.deliver(GOSSIP_JOB_ACCEPT, {
        "job_id": "job-123",
        "provider_id": "node-provider-2",
        "public_key": "fake-key-2",
    }, origin="node-provider-2")

    # Only ONE confirm should have been published
    confirm_msgs = [m for m in gossip.published if m[0] == GOSSIP_JOB_CONFIRM]
    assert len(confirm_msgs) == 1
    assert confirm_msgs[0][1]["provider_id"] == "node-provider-1"


# ── Signature verification ───────────────────────────────────────


@pytest.mark.asyncio
async def test_requester_rejects_unsigned_remote_result(requester_identity, gossip, ledger):
    """Remote provider results without valid signature should be rejected."""
    requester = ComputeRequester(
        identity=requester_identity,
        transport=FakeTransport(peer_count=3),
        gossip=gossip,
        ledger=ledger,
    )
    await requester.start()

    job = SubmittedJob(
        job_id="job-456",
        job_type=JobType.INFERENCE,
        payload={"prompt": "test"},
        ftns_budget=0.01,
    )
    requester.submitted_jobs["job-456"] = job

    # Result WITHOUT signature from remote provider
    await gossip.deliver(GOSSIP_JOB_RESULT, {
        "job_id": "job-456",
        "provider_id": "node-provider-remote",
        "status": "completed",
        "result": {"response": "fake answer"},
        # No signature!
    }, origin="node-provider-remote")

    # Job should NOT be marked completed (result rejected)
    assert job.status != JobStatus.COMPLETED
    assert job.result is None


# ── Job ID passthrough ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_job_uses_provided_job_id(requester_identity, gossip, ledger):
    """submit_job should use the provided job_id if given."""
    requester = ComputeRequester(
        identity=requester_identity,
        transport=FakeTransport(peer_count=0),
        gossip=gossip,
        ledger=ledger,
    )
    await requester.start()

    submitted = await requester.submit_job(
        job_type=JobType.INFERENCE,
        payload={"prompt": "test"},
        ftns_budget=0.0,
        job_id="my-custom-job-id-123",
    )

    assert submitted.job_id == "my-custom-job-id-123"


@pytest.mark.asyncio
async def test_submit_job_generates_id_when_not_provided(requester_identity, gossip, ledger):
    """submit_job should auto-generate job_id when none given."""
    requester = ComputeRequester(
        identity=requester_identity,
        transport=FakeTransport(peer_count=0),
        gossip=gossip,
        ledger=ledger,
    )
    await requester.start()

    submitted = await requester.submit_job(
        job_type=JobType.INFERENCE,
        payload={"prompt": "test"},
        ftns_budget=0.0,
    )

    assert submitted.job_id  # Should be auto-generated
    assert submitted.job_id != ""
