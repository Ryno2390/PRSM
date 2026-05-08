"""SwarmDispatcherAdapter — thin orchestration around AgentDispatcher.

The adapter satisfies the `SwarmDispatcher` Protocol (from
`swarm_runner.py`) by fanning out one `MobileAgent` per shard via the
existing requester-side `AgentDispatcher`, then collecting the results
back into `PartialResult` objects.

This module's tests verify that:
  - per-shard MobileAgents are built carrying the input manifest
  - `dispatch(agent)` and `wait_for_result(agent_id)` are awaited
    once per shard
  - result-dict fields are threaded into PartialResult correctly
    (shard_cid + creator_id from shard; payload + agent_signature +
    dp_noise_applied from the result)
  - timeouts (None from wait_for_result) cause that shard to be
    skipped — orchestrator's retry-loop owns retry policy
  - empty shard list short-circuits without invoking dispatcher
  - per-shard FTNS budget is propagated onto each MobileAgent
  - the adapter satisfies the runtime SwarmDispatcher Protocol

The adapter does NOT apply DP noise, sign partials, verify signatures,
or do bid selection — those belong to the source agent / aggregator /
AgentDispatcher respectively.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.compute.agents.instruction_set import (
    AgentInstruction,
    AgentOp,
    InstructionManifest,
)
from prsm.compute.agents.models import (
    DispatchRecord,
    DispatchStatus,
    MobileAgent,
)
from prsm.compute.query_orchestrator import (
    PartialResult,
    ShardCandidate,
    SwarmDispatcher,
)
from prsm.compute.query_orchestrator.swarm_dispatcher_adapter import (
    SwarmDispatcherAdapter,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

# Minimal valid WASM binary — magic bytes + version. MobileAgent's
# __post_init__ enforces the magic-byte check.
_WASM_STUB = b"\x00asm" + b"\x01\x00\x00\x00"


def _manifest() -> InstructionManifest:
    return InstructionManifest(
        query="count records",
        instructions=[AgentInstruction(op=AgentOp.COUNT)],
    )


def _shards(n: int = 3) -> list[ShardCandidate]:
    return [
        ShardCandidate(
            cid=f"prsm:shard-{i}",
            similarity=0.9 - i * 0.01,
            creator_id=f"creator-{i}",
            holder_node_ids=(f"node-{i}",),
        )
        for i in range(n)
    ]


def _result_dict(
    *,
    payload: bytes = b"shard-output",
    signature: bytes = b"\x99" * 64,
    dp_noise: bool = True,
) -> dict:
    return {
        "payload": payload,
        "agent_signature": signature,
        "dp_noise_applied": dp_noise,
    }


def _stub_dispatcher(results: list[dict | None]) -> MagicMock:
    """Build a MagicMock AgentDispatcher whose dispatch+wait return
    one canned result per call (in order)."""
    ad = MagicMock()

    # dispatch returns a DispatchRecord for the agent it was given
    async def _dispatch(agent: MobileAgent, **_kwargs) -> DispatchRecord:
        return DispatchRecord(
            agent_id=agent.agent_id,
            origin_node="origin-node",
            target_node="",
            ftns_budget=agent.ftns_budget,
            status=DispatchStatus.BIDDING,
        )

    ad.dispatch = AsyncMock(side_effect=_dispatch)

    # wait_for_result yields the canned results in order
    results_iter = iter(results)

    async def _wait(agent_id: str, *, timeout: float | None = None):
        try:
            return next(results_iter)
        except StopIteration:
            return None

    ad.wait_for_result = AsyncMock(side_effect=_wait)
    return ad


def _adapter(
    ad: MagicMock,
    *,
    per_shard_budget_ftns: int = 10,
    ttl_seconds: int = 60,
    result_timeout_seconds: float = 30.0,
) -> SwarmDispatcherAdapter:
    return SwarmDispatcherAdapter(
        ad,
        wasm_executor_binary=_WASM_STUB,
        per_shard_budget_ftns=per_shard_budget_ftns,
        ttl_seconds=ttl_seconds,
        result_timeout_seconds=result_timeout_seconds,
    )


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_three_shards_three_partials():
    """3 shards → 3 PartialResults with fields threaded correctly."""
    shards = _shards(3)
    results = [
        _result_dict(payload=f"out-{i}".encode(), signature=bytes([i]) * 64)
        for i in range(3)
    ]
    ad = _stub_dispatcher(results)
    adapter = _adapter(ad)

    out = await adapter.fan_out(_manifest(), shards)

    assert len(out) == 3
    for i, partial in enumerate(out):
        assert isinstance(partial, PartialResult)
        # shard-derived fields
        assert partial.shard_cid == shards[i].cid
        assert partial.creator_id == shards[i].creator_id
        # result-derived fields
        assert partial.payload == f"out-{i}".encode()
        assert partial.agent_signature == bytes([i]) * 64
        assert partial.dp_noise_applied is True


@pytest.mark.asyncio
async def test_per_shard_mobile_agent_built_with_proper_agent_manifest():
    """The dispatcher's `dispatch` receives an agent whose manifest
    is a real `AgentManifest` (resource declaration), NOT the
    InstructionManifest. The InstructionManifest is the WASM-input
    payload — it's a separate concern from the agent's resource
    declaration. This pin closes the type-unification follow-on:
    `agent.manifest.to_dict()` (called by the dispatcher's gossip
    payload at `prsm/compute/agents/dispatcher.py:135`) now produces
    the correct AgentManifest dict shape."""
    from prsm.compute.agents.models import AgentManifest

    shards = _shards(2)
    manifest = _manifest()
    ad = _stub_dispatcher([_result_dict(), _result_dict()])
    adapter = _adapter(ad)

    await adapter.fan_out(manifest, shards)

    assert ad.dispatch.await_count == 2
    for i, call in enumerate(ad.dispatch.await_args_list):
        agent = call.args[0] if call.args else call.kwargs["agent"]
        assert isinstance(agent, MobileAgent)
        # The agent's manifest is a real AgentManifest with the
        # shard's CID in required_content_ids — NOT the
        # InstructionManifest payload.
        assert isinstance(agent.manifest, AgentManifest)
        assert agent.manifest.required_content_ids == [shards[i].cid]
        # to_dict() produces the AgentManifest dict shape, not the
        # InstructionManifest dict shape.
        d = agent.manifest.to_dict()
        assert "required_content_ids" in d
        assert "min_hardware_tier" in d


@pytest.mark.asyncio
async def test_timeout_skips_shard():
    """1 shard returning None from wait_for_result is skipped; the
    other two still appear in the output, in shard-order."""
    shards = _shards(3)
    # Middle shard times out (None)
    results = [
        _result_dict(payload=b"out-0"),
        None,
        _result_dict(payload=b"out-2"),
    ]
    ad = _stub_dispatcher(results)
    adapter = _adapter(ad)

    out = await adapter.fan_out(_manifest(), shards)

    assert len(out) == 2
    assert out[0].shard_cid == shards[0].cid
    assert out[0].payload == b"out-0"
    assert out[1].shard_cid == shards[2].cid
    assert out[1].payload == b"out-2"


@pytest.mark.asyncio
async def test_empty_shards_returns_empty_dispatcher_untouched():
    """No shards → no dispatcher calls, empty output list."""
    ad = _stub_dispatcher([])
    adapter = _adapter(ad)

    out = await adapter.fan_out(_manifest(), [])

    assert out == []
    ad.dispatch.assert_not_awaited()
    ad.wait_for_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_dp_noise_applied_threading_false_propagates():
    """Adapter does NOT inject DP noise — if the source agent reports
    `dp_noise_applied=False`, the resulting PartialResult must carry
    False forward (the runner's A5 enforcement rejects it downstream;
    the adapter is not the security boundary)."""
    shards = _shards(1)
    ad = _stub_dispatcher([_result_dict(dp_noise=False)])
    adapter = _adapter(ad)

    out = await adapter.fan_out(_manifest(), shards)

    assert len(out) == 1
    assert out[0].dp_noise_applied is False


@pytest.mark.asyncio
async def test_per_shard_budget_propagated_to_mobile_agent():
    """Each MobileAgent built by the adapter carries
    `ftns_budget == per_shard_budget_ftns`."""
    shards = _shards(2)
    ad = _stub_dispatcher([_result_dict(), _result_dict()])
    adapter = _adapter(ad, per_shard_budget_ftns=42)

    await adapter.fan_out(_manifest(), shards)

    assert ad.dispatch.await_count == 2
    for call in ad.dispatch.await_args_list:
        agent = call.args[0] if call.args else call.kwargs["agent"]
        assert agent.ftns_budget == 42


def test_adapter_satisfies_swarm_dispatcher_protocol():
    """Structural check that the adapter is a runtime-checkable
    SwarmDispatcher."""
    ad = _stub_dispatcher([])
    adapter = _adapter(ad)
    assert isinstance(adapter, SwarmDispatcher)


# ──────────────────────────────────────────────────────────────────────
# Bonus — defensive shape checks for the result-dict adapter
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dp_noise_applied_defaults_to_false_when_marker_missing():
    """If the result dict omits `dp_noise_applied`, the PartialResult
    carries False (defensive — the runner will reject, which is the
    correct fail-fast behavior for a malformed source-agent reply)."""
    shards = _shards(1)
    # Missing dp_noise_applied entirely
    bad_result = {"payload": b"out", "agent_signature": b"\x00" * 64}
    ad = _stub_dispatcher([bad_result])
    adapter = _adapter(ad)

    out = await adapter.fan_out(_manifest(), shards)

    assert len(out) == 1
    assert out[0].dp_noise_applied is False
