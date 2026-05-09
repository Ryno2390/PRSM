"""DaemonWatchdog — daemon-crash → webhook event."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.daemon_watchdog import DaemonWatchdog
from prsm.node.webhook_delivery import DeliveryResult


def _node_with_tasks(**task_states):
    """Build a node-like with task attributes set to MagicMocks
    whose .done() returns the given state."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    for attr, alive in task_states.items():
        if alive is None:
            setattr(node, attr, None)
            continue
        task = MagicMock()
        task.done.return_value = not alive
        setattr(node, attr, task)
    return node


def _stub_deliverer(success=True):
    deliverer = MagicMock()
    deliverer.deliver = AsyncMock(
        return_value=DeliveryResult(
            success=success,
            status_code=200 if success else 503,
            attempts=1,
            error=None if success else "down",
        ),
    )
    return deliverer


class TestValidation:
    def test_webhook_url_required(self):
        with pytest.raises(ValueError):
            DaemonWatchdog(
                node=MagicMock(),
                webhook_deliverer=MagicMock(),
                webhook_url="",
            )

    def test_interval_must_be_positive(self):
        with pytest.raises(ValueError):
            DaemonWatchdog(
                node=MagicMock(),
                webhook_deliverer=MagicMock(),
                webhook_url="https://hook.example.com",
                interval_seconds=0,
            )


class TestCheckOnce:
    @pytest.mark.asyncio
    async def test_first_sweep_establishes_baseline_no_emit(self):
        """First sweep can't fire crashes — even if .done()
        is True, could be inherited from before watchdog started."""
        node = _node_with_tasks(
            _heartbeat_scheduler_task=False,  # already done
        )
        deliverer = _stub_deliverer()
        watchdog = DaemonWatchdog(
            node=node,
            webhook_deliverer=deliverer,
            webhook_url="https://hook.example.com",
        )
        emitted = await watchdog.check_once()
        assert emitted == []
        deliverer.deliver.assert_not_called()

    @pytest.mark.asyncio
    async def test_running_to_done_transition_fires(self):
        """The load-bearing case: was running last sweep, now
        crashed → emit webhook."""
        node = _node_with_tasks(
            _heartbeat_scheduler_task=True,  # alive
        )
        deliverer = _stub_deliverer()
        watchdog = DaemonWatchdog(
            node=node,
            webhook_deliverer=deliverer,
            webhook_url="https://hook.example.com",
        )
        # First sweep: establish baseline.
        await watchdog.check_once()
        # Simulate crash.
        node._heartbeat_scheduler_task.done.return_value = True
        # Second sweep: detect transition.
        emitted = await watchdog.check_once()
        assert "heartbeat_scheduler" in emitted
        deliverer.deliver.assert_awaited_once()
        # Verify payload shape.
        call = deliverer.deliver.await_args
        kwargs = call.kwargs
        assert kwargs["event"] == "daemon.crashed"
        assert kwargs["payload"]["daemon"] == "heartbeat_scheduler"
        assert kwargs["payload"]["node_id"] == "test-node"

    @pytest.mark.asyncio
    async def test_still_alive_does_not_fire(self):
        node = _node_with_tasks(
            _heartbeat_scheduler_task=True,
        )
        deliverer = _stub_deliverer()
        watchdog = DaemonWatchdog(
            node=node,
            webhook_deliverer=deliverer,
            webhook_url="https://hook.example.com",
        )
        await watchdog.check_once()
        await watchdog.check_once()
        await watchdog.check_once()
        deliverer.deliver.assert_not_called()

    @pytest.mark.asyncio
    async def test_unwired_task_does_not_fire(self):
        node = _node_with_tasks(
            _heartbeat_scheduler_task=None,  # not wired
        )
        deliverer = _stub_deliverer()
        watchdog = DaemonWatchdog(
            node=node,
            webhook_deliverer=deliverer,
            webhook_url="https://hook.example.com",
        )
        await watchdog.check_once()
        await watchdog.check_once()
        deliverer.deliver.assert_not_called()

    @pytest.mark.asyncio
    async def test_secret_threaded_to_deliverer(self):
        node = _node_with_tasks(_heartbeat_scheduler_task=True)
        deliverer = _stub_deliverer()
        watchdog = DaemonWatchdog(
            node=node,
            webhook_deliverer=deliverer,
            webhook_url="https://hook.example.com",
            webhook_secret="ops-shared-secret",
        )
        await watchdog.check_once()
        node._heartbeat_scheduler_task.done.return_value = True
        await watchdog.check_once()
        # Secret threaded through to deliverer.
        kwargs = deliverer.deliver.await_args.kwargs
        assert kwargs["secret"] == "ops-shared-secret"

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_crashes(self):
        node = _node_with_tasks(
            _heartbeat_scheduler_task=True,
            _job_reaper_task=True,
        )
        deliverer = _stub_deliverer()
        watchdog = DaemonWatchdog(
            node=node,
            webhook_deliverer=deliverer,
            webhook_url="https://hook.example.com",
        )
        await watchdog.check_once()
        # Both crash.
        node._heartbeat_scheduler_task.done.return_value = True
        node._job_reaper_task.done.return_value = True
        emitted = await watchdog.check_once()
        # Both events emitted.
        assert "heartbeat_scheduler" in emitted
        assert "job_reaper" in emitted
        assert deliverer.deliver.await_count == 2


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_watch_can_be_stopped(self):
        node = _node_with_tasks()
        deliverer = _stub_deliverer()
        watchdog = DaemonWatchdog(
            node=node,
            webhook_deliverer=deliverer,
            webhook_url="https://hook.example.com",
            interval_seconds=0.05,
        )
        task = asyncio.create_task(watchdog.watch())
        await asyncio.sleep(0.15)
        await watchdog.stop()
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done()
