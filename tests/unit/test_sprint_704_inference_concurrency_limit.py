"""Sprint 704 — pin tests for PRSM_INFERENCE_CONCURRENCY_LIMIT.

Audit-doc §7.4 documented NYC's 2GB droplet OOM-cycling under sprint
698's cross-host coordination cold-load. Root cause: concurrent
inference requests each load gpt2 + allocate activations; peak
memory exceeds the 2GB budget.

Sprint 704 ships a global asyncio.Semaphore gate on both
/compute/inference and /compute/inference/stream paths. Operators
on tight-RAM nodes set PRSM_INFERENCE_CONCURRENCY_LIMIT=1 to
serialize inference; nodes with plenty of RAM leave it unset
(default = no cap, preserves pre-704 behavior).
"""
from __future__ import annotations

import asyncio

import pytest


def test_semaphore_disabled_when_env_unset(monkeypatch):
    """No env → returns None → caller code falls through to
    unbounded execution. Preserves pre-704 default behavior."""
    monkeypatch.delenv("PRSM_INFERENCE_CONCURRENCY_LIMIT", raising=False)
    # Reset module-level singleton
    from prsm.node import api as _api
    _api._inference_semaphore = None
    _api._inference_semaphore_limit = None
    assert _api._get_inference_semaphore() is None


def test_semaphore_constructed_when_env_set(monkeypatch):
    """PRSM_INFERENCE_CONCURRENCY_LIMIT=1 → returns
    asyncio.Semaphore(1) on first call + reuses on subsequent."""
    monkeypatch.setenv("PRSM_INFERENCE_CONCURRENCY_LIMIT", "1")
    from prsm.node import api as _api
    _api._inference_semaphore = None
    _api._inference_semaphore_limit = None
    sem = _api._get_inference_semaphore()
    assert isinstance(sem, asyncio.Semaphore)
    # Singleton reuse
    sem2 = _api._get_inference_semaphore()
    assert sem is sem2


def test_semaphore_invalid_value_falls_through(monkeypatch):
    """Non-integer env → returns None (no crash, no enforcement).
    Operator gets the existing pre-704 behavior + their typo'd env
    var is reported by the sprint-696 parallax-readiness CLI."""
    monkeypatch.setenv("PRSM_INFERENCE_CONCURRENCY_LIMIT", "garbage")
    from prsm.node import api as _api
    _api._inference_semaphore = None
    assert _api._get_inference_semaphore() is None


def test_semaphore_non_positive_falls_through(monkeypatch):
    """Zero or negative → returns None (no enforcement). Negative
    Semaphore would raise; zero would block ALL inference forever."""
    monkeypatch.setenv("PRSM_INFERENCE_CONCURRENCY_LIMIT", "0")
    from prsm.node import api as _api
    _api._inference_semaphore = None
    assert _api._get_inference_semaphore() is None
    monkeypatch.setenv("PRSM_INFERENCE_CONCURRENCY_LIMIT", "-3")
    _api._inference_semaphore = None
    assert _api._get_inference_semaphore() is None


def test_semaphore_rebuilds_when_limit_changes(monkeypatch):
    """Operator can flip the limit at runtime via systemctl edit
    + reload; the next call sees the new value + builds a fresh
    semaphore with the updated limit."""
    monkeypatch.setenv("PRSM_INFERENCE_CONCURRENCY_LIMIT", "1")
    from prsm.node import api as _api
    _api._inference_semaphore = None
    _api._inference_semaphore_limit = None
    sem1 = _api._get_inference_semaphore()
    assert sem1._value == 1
    monkeypatch.setenv("PRSM_INFERENCE_CONCURRENCY_LIMIT", "4")
    sem2 = _api._get_inference_semaphore()
    assert sem2 is not sem1
    assert sem2._value == 4


@pytest.mark.asyncio
async def test_semaphore_actually_serializes(monkeypatch):
    """End-to-end behavior: when 2 coroutines try to acquire a
    limit=1 semaphore concurrently, the second waits for the
    first."""
    monkeypatch.setenv("PRSM_INFERENCE_CONCURRENCY_LIMIT", "1")
    from prsm.node import api as _api
    _api._inference_semaphore = None
    _api._inference_semaphore_limit = None
    sem = _api._get_inference_semaphore()
    order = []

    async def task(name: str, delay: float):
        async with sem:
            order.append(f"{name}:enter")
            await asyncio.sleep(delay)
            order.append(f"{name}:exit")

    # Start both — second should not enter until first exits.
    await asyncio.gather(task("A", 0.05), task("B", 0.05))
    # Strict ordering: A:enter → A:exit → B:enter → B:exit
    assert order == ["A:enter", "A:exit", "B:enter", "B:exit"]


def test_env_listed_in_parallax_readiness_registry():
    """Sprint 696's CLI must list the new env var so operators
    discover it via `prsm node parallax-readiness`."""
    from prsm.cli import _PARALLAX_ENV_REGISTRY
    names = {row[0] for row in _PARALLAX_ENV_REGISTRY}
    assert "PRSM_INFERENCE_CONCURRENCY_LIMIT" in names
