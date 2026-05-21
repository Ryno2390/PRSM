"""Sprint 681 — node.py loads/computes the local hardware_profile
at daemon start, threads it into PeerDiscovery via sprint 680's
local_hardware_profile kwarg.

Sprint 680 opened the schema pass-through path; this sprint makes
peers actually advertise hardware. Loading strategy:

  1. PRSM_HARDWARE_PROFILE_FILE env → explicit JSON path (operator
     can pin a profile, e.g. for CI / synthetic benchmarks).
  2. ~/.prsm/hardware_profile.json — cache populated by prior runs.
  3. Otherwise compute fresh via HardwareProfiler.detect() + write
     the cache for next start.
  4. Any error (psutil unavailable, profiler exception, write
     failure) → return None gracefully. The peer simply doesn't
     advertise hardware, and the DHT pool excludes it.

The helper is pure: takes an optional path override + a profiler
factory, returns Dict or None.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_load_uses_env_var_when_set(tmp_path, monkeypatch):
    """PRSM_HARDWARE_PROFILE_FILE points to a JSON file → load it
    directly. Operator-pin path; never runs the profiler."""
    from prsm.node.hardware_profile_loader import (
        load_local_hardware_profile,
    )
    profile_file = tmp_path / "pinned.json"
    pinned = {"tflops_fp16": 9.9, "memory_gb": 24.0, "gpu_name": "Pinned"}
    profile_file.write_text(json.dumps(pinned))
    monkeypatch.setenv("PRSM_HARDWARE_PROFILE_FILE", str(profile_file))
    result = load_local_hardware_profile(cache_dir=tmp_path)
    assert result == pinned


def test_load_uses_cache_when_present(tmp_path, monkeypatch):
    """No env override → read ~/.prsm/hardware_profile.json (or
    the cache_dir override for tests). Avoids re-running the
    profiler on every daemon restart."""
    from prsm.node.hardware_profile_loader import (
        load_local_hardware_profile,
    )
    monkeypatch.delenv("PRSM_HARDWARE_PROFILE_FILE", raising=False)
    cached = {"tflops_fp16": 4.6, "memory_gb": 16.0, "gpu_name": "Apple M4"}
    cache_file = tmp_path / "hardware_profile.json"
    cache_file.write_text(json.dumps(cached))
    result = load_local_hardware_profile(cache_dir=tmp_path)
    assert result == cached


def test_load_computes_and_caches_when_no_cache(tmp_path, monkeypatch):
    """No env, no cache → compute via the supplied profiler
    factory, write the cache, return the dict."""
    from prsm.node.hardware_profile_loader import (
        load_local_hardware_profile,
    )
    monkeypatch.delenv("PRSM_HARDWARE_PROFILE_FILE", raising=False)
    fresh = {"tflops_fp16": 7.5, "memory_gb": 32.0, "gpu_name": "Fresh"}

    class _FakeProfile:
        def to_dict(self):
            return fresh

    class _FakeProfiler:
        def detect(self):
            return _FakeProfile()

    result = load_local_hardware_profile(
        cache_dir=tmp_path, profiler_factory=_FakeProfiler,
    )
    assert result == fresh
    cache_file = tmp_path / "hardware_profile.json"
    assert cache_file.exists()
    assert json.loads(cache_file.read_text()) == fresh


def test_load_returns_none_on_profiler_exception(tmp_path, monkeypatch):
    """Profiler blew up (psutil missing, GPU detection crashed,
    etc.) → return None. Peer simply doesn't advertise hardware,
    DHT pool excludes it — graceful degradation, NEVER a daemon-
    start failure."""
    from prsm.node.hardware_profile_loader import (
        load_local_hardware_profile,
    )
    monkeypatch.delenv("PRSM_HARDWARE_PROFILE_FILE", raising=False)

    class _BrokenProfiler:
        def detect(self):
            raise RuntimeError("psutil unavailable")

    result = load_local_hardware_profile(
        cache_dir=tmp_path, profiler_factory=_BrokenProfiler,
    )
    assert result is None


def test_load_returns_none_when_env_file_missing(tmp_path, monkeypatch):
    """Env points at a nonexistent file → None (operator typo'd
    the path); never silently fall through to the cache or fresh
    compute since the operator was being explicit."""
    from prsm.node.hardware_profile_loader import (
        load_local_hardware_profile,
    )
    monkeypatch.setenv(
        "PRSM_HARDWARE_PROFILE_FILE", str(tmp_path / "missing.json"),
    )
    result = load_local_hardware_profile(cache_dir=tmp_path)
    assert result is None


def test_load_returns_none_when_env_file_invalid_json(tmp_path, monkeypatch):
    """Env points at a file that exists but doesn't parse → None.
    Same operator-explicit-mistake rationale."""
    from prsm.node.hardware_profile_loader import (
        load_local_hardware_profile,
    )
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{")
    monkeypatch.setenv("PRSM_HARDWARE_PROFILE_FILE", str(bad))
    result = load_local_hardware_profile(cache_dir=tmp_path)
    assert result is None


def test_load_skips_cache_when_invalid_falls_through_to_compute(
    tmp_path, monkeypatch,
):
    """Cache file exists but is corrupt → ignore + recompute via
    the profiler. Defends against partially-written cache files
    from killed daemons. Then re-writes a valid cache."""
    from prsm.node.hardware_profile_loader import (
        load_local_hardware_profile,
    )
    monkeypatch.delenv("PRSM_HARDWARE_PROFILE_FILE", raising=False)
    cache_file = tmp_path / "hardware_profile.json"
    cache_file.write_text("not json")
    fresh = {"tflops_fp16": 2.0, "memory_gb": 8.0, "gpu_name": "Recovered"}

    class _FakeProfile:
        def to_dict(self):
            return fresh

    class _FakeProfiler:
        def detect(self):
            return _FakeProfile()

    result = load_local_hardware_profile(
        cache_dir=tmp_path, profiler_factory=_FakeProfiler,
    )
    assert result == fresh
    assert json.loads(cache_file.read_text()) == fresh


def test_load_tolerates_unwritable_cache_dir(tmp_path, monkeypatch):
    """Cache dir can't be written to → still return the freshly-
    computed profile (we just don't cache for next time). Defends
    against read-only filesystems / permission-denied home
    directories."""
    from prsm.node.hardware_profile_loader import (
        load_local_hardware_profile,
    )
    monkeypatch.delenv("PRSM_HARDWARE_PROFILE_FILE", raising=False)
    fresh = {"tflops_fp16": 1.0, "memory_gb": 4.0}

    class _FakeProfile:
        def to_dict(self):
            return fresh

    class _FakeProfiler:
        def detect(self):
            return _FakeProfile()

    nonexistent = tmp_path / "nope" / "nested"
    # Force the dir creation to fail by making a file at the
    # intermediate path
    (tmp_path / "nope").write_text("blocker")

    result = load_local_hardware_profile(
        cache_dir=nonexistent, profiler_factory=_FakeProfiler,
    )
    # Compute path still succeeded even though cache write failed
    assert result == fresh
