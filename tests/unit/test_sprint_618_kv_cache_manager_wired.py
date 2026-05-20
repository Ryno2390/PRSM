"""Sprint 618 (Phase 2F-5i) — KVCacheManager wired into LayerStageServer factory.

Sprint 607's build_layer_stage_server_executor constructed
LayerStageServer with kv_cache_manager=None — INCREMENTAL decode
mode silently degraded. Sprint 618 wires KVCacheManager opt-in via:

  PRSM_PARALLAX_KV_CACHE_ENABLED=1
  PRSM_PARALLAX_KV_CACHE_MAX_REQUESTS=<int>   (default 64)
  PRSM_PARALLAX_KV_CACHE_TTL_SECONDS=<float>  (default 300.0)

Source-grep tests (sys.modules patching of chain_rpc.* triggers a
numpy "cannot load module more than once" — covered the same way
in sprints 607/612). Behavioral check via the real KVCacheManager
constructor uses defaults safely.
"""
from __future__ import annotations

import inspect


def test_factory_reads_kv_cache_enabled_env():
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    assert "PRSM_PARALLAX_KV_CACHE_ENABLED" in src


def test_factory_reads_max_requests_env():
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    assert "PRSM_PARALLAX_KV_CACHE_MAX_REQUESTS" in src


def test_factory_reads_ttl_seconds_env():
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    assert "PRSM_PARALLAX_KV_CACHE_TTL_SECONDS" in src


def test_factory_imports_kv_cache_manager_lazily():
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    assert "from prsm.compute.chain_rpc.kv_cache import KVCacheManager" in src


def test_factory_threads_kv_cache_manager_into_layer_stage_server():
    """The LayerStageServer constructor must receive kv_cache_manager
    kwarg (None when disabled, instance when enabled).
    """
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    assert "kv_cache_manager=kv_cache_manager" in src


def test_real_kv_cache_manager_construction_default_params():
    """Real KVCacheManager constructor accepts the defaults sprint 618
    uses (max_cached_requests=64, ttl_seconds=300.0).
    """
    from prsm.compute.chain_rpc.kv_cache import KVCacheManager
    cm = KVCacheManager(max_cached_requests=64, ttl_seconds=300.0)
    assert cm is not None


def test_implicit_enable_when_any_tuning_var_set():
    """Operator setting PRSM_PARALLAX_KV_CACHE_MAX_REQUESTS without
    ENABLED=1 still gets the cache (less-surprising default).
    """
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    # Source must check for any of the tuning vars as opt-in
    assert (
        "or bool(_kv_max_raw)" in src
        or "or _kv_max_raw" in src.lower().replace("_kv_max_raw", "_kv_max_raw")
    ), "Implicit opt-in via tuning vars missing"
