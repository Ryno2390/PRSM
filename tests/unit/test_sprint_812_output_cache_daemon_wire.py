"""Sprint 812 — daemon-startup output cache wire-up.

Sprint 811 added `_output_cache` field to
ParallaxScheduledExecutor but left it None on construction.
Operators had to manually set the attribute post-construct,
which means production deploys silently never enable cache
even with PRSM_INFERENCE_OUTPUT_CACHE_ENABLED=1.

Sprint 812 closes that: `build_parallax_executor_or_none`
calls `resolve_output_cache_from_env` after construction and
injects the result. Operators opt in via env (sprint 810's
`PRSM_INFERENCE_OUTPUT_CACHE_ENABLED=1`) — no code change
required.

Also surfaces the 3 env vars in sprint-800's
`_PARALLAX_ENV_REGISTRY` so `prsm node init` writes them into
the operator's `.env` template with descriptions.

Pin tests (source-shape on inference_wiring):
- build_parallax_executor_or_none imports
  resolve_output_cache_from_env.
- The function assigns the cache to ._output_cache on the
  constructed executor.
- The assignment is wrapped in try/except (fail-soft like the
  rest of the wiring).

Pin tests (env registry):
- PRSM_INFERENCE_OUTPUT_CACHE_ENABLED appears in registry.
- PRSM_INFERENCE_OUTPUT_CACHE_TTL_S appears.
- PRSM_INFERENCE_OUTPUT_CACHE_MAX_ENTRIES appears.
- Their descriptions surface the privacy invariant.
"""
from __future__ import annotations

import inspect


# ---- Source-shape on build_parallax_executor_or_none ----------


def test_wiring_imports_resolve_output_cache():
    from prsm.node import inference_wiring as _wiring
    src = inspect.getsource(_wiring)
    assert "resolve_output_cache_from_env" in src, (
        "Sprint 812: inference_wiring must call "
        "resolve_output_cache_from_env to enable output cache "
        "via PRSM_INFERENCE_OUTPUT_CACHE_ENABLED env."
    )


def test_wiring_assigns_to_output_cache():
    """The resolved cache is assigned to the executor's
    _output_cache attribute (sprint 811's slot)."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    src = inspect.getsource(build_parallax_executor_or_none)
    assert "_output_cache" in src, (
        "Sprint 812: must set executor._output_cache"
    )


def test_wiring_is_fail_soft():
    """Cache wire-up wrapped in try/except so daemon doesn't
    crash on a cache-init failure (e.g. env value parse error
    that the resolver didn't catch). Same fail-soft pattern as
    the existing executor construction."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    src = inspect.getsource(build_parallax_executor_or_none)
    # Find a try block that wraps the output_cache wiring.
    # Source-shape: "resolve_output_cache_from_env" appears
    # INSIDE a try: block (not at top-level of the function).
    # We grep for "try:" + "resolve_output_cache_from_env" in
    # the same vicinity.
    rc_idx = src.find("resolve_output_cache_from_env")
    assert rc_idx > 0
    # Look backward for the preceding try:
    before = src[:rc_idx]
    last_try = before.rfind("try:")
    assert last_try > 0, (
        "resolve_output_cache_from_env must be inside a try: block"
    )


# ---- Env registry surfaces the cache vars ---------------------


def test_registry_includes_cache_enabled():
    from prsm.cli import _PARALLAX_ENV_REGISTRY
    names = {entry[0] for entry in _PARALLAX_ENV_REGISTRY}
    assert "PRSM_INFERENCE_OUTPUT_CACHE_ENABLED" in names


def test_registry_includes_cache_ttl():
    from prsm.cli import _PARALLAX_ENV_REGISTRY
    names = {entry[0] for entry in _PARALLAX_ENV_REGISTRY}
    assert "PRSM_INFERENCE_OUTPUT_CACHE_TTL_S" in names


def test_registry_includes_cache_max_entries():
    from prsm.cli import _PARALLAX_ENV_REGISTRY
    names = {entry[0] for entry in _PARALLAX_ENV_REGISTRY}
    assert "PRSM_INFERENCE_OUTPUT_CACHE_MAX_ENTRIES" in names


def test_registry_description_mentions_privacy_invariant():
    """Operators reading the template must see WHY only
    `privacy_tier=none` is cacheable. The description for the
    ENABLED var surfaces this so they don't accidentally
    assume it caches everything."""
    from prsm.cli import _PARALLAX_ENV_REGISTRY
    for name, _req, _vals, desc in _PARALLAX_ENV_REGISTRY:
        if name == "PRSM_INFERENCE_OUTPUT_CACHE_ENABLED":
            low = desc.lower()
            assert (
                "privacy_tier" in low
                or "tier=none" in low
                or "privacy" in low
            ), (
                "Sprint 812: ENABLED description must mention "
                "the privacy invariant (only tier=none cached)"
            )
            return
    assert False, "ENABLED entry not found in registry"
