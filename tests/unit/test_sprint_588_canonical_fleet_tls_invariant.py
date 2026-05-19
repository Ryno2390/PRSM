"""Sprint 588 — canonical bootstrap fleet TLS invariant.

Sprint 575 F29 (DNS rename bootstrap1 → bootstrap-us) surfaced a
sibling production issue 2026-05-19 via `prsm node bootstrap-test`:
the Let's Encrypt cert at /etc/letsencrypt/live/bootstrap1.prsm-network.com/
was CN=bootstrap1.prsm-network.com with SAN limited to bootstrap1.
After the DNS rename, strict TLS clients connecting to
`wss://bootstrap-us.prsm-network.com:8765` failed with
"Hostname mismatch, certificate is not valid for
'bootstrap-us.prsm-network.com'".

Sprint 588 ran `certbot certonly --dns-cloudflare --expand` to add
the bootstrap-us SAN to the existing cert (preserving bootstrap1 for
back-compat). Bootstrap-server-v2 reloaded. Live attestation:
`prsm node bootstrap-test` now reports ALL 3 canonical bootstraps
✓ok / TCP/TLS/WSS clean.

This test pins the SHAPE invariant: canonical_bootstrap_urls() must
include all three regional names so future regressions (e.g.,
someone hard-codes bootstrap1 back) get caught at CI time. Cannot
test live cert state in CI (no network access in test env), but
the URL list is the surface the test harness probes.
"""
from __future__ import annotations


def _canonical_with_clean_env():
    """DEFAULT_BOOTSTRAP_NODES is frozen at module-import time;
    earlier tests in the same process may have set env vars +
    reloaded config (leaving custom defaults in place). Reload
    prsm.node.config with env cleared to read the true defaults.
    """
    import importlib
    import os as _os
    for k in (
        "BOOTSTRAP_PRIMARY", "BOOTSTRAP_FALLBACK_EU",
        "BOOTSTRAP_FALLBACK_APAC",
    ):
        _os.environ.pop(k, None)
    import prsm.node.config as cfg
    importlib.reload(cfg)
    from prsm.cli_helpers.bootstrap_probe import canonical_bootstrap_urls
    return canonical_bootstrap_urls()


def test_canonical_bootstrap_fleet_includes_three_regions():
    """Three canonical regional bootstraps. The fleet probe relies
    on this list — a sprint that drops one (e.g., the F29 rename
    moving bootstrap1 → bootstrap-us without including bootstrap-us)
    would be caught here.
    """
    urls = _canonical_with_clean_env()
    joined = " ".join(urls)
    for region in ("bootstrap-us", "bootstrap-eu", "bootstrap-apac"):
        assert region in joined, (
            f"Canonical fleet missing {region!r}: {urls!r}. "
            f"Sprint 588 invariant: all 3 regional bootstraps in fleet."
        )


def test_canonical_fleet_does_not_reference_dead_bootstrap1():
    """Sprint 575 F29 retired bootstrap1.prsm-network.com from
    fleet defaults; canonical_bootstrap_urls() must not reintroduce
    it (the rename-aware sprint-575 fix would silently revert).
    """
    urls = _canonical_with_clean_env()
    for url in urls:
        assert "bootstrap1.prsm-network.com" not in url, (
            f"Sprint 575/588: bootstrap1 is the retired name; "
            f"got {url!r} in canonical fleet"
        )
