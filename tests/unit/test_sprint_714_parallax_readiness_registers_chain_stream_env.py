"""Sprint 714 — `prsm node parallax-readiness` must enumerate
sprint-713's `PRSM_CHAIN_STREAM_QUEUE_MAXSIZE` so operators see it
in the preflight check alongside the other 22 parallax env vars.

Without this, operators deploying sprint-713's streaming
back-pressure gate have no preflight visibility into whether the
env is set — they'd have to read the audit doc or grep the source.
This is the same gap sprint 696 closed for the original 22 vars
and sprint 705 closed for `PRSM_INFERENCE_CONCURRENCY_LIMIT`.
"""
from __future__ import annotations


def test_parallax_env_registry_includes_chain_stream_queue_maxsize():
    """Pin: the sprint-713 env var is registered in the CLI's
    `_PARALLAX_ENV_REGISTRY` so it appears in `parallax-readiness`
    output."""
    from prsm.cli import _PARALLAX_ENV_REGISTRY
    names = [row[0] for row in _PARALLAX_ENV_REGISTRY]
    assert "PRSM_CHAIN_STREAM_QUEUE_MAXSIZE" in names, (
        "sprint 713's bounded-queue env var must appear in the "
        "readiness preflight — otherwise operators have no visibility"
    )


def test_parallax_env_registry_chain_stream_is_optional():
    """The env is optional (default 64 ships safely without operator
    action). Required-True would make the CLI exit nonzero for any
    operator who didn't deliberately set the env."""
    from prsm.cli import _PARALLAX_ENV_REGISTRY
    row = next(
        r for r in _PARALLAX_ENV_REGISTRY
        if r[0] == "PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"
    )
    name, required, valid_values, desc = row
    assert required is False, (
        "sprint 713 ships a safe default (64) so the env is optional;"
        " required=True would break operators who haven't set it"
    )
    # Description must reference the sprint-713 semantics so operators
    # understand what 0/negative means.
    assert "back-pressure" in desc.lower() or "sprint 713" in desc.lower()


def test_parallax_env_registry_total_count_post_sprint_714():
    """Pin: sprint 714 added `PRSM_CHAIN_STREAM_QUEUE_MAXSIZE` —
    registry must now contain exactly 26 entries. If this count
    changes, update the audit doc + parallax-readiness docstring
    so operators know which env vars exist."""
    from prsm.cli import _PARALLAX_ENV_REGISTRY
    # Sprint 734 added PRSM_ADMIN_REMOTE_ALLOWED → 33.
    assert len(_PARALLAX_ENV_REGISTRY) == 33, (
        f"expected 33 vars post-sprint-734, got "
        f"{len(_PARALLAX_ENV_REGISTRY)}"
    )
