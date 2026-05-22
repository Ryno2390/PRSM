"""Sprint 740 — runbook update for F65-F68 admin-auth arc.

Sprint 732's runbook refresh covered the streaming wire-protocol
hardening (sprints 713-731) but predated the admin-auth arc
(sprints 734-739, F65-F68). An operator upgrading past sprint 734
would see their grafana scraper / remote monitoring 403 without
any guidance in the runbook on:

- WHY their tooling broke (the F65 default-deny behavior change)
- HOW to fix it (3 paths: localhost-only, reverse-proxy + auth,
  or PRSM_ADMIN_REMOTE_ALLOWED opt-out)
- WHAT additional defenses F66/F67/F68 add for proxy patterns

Sprint 740 adds an "Admin endpoint auth (sprints 734-739, F65-
F68)" section to the runbook with all three remediation paths,
the symptoms operators should expect, and the loopback
representations accepted post-739.
"""
from __future__ import annotations

from pathlib import Path


RUNBOOK = (
    Path(__file__).parent.parent.parent
    / "docs" / "operations" / "parallax-inference-deploy.md"
)


def _text() -> str:
    return RUNBOOK.read_text()


def test_runbook_has_admin_endpoint_auth_section():
    """Pin: section title is grep-findable for operators
    troubleshooting after upgrade."""
    text = _text()
    assert "Admin endpoint auth" in text


def test_runbook_warns_about_behavior_change():
    """Pin: explicit BEHAVIOR CHANGE warning so operators reading
    the runbook know to expect 403s on remote tooling."""
    text = _text()
    assert "BEHAVIOR CHANGE" in text


def test_runbook_documents_prsm_admin_remote_allowed():
    """Pin: the env var operators most need is documented inline
    so they can grep for it."""
    text = _text()
    assert "PRSM_ADMIN_REMOTE_ALLOWED" in text


def test_runbook_lists_three_remediation_paths():
    """Pin: localhost-only / behind-reverse-proxy / opt-out env
    — all three remediation paths must be named so operators
    pick the one matching their deployment."""
    text = _text()
    # Localhost path
    assert (
        "same host" in text.lower()
        or "localhost" in text.lower()
    )
    # Reverse-proxy path
    assert "reverse proxy" in text.lower() or "nginx" in text.lower()
    # Opt-out env path
    assert "PRSM_ADMIN_REMOTE_ALLOWED=1" in text


def test_runbook_warns_against_opt_out_without_proxy_auth():
    """Pin: the opt-out path must explicitly warn that
    PRSM_ADMIN_REMOTE_ALLOWED=1 WITHOUT proxy auth re-opens
    the F65 vulnerability. Otherwise an operator just sets the
    env var to get past the 403 and ships an insecure deploy."""
    text = _text()
    # Loose match — any warning that opt-out alone is insecure.
    assert (
        "re-opens the F65" in text
        or "re-opens" in text.lower()
        or "ONLY if you" in text
        or "least secure" in text.lower()
    )


def test_runbook_lists_loopback_representations():
    """Pin: sprint 739 widened loopback acceptance. Operators
    running dual-stack or with 127/8 aliases should see this so
    they don't false-alarm a 403."""
    text = _text()
    assert "::ffff:127.0.0.1" in text or "IPv4-mapped" in text
    assert "127.0.0.0/8" in text or "RFC 1122" in text


def test_runbook_changelog_includes_f65_through_f68():
    """Pin: the sprint changelog table at the bottom of the
    runbook references F65-F68 so an operator scanning the
    sprint history sees the admin-auth arc."""
    text = _text()
    assert "F65" in text and "F66" in text and "F67" in text and "F68" in text
