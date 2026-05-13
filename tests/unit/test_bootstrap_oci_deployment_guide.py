"""Pin the OCI deployment guide for bootstrap servers.

The guide closes the operator-side ops gap that PRSM-CR-
2026-05-13-2 §5 non-scope item 6 flagged as "operator-driven,
not engineering-driven." When the EU + APAC droplets land,
the sprint-375 fallback code path immediately benefits.

This test gates the guide against accidental deletion +
ensures the load-bearing operator-facing claims stay
consistent (cert paths, env-var names, port numbers, service
unit shape).
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE_PATH = (
    REPO_ROOT
    / "docs" / "operations"
    / "bootstrap-server-oci-deployment-guide.md"
)


def _read_guide() -> str:
    return GUIDE_PATH.read_text(encoding="utf-8")


def test_guide_file_exists():
    """Pin against accidental deletion."""
    assert GUIDE_PATH.is_file()


def test_guide_names_canonical_hostnames():
    """The two canonical hostnames the sprint-375 fallback
    list expects MUST appear in the guide. Drift breaks the
    one-to-one mapping operators rely on."""
    text = _read_guide()
    assert "bootstrap-eu.prsm-network.com" in text
    assert "bootstrap-apac.prsm-network.com" in text
    # Reference to the existing US bootstrap for symmetry
    assert "bootstrap1.prsm-network.com" in text


def test_guide_documents_canonical_port():
    """Port 8765 is the canonical bootstrap WSS port; the
    guide must teach operators to open it both in OCI
    security list AND in ufw."""
    text = _read_guide()
    # Port appears in firewall instructions
    assert "8765" in text
    # Both layers documented
    assert "security list" in text.lower()
    assert "ufw" in text.lower()


def test_guide_documents_systemd_unit():
    """The systemd service shape is the operator's only
    real surface for restart/status/logs. Pin the canonical
    path + ExecStart so future revisions don't silently
    rename either."""
    text = _read_guide()
    assert "/etc/systemd/system/prsm-bootstrap.service" in text
    assert "prsm.bootstrap.server" in text
    assert "EnvironmentFile=" in text


def test_guide_documents_env_var_names():
    """The bootstrap server reads these env vars verbatim
    via prsm/bootstrap/config.py — renaming any without
    updating the guide leaves operators with a broken
    deployment. Pin them."""
    text = _read_guide()
    canonical_env_vars = [
        "PRSM_BOOTSTRAP_HOST",
        "PRSM_BOOTSTRAP_PORT",
        "PRSM_SSL_ENABLED",
        "PRSM_SSL_CERT_PATH",
        "PRSM_SSL_KEY_PATH",
        "PRSM_DOMAIN",
    ]
    for var in canonical_env_vars:
        assert var in text, (
            f"OCI guide missing env var {var!r}; without "
            f"it the operator's deployment will use defaults "
            f"that may not match the canonical config."
        )


def test_guide_documents_letsencrypt_workflow():
    """TLS cert issuance is the most-likely-to-fail step;
    operators need certbot + renewal instructions."""
    text = _read_guide()
    assert "certbot" in text.lower()
    assert "letsencrypt" in text.lower()
    assert "renew" in text.lower()


def test_guide_documents_dns_a_record():
    """Without DNS pointing at the new IP, the sprint-375
    fallback list still resolves to nothing. DNS is the
    integration point + must be clearly called out."""
    text = _read_guide()
    assert "A record" in text or "A-record" in text
    # Verification step
    assert "dig" in text.lower()


def test_guide_documents_end_to_end_verification():
    """The verification section is what tells operators
    they're done. Must include TCP, TLS, and WSS checks."""
    text = _read_guide()
    # Section header
    assert "verification" in text.lower()
    # TCP check
    assert (
        "nc -zv" in text
        or "netcat" in text.lower()
        or "Connection succeeded" in text
    )
    # TLS handshake
    assert "openssl s_client" in text
    # CLI verification (prsm node bootstrap from sprint 380)
    assert "prsm node bootstrap" in text


def test_guide_cross_references_load_bearing_docs():
    text = _read_guide()
    expected_refs = [
        "PRSM-CR-2026-05-13-2",
        "fleet-kill-switch-operator-runbook",
        "audit-prep",
        "§7.29",
        "config.py",
        "libp2p_discovery.py",
    ]
    for ref in expected_refs:
        assert ref in text, (
            f"OCI guide missing cross-reference to {ref!r}."
        )


def test_guide_documents_free_tier_constraints():
    """Operators must know the free-tier limits to avoid
    accidentally provisioning paid resources."""
    text = _read_guide()
    assert "Always Free" in text
    assert "4 ARM" in text or "4 OCPU" in text or "Ampere A1" in text
    assert "$0" in text


def test_guide_documents_migration_path():
    """When (not if) usage outgrows free tier, the
    migration path must be obvious so the upgrade isn't
    a surprise."""
    text = _read_guide()
    assert "Migration" in text or "migration" in text
    # The cloud-agnostic claim: change DNS, swap host
    assert (
        "change DNS" in text.lower()
        or "DNS A record" in text
        or "repoint DNS" in text
    )
