"""Sprint 384 — pin cloud-init userdata for turnkey bootstrap.

When a cloud instance boots with this userdata, it must
produce a working bootstrap server without operator
SSH-in. This test gates the load-bearing behaviors so a
future edit can't silently break the auto-deploy:

  - Bash strict mode (set -euxo pipefail)
  - apt-lock wait loop (Ubuntu cloud-image first-boot)
  - All 4 ufw ports opened (22, 80, 443, 8765)
  - Sprint-383 PRSM_PEER_DB_PATH baked into env file
  - Sprint-383 ReadWritePaths uses /var/lib/prsm-bootstrap
  - DNS-wait gate before certbot (HTTP-01 requires DNS)
  - Idempotent fallback path when DNS doesn't resolve
  - Render helper produces a paste-ready script
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "scripts" / "bootstrap-server-cloud-init.sh"
RENDER = REPO_ROOT / "scripts" / "render-bootstrap-cloud-init.sh"
RETRY_LOOP = (
    REPO_ROOT / "scripts" / "oci-bootstrap-launch-retry.sh"
)


def _read_template() -> str:
    return TEMPLATE.read_text()


def _render(hostname: str, email: str | None = None) -> str:
    """Test helper. Passes email through only when caller
    sets it; otherwise lets the script's own default fire
    (so we can test the default-email behavior)."""
    args = ["bash", str(RENDER), hostname]
    if email is not None:
        args.append(email)
    result = subprocess.run(
        args,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


# ── Template file presence + perms ───────────────────


def test_template_file_exists():
    assert TEMPLATE.is_file()


def test_template_is_executable():
    mode = TEMPLATE.stat().st_mode
    assert mode & stat.S_IXUSR, "template not executable"


def test_render_helper_exists_and_executable():
    assert RENDER.is_file()
    assert RENDER.stat().st_mode & stat.S_IXUSR


# ── Bash hygiene ─────────────────────────────────────


def test_template_uses_strict_mode():
    """Bash strict mode is the difference between cloud-
    init silently completing on a half-broken instance vs
    failing fast at the broken step. Pin it."""
    text = _read_template()
    assert "set -euxo pipefail" in text


def test_template_waits_for_apt_lock():
    """Ubuntu cloud images run unattended-upgrades on
    first boot; apt-lock contention is the most common
    cloud-init failure on fresh instances. Loop must wait."""
    text = _read_template()
    assert "lock-frontend" in text
    assert "fuser" in text or "lsof" in text


# ── Firewall ─────────────────────────────────────────


def test_template_opens_all_four_canonical_ports():
    """ufw must allow 22 (SSH), 80 (Let's Encrypt HTTP-01),
    443 (TLS), 8765 (PRSM bootstrap WSS). Missing any of
    these = silent failure mode."""
    text = _read_template()
    for port in ["22", "80", "443", "8765"]:
        assert f"ufw allow {port}/tcp" in text, (
            f"cloud-init missing ufw rule for port {port}"
        )


def test_template_enables_ufw_force():
    """ufw enable is interactive by default; --force is
    required for non-interactive cloud-init context."""
    text = _read_template()
    assert "ufw --force enable" in text


# ── Sprint-383 fix carried forward ───────────────────


def test_env_file_includes_prsm_peer_db_path():
    """Sprint 383 added PRSM_PEER_DB_PATH env-var support;
    this cloud-init must use it instead of relying on the
    Docker-conventional /app/data default."""
    text = _read_template()
    assert "PRSM_PEER_DB_PATH=" in text
    # Canonical value points at /var/lib/prsm-bootstrap/
    assert "/var/lib/prsm-bootstrap" in text


def test_template_does_not_use_app_data():
    """Pre-sprint-383 hack created /app/data/ on the host;
    sprint-384 cloud-init must NOT replicate the hack
    (canonical path is /var/lib/prsm-bootstrap)."""
    text = _read_template()
    assert "/app/data" not in text, (
        "cloud-init must use sprint-383 canonical path, "
        "not the pre-fix /app/data workaround"
    )


def test_systemd_unit_readwritepaths_canonical():
    """systemd ProtectSystem=strict + ReadWritePaths is the
    sandboxing model; the peer-DB path must be writable
    so the bootstrap server can persist state."""
    text = _read_template()
    assert "ReadWritePaths=/var/log $PEER_DB_DIR" in text or \
           "ReadWritePaths=/var/log /var/lib/prsm-bootstrap" in text


# ── Lifecycle hardening ──────────────────────────────


def test_template_creates_peer_db_directory():
    """Without explicit mkdir + chown, the bootstrap
    process can't write its peer DB."""
    text = _read_template()
    assert "mkdir -p " in text
    assert "chown ubuntu:ubuntu" in text


def test_template_includes_systemd_unit():
    """systemd unit must be written + enabled."""
    text = _read_template()
    assert "/etc/systemd/system/prsm-bootstrap.service" in text
    assert "systemctl daemon-reload" in text
    assert "systemctl enable" in text


def test_template_includes_protectionsystem_hardening():
    """Sprint-379 + sprint-382 systemd hardening pattern
    (NoNewPrivileges + ProtectSystem + PrivateTmp) must be
    in the unit file."""
    text = _read_template()
    assert "NoNewPrivileges=true" in text
    assert "ProtectSystem=strict" in text
    assert "PrivateTmp=true" in text


# ── DNS gate before certbot ──────────────────────────


def test_template_waits_for_dns_before_certbot():
    """Let's Encrypt HTTP-01 requires the A record to
    already resolve to the instance's public IP. If we
    fire certbot before DNS has propagated, it fails +
    leaves the instance certless. The wait loop is the
    load-bearing fix."""
    text = _read_template()
    assert "dig +short" in text
    # Must wait some number of iterations
    assert "for i in {1..120}" in text or "for i in" in text
    # DNS_RESOLVED flag gates the certbot invocation
    assert "DNS_RESOLVED" in text


def test_template_skips_certbot_gracefully_when_dns_unresolved():
    """If DNS still doesn't resolve after the wait, the
    script must NOT crash — instead it leaves manual-
    completion instructions in the log so the operator
    can SSH in and finish."""
    text = _read_template()
    assert "skipping certbot" in text.lower() or \
           "did not resolve" in text.lower()
    # Manual remediation instruction present
    assert "certbot certonly --standalone" in text


def test_template_uses_curl_metadata_fallback_for_ip():
    """Different clouds expose public IP via different
    metadata services. Falling back through multiple
    public-IP-echo services is the cloud-agnostic move."""
    text = _read_template()
    assert "checkip.amazonaws.com" in text or \
           "ifconfig.me" in text


# ── Render helper ────────────────────────────────────


@pytest.mark.requires_halmos
def test_render_replaces_hostname():
    """Marked requires_halmos to bypass the session-wide
    subprocess mock (sprint 366) — semantically the marker
    means 'this test needs real subprocess.run' which fits
    here even though halmos isn't involved."""
    out = _render("bootstrap-eu.prsm-network.com")
    assert (
        'PRSM_HOSTNAME="bootstrap-eu.prsm-network.com"'
        in out
    )
    # The CHANGEME placeholder is gone
    assert "CHANGEME" not in out


@pytest.mark.requires_halmos
def test_render_replaces_email():
    out = _render(
        "bootstrap-eu.prsm-network.com",
        email="custom@example.com",
    )
    assert 'ADMIN_EMAIL="custom@example.com"' in out


@pytest.mark.requires_halmos
def test_render_default_email_is_foundation_ops():
    out = _render("bootstrap-eu.prsm-network.com")
    assert (
        'ADMIN_EMAIL="foundation-ops@prsm-network.com"'
        in out
    )


@pytest.mark.requires_halmos
def test_render_fails_without_hostname():
    result = subprocess.run(
        ["bash", str(RENDER)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "usage" in result.stderr.lower()


# ── Retry loop integration ───────────────────────────


def test_retry_loop_includes_user_data_flag():
    """Sprint 384 — the OCI retry loop must pass
    --user-data-file to oci compute instance launch so
    when Frankfurt eventually frees capacity, the new
    instance auto-deploys via cloud-init."""
    text = RETRY_LOOP.read_text()
    assert "--user-data-file" in text


def test_retry_loop_renders_userdata_before_loop():
    """Render must happen ONCE at script start, not on
    every retry iteration (would be wasteful + slow)."""
    text = RETRY_LOOP.read_text()
    assert "render-bootstrap-cloud-init.sh" in text
    # Look for invocation outside the while loop — should
    # be in the setup section.
    setup_section = text.split("ATTEMPT=0")[0]
    assert "render-bootstrap-cloud-init.sh" in setup_section
