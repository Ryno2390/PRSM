"""Sprint 591 — surface san_mismatch in bootstrap-test text output.

Sprint 590 set HostProbe.san_mismatch when TLS handshake succeeded
but cert SAN doesn't cover the probe hostname. JSON output already
surfaced it. Sprint 591 adds the visible warning in text output so
operators eyeballing the CLI see the regression.

Tests assert the text output renders "SAN mismatch" when a probe
result has san_mismatch=True.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner


def _fake_fleet(san_mismatch: bool):
    from prsm.cli_helpers.bootstrap_probe import (
        FleetProbe, HostProbe, ProbeStatus,
    )
    h = HostProbe(
        url="wss://example.com:8765",
        host="example.com",
        port=8765,
        status=ProbeStatus.OK,
        tcp_ok=True, tls_ok=True, wss_ok=True,
        latency_ms=100.0,
        cert_subject="bootstrap1.prsm-network.com",
        cert_issuer="Let's Encrypt",
        cert_san_dns=["bootstrap1.prsm-network.com"],
        san_mismatch=san_mismatch,
    )
    return FleetProbe(hosts=[h])


def test_text_output_shows_san_mismatch_warning():
    """When san_mismatch=True, output includes the warning."""
    from prsm.cli import node

    fake = _fake_fleet(san_mismatch=True)
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=fake),
    ):
        result = CliRunner().invoke(
            node, ["bootstrap-test", "--url", "wss://example.com:8765"],
        )
    # exit code may be nonzero due to single-host degraded path; that's OK
    assert "SAN mismatch" in result.output, (
        f"Sprint 591: text output must render SAN-mismatch warning; "
        f"got {result.output!r}"
    )


def test_text_output_silent_when_no_mismatch():
    """When san_mismatch=False, no SAN warning text — keeps the
    happy path clean.
    """
    from prsm.cli import node

    fake = _fake_fleet(san_mismatch=False)
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=fake),
    ):
        result = CliRunner().invoke(
            node, ["bootstrap-test", "--url", "wss://example.com:8765"],
        )
    assert "SAN mismatch" not in result.output, (
        "Sprint 591: clean path must NOT show SAN warning"
    )
