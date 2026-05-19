"""Sprint 590 — bootstrap-probe SAN-mismatch warning.

Sprint 589 captured the cert SAN DNS list. Sprint 590 USES it:
after a successful TLS handshake, check whether the probe URL's
hostname appears in cert_san_dns. If not, set a warning flag.

This catches F30-class bugs at probe time: TLS handshake can
succeed when default verification is permissive (older OpenSSL,
distrusted intermediates accepted, etc.) but downstream clients
that DO strict-verify will fail. Operators see "✓ ok (SAN
mismatch warning)" in probe output.

Test plan:
  1. HostProbe.san_mismatch field exists; defaults to False.
  2. to_dict() round-trips it.
  3. Helper _check_san_match(hostname, san_dns) returns True
     when hostname is in SAN, False otherwise.
  4. Wildcard SAN (e.g., '*.example.com') matches single-label
     subdomains.
"""
from __future__ import annotations


def test_host_probe_has_san_mismatch_field():
    from prsm.cli_helpers.bootstrap_probe import HostProbe, ProbeStatus
    hp = HostProbe(
        url="wss://x:8765",
        host="x",
        port=8765,
        status=ProbeStatus.OK,
    )
    assert hasattr(hp, "san_mismatch")
    assert hp.san_mismatch is False  # default safe


def test_host_probe_to_dict_includes_san_mismatch():
    from prsm.cli_helpers.bootstrap_probe import HostProbe, ProbeStatus
    hp = HostProbe(
        url="wss://x:8765",
        host="x",
        port=8765,
        status=ProbeStatus.OK,
        san_mismatch=True,
    )
    d = hp.to_dict()
    assert d["san_mismatch"] is True


def test_check_san_match_exact_match():
    from prsm.cli_helpers.bootstrap_probe import _check_san_match
    assert _check_san_match(
        "bootstrap-us.prsm-network.com",
        ["bootstrap-us.prsm-network.com", "bootstrap1.prsm-network.com"],
    ) is True


def test_check_san_match_not_in_list():
    from prsm.cli_helpers.bootstrap_probe import _check_san_match
    assert _check_san_match(
        "bootstrap-us.prsm-network.com",
        ["bootstrap1.prsm-network.com"],  # F30 pre-fix shape
    ) is False


def test_check_san_match_empty_san_list_returns_false():
    """No SAN data → can't claim a match. Conservative."""
    from prsm.cli_helpers.bootstrap_probe import _check_san_match
    assert _check_san_match("x.example.com", []) is False


def test_check_san_match_wildcard():
    """A wildcard SAN like *.example.com matches single-label subs."""
    from prsm.cli_helpers.bootstrap_probe import _check_san_match
    assert _check_san_match("a.example.com", ["*.example.com"]) is True
    # Wildcard does NOT match 2-level subdomain
    assert _check_san_match("a.b.example.com", ["*.example.com"]) is False
    # Wildcard does NOT match the bare apex
    assert _check_san_match("example.com", ["*.example.com"]) is False


def test_check_san_match_case_insensitive():
    """DNS is case-insensitive."""
    from prsm.cli_helpers.bootstrap_probe import _check_san_match
    assert _check_san_match(
        "Bootstrap-US.PRSM-Network.com",
        ["bootstrap-us.prsm-network.com"],
    ) is True
