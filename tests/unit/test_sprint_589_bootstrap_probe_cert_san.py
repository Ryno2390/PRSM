"""Sprint 589 — bootstrap-probe surfaces cert SAN DNS list.

Pre-589 HostProbe captures cert_subject (CN) + cert_issuer but
drops the subjectAltName list. Operators can't see which hostnames
a cert actually covers — exactly the F30 class of bug (sprint 588
fixed bootstrap-us SAN; without SAN visibility, future regressions
are silent).

Sprint 589 adds:
  HostProbe.cert_san_dns: List[str]    # populated from cert["subjectAltName"]
  to_dict() includes it

Test plan:
  1. dataclass field exists + defaults to []
  2. to_dict() round-trip preserves it
  3. The extraction logic handles the (('DNS', 'name'), ...) tuple
     shape Python's ssl module returns
"""
from __future__ import annotations


def test_host_probe_dataclass_has_cert_san_dns_field():
    """HostProbe gains cert_san_dns: List[str]."""
    from prsm.cli_helpers.bootstrap_probe import HostProbe, ProbeStatus
    hp = HostProbe(
        url="wss://x:8765",
        host="x",
        port=8765,
        status=ProbeStatus.OK,
    )
    assert hasattr(hp, "cert_san_dns")
    assert hp.cert_san_dns == []


def test_host_probe_to_dict_includes_cert_san_dns():
    from prsm.cli_helpers.bootstrap_probe import HostProbe, ProbeStatus
    hp = HostProbe(
        url="wss://x:8765",
        host="x",
        port=8765,
        status=ProbeStatus.OK,
        cert_san_dns=["a.example", "b.example"],
    )
    d = hp.to_dict()
    assert d["cert_san_dns"] == ["a.example", "b.example"]


def test_extract_san_dns_from_cert_dict_helper():
    """Sprint 589 introduces _extract_san_dns(cert) helper that
    pulls DNS entries from the ssl.getpeercert() output. The cert
    dict's subjectAltName is a tuple of (type, value) pairs.
    """
    from prsm.cli_helpers.bootstrap_probe import _extract_san_dns
    cert = {
        "subjectAltName": (
            ("DNS", "bootstrap1.prsm-network.com"),
            ("DNS", "bootstrap-us.prsm-network.com"),
            ("IP Address", "1.2.3.4"),  # non-DNS — must be skipped
        ),
    }
    assert _extract_san_dns(cert) == [
        "bootstrap1.prsm-network.com",
        "bootstrap-us.prsm-network.com",
    ]


def test_extract_san_dns_missing_subjectAltName_returns_empty():
    from prsm.cli_helpers.bootstrap_probe import _extract_san_dns
    assert _extract_san_dns({}) == []
    assert _extract_san_dns({"subject": [(("CN", "x"),)]}) == []


def test_extract_san_dns_handles_none_cert():
    """Defensive: getpeercert() may return None on some paths."""
    from prsm.cli_helpers.bootstrap_probe import _extract_san_dns
    assert _extract_san_dns(None) == []
