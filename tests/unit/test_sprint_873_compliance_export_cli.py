"""Sprint 873 — `prsm node compliance-export` CLI pin tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner


_real_Client = httpx.Client


@pytest.fixture(autouse=True)
def _restore_real_httpx(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _real_Client)
    yield


from prsm.cli import node_compliance_export  # noqa: E402


_CSV_HEADER = (
    "entry_id,timestamp_unix,timestamp_iso8601_utc,kind,user_id,"
    "usd_amount,ftns_amount,status,kyc_status,tx_hash,vendor_ref,"
    "address,jurisdiction,metadata_json\n"
)
_CSV_ROW = (
    "e1,1700000000.0,2023-11-14T22:13:20+00:00,onramp_quote,alice,"
    "100.000000,0,OK,VERIFIED,,,,AU,\n"
)


def _mock_csv(body: str, status: int = 200):
    def handler(request):
        return httpx.Response(
            status, content=body,
            headers={"content-type": "text/csv"},
        )
    return handler


@pytest.fixture
def patched_client():
    def _patch(handler):
        return patch(
            "httpx.Client",
            lambda *a, **kw: _real_Client(
                transport=httpx.MockTransport(handler),
            ),
        )
    return _patch


# ── stdout mode ──────────────────────────────────────────────

def test_stdout_emits_raw_csv(patched_client):
    runner = CliRunner()
    with patched_client(_mock_csv(_CSV_HEADER + _CSV_ROW)):
        result = runner.invoke(node_compliance_export, [])
    assert result.exit_code == 0
    # No Rich formatting in stdout mode — pipeable CSV
    assert "entry_id" in result.output
    assert "alice" in result.output


def test_stdout_no_filter_no_filter_line(patched_client):
    runner = CliRunner()
    with patched_client(_mock_csv(_CSV_HEADER)):
        result = runner.invoke(node_compliance_export, [])
    # stdout mode is pure CSV — no "Filters:" line
    assert "Filters:" not in result.output


# ── --output file mode ───────────────────────────────────────

def test_output_writes_file(tmp_path, patched_client):
    out = tmp_path / "exp.csv"
    runner = CliRunner()
    with patched_client(_mock_csv(_CSV_HEADER + _CSV_ROW)):
        result = runner.invoke(
            node_compliance_export,
            ["--output", str(out)],
        )
    assert result.exit_code == 0
    assert out.exists()
    body = out.read_text()
    assert body.startswith("entry_id,")
    assert "alice" in body
    # Operator feedback
    assert "Wrote 1 row" in result.output


def test_output_zero_rows_clean(tmp_path, patched_client):
    out = tmp_path / "exp.csv"
    runner = CliRunner()
    with patched_client(_mock_csv(_CSV_HEADER)):
        result = runner.invoke(
            node_compliance_export,
            ["--output", str(out)],
        )
    assert result.exit_code == 0
    assert "Wrote 0 row" in result.output


def test_output_shows_filter_summary(tmp_path, patched_client):
    """Filter context line gives operators an audit trail of
    what the export covered."""
    out = tmp_path / "exp.csv"
    runner = CliRunner()
    with patched_client(_mock_csv(_CSV_HEADER + _CSV_ROW)):
        result = runner.invoke(
            node_compliance_export,
            [
                "--output", str(out),
                "--since", "1700000000",
                "--user-id", "alice",
                "--min-usd", "10000",
            ],
        )
    assert result.exit_code == 0
    assert "Filters:" in result.output
    assert "since=1700000000" in result.output
    assert "user_id=alice" in result.output
    assert "min_usd=10000" in result.output


# ── Filter query params passed through ───────────────────────

def test_filters_passed_as_query_params(patched_client):
    captured_urls = []

    def handler(request):
        captured_urls.append(str(request.url))
        return httpx.Response(
            200, content=_CSV_HEADER,
            headers={"content-type": "text/csv"},
        )

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(
            node_compliance_export,
            [
                "--since", "1700000000",
                "--until", "1800000000",
                "--user-id", "alice",
                "--kind", "onramp_quote",
                "--min-usd", "10000",
            ],
        )
    assert result.exit_code == 0
    url = captured_urls[0]
    assert "since=1700000000" in url
    assert "until=1800000000" in url
    assert "user_id=alice" in url
    assert "kind=onramp_quote" in url
    assert "min_usd=10000" in url


def test_no_filters_no_query_params(patched_client):
    """When operator passes no filters, the URL must have no
    query string OR an empty one — defends against accidentally
    sending None values that the server might mis-parse."""
    captured_urls = []

    def handler(request):
        captured_urls.append(str(request.url))
        return httpx.Response(
            200, content=_CSV_HEADER,
            headers={"content-type": "text/csv"},
        )

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_compliance_export, [])
    assert result.exit_code == 0
    url = captured_urls[0]
    # No filter params present
    assert "since=" not in url
    assert "user_id=" not in url


# ── Error paths ──────────────────────────────────────────────

def test_exit_2_on_connection_error(patched_client):
    def handler(request):
        raise httpx.ConnectError("refused")

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_compliance_export, [])
    assert result.exit_code == 2
    assert "Cannot reach PRSM node" in result.output


def test_exit_1_on_non_200(patched_client):
    runner = CliRunner()
    with patched_client(_mock_csv("internal", status=500)):
        result = runner.invoke(node_compliance_export, [])
    assert result.exit_code == 1
    assert "500" in result.output


# ── --api-port flag ─────────────────────────────────────────

def test_custom_api_port_honored(patched_client):
    captured_urls = []

    def handler(request):
        captured_urls.append(str(request.url))
        return httpx.Response(
            200, content=_CSV_HEADER,
            headers={"content-type": "text/csv"},
        )

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(
            node_compliance_export, ["--api-port", "9999"],
        )
    assert result.exit_code == 0
    assert any(":9999/" in u for u in captured_urls)
