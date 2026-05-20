"""Sprint 646 (F25) — `_node_admin_list_details` must surface the
HTTPException `detail` field from the server.

Pre-fix `prsm node pipeline list` against a node without
`PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY` set printed:

    Failed to fetch pipeline jobs: HTTP Error 503: Service Unavailable

…silently swallowing the server's actionable detail (which named
the missing env var). Post-fix:

    Failed to fetch pipeline jobs: HTTP 503 — Pipeline inference
    orchestrator not initialized (set PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY
    env)

Same path serves `prsm node federated list`, so both endpoints
benefit.
"""
from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prsm.cli import node


@pytest.fixture
def runner():
    return CliRunner()


def _http_error(code: int, detail: str) -> urllib.error.HTTPError:
    """Build an HTTPError whose .read() returns a JSON
    {"detail": "..."} body — matches the FastAPI HTTPException
    shape the daemon emits.
    """
    body = json.dumps({"detail": detail}).encode("utf-8")
    return urllib.error.HTTPError(
        url="http://127.0.0.1:8000/admin/inference/pipeline/job",
        code=code,
        msg="Service Unavailable",
        hdrs={},
        fp=io.BytesIO(body),
    )


def test_pipeline_list_surfaces_detail_on_503(runner):
    """The sprint-435+F25 path: pipeline orchestrator unwired → 503
    with breadcrumb. CLI must print the breadcrumb, not just
    "Service Unavailable".
    """
    err = _http_error(
        503,
        "Pipeline inference orchestrator not initialized "
        "(set PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY env)",
    )
    with patch("urllib.request.urlopen", side_effect=err):
        result = runner.invoke(node, ["pipeline", "list"])
    assert result.exit_code == 2
    output = result.output + (result.stderr_bytes or b"").decode()
    # Either the detail breadcrumb OR (less ideal) the env-var
    # name itself must be visible to operators
    assert (
        "PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY" in output
        or "not initialized" in output
    ), f"breadcrumb missing from output: {output!r}"
    # Old behavior surfaced "Service Unavailable" with no detail
    # — make sure we don't regress to that
    assert "HTTP 503" in output


def test_federated_list_surfaces_detail_on_500(runner):
    """Same path serves federated. A hypothetical 500 with detail
    must reach the operator too.
    """
    err = _http_error(500, "ledger backend down — retry later")
    with patch("urllib.request.urlopen", side_effect=err):
        result = runner.invoke(node, ["federated", "list"])
    assert result.exit_code == 2
    output = result.output + (result.stderr_bytes or b"").decode()
    assert "ledger backend down" in output
    assert "HTTP 500" in output


def test_falls_back_when_body_not_json(runner):
    """Server emits non-JSON body — CLI must not crash; falls back
    to default error rendering.
    """
    body = b"<html>nginx 503</html>"
    err = urllib.error.HTTPError(
        url="http://127.0.0.1:8000/admin/inference/pipeline/job",
        code=503, msg="Service Unavailable", hdrs={},
        fp=io.BytesIO(body),
    )
    with patch("urllib.request.urlopen", side_effect=err):
        result = runner.invoke(node, ["pipeline", "list"])
    assert result.exit_code == 2
    # Doesn't crash; either renders the html as the message or
    # falls back to the default urllib repr
    output = result.output + (result.stderr_bytes or b"").decode()
    assert "Failed to fetch pipeline jobs" in output


def test_happy_path_renders_jobs(runner):
    """Server returns 200 with one job → CLI renders it."""
    mock_resp = MagicMock()
    mock_resp.read = MagicMock(return_value=json.dumps({
        "jobs": [{"job_id": "j-001", "status": "active"}],
    }).encode())
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = runner.invoke(node, ["pipeline", "list"])
    assert result.exit_code == 0, result.output
    assert "j-001" in result.output
    assert "active" in result.output


# --------------------------------------------------------------------------
# Sprint 647 — F25 sweep
# --------------------------------------------------------------------------


def test_render_http_error_surfaces_detail():
    """Direct unit test for the shared helper."""
    from prsm.cli_modules.http_errors import render_http_error
    err = _http_error(503, "missing PRSM_FOO env")
    msg = render_http_error(err, "things")
    assert "HTTP 503" in msg
    assert "missing PRSM_FOO env" in msg


def test_render_http_error_falls_back_to_str_on_non_json():
    from prsm.cli_modules.http_errors import render_http_error
    err = urllib.error.HTTPError(
        url="http://x", code=500, msg="Server Error", hdrs={},
        fp=io.BytesIO(b"<html>nginx</html>"),
    )
    msg = render_http_error(err, "things")
    # No detail extracted; falls back to default urllib repr
    assert "Failed to fetch things" in msg


def test_render_http_error_handles_non_http_exceptions():
    from prsm.cli_modules.http_errors import render_http_error
    exc = ConnectionError("connection refused")
    msg = render_http_error(exc, "things")
    assert "connection refused" in msg
    # Should NOT mention HTTP code (no HTTPError)
    assert "HTTP " not in msg


def test_incident_list_uses_shared_helper(runner):
    """Sprint 647 fix: incidents list site previously had
    `except Exception` swallowing detail. Now uses shared helper.
    """
    err = _http_error(503, "incident store unavailable")
    with patch("urllib.request.urlopen", side_effect=err):
        result = runner.invoke(node, ["incident", "list"])
    assert result.exit_code == 2
    out = result.output + (result.stderr_bytes or b"").decode()
    assert "incident store unavailable" in out


def test_tee_status_uses_shared_helper(runner):
    """Sprint 647 fix: TEE status site previously had
    `except Exception` swallowing detail.
    """
    err = _http_error(503, "TEE policy not configured")
    with patch("urllib.request.urlopen", side_effect=err):
        result = runner.invoke(node, ["tee", "status"])
    assert result.exit_code == 2
    out = result.output + (result.stderr_bytes or b"").decode()
    assert "TEE policy not configured" in out
