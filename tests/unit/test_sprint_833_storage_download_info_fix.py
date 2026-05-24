"""Sprint 833 — F29 fix: `prsm storage download` + `info` hit
working endpoints.

Third cascade fix from sprint 830's audit. Both
`prsm storage download <cid>` and `prsm storage info <cid>`
hit /api/v1/storage/{cid}/... routes (legacy storage_api router,
unmounted on production daemon per sp830) so every operator
got a bare 404.

Sprint 833 switches both to inline /content/retrieve/{cid}
(node/api.py — same surface `prsm content fetch` sprint 805
uses successfully). Response shape:
  {status: "success", data: <base64>, filename, size_bytes,
   content_hash, providers_tried}

For `download`: base64-decode data → write to --output or
stdout (text-decoded if UTF-8, raw bytes otherwise).
For `info`: ignore data payload, render metadata only.

Live-attested 2026-05-24 — full round-trip:
  upload payload.txt (37 bytes)
  → CID: 9946fa40...
  info <CID> → renders {filename, size_bytes=37, content_hash,
               providers_tried=0}
  download <CID> -o got.txt → "37 bytes downloaded"
  diff payload.txt got.txt → BYTES MATCH

Pre-833 both commands 404'd.

Pin tests:
- download URL is /content/retrieve/{cid}
- info URL is /content/retrieve/{cid}
- phantom /api/v1/storage/ never appears
- download base64-decodes correctly
- info renders {filename, size_bytes, content_hash}
- status != success → actionable not_found error
- ConnectError → "node start" hint
"""
from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from click.testing import CliRunner


def _invoke(cmd, args):
    from prsm.cli import storage as _storage_group
    return CliRunner().invoke(_storage_group, [cmd] + list(args))


def _retrieve_ok(text="payload content\n"):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "status": "success",
        "data": base64.b64encode(text.encode("utf-8")).decode(),
        "filename": "test.txt",
        "size_bytes": len(text),
        "content_hash": "deadbeef" * 8,
        "providers_tried": 0,
    }
    return r


def _not_found():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "status": "not_found",
        "error": "CID not in any provider",
    }
    return r


# ---- URL regression guards (F29 lesson) ----------------------


def test_download_hits_inline_content_retrieve():
    with patch("httpx.get", return_value=_retrieve_ok()) as mg:
        result = _invoke("download", [
            "abc123", "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 0, result.output
    call_url = mg.call_args.args[0]
    assert call_url == "http://node:8000/content/retrieve/abc123"


def test_download_never_hits_phantom_storage_url():
    with patch("httpx.get", return_value=_retrieve_ok()) as mg:
        _invoke("download", ["abc", "--api-url", "http://node:8000"])
    call_url = mg.call_args.args[0]
    assert "/api/v1/storage/" not in call_url


def test_info_hits_inline_content_retrieve():
    with patch("httpx.get", return_value=_retrieve_ok()) as mg:
        result = _invoke("info", [
            "abc123", "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 0, result.output
    call_url = mg.call_args.args[0]
    assert call_url == "http://node:8000/content/retrieve/abc123"


def test_info_never_hits_phantom_storage_url():
    with patch("httpx.get", return_value=_retrieve_ok()) as mg:
        _invoke("info", ["abc", "--api-url", "http://node:8000"])
    call_url = mg.call_args.args[0]
    assert "/api/v1/storage/" not in call_url


# ---- Successful render paths ---------------------------------


def test_download_writes_to_output_file(tmp_path: Path):
    out = tmp_path / "got.txt"
    with patch(
        "httpx.get",
        return_value=_retrieve_ok("hello world\n"),
    ):
        result = _invoke("download", [
            "abc", "--output", str(out),
            "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 0
    assert out.read_bytes() == b"hello world\n"


def test_info_renders_metadata():
    with patch("httpx.get", return_value=_retrieve_ok()):
        result = _invoke("info", [
            "abc", "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 0
    assert "test.txt" in result.output
    # Size — strip table formatting
    flat = " ".join(result.output.split())
    assert "16 bytes" in flat or "size" in flat.lower()


# ---- Error paths ---------------------------------------------


def test_download_not_found_status_exits_1():
    with patch("httpx.get", return_value=_not_found()):
        result = _invoke("download", [
            "missingcid", "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "not_found" in flat or "CID not in" in flat


def test_info_not_found_status_exits_1():
    with patch("httpx.get", return_value=_not_found()):
        result = _invoke("info", [
            "missingcid", "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "not_found" in flat or "CID not in" in flat


def test_download_connect_error_points_at_node_start():
    with patch(
        "httpx.get",
        side_effect=httpx.ConnectError("conn refused"),
    ):
        result = _invoke("download", [
            "abc", "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "node start" in flat


def test_info_connect_error_points_at_node_start():
    with patch(
        "httpx.get",
        side_effect=httpx.ConnectError("conn refused"),
    ):
        result = _invoke("info", [
            "abc", "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "node start" in flat
