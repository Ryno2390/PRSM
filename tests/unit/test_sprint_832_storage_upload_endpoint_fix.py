"""Sprint 832 — F29 fix: `prsm storage upload` hits working endpoint.

Sprint 830's audit + sprint 831 closed the first cascade fix
(prsm ftns balance). Sprint 832 closes the second:
`prsm storage upload <file>` POSTed multipart to
/api/v1/content/upload (legacy content_api router, allow-listed
inert per sp830) so every operator got a 404.

Sprint 832 switches to the inline /content/upload endpoint at
node/api.py:7654, which is the same JSON-ContentUploadRequest
surface that `prsm content publish` (sprint 806) and `prsm node
share` (sprint 574) use successfully.

Body translation:
- Pre-832: multipart files={"file": ...} + data={description,
  royalty_rate, parent_cids, replicas}
- Sprint 832: JSON {text, filename, royalty_rate, replicas,
  parent_cids:[...]}

Binary files: inline endpoint takes UTF-8 text only. If the
file isn't UTF-8-decodable, sprint 832 redirects operators at
sprint-817's `prsm content publish-shard` (base64 binary path).

Auth gate removed: legacy /api/v1/content/upload required login
(Depends(get_current_user)); inline /content/upload doesn't.
Pre-832 the command hard-failed on missing PRIVATE_KEY even
though the server would have accepted the upload.

Pin tests:
- POST URL is /content/upload, NOT /api/v1/content/upload
- Body uses JSON shape (text/filename/royalty_rate/replicas)
- Successful response renders CID
- Binary file → actionable "use publish-shard" error
- 503 surfaces "Content uploader not initialized" hint
- ConnectError points at `prsm node start`
- No-auth (no PRIVATE_KEY) does NOT block command

Live-attested 2026-05-24 against mock-executor daemon:
  $ prsm storage upload payload.txt --description "..."
  → CID: 7fe7aaccebf762d8699c14f5269fad225b7919c9
  → exit 0
Pre-832 same command 404'd or "Not logged in"'d.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from click.testing import CliRunner


def _invoke(args=None):
    from prsm.cli import storage as _storage_group
    return CliRunner().invoke(
        _storage_group, ["upload"] + (args or []),
    )


def _upload_ok():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "cid": "test-cid-abc123",
        "filename": "test.txt",
        "size_bytes": 27,
        "royalty_rate": 0.01,
    }
    return r


# ---- URL regression guard (F29 lesson) -----------------------


def test_upload_hits_inline_content_upload_endpoint(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("sprint 832 dogfood payload\n")
    with patch("httpx.post", return_value=_upload_ok()) as mp:
        result = _invoke([
            str(f), "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 0, result.output
    call_url = mp.call_args.args[0]
    assert call_url == "http://node:8000/content/upload", (
        f"Sprint 832 regressed — CLI now targets {call_url!r}. "
        f"Must hit /content/upload, not /api/v1/content/upload."
    )


def test_upload_never_hits_phantom_url(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("test\n")
    with patch("httpx.post", return_value=_upload_ok()) as mp:
        _invoke([str(f), "--api-url", "http://node:8000"])
    call_url = mp.call_args.args[0]
    assert "/api/v1/content/upload" not in call_url


def test_upload_body_is_json_not_multipart(tmp_path: Path):
    """Inline /content/upload accepts JSON ContentUploadRequest.
    Pre-832 the CLI sent multipart; that won't validate against
    the Pydantic model."""
    f = tmp_path / "test.txt"
    f.write_text("hello\n")
    with patch("httpx.post", return_value=_upload_ok()) as mp:
        _invoke([str(f), "--api-url", "http://node:8000"])
    kwargs = mp.call_args.kwargs
    assert "json" in kwargs, "Body must be JSON, not multipart"
    body = kwargs["json"]
    assert body["text"] == "hello\n"
    assert body["filename"] == "test.txt"
    assert body["royalty_rate"] == 0.01
    assert body["replicas"] == 3


def test_upload_parent_cids_parsed_to_list(tmp_path: Path):
    """CLI accepts comma-separated parent_cids string; the
    inline endpoint expects a list of strings (sprint 821 shape)."""
    f = tmp_path / "test.txt"
    f.write_text("hello\n")
    with patch("httpx.post", return_value=_upload_ok()) as mp:
        _invoke([
            str(f), "--api-url", "http://node:8000",
            "--parent-cids", "QmAbc, QmDef ,QmGhi",
        ])
    body = mp.call_args.kwargs["json"]
    assert body["parent_cids"] == ["QmAbc", "QmDef", "QmGhi"]


# ---- Successful response rendering ---------------------------


def test_upload_renders_cid(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("payload\n")
    with patch("httpx.post", return_value=_upload_ok()):
        result = _invoke([str(f), "--api-url", "http://node:8000"])
    assert result.exit_code == 0
    assert "test-cid-abc123" in result.output


# ---- Binary file → actionable error --------------------------


def test_upload_binary_file_redirects_to_publish_shard(tmp_path: Path):
    """Inline endpoint is UTF-8 text only. Binary content must
    fail fast with a redirect to publish-shard."""
    f = tmp_path / "binary.bin"
    # Non-UTF-8 bytes (mid-codepoint continuation)
    f.write_bytes(b"\xff\xfe\xff\xfd\x80\x81")
    with patch("httpx.post") as mp:
        result = _invoke([str(f), "--api-url", "http://node:8000"])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "publish-shard" in flat
    # Should NOT have made an HTTP call
    mp.assert_not_called()


# ---- Actionable error paths ----------------------------------


def test_upload_503_surfaces_actionable_message(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("hello\n")
    bad = MagicMock()
    bad.status_code = 503
    bad.text = '{"detail":"Content uploader not initialized"}'
    with patch("httpx.post", return_value=bad):
        result = _invoke([str(f), "--api-url", "http://node:8000"])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "Content uploader not initialized" in flat
    assert "node start" in flat


def test_upload_connect_error_points_at_node_start(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("hello\n")
    with patch(
        "httpx.post",
        side_effect=httpx.ConnectError("connection refused"),
    ):
        result = _invoke([str(f), "--api-url", "http://node:8000"])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "node start" in flat


def test_upload_no_auth_does_not_block(tmp_path: Path):
    """Sprint 832 removes the legacy auth gate (inline endpoint
    doesn't require login). Pre-832 the command hard-failed on
    no PRIVATE_KEY/JWT; sprint 832 lets the server decide."""
    f = tmp_path / "test.txt"
    f.write_text("hello\n")
    # Patch _auth_headers to return None (no creds available)
    with patch("prsm.cli._auth_headers", return_value=None), \
         patch("httpx.post", return_value=_upload_ok()):
        result = _invoke([str(f), "--api-url", "http://node:8000"])
    # Pre-832 would exit 1 with "Not logged in"; sprint 832 proceeds.
    assert result.exit_code == 0
    assert "Not logged in" not in result.output
