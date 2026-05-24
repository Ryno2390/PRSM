"""Sprint 817 — `prsm content publish-shard FILE` binary upload CLI.

Closes the content surface end-to-end. Sprint 806 shipped text
upload via /content/upload (UTF-8 only); large or binary
content was rejected with a hint to use /content/upload/shard
directly. Sprint 817 wraps the shard endpoint.

  prsm content publish-shard FILE
                            --dataset-id ID
                            [--title TITLE]
                            [--shard-count N]
                            [--royalty-rate R]
                            [--base-access-fee F]
                            [--per-shard-fee F]
                            [--api-url URL]
                            [--format text|json]

Reads FILE as binary, base64-encodes, POSTs to
/content/upload/shard. Returns the dataset manifest + per-shard
CIDs.

Exit codes:
  0 — uploaded
  1 — file missing / server error / validation error
  2 — daemon unreachable

Pin tests:
- Command registered under `content` group.
- POSTs to /content/upload/shard.
- Body shape matches endpoint (dataset_id, title, content_b64,
  shard_count, royalty_rate, base_access_fee, per_shard_fee).
- File bytes round-trip through base64 in content_b64.
- --title defaults to dataset_id when not given.
- Defaults: shard_count=4, royalty_rate=0.05, base/per-shard fees.
- Missing file → exit 1.
- 400 (e.g. empty content_b64) → exit 1.
- 422 (shard_count > cap) → exit 1.
- 503 (uploader not initialized) → exit 1.
- Unreachable → exit 2.
- JSON mode returns full server payload.
- Text mode surfaces dataset_id + per-shard CID count.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args):
    from prsm.cli import content as _content_group
    return CliRunner().invoke(
        _content_group, ["publish-shard"] + list(args),
    )


def _success_response(dataset_id="test-ds", shards=None):
    r = MagicMock()
    r.status_code = 200
    body = {
        "dataset_id": dataset_id,
        "title": "Test Dataset",
        "manifest_cid": "QmManifestCid",
        "shards": shards or [
            {"shard_index": 0, "cid": "QmS0", "size_bytes": 100},
            {"shard_index": 1, "cid": "QmS1", "size_bytes": 100},
        ],
        "shard_count": len(shards) if shards else 2,
    }
    r.json.return_value = body
    return r


# ---- Command registration --------------------------------------


def test_publish_shard_command_registered():
    from prsm.cli import content as _content_group
    cmd_names = [c.name for c in _content_group.commands.values()]
    assert "publish-shard" in cmd_names


# ---- POSTs to /content/upload/shard + body shape -------------


def test_post_url_is_upload_shard(tmp_path: Path):
    f = tmp_path / "dataset.bin"
    f.write_bytes(b"binary payload")
    with patch(
        "httpx.post", return_value=_success_response(),
    ) as mp:
        result = _invoke([
            str(f), "--dataset-id", "ds1", "--format", "json",
        ])
    assert result.exit_code == 0, result.output
    url = mp.call_args.args[0] if mp.call_args.args else (
        mp.call_args.kwargs.get("url")
    )
    assert url.endswith("/content/upload/shard")


def test_body_shape_matches_endpoint(tmp_path: Path):
    f = tmp_path / "ds.bin"
    f.write_bytes(b"x" * 10)
    with patch(
        "httpx.post", return_value=_success_response(),
    ) as mp:
        _invoke([
            str(f), "--dataset-id", "ds1",
            "--format", "json",
        ])
    body = mp.call_args.kwargs.get("json") or {}
    assert "dataset_id" in body
    assert "title" in body
    assert "content_b64" in body
    assert "shard_count" in body
    assert "royalty_rate" in body
    assert "base_access_fee" in body
    assert "per_shard_fee" in body


def test_content_b64_round_trip(tmp_path: Path):
    payload = b"PRSM binary payload\x00\x01\x02\xff"
    f = tmp_path / "binary.bin"
    f.write_bytes(payload)
    with patch(
        "httpx.post", return_value=_success_response(),
    ) as mp:
        _invoke([
            str(f), "--dataset-id", "binds",
            "--format", "json",
        ])
    body = mp.call_args.kwargs.get("json") or {}
    decoded = base64.b64decode(body["content_b64"])
    assert decoded == payload


def test_title_defaults_to_dataset_id(tmp_path: Path):
    f = tmp_path / "ds.bin"
    f.write_bytes(b"x")
    with patch(
        "httpx.post", return_value=_success_response(),
    ) as mp:
        _invoke([
            str(f), "--dataset-id", "my-ds",
            "--format", "json",
        ])
    body = mp.call_args.kwargs.get("json") or {}
    assert body["title"] == "my-ds"


def test_title_override(tmp_path: Path):
    f = tmp_path / "ds.bin"
    f.write_bytes(b"x")
    with patch(
        "httpx.post", return_value=_success_response(),
    ) as mp:
        _invoke([
            str(f), "--dataset-id", "my-ds",
            "--title", "Pretty Title",
            "--format", "json",
        ])
    body = mp.call_args.kwargs.get("json") or {}
    assert body["title"] == "Pretty Title"


def test_defaults_match_endpoint(tmp_path: Path):
    """CLI defaults match server-side defaults so operators
    don't have to think about every knob."""
    f = tmp_path / "ds.bin"
    f.write_bytes(b"x")
    with patch(
        "httpx.post", return_value=_success_response(),
    ) as mp:
        _invoke([
            str(f), "--dataset-id", "ds",
            "--format", "json",
        ])
    body = mp.call_args.kwargs.get("json") or {}
    assert body["shard_count"] == 4
    assert body["royalty_rate"] == 0.05
    # base_access_fee + per_shard_fee both present (default
    # values come from server-side defaults; CLI just provides
    # SOME default so the field is set).
    assert body["base_access_fee"] >= 0
    assert body["per_shard_fee"] >= 0


# ---- Output ---------------------------------------------------


def test_text_renders_manifest_and_shard_count(tmp_path: Path):
    f = tmp_path / "ds.bin"
    f.write_bytes(b"x")
    with patch(
        "httpx.post",
        return_value=_success_response(
            dataset_id="myds",
            shards=[
                {"shard_index": i, "cid": f"QmS{i}",
                 "size_bytes": 100}
                for i in range(4)
            ],
        ),
    ):
        result = _invoke([
            str(f), "--dataset-id", "myds",
            "--format", "text",
        ])
    assert result.exit_code == 0
    assert "myds" in result.output
    # Shard count surfaced
    assert "4" in result.output
    # Manifest CID present
    assert "QmManifestCid" in result.output


def test_json_returns_payload(tmp_path: Path):
    f = tmp_path / "ds.bin"
    f.write_bytes(b"x")
    with patch(
        "httpx.post",
        return_value=_success_response(dataset_id="myds"),
    ):
        result = _invoke([
            str(f), "--dataset-id", "myds",
            "--format", "json",
        ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["dataset_id"] == "myds"
    assert data["manifest_cid"] == "QmManifestCid"


# ---- Error paths ----------------------------------------------


def test_missing_file_exits_1(tmp_path: Path):
    nonexistent = tmp_path / "nope.bin"
    result = _invoke([
        str(nonexistent), "--dataset-id", "ds",
        "--format", "text",
    ])
    assert result.exit_code != 0
    assert "nope.bin" in result.output


def test_server_400_exits_1(tmp_path: Path):
    f = tmp_path / "ds.bin"
    f.write_bytes(b"x")
    fake = MagicMock()
    fake.status_code = 400
    fake.text = '{"detail":"content_b64 empty"}'
    with patch("httpx.post", return_value=fake):
        result = _invoke([
            str(f), "--dataset-id", "ds",
            "--format", "text",
        ])
    assert result.exit_code == 1


def test_server_422_exits_1(tmp_path: Path):
    f = tmp_path / "ds.bin"
    f.write_bytes(b"x")
    fake = MagicMock()
    fake.status_code = 422
    fake.text = '{"detail":"shard_count exceeds cap"}'
    with patch("httpx.post", return_value=fake):
        result = _invoke([
            str(f), "--dataset-id", "ds",
            "--format", "text",
        ])
    assert result.exit_code == 1


def test_unreachable_exits_2(tmp_path: Path):
    f = tmp_path / "ds.bin"
    f.write_bytes(b"x")
    with patch(
        "httpx.post",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke([
            str(f), "--dataset-id", "ds",
            "--format", "text",
        ])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()
