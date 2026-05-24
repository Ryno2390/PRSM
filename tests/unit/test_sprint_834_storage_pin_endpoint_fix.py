"""Sprint 834 — F29 fix: `prsm storage pin` + `pins` + new
inline /content/{cid}/pin endpoint.

Fourth cascade fix from sprint 830's audit. Both `prsm storage
pin <cid>` and `prsm storage pins` hit /api/v1/storage/...
phantoms — but unlike sprints 831/832/833, no router file
actually defines those endpoints (no prsm/interface/api/
storage_api.py exists). The CLI was pointing at URLs that have
NEVER existed.

Sprint 834 closes the gap by:
1. Adding inline /content/{cid}/pin POST endpoint in
   node/api.py wired to StorageProvider.pin_content (sprint 263).
2. Switching `prsm storage pin` to the new endpoint.
3. Switching `prsm storage pins` to inline /storage/pinned-stats
   (sprint 263 existing surface).

CLI signature simplification: dropped `--replication` option
(the inline endpoint promotes a single-node pin; cross-node
replication is sprint-263's replica management territory).

Response-shape adjustment: pre-834 the CLI expected
{replication, monthly_cost}; inline /storage/pinned-stats
returns {pinned:[{cid, size_bytes, pinned_at,
successful_challenges, failed_challenges}], count}. The CLI is
updated to match.

Live-attested 2026-05-24:
  $ prsm storage pin deadbeef000000...
  → exit 1, "CID not present locally" (correct 404 path)
  $ prsm storage pins
  → exit 0, "No pinned content." (correct empty-state)

**Follow-on F30 surfaced**: After `prsm storage upload`, the
returned CID does NOT resolve via StorageProvider.exists_local
even though the upload succeeded. Likely a CID-vs-content_hash
representation mismatch between upload path + ContentStore
lookup. Deferred to sprint 835. Sprint 834 ships the URL fix +
new endpoint; the deeper exists_local gap is a separate sprint.

Pin tests:
- pin URL is /content/{cid}/pin
- pins URL is /storage/pinned-stats
- phantom /api/v1/storage/ never appears
- 404 → "CID not present locally" actionable message
- 503 → "Storage provider not initialized"
- ConnectError → "node start" hint
- pins empty-state renders cleanly
- pins with entries renders the new shape (size_bytes,
  successful_challenges, failed_challenges)
- Inline endpoint exists in create_api_app (source pin)
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import httpx
from click.testing import CliRunner


def _invoke(cmd, args):
    from prsm.cli import storage as _storage_group
    return CliRunner().invoke(_storage_group, [cmd] + list(args))


def _pin_ok():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "pinned": True,
        "cid": "abc123",
        "size_bytes": 42,
    }
    return r


def _pin_not_found():
    r = MagicMock()
    r.status_code = 404
    r.text = '{"detail":"CID not present locally"}'
    return r


def _pin_503():
    r = MagicMock()
    r.status_code = 503
    r.text = '{"detail":"Storage provider not initialized."}'
    return r


def _pins_with_entries():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "pinned": [
            {
                "cid": "cid1abcdef",
                "size_bytes": 100,
                "pinned_at": 1779631000.0,
                "successful_challenges": 5,
                "failed_challenges": 0,
            },
            {
                "cid": "cid2",
                "size_bytes": 200,
                "pinned_at": 1779631100.0,
                "successful_challenges": 3,
                "failed_challenges": 1,
            },
        ],
        "count": 2,
    }
    return r


def _pins_empty():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"pinned": [], "count": 0}
    return r


# ---- URL regression guards (F29 lesson) ----------------------


def test_pin_hits_inline_content_pin_endpoint():
    with patch("httpx.post", return_value=_pin_ok()) as mp:
        result = _invoke("pin", [
            "abc123", "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 0
    call_url = mp.call_args.args[0]
    assert call_url == "http://node:8000/content/abc123/pin"


def test_pin_never_hits_phantom_storage_url():
    with patch("httpx.post", return_value=_pin_ok()) as mp:
        _invoke("pin", ["abc", "--api-url", "http://node:8000"])
    call_url = mp.call_args.args[0]
    assert "/api/v1/storage/" not in call_url


def test_pins_hits_inline_pinned_stats_endpoint():
    with patch("httpx.get", return_value=_pins_empty()) as mg:
        _invoke("pins", ["--api-url", "http://node:8000"])
    call_url = mg.call_args.args[0]
    assert call_url == "http://node:8000/storage/pinned-stats"


def test_pins_never_hits_phantom_storage_url():
    with patch("httpx.get", return_value=_pins_empty()) as mg:
        _invoke("pins", ["--api-url", "http://node:8000"])
    call_url = mg.call_args.args[0]
    assert "/api/v1/storage/pins" not in call_url


# ---- Pin success + error paths -------------------------------


def test_pin_success_renders_size():
    with patch("httpx.post", return_value=_pin_ok()):
        result = _invoke("pin", [
            "abc123", "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 0
    assert "Content pinned" in result.output
    assert "42 bytes" in result.output


def test_pin_404_surfaces_actionable_message():
    with patch("httpx.post", return_value=_pin_not_found()):
        result = _invoke("pin", [
            "missingcid", "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "CID not present locally" in flat
    assert "Upload or retrieve" in flat


def test_pin_503_surfaces_node_start_hint():
    with patch("httpx.post", return_value=_pin_503()):
        result = _invoke("pin", [
            "abc", "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "Storage provider not initialized" in flat
    assert "node start" in flat


def test_pin_connect_error_points_at_node_start():
    with patch(
        "httpx.post",
        side_effect=httpx.ConnectError("conn refused"),
    ):
        result = _invoke("pin", [
            "abc", "--api-url", "http://node:8000",
        ])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "node start" in flat


# ---- Pins list rendering -------------------------------------


def test_pins_empty_state_renders_cleanly():
    with patch("httpx.get", return_value=_pins_empty()):
        result = _invoke("pins", ["--api-url", "http://node:8000"])
    assert result.exit_code == 0
    assert "No pinned content" in result.output


def test_pins_with_entries_renders_new_shape():
    """Sprint 834 shape: size_bytes/pinned_at/successful_challenges/
    failed_challenges columns. Pre-834 expected replication +
    monthly_cost which the inline endpoint doesn't provide."""
    with patch("httpx.get", return_value=_pins_with_entries()):
        result = _invoke("pins", ["--api-url", "http://node:8000"])
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "100 bytes" in flat
    assert "200 bytes" in flat
    # Challenges column: "5 / 0" and "3 / 1"
    assert "5 / 0" in flat
    assert "3 / 1" in flat


# ---- Endpoint source pin -------------------------------------


def test_create_api_app_defines_content_pin_endpoint():
    """Source-shape pin: defends sprint 834's new
    /content/{cid}/pin endpoint against accidental removal."""
    from prsm.node import api as _api_mod
    src = inspect.getsource(_api_mod.create_api_app)
    assert '"/content/{cid}/pin"' in src
    assert "pin_content" in src
