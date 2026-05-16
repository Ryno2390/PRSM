"""Sprint 491 — content scale stress harness.

Coverage matrix priority #2 from sprint 486: zero scale
testing. Largest content tested as of sprint 490 was
~500B. Vision §11 claims multi-GB Tier B/C support.

Sprint 491 walks the size axis: 1KB → 100KB → 1MB → 10MB
→ 100MB. For each:
  - upload latency
  - retrieve latency
  - byte-identity (SHA-256 round-trip)
  - daemon RSS growth (operator-observable resource cost)

Hits two documented limits as it walks:
  - PRSM_MAX_UPLOAD_BYTES env-driven runtime cap (default
    10MB; tests use higher for the > 10MB cases)
  - Pydantic max_length=100MB (sprint 333 — anything larger
    requires /content/upload/shard)

Surfaces NEW findings (F29+) where observed costs are
significantly worse than linear.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

import pytest


DAEMON_URL = os.environ.get(
    "PRSM_DAEMON_URL", "http://127.0.0.1:8000",
).rstrip("/")


def _daemon_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{DAEMON_URL}/health", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _daemon_reachable(),
    reason=f"no daemon at {DAEMON_URL}",
)


def _rss_kb_for(pid: int) -> int:
    """Read RSS in KB for a process (macOS)."""
    import subprocess
    out = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(pid)], text=True,
    )
    return int(out.strip())


def _daemon_pid() -> int:
    """Find the daemon PID. Returns 0 if not findable —
    the harness then skips memory-growth assertions but
    still runs the upload/retrieve tests."""
    import subprocess
    try:
        # `pgrep -f` matching is unreliable under pytest's
        # subprocess context on macOS. Fall back to `ps ax`
        # which always works.
        out = subprocess.run(
            ["ps", "ax", "-o", "pid,command"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            if "prsm.cli node start" in line and "grep" not in line:
                return int(line.strip().split()[0])
    except Exception:  # noqa: BLE001
        pass
    return 0


def _rss_kb_for_safe(pid: int) -> int:
    """RSS in KB, or 0 if pid is 0 or process gone."""
    if pid == 0:
        return 0
    try:
        return _rss_kb_for(pid)
    except Exception:  # noqa: BLE001
        return 0


def _upload(text: bytes, *, timeout_sec: float = 120.0) -> dict:
    """POST to /content/upload + measure timing."""
    payload = json.dumps({
        "text": text.decode("utf-8", errors="replace")
        if len(text) < 50_000_000  # avoid huge str conversions
        else text.decode("latin-1"),
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{DAEMON_URL}/content/upload",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:
            body = json.loads(r.read())
            return {
                "status": r.status,
                "body": body,
                "elapsed_sec": time.time() - t0,
            }
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read())
        except Exception:  # noqa: BLE001
            err_body = None
        return {
            "status": e.code,
            "body": err_body,
            "elapsed_sec": time.time() - t0,
        }


def _retrieve(cid: str, *, timeout_sec: float = 120.0) -> dict:
    req = urllib.request.Request(
        f"{DAEMON_URL}/content/retrieve/{cid}?timeout=60",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:
            body = json.loads(r.read())
            return {
                "status": r.status,
                "body": body,
                "elapsed_sec": time.time() - t0,
            }
    except urllib.error.HTTPError as e:
        return {
            "status": e.code,
            "body": None,
            "elapsed_sec": time.time() - t0,
        }


def _run_round_trip(size_bytes: int) -> dict:
    """Upload size_bytes of random ASCII content + retrieve +
    verify byte-identity. Returns timing + memory metrics."""
    pid = _daemon_pid()
    rss_before = _rss_kb_for_safe(pid)

    # Random content with deterministic hash for byte-identity check
    rng = os.urandom(size_bytes)
    # Use printable hex so it survives JSON encoding cleanly
    text = rng.hex()[:size_bytes].encode("ascii")
    expected_hash = hashlib.sha256(text).hexdigest()

    up = _upload(text)
    rss_after_upload = _rss_kb_for_safe(pid)

    if up["status"] != 200:
        return {
            "size_bytes": size_bytes,
            "upload_status": up["status"],
            "upload_elapsed_sec": up["elapsed_sec"],
            "upload_error": up.get("body"),
            "rss_growth_kb": rss_after_upload - rss_before,
        }

    cid = up["body"]["cid"]
    actual_hash = up["body"]["content_hash"]

    # Wait briefly for BT-seed registration to settle
    time.sleep(0.5)

    re = _retrieve(cid)
    rss_after_retrieve = _rss_kb_for_safe(pid)

    byte_identity_ok = None
    if re["status"] == 200 and re["body"]["status"] == "success":
        import base64
        recovered = base64.b64decode(re["body"]["data"])
        byte_identity_ok = recovered == text

    return {
        "size_bytes": size_bytes,
        "upload_status": up["status"],
        "upload_elapsed_sec": round(up["elapsed_sec"], 3),
        "upload_throughput_mbps": (
            round(
                (size_bytes / 1_000_000) / up["elapsed_sec"], 2,
            )
            if up["elapsed_sec"] > 0 else None
        ),
        "retrieve_status": re["status"],
        "retrieve_elapsed_sec": round(re["elapsed_sec"], 3),
        "retrieve_inner_status": (
            re["body"]["status"] if re["body"] else None
        ),
        "content_hash_match": (
            actual_hash == expected_hash
            if up["status"] == 200 else None
        ),
        "byte_identity_ok": byte_identity_ok,
        "rss_growth_kb_upload": (
            rss_after_upload - rss_before
        ),
        "rss_growth_kb_total": (
            rss_after_retrieve - rss_before
        ),
        "cid": cid if up["status"] == 200 else None,
    }


def test_scale_1KB():
    r = _run_round_trip(1 * 1024)
    print(f"\n1KB: {r}")
    assert r["upload_status"] == 200
    assert r["content_hash_match"] is True
    assert r["byte_identity_ok"] is True
    # Sub-second for 1KB on any reasonable hardware
    assert r["upload_elapsed_sec"] < 5.0
    assert r["retrieve_elapsed_sec"] < 5.0


def test_scale_100KB():
    r = _run_round_trip(100 * 1024)
    print(f"\n100KB: {r}")
    assert r["upload_status"] == 200
    assert r["content_hash_match"] is True
    assert r["byte_identity_ok"] is True


def test_scale_1MB():
    r = _run_round_trip(1 * 1024 * 1024)
    print(f"\n1MB: {r}")
    assert r["upload_status"] == 200, (
        f"1MB upload failed: {r}"
    )
    assert r["content_hash_match"] is True
    assert r["byte_identity_ok"] is True


def test_scale_10MB():
    r = _run_round_trip(10 * 1024 * 1024)
    print(f"\n10MB: {r}")
    # 10MB hits the default PRSM_MAX_UPLOAD_BYTES cap —
    # may need env override OR may succeed if cap was lifted.
    if r["upload_status"] == 413:
        pytest.skip(
            f"10MB upload rejected by daemon cap "
            f"(PRSM_MAX_UPLOAD_BYTES). To run this test, "
            f"start daemon with PRSM_MAX_UPLOAD_BYTES "
            f"unset or >10MB. Got: {r}"
        )
    assert r["upload_status"] == 200, (
        f"10MB upload failed: {r}"
    )
    assert r["content_hash_match"] is True
    assert r["byte_identity_ok"] is True
    # Operator-observable threshold: 10MB should round-trip
    # in under 60s on dev hardware. Exceeds → log as F29 candidate.
    assert r["upload_elapsed_sec"] < 60.0
    assert r["retrieve_elapsed_sec"] < 60.0


def test_scale_50MB_clean_413():
    """50MB exceeds sharding_threshold (10MB). Sprint 491 (F29)
    confirmed the internal sharding code path references a
    ContentSharder class that doesn't exist — pre-fix this
    crashed with AttributeError + returned a generic 502.

    Sprint 491 fix: `upload_text` raises NotImplementedError
    when content > sharding_threshold; handler converts to
    a clean 413 directing operators to /content/upload/shard.

    This test pins that contract: 50MB upload must return
    413 with the canonical NotImplementedError detail."""
    r = _run_round_trip(50 * 1024 * 1024)
    print(f"\n50MB: {r}")
    # Either 413 from the runtime cap (PRSM_MAX_UPLOAD_BYTES
    # default 10MB) OR 413 from the F29 NotImplementedError
    # path. Both are clean operator signals.
    assert r["upload_status"] == 413, (
        f"50MB should return 413 (not 200 — sharding not "
        f"wired; not 502 — F29 regression). Got: {r}"
    )
    detail = (r.get("upload_error") or {}).get("detail", "")
    # The detail must direct operators to the sharding path.
    assert (
        "upload/shard" in detail
        or "PRSM_MAX_UPLOAD_BYTES" in detail
    ), (
        f"413 detail must surface actionable path "
        f"(/content/upload/shard or env-var name). Got: "
        f"{detail!r}"
    )
