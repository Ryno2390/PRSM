"""Sprint 491 — F29 sharding-path source pins.

F29: `ContentUploader.upload_text` had a code path that
called `self._get_content_sharder()` (line 1587 + line
2555) but the method was never defined and ContentSharder
class doesn't exist in the codebase. For content >
sharding_threshold (10MB default), this raised
AttributeError which the outer try/except swallowed, then
upload_text returned None, then the API handler bubbled a
generic 502 "upload_text returned None" with no path
forward for the operator.

Live impact: any upload between 10MB and the PRSM_MAX_UPLOAD_
BYTES env cap silently failed with a cryptic 502. Vision
§11's multi-GB Tier B/C claim was actually broken at 10MB
unless operators used /content/upload/shard explicitly.

Sprint 491 fix:
  - upload_text now raises NotImplementedError when
    content > sharding_threshold, with an actionable
    detail directing operators to /content/upload/shard.
  - The API handler converts NotImplementedError to a
    clean 413 with that detail.

These pins defend the contract.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_upload_text_raises_not_implemented_for_oversize():
    """Source pin: the should_shard branch must raise
    NotImplementedError with an actionable message. Pre-fix
    it called `await self._upload_with_sharding(...)` which
    hit AttributeError on the non-existent
    `_get_content_sharder` method."""
    src = (
        REPO_ROOT / "prsm" / "node" / "content_uploader.py"
    ).read_text()
    idx = src.find("# Check if content should be sharded")
    assert idx >= 0, "should_shard branch missing"
    region = src[idx:idx + 3000]
    assert "raise NotImplementedError(" in region, (
        "F29 regression: should_shard branch must raise "
        "NotImplementedError until ContentSharder is "
        "actually implemented"
    )
    assert "/content/upload/shard" in region, (
        "actionable detail must point operators to "
        "/content/upload/shard"
    )


def test_api_handler_converts_not_implemented_to_413():
    """Source pin: the /content/upload handler must catch
    NotImplementedError specifically and convert to 413
    (Payload Too Large) — not let it fall through to the
    generic 502 catch-all."""
    api_src = (
        REPO_ROOT / "prsm" / "node" / "api.py"
    ).read_text()
    idx = api_src.find(
        "result = await node.content_uploader.upload_text"
    )
    assert idx >= 0
    region = api_src[idx:idx + 3000]
    assert "except NotImplementedError" in region, (
        "/content/upload handler must catch "
        "NotImplementedError as a distinct error class — "
        "without this, F29's actionable detail gets buried "
        "in a generic 502"
    )
    assert "status_code=413" in region


def test_content_sharder_class_does_not_exist_yet():
    """Defensive pin: if a future contributor adds a
    `ContentSharder` class, this test fails and reminds them
    to also wire it into `_get_content_sharder` so the
    F29 NotImplementedError guard can be safely removed.

    The test is INVERTED — it asserts the class STILL doesn't
    exist as documentation of the deferred work."""
    import subprocess
    # Search all prsm/ Python files (excluding tests).
    out = subprocess.run(
        ["grep", "-rn", "class ContentSharder",
         str(REPO_ROOT / "prsm")],
        capture_output=True, text=True,
    )
    if out.stdout.strip():
        import pytest
        pytest.fail(
            f"ContentSharder class now exists in:\n"
            f"{out.stdout}\n"
            f"Wire it through `_get_content_sharder` in "
            f"ContentUploader and remove the F29 "
            f"NotImplementedError guard in "
            f"`content_uploader.py` line ~1393."
        )


def test_sharding_threshold_constant_exists():
    """Pin the threshold constant — operators rely on the
    default (10MB) matching the env cap default."""
    src = (
        REPO_ROOT / "prsm" / "node" / "content_uploader.py"
    ).read_text()
    assert "DEFAULT_SHARDING_THRESHOLD" in src
    # The default should be 10MB (10 * 1024 * 1024 =
    # 10485760).
    assert "10485760" in src or "10 * 1024 * 1024" in src or (
        "10MB" in src and "sharding threshold" in src.lower()
    )
