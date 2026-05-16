"""Sprint 495 — F37 staging-file cleanup on BT seed failure.

Coverage matrix priority #6 (LR column): long-running
stability. 60-second sustained-load soak revealed:

  - Pre-soak RSS: 147 MB / 338 FDs / 23 threads
  - Post-soak RSS: 545 MB / 341 FDs / 26 threads
  - RSS growth: +388 MB in 60s (≈ 6.5 MB/s)
  - Staging dir growth: 36,337 leaked files (0→36k+)
  - Daemon log: thousands of "BitTorrent seed_content
    returned None" errors

Root cause (F37): `ContentPublisher._publish_tier_a`
wrote the staged file BEFORE calling `bt_provider.
seed_content`. On seed failure, the code raised
RuntimeError with the message "The staged file is
preserved; retry is safe." but the daemon has no
retry mechanism that uses the preserved file. Under
sustained-failure load (libtorrent flakiness, disk
pressure, etc.), staged files accumulated indefinitely
+ in-process libtorrent state grew with each failed
seed.

Sprint 495 fix: clean up the staged file on BT seed
failure IF this publish call created it (tracked via
`we_wrote_file` flag — preserves the file when a
concurrent publish of identical bytes is in flight).
Two failure modes covered:
  - seed_content raises Exception → cleanup + re-raise
  - seed_content returns None → cleanup + raise the
    "returned None" RuntimeError (sprint 493 F33 fix
    surfaces this to the operator as a clean 502)

Live-verified post-fix (30s soak, same unique-content
pattern):
  - RSS growth: +13 MB (vs +388 MB pre-fix)
  - Staging files: +50 (= number of successful ops; vs
    36,337 leaked pre-fix)

F36 — total RSS growth even pre-F37 had a component
not attributable to staging files alone (some
libtorrent in-process state). Properly profiling that
requires a Python heap profiler beyond single-sprint
scope; F37 closure captures the bulk of the leak.

These pins defend the cleanup contract.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_f37_staged_file_cleanup_on_seed_exception():
    """Source pin: `_publish_tier_a` must clean up the
    staged file when `bt_provider.seed_content` raises.
    Pre-fix it propagated the exception without cleanup."""
    src = (
        REPO_ROOT / "prsm" / "node" / "content_publisher.py"
    ).read_text()
    idx = src.find("async def _publish_tier_a")
    assert idx >= 0
    body = src[idx:idx + 6000]
    assert "Sprint 495 (F37 fix)" in body, (
        "F37 marker missing from _publish_tier_a"
    )
    # Cleanup must reference unlink + missing_ok=True
    assert "staged_path.unlink(missing_ok=True)" in body, (
        "F37 regression: cleanup-on-failure missing"
    )


def test_f37_we_wrote_file_flag_guards_cleanup():
    """Defense-in-depth: cleanup must be gated by
    `we_wrote_file` so a concurrent publish of identical
    bytes (deterministic content_hash naming) doesn't
    have its file deleted by a sibling's failure."""
    src = (
        REPO_ROOT / "prsm" / "node" / "content_publisher.py"
    ).read_text()
    idx = src.find("async def _publish_tier_a")
    body = src[idx:idx + 6000]
    assert "we_wrote_file = False" in body
    assert "we_wrote_file = True" in body
    assert "if we_wrote_file:" in body, (
        "cleanup must be guarded by we_wrote_file flag — "
        "without this, a concurrent publish of identical "
        "bytes could have its file pulled out from under it"
    )


def test_f37_both_failure_modes_covered():
    """The fix covers BOTH:
    - seed_content raises Exception
    - seed_content returns None (no exception)
    Both must trigger cleanup."""
    src = (
        REPO_ROOT / "prsm" / "node" / "content_publisher.py"
    ).read_text()
    idx = src.find("async def _publish_tier_a")
    body = src[idx:idx + 6000]
    # Count unlink calls in the method body — must be 2
    # (one in except, one in `if manifest is None`).
    unlink_count = body.count("staged_path.unlink")
    assert unlink_count >= 2, (
        f"F37 fix must clean up in BOTH failure paths "
        f"(except + None-return); got {unlink_count} "
        f"unlink calls"
    )


def test_f37_misleading_preserve_message_removed():
    """The old "The staged file is preserved; retry is
    safe." message was misleading — no retry path
    consumed the preserved file. Sprint 495 removed it
    from the RuntimeError detail."""
    src = (
        REPO_ROOT / "prsm" / "node" / "content_publisher.py"
    ).read_text()
    idx = src.find("async def _publish_tier_a")
    body = src[idx:idx + 6000]
    assert "The staged file is preserved" not in body, (
        "F37 regression: misleading 'preserved' message "
        "reintroduced — no retry path actually uses the "
        "preserved file, so the claim was false advertising"
    )
