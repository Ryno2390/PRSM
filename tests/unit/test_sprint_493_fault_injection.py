"""Sprint 493 — fault injection / chaos fixes (F33).

Coverage matrix priority #4: graceful degradation under
fault. Sprint 493 walked these scenarios:

  A. Staged file deleted mid-life → retrieve returns clean
     `not_found` in <5s (sprint 484 timeout-propagation
     + sprint 485 seed-registration + inherent
     `local_path.is_file()` check combine for fail-safe).
     No fix needed. ✅

  B. Staging dir read-only (chmod -w) → F33: upload
     returned cryptic "upload_text returned None" 502
     because `_publish_content` swallowed the
     PermissionError with `except Exception: return None`.
     Operator had NO clue what failed.

  C. Staging dir renamed mid-life → FileNotFoundError,
     same swallow path as B.

  D. SQLite DB locked by external writer (6s IMMEDIATE
     lock) → daemon WAITS cleanly, request succeeds when
     lock releases. Aiosqlite's busy_timeout handles
     contention. No fix needed. ✅

Sprint 493 fix (F33): `_publish_content` re-raises
exceptions instead of swallowing. The API handler's
existing `except Exception: HTTP 502 with detail=<type:msg>`
then surfaces the real error to operators.

Live-verified post-fix:
  - chmod -w staging dir → 502 "PermissionError: [Errno 13]
    Permission denied: '/.../staging/<hash>.tmp'"
  - rename staging dir → 502 "FileNotFoundError: [Errno 2]
    No such file or directory: '/.../staging/<hash>.tmp'"
  - daemon stays healthy across both scenarios

These pins defend the F33 fix in source.
"""
from __future__ import annotations

import inspect
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_f33_publish_content_re_raises_exceptions():
    """Source pin: `_publish_content` must `raise` (not
    `return None`) in the except block. Pre-fix it
    swallowed everything and returned None, making the
    handler bubble a generic 502 with no diagnostic detail."""
    src = (
        REPO_ROOT / "prsm" / "node" / "content_uploader.py"
    ).read_text()
    idx = src.find("async def _publish_content")
    assert idx >= 0
    # Take the next ~1500 chars covering the method body.
    body = src[idx:idx + 1500]
    # The sprint-493 fix marker must remain.
    assert "Sprint 493 (F33 fix)" in body, (
        "F33 regression risk: sprint 493 marker missing — "
        "_publish_content may have reverted to swallowing"
    )
    # Find the except clause + verify it RAISES not RETURNS NONE.
    except_idx = body.find("except Exception as e:")
    assert except_idx >= 0, (
        "_publish_content missing except Exception block"
    )
    except_block = body[except_idx:except_idx + 800]
    assert "raise" in except_block, (
        "F33 regression: except Exception block must RAISE, "
        "not silently return None — operators need the error "
        "surfaced in the 502 detail"
    )
    # Defensive: the buggy 'return None' pattern inside the
    # except block must NOT appear.
    # (logger.error followed immediately by `return None`)
    assert "return None\n" not in (
        except_block.replace("logger.error", "")[:200]
    ), (
        "F33 regression: silent `return None` in except "
        "block re-introduced"
    )


def test_f33_api_handler_surfaces_exception_in_502_detail():
    """Source pin: the /content/upload handler's broad
    `except Exception` clause must include the exception
    type + message in the 502 detail. Sprint 491 (F29)
    added a specific NotImplementedError catch above; the
    catch-all below must remain rich for F33-class errors."""
    api_src = (
        REPO_ROOT / "prsm" / "node" / "api.py"
    ).read_text()
    idx = api_src.find(
        "result = await node.content_uploader.upload_text"
    )
    assert idx >= 0
    region = api_src[idx:idx + 3000]
    # Confirm the rich-detail pattern is preserved.
    assert (
        'f"Upload failed: {type(exc).__name__}: {exc}"'
        in region
        or "f\"Upload failed: {type(exc).__name__}" in region
    ), (
        "API handler must include exception type + message "
        "in 502 detail for F33-class operator visibility"
    )


def test_f33_retrieve_fails_safe_when_staged_file_missing():
    """Documented invariant pin (no source check — this
    just records that sprint 493 chaos-tested this
    behavior). Sprint 484's timeout propagation + sprint
    485's seed registration + the `local_path.is_file()`
    guard in ContentRetriever.fetch combine so that a
    retrieve for a CID whose staged file has been DELETED
    returns a clean `not_found` in <5s rather than:
      - hanging forever (pre-sprint 484)
      - crashing (no fix shipped, but worth pinning)
      - serving stale or partial bytes
    """
    src = (
        REPO_ROOT / "prsm" / "node" / "content_publisher.py"
    ).read_text()
    # The fetch path must check is_file() before reading.
    idx = src.find("async def fetch")
    assert idx >= 0
    body = src[idx:idx + 3000]
    assert "is_file()" in body, (
        "ContentRetriever.fetch must check local_path.is_file() "
        "before read_bytes() — otherwise deleted staged files "
        "raise FileNotFoundError mid-retrieve"
    )
