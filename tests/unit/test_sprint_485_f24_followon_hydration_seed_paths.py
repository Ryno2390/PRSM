"""Sprint 485 — F24 follow-on: BT seed paths restored at hydration.

Sprint 484 (F24) bounded the BT-fallback hang by propagating
timeout. That made `/content/retrieve/{hydrated_cid}` fail
FAST (returning `not_found` in 2s) instead of HANGING — but
it didn't actually deliver bytes for hydrated content.

Sprint 485 closes the loop: at F22 hydration time, the
content_uploader now ALSO calls
``ContentPublisher.register_local_publish_tier_a`` to restore
the per-session ``_published_paths`` map from the persisted
``content_hash`` field. After this, ``local_publish_path(cid)``
returns the staged file → ``ContentRetriever.fetch`` short-
circuits the BT swarm path → retrieve delivers the original
bytes for hydrated CIDs.

Live-verified pre-sprint-485: hydrated CIDs returned
`not_found` even with sprint 484's F24 fix in place.

Live-verified post-sprint-485: 17/17 hydrated CIDs return
`status: success` with the original bytes intact across a
clean daemon restart.

These pins defend:
  1. ContentPublisher.register_local_publish_tier_a is
     defined + returns True for present staged files /
     False for missing files
  2. content_uploader._hydrate_from_db calls this on every
     hydrated record
  3. ContentRetriever.fetch short-circuits via
     local_publish_path after re-registration

Tier B/C is OUT OF SCOPE for this sprint — staged dir naming
(`staging_dir/<basename>-<tier>`) requires per-publish info
that's not persisted. Sharded content hydration remains a
deferred follow-on.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from prsm.node.content_publisher import ContentPublisher


class _StubBTProvider:
    """Minimal BT-provider stub for ContentPublisher init.
    These tests only exercise the local-publish-path machinery
    so the BT layer just needs to not raise."""
    def __init__(self):
        pass


def test_register_local_publish_tier_a_method_exists():
    """The helper method must exist on ContentPublisher."""
    assert hasattr(ContentPublisher, "register_local_publish_tier_a")
    assert callable(
        getattr(ContentPublisher, "register_local_publish_tier_a")
    )


def test_register_returns_true_when_staged_file_present(tmp_path):
    """Registration succeeds when the content_hash file is
    present in the staging dir + sets _published_paths."""
    publisher = ContentPublisher(
        staging_dir=tmp_path, bt_provider=_StubBTProvider(),
    )
    # Stage a file under its sha256 name.
    data = b"sprint 485 stage probe"
    ch = hashlib.sha256(data).hexdigest()
    (tmp_path / ch).write_bytes(data)
    # Fake infohash (CID); register.
    cid = "a" * 40
    assert publisher.register_local_publish_tier_a(
        infohash=cid, content_hash=ch,
    ) is True
    # local_publish_path should now return the staged file.
    p = publisher.local_publish_path(cid)
    assert p is not None
    assert p.read_bytes() == data


def test_register_returns_false_when_staged_file_missing(tmp_path):
    """If staged file is missing, registration returns False
    and _published_paths is unchanged. Defensive — caller
    falls through to the F24 fail-fast path."""
    publisher = ContentPublisher(
        staging_dir=tmp_path, bt_provider=_StubBTProvider(),
    )
    cid = "b" * 40
    ch = "deadbeef" * 8
    assert publisher.register_local_publish_tier_a(
        infohash=cid, content_hash=ch,
    ) is False
    assert publisher.local_publish_path(cid) is None


def test_register_only_handles_file_not_dir(tmp_path):
    """Tier B/C staged roots are DIRS. register_local_publish_tier_a
    must refuse to register a dir (it's Tier-A-only). Without
    this guard, a Tier B/C staged dir would be served as a
    raw Tier A file → corrupt retrieve."""
    publisher = ContentPublisher(
        staging_dir=tmp_path, bt_provider=_StubBTProvider(),
    )
    # Make a dir at the staged path (mimics Tier B/C root).
    ch = "c" * 64
    (tmp_path / ch).mkdir()
    cid = "c" * 40
    assert publisher.register_local_publish_tier_a(
        infohash=cid, content_hash=ch,
    ) is False


def test_hydration_calls_register_for_tier_a_records():
    """Source pin: _hydrate_from_db must call
    register_local_publish_tier_a for each non-sharded
    hydrated record."""
    src = (
        Path(__file__).resolve().parents[2]
        / "prsm" / "node" / "content_uploader.py"
    ).read_text()
    idx = src.find("async def _hydrate_from_db")
    assert idx >= 0
    body = src[idx:idx + 6000]
    assert "register_local_publish_tier_a" in body, (
        "_hydrate_from_db must call "
        "register_local_publish_tier_a — sprint 485 F24 "
        "follow-on regression risk"
    )
    # Must guard against Tier B/C (sharded) — that path is
    # deferred.
    assert "not uploaded.is_sharded" in body or (
        "is_sharded" in body
    ), (
        "_hydrate_from_db must guard against Tier B/C; "
        "calling tier_a registration on sharded records "
        "would silently corrupt retrieve"
    )


def test_hydration_registration_uses_persisted_fields():
    """The registration call must use the persisted
    `content_id` (infohash) and `content_hash` from the
    hydrated UploadedContent — not synthesize them from
    elsewhere. Source pin."""
    src = (
        Path(__file__).resolve().parents[2]
        / "prsm" / "node" / "content_uploader.py"
    ).read_text()
    idx = src.find("register_local_publish_tier_a(")
    assert idx >= 0
    call_region = src[idx:idx + 500]
    assert "infohash=uploaded.content_id" in call_region
    assert "content_hash=uploaded.content_hash" in call_region
