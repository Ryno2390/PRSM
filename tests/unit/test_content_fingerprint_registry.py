"""Sprint 291 — ContentFingerprintRegistry.

Vision §14 mitigation item (3): "Cryptographic deduplication
— the same content under different hashes still pays the
same creator by on-chain record."

The spam pattern this defends against: a creator downloads
someone else's content + re-uploads it under their own
address to claim royalties. Defense: SHA-256 content
fingerprint with first-creator-wins semantics. Subsequent
uploads of the same fingerprint are recognized as duplicates
and royalties route to the canonical (first) creator.

v1 is per-node, in-memory + opt-in filesystem persistence
(mirrors sprint-272 TakedownNoticeRing pattern). Federated
gossip-of-fingerprints is a follow-on (operators eventually
need to agree on global first-uploader status; for now each
operator builds their own per-node view).
"""
from __future__ import annotations

import json

import pytest

from prsm.marketplace.content_fingerprint_registry import (
    ContentFingerprintEntry,
    ContentFingerprintRegistry,
)


# ── First-creator-wins semantics ─────────────────────────


def test_register_first_uploader_wins():
    r = ContentFingerprintRegistry()
    canonical, is_new = r.register(
        content_hash="sha256-abc",
        creator_eth_address="0xalice",
    )
    assert canonical == "0xalice"
    assert is_new is True


def test_register_duplicate_returns_original_creator():
    r = ContentFingerprintRegistry()
    r.register(
        content_hash="sha256-abc",
        creator_eth_address="0xalice",
    )
    canonical, is_new = r.register(
        content_hash="sha256-abc",
        creator_eth_address="0xbob",
    )
    assert canonical == "0xalice"  # Alice was first
    assert is_new is False


def test_register_same_creator_twice_idempotent():
    """Same creator re-uploading their own content is not a
    duplicate-attack — they get back their own address."""
    r = ContentFingerprintRegistry()
    r.register("sha256-abc", "0xalice")
    canonical, is_new = r.register(
        "sha256-abc", "0xalice",
    )
    assert canonical == "0xalice"
    assert is_new is False  # registry already knew


def test_register_validates_content_hash():
    r = ContentFingerprintRegistry()
    with pytest.raises(ValueError):
        r.register(content_hash="", creator_eth_address="0xa")


def test_register_validates_creator_address():
    r = ContentFingerprintRegistry()
    with pytest.raises(ValueError):
        r.register(content_hash="sha256-x", creator_eth_address="")


# ── canonical_creator query ──────────────────────────────


def test_canonical_creator_returns_none_for_unknown():
    r = ContentFingerprintRegistry()
    assert r.canonical_creator("sha256-unknown") is None


def test_canonical_creator_after_register():
    r = ContentFingerprintRegistry()
    r.register("sha256-x", "0xalice")
    assert r.canonical_creator("sha256-x") == "0xalice"


def test_canonical_creator_unaffected_by_duplicate():
    r = ContentFingerprintRegistry()
    r.register("sha256-x", "0xalice")
    r.register("sha256-x", "0xbob")  # bob tries to claim
    # Alice still canonical
    assert r.canonical_creator("sha256-x") == "0xalice"


# ── Duplicate detection helper ───────────────────────────


def test_is_duplicate_for_unknown_returns_false():
    r = ContentFingerprintRegistry()
    assert r.is_duplicate(
        content_hash="sha256-x", creator_eth_address="0xa",
    ) is False


def test_is_duplicate_same_creator_returns_false():
    r = ContentFingerprintRegistry()
    r.register("sha256-x", "0xalice")
    assert r.is_duplicate(
        content_hash="sha256-x", creator_eth_address="0xalice",
    ) is False


def test_is_duplicate_different_creator_returns_true():
    r = ContentFingerprintRegistry()
    r.register("sha256-x", "0xalice")
    assert r.is_duplicate(
        content_hash="sha256-x", creator_eth_address="0xbob",
    ) is True


# ── Entry detail ─────────────────────────────────────────


def test_get_entry_returns_record():
    r = ContentFingerprintRegistry()
    r.register("sha256-x", "0xalice", timestamp=100.0)
    e = r.get_entry("sha256-x")
    assert isinstance(e, ContentFingerprintEntry)
    assert e.content_hash == "sha256-x"
    assert e.canonical_creator == "0xalice"
    assert e.first_seen_unix == 100
    assert e.duplicate_attempt_count == 0


def test_duplicate_attempts_incremented():
    r = ContentFingerprintRegistry()
    r.register("sha256-x", "0xalice")
    r.register("sha256-x", "0xbob")
    r.register("sha256-x", "0xcarol")
    e = r.get_entry("sha256-x")
    assert e.duplicate_attempt_count == 2


def test_same_creator_re_register_not_counted_as_duplicate_attempt():
    r = ContentFingerprintRegistry()
    r.register("sha256-x", "0xalice")
    r.register("sha256-x", "0xalice")  # alice again
    e = r.get_entry("sha256-x")
    assert e.duplicate_attempt_count == 0


def test_entry_to_dict():
    e = ContentFingerprintEntry(
        content_hash="sha256-x",
        canonical_creator="0xalice",
        first_seen_unix=100,
        duplicate_attempt_count=3,
    )
    d = e.to_dict()
    assert d["content_hash"] == "sha256-x"
    assert d["canonical_creator"] == "0xalice"
    assert d["duplicate_attempt_count"] == 3


# ── Recent + count ───────────────────────────────────────


def test_count_empty_zero():
    r = ContentFingerprintRegistry()
    assert r.count() == 0


def test_count_distinct_fingerprints():
    r = ContentFingerprintRegistry()
    r.register("sha256-x", "0xalice")
    r.register("sha256-y", "0xbob")
    r.register("sha256-x", "0xspammer")  # not new
    assert r.count() == 2


def test_recent_newest_first():
    r = ContentFingerprintRegistry()
    r.register("sha256-x", "0xalice", timestamp=100.0)
    r.register("sha256-y", "0xbob", timestamp=200.0)
    recent = r.recent(limit=10)
    hashes = [e.content_hash for e in recent]
    assert hashes == ["sha256-y", "sha256-x"]


def test_recent_invalid_limit():
    r = ContentFingerprintRegistry()
    with pytest.raises(ValueError):
        r.recent(limit=0)
    with pytest.raises(ValueError):
        r.recent(limit=10001)


# ── Persistence ──────────────────────────────────────────


def test_persistence_round_trip(tmp_path):
    r1 = ContentFingerprintRegistry(persist_dir=tmp_path)
    r1.register("sha256-x", "0xalice", timestamp=100.0)
    r1.register("sha256-x", "0xspammer", timestamp=200.0)
    r2 = ContentFingerprintRegistry(persist_dir=tmp_path)
    assert r2.canonical_creator("sha256-x") == "0xalice"
    e = r2.get_entry("sha256-x")
    assert e.duplicate_attempt_count == 1


def test_persistence_corrupt_file_fail_soft(tmp_path):
    (tmp_path / "garbage.json").write_text("{not valid json")
    r = ContentFingerprintRegistry(persist_dir=tmp_path)
    assert r.count() == 0


# ── from_env ─────────────────────────────────────────────


def test_from_env_no_dir(monkeypatch):
    monkeypatch.delenv(
        "PRSM_FINGERPRINT_REGISTRY_DIR", raising=False,
    )
    r = ContentFingerprintRegistry.from_env()
    assert r._persist_dir is None


def test_from_env_with_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PRSM_FINGERPRINT_REGISTRY_DIR", str(tmp_path),
    )
    r = ContentFingerprintRegistry.from_env()
    assert r._persist_dir == tmp_path
