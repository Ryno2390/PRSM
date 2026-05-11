"""Sprint 269 — ContentFilterStore tests.

Operator-side runtime-editable blocklist store. Mirrors the
ReceiptStore + JobHistoryStore patterns (in-memory primary,
opt-in filesystem persistence). Covers CRUD, persistence,
malformed-input rejection, and the snapshot-to-filter
conversion.
"""
from __future__ import annotations

import json

import pytest

from prsm.node.content_filter_store import ContentFilterStore
from prsm.node.content_self_filter import (
    ContentSelfFilter, DispatchContext, FilterAction,
)


def test_empty_default():
    s = ContentFilterStore()
    snap = s.to_dict()
    assert snap["blocked_content_ids"] == []
    assert snap["blocked_model_tags"] == []
    assert snap["blocked_input_patterns"] == []
    assert snap["action_on_match"] == "refuse"
    assert s.is_cid_blocked("anything") is False


def test_add_cids_idempotent():
    s = ContentFilterStore()
    assert s.add_cids(["a", "b", "c"]) == 3
    assert s.add_cids(["b", "c", "d"]) == 1  # only "d" new
    assert s.to_dict()["count_cids"] == 4


def test_remove_cid_returns_state():
    s = ContentFilterStore()
    s.add_cids(["a"])
    assert s.remove_cid("a") is True
    assert s.remove_cid("a") is False  # already gone
    assert s.is_cid_blocked("a") is False


def test_add_tags_lowercases():
    s = ContentFilterStore()
    s.add_tags(["Safety-Flagged", "BIOWEAPON"])
    snap = s.to_dict()
    assert "safety-flagged" in snap["blocked_model_tags"]
    assert "bioweapon" in snap["blocked_model_tags"]


def test_add_patterns_validates_regex():
    s = ContentFilterStore()
    s.add_patterns([r"valid.*pattern"])
    with pytest.raises(ValueError):
        s.add_patterns([r"unclosed[bracket"])


def test_set_action_validates():
    s = ContentFilterStore()
    s.set_action("log_and_refuse")
    assert s.to_dict()["action_on_match"] == "log_and_refuse"
    with pytest.raises(ValueError):
        s.set_action("nuke")


def test_current_returns_immutable_filter():
    s = ContentFilterStore()
    s.add_cids(["bafy123"])
    f = s.current()
    assert isinstance(f, ContentSelfFilter)
    decision = f.evaluate(DispatchContext(content_id="bafy123"))
    assert decision.allow is False
    assert decision.reason == "blocked_content_id"
    # Adding more after snapshot doesn't mutate the snapshot
    s.add_cids(["bafy456"])
    decision2 = f.evaluate(DispatchContext(content_id="bafy456"))
    assert decision2.allow is True  # old snapshot didn't see new add


def test_persistence_round_trip(tmp_path):
    s1 = ContentFilterStore(persist_dir=tmp_path)
    s1.add_cids(["bafy-persisted"])
    s1.add_tags(["safety"])
    s1.add_patterns([r"^evil"])
    s1.set_action("log_and_refuse")

    # New instance reads same dir
    s2 = ContentFilterStore(persist_dir=tmp_path)
    snap = s2.to_dict()
    assert "bafy-persisted" in snap["blocked_content_ids"]
    assert "safety" in snap["blocked_model_tags"]
    assert "^evil" in snap["blocked_input_patterns"]
    assert snap["action_on_match"] == "log_and_refuse"


def test_disk_corruption_fail_soft(tmp_path):
    (tmp_path / "filter_state.json").write_text("{not valid json")
    s = ContentFilterStore(persist_dir=tmp_path)
    # Starts empty; doesn't raise
    assert s.to_dict()["count_cids"] == 0


def test_disk_wrong_shape_fail_soft(tmp_path):
    (tmp_path / "filter_state.json").write_text(
        json.dumps(["this", "is", "a", "list"])
    )
    s = ContentFilterStore(persist_dir=tmp_path)
    assert s.to_dict()["count_cids"] == 0


def test_disk_invalid_action_falls_back_to_refuse(tmp_path):
    (tmp_path / "filter_state.json").write_text(json.dumps({
        "blocked_content_ids": [],
        "blocked_model_tags": [],
        "blocked_input_patterns": [],
        "action_on_match": "explode",
    }))
    s = ContentFilterStore(persist_dir=tmp_path)
    # Action falls back to default
    assert s.to_dict()["action_on_match"] == "refuse"


def test_from_env_no_var(monkeypatch):
    monkeypatch.delenv("PRSM_CONTENT_FILTER_DIR", raising=False)
    s = ContentFilterStore.from_env()
    assert s._persist_dir is None


def test_from_env_with_var(tmp_path, monkeypatch):
    monkeypatch.setenv("PRSM_CONTENT_FILTER_DIR", str(tmp_path))
    s = ContentFilterStore.from_env()
    assert s._persist_dir == tmp_path


def test_add_cids_filters_empty_strings():
    s = ContentFilterStore()
    s.add_cids(["good", "", "  ", "another"])
    assert s.to_dict()["count_cids"] == 2


def test_non_string_cids_silently_filtered():
    s = ContentFilterStore()
    added = s.add_cids(["good", 12345, None, "another"])  # type: ignore
    assert added == 2
    assert "good" in s.to_dict()["blocked_content_ids"]
