"""Unit tests for prsm.node.content_self_filter — R9 Phase 6.4."""
from __future__ import annotations

import logging
import re

import pytest

from prsm.node.content_self_filter import (
    ContentSelfFilter,
    DispatchContext,
    FilterAction,
    FilterDecision,
)


class TestDispatchContext:
    def test_defaults_are_empty(self):
        ctx = DispatchContext()
        assert ctx.content_id == ""
        assert ctx.model_tags == frozenset()
        assert ctx.prompt_text == ""

    def test_immutable(self):
        ctx = DispatchContext(content_id="cid1")
        with pytest.raises(Exception):
            ctx.content_id = "cid2"  # type: ignore[misc]


class TestFilterConstruction:
    def test_empty_filter_allows_everything(self):
        f = ContentSelfFilter()
        decision = f.evaluate(DispatchContext(
            content_id="anything",
            model_tags=frozenset({"any-tag"}),
            prompt_text="any prompt",
        ))
        assert decision.allow is True
        assert decision.reason == "allowed"

    def test_content_ids_empty_strings_filtered_out(self):
        """Empty strings in blocklists are no-ops — skipped at construction."""
        f = ContentSelfFilter(blocked_content_ids=["", "real-cid", ""])
        assert f.blocked_content_ids == frozenset({"real-cid"})

    def test_tags_normalized_to_lowercase(self):
        f = ContentSelfFilter(blocked_model_tags=["SAFETY", " FlagGed ", "low"])
        assert f.blocked_model_tags == frozenset({"safety", "flagged", "low"})

    def test_tags_empty_filtered(self):
        f = ContentSelfFilter(blocked_model_tags=["real", "", "  ", "also"])
        assert f.blocked_model_tags == frozenset({"real", "also"})

    def test_string_patterns_compiled(self):
        f = ContentSelfFilter(blocked_input_patterns=[r"foo.*bar", r"baz"])
        assert len(f.blocked_input_patterns) == 2
        assert all(hasattr(p, "search") for p in f.blocked_input_patterns)

    def test_precompiled_patterns_preserved(self):
        original = re.compile(r"pre-compiled", re.IGNORECASE)
        f = ContentSelfFilter(blocked_input_patterns=[original])
        assert f.blocked_input_patterns[0] is original

    def test_mixed_string_and_compiled_patterns(self):
        compiled = re.compile(r"compiled")
        f = ContentSelfFilter(blocked_input_patterns=[r"raw-string", compiled])
        assert len(f.blocked_input_patterns) == 2

    def test_invalid_regex_raises_at_construction(self):
        with pytest.raises(ValueError) as excinfo:
            ContentSelfFilter(blocked_input_patterns=[r"[unclosed"])
        assert "invalid regex" in str(excinfo.value)

    def test_invalid_action_rejected(self):
        with pytest.raises(ValueError):
            ContentSelfFilter(action_on_match="not-an-action-enum")  # type: ignore[arg-type]

    def test_default_action_is_refuse(self):
        f = ContentSelfFilter()
        assert f.action_on_match == FilterAction.REFUSE


class TestContentIdRule:
    def test_blocks_matching_cid(self):
        f = ContentSelfFilter(blocked_content_ids={"bafkreiabc"})
        decision = f.evaluate(DispatchContext(content_id="bafkreiabc"))
        assert decision.allow is False
        assert decision.reason == "blocked_content_id"
        assert decision.matched_value == "bafkreiabc"

    def test_allows_nonmatching_cid(self):
        f = ContentSelfFilter(blocked_content_ids={"bafkreiabc"})
        decision = f.evaluate(DispatchContext(content_id="bafkreizzz"))
        assert decision.allow is True

    def test_empty_cid_not_spuriously_blocked(self):
        """Empty content_id must not match an empty-string CID in the
        blocklist (we also filter empties at construction time, but
        this covers the evaluator's own behavior)."""
        f = ContentSelfFilter(blocked_content_ids={"real-cid"})
        decision = f.evaluate(DispatchContext(content_id=""))
        assert decision.allow is True

    def test_cid_match_is_exact(self):
        """CID matching is exact string comparison; substring matches
        must NOT trigger (would be a security risk)."""
        f = ContentSelfFilter(blocked_content_ids={"bafkrei-abc"})
        decision = f.evaluate(DispatchContext(content_id="bafkrei-abcdef"))
        assert decision.allow is True


class TestModelTagRule:
    def test_blocks_matching_tag(self):
        f = ContentSelfFilter(blocked_model_tags={"safety-flagged"})
        decision = f.evaluate(DispatchContext(
            model_tags=frozenset({"safety-flagged", "text-gen"})
        ))
        assert decision.allow is False
        assert decision.reason == "blocked_model_tag"
        assert decision.matched_value == "safety-flagged"

    def test_case_insensitive_tag_match(self):
        f = ContentSelfFilter(blocked_model_tags={"SAFETY"})
        decision = f.evaluate(DispatchContext(
            model_tags=frozenset({"Safety", "text-gen"})
        ))
        assert decision.allow is False
        assert decision.matched_value == "safety"

    def test_no_tags_allows(self):
        f = ContentSelfFilter(blocked_model_tags={"safety"})
        decision = f.evaluate(DispatchContext(model_tags=frozenset()))
        assert decision.allow is True

    def test_nonmatching_tags_allow(self):
        f = ContentSelfFilter(blocked_model_tags={"safety"})
        decision = f.evaluate(DispatchContext(
            model_tags=frozenset({"text-gen", "image-gen"})
        ))
        assert decision.allow is True

    def test_multiple_tags_match_first_wins(self):
        """When a dispatch's tags include multiple blocked ones, report
        the alphabetically-first match (deterministic ordering)."""
        f = ContentSelfFilter(blocked_model_tags={"alpha", "beta"})
        decision = f.evaluate(DispatchContext(
            model_tags=frozenset({"alpha", "beta", "gamma"})
        ))
        assert decision.matched_value == "alpha"


class TestInputPatternRule:
    def test_blocks_matching_pattern(self):
        f = ContentSelfFilter(blocked_input_patterns=[r"sensitive"])
        decision = f.evaluate(DispatchContext(prompt_text="This is a sensitive prompt"))
        assert decision.allow is False
        assert decision.reason == "blocked_input_pattern"
        assert decision.matched_value == "sensitive"

    def test_case_insensitive_via_flag(self):
        pattern = re.compile(r"sensitive", re.IGNORECASE)
        f = ContentSelfFilter(blocked_input_patterns=[pattern])
        decision = f.evaluate(DispatchContext(prompt_text="SENSITIVE content"))
        assert decision.allow is False

    def test_empty_prompt_allowed(self):
        f = ContentSelfFilter(blocked_input_patterns=[r"anything"])
        decision = f.evaluate(DispatchContext(prompt_text=""))
        assert decision.allow is True

    def test_first_matching_pattern_wins(self):
        f = ContentSelfFilter(
            blocked_input_patterns=[r"first", r"second"]
        )
        decision = f.evaluate(DispatchContext(prompt_text="first and second"))
        assert decision.matched_value == "first"

    def test_matched_value_is_pattern_source_not_substring(self):
        """Privacy: don't leak the actual matched substring from prompt
        text into the decision (it might contain the operator-restricted
        content itself)."""
        f = ContentSelfFilter(blocked_input_patterns=[r"topic-[a-z]+"])
        decision = f.evaluate(DispatchContext(prompt_text="I need help with topic-xyzabc"))
        assert decision.matched_value == "topic-[a-z]+"
        # The actual "topic-xyzabc" string must NOT leak into matched_value.
        assert "xyzabc" not in decision.matched_value


class TestRuleOrdering:
    """Rules evaluate in order: content-ID → tag → pattern. First match
    wins. Test that a CID match short-circuits before tag/pattern
    evaluation."""

    def test_cid_wins_over_tag(self):
        f = ContentSelfFilter(
            blocked_content_ids={"cid-1"},
            blocked_model_tags={"bad-tag"},
        )
        ctx = DispatchContext(
            content_id="cid-1",
            model_tags=frozenset({"bad-tag"}),
        )
        decision = f.evaluate(ctx)
        assert decision.reason == "blocked_content_id"

    def test_tag_wins_over_pattern(self):
        f = ContentSelfFilter(
            blocked_model_tags={"bad-tag"},
            blocked_input_patterns=[r"bad-phrase"],
        )
        ctx = DispatchContext(
            model_tags=frozenset({"bad-tag"}),
            prompt_text="this is a bad-phrase",
        )
        decision = f.evaluate(ctx)
        assert decision.reason == "blocked_model_tag"


class TestActionModes:
    def test_refuse_logs_at_info(self, caplog):
        f = ContentSelfFilter(
            blocked_content_ids={"cid-1"},
            action_on_match=FilterAction.REFUSE,
        )
        with caplog.at_level(logging.INFO, logger="prsm.node.content_self_filter"):
            f.evaluate(DispatchContext(content_id="cid-1"))
        assert any("blocking dispatch" in r.message for r in caplog.records)
        # INFO level — not WARNING
        assert any(r.levelno == logging.INFO for r in caplog.records)

    def test_log_and_refuse_logs_at_warning(self, caplog):
        f = ContentSelfFilter(
            blocked_content_ids={"cid-1"},
            action_on_match=FilterAction.LOG_AND_REFUSE,
        )
        with caplog.at_level(logging.WARNING, logger="prsm.node.content_self_filter"):
            f.evaluate(DispatchContext(content_id="cid-1"))
        # WARNING level + includes matched_value
        assert any(
            "blocking dispatch" in r.message and "cid-1" in r.message
            for r in caplog.records
        )

    def test_silent_refuse_emits_no_log(self, caplog):
        f = ContentSelfFilter(
            blocked_content_ids={"cid-1"},
            action_on_match=FilterAction.SILENT_REFUSE,
        )
        with caplog.at_level(logging.INFO, logger="prsm.node.content_self_filter"):
            decision = f.evaluate(DispatchContext(content_id="cid-1"))
        # No log records emitted at all.
        assert not caplog.records
        # Decision still records blocked — caller knows but log sink
        # doesn't.
        assert decision.allow is False

    def test_silent_refuse_redacts_matched_value(self):
        """Silent mode must not leak matched_value into the returned
        decision — caller may pass the decision to a logger that is
        not in the operator's trust boundary."""
        f = ContentSelfFilter(
            blocked_content_ids={"sensitive-cid"},
            action_on_match=FilterAction.SILENT_REFUSE,
        )
        decision = f.evaluate(DispatchContext(content_id="sensitive-cid"))
        assert decision.allow is False
        assert decision.matched_value == ""
        assert decision.reason == "blocked"  # Generic; not the specific rule
        assert decision.action == FilterAction.SILENT_REFUSE

    def test_allow_decision_has_refuse_action_sentinel(self):
        """For allowed decisions, action field defaults to REFUSE as a
        sentinel — no action was taken, but the field is typed
        consistently."""
        f = ContentSelfFilter()
        decision = f.evaluate(DispatchContext(content_id="anything"))
        assert decision.allow is True
        assert decision.action == FilterAction.REFUSE  # sentinel, not applied


class TestAllThreeRulesInteract:
    """End-to-end smoke tests with realistic-looking config."""

    def test_realistic_config_blocks_cid(self):
        f = ContentSelfFilter(
            blocked_content_ids={
                "bafkreiabcdefghijk",
                "QmXYZ...123",
            },
            blocked_model_tags={"safety-flagged", "bioweapon-assist"},
            blocked_input_patterns=[
                re.compile(r"politically-sensitive-keyword", re.IGNORECASE),
            ],
            action_on_match=FilterAction.LOG_AND_REFUSE,
        )
        ctx = DispatchContext(
            content_id="bafkreiabcdefghijk",
            model_tags=frozenset({"text-generation"}),
            prompt_text="benign prompt",
        )
        decision = f.evaluate(ctx)
        assert decision.allow is False
        assert decision.reason == "blocked_content_id"

    def test_realistic_config_allows_benign_dispatch(self):
        f = ContentSelfFilter(
            blocked_content_ids={"bafkreiabc"},
            blocked_model_tags={"safety-flagged"},
            blocked_input_patterns=[r"sensitive-keyword"],
        )
        ctx = DispatchContext(
            content_id="bafkreixyz",
            model_tags=frozenset({"text-generation", "gpt-style"}),
            prompt_text="Please explain quantum mechanics.",
        )
        decision = f.evaluate(ctx)
        assert decision.allow is True

    def test_no_op_filter_is_truly_a_no_op(self):
        """A filter with no rules must never block, regardless of
        dispatch content."""
        f = ContentSelfFilter()
        for ctx in [
            DispatchContext(content_id="any"),
            DispatchContext(model_tags=frozenset({"any", "tag"})),
            DispatchContext(prompt_text="any content"),
            DispatchContext(
                content_id="a", model_tags=frozenset({"b"}), prompt_text="c"
            ),
        ]:
            assert f.evaluate(ctx).allow is True
