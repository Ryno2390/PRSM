"""R9 Phase 6.4 — content self-filter (operator-side defensive filter).

Per R9-SCOPING-1 §7, an operator whose jurisdiction holds them legally
responsible for the content their node serves needs a mechanism to
refuse specific dispatch requests BEFORE running inference or returning
content. This module implements that filter.

Important framing (per R9 §7.3, §8)
-----------------------------------

**This is NOT a censorship mechanism.** The filter is local to a single
operator's node. It does not propagate to the network, does not remove
content from PRSM, and does not prevent other operators from serving
what this operator refuses. It is defensive compliance — a single
operator saying "I will not serve this" — not an enforcement primitive.

**The filter protects the operator, not the content.** An operator
uses the filter to reduce their personal legal exposure under their
local jurisdiction's content-liability rules. The content remains
available on the network through any other operator that does serve
it.

**No Foundation-shipped presets.** Per R9 §8, the Foundation does not
ship:
- Jurisdiction-specific preset filters ("Mode: China" configurations).
- Curated blocklists.
- Central content-classification services.
Every operator configures their own filter rules based on their own
compliance analysis + legal advice.

Filter rules
------------

The filter evaluates three independent rule axes. A dispatch is blocked
if ANY rule matches:

1. **``blocked_content_ids``** — explicit CID blocklist. Exact-match.
   Operator manages this via their config file or operator dashboard.
2. **``blocked_model_tags``** — tags from the ModelCatalog (if
   implemented; designed to integrate with PRSM's tagging system).
   Case-insensitive exact match against any tag on the dispatched model.
3. **``blocked_input_patterns``** — compiled regex patterns evaluated
   against the dispatch's prompt text. First match wins.

Action modes
------------

Operators choose how blocked dispatches surface observationally:

- ``FilterAction.REFUSE`` — default. Return a blocked decision; the
  orchestrator logs at warning level with the matched rule in the
  reason.
- ``FilterAction.LOG_AND_REFUSE`` — additionally emits a structured
  audit log entry (operator's compliance records).
- ``FilterAction.SILENT_REFUSE`` — return blocked without emitting
  any log entry. Use when logging the match itself would be a risk
  (e.g., if the log sink is in scope of compelled-disclosure).

Defaults to ``REFUSE``. Per R9 §8, operators explicitly choose the
action — the Foundation does not pick a default that implies an
opinion on their threat model.

Usage
-----

.. code-block:: python

    from prsm.node.content_self_filter import (
        ContentSelfFilter, DispatchContext, FilterAction,
    )
    import re

    filter = ContentSelfFilter(
        blocked_content_ids={"bafkreiabcdef...", "QmYwA..."},
        blocked_model_tags={"safety-flagged", "bioweapon-assist"},
        blocked_input_patterns=[
            re.compile(r"political-sensitive-pattern", re.IGNORECASE),
        ],
        action_on_match=FilterAction.LOG_AND_REFUSE,
    )

    ctx = DispatchContext(
        content_id="bafkreiabcdef...",
        model_tags=frozenset({"text-generation", "gpt-style"}),
        prompt_text="hello world",
    )
    decision = filter.evaluate(ctx)
    if not decision.allow:
        # Do not dispatch. Return a refusal to the caller.
        return

Integration point
-----------------

The orchestrator calling this filter should consult it BEFORE
accepting a dispatch request (before any compute cost is paid, before
any FTNS is escrowed, before any shard routing). The filter is a
pre-condition, not a post-hoc censorship step. See
``docs/2026-04-23-r9-transport-censorship-resistance-scoping.md`` §7
for the full intent.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Iterable, List, Optional, Pattern, Union

logger = logging.getLogger(__name__)


class FilterAction(Enum):
    """How a matching rule is surfaced observationally.

    - ``REFUSE``: refuse the dispatch; default-level logging in the
      caller's flow.
    - ``LOG_AND_REFUSE``: additionally emit a compliance audit log.
    - ``SILENT_REFUSE``: refuse silently; no log emission from the
      filter itself (caller may still log its own refusal).
    """

    REFUSE = "refuse"
    LOG_AND_REFUSE = "log_and_refuse"
    SILENT_REFUSE = "silent_refuse"


@dataclass(frozen=True)
class DispatchContext:
    """Inputs the filter evaluates.

    Kept minimal + immutable so the filter's evaluation is pure. The
    orchestrator populates this from its dispatch request before
    consulting the filter.

    :param content_id: CID of the content to be served (model weights,
        dataset shard, etc.). Empty string when not applicable (e.g., a
        pure inference dispatch with no content-ID to serve).
    :param model_tags: Frozen set of tags on the model being dispatched.
        Empty frozenset when no tags known.
    :param prompt_text: The text prompt or input payload for the
        dispatch. Filter's input-pattern rules run against this.
        Empty string when not applicable.
    """

    content_id: str = ""
    model_tags: FrozenSet[str] = field(default_factory=frozenset)
    prompt_text: str = ""


@dataclass(frozen=True)
class FilterDecision:
    """Result of a content-filter evaluation.

    :param allow: True if the dispatch is allowed, False if blocked.
    :param reason: Short identifier for the triggering rule, bounded
        vocabulary for telemetry labeling. Empty on allow.
    :param matched_value: The specific value that matched (CID, tag
        name, or regex pattern source). Empty on allow or for privacy-
        sensitive matches at ``SILENT_REFUSE``.
    :param action: Action the operator's config selected. ``REFUSE``
        for allowed decisions since no action was taken.
    """

    allow: bool
    reason: str = ""
    matched_value: str = ""
    action: FilterAction = FilterAction.REFUSE


# Reason-code vocabulary for telemetry labeling. Bounded so operators +
# dashboard code can enumerate all possible labels without the set
# growing unboundedly with each new user-configured rule.
_REASON_CONTENT_ID = "blocked_content_id"
_REASON_MODEL_TAG = "blocked_model_tag"
_REASON_INPUT_PATTERN = "blocked_input_pattern"
_REASON_ALLOWED = "allowed"


@dataclass(frozen=True)
class ContentSelfFilter:
    """Operator-side defensive content filter.

    All rule sets default to empty (the filter allows everything). A
    filter with all rules empty is a no-op — useful as a placeholder
    when the operator plans to configure rules later.

    :param blocked_content_ids: Iterable of CIDs to refuse serving.
        Stored as a frozenset for O(1) lookup.
    :param blocked_model_tags: Iterable of model tags to refuse. Stored
        case-normalized (lowercased) for case-insensitive matching.
    :param blocked_input_patterns: Iterable of compiled regex patterns
        to evaluate against the prompt text. Callers can pass raw
        strings in lieu of pre-compiled patterns; ``__post_init__``
        compiles them.
    :param action_on_match: Observability mode for matches.
    """

    blocked_content_ids: Union[FrozenSet[str], Iterable[str]] = field(
        default_factory=frozenset
    )
    blocked_model_tags: Union[FrozenSet[str], Iterable[str]] = field(
        default_factory=frozenset
    )
    blocked_input_patterns: List[Union[Pattern[str], str]] = field(
        default_factory=list
    )
    action_on_match: FilterAction = FilterAction.REFUSE

    def __post_init__(self) -> None:
        # Normalize + validate the rule sets. Frozen dataclass so we
        # use object.__setattr__ to mutate during __post_init__.
        object.__setattr__(
            self,
            "blocked_content_ids",
            frozenset(cid for cid in self.blocked_content_ids if cid),
        )
        object.__setattr__(
            self,
            "blocked_model_tags",
            frozenset(
                tag.strip().lower()
                for tag in self.blocked_model_tags
                if tag and tag.strip()
            ),
        )

        # Compile any raw-string patterns. Operators can pass either a
        # pre-compiled Pattern or a raw string; both are valid config
        # shapes. Invalid regexes raise at construction time, not at
        # evaluate time — fail fast on operator misconfiguration.
        compiled: List[Pattern[str]] = []
        for p in self.blocked_input_patterns:
            if isinstance(p, str):
                try:
                    compiled.append(re.compile(p))
                except re.error as exc:
                    raise ValueError(
                        f"invalid regex in blocked_input_patterns: {p!r}: {exc}"
                    ) from exc
            else:
                compiled.append(p)
        object.__setattr__(self, "blocked_input_patterns", compiled)

        if not isinstance(self.action_on_match, FilterAction):
            raise ValueError(
                f"action_on_match must be FilterAction; got {self.action_on_match!r}"
            )

    def evaluate(self, ctx: DispatchContext) -> FilterDecision:
        """Decide whether to allow the dispatch.

        Rules evaluate in order; first match wins. The order matches
        the stability ranking (explicit CIDs most specific; regex
        patterns most fuzzy):

        1. blocked_content_ids (exact CID match)
        2. blocked_model_tags (exact tag match, case-insensitive)
        3. blocked_input_patterns (regex match on prompt)

        :returns: ``FilterDecision`` with ``allow=False`` and the
            matching rule's reason+value on block, or ``allow=True``
            on pass.
        """
        # Rule 1: content-ID exact match.
        if ctx.content_id and ctx.content_id in self.blocked_content_ids:
            return self._block(_REASON_CONTENT_ID, ctx.content_id)

        # Rule 2: model-tag exact match (case-insensitive).
        if ctx.model_tags:
            normalized_tags = {
                t.strip().lower() for t in ctx.model_tags if t and t.strip()
            }
            match = normalized_tags & self.blocked_model_tags  # type: ignore[operator]
            if match:
                return self._block(_REASON_MODEL_TAG, sorted(match)[0])

        # Rule 3: regex pattern match on prompt.
        # __post_init__ normalizes the field to List[Pattern[str]] — any
        # raw-string inputs have already been compiled. Cast at use site.
        if ctx.prompt_text and self.blocked_input_patterns:
            from typing import cast as _cast
            for pattern in _cast(List[Pattern[str]], self.blocked_input_patterns):
                if pattern.search(ctx.prompt_text):
                    # For pattern matches: matched_value is the pattern's
                    # source, not the actual matched substring. This
                    # avoids leaking prompt content to log aggregators.
                    return self._block(_REASON_INPUT_PATTERN, pattern.pattern)

        return FilterDecision(
            allow=True, reason=_REASON_ALLOWED, action=FilterAction.REFUSE
        )

    def _block(self, reason: str, matched_value: str) -> FilterDecision:
        """Construct a block decision with the configured action mode."""
        action = self.action_on_match

        if action == FilterAction.LOG_AND_REFUSE:
            logger.warning(
                "content_self_filter: blocking dispatch; reason=%s matched_value=%s",
                reason, matched_value,
            )
        elif action == FilterAction.REFUSE:
            # Info-level log only — operator's orchestrator decides
            # whether to surface this higher.
            logger.info(
                "content_self_filter: blocking dispatch; reason=%s", reason,
            )
        # SILENT_REFUSE: emit nothing.

        # For SILENT_REFUSE, redact matched_value from the returned
        # decision so any downstream logger doesn't accidentally surface
        # it via the decision object.
        returned_matched = "" if action == FilterAction.SILENT_REFUSE else matched_value
        returned_reason = reason if action != FilterAction.SILENT_REFUSE else "blocked"

        return FilterDecision(
            allow=False,
            reason=returned_reason,
            matched_value=returned_matched,
            action=action,
        )


__all__ = [
    "ContentSelfFilter",
    "DispatchContext",
    "FilterAction",
    "FilterDecision",
]
