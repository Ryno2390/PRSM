"""
NWTN Interview Mode
===================

Gathers structured project requirements from the user through a focused,
multi-round conversation before assembling the Agent Team.

The interview is UI-agnostic: callers supply a ``QuestionCallback`` —
an async function that presents a question string to the user and returns
their answer string.  This lets the same interview logic work in a CLI,
a web endpoint, a Slack bot, or an automated test fixture.

Flow
----
1. User supplies an initial goal prompt.
2. NWTN (via BackendRegistry) identifies information gaps and generates
   targeted questions (up to ``max_rounds``).
3. Each question–answer pair is recorded.
4. NWTN synthesises a structured ``ProjectBrief`` from the collected QA.

Fallback
--------
If no LLM backend is available (no API keys, offline), the interviewer
falls back to a fixed set of template questions covering the five essential
areas: goal clarity, technology stack, constraints, success criteria, and
timeline.  This ensures the interview always produces a usable
``ProjectBrief`` even without a live model.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Type alias: async (question: str) -> answer: str
QuestionCallback = Callable[[str], Awaitable[str]]


# ======================================================================
# Data models
# ======================================================================

@dataclass
class QAPair:
    """A single question–answer exchange during the interview."""
    question: str
    answer: str
    category: str = "general"


@dataclass
class ProjectBrief:
    """
    Structured project requirements gathered by the interview.

    This is the input to the MetaPlanner (``planner.py``).
    """
    session_id: str
    goal: str
    """The core objective in the user's own words."""

    technology_stack: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    existing_codebase: Optional[str] = None
    timeline: Optional[str] = None
    preferred_roles: List[str] = field(default_factory=list)
    """Agent role preferences stated by the user (e.g. 'security auditor')."""

    raw_qa: List[QAPair] = field(default_factory=list)
    """Complete interview transcript."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    llm_assisted: bool = True
    """False if the fallback template path was used."""

    def to_prompt_block(self) -> str:
        """
        Render the brief as a readable block for the MetaPlanner prompt.
        """
        lines = [
            f"PROJECT BRIEF",
            f"=============",
            f"Goal: {self.goal}",
        ]
        if self.technology_stack:
            lines.append(f"Technology stack: {', '.join(self.technology_stack)}")
        if self.constraints:
            lines.append("Constraints:")
            for c in self.constraints:
                lines.append(f"  - {c}")
        if self.success_criteria:
            lines.append("Success criteria:")
            for s in self.success_criteria:
                lines.append(f"  - {s}")
        if self.existing_codebase:
            lines.append(f"Existing codebase: {self.existing_codebase}")
        if self.timeline:
            lines.append(f"Timeline: {self.timeline}")
        if self.preferred_roles:
            lines.append(f"Preferred agent roles: {', '.join(self.preferred_roles)}")
        return "\n".join(lines)


# ======================================================================
# Template fallback questions
# ======================================================================

_TEMPLATE_QUESTIONS: List[Dict[str, str]] = [
    {
        "category": "goal",
        "question": (
            "Can you describe your project goal in more detail? What specific "
            "outcome or deliverable are you aiming for?"
        ),
    },
    {
        "category": "technology_stack",
        "question": (
            "What programming languages, frameworks, or technologies are involved? "
            "(e.g. Python / FastAPI, TypeScript / React, Rust, etc.)"
        ),
    },
    {
        "category": "constraints",
        "question": (
            "Are there any constraints or hard requirements? "
            "(e.g. must run on Apple Silicon, no external APIs, specific library versions)"
        ),
    },
    {
        "category": "success_criteria",
        "question": (
            "How will you know the project is complete? "
            "What are the 2–3 most important success criteria?"
        ),
    },
    {
        "category": "timeline",
        "question": (
            "What is your target timeline or deadline, if any?"
        ),
    },
]


# ======================================================================
# InterviewSession
# ======================================================================

class InterviewSession:
    """
    Conducts a structured requirements-gathering interview.

    Parameters
    ----------
    ask : QuestionCallback
        Async callable ``(question: str) -> answer: str``.  In production
        this drives a CLI prompt or web form; in tests it returns predefined
        answers.
    backend_registry : optional
        A ``BackendRegistry`` instance for LLM-powered question generation.
        If ``None`` or if all backends fail, the template fallback is used.
    max_rounds : int
        Maximum number of clarifying-question rounds (default: 5).
    session_id : str
        Optional session identifier; auto-generated if omitted.
    """

    def __init__(
        self,
        ask: QuestionCallback,
        backend_registry=None,
        max_rounds: int = 5,
        session_id: Optional[str] = None,
    ) -> None:
        self._ask = ask
        self._backend = backend_registry
        self._max_rounds = max_rounds
        self._session_id = session_id or str(uuid4())

    async def run(self, initial_prompt: str) -> ProjectBrief:
        """
        Run the full interview and return a ``ProjectBrief``.

        Parameters
        ----------
        initial_prompt : str
            The user's opening statement of their goal.

        Returns
        -------
        ProjectBrief
        """
        logger.info("InterviewSession starting: session=%s", self._session_id)
        qa_pairs: List[QAPair] = []
        llm_assisted = False

        if self._backend is not None:
            try:
                qa_pairs, llm_assisted = await self._llm_interview(
                    initial_prompt, qa_pairs
                )
            except Exception as exc:
                logger.warning(
                    "LLM interview failed (%s); falling back to template questions",
                    exc,
                )

        if not llm_assisted:
            qa_pairs = await self._template_interview(initial_prompt)

        brief = await self._synthesise(
            initial_prompt, qa_pairs, llm_assisted=llm_assisted
        )
        logger.info(
            "InterviewSession complete: session=%s, rounds=%d, llm=%s",
            self._session_id, len(qa_pairs), llm_assisted,
        )
        return brief

    # ------------------------------------------------------------------
    # LLM-driven interview
    # ------------------------------------------------------------------

    async def _llm_interview(
        self, initial_prompt: str, qa_pairs: List[QAPair]
    ) -> tuple[List[QAPair], bool]:
        """Generate and ask targeted questions using the LLM backend."""
        gap_prompt = (
            "You are NWTN, a project coordinator. A user has described a project.\n\n"
            f"User's goal: {initial_prompt}\n\n"
            f"Identify up to {self._max_rounds} specific information gaps that would "
            "help you build a detailed project plan. Return ONLY a JSON array of "
            "objects with keys 'category' and 'question'. Example:\n"
            '[{"category": "constraints", "question": "Are there any ...?"}]\n\n'
            "Focus on: goal clarity, technology stack, constraints, success criteria, "
            "timeline, and required specialist roles. Do not ask about things already "
            "clear from the goal statement."
        )

        result = await self._backend.generate(
            prompt=gap_prompt,
            max_tokens=600,
            temperature=0.3,
        )
        raw = result.text.strip()

        # Extract JSON array from the response
        questions = _extract_json_array(raw)
        if not questions:
            raise ValueError("LLM returned no parseable questions")

        for q_obj in questions[: self._max_rounds]:
            question = q_obj.get("question", "").strip()
            category = q_obj.get("category", "general")
            if not question:
                continue
            answer = await self._ask(question)
            qa_pairs.append(QAPair(question=question, answer=answer, category=category))

        return qa_pairs, True

    # ------------------------------------------------------------------
    # Template fallback interview
    # ------------------------------------------------------------------

    async def _template_interview(self, initial_prompt: str) -> List[QAPair]:
        """Ask the fixed set of template questions."""
        qa_pairs: List[QAPair] = []
        for tq in _TEMPLATE_QUESTIONS[: self._max_rounds]:
            answer = await self._ask(tq["question"])
            # Skip if the user explicitly skips (empty / "n/a" / "skip")
            if answer.strip().lower() in ("", "n/a", "na", "skip", "none"):
                continue
            qa_pairs.append(
                QAPair(
                    question=tq["question"],
                    answer=answer,
                    category=tq["category"],
                )
            )
        return qa_pairs

    # ------------------------------------------------------------------
    # ProjectBrief synthesis
    # ------------------------------------------------------------------

    async def _synthesise(
        self,
        initial_prompt: str,
        qa_pairs: List[QAPair],
        *,
        llm_assisted: bool,
    ) -> ProjectBrief:
        """
        Build a ``ProjectBrief`` from the collected QA pairs.

        Tries the LLM-synthesis path first; falls back to rule-based extraction
        if the LLM is unavailable or returns unparseable output.
        """
        if self._backend is not None and llm_assisted:
            try:
                return await self._llm_synthesise(initial_prompt, qa_pairs)
            except Exception as exc:
                logger.warning("LLM synthesis failed (%s); using rule-based extraction", exc)

        return self._rule_based_synthesise(initial_prompt, qa_pairs)

    async def _llm_synthesise(
        self, initial_prompt: str, qa_pairs: List[QAPair]
    ) -> ProjectBrief:
        qa_block = "\n".join(
            f"Q ({p.category}): {p.question}\nA: {p.answer}" for p in qa_pairs
        )
        synth_prompt = (
            "You are NWTN. Synthesise the following project interview into a structured "
            "JSON ProjectBrief.\n\n"
            f"Initial goal: {initial_prompt}\n\n"
            f"Interview transcript:\n{qa_block}\n\n"
            "Return ONLY a JSON object with these keys:\n"
            "  goal (str), technology_stack (list), constraints (list),\n"
            "  success_criteria (list), existing_codebase (str or null),\n"
            "  timeline (str or null), preferred_roles (list)\n"
        )
        result = await self._backend.generate(
            prompt=synth_prompt,
            max_tokens=800,
            temperature=0.2,
        )
        data = _extract_json_object(result.text.strip())

        return ProjectBrief(
            session_id=self._session_id,
            goal=data.get("goal", initial_prompt),
            technology_stack=_as_list(data.get("technology_stack")),
            constraints=_as_list(data.get("constraints")),
            success_criteria=_as_list(data.get("success_criteria")),
            existing_codebase=data.get("existing_codebase"),
            timeline=data.get("timeline"),
            preferred_roles=_as_list(data.get("preferred_roles")),
            raw_qa=qa_pairs,
            llm_assisted=True,
        )

    def _rule_based_synthesise(
        self, initial_prompt: str, qa_pairs: List[QAPair]
    ) -> ProjectBrief:
        """Extract fields from QA pairs by category."""
        by_cat: Dict[str, List[str]] = {}
        for pair in qa_pairs:
            by_cat.setdefault(pair.category, []).append(pair.answer)

        return ProjectBrief(
            session_id=self._session_id,
            goal=initial_prompt,
            technology_stack=_split_answers(by_cat.get("technology_stack", [])),
            constraints=_split_answers(by_cat.get("constraints", [])),
            success_criteria=_split_answers(by_cat.get("success_criteria", [])),
            existing_codebase=by_cat.get("existing_codebase", [None])[0],
            timeline=by_cat.get("timeline", [None])[0],
            preferred_roles=_split_answers(by_cat.get("preferred_roles", [])),
            raw_qa=qa_pairs,
            llm_assisted=False,
        )


# ======================================================================
# Helpers
# ======================================================================

def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Extract the first JSON array from *text*."""
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from *text*."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _split_answers(answers: List[str]) -> List[str]:
    """Split comma-separated or newline-separated answers into a flat list."""
    items: List[str] = []
    for ans in answers:
        for part in re.split(r"[,\n;]+", ans):
            part = part.strip(" -•*")
            if part:
                items.append(part)
    return items
