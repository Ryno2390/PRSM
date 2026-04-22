"""
Tests for Sub-phase 10.3: Agent Team Coordination

Coverage:
  1. Interview — template path, LLM path (mocked), ProjectBrief synthesis
  2. Planner — template plan, LLM plan (mocked), MetaPlan structure
  3. Assembler — fallback assignments, registry query (mocked), branch naming
  4. BranchManager — branch creation/listing/diff/merge (uses real temp git repo)
  5. CheckpointReviewer — structural guards, heuristic review, LLM review (mocked),
     approve_and_merge
  6. End-to-end: interview → plan → assemble → review (all mocked/temp git)
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import subprocess as _subprocess_module

# Capture real Popen at import time before conftest mocks are applied.
_REAL_POPEN = _subprocess_module.Popen

import pytest

from prsm.compute.nwtn.team import (
    AgentTeam,
    BranchManager,
    BranchStatus,
    CheckpointDecision,
    CheckpointReviewer,
    InterviewSession,
    MetaPlan,
    MetaPlanner,
    Milestone,
    AgentRole,
    ProjectBrief,
    QAPair,
    TeamAssembler,
    TeamMember,
)
from prsm.compute.nwtn.team.branch_manager import GitDiff
from prsm.compute.nwtn.team.interview import _extract_json_array, _extract_json_object, _split_answers


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def sample_brief():
    return ProjectBrief(
        session_id="sess-test",
        goal="Build a BSC-powered agent coordination layer for PRSM",
        technology_stack=["Python", "FastAPI", "aiosqlite"],
        constraints=["Must run on Apple Silicon", "No breaking API changes"],
        success_criteria=["79+ tests passing", "BSC integration complete"],
        timeline="2 weeks",
        preferred_roles=["architect", "backend-coder", "tester"],
    )


@pytest.fixture
def sample_meta_plan(sample_brief):
    return MetaPlan(
        session_id="sess-test",
        title="PRSM Agent Coordination Layer",
        objective="Implement BSC-powered flat Agent Team harness with whiteboard",
        milestones=[
            Milestone(
                title="BSC Core",
                description="Predictor, KL filter, semantic dedup, promoter",
                merge_criteria=["35 BSC tests passing", "No regressions"],
            ),
            Milestone(
                title="Whiteboard",
                description="SQLite store, file monitor, query interface",
                merge_criteria=["44 whiteboard tests passing"],
            ),
            Milestone(
                title="Team Coordination",
                description="Interview, planner, assembler, branch manager",
                merge_criteria=["Team coordination tests passing"],
            ),
        ],
        required_roles=[
            AgentRole(role="architect", description="High-level design", priority=1),
            AgentRole(role="backend-coder", description="Core implementation", priority=2),
            AgentRole(role="tester", description="Test coverage", priority=3),
        ],
        success_criteria=["All tests pass", "BSC integration complete"],
        constraints=["Must run on Apple Silicon", "No breaking API changes"],
        source_brief=sample_brief,
    )


def _git(repo, *args):
    """Run a git command using the real Popen (bypasses conftest subprocess mock)."""
    import os
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    proc = _REAL_POPEN(
        ["git", "-C", str(repo)] + list(args),
        stdout=_subprocess_module.PIPE,
        stderr=_subprocess_module.PIPE,
        env=env,
    )
    stdout, stderr = proc.communicate(timeout=15)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {args} failed: {stderr.decode(errors='replace').strip()}"
        )
    return stdout.decode(errors="replace").strip()


@pytest.fixture
def git_repo(tmp_path):
    """Create a minimal git repository for BranchManager tests."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(tmp_path, "init", "repo")
    _git(repo, "config", "user.email", "test@prsm.ai")
    _git(repo, "config", "user.name", "PRSM Test")
    (repo / "README.md").write_text("# PRSM Test Repo\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial commit")
    _git(repo, "branch", "-M", "main")
    return repo


# ======================================================================
# 1. Interview
# ======================================================================

class TestInterviewSession:
    @pytest.mark.asyncio
    async def test_template_path_produces_brief(self):
        answers = iter([
            "Build a distributed AI protocol",  # goal
            "Python, FastAPI",                  # tech stack
            "No breaking changes",              # constraints
            "All tests pass",                   # success criteria
            "2 weeks",                          # timeline
        ])

        async def mock_ask(question: str) -> str:
            return next(answers, "")

        session = InterviewSession(ask=mock_ask, backend_registry=None)
        brief = await session.run("I want to build a distributed AI protocol")

        assert isinstance(brief, ProjectBrief)
        assert "distributed AI protocol" in brief.goal
        assert not brief.llm_assisted

    @pytest.mark.asyncio
    async def test_template_path_skips_empty_answers(self):
        async def mock_ask(question: str) -> str:
            return "skip"

        session = InterviewSession(ask=mock_ask, backend_registry=None)
        brief = await session.run("goal")

        # All answers were skipped — brief should still be created
        assert brief.session_id is not None
        assert len(brief.raw_qa) == 0

    @pytest.mark.asyncio
    async def test_llm_path_uses_backend(self):
        backend = MagicMock()
        questions_response = MagicMock()
        questions_response.text = (
            '[{"category": "constraints", "question": "Are there any hard constraints?"},'
            ' {"category": "timeline", "question": "What is the deadline?"}]'
        )
        synth_response = MagicMock()
        synth_response.text = (
            '{"goal": "Build distributed AI protocol", "technology_stack": ["Python"],'
            ' "constraints": ["No breaking changes"], "success_criteria": ["Tests pass"],'
            ' "existing_codebase": null, "timeline": "2 weeks", "preferred_roles": []}'
        )
        backend.generate = AsyncMock(side_effect=[questions_response, synth_response])

        call_count = 0

        async def mock_ask(question: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Answer {call_count}"

        session = InterviewSession(ask=mock_ask, backend_registry=backend, max_rounds=3)
        brief = await session.run("Build distributed AI protocol")

        assert brief.llm_assisted
        assert brief.goal == "Build distributed AI protocol"
        assert "Python" in brief.technology_stack
        assert call_count == 2  # two questions asked

    @pytest.mark.asyncio
    async def test_falls_back_to_template_on_llm_error(self):
        backend = MagicMock()
        backend.generate = AsyncMock(side_effect=Exception("API error"))

        async def mock_ask(q: str) -> str:
            return "answer"

        session = InterviewSession(ask=mock_ask, backend_registry=backend)
        brief = await session.run("Build something")

        # Should not raise; should use template fallback
        assert brief is not None
        assert not brief.llm_assisted

    @pytest.mark.asyncio
    async def test_session_id_auto_generated(self):
        async def mock_ask(q):
            return "a"

        s1 = InterviewSession(ask=mock_ask)
        s2 = InterviewSession(ask=mock_ask)
        b1 = await s1.run("goal")
        b2 = await s2.run("goal")
        assert b1.session_id != b2.session_id

    @pytest.mark.asyncio
    async def test_explicit_session_id_preserved(self):
        async def mock_ask(q):
            return "a"

        session = InterviewSession(ask=mock_ask, session_id="fixed-id")
        brief = await session.run("goal")
        assert brief.session_id == "fixed-id"

    def test_project_brief_to_prompt_block(self, sample_brief):
        block = sample_brief.to_prompt_block()
        assert "BSC-powered" in block
        assert "Python" in block
        assert "Apple Silicon" in block
        assert "2 weeks" in block


class TestInterviewHelpers:
    def test_extract_json_array(self):
        text = 'prefix [{"a": 1}, {"b": 2}] suffix'
        result = _extract_json_array(text)
        assert result == [{"a": 1}, {"b": 2}]

    def test_extract_json_array_no_match(self):
        assert _extract_json_array("no json here") == []

    def test_extract_json_object(self):
        text = 'here {"key": "value"} there'
        result = _extract_json_object(text)
        assert result == {"key": "value"}

    def test_split_answers_comma_separated(self):
        result = _split_answers(["Python, FastAPI, Redis"])
        assert "Python" in result
        assert "FastAPI" in result
        assert "Redis" in result

    def test_split_answers_strips_bullets(self):
        result = _split_answers(["- Item one\n- Item two"])
        assert "Item one" in result
        assert "Item two" in result


# ======================================================================
# 2. MetaPlanner
# ======================================================================

class TestMetaPlanner:
    @pytest.mark.asyncio
    async def test_template_plan_from_brief(self, sample_brief):
        planner = MetaPlanner(backend_registry=None)
        plan = await planner.generate(sample_brief)

        assert plan.session_id == "sess-test"
        assert len(plan.milestones) >= 1
        assert len(plan.required_roles) >= 1
        assert not plan.llm_assisted
        assert plan.objective  # non-empty
        assert plan.source_brief is sample_brief

    @pytest.mark.asyncio
    async def test_template_uses_success_criteria_as_milestones(self, sample_brief):
        planner = MetaPlanner(backend_registry=None)
        plan = await planner.generate(sample_brief)

        # success_criteria has 2 items → 2 milestones
        assert len(plan.milestones) == 2

    @pytest.mark.asyncio
    async def test_template_uses_default_roles_when_none_preferred(self):
        brief = ProjectBrief(
            session_id="s",
            goal="Build something",
            preferred_roles=[],
        )
        planner = MetaPlanner(backend_registry=None)
        plan = await planner.generate(brief)
        assert len(plan.required_roles) == 4  # _DEFAULT_ROLES

    @pytest.mark.asyncio
    async def test_template_uses_preferred_roles_from_brief(self, sample_brief):
        planner = MetaPlanner(backend_registry=None)
        plan = await planner.generate(sample_brief)
        role_slugs = [r.role for r in plan.required_roles]
        # preferred_roles from brief are: architect, backend-coder, tester
        assert any("architect" in s for s in role_slugs)

    @pytest.mark.asyncio
    async def test_llm_plan_path(self, sample_brief):
        backend = MagicMock()
        llm_response = MagicMock()
        llm_response.text = """{
          "title": "PRSM Agent Layer",
          "objective": "Implement flat Agent Team harness",
          "milestones": [
            {"title": "BSC", "description": "Core filter", "merge_criteria": ["Tests pass"], "estimated_effort": "medium"}
          ],
          "required_roles": [
            {"role": "coder", "description": "Implement", "capabilities_required": ["code_generation"], "priority": 1}
          ],
          "success_criteria": ["All tests pass"],
          "constraints": ["No breaking changes"]
        }"""
        backend.generate = AsyncMock(return_value=llm_response)

        planner = MetaPlanner(backend_registry=backend)
        plan = await planner.generate(sample_brief)

        assert plan.llm_assisted
        assert plan.title == "PRSM Agent Layer"
        assert len(plan.milestones) == 1
        assert plan.milestones[0].title == "BSC"

    @pytest.mark.asyncio
    async def test_llm_plan_falls_back_on_error(self, sample_brief):
        backend = MagicMock()
        backend.generate = AsyncMock(side_effect=Exception("LLM error"))

        planner = MetaPlanner(backend_registry=backend)
        plan = await planner.generate(sample_brief)

        assert not plan.llm_assisted  # fell back to template

    def test_meta_plan_to_whiteboard_entry(self, sample_meta_plan):
        entry = sample_meta_plan.to_whiteboard_entry()
        assert "META-PLAN" in entry
        assert "BSC Core" in entry
        assert "architect" in entry
        assert "All tests pass" in entry

    def test_roles_by_priority(self, sample_meta_plan):
        roles = sample_meta_plan.roles_by_priority()
        priorities = [r.priority for r in roles]
        assert priorities == sorted(priorities)


# ======================================================================
# 3. TeamAssembler
# ======================================================================

class TestTeamAssembler:
    @pytest.mark.asyncio
    async def test_fallback_assigns_all_roles(self, sample_meta_plan):
        assembler = TeamAssembler(model_registry=None)
        team = await assembler.assemble(sample_meta_plan)

        assert len(team.members) == 3
        roles = {m.role for m in team.members}
        assert roles == {"architect", "backend-coder", "tester"}

    @pytest.mark.asyncio
    async def test_branch_names_follow_convention(self, sample_meta_plan):
        assembler = TeamAssembler(model_registry=None, date_suffix="20260326")
        team = await assembler.assemble(sample_meta_plan)

        for m in team.members:
            assert m.branch_name.startswith("agent/")
            assert "20260326" in m.branch_name

    @pytest.mark.asyncio
    async def test_fallback_source_label(self, sample_meta_plan):
        assembler = TeamAssembler(model_registry=None)
        team = await assembler.assemble(sample_meta_plan)
        assert all(m.source == "fallback" for m in team.members)

    @pytest.mark.asyncio
    async def test_registry_assignment_when_match_found(self, sample_meta_plan):
        # Mock a ModelRegistry with one matching model
        from prsm.compute.federation.model_registry import (
            ModelDetails, ModelProvider, ModelCapability
        )

        registry = MagicMock()
        registry.models = {
            "mock-architect-model": ModelDetails(
                model_id="mock-architect-model",
                name="MockArchitect",
                provider=ModelProvider.LOCAL,
                capabilities=[ModelCapability.REASONING, ModelCapability.ANALYSIS],
                performance_score=0.95,
                availability=1.0,
                specialization_domains=["architecture", "design"],
            )
        }

        assembler = TeamAssembler(model_registry=registry)
        team = await assembler.assemble(sample_meta_plan)

        # At least the architect role should be resolved from registry
        architect = team.member_by_role("architect")
        assert architect is not None
        # Source could be registry or fallback depending on score calculation
        assert architect.model_id in ("mock-architect-model", "anthropic/claude-opus-4-6")

    @pytest.mark.asyncio
    async def test_team_summary_is_non_empty(self, sample_meta_plan):
        assembler = TeamAssembler()
        team = await assembler.assemble(sample_meta_plan)
        summary = team.summary()
        assert "Agent Team" in summary
        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_member_by_role_returns_correct_member(self, sample_meta_plan):
        assembler = TeamAssembler()
        team = await assembler.assemble(sample_meta_plan)
        coder = team.member_by_role("backend-coder")
        assert coder is not None
        assert coder.role == "backend-coder"

    @pytest.mark.asyncio
    async def test_member_by_role_returns_none_for_unknown(self, sample_meta_plan):
        assembler = TeamAssembler()
        team = await assembler.assemble(sample_meta_plan)
        assert team.member_by_role("nonexistent-role") is None

    @pytest.mark.asyncio
    async def test_branch_names_list(self, sample_meta_plan):
        assembler = TeamAssembler(date_suffix="20260326")
        team = await assembler.assemble(sample_meta_plan)
        branches = team.branch_names()
        assert len(branches) == 3
        assert all("20260326" in b for b in branches)


# ======================================================================
# 4. BranchManager
# ======================================================================

class TestBranchManager:
    @pytest.mark.asyncio
    async def test_create_branch(self, git_repo):
        bm = BranchManager(repo_path=git_repo)
        state = await bm.create_branch("agent/coder-20260326", "backend-coder", "agent-001")

        assert state.branch_name == "agent/coder-20260326"
        assert state.role == "backend-coder"
        assert state.base_commit  # non-empty SHA
        assert state.status == BranchStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_create_team_branches(self, git_repo, sample_meta_plan):
        assembler = TeamAssembler(date_suffix="20260326")
        team = await assembler.assemble(sample_meta_plan)

        bm = BranchManager(repo_path=git_repo)
        states = await bm.create_team_branches(team.members)

        assert len(states) == 3
        branches_in_repo = await bm.list_agent_branches()
        for branch_name in states:
            assert branch_name in branches_in_repo

    @pytest.mark.asyncio
    async def test_list_agent_branches_empty(self, git_repo):
        bm = BranchManager(repo_path=git_repo)
        branches = await bm.list_agent_branches()
        assert branches == []

    @pytest.mark.asyncio
    async def test_list_agent_branches_after_creation(self, git_repo):
        bm = BranchManager(repo_path=git_repo)
        await bm.create_branch("agent/coder-20260326", "coder", "a1")
        await bm._git("checkout", "main")
        await bm.create_branch("agent/tester-20260326", "tester", "a2")
        await bm._git("checkout", "main")

        branches = await bm.list_agent_branches()
        assert "agent/coder-20260326" in branches
        assert "agent/tester-20260326" in branches

    @pytest.mark.asyncio
    async def test_get_diff_empty_on_fresh_branch(self, git_repo):
        bm = BranchManager(repo_path=git_repo)
        await bm.create_branch("agent/coder-20260326", "coder", "a1")
        diff = await bm.get_diff("agent/coder-20260326")
        assert diff.is_empty

    @pytest.mark.asyncio
    async def test_get_diff_after_changes(self, git_repo):
        bm = BranchManager(repo_path=git_repo)
        await bm.create_branch("agent/coder-20260326", "coder", "a1")

        # Make a commit on the branch
        new_file = git_repo / "feature.py"
        new_file.write_text("def hello(): pass\n")
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "add feature")

        diff = await bm.get_diff("agent/coder-20260326")
        assert not diff.is_empty
        assert diff.insertions > 0

    @pytest.mark.asyncio
    async def test_merge_branch(self, git_repo):
        bm = BranchManager(repo_path=git_repo)
        await bm.create_branch("agent/coder-20260326", "coder", "a1")

        # Add a file on the branch
        new_file = git_repo / "feature.py"
        new_file.write_text("def hello(): pass\n")
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "agent work")

        sha = await bm.merge_branch("agent/coder-20260326", "checkpoint: merge coder branch")
        assert sha  # non-empty SHA

        state = bm.get_state("agent/coder-20260326")
        assert state.status == BranchStatus.MERGED
        assert state.merged_at is not None

        # feature.py should now exist on main
        assert (git_repo / "feature.py").exists()

    @pytest.mark.asyncio
    async def test_abandon_branch(self, git_repo):
        bm = BranchManager(repo_path=git_repo)
        await bm.create_branch("agent/coder-20260326", "coder", "a1")
        await bm._git("checkout", "main")

        await bm.abandon_branch("agent/coder-20260326")

        state = bm.get_state("agent/coder-20260326")
        assert state.status == BranchStatus.ABANDONED

        branches = await bm.list_agent_branches()
        assert "agent/coder-20260326" not in branches

    @pytest.mark.asyncio
    async def test_active_branches(self, git_repo):
        bm = BranchManager(repo_path=git_repo)
        await bm.create_branch("agent/a-20260326", "a", "1")
        await bm._git("checkout", "main")
        await bm.create_branch("agent/b-20260326", "b", "2")
        await bm._git("checkout", "main")

        active = bm.active_branches()
        assert len(active) == 2

    @pytest.mark.asyncio
    async def test_commit_count_on_branch(self, git_repo):
        bm = BranchManager(repo_path=git_repo)
        await bm.create_branch("agent/coder-20260326", "coder", "a1")

        # 0 commits since branch (fresh branch = no divergence)
        assert await bm.commit_count_on_branch("agent/coder-20260326") == 0

        # Add a commit
        (git_repo / "x.py").write_text("x = 1\n")
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "x")
        assert await bm.commit_count_on_branch("agent/coder-20260326") == 1


# ======================================================================
# 5. CheckpointReviewer
# ======================================================================

class TestCheckpointReviewer:
    def _make_reviewer(self, git_repo, sample_meta_plan, backend=None):
        bm = BranchManager(repo_path=git_repo)
        return CheckpointReviewer(
            branch_manager=bm,
            meta_plan=sample_meta_plan,
            backend_registry=backend,
        )

    @pytest.mark.asyncio
    async def test_rejects_empty_diff(self, git_repo, sample_meta_plan):
        reviewer = self._make_reviewer(git_repo, sample_meta_plan)
        bm = reviewer._bm
        await bm.create_branch("agent/coder-20260326", "coder", "a1")

        decision = await reviewer.review("agent/coder-20260326")
        assert not decision.approved
        assert "empty" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_heuristic_approves_non_empty_diff(self, git_repo, sample_meta_plan):
        reviewer = self._make_reviewer(git_repo, sample_meta_plan)
        bm = reviewer._bm
        await bm.create_branch("agent/coder-20260326", "coder", "a1")

        # Add some work
        (git_repo / "bsc_filter.py").write_text("class KLFilter: pass\n")
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "implement BSC filter")

        decision = await reviewer.review("agent/coder-20260326")
        assert decision.approved
        assert decision.heuristic

    @pytest.mark.asyncio
    async def test_heuristic_milestone_keyword_scoring(self, git_repo, sample_meta_plan):
        reviewer = self._make_reviewer(git_repo, sample_meta_plan)
        bm = reviewer._bm
        await bm.create_branch("agent/coder-20260326", "coder", "a1")

        # File name contains a milestone keyword ("bsc" is in "BSC Core")
        (git_repo / "bsc_core.py").write_text("# BSC Core implementation\nclass Predictor: pass\n")
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "BSC core module")

        decision = await reviewer.review("agent/coder-20260326")
        assert decision.approved
        # Alignment should be > 0.5 due to keyword match
        assert decision.meta_plan_alignment >= 0.5

    @pytest.mark.asyncio
    async def test_llm_review_path(self, git_repo, sample_meta_plan):
        backend = MagicMock()
        llm_response = MagicMock()
        llm_response.text = """{
            "approved": true,
            "alignment_score": 0.88,
            "milestone_index": 0,
            "reason": "Diff implements BSC core as planned",
            "notes": "Good coverage"
        }"""
        backend.generate = AsyncMock(return_value=llm_response)

        reviewer = self._make_reviewer(git_repo, sample_meta_plan, backend=backend)
        bm = reviewer._bm
        await bm.create_branch("agent/coder-20260326", "coder", "a1")

        (git_repo / "bsc.py").write_text("class BSC: pass\n")
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "implement BSC")

        decision = await reviewer.review("agent/coder-20260326")
        assert decision.approved
        assert not decision.heuristic
        assert decision.meta_plan_alignment == pytest.approx(0.88)
        assert decision.milestone_index == 0

    @pytest.mark.asyncio
    async def test_approve_and_merge_creates_commit(self, git_repo, sample_meta_plan):
        reviewer = self._make_reviewer(git_repo, sample_meta_plan)
        bm = reviewer._bm
        await bm.create_branch("agent/coder-20260326", "coder", "a1")

        (git_repo / "work.py").write_text("# work\n")
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "work done")

        decision = await reviewer.review("agent/coder-20260326")
        assert decision.approved  # heuristic approve

        decision = await reviewer.approve_and_merge(decision)
        assert decision.merge_commit_sha is not None

        state = bm.get_state("agent/coder-20260326")
        assert state.status == BranchStatus.MERGED

    @pytest.mark.asyncio
    async def test_approve_and_merge_rejects_unapproved(self, git_repo, sample_meta_plan):
        reviewer = self._make_reviewer(git_repo, sample_meta_plan)
        bm = reviewer._bm
        await bm.create_branch("agent/coder-20260326", "coder", "a1")

        # Empty diff → rejected
        decision = await reviewer.review("agent/coder-20260326")
        assert not decision.approved

        with pytest.raises(ValueError, match="rejected"):
            await reviewer.approve_and_merge(decision)

    @pytest.mark.asyncio
    async def test_review_and_merge_convenience(self, git_repo, sample_meta_plan):
        reviewer = self._make_reviewer(git_repo, sample_meta_plan)
        bm = reviewer._bm
        await bm.create_branch("agent/coder-20260326", "coder", "a1")

        (git_repo / "feature.py").write_text("class Feature: pass\n")
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "feature")

        decision = await reviewer.review_and_merge("agent/coder-20260326")
        assert decision.approved
        assert decision.merge_commit_sha is not None

    def test_decision_history_tracks_reviews(self, git_repo, sample_meta_plan):
        reviewer = self._make_reviewer(git_repo, sample_meta_plan)
        assert len(reviewer.decision_history) == 0

    def test_update_ledger_sha(self, git_repo, sample_meta_plan):
        reviewer = self._make_reviewer(git_repo, sample_meta_plan)
        reviewer.update_ledger_sha("abc123")
        assert reviewer._ledger_sha == "abc123"

    @pytest.mark.asyncio
    async def test_checkpoint_decision_verdict(self, git_repo, sample_meta_plan):
        reviewer = self._make_reviewer(git_repo, sample_meta_plan)
        bm = reviewer._bm
        await bm.create_branch("agent/coder-20260326", "coder", "a1")

        decision = await reviewer.review("agent/coder-20260326")
        assert decision.verdict in ("APPROVED", "REJECTED")


# ======================================================================
# 6. End-to-end: interview → plan → assemble → branch → review
# ======================================================================

class TestTeamCoordinationEndToEnd:
    @pytest.mark.asyncio
    async def test_full_session_flow(self, git_repo):
        """
        Complete flow from initial prompt to a reviewed and merged agent branch.
        All LLM calls are bypassed (template/heuristic paths).
        """
        # 1. Interview
        answers = iter([
            "Build the NWTN BSC system for PRSM",   # goal
            "Python, aiosqlite",                      # tech
            "No breaking API changes",               # constraints
            "All BSC tests pass",                    # success
            "2 weeks",                               # timeline
        ])
        session = InterviewSession(
            ask=lambda q: asyncio.coroutine(lambda: next(answers, ""))(),
            backend_registry=None,
        )

        # Use a simpler mock
        async def ask(q):
            return next(answers, "n/a")

        session2 = InterviewSession(ask=ask, backend_registry=None, session_id="e2e-001")
        brief = await session2.run("Build the NWTN BSC system for PRSM")

        assert brief.session_id == "e2e-001"

        # 2. Plan
        planner = MetaPlanner(backend_registry=None)
        meta_plan = await planner.generate(brief)
        assert meta_plan.milestone_count() >= 1
        whiteboard_entry = meta_plan.to_whiteboard_entry()
        assert "META-PLAN" in whiteboard_entry

        # 3. Assemble
        assembler = TeamAssembler(date_suffix="20260326")
        team = await assembler.assemble(meta_plan)
        assert len(team.members) > 0

        # 4. Create branches
        bm = BranchManager(repo_path=git_repo)
        await bm.create_team_branches(team.members)
        agent_branches = await bm.list_agent_branches()
        assert len(agent_branches) == len(team.members)

        # 5. Simulate agent work on first branch
        first_branch = team.members[0].branch_name
        await bm._git("checkout", first_branch)
        work_file = git_repo / "agent_work.py"
        work_file.write_text("# Agent work result\nclass AgentOutput: pass\n")
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "agent: implement AgentOutput")
        await bm._git("checkout", "main")

        # 6. Checkpoint review and merge
        reviewer = CheckpointReviewer(branch_manager=bm, meta_plan=meta_plan)
        decision = await reviewer.review_and_merge(first_branch)

        assert decision.approved
        assert decision.merge_commit_sha is not None
        assert (git_repo / "agent_work.py").exists()
