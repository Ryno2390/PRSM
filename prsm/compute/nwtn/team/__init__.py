"""
prsm.compute.nwtn.team — Agent Team Coordination
=================================================

Sub-phase 10.3: the coordination layer that takes a user's goal from first
conversation to a fully assembled, branch-isolated Agent Team with a MetaPlan
anchoring all work.

Components
----------
InterviewSession / ProjectBrief
    NWTN interview mode.  Gathers structured requirements through a multi-round
    conversation (LLM-driven or template fallback).

MetaPlanner / MetaPlan
    Transforms a ``ProjectBrief`` into a structured plan: objective, milestones
    with merge criteria, required agent roles, success criteria, constraints.

TeamAssembler / AgentTeam / TeamMember
    Maps MetaPlan roles to available agents from the PRSM model registry
    (or fallback defaults).  Assigns git branch names.

BranchManager / BranchState
    Creates and manages per-agent git branches.  Provides diff generation and
    merge execution.  All agent work is branch-isolated; no contamination
    between teammates.

CheckpointReviewer / CheckpointDecision
    NWTN as merge manager.  Reviews branch diffs against the MetaPlan before
    allowing any merge.  Writes tamper-evident commit messages cross-linking
    the Project Ledger.

Typical session flow
--------------------
.. code-block:: python

    from prsm.compute.nwtn.team import (
        InterviewSession, MetaPlanner, TeamAssembler,
        BranchManager, CheckpointReviewer,
    )

    # 1. Interview
    session = InterviewSession(ask=cli_prompt, backend_registry=backend)
    brief = await session.run("Build a BSC-powered agent coordination layer for PRSM")

    # 2. Plan
    planner = MetaPlanner(backend_registry=backend)
    meta_plan = await planner.generate(brief)

    # 3. Assemble team
    assembler = TeamAssembler()
    team = await assembler.assemble(meta_plan)

    # 4. Create branches
    bm = BranchManager(repo_path="/path/to/repo")
    await bm.create_team_branches(team.members)

    # 5. Agents work … whiteboard fills …

    # 6. Checkpoint review
    reviewer = CheckpointReviewer(branch_manager=bm, meta_plan=meta_plan)
    decision = await reviewer.review_and_merge("agent/backend-coder-20260326")
"""

from .assembler import AgentTeam, AgentRole, TeamAssembler, TeamMember
from .branch_manager import BranchManager, BranchState, BranchStatus, GitDiff
from .checkpoint import CheckpointDecision, CheckpointReviewer
from .convergence import ConvergenceTracker, ConvergenceSignal, AgentConvergenceState
from .interview import InterviewSession, ProjectBrief, QAPair, QuestionCallback
from .planner import MetaPlan, MetaPlanner, Milestone

__all__ = [
    # Interview
    "InterviewSession",
    "ProjectBrief",
    "QAPair",
    "QuestionCallback",
    # Planner
    "MetaPlanner",
    "MetaPlan",
    "Milestone",
    "AgentRole",
    # Assembler
    "TeamAssembler",
    "AgentTeam",
    "TeamMember",
    # Branch manager
    "BranchManager",
    "BranchState",
    "BranchStatus",
    "GitDiff",
    # Checkpoint
    "CheckpointReviewer",
    "CheckpointDecision",
    # Convergence
    "ConvergenceTracker",
    "ConvergenceSignal",
    "AgentConvergenceState",
]
