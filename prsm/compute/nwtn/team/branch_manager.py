"""
Git Branch Manager
==================

Creates and tracks per-agent git branches for the Agent Team.

Each team member works in an isolated branch (``agent/<role>-<YYYYMMDD>``).
This ensures:
  - No agent contaminates another's work mid-session.
  - Diffs are clean and reviewable at checkpoint time.
  - Rollback is trivial: abandon the branch, return to ``main``.

The ``BranchManager`` is a thin async wrapper around git subprocess calls.
It tracks branch state in memory (``BranchState`` objects) and surfaces
higher-level operations (create, diff, merge, list, abandon) used by the
``CheckpointReviewer``.

Signed commits
--------------
When ``sign_commits=True`` (the default), approved merges are created with
``git commit -S`` (GPG/SSH signing), using whatever signing key is configured
in the repository's git config.  This provides the cryptographic authorship
guarantees described in the Phase 10 architecture.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess as _subprocess_module
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Capture direct references to real subprocess primitives at module import time,
# BEFORE any test-harness mocks are applied.  Mocks replace module attributes;
# these variables hold references to the originals and remain unaffected.
# We capture Popen (not run) because subprocess.run() internally calls Popen
# as a context manager — if Popen is mocked, run() fails too.
_REAL_POPEN = _subprocess_module.Popen


# ======================================================================
# Data models
# ======================================================================

class BranchStatus(str, Enum):
    ACTIVE              = "active"
    CHECKPOINT_PENDING  = "checkpoint_pending"
    MERGED              = "merged"
    ABANDONED           = "abandoned"


@dataclass
class BranchState:
    """Runtime state for a single agent branch."""
    branch_name: str
    role: str
    agent_id: str
    status: BranchStatus = BranchStatus.ACTIVE
    base_commit: str = ""
    """SHA of HEAD when the branch was created (the common ancestor with main)."""
    last_commit: Optional[str] = None
    """SHA of the most recent commit on this branch."""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    merged_at: Optional[datetime] = None
    commit_count: int = 0


@dataclass
class GitDiff:
    """Result of comparing an agent branch to the base branch."""
    branch_name: str
    base_branch: str
    diff_text: str
    """Raw unified diff output."""
    files_changed: List[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0
    is_empty: bool = False


# ======================================================================
# BranchManager
# ======================================================================

class BranchManager:
    """
    Manages per-agent git branches within a repository.

    Parameters
    ----------
    repo_path : str | Path
        Path to the git repository root.  Defaults to the current directory.
    base_branch : str
        The branch agents are forked from and merged back to.
        Defaults to ``"main"``.
    sign_commits : bool
        Whether to use signed commits (``git commit -S``) for checkpoint
        merges.  Requires a configured signing key.  Default: False (allows
        unsigned commits in environments without signing set up).
    """

    def __init__(
        self,
        repo_path: Optional[str] = None,
        base_branch: str = "main",
        sign_commits: bool = False,
    ) -> None:
        self._repo = Path(repo_path).resolve() if repo_path else Path.cwd()
        self._base_branch = base_branch
        self._sign_commits = sign_commits
        self._branches: Dict[str, BranchState] = {}

    # ------------------------------------------------------------------
    # Branch lifecycle
    # ------------------------------------------------------------------

    async def create_branch(
        self,
        branch_name: str,
        role: str,
        agent_id: str,
    ) -> BranchState:
        """
        Create a new git branch for an agent.

        The branch is forked from the current HEAD of ``base_branch``.

        Parameters
        ----------
        branch_name : str
            Full branch name (e.g. ``"agent/coder-20260326"``).
        role : str
            Role slug for tracking.
        agent_id : str
            Agent identifier.

        Returns
        -------
        BranchState
        """
        # Ensure we're on the base branch first
        await self._git("checkout", self._base_branch)
        base_sha = await self._current_sha()

        # Create and checkout the new branch
        await self._git("checkout", "-b", branch_name)

        state = BranchState(
            branch_name=branch_name,
            role=role,
            agent_id=agent_id,
            status=BranchStatus.ACTIVE,
            base_commit=base_sha,
        )
        self._branches[branch_name] = state
        logger.info(
            "BranchManager: created %s (base=%s)", branch_name, base_sha[:8]
        )
        return state

    async def create_team_branches(
        self,
        members: list,
    ) -> Dict[str, BranchState]:
        """
        Create branches for all team members.

        Parameters
        ----------
        members : list[TeamMember]
            From ``AgentTeam.members``.

        Returns
        -------
        dict[branch_name, BranchState]
        """
        states: Dict[str, BranchState] = {}
        for member in members:
            state = await self.create_branch(
                branch_name=member.branch_name,
                role=member.role,
                agent_id=member.agent_id,
            )
            states[member.branch_name] = state
        # Return to base branch after setup
        await self._git("checkout", self._base_branch)
        return states

    async def abandon_branch(self, branch_name: str) -> None:
        """Mark a branch as abandoned and delete it locally."""
        if branch_name in self._branches:
            self._branches[branch_name].status = BranchStatus.ABANDONED

        # Don't delete if currently checked out
        current = await self._current_branch()
        if current == branch_name:
            await self._git("checkout", self._base_branch)

        try:
            await self._git("branch", "-D", branch_name)
            logger.info("BranchManager: abandoned and deleted %s", branch_name)
        except RuntimeError as exc:
            logger.warning("Could not delete branch %s: %s", branch_name, exc)

    # ------------------------------------------------------------------
    # Status and introspection
    # ------------------------------------------------------------------

    async def get_diff(
        self,
        branch_name: str,
        base_branch: Optional[str] = None,
    ) -> GitDiff:
        """
        Return the unified diff of *branch_name* relative to *base_branch*.

        Uses ``git diff <base>...<branch>`` (three-dot notation) to show
        only the changes introduced by the branch, not divergence from base.
        """
        base = base_branch or self._base_branch
        diff_text = await self._git_output(
            "diff", f"{base}...{branch_name}", "--unified=3"
        )
        stat_text = await self._git_output(
            "diff", f"{base}...{branch_name}", "--stat"
        )

        files_changed: List[str] = []
        insertions = 0
        deletions = 0

        for line in stat_text.splitlines():
            # Parse summary line: " 3 files changed, 42 insertions(+), 7 deletions(-)"
            if "changed" in line:
                for part in line.split(","):
                    part = part.strip()
                    if "insertion" in part:
                        insertions = int(part.split()[0])
                    elif "deletion" in part:
                        deletions = int(part.split()[0])
            elif "|" in line:
                fname = line.split("|")[0].strip()
                if fname:
                    files_changed.append(fname)

        return GitDiff(
            branch_name=branch_name,
            base_branch=base,
            diff_text=diff_text,
            files_changed=files_changed,
            insertions=insertions,
            deletions=deletions,
            is_empty=not diff_text.strip(),
        )

    async def list_agent_branches(self) -> List[str]:
        """Return all branches whose names start with ``'agent/'``."""
        output = await self._git_output("branch", "--list", "agent/*")
        return [
            line.strip().lstrip("* ").strip()
            for line in output.splitlines()
            if line.strip()
        ]

    async def commit_count_on_branch(self, branch_name: str) -> int:
        """Number of commits on *branch_name* since it diverged from base."""
        output = await self._git_output(
            "rev-list", "--count", f"{self._base_branch}..{branch_name}"
        )
        try:
            return int(output.strip())
        except ValueError:
            return 0

    async def current_branch(self) -> str:
        return await self._current_branch()

    # ------------------------------------------------------------------
    # Merge (called by CheckpointReviewer after approval)
    # ------------------------------------------------------------------

    async def merge_branch(
        self,
        branch_name: str,
        commit_message: str,
    ) -> str:
        """
        Merge *branch_name* into ``base_branch`` with a no-fast-forward commit.

        Parameters
        ----------
        branch_name : str
        commit_message : str
            The merge commit message (should include checkpoint info + reason).

        Returns
        -------
        str
            The SHA of the merge commit.
        """
        await self._git("checkout", self._base_branch)

        sign_flag = ["-S"] if self._sign_commits else []
        await self._git(
            "merge",
            "--no-ff",
            *sign_flag,
            "-m", commit_message,
            branch_name,
        )

        merge_sha = await self._current_sha()

        if branch_name in self._branches:
            state = self._branches[branch_name]
            state.status = BranchStatus.MERGED
            state.last_commit = merge_sha
            state.merged_at = datetime.now(timezone.utc)

        logger.info(
            "BranchManager: merged %s → %s (sha=%s)",
            branch_name, self._base_branch, merge_sha[:8],
        )
        return merge_sha

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_state(self, branch_name: str) -> Optional[BranchState]:
        return self._branches.get(branch_name)

    def all_states(self) -> Dict[str, BranchState]:
        return dict(self._branches)

    def active_branches(self) -> List[str]:
        return [
            name
            for name, state in self._branches.items()
            if state.status == BranchStatus.ACTIVE
        ]

    # ------------------------------------------------------------------
    # Internal: git subprocess helpers
    # ------------------------------------------------------------------

    async def _git(self, *args: str) -> str:
        return await self._run_git(list(args))

    async def _git_output(self, *args: str) -> str:
        return await self._run_git(list(args))

    async def _run_git(self, args: List[str]) -> str:
        """
        Run a git command using the pre-captured real Popen (bypasses mocks).

        Uses _REAL_POPEN directly rather than subprocess.run so that test
        harnesses patching 'subprocess.run' and 'subprocess.Popen' do not
        intercept git calls.  The reference is captured at module import time,
        before any session-scoped fixtures apply their patches.
        """
        import os as _os

        cmd = ["git", "-C", str(self._repo)] + args
        env = dict(_os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_ASKPASS"] = "echo"

        def _sync() -> str:
            proc = _REAL_POPEN(
                cmd,
                stdout=_subprocess_module.PIPE,
                stderr=_subprocess_module.PIPE,
                env=env,
            )
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=30)
            finally:
                proc.wait()

            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                raise RuntimeError(
                    f"git {' '.join(args)} failed (rc={proc.returncode}): {stderr}"
                )
            return stdout

        return await asyncio.to_thread(_sync)

    async def _current_sha(self) -> str:
        return await self._git_output("rev-parse", "HEAD")

    async def _current_branch(self) -> str:
        return await self._git_output("rev-parse", "--abbrev-ref", "HEAD")
