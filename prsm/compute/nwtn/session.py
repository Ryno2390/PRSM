"""
NWTNSession — Simple entry point for launching an NWTN agent team session.

Usage:
    session = await NWTNSession.create(
        goal="Build a distributed caching layer for PRSM",
        repo_path=Path("/path/to/repo"),
    )
    await session.run()          # blocks until converged
    await session.close()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prsm.compute.nwtn.openclaw.adapter import NWTNOpenClawAdapter, SessionState


class NWTNSession:
    """
    Convenience factory for launching an NWTN agent team session.

    This class provides a simple API for creating and managing NWTN sessions
    without needing to manually wire up all the components.

    Example
    -------
    >>> session = await NWTNSession.create(
    ...     goal="Implement a REST API for user management",
    ...     repo_path=Path("/home/user/myproject"),
    ... )
    >>> print(session.session_id)
    'a1b2c3d4'
    >>> print(session.team_members)
    ['architect-main', 'coder-auth', 'tester-api']
    >>> print(session.scribe_running)
    True
    >>> await session.close()
    """

    def __init__(self, adapter: "NWTNOpenClawAdapter", session_state: "SessionState"):
        """
        Initialize an NWTNSession.

        Parameters
        ----------
        adapter : NWTNOpenClawAdapter
            The adapter managing this session.
        session_state : SessionState
            The state object for this session.
        """
        self._adapter = adapter
        self._state = session_state

    @classmethod
    async def create(
        cls,
        goal: str,
        repo_path: Optional[Path] = None,
        ask=None,
        orchestrator_backend=None,
    ) -> "NWTNSession":
        """
        Bootstrap a complete NWTNSession with sensible defaults.

        Creates all required components (WhiteboardStore in-memory, promoter, etc.)
        using lightweight defaults suitable for standalone use.

        Parameters
        ----------
        goal : str
            The user's goal statement for the session.
        repo_path : Path, optional
            Git repository for agent branches. Defaults to cwd.
        ask : Callable, optional
            Async function to present questions to the user.
        orchestrator_backend : optional
            LLM backend for meta-level orchestration.

        Returns
        -------
        NWTNSession
            A fully initialized session ready to run.
        """
        from prsm.compute.nwtn.openclaw.adapter import NWTNOpenClawAdapter
        from prsm.compute.nwtn.whiteboard.store import WhiteboardStore
        from prsm.compute.nwtn.bsc.promoter import BSCPromoter
        from prsm.compute.nwtn.bsc.deployment import BSCDeploymentConfig

        # Build store and promoter with defaults (no ML models loaded)
        store = WhiteboardStore()
        config = BSCDeploymentConfig.auto()
        promoter = BSCPromoter.from_config(config, with_quality_gate=True)

        # For ledger/signer/synthesizer — use None for now (graceful degradation)
        adapter = NWTNOpenClawAdapter(
            whiteboard_store=store,
            promoter=promoter,
            ledger=None,
            signer=None,
            synthesizer=None,
            repo_path=repo_path,
            orchestrator_backend=orchestrator_backend,
        )

        state = await adapter.start_session(goal=goal, ask=ask)
        return cls(adapter=adapter, session_state=state)

    @property
    def session_id(self) -> str:
        """Return the session ID."""
        return self._state.session_id

    @property
    def team_members(self) -> list:
        """Return the list of team member branch names."""
        return self._state.team_members

    @property
    def scribe_running(self) -> bool:
        """Return True if the ScribeAgent is active."""
        return self._state.scribe_running

    async def close(self) -> None:
        """Gracefully close the session, stopping all components."""
        await self._adapter.end_session(self._state.session_id)

    def status(self) -> dict:
        """
        Return a status dictionary for the session.

        Returns
        -------
        dict
            Dictionary with keys: session_id, goal, status, team_members,
            scribe_running.
        """
        return {
            "session_id": self._state.session_id,
            "goal": self._state.goal,
            "status": self._state.status,
            "team_members": self._state.team_members,
            "scribe_running": self._state.scribe_running,
        }
