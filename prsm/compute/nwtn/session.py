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

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from prsm.compute.nwtn.openclaw.adapter import NWTNOpenClawAdapter, SessionState

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of an NWTNSession.run() execution."""

    session_id: str
    rounds_completed: int
    converged: bool
    convergence_summary: dict
    final_status: str
    elapsed_seconds: float
    context_resets_triggered: int = 0


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
        self._rounds_completed = 0
        self._converged = False

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

    async def run(
        self,
        max_rounds: int = 20,
        round_poll_interval: float = 5.0,
        agent_output_fn: Optional[Callable[[str, int], Awaitable[str]]] = None,
        context_monitor=None,
    ) -> RunResult:
        """
        Drive the NWTN checkpoint loop until convergence or max_rounds.

        Parameters
        ----------
        max_rounds : int
            Maximum checkpoint rounds before giving up.
        round_poll_interval : float
            Seconds to wait between convergence polls (default 5s).
        agent_output_fn : async callable(agent_id, round_number) -> str, optional
            Called each round to get each agent's output for convergence scanning.
            If None, convergence is never detected (useful for externally-driven sessions).
        context_monitor : ContextPressureMonitor, optional
            If provided, monitors token usage per agent and triggers context resets
            when agents exceed critical thresholds. Enables long-running sessions
            to continue via structured handoff artifacts.

        Returns
        -------
        RunResult
            Summary of the run: rounds completed, converged, final status, context resets.
        """
        start_time = time.perf_counter()
        context_resets_triggered = 0

        for round_num in range(1, max_rounds + 1):
            # Step 1: Scan agent outputs for convergence signals if fn provided
            if agent_output_fn is not None:
                for agent_id in self.team_members:
                    try:
                        output = await agent_output_fn(agent_id, round_num)
                        self._adapter.scan_agent_convergence(
                            self.session_id, agent_id, output, round_num
                        )
                        # Record tokens for context pressure monitoring
                        if context_monitor is not None:
                            token_count = len(output.split()) * 4  # Rough estimate: 4 chars per token
                            context_monitor.record_tokens(agent_id, round_num, token_count)
                    except Exception as e:
                        logger.warning(
                            f"NWTNSession [{self.session_id}] agent_output_fn error for {agent_id}: {e}"
                        )

            # Step 1.5: Check for context pressure resets (if monitor provided)
            if context_monitor is not None:
                agents_needing_reset = context_monitor.agents_needing_reset()
                if agents_needing_reset:
                    for agent_id in agents_needing_reset:
                        try:
                            # Build handoff artifact for the fresh agent
                            # Get whiteboard entries from convergence summary
                            summary = self._adapter.convergence_summary(self.session_id)
                            whiteboard_entries = summary.get("whiteboard_entries", [])
                            meta_plan_summary = summary.get("meta_plan_summary", "")

                            handoff = context_monitor.build_handoff_artifact(
                                agent_id=agent_id,
                                whiteboard_entries=whiteboard_entries,
                                meta_plan_summary=meta_plan_summary,
                                goal=self._state.goal,
                            )
                            logger.info(
                                f"NWTNSession [{self.session_id}] built handoff artifact for agent={agent_id} "
                                f"(round={round_num}, tokens={handoff.get('tokens_before_reset', 0)})"
                            )

                            # Reset the agent's context
                            context_monitor.reset_agent_context(agent_id)
                            context_resets_triggered += 1
                            logger.info(
                                f"NWTNSession [{self.session_id}] context reset for agent={agent_id} "
                                f"(total_resets={context_resets_triggered})"
                            )
                        except Exception as e:
                            logger.warning(
                                f"NWTNSession [{self.session_id}] context reset error for {agent_id}: {e}"
                            )

            # Step 2: Publish ROUND_ADVANCED event (triggers ScribeAgent checkpoint)
            await self._adapter.advance_bsc_round(
                self.session_id, f"Round {round_num} complete"
            )

            # Step 3: Wait for convergence check
            await asyncio.sleep(round_poll_interval)

            # Step 4: Check convergence
            if self._adapter.is_session_converged(self.session_id):
                self._converged = True
                self._rounds_completed = round_num
                logger.info(
                    f"NWTNSession [{self.session_id}] round {round_num}/{max_rounds} — CONVERGED"
                )
                break

            # Step 5: Log progress
            pending = self._adapter.convergence_summary(self.session_id).get(
                "pending_agents", []
            )
            logger.info(
                f"NWTNSession [{self.session_id}] round {round_num}/{max_rounds} — pending: {pending}"
            )
            self._rounds_completed = round_num
        else:
            # Completed max_rounds without convergence
            self._rounds_completed = max_rounds

        elapsed_seconds = time.perf_counter() - start_time

        return RunResult(
            session_id=self.session_id,
            rounds_completed=self._rounds_completed,
            converged=self._converged,
            convergence_summary=self._adapter.convergence_summary(self.session_id),
            final_status=self._state.status,
            elapsed_seconds=elapsed_seconds,
            context_resets_triggered=context_resets_triggered,
        )

    def status(self) -> dict:
        """
        Return a status dictionary for the session.

        Returns
        -------
        dict
            Dictionary with keys: session_id, goal, status, team_members,
            scribe_running, rounds_completed, converged.
        """
        return {
            "session_id": self._state.session_id,
            "goal": self._state.goal,
            "status": self._state.status,
            "team_members": self._state.team_members,
            "scribe_running": self._state.scribe_running,
            "rounds_completed": self._rounds_completed,
            "converged": self._converged,
        }
