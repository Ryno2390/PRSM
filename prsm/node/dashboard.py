"""
PRSM Node Dashboard
====================

Rich-based live terminal dashboard for monitoring a running PRSM node.
Polls ``node.get_status()`` every 2 seconds and renders a multi-panel
layout showing peers, content, compute, agents, and recent activity.

No new dependencies — uses Rich 13.7+ (already installed).
"""

import asyncio
import collections
import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from prsm.node.node import PRSMNode

logger = logging.getLogger(__name__)


class _ActivityLogHandler(logging.Handler):
    """Captures recent log lines from ``prsm.node.*`` into a deque."""

    def __init__(self, maxlen: int = 8):
        super().__init__()
        self.records: collections.deque = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("prsm.node"):
            ts = time.strftime("%H:%M", time.localtime(record.created))
            msg = record.getMessage()
            # Truncate long messages for the dashboard
            if len(msg) > 60:
                msg = msg[:57] + "..."
            self.records.append(f"{ts} {msg}")


class NodeDashboard:
    """Live terminal dashboard for a running PRSM node.

    Usage::

        dashboard = NodeDashboard(node)
        await dashboard.run()  # blocks until Ctrl+C
    """

    REFRESH_INTERVAL = 2  # seconds

    def __init__(self, node: "PRSMNode") -> None:
        self.node = node
        self._log_handler = _ActivityLogHandler(maxlen=8)

        # Attach handler to the prsm.node logger tree
        node_logger = logging.getLogger("prsm.node")
        node_logger.addHandler(self._log_handler)

    async def run(self) -> None:
        """Run the live dashboard until cancelled."""
        with Live(self._build_display({}), refresh_per_second=1, screen=False) as live:
            try:
                while True:
                    status = await self.node.get_status()
                    live.update(self._build_display(status))
                    await asyncio.sleep(self.REFRESH_INTERVAL)
            except asyncio.CancelledError:
                pass
            finally:
                # Remove our log handler on exit
                node_logger = logging.getLogger("prsm.node")
                node_logger.removeHandler(self._log_handler)

    # ── Layout builders ──────────────────────────────────────────

    def _build_display(self, status: Dict[str, Any]) -> Table:
        """Build the full dashboard as a Rich Table (grid mode)."""
        grid = Table.grid(padding=0)
        grid.add_column(ratio=1)

        # Header
        grid.add_row(self._build_header(status))

        # Middle: 2-column layout
        middle = Table.grid(padding=1)
        middle.add_column(ratio=1)
        middle.add_column(ratio=1)
        middle.add_row(
            self._build_network(status),
            self._build_content(status),
        )
        middle.add_row(
            self._build_compute(status),
            self._build_agents(status),
        )
        grid.add_row(middle)

        # Footer
        grid.add_row(
            Text("Press Ctrl+C to stop", style="dim", justify="center")
        )
        return grid

    def _build_header(self, status: Dict[str, Any]) -> Panel:
        """Top header with node ID, uptime, and FTNS balance."""
        node_id = status.get("node_id", "initializing...")
        if node_id and len(node_id) > 12:
            node_id_short = node_id[:12]
        else:
            node_id_short = node_id or "..."

        uptime = self._format_uptime(status.get("uptime_seconds", 0))
        balance = status.get("ftns_balance", 0.0)

        header = Text()
        header.append("  Node: ", style="bold")
        header.append(node_id_short, style="cyan")
        header.append("  |  Uptime: ", style="bold")
        header.append(uptime, style="green")
        header.append("  |  FTNS: ", style="bold")
        header.append(f"{balance:.2f}", style="yellow")

        return Panel(header, title="PRSM Node Dashboard", style="bold blue")

    def _build_network(self, status: Dict[str, Any]) -> Panel:
        """Network panel: peer counts, gossip status, peer list."""
        peers = status.get("peers", {})
        connected = peers.get("connected", 0)
        known = peers.get("known", 0)

        lines = Text()
        lines.append(f"  {connected}", style="bold green" if connected > 0 else "bold red")
        lines.append(f" peers connected\n")
        lines.append(f"  {known}", style="cyan")
        lines.append(f" peers known\n")

        # Gossip status
        gossip_active = status.get("started", False)
        if gossip_active:
            lines.append("  Gossip: ", style="")
            lines.append("active\n", style="bold green")
        else:
            lines.append("  Gossip: ", style="")
            lines.append("inactive\n", style="dim")

        # Show peer list from transport
        if self.node.transport and self.node.transport.peers:
            lines.append("\n  Peers:\n", style="bold")
            for pid, peer in list(self.node.transport.peers.items())[:6]:
                direction = "outbound" if peer.outbound else "inbound"
                name = peer.display_name or pid[:8]
                lines.append(f"  {name}", style="cyan")
                lines.append(f" ({direction})\n", style="dim")
        else:
            lines.append("\n  No peers connected\n", style="dim")

        return Panel(lines, title="Network", border_style="blue")

    def _build_content(self, status: Dict[str, Any]) -> Panel:
        """Content & economy panel: CIDs, uploads, royalties, ledger sync."""
        content_index = status.get("content_index") or {}
        content = status.get("content") or {}
        ledger_sync = status.get("ledger_sync") or {}

        indexed = content_index.get("indexed_cids", 0)
        uploaded = content.get("uploaded_count", 0)
        royalties = content.get("total_royalties_ftns", 0.0)
        txs_broadcast = ledger_sync.get("txs_broadcast", 0)
        txs_received = ledger_sync.get("txs_received", 0)
        discrepancies = ledger_sync.get("discrepancies_found", 0)

        lines = Text()
        lines.append(f"  {indexed}", style="cyan")
        lines.append(" CIDs indexed\n")
        lines.append(f"  {uploaded}", style="cyan")
        lines.append(" files uploaded\n")
        lines.append(f"  {royalties:.3f}", style="yellow")
        lines.append(" FTNS royalties\n")
        lines.append(f"  {txs_broadcast}", style="cyan")
        lines.append(" txs broadcast / ")
        lines.append(f"{txs_received}", style="cyan")
        lines.append(" received\n")
        lines.append(f"  {discrepancies}", style="green" if discrepancies == 0 else "red")
        lines.append(" discrepancies\n")

        return Panel(lines, title="Content & Economy", border_style="yellow")

    def _build_compute(self, status: Dict[str, Any]) -> Panel:
        """Compute & storage panel: CPU, RAM, jobs, IPFS, storage."""
        compute = status.get("compute") or {}
        storage = status.get("storage") or {}

        lines = Text()

        if compute:
            res = compute.get("resources", {})
            alloc = compute.get("allocation", {})
            cpu_count = res.get("cpu_count", "?")
            cpu_pct = alloc.get("cpu_pct", "?")
            mem_gb = res.get("memory_total_gb", "?")
            mem_pct = alloc.get("memory_pct", "?")
            active_jobs = compute.get("active_jobs", 0)

            lines.append(f"  CPU: {cpu_count} cores ({cpu_pct}%)\n")
            lines.append(f"  RAM: {mem_gb} GB ({mem_pct}%)\n")
            lines.append(f"  Jobs: ", style="")
            lines.append(f"{active_jobs}", style="bold green" if active_jobs == 0 else "bold yellow")
            lines.append(" active\n")
        else:
            lines.append("  Compute: not enabled\n", style="dim")

        if storage:
            ipfs_ok = storage.get("ipfs_available", False)
            pledged = storage.get("pledged_gb", 0)
            used = storage.get("used_gb", 0)
            pinned = storage.get("pinned_cids", 0)

            lines.append("  IPFS: ", style="")
            lines.append("connected\n" if ipfs_ok else "not available\n",
                          style="green" if ipfs_ok else "dim")
            lines.append(f"  Storage: {used:.1f} / {pledged} GB\n")
            lines.append(f"  Pinned: {pinned} CIDs\n")
        else:
            lines.append("  Storage: not enabled\n", style="dim")

        return Panel(lines, title="Compute & Storage", border_style="green")

    def _build_agents(self, status: Dict[str, Any]) -> Panel:
        """Agents panel: registry stats, collaboration, and recent activity log."""
        agents = status.get("agents") or {}
        collab = status.get("collaboration") or {}

        local = agents.get("local_agents", 0)
        total = agents.get("total_agents", 0)
        online = agents.get("online_agents", 0)
        conversations = agents.get("active_conversations", 0)
        open_tasks = collab.get("open_tasks", 0)

        lines = Text()
        if agents:
            lines.append(f"  {local}", style="cyan")
            lines.append(f" local / ")
            lines.append(f"{total}", style="cyan")
            lines.append(f" known agents")
            if online:
                lines.append(f" ({online} online)", style="green")
            lines.append("\n")
            lines.append(f"  {open_tasks}", style="cyan")
            lines.append(f" open tasks\n")
            lines.append(f"  {conversations}", style="cyan")
            lines.append(f" conversations\n")
        else:
            lines.append("  Agent registry: not enabled\n", style="dim")

        # Recent activity from captured log lines
        if self._log_handler.records:
            lines.append("\n  Recent Activity\n", style="bold")
            for entry in list(self._log_handler.records)[-5:]:
                lines.append(f"  {entry}\n", style="dim")

        return Panel(lines, title="Agents", border_style="magenta")

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format seconds into a human-readable uptime string."""
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m {seconds % 60}s"
        hours = minutes // 60
        remaining_min = minutes % 60
        if hours < 24:
            return f"{hours}h {remaining_min}m"
        days = hours // 24
        remaining_hrs = hours % 24
        return f"{days}d {remaining_hrs}h"
