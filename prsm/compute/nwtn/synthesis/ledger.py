"""
Project Ledger
==============

Append-only, tamper-evident record of every session's Nightly Synthesis.

The ledger is the long-term memory of the project — the primary onboarding
artefact for agents joining mid-project and for humans auditing progress.

Storage layout
--------------
Two files live side-by-side at the configured *ledger_dir*:

    <ledger_dir>/project_ledger.md    — human-readable Markdown narrative
    <ledger_dir>/project_ledger.json  — machine-readable entry records

The Markdown file is what agents and humans read.
The JSON file is what the verifier reads to check the hash chain.

Thread/process safety
---------------------
The ledger uses a simple file-append model — safe for a single writer
(the Nightly Synthesis job).  Concurrent writes from multiple processes
are not supported; use an external lock file if needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .signer import (
    GENESIS_HASH,
    EntrySignature,
    LedgerSigner,
    VerificationResult,
    hash_content,
)
from .reconstructor import SynthesisResult

logger = logging.getLogger(__name__)

_LEDGER_MD   = "project_ledger.md"
_LEDGER_JSON = "project_ledger.json"


# ======================================================================
# Data models
# ======================================================================

@dataclass
class LedgerEntry:
    """
    One entry in the Project Ledger — corresponds to one Nightly Synthesis.
    """
    entry_index: int
    session_id: str

    content: str
    """The Markdown narrative produced by the Re-constructor Agent."""

    timestamp: datetime

    content_hash: str
    """SHA-256 of content (UTF-8)."""

    chain_hash: str
    """SHA-256(content_hash + ':' + previous_hash) — also what is signed."""

    previous_hash: str
    """chain_hash of the preceding entry, or GENESIS_HASH for entry 0."""

    signature_b64: str
    """Ed25519 signature over chain_hash."""

    public_key_b64: str
    """Base64 raw public key bytes used for this entry's signature."""

    agents_involved: List[str] = field(default_factory=list)
    whiteboard_entry_count: int = 0
    llm_assisted: bool = True
    dag_anchor_tx: Optional[str] = None
    """DAG transaction ID if this entry was anchored (set post-write)."""

    def to_markdown_block(self) -> str:
        """
        Render this entry as the Markdown block appended to the ledger file.

        Includes the narrative + a metadata table for human inspection.
        """
        ts = self.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        agents_str = ", ".join(
            a.removeprefix("agent/") for a in self.agents_involved
        ) or "—"
        anchor_row = (
            f"| DAG Anchor TX | `{self.dag_anchor_tx}` |\n"
            if self.dag_anchor_tx
            else ""
        )

        return (
            f"---\n\n"
            f"### Entry #{self.entry_index}  —  {ts}\n\n"
            f"{self.content}\n\n"
            f"<details>\n<summary>Entry Metadata</summary>\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| Entry Index | `{self.entry_index}` |\n"
            f"| Session | `{self.session_id}` |\n"
            f"| Agents | {agents_str} |\n"
            f"| Whiteboard entries | {self.whiteboard_entry_count} |\n"
            f"| LLM-assisted | {self.llm_assisted} |\n"
            f"| Content Hash | `{self.content_hash[:16]}…` |\n"
            f"| Chain Hash | `{self.chain_hash[:16]}…` |\n"
            f"| Previous Hash | `{self.previous_hash[:16]}…` |\n"
            f"| Signature | `{self.signature_b64[:24]}…` |\n"
            f"{anchor_row}"
            f"\n</details>\n\n"
        )

    def to_dict(self) -> Dict:
        """Serialise to a JSON-safe dict."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "LedgerEntry":
        d = dict(d)
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


# ======================================================================
# ProjectLedger
# ======================================================================

class ProjectLedger:
    """
    Append-only Project Ledger backed by a Markdown + JSON file pair.

    Parameters
    ----------
    ledger_dir : Path
        Directory where the two ledger files are stored.  Created if absent.
    project_title : str
        Short project title for the Markdown header.
    """

    def __init__(
        self,
        ledger_dir: Path,
        project_title: str = "PRSM Project",
    ) -> None:
        self._dir = Path(ledger_dir)
        self._title = project_title
        self._md_path = self._dir / _LEDGER_MD
        self._json_path = self._dir / _LEDGER_JSON
        self._entries: List[LedgerEntry] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load existing entries from the JSON sidecar file."""
        self._dir.mkdir(parents=True, exist_ok=True)

        if self._json_path.exists():
            try:
                raw = json.loads(self._json_path.read_text(encoding="utf-8"))
                self._entries = [LedgerEntry.from_dict(e) for e in raw]
                logger.info(
                    "ProjectLedger: loaded %d entries from %s",
                    len(self._entries), self._json_path,
                )
            except Exception as exc:
                logger.error(
                    "ProjectLedger: failed to load %s: %s — starting fresh",
                    self._json_path, exc,
                )
                self._entries = []
        else:
            self._entries = []

        if not self._md_path.exists():
            self._write_md_header()

        self._loaded = True

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def append(
        self,
        synthesis: SynthesisResult,
        signer: LedgerSigner,
    ) -> LedgerEntry:
        """
        Sign and append a ``SynthesisResult`` to the ledger.

        Parameters
        ----------
        synthesis : SynthesisResult
            The Nightly Synthesis output from the Re-constructor Agent.
        signer : LedgerSigner
            Signing key to use for this entry.

        Returns
        -------
        LedgerEntry
            The newly created, persisted entry.
        """
        if not self._loaded:
            self.load()

        entry_index = len(self._entries)
        previous_hash = (
            self._entries[-1].chain_hash if self._entries else GENESIS_HASH
        )

        content_hash = hash_content(synthesis.narrative)
        entry_sig: EntrySignature = signer.sign_entry(content_hash, previous_hash)

        entry = LedgerEntry(
            entry_index=entry_index,
            session_id=synthesis.session_id,
            content=synthesis.narrative,
            timestamp=synthesis.timestamp,
            content_hash=content_hash,
            chain_hash=entry_sig.chain_hash,
            previous_hash=previous_hash,
            signature_b64=entry_sig.signature_b64,
            public_key_b64=entry_sig.public_key_b64,
            agents_involved=synthesis.agents_involved,
            whiteboard_entry_count=synthesis.whiteboard_entry_count,
            llm_assisted=synthesis.llm_assisted,
        )

        self._entries.append(entry)
        self._persist()

        logger.info(
            "ProjectLedger: entry #%d appended (session=%s, hash=%s…)",
            entry_index, synthesis.session_id, entry.chain_hash[:12],
        )
        return entry

    def update_dag_anchor(self, entry_index: int, dag_tx: str) -> None:
        """
        Back-fill the DAG transaction ID on an existing entry after anchoring.
        Rewrites the JSON sidecar but does NOT recompute the hash/signature
        (the anchor is metadata, not part of the signed content).
        """
        if not 0 <= entry_index < len(self._entries):
            raise IndexError(f"No entry #{entry_index}")
        # dataclass is mutable — update in place
        object.__setattr__(self._entries[entry_index], "dag_anchor_tx", dag_tx)
        self._persist_json()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_entry(self, index: int) -> LedgerEntry:
        if not self._loaded:
            self.load()
        if not 0 <= index < len(self._entries):
            raise IndexError(f"No entry #{index}")
        return self._entries[index]

    def latest_entry(self) -> Optional[LedgerEntry]:
        if not self._loaded:
            self.load()
        return self._entries[-1] if self._entries else None

    def latest_chain_hash(self) -> str:
        """Return the chain_hash of the most recent entry (or GENESIS_HASH)."""
        entry = self.latest_entry()
        return entry.chain_hash if entry else GENESIS_HASH

    @property
    def entry_count(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._entries)

    def read_markdown(self) -> str:
        """Return the full ledger Markdown content."""
        if self._md_path.exists():
            return self._md_path.read_text(encoding="utf-8")
        return f"# {self._title}\n\n*(empty)*\n"

    def to_onboarding_context(
        self,
        max_entries: int = 10,
        max_chars: int = 6000,
    ) -> str:
        """
        Return a condensed version of the ledger suitable for feeding to
        an agent at session start.

        Uses the most recent *max_entries* entries and truncates to *max_chars*.
        """
        if not self._loaded:
            self.load()

        if not self._entries:
            return f"# {self._title}\n\n*(No history yet)*\n"

        recent = self._entries[-max_entries:]
        header = (
            f"# {self._title} — Project History\n\n"
            f"Total sessions: {len(self._entries)}  \n"
            f"Showing: last {len(recent)} session(s)\n\n"
            "---\n\n"
        )
        body_parts: List[str] = []
        budget = max_chars - len(header)

        for entry in reversed(recent):
            block = (
                f"### Entry #{entry.entry_index} — "
                f"{entry.timestamp.strftime('%Y-%m-%d')}\n\n"
                f"{entry.content[:1200]}"
                f"{'…' if len(entry.content) > 1200 else ''}\n\n"
            )
            if budget - len(block) < 0:
                break
            body_parts.insert(0, block)
            budget -= len(block)

        return header + "\n".join(body_parts)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self) -> VerificationResult:
        """Verify the integrity of the entire ledger chain."""
        if not self._loaded:
            self.load()
        return LedgerSigner.verify_chain(self._entries)

    # ------------------------------------------------------------------
    # Internal: persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Write both the JSON sidecar and append to the Markdown file."""
        self._persist_json()
        self._append_to_markdown(self._entries[-1])

    def _persist_json(self) -> None:
        raw = [e.to_dict() for e in self._entries]
        self._json_path.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _write_md_header(self) -> None:
        header = (
            f"# {self._title} — Project Ledger\n\n"
            f"*Append-only tamper-evident record of all session syntheses.*  \n"
            f"*Each entry is Ed25519-signed and SHA-256 hash-chained.*\n\n"
        )
        self._md_path.write_text(header, encoding="utf-8")

    def _append_to_markdown(self, entry: LedgerEntry) -> None:
        block = entry.to_markdown_block()
        with self._md_path.open("a", encoding="utf-8") as f:
            f.write(block)
