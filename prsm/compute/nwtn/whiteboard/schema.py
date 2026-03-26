"""
Whiteboard Schema
=================

Pydantic models for all data structures stored in and returned from the
Active Whiteboard.  These are the canonical data types shared across the
store, monitor, and query layers.

Design principles
-----------------
- Every entry is immutable after creation (frozen=True).
- The ``extra`` field provides a forward-compatible extension point without
  requiring schema migrations.
- ``WhiteboardSnapshot`` is the complete, serialisable state of a session's
  whiteboard at a point in time — suitable for the Nightly Synthesis agent
  to read and synthesise into the Project Ledger (Sub-phase 10.4).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class WhiteboardEntry(BaseModel):
    """
    A single fact promoted to the Active Whiteboard by the BSC.

    Entries are written by the ``WhiteboardStore`` and are immutable
    after creation.  The ``id`` is assigned by SQLite on insert.
    """

    model_config = {"frozen": True}

    id: Optional[int] = None
    """Row ID assigned by the store.  None until persisted."""

    session_id: str
    """Identifier of the working session this entry belongs to."""

    source_agent: str
    """Agent that produced the chunk.
    Convention: ``"agent/<role>-<YYYYMMDD>"``, e.g. ``"agent/coder-20260326"``."""

    chunk: str
    """The promoted text chunk — the actual information being recorded."""

    surprise_score: float = Field(ge=0.0, le=1.0)
    """Normalised surprise score from the BSC predictor."""

    raw_perplexity: float = Field(ge=0.0)
    """Raw perplexity value from the predictor model."""

    similarity_score: float = Field(ge=0.0, le=1.0)
    """Max cosine similarity with previously promoted entries at time of write."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    kl_reason: str = ""
    """Human-readable reason the KL filter promoted this chunk."""

    dedup_reason: str = ""
    """Human-readable reason the semantic de-duplicator accepted this chunk."""

    extra: Dict[str, Any] = Field(default_factory=dict)
    """Arbitrary provenance metadata (git branch, skill name, etc.)."""

    @property
    def agent_short(self) -> str:
        """Display name: strips ``"agent/"`` prefix for compact rendering."""
        return self.source_agent.removeprefix("agent/")

    @property
    def timestamp_short(self) -> str:
        """ISO-8601 timestamp truncated to seconds."""
        return self.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


class WhiteboardSnapshot(BaseModel):
    """
    Complete, point-in-time view of a session's whiteboard.

    Produced by ``WhiteboardQuery.snapshot()`` and consumed by the Nightly
    Synthesis agent (Sub-phase 10.4) to generate the Project Ledger entry.
    """

    model_config = {"frozen": True}

    session_id: str
    entries: List[WhiteboardEntry] = Field(default_factory=list)
    entry_count: int = 0
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    @model_validator(mode="after")
    def _sync_count(self) -> "WhiteboardSnapshot":
        # pydantic frozen models require object.__setattr__ for post-init mutation
        object.__setattr__(self, "entry_count", len(self.entries))
        return self

    @property
    def agents(self) -> List[str]:
        """Sorted list of unique agent identifiers in this snapshot."""
        return sorted({e.source_agent for e in self.entries})

    def entries_by_agent(self, source_agent: str) -> List[WhiteboardEntry]:
        return [e for e in self.entries if e.source_agent == source_agent]

    def top_surprise(self, n: int = 5) -> List[WhiteboardEntry]:
        """Return the *n* entries with the highest surprise scores."""
        return sorted(self.entries, key=lambda e: e.surprise_score, reverse=True)[:n]


class SessionRecord(BaseModel):
    """Metadata row for a session in the ``sessions`` table."""

    model_config = {"frozen": True}

    session_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_updated: Optional[datetime] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
