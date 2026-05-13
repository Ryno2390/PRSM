"""Sprint 378 — source-identity CI parity check.

Halmos symbolic specs cite canonical source line ranges in
their STRUCTURAL EQUIVALENCE blocks (e.g.,
``prsm/compute/chain_rpc/client.py:1431``). When canonical
source mutates, the cited spec must be re-validated by an
engineer — otherwise the symbolic proof silently drifts
out of sync with what it claims to mirror.

This module provides the machinery:

  parse_citations(spec_text)       — extract (path, start, end)
  hash_canonical_range(path, ...)  — SHA-256 of the line range
  scan_specs_dir(dir)              — citations from all .t.sol files
  load_pins(path) / save_pins(...) — JSON pin registry I/O

Plus a `verify_parity` entry point used by CI: returns
PASS/FAIL with the list of drifted citations.

The §7.34 honest-scope was explicit about this: "Source-
identity CI parity check between spec contracts + canonical
source — would catch silent drift if someone modifies one
without the other." Sprint 378 closes it.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Citation regex: matches `path/to/file.{py,sol}:NNN(-NNN)?`
# anchored to either prsm/ or contracts/ to avoid false
# matches on narrative text / URLs / etc.
_CITATION_RE = re.compile(
    r"\b((?:prsm|contracts)/[a-zA-Z0-9_/.]+\.(?:py|sol))"
    r":(\d+)(?:-(\d+))?\b"
)


@dataclass(frozen=True)
class Citation:
    """One canonical-source citation from a halmos spec."""
    canonical_path: str
    line_start: int
    line_end: int

    @property
    def key(self) -> str:
        """Canonical identifier for the pin registry."""
        if self.line_start == self.line_end:
            return f"{self.canonical_path}:{self.line_start}"
        return (
            f"{self.canonical_path}:"
            f"{self.line_start}-{self.line_end}"
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "canonical_path": self.canonical_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "key": self.key,
        }


@dataclass
class ParityResult:
    """Outcome of a full-directory parity check."""
    passed: List[str] = field(default_factory=list)
    drifted: List[Tuple[str, str, str]] = field(
        default_factory=list,
    )  # (key, expected_hash, actual_hash)
    missing_source: List[str] = field(default_factory=list)
    missing_pin: List[str] = field(default_factory=list)
    out_of_range: List[Tuple[str, int]] = field(
        default_factory=list,
    )

    @property
    def ok(self) -> bool:
        return not (
            self.drifted
            or self.missing_source
            or self.missing_pin
            or self.out_of_range
        )

    def summary(self) -> str:
        return (
            f"passed={len(self.passed)} "
            f"drifted={len(self.drifted)} "
            f"missing_source={len(self.missing_source)} "
            f"missing_pin={len(self.missing_pin)} "
            f"out_of_range={len(self.out_of_range)}"
        )


def parse_citations(spec_text: str) -> List[Citation]:
    """Extract source-identity citations from a spec file.

    De-duplicates within a single file — if the same citation
    appears multiple times, it's hashed once.
    """
    seen: set = set()
    out: List[Citation] = []
    for m in _CITATION_RE.finditer(spec_text):
        path = m.group(1)
        start = int(m.group(2))
        end = int(m.group(3)) if m.group(3) else start
        key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(
            canonical_path=path,
            line_start=start,
            line_end=end,
        ))
    return out


def hash_canonical_range(
    citation: Citation, *, repo_root: Path,
) -> Optional[str]:
    """Hash the cited line range of the canonical source.

    Returns None when:
      - The canonical file doesn't exist
      - The cited line range is out of bounds (start <= 0 or
        end > file_length)

    Hash is SHA-256 of the line-range content with trailing
    whitespace stripped per line + joined with `\\n`. This
    keeps the pin stable under benign whitespace-only edits
    at end-of-line while catching any semantic change.
    """
    full_path = repo_root / citation.canonical_path
    if not full_path.is_file():
        return None
    try:
        text = full_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    lines = text.splitlines()
    if (
        citation.line_start <= 0
        or citation.line_end <= 0
        or citation.line_start > len(lines)
        or citation.line_end > len(lines)
    ):
        return None
    # Line numbers are 1-indexed in citations; convert to
    # 0-indexed slice (start-1, end inclusive).
    sliced = lines[
        citation.line_start - 1: citation.line_end
    ]
    normalized = "\n".join(line.rstrip() for line in sliced)
    return hashlib.sha256(
        normalized.encode("utf-8"),
    ).hexdigest()


def scan_specs_dir(specs_dir: Path) -> List[Citation]:
    """All citations across every .t.sol file in specs_dir,
    sorted by key for deterministic order."""
    seen: set = set()
    out: List[Citation] = []
    for spec_file in sorted(specs_dir.glob("*.t.sol")):
        text = spec_file.read_text(encoding="utf-8")
        for cit in parse_citations(text):
            if cit.key in seen:
                continue
            seen.add(cit.key)
            out.append(cit)
    return sorted(out, key=lambda c: c.key)


def load_pins(pins_path: Path) -> Dict[str, str]:
    """Load the pin registry. Returns empty dict if file
    doesn't exist (first-run case)."""
    if not pins_path.is_file():
        return {}
    try:
        data = json.loads(pins_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    pins = data.get("pins") or {}
    if not isinstance(pins, dict):
        return {}
    return {str(k): str(v) for k, v in pins.items()}


def save_pins(
    pins: Dict[str, str], pins_path: Path,
) -> None:
    """Write the pin registry. Sorted keys for deterministic
    diff under git."""
    body = {
        "version": 1,
        "pins": dict(sorted(pins.items())),
    }
    pins_path.write_text(
        json.dumps(body, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_parity(
    *,
    specs_dir: Path,
    pins_path: Path,
    repo_root: Path,
) -> ParityResult:
    """Run the full parity check.

    Returns a ParityResult with passed citations + any
    drifted / missing-source / missing-pin / out-of-range
    entries. CI gate: `result.ok` is the boolean to assert.
    """
    result = ParityResult()
    pins = load_pins(pins_path)
    citations = scan_specs_dir(specs_dir)
    for cit in citations:
        actual = hash_canonical_range(
            cit, repo_root=repo_root,
        )
        if actual is None:
            full_path = repo_root / cit.canonical_path
            if not full_path.is_file():
                result.missing_source.append(cit.key)
            else:
                # File exists but range was out of bounds
                try:
                    n_lines = len(
                        full_path.read_text(
                            encoding="utf-8",
                        ).splitlines()
                    )
                except Exception:  # noqa: BLE001
                    n_lines = 0
                result.out_of_range.append(
                    (cit.key, n_lines),
                )
            continue
        expected = pins.get(cit.key)
        if expected is None:
            result.missing_pin.append(cit.key)
            continue
        if expected != actual:
            result.drifted.append(
                (cit.key, expected, actual),
            )
            continue
        result.passed.append(cit.key)
    return result
