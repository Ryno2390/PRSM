#!/usr/bin/env python3
"""Sprint 378 — operator-run script to regenerate the
source-identity pin registry.

USE WITH CARE: this re-pins ALL citations to their CURRENT
canonical-source hashes. Before running, verify each spec
contract still semantically mirrors its cited canonical
source. Pin updates without spec re-verification defeat
the purpose of the parity gate.

Workflow:
  1. Canonical source legitimately changes (e.g., refactor)
  2. Update each affected spec to match
  3. Re-run halmos to confirm proofs still pass
  4. Run this script to refresh the pin file
  5. Commit the spec changes + pin update together

Usage:
  python scripts/update_source_identity_pins.py [--dry-run]
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from prsm.economy.web3.source_identity import (
        hash_canonical_range,
        save_pins,
        scan_specs_dir,
    )

    specs_dir = (
        repo_root / "contracts" / "symbolic-proofs" / "test"
    )
    pins_path = (
        repo_root
        / "contracts" / "symbolic-proofs"
        / "source_identity_pins.json"
    )
    dry_run = "--dry-run" in sys.argv

    citations = scan_specs_dir(specs_dir)
    pins: dict[str, str] = {}
    skipped: list[str] = []
    for cit in citations:
        h = hash_canonical_range(cit, repo_root=repo_root)
        if h is None:
            skipped.append(cit.key)
            continue
        pins[cit.key] = h

    print(f"Source-identity pin update — {len(pins)} pin(s):")
    for key, h in sorted(pins.items()):
        print(f"  {key:60s} {h[:16]}...")
    if skipped:
        print(f"\nSKIPPED ({len(skipped)} citations w/o source):")
        for key in skipped:
            print(f"  {key}")

    if dry_run:
        print("\n(dry-run; not writing)")
        return 0
    save_pins(pins, pins_path)
    print(f"\nWrote pins to {pins_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
