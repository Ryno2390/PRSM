#!/usr/bin/env python
"""
Run Harness Optimizer on existing traces and propose next config.

This script loads history from .nwtn_traces/ and uses the HarnessOptimizer
to propose an improved configuration for the next session.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.compute.nwtn.harness_optimizer import HarnessOptimizer


def main():
    parser = argparse.ArgumentParser(description="Run harness optimizer and propose next config")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path(".nwtn_traces"),
        help="Directory containing trace sessions (default: .nwtn_traces)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for proposed config (default: {traces_dir}/proposals)",
    )
    args = parser.parse_args()

    traces_dir = args.traces_dir
    output_dir = args.output_dir or (traces_dir / "proposals")

    # 1. Load history
    print(f"Loading traces from: {traces_dir}")
    optimizer = HarnessOptimizer(traces_dir=traces_dir)
    history = optimizer.load_history()

    # 2. Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(optimizer.summarize(history))

    # 3. Print prompt context (first 2000 chars)
    print("\n" + "=" * 60)
    print("PROMPT CONTEXT (first 2000 chars)")
    print("=" * 60)
    prompt_context = history.to_prompt_context()
    print(prompt_context[:2000])
    if len(prompt_context) > 2000:
        print(f"\n... [truncated, {len(prompt_context)} total chars]")

    # 4. Propose next config
    print("\n" + "=" * 60)
    print("PROPOSED NEXT CONFIG")
    print("=" * 60)
    proposal = optimizer.propose_next_config(history, goal="Optimize distributed consensus algorithm")
    print(json.dumps(proposal.to_dict(), indent=2))

    # 5. Save proposal
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "next_config.json"
    output_path.write_text(json.dumps(proposal.to_dict(), indent=2))
    print(f"\nProposal saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
