"""Sprint 318 — operator deployment CLI.

`prsm-enterprise-bringup status` — prints the current
config state (every env var, set/missing/required).
`prsm-enterprise-bringup generate` — emits a complete
.env file template with fresh keypairs.

Run from the shell via:
  python -m prsm.enterprise.bringup_cli status
  python -m prsm.enterprise.bringup_cli generate > .env
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from prsm.enterprise.bringup import (
    EnterpriseConfig,
    EnterpriseConfigValidationError,
    generate_starter_config,
    render_env_file,
)


def _cmd_status(_args: argparse.Namespace) -> int:
    cfg = EnterpriseConfig.from_env()
    sys.stdout.write(cfg.summary())
    sys.stdout.write("\n\n")
    try:
        cfg.validate()
        sys.stdout.write(
            "✓ Deployment validates: all required env "
            "vars set and well-formed.\n"
        )
        return 0
    except EnterpriseConfigValidationError as e:
        sys.stdout.write(
            f"✗ Deployment is incomplete: {e}\n",
        )
        sys.stdout.write(
            "  Run `prsm-enterprise-bringup generate > "
            ".env` for a fresh starter config.\n"
        )
        return 1


def _cmd_generate(args: argparse.Namespace) -> int:
    starter = generate_starter_config(
        base_dir=args.base_dir,
    )
    sys.stdout.write(render_env_file(starter))
    return 0


def _cmd_health(_args: argparse.Namespace) -> int:
    """Sprint 318c — runtime health check. Wraps
    `run_health_checks()` to print the summary + return
    rc=0 only if every check passes (so CI / deploy
    scripts can fail-fast)."""
    from prsm.enterprise.bringup_health import (
        run_health_checks,
    )
    outcome = run_health_checks()
    sys.stdout.write(outcome.summary())
    sys.stdout.write("\n")
    return 0 if outcome.ok else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    """Sprint 318b — run the end-to-end demo against a
    fresh in-process deployment. Useful as a post-deploy
    smoke test."""
    import tempfile
    from pathlib import Path as _Path

    from prsm.enterprise.demo import (
        run_full_demo,
        setup_demo_environment,
    )

    persist_root = (
        _Path(args.persist_root)
        if args.persist_root is not None
        else _Path(tempfile.mkdtemp(prefix="prsm-demo-"))
    )
    sys.stdout.write(
        f"PRSM Enterprise end-to-end demo — "
        f"persist_root={persist_root}\n\n"
    )
    env = setup_demo_environment(
        persist_root=persist_root,
    )
    results = run_full_demo(env)
    return 0 if all(r.ok for _, r in results) else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prsm-enterprise-bringup",
        description=(
            "PRSM Enterprise deployment bringup helper. "
            "Status / generate-starter / future "
            "subcommands."
        ),
    )
    sub = parser.add_subparsers(
        dest="command", required=True,
    )

    sub.add_parser(
        "status",
        help=(
            "Print the current deployment's env-var "
            "state and validation result."
        ),
    )

    gen = sub.add_parser(
        "generate",
        help=(
            "Emit a complete .env file template with "
            "fresh keypairs + sensible defaults."
        ),
    )
    gen.add_argument(
        "--base-dir", default=None,
        help=(
            "Base filesystem path for persistence dirs "
            "(default /var/lib/prsm). All directory env "
            "vars get values like {base-dir}/fl, "
            "{base-dir}/pipeline, etc."
        ),
    )

    sub.add_parser(
        "health",
        help=(
            "Runtime health check: persistence dirs "
            "writable + keypairs load + subsystems "
            "wire ok. Returns rc=0 only if every check "
            "passes."
        ),
    )

    demo = sub.add_parser(
        "demo",
        help=(
            "Run the §7 enterprise FL + federated "
            "inference demo against a fresh in-process "
            "deployment. Useful as a post-deploy smoke "
            "test."
        ),
    )
    demo.add_argument(
        "--persist-root", default=None,
        help=(
            "Base filesystem path for demo persistence "
            "(default: fresh temp dir per run)."
        ),
    )

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse exits on unknown commands; surface as
        # non-zero RC for the caller
        return int(e.code) if e.code is not None else 1
    if args.command == "status":
        return _cmd_status(args)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "health":
        return _cmd_health(args)
    if args.command == "demo":
        return _cmd_demo(args)
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
