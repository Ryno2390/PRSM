"""One-shot registration wrapper for the SFO droplet operator.

Hardcodes the SFO droplet's node_id + pubkey (sprint 675, identity
generated on 2026-05-21) so the user only needs to pass one secret:
the funded Base EOA private key for the anchor.register TX.

Usage (single line):
    PRSM_DEPLOYER_PRIVATE_KEY=0x<your-funded-key> python3 \\
        /Users/ryneschultz/Documents/GitHub/PRSM/scripts/register_sfo_droplet_2026_05_21.py

OR (if the env-var prefix gives you trouble):
    python3 /path/to/register_sfo_droplet_2026_05_21.py 0x<your-funded-key>

Either form works — the key is read from argv[1] if env is unset.

This wrapper just sets the two env vars expected by sprint 675's
generic registration script + delegates. Idempotent: skips re-
registration if the node_id already has a pubkey on the anchor.
"""
from __future__ import annotations

import os
import sys


# Sprint 675 — SFO droplet (146.190.175.239, DO Basic 1GB/$6/mo)
# Identity generated 2026-05-21
SFO_NODE_ID = "d437aa67d99cff4a6a17179f5c731b77"
SFO_PUBKEY_B64 = "EsnFYACpOfH2MSp97GmJ8gt7qoW1T+3MM50evnjr0jg="


def main() -> int:
    # Accept private key from env OR argv[1]; whichever is present
    pk_from_env = (os.environ.get("PRSM_DEPLOYER_PRIVATE_KEY", "") or "").strip()
    pk_from_argv = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    pk = pk_from_env or pk_from_argv
    if not pk:
        print(
            "ERROR: deployer private key not supplied.\n"
            "Pass it via env:  PRSM_DEPLOYER_PRIVATE_KEY=0x... python3 "
            "register_sfo_droplet_2026_05_21.py\n"
            "Or via argv:      python3 register_sfo_droplet_2026_05_21.py "
            "0x<your-key>",
            file=sys.stderr,
        )
        return 1
    os.environ["PRSM_DEPLOYER_PRIVATE_KEY"] = pk
    os.environ["OPERATOR_NODE_ID"] = SFO_NODE_ID
    os.environ["OPERATOR_PUBKEY_B64"] = SFO_PUBKEY_B64

    # Make the project root importable so the generic script can pull
    # in prsm.security.publisher_key_anchor.client.
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Delegate. Import via spec so we don't need scripts to be a package.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sprint_675_register_operator_pubkey",
        os.path.join(here, "sprint_675_register_operator_pubkey.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
