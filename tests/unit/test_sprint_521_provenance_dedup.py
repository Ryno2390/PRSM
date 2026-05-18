"""Sprint 521 — `prsm provenance register` dedup pre-flight check.

Sprint 520 registered content hash 0xa97f3411… on Base mainnet
ProvenanceRegistry V1. Sprint 521 verifies the dedup pre-flight
check: re-attempting registration of the same hash returns
"note: already registered, no-op" WITHOUT spending gas on a
revert.

Live-verified on Base mainnet operator wallet 0x4acdE458…:

  $ prsm provenance register /tmp/sprint520-test-content.txt
  hash:        0xa97f3411…  (same hash as sprint 520)
  creator:     0x4acdE458…
  royalty:     800 bps (8.00%)
  metadata:    (none)
  note: already registered, no-op   ← NEW: pre-flight detection

No TX broadcast — `get_content()` is called first and returns
the existing record. Operator avoids wasting gas on a revert.

This is the desired UX for the "double-register thinking the
first failed" production class — silent failure to revert
would burn ~20k gas per attempt.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def test_register_short_circuits_if_already_registered():
    """When get_content returns an existing record, the
    CLI must NOT send a TX. The output must surface
    "no-op" so the operator knows it's intentional."""
    from prsm.cli import main as cli
    import os

    runner = CliRunner()

    with patch.dict(
        os.environ,
        {
            "FTNS_WALLET_PRIVATE_KEY": "0x8e6f526e6c80d7b7aae970a9de243b147e21616f11fa5fdba76f8046f89390d7",
            "PRSM_PROVENANCE_REGISTRY_ADDRESS":
                "0xdF470BFa9eF310B196801D5105468515d0069915",
            "BASE_RPC_URL": "http://fake",
        },
        clear=False,
    ):
        with patch(
            "prsm.cli_modules.provenance._make_client",
        ) as make_client:
            inst = MagicMock()
            inst.address = (
                "0x626908c2572eBE841Bdf4401bfb9a4cF630F91D2"
            )
            inst.is_registered.return_value = True
            inst.register_content.side_effect = RuntimeError(
                "register_content called when already registered!"
            )
            make_client.return_value = inst
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt",
            ) as f:
                f.write("sprint 521 test content")
                path = f.name
            result = runner.invoke(
                cli, ["provenance", "register", path],
            )

    assert result.exit_code == 0, result.output
    assert "already registered" in result.output.lower()
    assert "no-op" in result.output.lower()


def test_register_proceeds_when_not_yet_registered():
    """When get_content returns None, register_content
    must be called and the resulting tx_hash surfaced."""
    from prsm.cli import main as cli
    import os

    runner = CliRunner()

    with patch.dict(
        os.environ,
        {
            "FTNS_WALLET_PRIVATE_KEY": "0x8e6f526e6c80d7b7aae970a9de243b147e21616f11fa5fdba76f8046f89390d7",
            "PRSM_PROVENANCE_REGISTRY_ADDRESS":
                "0xdF470BFa9eF310B196801D5105468515d0069915",
            "BASE_RPC_URL": "http://fake",
        },
        clear=False,
    ):
        with patch(
            "prsm.cli_modules.provenance._make_client",
        ) as make_client:
            inst = MagicMock()
            inst.address = (
                "0x626908c2572eBE841Bdf4401bfb9a4cF630F91D2"
            )
            inst.is_registered.return_value = False
            inst.register_content.return_value = "0x" + "ab" * 32
            make_client.return_value = inst
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt",
            ) as f:
                f.write("sprint 521 fresh content")
                path = f.name
            result = runner.invoke(
                cli, ["provenance", "register", path],
            )

    assert result.exit_code == 0, result.output
    assert "0x" + "ab" * 32 in result.output
    assert "confirmed" in result.output
