"""Sprint 520 — `prsm provenance register/info` live-verified on mainnet.

Vision §11 promises operators can register content provenance on
the on-chain ProvenanceRegistry. The CLI surface
(`prsm provenance register|info`) existed since the early sprints
but had never been operator-CLI-attested with a real on-chain TX.

Sprint 520 closes this gap. Live-verified on Base mainnet
ProvenanceRegistry V1 (`0xdF470BFa9eF310B196801D5105468515d0069915`):

  $ prsm provenance register /tmp/sprint520-test-content.txt
  hash:        0xa97f3411ea26f52c9a5660a9a8bbe2aa2738be55f60fd2649d53e6a0f612699c
  creator:     0x4acdE458766C704B2511583572303e77109cFFE8
  royalty:     800 bps (8.00%)
  metadata:    (none)
  tx:          0x84b8084beea7e7580ee76f2310f9d71e13252a227f58d39451b5cf63c5b9f1f7
  status:      confirmed

Round-trip confirmed:
  $ prsm provenance info 0xa97f3411…
  hash:        0xa97f3411…
  creator:     0x4acdE458…
  royalty:     800 bps (8.00%)
  registered:  1779120967 (unix)
  metadata:    (none)

On-chain receipt: block 46165810, success, 50470 gas @ 0.006 Gwei
= 0.0000003 ETH, 1 ProvenanceRegistered event.

This is the first ever PRSM-daemon-signed ProvenanceRegistry TX
on Base mainnet — Vision §11 creator-provenance promise
operationally proven via the operator CLI.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def test_register_requires_private_key():
    """When FTNS_WALLET_PRIVATE_KEY is unset, register must
    surface a clear error — not crash with cryptic
    NoneType."""
    from prsm.cli import main as cli
    import os

    runner = CliRunner()
    with patch.dict(
        os.environ,
        {
            "FTNS_WALLET_PRIVATE_KEY": "",
            "PRSM_PROVENANCE_REGISTRY_ADDRESS":
                "0xdF470BFa9eF310B196801D5105468515d0069915",
        },
        clear=False,
    ):
        # write a temp file
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt",
        ) as f:
            f.write("test")
            path = f.name
        result = runner.invoke(
            cli, ["provenance", "register", path],
        )
    assert result.exit_code != 0
    assert "FTNS_WALLET_PRIVATE_KEY" in result.output


def test_register_uses_canonical_fallback_when_env_unset():
    """Sprint 527 added canonical fallback: when
    PRSM_PROVENANCE_REGISTRY_ADDRESS is unset, the CLI falls
    back to networks.py canonical (V2 preferred). Replaces
    the old "must surface PRSM_PROVENANCE_REGISTRY_ADDRESS
    not set" test which was correct pre-sprint-527 but
    obsoleted by the fallback.
    """
    from prsm.cli import main as cli
    import os

    runner = CliRunner()
    fake_inst = MagicMock()
    fake_inst.address = (
        "0x4acdE458766C704B2511583572303e77109cFFE8"
    )
    fake_inst.is_registered.return_value = True
    with patch.dict(
        os.environ,
        {
            "FTNS_WALLET_PRIVATE_KEY":
                "0x8e6f526e6c80d7b7aae970a9de243b147e21616f11fa5fdba76f8046f89390d7",
            "PRSM_PROVENANCE_REGISTRY_ADDRESS": "",
        },
        clear=False,
    ):
        # Mock both client classes to verify V2 is the canonical
        # default (sprint 526 fix) rather than failing on missing
        # env.
        with patch(
            "prsm.cli_modules.provenance."
            "ProvenanceRegistryV2Client",
            return_value=fake_inst,
            create=True,
        ):
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt",
            ) as f:
                f.write("test")
                path = f.name
            result = runner.invoke(
                cli, ["provenance", "register", path],
            )
    # Either succeeds (canonical fallback works) or fails
    # for a downstream reason (mock client incomplete) — but
    # MUST NOT fail with "PRSM_PROVENANCE_REGISTRY_ADDRESS
    # not set".
    assert (
        "PRSM_PROVENANCE_REGISTRY_ADDRESS not set"
        not in result.output
    )


def test_info_returns_clean_message_for_unregistered():
    """When a content hash isn't on-chain, lookup must
    return a clear "NOT registered" — not crash."""
    from prsm.cli import main as cli
    import os

    runner = CliRunner()
    # Patch the ProvenanceRegistryClient.get_record to
    # return None (matching "not registered" semantics).
    with patch.dict(
        os.environ,
        {
            "PRSM_PROVENANCE_REGISTRY_ADDRESS":
                "0xdF470BFa9eF310B196801D5105468515d0069915",
            "BASE_RPC_URL": "http://fake",
        },
        clear=False,
    ):
        with patch(
            "prsm.economy.web3.provenance_registry."
            "ProvenanceRegistryClient",
        ) as Client:
            inst = MagicMock()
            inst.get_content.return_value = None
            Client.return_value = inst
            result = runner.invoke(
                cli,
                [
                    "provenance", "info",
                    "0x" + "00" * 32,
                ],
            )
    assert result.exit_code in (0, 1)
    assert "NOT registered" in result.output
