"""CLI: prsm provenance register|info|transfer

Phase 1 on-chain provenance commands. Requires:
  PRSM_PROVENANCE_REGISTRY_ADDRESS
  PRSM_BASE_RPC_URL              (defaults to https://mainnet.base.org)
  FTNS_WALLET_PRIVATE_KEY        (for register/transfer; not for info)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import click


def _read_file_bytes(path: Path) -> bytes:
    """Read file bytes for hashing. Streams in 64KB chunks."""
    chunks = []
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            chunks.append(chunk)
    return b"".join(chunks)


def _content_hash_for(path: Path, creator_address: str) -> bytes:
    """Canonical content hash: keccak256(creator_address || sha3_256(file_bytes)).

    Bound to a specific creator so the registration is unforgeable: a
    different creator cannot register the same file as themselves and
    pre-empt the rightful owner. Phase 1.1 Task 3.
    """
    from prsm.economy.web3.provenance_registry import compute_content_hash

    return compute_content_hash(creator_address, _read_file_bytes(path))


def _make_client():
    from prsm.economy.web3.provenance_registry import ProvenanceRegistryClient

    addr = os.getenv("PRSM_PROVENANCE_REGISTRY_ADDRESS")
    if not addr:
        click.echo("error: PRSM_PROVENANCE_REGISTRY_ADDRESS not set", err=True)
        sys.exit(1)
    rpc = os.getenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
    pk = os.getenv("FTNS_WALLET_PRIVATE_KEY")
    return ProvenanceRegistryClient(rpc_url=rpc, contract_address=addr, private_key=pk)


@click.group("provenance")
def provenance() -> None:
    """On-chain provenance registry commands (Base mainnet)."""


@provenance.command("register")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--royalty-bps",
    type=click.IntRange(0, 9800),  # MAX_ROYALTY_RATE_BPS — Phase 1.1 Task 1
    default=800,
    show_default=True,
    help="Royalty rate in basis points (max 9800; the distributor takes a fixed 200 bps network fee).",
)
@click.option(
    "--metadata-uri",
    type=str,
    default="",
    help="Off-chain metadata URI (ipfs://, https://, etc.). Keep this short — every payment reads the registry struct.",
)
def register(file: Path, royalty_bps: int, metadata_uri: str) -> None:
    """Register FILE in the on-chain provenance registry."""
    client = _make_client()
    if client.address is None:
        click.echo("error: FTNS_WALLET_PRIVATE_KEY required to register", err=True)
        sys.exit(1)

    content_hash = _content_hash_for(file, client.address)
    click.echo(f"file:        {file}")
    click.echo(f"hash:        0x{content_hash.hex()}")
    click.echo(f"creator:     {client.address}")
    click.echo(f"royalty:     {royalty_bps} bps ({royalty_bps / 100:.2f}%)")
    click.echo(f"metadata:    {metadata_uri or '(none)'}")

    if client.is_registered(content_hash):
        click.echo("note: already registered, no-op", err=True)
        sys.exit(0)

    result = client.register_content(content_hash, royalty_bps, metadata_uri)
    if isinstance(result, tuple):
        tx_hash, status = result
        status_str = getattr(status, "value", str(status))
    else:
        tx_hash = result
        status_str = "confirmed"
    click.echo(f"tx:          {tx_hash}")
    click.echo(f"status:      {status_str}")
    click.echo("")
    click.echo("Save this hash for upload metadata:")
    click.echo(f"  provenance_hash = 0x{content_hash.hex()}")


@provenance.command("info")
@click.argument("hash_or_file")
@click.option(
    "--creator",
    type=str,
    default="",
    help="Creator address (used when looking up by file path; defaults to the connected wallet).",
)
def info(hash_or_file: str, creator: str) -> None:
    """Show provenance record for a content hash (0x… 32 bytes) or file path.

    When given a file path, the canonical hash is
    keccak256(creator_address || sha3_256(file_bytes)). The creator
    defaults to the connected wallet — pass --creator to look up
    someone else's registration of the same file.
    """
    client = _make_client()

    if hash_or_file.startswith("0x") and len(hash_or_file) == 66:
        content_hash = bytes.fromhex(hash_or_file[2:])
    else:
        p = Path(hash_or_file)
        if not p.exists():
            click.echo(
                f"error: not a hash and not an existing file: {hash_or_file}",
                err=True,
            )
            sys.exit(1)
        if not creator:
            if client.address is None:
                click.echo(
                    "error: --creator is required for file lookups when no wallet is connected",
                    err=True,
                )
                sys.exit(1)
            creator = client.address
        content_hash = _content_hash_for(p, creator)

    rec = client.get_content(content_hash)
    if rec is None:
        click.echo(f"hash 0x{content_hash.hex()} is NOT registered")
        sys.exit(0)

    click.echo(f"hash:        0x{content_hash.hex()}")
    click.echo(f"creator:     {rec.creator}")
    click.echo(f"royalty:     {rec.royalty_rate_bps} bps ({rec.royalty_rate_bps / 100:.2f}%)")
    click.echo(f"registered:  {rec.registered_at} (unix)")
    click.echo(f"metadata:    {rec.metadata_uri or '(none)'}")


@provenance.command("transfer")
@click.argument("content_hash")
@click.argument("new_creator")
def transfer(content_hash: str, new_creator: str) -> None:
    """Transfer creator role for CONTENT_HASH (0x…) to NEW_CREATOR (0x…)."""
    client = _make_client()
    if client.address is None:
        click.echo("error: FTNS_WALLET_PRIVATE_KEY required to transfer", err=True)
        sys.exit(1)

    if not content_hash.startswith("0x") or len(content_hash) != 66:
        click.echo("error: content_hash must be 0x followed by 64 hex chars", err=True)
        sys.exit(1)
    if not new_creator.startswith("0x") or len(new_creator) != 42:
        click.echo("error: new_creator must be a 0x address", err=True)
        sys.exit(1)

    result = client.transfer_ownership(bytes.fromhex(content_hash[2:]), new_creator)
    if isinstance(result, tuple):
        tx_hash, status = result
        click.echo(f"tx:     {tx_hash}")
        click.echo(f"status: {getattr(status, 'value', status)}")
    else:
        click.echo(f"tx:     {result}")
