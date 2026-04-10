"""CLI: prsm provenance register|info|transfer

Phase 1 on-chain provenance commands. Requires:
  PRSM_PROVENANCE_REGISTRY_ADDRESS
  PRSM_BASE_RPC_URL              (defaults to https://mainnet.base.org)
  FTNS_WALLET_PRIVATE_KEY        (for register/transfer; not for info)
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import click


def _content_hash_for(path: Path) -> bytes:
    """Canonical content hash: sha3-256 of file bytes (matches contract bytes32)."""
    h = hashlib.sha3_256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.digest()


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
    type=click.IntRange(0, 10000),
    default=800,
    show_default=True,
    help="Royalty rate in basis points (800 = 8%).",
)
@click.option(
    "--metadata-uri",
    type=str,
    default="",
    help="Off-chain metadata URI (ipfs://, https://, etc.).",
)
def register(file: Path, royalty_bps: int, metadata_uri: str) -> None:
    """Register FILE in the on-chain provenance registry."""
    client = _make_client()
    if client.address is None:
        click.echo("error: FTNS_WALLET_PRIVATE_KEY required to register", err=True)
        sys.exit(1)

    content_hash = _content_hash_for(file)
    click.echo(f"file:        {file}")
    click.echo(f"hash:        0x{content_hash.hex()}")
    click.echo(f"creator:     {client.address}")
    click.echo(f"royalty:     {royalty_bps} bps ({royalty_bps / 100:.2f}%)")
    click.echo(f"metadata:    {metadata_uri or '(none)'}")

    if client.is_registered(content_hash):
        click.echo("note: already registered, no-op", err=True)
        sys.exit(0)

    tx = client.register_content(content_hash, royalty_bps, metadata_uri)
    click.echo(f"tx:          {tx}")


@provenance.command("info")
@click.argument("hash_or_file")
def info(hash_or_file: str) -> None:
    """Show provenance record for a content hash (0x… 32 bytes) or file path."""
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
        content_hash = _content_hash_for(p)

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

    tx = client.transfer_ownership(bytes.fromhex(content_hash[2:]), new_creator)
    click.echo(f"tx: {tx}")
