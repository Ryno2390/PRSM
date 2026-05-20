"""Sprint 620 — Phase 3.x.3 Path B Safe deploy payload generator.

Generates the calldata + predicted CREATE2 address for deploying
PublisherKeyAnchor from the Foundation Safe via the Singleton
Factory (Arachnid's Deterministic Deployment Proxy at
0xce0042B868300000d44A59004Da54A005ffdcf9f — deployed on Base
mainnet + most EVM chains).

Safe TX to enqueue at app.safe.global:
  To:        0xce0042B868300000d44A59004Da54A005ffdcf9f
  Value:     0
  Data:      <salt 32 bytes> || <bytecode> || <abi-encoded ctor args>
  Operation: 0 (Call)

Predicted contract address (deterministic from factory + salt +
keccak256(initcode)):
  address = last 20 bytes of keccak256(0xff || factory || salt || keccak256(initcode))

Salt is fixed to bytes32(0) — operator can pick any uint256 if a
collision becomes relevant (none expected; the PublisherKeyAnchor
initcode is unique to this Foundation Safe admin address).

Run:
  python3 scripts/generate_safe_deploy_payload.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


SINGLETON_FACTORY = "0xce0042B868300000d44A59004Da54A005ffdcf9f"
FOUNDATION_SAFE = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
SALT_HEX = "0x" + "00" * 32  # bytes32(0)


def _hex_to_bytes(h: str) -> bytes:
    return bytes.fromhex(h[2:] if h.startswith("0x") else h)


def _abi_encode_address(addr: str) -> bytes:
    """ABI-encode an address: left-pad to 32 bytes."""
    if addr.startswith("0x"):
        addr = addr[2:]
    if len(addr) != 40:
        raise ValueError(f"address must be 20 bytes (40 hex chars); got {len(addr)}")
    return bytes.fromhex(addr).rjust(32, b"\x00")


def _keccak256(data: bytes) -> bytes:
    # Solidity keccak256, NOT sha3-256. pip install eth-hash[pycryptodome] or use a wrapper.
    try:
        from Crypto.Hash import keccak
        k = keccak.new(digest_bits=256)
        k.update(data)
        return k.digest()
    except ImportError:
        try:
            from eth_hash.auto import keccak as _eth_keccak
            return _eth_keccak(data)
        except ImportError:
            # Fallback — pysha3 if available
            import sha3
            k = sha3.keccak_256()
            k.update(data)
            return k.digest()


def main() -> int:
    repo = Path(__file__).parent.parent
    artifact_path = (
        repo / "contracts" / "artifacts" / "contracts"
        / "PublisherKeyAnchor.sol" / "PublisherKeyAnchor.json"
    )
    if not artifact_path.exists():
        print(
            f"ERROR: artifact not found at {artifact_path}\n"
            f"Run `cd contracts && npx hardhat compile` first.",
            file=sys.stderr,
        )
        return 1
    artifact = json.loads(artifact_path.read_text())

    bytecode_hex = artifact["bytecode"]
    if not bytecode_hex.startswith("0x"):
        bytecode_hex = "0x" + bytecode_hex
    bytecode = _hex_to_bytes(bytecode_hex)

    # PublisherKeyAnchor constructor takes (address _admin)
    ctor_args = _abi_encode_address(FOUNDATION_SAFE)
    initcode = bytecode + ctor_args
    initcode_hash = _keccak256(initcode)

    salt = _hex_to_bytes(SALT_HEX)
    factory = _hex_to_bytes(SINGLETON_FACTORY)

    # CREATE2 address: last 20 bytes of keccak256(0xff || factory || salt || keccak256(initcode))
    predicted = _keccak256(b"\xff" + factory + salt + initcode_hash)[-20:]
    predicted_hex = "0x" + predicted.hex()

    # Calldata to factory = salt || initcode
    calldata = salt + initcode
    calldata_hex = "0x" + calldata.hex()

    print("=" * 70)
    print("Phase 3.x.3 Path B — Foundation Safe deploy payload")
    print("=" * 70)
    print()
    print("Paste these into app.safe.global → New Transaction → Contract Interaction")
    print("(or 'Custom Transaction' depending on your Safe wallet UI version):")
    print()
    print(f"  To:         {SINGLETON_FACTORY}")
    print( "              (Arachnid's Deterministic Deployment Proxy on Base mainnet)")
    print( "  Value:      0")
    print(f"  Data:       {calldata_hex[:80]}...")
    print(f"              ({len(calldata)} bytes total — copy full hex from below)")
    print( "  Operation:  Call (NOT DelegateCall)")
    print()
    print("Full data (single line, copy verbatim):")
    print(f"  {calldata_hex}")
    print()
    print("─" * 70)
    print("Predicted contract address (deterministic CREATE2):")
    print(f"  {predicted_hex}")
    print("─" * 70)
    print()
    print("Post-execution verification:")
    print(f"  1. Check {predicted_hex} exists on Basescan (https://basescan.org/address/" + predicted_hex + ")")
    print( "  2. Verify contract code matches PublisherKeyAnchor.sol (use Hardhat verify or Basescan flatten)")
    print(f"  3. Call .admin() — must return {FOUNDATION_SAFE} (Foundation Safe)")
    print(f"  4. Update prsm/config/networks.py: publisher_key_anchor='{predicted_hex}' on Base mainnet")
    print( "  5. Run `prsm node anchor-probe` — should report ok")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
