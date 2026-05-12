"""Sprint 304 — Enterprise CLI helper.

Lets an operator use the recipient-encryption primitives
without involving a running PRSM node. The encryption /
decryption flows are fully client-side and air-gappable —
the private key never has to touch a server.

Subcommands:
  keypair-gen        emit (privkey_b64, pubkey_b64)
  encrypt            seal stdin to a recipient list
                     (--recipients "id1:pubkey1,id2:pubkey2")
  decrypt            unseal stdin bundle with --privkey

Stdin/stdout binary-safe: encrypt reads raw bytes from
stdin, writes a JSON bundle to stdout; decrypt is the
mirror. The JSON bundle is the same shape returned by
the `encrypt_for_recipients` primitive.
"""
from __future__ import annotations

import argparse
import json
import sys

from prsm.enterprise.recipient_encryption import (
    EncryptedPayload,
    EnterpriseRecipient,
    decrypt_for_recipient,
    encrypt_for_recipients,
    generate_recipient_keypair,
)


def _cmd_keypair_gen(_args: argparse.Namespace) -> int:
    priv, pub = generate_recipient_keypair()
    sys.stdout.write(
        f"privkey_b64={priv}\npubkey_b64={pub}\n"
    )
    return 0


def _parse_recipients(spec: str) -> list[EnterpriseRecipient]:
    """Parse 'id1:pubkey1,id2:pubkey2' into recipient list."""
    out: list[EnterpriseRecipient] = []
    if not spec:
        raise SystemExit(
            "encrypt requires --recipients "
            "'id1:pubkey1,id2:pubkey2'"
        )
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise SystemExit(
                f"recipient {piece!r} missing ':' "
                f"between identifier and pubkey"
            )
        ident, pub = piece.split(":", 1)
        out.append(EnterpriseRecipient(
            identifier=ident.strip(),
            x25519_pubkey_b64=pub.strip(),
        ))
    if not out:
        raise SystemExit(
            "no recipients parsed from --recipients spec"
        )
    return out


def _cmd_encrypt(args: argparse.Namespace) -> int:
    recipients = _parse_recipients(args.recipients)
    plaintext = sys.stdin.buffer.read()
    payload = encrypt_for_recipients(plaintext, recipients)
    sys.stdout.write(json.dumps(payload.to_dict()))
    sys.stdout.write("\n")
    return 0


def _cmd_decrypt(args: argparse.Namespace) -> int:
    if not args.privkey:
        raise SystemExit("decrypt requires --privkey")
    blob = sys.stdin.read()
    payload = EncryptedPayload.from_dict(json.loads(blob))
    plaintext = decrypt_for_recipient(payload, args.privkey)
    sys.stdout.buffer.write(plaintext)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prsm-enterprise",
        description=(
            "PRSM Enterprise Confidentiality Mode CLI — "
            "recipient-encrypted upload primitives "
            "(Vision §7)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "keypair-gen",
        help="Generate an X25519 recipient keypair.",
    )

    p_enc = sub.add_parser(
        "encrypt",
        help=(
            "Encrypt stdin for a list of recipients; "
            "writes a JSON bundle to stdout."
        ),
    )
    p_enc.add_argument(
        "--recipients", required=True,
        help=(
            "Comma-separated 'identifier:pubkey_b64' "
            "pairs (e.g. 'alice@corp:AAA...,bob@corp:BBB...')."
        ),
    )

    p_dec = sub.add_parser(
        "decrypt",
        help=(
            "Decrypt a JSON bundle from stdin with the "
            "given recipient private key; writes plaintext "
            "to stdout."
        ),
    )
    p_dec.add_argument("--privkey", required=True)

    args = parser.parse_args(argv)
    if args.command == "keypair-gen":
        return _cmd_keypair_gen(args)
    if args.command == "encrypt":
        return _cmd_encrypt(args)
    if args.command == "decrypt":
        return _cmd_decrypt(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
