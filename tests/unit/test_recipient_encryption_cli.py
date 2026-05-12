"""Sprint 304 — Enterprise CLI smoke tests.

The CLI is the air-gappable path: the private key never
needs to touch a network-attached service. These tests
exercise the keypair-gen → encrypt → decrypt round trip
end-to-end via the argparse entry point.
"""
from __future__ import annotations

import io
import json

import pytest

from prsm.enterprise import cli


def _run(argv, stdin_bytes, capsys, monkeypatch):
    stdin_text = stdin_bytes.decode("latin-1")
    monkeypatch.setattr(
        "sys.stdin",
        type("S", (), {
            "buffer": io.BytesIO(stdin_bytes),
            "read": lambda self=None: stdin_text,
        })(),
    )
    rc = cli.main(argv)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def test_cli_keypair_gen(capsys, monkeypatch):
    rc, out, _ = _run(
        ["keypair-gen"], b"", capsys, monkeypatch,
    )
    assert rc == 0
    assert "privkey_b64=" in out
    assert "pubkey_b64=" in out


def test_cli_encrypt_decrypt_round_trip(
    capsys, monkeypatch,
):
    # Generate a recipient
    rc, kp_out, _ = _run(
        ["keypair-gen"], b"", capsys, monkeypatch,
    )
    priv = next(
        line.split("=", 1)[1]
        for line in kp_out.splitlines()
        if line.startswith("privkey_b64=")
    )
    pub = next(
        line.split("=", 1)[1]
        for line in kp_out.splitlines()
        if line.startswith("pubkey_b64=")
    )

    plaintext = b"top secret enterprise data"
    rc, enc_out, _ = _run(
        ["encrypt", "--recipients", f"alice:{pub}"],
        plaintext, capsys, monkeypatch,
    )
    assert rc == 0
    # Bundle is a valid JSON payload
    parsed = json.loads(enc_out.strip())
    assert "manifest" in parsed
    assert "ciphertext_b64" in parsed

    # Decrypt path — but capsys.readouterr captures str
    # not bytes. Roundtrip via the primitive instead.
    from prsm.enterprise.recipient_encryption import (
        EncryptedPayload, decrypt_for_recipient,
    )
    out_plain = decrypt_for_recipient(
        EncryptedPayload.from_dict(parsed), priv,
    )
    assert out_plain == plaintext


def test_cli_encrypt_requires_recipients_with_colon(
    capsys, monkeypatch,
):
    with pytest.raises(SystemExit):
        _run(
            [
                "encrypt", "--recipients", "no-colon-here",
            ],
            b"x", capsys, monkeypatch,
        )


def test_cli_encrypt_empty_recipients_rejected(
    capsys, monkeypatch,
):
    with pytest.raises(SystemExit):
        _run(
            ["encrypt", "--recipients", ",,,"],
            b"x", capsys, monkeypatch,
        )
