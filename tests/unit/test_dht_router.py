"""DHTRequestRouter tests — verifies the demultiplexer routes DHT
request envelopes to the right server and degrades safely on every
failure path.
"""
from __future__ import annotations

import json

import pytest

from prsm.network.dht_router import DHTRequestRouter


# ---- helpers ------------------------------------------------------


class _StubServer:
    """Records the most recent request_bytes for assertion."""

    def __init__(self, response_bytes: bytes = b"OK"):
        self.response_bytes = response_bytes
        self.calls: list = []

    def handle(self, request_bytes: bytes) -> bytes:
        self.calls.append(request_bytes)
        return self.response_bytes


class _RaisingServer:
    """Used to verify the router catches downstream raises."""

    def handle(self, request_bytes: bytes) -> bytes:
        raise RuntimeError("server bug")


def _envelope(msg_type: str, **extras) -> bytes:
    return json.dumps({"type": msg_type, "version": 1, **extras}).encode("utf-8")


def _decode_error(response: bytes) -> dict:
    return json.loads(response.decode("utf-8"))


# ---- constructor --------------------------------------------------


def test_constructor_rejects_all_none():
    with pytest.raises(ValueError, match="at least one server"):
        DHTRequestRouter()


def test_constructor_accepts_manifest_only():
    DHTRequestRouter(manifest_server=_StubServer())  # no raise


def test_constructor_accepts_embedding_only():
    DHTRequestRouter(embedding_server=_StubServer())  # no raise


def test_constructor_accepts_both():
    DHTRequestRouter(
        manifest_server=_StubServer(), embedding_server=_StubServer(),
    )


# ---- routing — happy paths ----------------------------------------


def test_routes_find_providers_to_manifest():
    manifest = _StubServer(response_bytes=b"manifest-resp")
    embedding = _StubServer(response_bytes=b"embedding-resp")
    router = DHTRequestRouter(
        manifest_server=manifest, embedding_server=embedding,
    )
    req = _envelope("find_providers", model_id="x", request_id="r1")
    response = router.handle(req)
    assert response == b"manifest-resp"
    assert manifest.calls == [req]
    assert embedding.calls == []


def test_routes_fetch_manifest_to_manifest():
    manifest = _StubServer(response_bytes=b"m")
    embedding = _StubServer()
    router = DHTRequestRouter(
        manifest_server=manifest, embedding_server=embedding,
    )
    req = _envelope("fetch_manifest", model_id="x", request_id="r1")
    response = router.handle(req)
    assert response == b"m"
    assert len(manifest.calls) == 1
    assert embedding.calls == []


def test_routes_find_embedding_to_embedding():
    manifest = _StubServer()
    embedding = _StubServer(response_bytes=b"emb-resp")
    router = DHTRequestRouter(
        manifest_server=manifest, embedding_server=embedding,
    )
    req = _envelope("find_embedding", request_id="r1")
    response = router.handle(req)
    assert response == b"emb-resp"
    assert manifest.calls == []
    assert len(embedding.calls) == 1


def test_routes_fetch_embedding_to_embedding():
    manifest = _StubServer()
    embedding = _StubServer(response_bytes=b"e")
    router = DHTRequestRouter(
        manifest_server=manifest, embedding_server=embedding,
    )
    req = _envelope("fetch_embedding", request_id="r1")
    response = router.handle(req)
    assert response == b"e"
    assert len(embedding.calls) == 1


def test_works_with_only_manifest_server_registered():
    """A bootstrap node that only runs ManifestDHT must still respond
    to manifest-protocol requests; embedding requests get a structured
    error rather than a crash."""
    router = DHTRequestRouter(
        manifest_server=_StubServer(response_bytes=b"manifest-only"),
    )
    response = router.handle(_envelope("find_providers", model_id="x"))
    assert response == b"manifest-only"

    response = router.handle(_envelope("find_embedding"))
    err = _decode_error(response)
    assert err["type"] == "error"
    assert err["code"] == "UNSUPPORTED_VERSION"


def test_works_with_only_embedding_server_registered():
    router = DHTRequestRouter(
        embedding_server=_StubServer(response_bytes=b"emb-only"),
    )
    response = router.handle(_envelope("find_embedding"))
    assert response == b"emb-only"

    response = router.handle(_envelope("find_providers", model_id="x"))
    err = _decode_error(response)
    assert err["code"] == "UNSUPPORTED_VERSION"


# ---- routing — error paths ----------------------------------------


def test_returns_error_envelope_for_invalid_json():
    router = DHTRequestRouter(manifest_server=_StubServer())
    response = router.handle(b"not-json{{{")
    err = _decode_error(response)
    assert err["type"] == "error"
    assert err["code"] == "MALFORMED_REQUEST"
    assert "JSON parse" in err["message"]


def test_returns_error_envelope_for_non_dict_top_level():
    router = DHTRequestRouter(manifest_server=_StubServer())
    response = router.handle(b'["not", "a", "dict"]')
    err = _decode_error(response)
    assert err["code"] == "MALFORMED_REQUEST"
    assert "must be a dict" in err["message"]


def test_returns_error_envelope_for_missing_type():
    router = DHTRequestRouter(manifest_server=_StubServer())
    response = router.handle(json.dumps({"version": 1}).encode("utf-8"))
    err = _decode_error(response)
    assert err["code"] == "MALFORMED_REQUEST"
    assert "type" in err["message"]


def test_returns_error_envelope_for_non_string_type():
    router = DHTRequestRouter(manifest_server=_StubServer())
    response = router.handle(
        json.dumps({"type": 42, "version": 1}).encode("utf-8"),
    )
    err = _decode_error(response)
    assert err["code"] == "MALFORMED_REQUEST"


def test_returns_error_envelope_for_unknown_type():
    router = DHTRequestRouter(manifest_server=_StubServer())
    response = router.handle(_envelope("invented_kind", request_id="r1"))
    err = _decode_error(response)
    assert err["code"] == "UNSUPPORTED_VERSION"
    assert "no DHT server" in err["message"]


def test_returns_error_envelope_for_invalid_utf8():
    router = DHTRequestRouter(manifest_server=_StubServer())
    response = router.handle(b"\xff\xfe\xfd")
    err = _decode_error(response)
    assert err["code"] == "MALFORMED_REQUEST"


def test_router_catches_server_raise_and_returns_error():
    """Both DHTs' handle() are documented never-raises, but defense in
    depth: if a regression introduces a raise, the router converts to
    an internal-error envelope so the TCP listener can still ship the
    response."""
    router = DHTRequestRouter(manifest_server=_RaisingServer())
    response = router.handle(_envelope("find_providers", model_id="x"))
    err = _decode_error(response)
    assert err["code"] == "INTERNAL_ERROR"
    assert "RuntimeError" in err["message"]


def test_router_catches_server_returning_non_bytes():
    class _BadServer:
        def handle(self, request_bytes):
            return "should-be-bytes-not-str"

    router = DHTRequestRouter(manifest_server=_BadServer())
    response = router.handle(_envelope("find_providers", model_id="x"))
    err = _decode_error(response)
    assert err["code"] == "INTERNAL_ERROR"
    assert "non-bytes" in err["message"]


def test_router_returns_bytearray_as_bytes():
    """Defensive: if a server returns bytearray instead of bytes, it's
    still a valid response — coerce to immutable bytes for the wire."""

    class _BAServer:
        def handle(self, request_bytes):
            return bytearray(b"ok")

    router = DHTRequestRouter(manifest_server=_BAServer())
    response = router.handle(_envelope("find_providers", model_id="x"))
    assert response == b"ok"
    assert isinstance(response, bytes)


# ---- determinism on overlapping types -----------------------------


def test_error_type_overlap_resolves_to_manifest_when_both_registered():
    """Both protocols define an "error" message type. Errors never
    arrive at server entry, but resolve them deterministically anyway —
    we picked manifest as the older registry."""
    manifest = _StubServer(response_bytes=b"manifest-handled-error")
    embedding = _StubServer(response_bytes=b"embedding-handled-error")
    router = DHTRequestRouter(
        manifest_server=manifest, embedding_server=embedding,
    )
    response = router.handle(_envelope("error", code="X"))
    assert response == b"manifest-handled-error"
    assert len(manifest.calls) == 1
    assert embedding.calls == []


def test_error_type_falls_through_to_embedding_when_only_embedding_registered():
    embedding = _StubServer(response_bytes=b"embedding-handled")
    router = DHTRequestRouter(embedding_server=embedding)
    response = router.handle(_envelope("error", code="X"))
    assert response == b"embedding-handled"
