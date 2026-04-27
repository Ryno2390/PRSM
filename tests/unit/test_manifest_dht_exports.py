"""
Smoke test — Phase 3.x.5 Task 6 — public-surface contract.

Pins every name in ``prsm.network.manifest_dht.__all__`` so accidental
drift breaks at PR time, not on first downstream user report.

Mirrors the pattern from
``test_publisher_key_anchor_exports.py`` (Phase 3.x.3 Task 6) and
``test_model_registry_exports.py`` (Phase 3.x.2 Task 6) — same six
guarantees:
  1. ``__all__`` matches an explicit expected set
  2. No duplicates in ``__all__``
  3. Every advertised name resolves at import time
  4. Type expectations are met (classes are classes, callables callable)
  5. Re-export identity preserved across import paths
  6. Star-import surface = ``__all__`` (no internal helper leakage)
"""

from __future__ import annotations

import inspect

import prsm.network.manifest_dht as dht_pkg


# ──────────────────────────────────────────────────────────────────────────
# __all__ contract
# ──────────────────────────────────────────────────────────────────────────


EXPECTED_PUBLIC_NAMES = frozenset({
    # Protocol constants (Task 1)
    "DHT_PROTOCOL_VERSION",
    "MAX_MESSAGE_BYTES",
    "MAX_PROVIDERS_PER_RESPONSE",
    "MESSAGE_TYPE_REGISTRY",
    # Enums
    "ErrorCode",
    "MessageType",
    # Message types
    "FindProvidersRequest",
    "FetchManifestRequest",
    "ProviderInfo",
    "ProvidersResponse",
    "ManifestResponse",
    "ErrorResponse",
    # Codec
    "encode_message",
    "parse_message",
    # Exceptions (protocol layer)
    "ProtocolError",
    "MalformedMessageError",
    "UnknownMessageTypeError",
    "IncompatibleProtocolVersionError",
    # Local index (Task 2)
    "LocalManifestIndex",
    # DHT client (Task 3)
    "ManifestDHTClient",
    "RoutingTable",
    "SendMessageFn",
    "DEFAULT_K",
    "DHTClientError",
    "ManifestNotFoundError",
    "TransportFailureError",
    # DHT server (Task 4)
    "ManifestDHTServer",
    "UNKNOWN_REQUEST_ID",
})


class TestPublicSurfacePins:
    def test_all_matches_expected_set(self):
        # If this fails, either __all__ has drifted or the test needs
        # an explicit decision about whether to add/remove a name.
        assert set(dht_pkg.__all__) == EXPECTED_PUBLIC_NAMES

    def test_no_duplicate_names_in_all(self):
        all_list = dht_pkg.__all__
        assert len(all_list) == len(set(all_list)), (
            f"duplicates in __all__: {all_list}"
        )

    def test_every_name_resolves(self):
        for name in dht_pkg.__all__:
            assert hasattr(dht_pkg, name), (
                f"__all__ advertises {name!r} but it doesn't resolve "
                f"on the module"
            )
            assert getattr(dht_pkg, name) is not None, (
                f"{name!r} resolved to None — likely a stale import"
            )


# ──────────────────────────────────────────────────────────────────────────
# Type expectations — catches accidental shadowing
# ──────────────────────────────────────────────────────────────────────────


class TestExportedTypes:
    def test_protocol_version_is_int(self):
        assert isinstance(dht_pkg.DHT_PROTOCOL_VERSION, int)
        assert dht_pkg.DHT_PROTOCOL_VERSION >= 1

    def test_message_type_registry_is_dict(self):
        # Map type-tag string → from_dict constructor.
        assert isinstance(dht_pkg.MESSAGE_TYPE_REGISTRY, dict)
        # Every value must be callable (a from_dict classmethod).
        for tag, ctor in dht_pkg.MESSAGE_TYPE_REGISTRY.items():
            assert isinstance(tag, str), f"tag {tag!r} not a string"
            assert callable(ctor), f"ctor for {tag!r} not callable"

    def test_enums_are_str_enums(self):
        # ErrorCode and MessageType are str-Enum subclasses; tools
        # log/serialize them as strings so this matters.
        from enum import Enum
        for cls_name in ("ErrorCode", "MessageType"):
            cls = getattr(dht_pkg, cls_name)
            assert inspect.isclass(cls)
            assert issubclass(cls, Enum)
            assert issubclass(cls, str), (
                f"{cls_name} must inherit from str for log readability"
            )

    def test_message_dataclasses_are_classes(self):
        for name in (
            "FindProvidersRequest",
            "FetchManifestRequest",
            "ProviderInfo",
            "ProvidersResponse",
            "ManifestResponse",
            "ErrorResponse",
        ):
            cls = getattr(dht_pkg, name)
            assert inspect.isclass(cls), f"{name} must be a class"
            # All five wire messages have a `to_dict()` method.
            assert hasattr(cls, "to_dict"), f"{name} missing to_dict()"
            assert hasattr(cls, "from_dict"), f"{name} missing from_dict()"

    def test_codec_callable(self):
        for name in ("encode_message", "parse_message"):
            assert callable(getattr(dht_pkg, name)), (
                f"{name} must be callable"
            )

    def test_protocol_exceptions_form_a_hierarchy(self):
        # ProtocolError is the documented base for protocol-layer
        # failures (parse / version / unknown-type). Anything in __all__
        # under that umbrella MUST inherit from it.
        assert issubclass(dht_pkg.ProtocolError, Exception)
        for name in (
            "MalformedMessageError",
            "UnknownMessageTypeError",
            "IncompatibleProtocolVersionError",
        ):
            cls = getattr(dht_pkg, name)
            assert issubclass(cls, dht_pkg.ProtocolError), (
                f"{name} must subclass ProtocolError"
            )

    def test_dht_client_exceptions_form_a_hierarchy(self):
        # DHTClientError is the base for client-layer failures —
        # ManifestNotFound / TransportFailure surface to callers under
        # this umbrella.
        assert issubclass(dht_pkg.DHTClientError, Exception)
        for name in ("ManifestNotFoundError", "TransportFailureError"):
            cls = getattr(dht_pkg, name)
            assert issubclass(cls, dht_pkg.DHTClientError), (
                f"{name} must subclass DHTClientError"
            )

    def test_local_index_is_class(self):
        assert inspect.isclass(dht_pkg.LocalManifestIndex)

    def test_dht_client_is_class(self):
        assert inspect.isclass(dht_pkg.ManifestDHTClient)

    def test_dht_server_is_class(self):
        assert inspect.isclass(dht_pkg.ManifestDHTServer)

    def test_routing_table_is_protocol(self):
        # Protocol class — duck-typing contract for what a routing
        # table must expose. Should be a class-like object, not an
        # instance.
        assert inspect.isclass(dht_pkg.RoutingTable)

    def test_send_message_fn_is_a_callable_alias(self):
        # SendMessageFn is a ``typing.Callable`` alias. We don't pin
        # the exact runtime form (Callable internals shift across
        # Python versions); just ensure it resolved.
        assert dht_pkg.SendMessageFn is not None

    def test_default_k_is_positive_int(self):
        assert isinstance(dht_pkg.DEFAULT_K, int)
        assert dht_pkg.DEFAULT_K > 0

    def test_unknown_request_id_is_non_empty_string(self):
        # The sentinel returned by the server for parse failures —
        # client-side correlation expects a non-empty string here.
        assert isinstance(dht_pkg.UNKNOWN_REQUEST_ID, str)
        assert dht_pkg.UNKNOWN_REQUEST_ID != ""


# ──────────────────────────────────────────────────────────────────────────
# Re-export integrity — package surface and submodule attrs identity-equal
# ──────────────────────────────────────────────────────────────────────────


class TestReExportIdentity:
    """Sneaky regression mode: someone re-defines a name at the
    package level (e.g., adds a wrapper function) and the direct-import
    path stops being identity-equal to the submodule-import path.
    Downstream code using ``isinstance(x, ManifestNotFoundError)``
    would then start failing on objects produced by the submodule.
    These tests pin identity."""

    def test_protocol_re_exports_match_submodule(self):
        from prsm.network.manifest_dht import protocol as _proto
        for name in (
            "DHT_PROTOCOL_VERSION",
            "MESSAGE_TYPE_REGISTRY",
            "ErrorCode",
            "MessageType",
            "FindProvidersRequest",
            "FetchManifestRequest",
            "ProviderInfo",
            "ProvidersResponse",
            "ManifestResponse",
            "ErrorResponse",
            "encode_message",
            "parse_message",
            "ProtocolError",
            "MalformedMessageError",
            "UnknownMessageTypeError",
            "IncompatibleProtocolVersionError",
        ):
            assert getattr(dht_pkg, name) is getattr(_proto, name), (
                f"{name} re-export drift: package vs submodule"
            )

    def test_local_index_re_exports_match_submodule(self):
        from prsm.network.manifest_dht import local_index as _li
        assert dht_pkg.LocalManifestIndex is _li.LocalManifestIndex

    def test_dht_client_re_exports_match_submodule(self):
        from prsm.network.manifest_dht import dht_client as _dc
        for name in (
            "ManifestDHTClient",
            "RoutingTable",
            "SendMessageFn",
            "DEFAULT_K",
            "DHTClientError",
            "ManifestNotFoundError",
            "TransportFailureError",
        ):
            assert getattr(dht_pkg, name) is getattr(_dc, name), (
                f"{name} re-export drift: package vs dht_client"
            )

    def test_dht_server_re_exports_match_submodule(self):
        from prsm.network.manifest_dht import dht_server as _ds
        assert dht_pkg.ManifestDHTServer is _ds.ManifestDHTServer
        assert dht_pkg.UNKNOWN_REQUEST_ID is _ds.UNKNOWN_REQUEST_ID


# ──────────────────────────────────────────────────────────────────────────
# Star-import smoke
# ──────────────────────────────────────────────────────────────────────────


class TestStarImport:
    def test_star_import_yields_only_public_names(self):
        # `from prsm.network.manifest_dht import *` should bring in
        # EXACTLY the public surface. Catches accidental leakage of
        # internal helpers (logger, _validate_*, etc.).
        ns: dict = {}
        exec("from prsm.network.manifest_dht import *", ns)
        imported = {k for k in ns if not k.startswith("_")}
        assert imported == EXPECTED_PUBLIC_NAMES, (
            f"star-import surface drift:\n"
            f"  unexpected: {imported - EXPECTED_PUBLIC_NAMES}\n"
            f"  missing:    {EXPECTED_PUBLIC_NAMES - imported}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Round-trip smoke — encode/parse goes through __all__ surface only
# ──────────────────────────────────────────────────────────────────────────


class TestRoundTripViaPublicSurface:
    """Don't just pin names — confirm a working DHT message round-trips
    using ONLY symbols from ``__all__``. If a future refactor breaks the
    encode/parse pairing for a public message type, this catches it."""

    def test_find_providers_round_trip(self):
        req = dht_pkg.FindProvidersRequest(
            model_id="alpha", request_id="rid-1"
        )
        wire = dht_pkg.encode_message(req)
        parsed = dht_pkg.parse_message(wire)
        assert isinstance(parsed, dht_pkg.FindProvidersRequest)
        assert parsed.model_id == "alpha"
        assert parsed.request_id == "rid-1"

    def test_fetch_manifest_round_trip(self):
        req = dht_pkg.FetchManifestRequest(
            model_id="beta", request_id="rid-2"
        )
        wire = dht_pkg.encode_message(req)
        parsed = dht_pkg.parse_message(wire)
        assert isinstance(parsed, dht_pkg.FetchManifestRequest)

    def test_providers_response_round_trip(self):
        provider = dht_pkg.ProviderInfo(node_id="n", address="h:1")
        resp = dht_pkg.ProvidersResponse(
            request_id="rid-3", providers=(provider,)
        )
        wire = dht_pkg.encode_message(resp)
        parsed = dht_pkg.parse_message(wire)
        assert isinstance(parsed, dht_pkg.ProvidersResponse)
        assert parsed.providers[0].node_id == "n"

    def test_manifest_response_round_trip(self):
        resp = dht_pkg.ManifestResponse(
            request_id="rid-4", manifest={"k": "v"}
        )
        wire = dht_pkg.encode_message(resp)
        parsed = dht_pkg.parse_message(wire)
        assert isinstance(parsed, dht_pkg.ManifestResponse)
        assert parsed.manifest == {"k": "v"}

    def test_error_response_round_trip(self):
        resp = dht_pkg.ErrorResponse(
            request_id="rid-5",
            code=dht_pkg.ErrorCode.NOT_FOUND.value,
            message="missing",
        )
        wire = dht_pkg.encode_message(resp)
        parsed = dht_pkg.parse_message(wire)
        assert isinstance(parsed, dht_pkg.ErrorResponse)
        assert parsed.code == dht_pkg.ErrorCode.NOT_FOUND.value
