"""R9 Phase 6.3 — peer-jurisdiction filter.

Per R9-SCOPING-1 §6, an operator in a censoring jurisdiction may prefer
their PRSM traffic not transit any peer identified as within that
jurisdiction — both to avoid domestic legal-process compelled-disclosure
risk on that infrastructure and because traffic staying domestic signals
a lower threat model than the user is actually operating under.

This module provides:

- ``GeoIPResolver`` Protocol — host/IP → ISO 3166-1 alpha-2 country
  code (lowercase), or ``None`` when resolution fails.
- ``StaticGeoIPResolver`` — operator-provided host → country mapping.
  No external dependencies. Useful for operator overrides (the CDN
  front for a known-PRC service resolves to a different country by
  default; operator knows better and pins it).
- ``MaxMindGeoIPResolver`` — reads MaxMind GeoLite2-Country.mmdb.
  Optional runtime dependency on ``maxminddb``; the resolver is built
  lazily so users who only need StaticGeoIPResolver don't pay the dep
  cost.
- ``ChainedGeoIPResolver`` — try resolvers in order; first hit wins.
  Typical chain: [Static overrides, MaxMind automatic, None].
- ``PeerJurisdictionFilter`` — configurable policy over a resolver.
  excluded jurisdictions (blocklist), required jurisdictions (allowlist,
  if set), and strict-vs-soft behavior on resolution failure.

What this module does NOT do (per R9 §8 Foundation boundary commits):

- **No default jurisdiction lists.** No "block PRC" preset. No curated
  "safe jurisdictions" list. Every operator configures their own.
- **No automatic GeoIP download.** If operator wants MaxMind, they
  download + maintain the .mmdb file themselves. Foundation doesn't
  host or distribute GeoIP data.
- **No central policy service.** Each node evaluates its own filter
  locally. No Foundation-operated "peer allowlist" endpoint.
- **No enforcement by network coordination.** A node that blocks PRC
  peers is making a local decision; other nodes are free to connect
  to PRC peers and each other as they choose.

Integration with existing primitives:

- ``WebSocketTransport.connect_to_peer`` (R9 Phase 6.2 Task 1-2)
  consults the filter before handing off to the TransportAdapter.
- ``PRSM-SUPPLY-1`` geographic concentration metrics *consume* the
  same GeoIP resolution — a future integration pass will share the
  resolver instance to avoid double-resolving peer country codes.

Usage
-----

.. code-block:: python

    from prsm.node.jurisdiction_filter import (
        PeerJurisdictionFilter,
        StaticGeoIPResolver,
        ChainedGeoIPResolver,
    )

    # Operator-configured override + maxmind fallback.
    static = StaticGeoIPResolver({
        "known-prc-relay.example": "cn",
        "known-ru-node.example": "ru",
    })
    # maxmind = MaxMindGeoIPResolver(Path("./GeoLite2-Country.mmdb"))
    # resolver = ChainedGeoIPResolver([static, maxmind])

    filter = PeerJurisdictionFilter(
        resolver=static,
        excluded={"cn", "ru", "ir"},
        policy="strict",  # block peers we can't resolve
    )

    decision = filter.evaluate("peer.example.com")
    if decision.allow:
        # proceed with transport.connect_to_peer(...)
        ...
    else:
        log.info("blocked peer %s: %s", peer_host, decision.reason)
"""
from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


class GeoIPError(Exception):
    """Base class for GeoIP resolver failures."""


class GeoIPConfigError(GeoIPError):
    """Invalid resolver configuration (bad country code, missing file, etc.)."""


@runtime_checkable
class GeoIPResolver(Protocol):
    """Maps a hostname or IP address to a lowercase ISO 3166-1 alpha-2
    country code, or ``None`` when the host cannot be resolved.

    Implementations MUST:

    - Return lowercase ASCII 2-char codes (``"us"``, ``"cn"``, ``"ru"``).
    - Return ``None`` on any failure — lookup miss, DNS failure, file
      error, invalid input. No exceptions propagate to callers.
    - Be safe to call from multiple threads. The common case
      (``StaticGeoIPResolver``) is immutable after construction;
      ``MaxMindGeoIPResolver`` relies on maxminddb's reader thread-
      safety guarantees.
    """

    def resolve(self, host: str) -> Optional[str]:
        """Return lowercase ISO 3166-1 alpha-2 code for ``host``, or None."""
        ...


def _normalize_country_code(code: str) -> str:
    """Normalize a country code: lowercase, 2-char ASCII, strip whitespace."""
    code = code.strip().lower()
    if len(code) != 2 or not code.isalpha() or not code.isascii():
        raise GeoIPConfigError(
            f"invalid country code {code!r}; expected ISO 3166-1 alpha-2"
        )
    return code


def _normalize_country_set(codes: Iterable[str]) -> FrozenSet[str]:
    """Normalize a set of country codes; raise on any invalid entry."""
    return frozenset(_normalize_country_code(c) for c in codes)


# ──────────────────────────────────────────────────────────────────────
# Resolvers
# ──────────────────────────────────────────────────────────────────────


class StaticGeoIPResolver:
    """Resolver backed by an in-memory hostname/IP → country-code map.

    Useful for:

    - Operator overrides when MaxMind's automatic resolution is wrong
      (e.g., a CDN front for a known-PRC service resolves to the CDN's
      country, not the backing service's).
    - Testing (no external file / network dependency).
    - Small deployments that only need to block a known-bad list of
      named peers.

    Exact-match only. Does NOT do CIDR / subnet matching — that's
    ``MaxMindGeoIPResolver``'s job.
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        normalized: Dict[str, str] = {}
        for host, code in mapping.items():
            if not host:
                raise GeoIPConfigError("empty host in static mapping")
            normalized[host.lower()] = _normalize_country_code(code)
        self._mapping = normalized

    def resolve(self, host: str) -> Optional[str]:
        if not host:
            return None
        return self._mapping.get(host.lower())


class MaxMindGeoIPResolver:
    """Resolver backed by a MaxMind GeoLite2-Country.mmdb file.

    Requires the ``maxminddb`` package and a valid MaxMind database
    file. The Foundation does NOT distribute the file — operators
    download from MaxMind directly (free tier available with
    registration) and update it on their own cadence.

    For IP inputs, queries the database directly. For hostname inputs,
    resolves via DNS first, then queries by IP. DNS resolution goes
    through the OS resolver; operators who need DNS-over-transport
    should combine this with ``bootstrap_transport.make_doh_resolve_txt``
    or equivalent at a higher layer.
    """

    def __init__(self, mmdb_path: Path) -> None:
        if not mmdb_path or not Path(mmdb_path).is_file():
            raise GeoIPConfigError(
                f"MaxMind database not found at {mmdb_path!r}"
            )
        try:
            import maxminddb
        except ImportError as exc:  # pragma: no cover
            raise GeoIPConfigError(
                "MaxMindGeoIPResolver requires maxminddb. "
                "Install with: pip install maxminddb"
            ) from exc
        self._path = Path(mmdb_path)
        self._reader = maxminddb.open_database(str(self._path))

    def resolve(self, host: str) -> Optional[str]:
        if not host:
            return None

        # If host is an IP address, query directly.
        ip_str = _host_to_ip_maybe(host)
        if ip_str is None:
            return None

        try:
            record = self._reader.get(ip_str)
        except Exception as exc:
            logger.debug("MaxMind lookup failed for %s: %s", ip_str, exc)
            return None

        if not record or not isinstance(record, dict):
            return None

        country = record.get("country") or {}
        iso_code = country.get("iso_code") if isinstance(country, dict) else None
        if not iso_code or not isinstance(iso_code, str):
            return None
        try:
            return _normalize_country_code(iso_code)
        except GeoIPConfigError:
            return None

    def close(self) -> None:
        """Release the maxminddb file handle."""
        try:
            self._reader.close()
        except Exception:
            pass


def _host_to_ip_maybe(host: str) -> Optional[str]:
    """Return ``host`` as a canonical IP string if it parses as IPv4/IPv6,
    or attempt DNS resolution. Returns None on failure.

    Kept as a module-level helper so tests can mock it.
    """
    # Parse as-is for IP literals.
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass

    # Fall back to DNS. getaddrinfo is blocking; callers aware of
    # latency should use StaticGeoIPResolver for hot paths.
    import socket
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, OSError) as exc:
        logger.debug("DNS resolution failed for %s: %s", host, exc)
        return None
    if not infos:
        return None
    # Prefer IPv4 if available — MaxMind coverage is better for v4.
    # sockaddr[0] is always the host string per getaddrinfo contract;
    # mypy's Union[str, int] inference over the sockaddr tuple shape
    # requires an explicit cast.
    for _family, _type, _proto, _canonname, sockaddr in infos:
        if _family == socket.AF_INET:
            return str(sockaddr[0])
    # Fall through to first address (likely IPv6).
    return str(infos[0][4][0])


class ChainedGeoIPResolver:
    """Try resolvers in order; return the first non-None result.

    Typical configuration::

        resolver = ChainedGeoIPResolver([
            StaticGeoIPResolver(operator_overrides),
            MaxMindGeoIPResolver(Path("./GeoLite2-Country.mmdb")),
        ])

    Operator overrides win; automatic GeoIP is the fallback.
    """

    def __init__(self, resolvers: List[GeoIPResolver]) -> None:
        if not resolvers:
            raise GeoIPConfigError("ChainedGeoIPResolver requires >=1 resolver")
        self._resolvers = list(resolvers)

    def resolve(self, host: str) -> Optional[str]:
        for resolver in self._resolvers:
            try:
                result = resolver.resolve(host)
            except Exception as exc:
                logger.debug(
                    "resolver %s raised %r; continuing to next",
                    type(resolver).__name__, exc,
                )
                continue
            if result is not None:
                return result
        return None


# ──────────────────────────────────────────────────────────────────────
# Filter
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FilterDecision:
    """Result of a jurisdiction-filter evaluation.

    :param allow: True if the peer should be allowed, False if blocked.
    :param reason: Short human-readable explanation, safe for logging /
        telemetry labels.
    :param detected_jurisdiction: Country code resolved for the peer,
        or None if resolution failed.
    """

    allow: bool
    reason: str
    detected_jurisdiction: Optional[str] = None


Policy = Literal["strict", "soft"]


@dataclass(frozen=True)
class PeerJurisdictionFilter:
    """Filter peers by their GeoIP-resolved jurisdiction.

    Three evaluation modes, combinable:

    - **Excluded set.** Peers resolving to any code in ``excluded`` are
      blocked. Overrides ``required``.
    - **Required set.** If non-empty, peers MUST resolve to a code in
      ``required`` to be allowed. If empty / None, all non-excluded
      resolutions allowed.
    - **Policy on resolution failure:**

        - ``"strict"`` — peer blocked if we can't resolve its
          jurisdiction. Safer for the operator who cares about where
          their traffic goes.
        - ``"soft"`` — peer allowed if we can't resolve. Better
          availability; worse information.

    Operator chooses. Per R9 §8, no default policy is shipped —
    ``policy`` has no default value and must be set explicitly.

    :param resolver: GeoIPResolver instance.
    :param excluded: Lowercase ISO 3166-1 alpha-2 codes to block.
    :param required: Optional set; if set, ONLY these allowed (subject
        to excluded winning conflicts).
    :param policy: Behavior when the resolver returns None.
    """

    resolver: GeoIPResolver
    excluded: FrozenSet[str] = field(default_factory=frozenset)
    required: Optional[FrozenSet[str]] = None
    policy: Policy = "strict"

    def __post_init__(self) -> None:
        # Normalize the sets. Dataclass frozen so use object.__setattr__.
        object.__setattr__(self, "excluded", _normalize_country_set(self.excluded))
        if self.required is not None:
            object.__setattr__(
                self, "required", _normalize_country_set(self.required)
            )
        if self.policy not in ("strict", "soft"):
            raise GeoIPConfigError(
                f"policy must be 'strict' or 'soft'; got {self.policy!r}"
            )
        # Mutual-exclusivity check: if a code is in both excluded AND
        # required, flag as a config error — the operator's intent is
        # ambiguous.
        if self.required is not None:
            conflicts = self.excluded & self.required
            if conflicts:
                raise GeoIPConfigError(
                    f"country codes in both excluded and required: "
                    f"{sorted(conflicts)}"
                )

    def evaluate(self, host: str) -> FilterDecision:
        """Decide whether to allow or block a peer by hostname/IP."""
        if not host:
            return FilterDecision(
                allow=False,
                reason="empty_host",
                detected_jurisdiction=None,
            )

        code = self.resolver.resolve(host)

        if code is None:
            if self.policy == "strict":
                return FilterDecision(
                    allow=False,
                    reason="resolution_failed_strict",
                    detected_jurisdiction=None,
                )
            return FilterDecision(
                allow=True,
                reason="resolution_failed_soft",
                detected_jurisdiction=None,
            )

        # Resolved. Apply set logic.
        if code in self.excluded:
            return FilterDecision(
                allow=False,
                reason=f"excluded_jurisdiction:{code}",
                detected_jurisdiction=code,
            )

        if self.required is not None and code not in self.required:
            return FilterDecision(
                allow=False,
                reason=f"not_in_required_jurisdictions:{code}",
                detected_jurisdiction=code,
            )

        return FilterDecision(
            allow=True,
            reason="allowed",
            detected_jurisdiction=code,
        )


__all__ = [
    "GeoIPError",
    "GeoIPConfigError",
    "GeoIPResolver",
    "StaticGeoIPResolver",
    "MaxMindGeoIPResolver",
    "ChainedGeoIPResolver",
    "FilterDecision",
    "Policy",
    "PeerJurisdictionFilter",
]
