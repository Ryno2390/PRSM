// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/// @title Three-band similarity routing invariants (sprint 374)
/// @notice Halmos-provable mirror of the PRSM-PROV-1 Item 6
///         three-band similarity-score routing from audit-prep
///         §7.19. EffectiveThresholds carries
///         (arbitration_floor, derivative, duplicate) bands;
///         this proof verifies (a) the band-ordering invariant
///         (0 <= arbitration_floor <= derivative <= duplicate
///         <= 1.0) is rejected at construction when violated,
///         and (b) the routing function is monotonic + correct.
///
/// @dev STRUCTURAL EQUIVALENCE (audit-visible):
///   prsm/data/dedup/thresholds.py:55-99 — EffectiveThresholds
///     dataclass __post_init__ validators.
///   Routing logic at usage sites (multiple callers in
///   prsm/data/dedup/) — modeled here as a single
///   `route(score)` function for symbolic verification.
///
///   Similarity scores represented as bps (0-10000 = [0, 1]
///   in float) to keep the math integer-clean. The proof
///   would carry over to float semantics modulo the well-
///   known IEEE 754 boundary issues, which are out of
///   scope for the routing-correctness invariant.
///
/// Run:
///   cd contracts/symbolic-proofs && \
///   halmos --contract ThreeBandRoutingSpec

contract ThreeBandRouting {
    /// Routing decision enum. Numbered to allow halmos
    /// comparison + ordering checks.
    uint8 public constant ROUTE_CLEAN = 0;
    uint8 public constant ROUTE_ARBITRATE = 1;
    uint8 public constant ROUTE_DERIVATIVE = 2;
    uint8 public constant ROUTE_DUPLICATE = 3;

    /// Constructor enforces the band-ordering invariant —
    /// mirror of __post_init__ at thresholds.py:75-99.
    uint256 public arbitrationFloorBps;
    uint256 public derivativeBps;
    uint256 public duplicateBps;

    constructor(
        uint256 _arbitrationFloorBps,
        uint256 _derivativeBps,
        uint256 _duplicateBps
    ) {
        require(_derivativeBps <= 10000, "derivative out of range");
        require(_duplicateBps <= 10000, "duplicate out of range");
        require(
            _arbitrationFloorBps <= 10000,
            "arbitration_floor out of range"
        );
        require(
            _duplicateBps >= _derivativeBps,
            "duplicate must be >= derivative"
        );
        require(
            _arbitrationFloorBps <= _derivativeBps,
            "arbitration_floor must be <= derivative"
        );
        arbitrationFloorBps = _arbitrationFloorBps;
        derivativeBps = _derivativeBps;
        duplicateBps = _duplicateBps;
    }

    /// Route a similarity score into one of four bands.
    /// Score in bps (0-10000).
    function route(uint256 scoreBps)
        external
        view
        returns (uint8)
    {
        if (scoreBps >= duplicateBps) return ROUTE_DUPLICATE;
        if (scoreBps >= derivativeBps) return ROUTE_DERIVATIVE;
        if (scoreBps >= arbitrationFloorBps) {
            return ROUTE_ARBITRATE;
        }
        return ROUTE_CLEAN;
    }
}


contract ThreeBandRoutingSpec {
    /// Canonical three-band setup for the routing proofs.
    /// Halmos's setUp runs once per check_*.
    function _canonical()
        internal
        returns (ThreeBandRouting)
    {
        // arbitration_floor=5500, derivative=6500,
        // duplicate=9500 — well-formed bands matching the
        // T6.5 default profile.
        return new ThreeBandRouting(5500, 6500, 9500);
    }

    /// Construction-time invariant: out-of-order bands MUST
    /// reject. Halmos explores ALL (af, der, dup) symbolic
    /// triples and proves the constructor either succeeds
    /// with the canonical ordering or reverts.
    function check_construction_enforces_band_ordering(
        uint256 af,
        uint256 der,
        uint256 dup
    ) public {
        vm_assume(af <= 10000);
        vm_assume(der <= 10000);
        vm_assume(dup <= 10000);
        try new ThreeBandRouting(af, der, dup) returns (
            ThreeBandRouting tb
        ) {
            // Success → invariants hold post-construction
            assert(tb.arbitrationFloorBps() <= tb.derivativeBps());
            assert(tb.derivativeBps() <= tb.duplicateBps());
            assert(tb.duplicateBps() <= 10000);
        } catch {
            // Revert path: at least one inequality was
            // violated. Acceptable.
            assert(true);
        }
    }

    /// Routing monotonicity: higher score → same-or-stricter
    /// band. For ALL (s1, s2) with s1 <= s2, route(s1) <=
    /// route(s2). The numeric ordering of the enum is
    /// load-bearing — codified by the constants.
    function check_routing_monotonic(
        uint256 s1,
        uint256 s2
    ) public {
        vm_assume(s1 <= 10000);
        vm_assume(s2 <= 10000);
        vm_assume(s1 <= s2);
        ThreeBandRouting tb = _canonical();
        uint8 r1 = tb.route(s1);
        uint8 r2 = tb.route(s2);
        assert(r1 <= r2);
    }

    /// Band-boundary correctness: score exactly AT the
    /// duplicate threshold routes to DUPLICATE (inclusive
    /// lower bound).
    function check_at_duplicate_threshold_routes_duplicate(
    ) public {
        ThreeBandRouting tb = _canonical();
        uint8 r = tb.route(tb.duplicateBps());
        assert(r == tb.ROUTE_DUPLICATE());
    }

    /// Band-boundary correctness: score exactly AT the
    /// derivative threshold routes to DERIVATIVE.
    function check_at_derivative_threshold_routes_derivative(
    ) public {
        ThreeBandRouting tb = _canonical();
        uint8 r = tb.route(tb.derivativeBps());
        assert(r == tb.ROUTE_DERIVATIVE());
    }

    /// Band-boundary correctness: score exactly AT the
    /// arbitration_floor routes to ARBITRATE.
    function check_at_arbitration_floor_routes_arbitrate(
    ) public {
        ThreeBandRouting tb = _canonical();
        uint8 r = tb.route(tb.arbitrationFloorBps());
        assert(r == tb.ROUTE_ARBITRATE());
    }

    /// Below arbitration_floor → CLEAN.
    function check_below_floor_routes_clean(
        uint256 score
    ) public {
        ThreeBandRouting tb = _canonical();
        vm_assume(score < tb.arbitrationFloorBps());
        uint8 r = tb.route(score);
        assert(r == tb.ROUTE_CLEAN());
    }

    /// Coverage completeness: route ALWAYS returns one of
    /// the four canonical values, never a fifth.
    function check_route_always_returns_valid(
        uint256 score
    ) public {
        vm_assume(score <= 10000);
        ThreeBandRouting tb = _canonical();
        uint8 r = tb.route(score);
        assert(r <= tb.ROUTE_DUPLICATE());
    }

    function vm_assume(bool cond) internal {
        (bool ok,) = address(
            uint160(uint256(keccak256("hevm cheat code")))
        ).call(
            abi.encodeWithSelector(bytes4(0x4c63e562), cond)
        );
        ok;
    }
}
