// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/// @title M1 cadence-driven yield invariant (sprint 370)
/// @notice Halmos-provable mirror of the Phase 3.x.11.q
///         FixedRateShardedExecutor inter-emission cadence
///         invariant from audit-prep §7.13. For ALL inner-
///         executor emission timings, the post-decorator
///         inter-yield gap is bounded BELOW by cadence —
///         the wire observer cannot infer per-token compute
///         time from emission timing.
///
/// @dev STRUCTURAL EQUIVALENCE (audit-visible):
///   prsm/compute/chain_rpc/tier_c_sharded_executors.py:
///     FixedRateShardedExecutor.execute_chain_streaming
///     (lines 328-351). The load-bearing block is the
///     pre-yield sleep gate:
///
///       if last_emit is not None:
///           target = last_emit + cadence
///           now = clock()
///           if now < target:
///               sleep(target - now)
///       yield event
///       last_emit = clock()
///
///   Invariant: for every yield after the first, the elapsed
///   clock time since the previous yield is >= cadence —
///   even when the inner executor's compute is faster than
///   cadence (the sleep slows it down). Faster computation
///   does NOT propagate to faster emission.
///
/// Run:
///   cd contracts/symbolic-proofs && \
///   halmos --contract M1CadenceDrivenYieldSpec

contract M1CadenceDrivenYield {
    uint256 public cadence;
    uint256 public lastEmit;
    bool public hasFirstEmit;

    constructor(uint256 _cadence) {
        require(_cadence > 0, "cadence must be positive");
        cadence = _cadence;
    }

    /// Mirrors the per-yield gating in
    /// execute_chain_streaming. `nowAtArrival` is the
    /// clock-read inside the executor BEFORE the sleep
    /// gate fires; returns the effective post-sleep emit
    /// timestamp.
    ///
    /// Halmos models the symbolic clock as caller-supplied
    /// monotonically-increasing values. Real implementation
    /// uses time.monotonic() injected via constructor for
    /// deterministic testability.
    function tick(uint256 nowAtArrival)
        external
        returns (uint256 emitTimestamp)
    {
        if (!hasFirstEmit) {
            // First emission — no cadence gate applies
            // (audit-prep §7.13: first yield is the seed).
            lastEmit = nowAtArrival;
            hasFirstEmit = true;
            return nowAtArrival;
        }
        uint256 target = lastEmit + cadence;
        // If arrival is earlier than target, sleep to target;
        // otherwise no sleep (inner executor is slower than
        // cadence, in which case wire-observer already sees
        // a >= cadence gap from inner compute alone).
        uint256 effectiveEmit = nowAtArrival < target
            ? target
            : nowAtArrival;
        lastEmit = effectiveEmit;
        return effectiveEmit;
    }
}


contract M1CadenceDrivenYieldSpec {
    uint256 internal constant CADENCE = 100;

    /// Fresh deployment per test to keep the state-machine
    /// pre-state deterministic. Halmos's setUp runs once
    /// per check_* — perfect for this proof shape.
    function _fresh() internal returns (
        M1CadenceDrivenYield
    ) {
        return new M1CadenceDrivenYield(CADENCE);
    }

    /// THE M1 invariant: for ALL symbolic (nowA, nowB)
    /// where nowB >= nowA (monotonic clock) AND nowA
    /// represents the first emission, the second emission
    /// timestamp is at least nowA + cadence.
    ///
    /// Concretely: the inter-emission gap is bounded BELOW
    /// by cadence regardless of how fast the inner
    /// executor produced the second event.
    function check_inter_emission_gap_bounded(
        uint256 nowA,
        uint256 nowB
    ) public {
        vm_assume(nowA <= 1_000_000);
        vm_assume(nowB <= 1_000_000);
        vm_assume(nowB >= nowA);  // Monotonic clock

        M1CadenceDrivenYield ex = _fresh();
        uint256 emitA = ex.tick(nowA);
        uint256 emitB = ex.tick(nowB);

        // First emission unchanged (no prior to gate against)
        assert(emitA == nowA);
        // Second emission >= emitA + cadence — the load-
        // bearing claim
        assert(emitB >= emitA + CADENCE);
    }

    /// Fast inner executor: nowB barely past nowA. The
    /// sleep gate MUST extend the emission to nowA +
    /// cadence. Closes the timing-leak threat where fast
    /// inner compute would reveal short-content tokens.
    function check_fast_inner_gated_to_cadence(
        uint256 nowA,
        uint256 nowB
    ) public {
        vm_assume(nowA <= 1_000_000);
        vm_assume(nowB >= nowA);
        vm_assume(nowB < nowA + CADENCE);  // Inner is FAST

        M1CadenceDrivenYield ex = _fresh();
        ex.tick(nowA);
        uint256 emitB = ex.tick(nowB);
        // Sleep gate fires — emit pushed to nowA + cadence
        assert(emitB == nowA + CADENCE);
    }

    /// Slow inner executor: nowB already past target. No
    /// sleep needed; emit is at nowB (inner compute alone
    /// produced a >= cadence gap). Sister proof to the
    /// fast case — symbolic execution exhausts both
    /// branches.
    function check_slow_inner_emits_immediately(
        uint256 nowA,
        uint256 nowB
    ) public {
        vm_assume(nowA <= 1_000_000);
        vm_assume(nowB >= nowA + CADENCE);  // Inner is SLOW
        vm_assume(nowB <= 2_000_000);

        M1CadenceDrivenYield ex = _fresh();
        ex.tick(nowA);
        uint256 emitB = ex.tick(nowB);
        // No sleep — emit is nowB unchanged
        assert(emitB == nowB);
    }

    /// First emission gets no gating: arrives at clock time.
    function check_first_emission_ungated(
        uint256 nowA
    ) public {
        vm_assume(nowA <= 1_000_000);
        M1CadenceDrivenYield ex = _fresh();
        uint256 emitA = ex.tick(nowA);
        assert(emitA == nowA);
    }

    /// Three-emission composition: A → B → C must satisfy
    /// (emitB - emitA) >= cadence AND (emitC - emitB) >=
    /// cadence. The state machine carries lastEmit across
    /// calls correctly.
    function check_three_emission_composition(
        uint256 nowA,
        uint256 nowB,
        uint256 nowC
    ) public {
        vm_assume(nowA <= 1_000_000);
        vm_assume(nowB >= nowA);
        vm_assume(nowB <= 1_500_000);
        vm_assume(nowC >= nowB);
        vm_assume(nowC <= 2_000_000);

        M1CadenceDrivenYield ex = _fresh();
        uint256 emitA = ex.tick(nowA);
        uint256 emitB = ex.tick(nowB);
        uint256 emitC = ex.tick(nowC);

        assert(emitB >= emitA + CADENCE);
        assert(emitC >= emitB + CADENCE);
    }

    /// Zero-cadence rejected at construction.
    function check_zero_cadence_rejected() public {
        try new M1CadenceDrivenYield(0) {
            assert(false);
        } catch {
            assert(true);
        }
    }

    /// Halmos cheatcode for symbolic pre-conditions.
    function vm_assume(bool cond) internal {
        (bool ok,) = address(
            uint160(uint256(keccak256("hevm cheat code")))
        ).call(
            abi.encodeWithSelector(bytes4(0x4c63e562), cond)
        );
        ok;
    }
}
