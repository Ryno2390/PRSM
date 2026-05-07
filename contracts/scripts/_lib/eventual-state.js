// Helper for the Base RPC propagation race that bit every deploy +
// transfer-ownership ceremony in the 2026-05-07 mainnet sprint.
//
// Symptom: `await tx.wait()` returns once the tx is mined, but an
// immediate-next state read (`ownable.pendingOwner()`,
// `bsr.escrowPool()`, etc.) returns the PRE-tx value because the
// state indexer hasn't propagated yet. The deploy/transfer-ownership
// scripts crash on a "post-transfer pendingOwner is 0x0..." invariant
// check even though the tx itself confirmed correctly.
//
// `waitForExpectedState(read, predicate, opts)` retries the getter
// until either the predicate matches or `attempts` runs out. Sleeps
// ~`delayMs` between attempts. If we never converge, throws with the
// last observed value so the operator can decide whether the tx
// actually failed (as opposed to RPC just being slow).
//
// This is a "trust the receipt, distrust the immediate-next read"
// pattern. The receipt is canonical (a confirmed tx is on-chain
// regardless of indexer state); the eventual-consistency read is
// what we need to retry.

const DEFAULT_ATTEMPTS = 8;
const DEFAULT_DELAY_MS = 750;

async function waitForExpectedState(read, predicate, opts = {}) {
  const attempts = opts.attempts ?? DEFAULT_ATTEMPTS;
  const delayMs = opts.delayMs ?? DEFAULT_DELAY_MS;
  let last;
  for (let i = 0; i < attempts; i++) {
    last = await read();
    if (predicate(last)) return last;
    if (i < attempts - 1) {
      await new Promise(r => setTimeout(r, delayMs));
    }
  }
  throw new Error(
    opts.errorPrefix
      ? `${opts.errorPrefix}: state never converged after ${attempts} attempts ` +
        `(${attempts * delayMs}ms total). Last value: ${last}`
      : `state never converged after ${attempts} attempts. Last value: ${last}`
  );
}

/** Convenience wrapper for the common "expect this address" case. */
async function waitForAddressEquals(read, expected, opts = {}) {
  const expectedLc = expected.toLowerCase();
  return waitForExpectedState(
    read,
    v => typeof v === "string" && v.toLowerCase() === expectedLc,
    opts,
  );
}

module.exports = { waitForExpectedState, waitForAddressEquals };
