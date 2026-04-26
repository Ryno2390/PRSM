/**
 * RoyaltyDistributor Detector
 *
 * Monitors `RoyaltyDistributor.sol` for:
 * - Distribution failures (alerts above 5% failure rate)
 * - Anomalous distribution amounts (>10× rolling p99)
 * - Split-percentage anomalies (creator+node+treasury+burn != 100%)
 *
 * Maps to Risk Register A1 (RoyaltyDistributor exploit) — High severity.
 * Maps to Tokenomics §10 invariant #2 ("Creator royalties are enforced on-chain").
 */

import {
  TransactionEvent,
  Finding,
  FindingSeverity,
  FindingType,
} from "forta-agent";

import {
  CONTRACT_ADDRESSES,
  EVENT_SIGNATURES,
  PAYMENT_SPLIT_BPS,
  THRESHOLDS,
  NetworkChainId,
} from "../config/contracts";

/** Rolling tracker of distributions for failure-rate calculation. */
class DistributionTracker {
  private successes: number[] = [];
  private failures: number[] = [];
  private readonly windowMs = 60 * 60 * 1000; // 1h

  recordSuccess(): void {
    this.successes.push(Date.now());
    this.prune();
  }

  recordFailure(): void {
    this.failures.push(Date.now());
    this.prune();
  }

  private prune(): void {
    const cutoff = Date.now() - this.windowMs;
    while (this.successes.length > 0 && this.successes[0] < cutoff) this.successes.shift();
    while (this.failures.length > 0 && this.failures[0] < cutoff) this.failures.shift();
  }

  failureRatePct(): number {
    const total = this.successes.length + this.failures.length;
    if (total < 10) return 0;
    return (this.failures.length / total) * 100;
  }
}

class AmountWindow {
  private samples: bigint[] = [];
  private readonly windowSize = 100;

  add(amount: bigint): void {
    this.samples.push(amount);
    if (this.samples.length > this.windowSize) this.samples.shift();
  }

  p99(): bigint | null {
    if (this.samples.length < 10) return null;
    const sorted = [...this.samples].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    return sorted[Math.floor(sorted.length * 0.99)];
  }
}

const _tracker = new DistributionTracker();
const _amountWindow = new AmountWindow();

export async function detectRoyaltyDistributorAnomalies(
  txEvent: TransactionEvent,
  chainId: NetworkChainId
): Promise<Finding[]> {
  const findings: Finding[] = [];

  const address = CONTRACT_ADDRESSES.RoyaltyDistributor[chainId];
  if (!address) return findings;
  if (!txEvent.addresses[address.toLowerCase()]) return findings;

  // ── Detector 1: Distribution failures ──
  const failures = txEvent.filterLog(EVENT_SIGNATURES.DISTRIBUTION_FAILED, address);
  for (const ev of failures) {
    _tracker.recordFailure();
    findings.push(
      Finding.fromObject({
        name: "RoyaltyDistributor distribution failure",
        description: `Distribution failed for content ${(ev.args.contentHash as string).slice(0, 10)}…: ${ev.args.reason}`,
        alertId: "PRSM-ROYALTY-DIST-FAILED",
        severity: FindingSeverity.Medium,
        type: FindingType.Info,
        protocol: "PRSM",
        metadata: {
          contentHash: ev.args.contentHash as string,
          reason: ev.args.reason as string,
          contract: address,
          txHash: txEvent.hash,
        },
      })
    );
  }

  const failureRatePct = _tracker.failureRatePct();
  if (failureRatePct > THRESHOLDS.DISTRIBUTION_FAILURE_RATE_PCT) {
    findings.push(
      Finding.fromObject({
        name: "RoyaltyDistributor elevated failure rate",
        description: `Distribution failure rate ${failureRatePct.toFixed(1)}% over rolling 1h window exceeds ${THRESHOLDS.DISTRIBUTION_FAILURE_RATE_PCT}% threshold. Investigate underlying cause.`,
        alertId: "PRSM-ROYALTY-FAILURE-RATE",
        severity: FindingSeverity.High,
        type: FindingType.Suspicious,
        protocol: "PRSM",
        metadata: {
          failureRatePct: failureRatePct.toFixed(2),
          threshold: THRESHOLDS.DISTRIBUTION_FAILURE_RATE_PCT.toString(),
          contract: address,
          txHash: txEvent.hash,
        },
      })
    );
  }

  // ── Detector 2: Successful distributions — check split + amount ──
  const distributions = txEvent.filterLog(EVENT_SIGNATURES.ROYALTY_PAID, address);
  for (const ev of distributions) {
    _tracker.recordSuccess();

    const creatorShare = BigInt(ev.args.creatorShare.toString());
    const nodeShare = BigInt(ev.args.nodeShare.toString());
    const treasuryShare = BigInt(ev.args.treasuryShare.toString());
    const total = creatorShare + nodeShare + treasuryShare;

    // Split-correctness: creator + node + treasury should be 80% of (total + 20% burn);
    // i.e., ratios CREATOR:NODE:TREASURY should be 6.4:72:1.6 (sum = 80%).
    // We can't know the burn portion from this event alone, but we can verify the
    // 6.4:72:1.6 RATIO between the three split components (which sum to 80% of payment).
    // Tolerance: ±0.5% of expected ratio.
    if (total > 0n) {
      const expectedCreatorBps = (PAYMENT_SPLIT_BPS.CREATOR * 10000) / (PAYMENT_SPLIT_BPS.CREATOR + PAYMENT_SPLIT_BPS.NODE + PAYMENT_SPLIT_BPS.TREASURY);
      const expectedNodeBps    = (PAYMENT_SPLIT_BPS.NODE    * 10000) / (PAYMENT_SPLIT_BPS.CREATOR + PAYMENT_SPLIT_BPS.NODE + PAYMENT_SPLIT_BPS.TREASURY);
      const actualCreatorBps   = Number((creatorShare * 10000n) / total);
      const actualNodeBps      = Number((nodeShare    * 10000n) / total);

      const tolerance = 50; // 0.5%
      const creatorDeviation = Math.abs(actualCreatorBps - expectedCreatorBps);
      const nodeDeviation    = Math.abs(actualNodeBps - expectedNodeBps);

      if (creatorDeviation > tolerance || nodeDeviation > tolerance) {
        findings.push(
          Finding.fromObject({
            name: "RoyaltyDistributor split-ratio anomaly",
            description: `Distribution split deviates from expected 6.4:72:1.6 ratio. Critical — Tokenomics §10 invariant #2 ("Creator royalties are enforced on-chain") may be violated.`,
            alertId: "PRSM-ROYALTY-SPLIT-ANOMALY",
            severity: FindingSeverity.Critical,
            type: FindingType.Suspicious,
            protocol: "PRSM",
            metadata: {
              contentHash: ev.args.contentHash as string,
              expectedCreatorBps: expectedCreatorBps.toString(),
              actualCreatorBps: actualCreatorBps.toString(),
              expectedNodeBps: expectedNodeBps.toString(),
              actualNodeBps: actualNodeBps.toString(),
              creatorShare: creatorShare.toString(),
              nodeShare: nodeShare.toString(),
              treasuryShare: treasuryShare.toString(),
              contract: address,
              txHash: txEvent.hash,
            },
          })
        );
      }
    }

    // Amount-spike detection
    const totalAmount = creatorShare + nodeShare + treasuryShare;
    const p99 = _amountWindow.p99();
    if (p99 !== null && totalAmount > (p99 * BigInt(THRESHOLDS.ANOMALOUS_DISTRIBUTION_AMOUNT_PCT)) / 100n) {
      findings.push(
        Finding.fromObject({
          name: "RoyaltyDistributor anomalous distribution amount",
          description: `Distribution of ${totalAmount.toString()} FTNS exceeds rolling p99 by ${THRESHOLDS.ANOMALOUS_DISTRIBUTION_AMOUNT_PCT / 100}×. Possible drain attempt or pricing oracle manipulation.`,
          alertId: "PRSM-ROYALTY-AMOUNT-SPIKE",
          severity: FindingSeverity.High,
          type: FindingType.Suspicious,
          protocol: "PRSM",
          metadata: {
            totalAmount: totalAmount.toString(),
            p99Baseline: p99.toString(),
            multiplier: ((totalAmount * 100n) / p99).toString(),
            contract: address,
            txHash: txEvent.hash,
          },
        })
      );
    }
    _amountWindow.add(totalAmount);
  }

  return findings;
}
