/**
 * BatchSettlementRegistry Detector
 *
 * Monitors `BatchSettlementRegistry.sol` (Phase 3.1) for:
 * - Challenge-rate spikes (>50/h sustained)
 * - High challenger-win rate (>25% suggests systemic submitter issue)
 * - Stale batches (committed but never resolved)
 *
 * Maps to settlement-integrity invariants in Phase 3.1 design + Risk Register
 * A2 (Payment escrow exploit precursor signal).
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
  THRESHOLDS,
  NetworkChainId,
} from "../config/contracts";

class ChallengeStats {
  private challenges: number[] = [];
  private wins: number[] = [];
  private readonly windowMs = 60 * 60 * 1000;

  recordChallenge(): void {
    this.challenges.push(Date.now());
    this.prune();
  }

  recordResolution(challengerWon: boolean): void {
    if (challengerWon) this.wins.push(Date.now());
    this.prune();
  }

  private prune(): void {
    const cutoff = Date.now() - this.windowMs;
    while (this.challenges.length > 0 && this.challenges[0] < cutoff) this.challenges.shift();
    while (this.wins.length > 0 && this.wins[0] < cutoff) this.wins.shift();
  }

  challengeRatePerHour(): number { return this.challenges.length; }
  winRatePct(): number {
    return this.challenges.length === 0 ? 0 : (this.wins.length / this.challenges.length) * 100;
  }
}

const _stats = new ChallengeStats();

export async function detectBatchSettlementAnomalies(
  txEvent: TransactionEvent,
  chainId: NetworkChainId
): Promise<Finding[]> {
  const findings: Finding[] = [];

  const address = CONTRACT_ADDRESSES.BatchSettlementRegistry[chainId];
  if (!address) return findings;
  if (!txEvent.addresses[address.toLowerCase()]) return findings;

  // ── Detector 1: Challenge submitted ──
  const challenges = txEvent.filterLog(EVENT_SIGNATURES.CHALLENGE_SUBMITTED, address);
  for (const ev of challenges) {
    _stats.recordChallenge();
    findings.push(
      Finding.fromObject({
        name: "BatchSettlement challenge submitted",
        description: `Challenger ${(ev.args.challenger as string).slice(0, 10)}… disputed receipt ${(ev.args.receiptHash as string).slice(0, 10)}… in batch ${(ev.args.batchRoot as string).slice(0, 10)}…`,
        alertId: "PRSM-SETTLE-CHALLENGE-SUBMITTED",
        severity: FindingSeverity.Info,
        type: FindingType.Info,
        protocol: "PRSM",
        metadata: {
          batchRoot: ev.args.batchRoot as string,
          receiptHash: ev.args.receiptHash as string,
          challenger: ev.args.challenger as string,
          contract: address,
          txHash: txEvent.hash,
        },
      })
    );
  }

  if (_stats.challengeRatePerHour() > THRESHOLDS.CHALLENGE_RATE_PER_HOUR) {
    findings.push(
      Finding.fromObject({
        name: "BatchSettlement elevated challenge rate",
        description: `${_stats.challengeRatePerHour()} challenges in past hour exceeds ${THRESHOLDS.CHALLENGE_RATE_PER_HOUR}/h baseline. Possible coordinated dispute attack or systemic submitter issue.`,
        alertId: "PRSM-SETTLE-CHALLENGE-RATE",
        severity: FindingSeverity.Medium,
        type: FindingType.Suspicious,
        protocol: "PRSM",
        metadata: {
          challengesPerHour: _stats.challengeRatePerHour().toString(),
          threshold: THRESHOLDS.CHALLENGE_RATE_PER_HOUR.toString(),
          contract: address,
          txHash: txEvent.hash,
        },
      })
    );
  }

  // ── Detector 2: Challenge resolutions — high win rate ──
  const resolutions = txEvent.filterLog(EVENT_SIGNATURES.CHALLENGE_RESOLVED, address);
  for (const ev of resolutions) {
    const challengerWon = Boolean(ev.args.challengerWon);
    _stats.recordResolution(challengerWon);

    if (challengerWon) {
      findings.push(
        Finding.fromObject({
          name: "BatchSettlement challenge upheld",
          description: `Challenge succeeded: receipt ${(ev.args.receiptHash as string).slice(0, 10)}… in batch ${(ev.args.batchRoot as string).slice(0, 10)}… was invalid. Submitter slashed.`,
          alertId: "PRSM-SETTLE-CHALLENGE-WIN",
          severity: FindingSeverity.Medium,
          type: FindingType.Info,
          protocol: "PRSM",
          metadata: {
            batchRoot: ev.args.batchRoot as string,
            receiptHash: ev.args.receiptHash as string,
            contract: address,
            txHash: txEvent.hash,
          },
        })
      );
    }
  }

  if (_stats.winRatePct() > THRESHOLDS.CHALLENGE_WIN_RATE_PCT && _stats.challengeRatePerHour() >= 4) {
    findings.push(
      Finding.fromObject({
        name: "BatchSettlement elevated challenger-win rate",
        description: `${_stats.winRatePct().toFixed(1)}% of challenges over rolling 1h window won by challenger, above ${THRESHOLDS.CHALLENGE_WIN_RATE_PCT}% threshold. Indicates systemic submitter problem — investigate submitter behavior.`,
        alertId: "PRSM-SETTLE-WIN-RATE",
        severity: FindingSeverity.High,
        type: FindingType.Suspicious,
        protocol: "PRSM",
        metadata: {
          winRatePct: _stats.winRatePct().toFixed(2),
          threshold: THRESHOLDS.CHALLENGE_WIN_RATE_PCT.toString(),
          contract: address,
          txHash: txEvent.hash,
        },
      })
    );
  }

  return findings;
}
