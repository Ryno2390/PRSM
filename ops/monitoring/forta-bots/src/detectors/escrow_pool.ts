/**
 * EscrowPool Detector
 *
 * Monitors `EscrowPool.sol` (Phase 3.1) for:
 * - Anomalous-amount escrows or releases (>5× rolling p99)
 * - Refund-rate spikes (operational signal of submitter trouble)
 * - Stuck escrows (created but not released/refunded after 48h)
 *
 * Maps to Risk Register A2 (Payment escrow exploit) — CRITICAL severity.
 * Escrow drain is the "Wormhole / Ronin" failure mode for PRSM.
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

class RefundTracker {
  private created: number[] = [];
  private refunded: number[] = [];
  private readonly windowMs = 60 * 60 * 1000;

  recordCreated(): void { this.created.push(Date.now()); this.prune(); }
  recordRefunded(): void { this.refunded.push(Date.now()); this.prune(); }

  private prune(): void {
    const cutoff = Date.now() - this.windowMs;
    while (this.created.length > 0 && this.created[0] < cutoff) this.created.shift();
    while (this.refunded.length > 0 && this.refunded[0] < cutoff) this.refunded.shift();
  }

  refundRatePct(): number {
    return this.created.length === 0 ? 0 : (this.refunded.length / this.created.length) * 100;
  }
}

const _amountWindow = new AmountWindow();
const _refundTracker = new RefundTracker();

/** Tracks escrows for stuck-detection. Map jobId → createdAtMs. */
const _pendingEscrows = new Map<string, number>();

export async function detectEscrowPoolAnomalies(
  txEvent: TransactionEvent,
  chainId: NetworkChainId
): Promise<Finding[]> {
  const findings: Finding[] = [];

  const address = CONTRACT_ADDRESSES.EscrowPool[chainId];
  if (!address) return findings;
  if (!txEvent.addresses[address.toLowerCase()]) return findings;

  // ── Detector 1: Escrow created — track for stuck detection ──
  const created = txEvent.filterLog(EVENT_SIGNATURES.ESCROW_CREATED, address);
  for (const ev of created) {
    const jobId = ev.args.jobId as string;
    const amount = BigInt(ev.args.amount.toString());

    _pendingEscrows.set(jobId, Date.now());
    _refundTracker.recordCreated();

    // Amount-spike detection on creation
    const p99 = _amountWindow.p99();
    if (p99 !== null && amount > (p99 * BigInt(THRESHOLDS.ESCROW_DRAIN_AMOUNT_PCT)) / 100n) {
      findings.push(
        Finding.fromObject({
          name: "EscrowPool anomalous escrow amount",
          description: `Escrow of ${amount.toString()} FTNS created for job ${jobId.slice(0, 10)}…, exceeding rolling p99 by ${THRESHOLDS.ESCROW_DRAIN_AMOUNT_PCT / 100}×. Possible drain-attempt setup.`,
          alertId: "PRSM-ESCROW-AMOUNT-SPIKE",
          severity: FindingSeverity.High,
          type: FindingType.Suspicious,
          protocol: "PRSM",
          metadata: {
            jobId,
            amount: amount.toString(),
            p99Baseline: p99.toString(),
            multiplier: ((amount * 100n) / p99).toString(),
            payer: ev.args.payer as string,
            contract: address,
            txHash: txEvent.hash,
          },
        })
      );
    }
    _amountWindow.add(amount);
  }

  // ── Detector 2: Escrow released — same anomaly check + clear pending ──
  const released = txEvent.filterLog(EVENT_SIGNATURES.ESCROW_RELEASED, address);
  for (const ev of released) {
    const jobId = ev.args.jobId as string;
    const amount = BigInt(ev.args.amount.toString());
    _pendingEscrows.delete(jobId);

    const p99 = _amountWindow.p99();
    if (p99 !== null && amount > (p99 * BigInt(THRESHOLDS.ESCROW_DRAIN_AMOUNT_PCT)) / 100n) {
      findings.push(
        Finding.fromObject({
          name: "EscrowPool anomalous release amount",
          description: `Escrow release of ${amount.toString()} FTNS to ${(ev.args.provider as string).slice(0, 10)}… exceeds rolling p99 by ${THRESHOLDS.ESCROW_DRAIN_AMOUNT_PCT / 100}×. Investigate for drain.`,
          alertId: "PRSM-ESCROW-RELEASE-SPIKE",
          severity: FindingSeverity.Critical,
          type: FindingType.Suspicious,
          protocol: "PRSM",
          metadata: {
            jobId,
            amount: amount.toString(),
            p99Baseline: p99.toString(),
            provider: ev.args.provider as string,
            contract: address,
            txHash: txEvent.hash,
          },
        })
      );
    }
  }

  // ── Detector 3: Escrow refunded ──
  const refunded = txEvent.filterLog(EVENT_SIGNATURES.ESCROW_REFUNDED, address);
  for (const ev of refunded) {
    const jobId = ev.args.jobId as string;
    _pendingEscrows.delete(jobId);
    _refundTracker.recordRefunded();
  }

  // Refund-rate threshold — separate from individual events
  if (_refundTracker.refundRatePct() > 30) {
    findings.push(
      Finding.fromObject({
        name: "EscrowPool elevated refund rate",
        description: `${_refundTracker.refundRatePct().toFixed(1)}% of escrows refunded over rolling 1h window. Above 30% threshold suggests systemic provider failures or attack on dispatch.`,
        alertId: "PRSM-ESCROW-REFUND-RATE",
        severity: FindingSeverity.Medium,
        type: FindingType.Suspicious,
        protocol: "PRSM",
        metadata: {
          refundRatePct: _refundTracker.refundRatePct().toFixed(2),
          contract: address,
          txHash: txEvent.hash,
        },
      })
    );
  }

  // ── Detector 4: Stuck escrows (block-level rather than tx-level — handled in agent.ts handleBlock) ──
  // Pending detection happens in block handler; we expose helper here.

  return findings;
}

/**
 * Block handler — checks for stuck escrows. Called from agent.ts handleBlock.
 */
export async function detectStuckEscrows(): Promise<Finding[]> {
  const findings: Finding[] = [];
  const cutoff = Date.now() - THRESHOLDS.STUCK_ESCROW_AGE_HOURS * 60 * 60 * 1000;

  for (const [jobId, createdAt] of _pendingEscrows.entries()) {
    if (createdAt < cutoff) {
      const ageHours = (Date.now() - createdAt) / (60 * 60 * 1000);
      findings.push(
        Finding.fromObject({
          name: "EscrowPool stuck escrow",
          description: `Escrow for job ${jobId.slice(0, 10)}… created ${ageHours.toFixed(1)}h ago, never released or refunded. Provider may have abandoned dispatch.`,
          alertId: "PRSM-ESCROW-STUCK",
          severity: FindingSeverity.Medium,
          type: FindingType.Info,
          protocol: "PRSM",
          metadata: {
            jobId,
            ageHours: ageHours.toFixed(1),
            threshold: THRESHOLDS.STUCK_ESCROW_AGE_HOURS.toString(),
          },
        })
      );
      // Once flagged, remove from tracking to avoid repeated alerts
      _pendingEscrows.delete(jobId);
    }
  }

  return findings;
}
