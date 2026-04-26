/**
 * ProvenanceRegistry Detector
 *
 * Monitors `ProvenanceRegistry.sol` for:
 * - Ownership transfers (creator-record changes)
 * - Registration spam from a single address
 * - High royalty rates (>9000 bps approaching 9800 cap)
 *
 * Maps to Risk Register A3 (ProvenanceRegistry exploit) — High severity.
 * Sybil-spam detection contributes to F4 (Sybil attack on creator/operator pools).
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

/** Per-address registration counter with hourly window. */
class RegistrationCounter {
  private counts = new Map<string, { count: number; windowStart: number }>();
  private readonly windowMs = 60 * 60 * 1000;

  record(address: string): number {
    const now = Date.now();
    const entry = this.counts.get(address.toLowerCase());
    if (!entry || now - entry.windowStart > this.windowMs) {
      this.counts.set(address.toLowerCase(), { count: 1, windowStart: now });
      return 1;
    }
    entry.count += 1;
    return entry.count;
  }
}

const _registrationCounter = new RegistrationCounter();

export async function detectProvenanceRegistryAnomalies(
  txEvent: TransactionEvent,
  chainId: NetworkChainId
): Promise<Finding[]> {
  const findings: Finding[] = [];

  const address = CONTRACT_ADDRESSES.ProvenanceRegistry[chainId];
  if (!address) return findings;
  if (!txEvent.addresses[address.toLowerCase()]) return findings;

  // ── Detector 1: Ownership transfers (creator-record changes) ──
  const ownershipTransfers = txEvent.filterLog(EVENT_SIGNATURES.OWNERSHIP_TRANSFERRED, address);
  for (const ev of ownershipTransfers) {
    const contentHash = ev.args.contentHash as string;
    const previousOwner = ev.args.previousOwner as string;
    const newOwner = ev.args.newOwner as string;

    findings.push(
      Finding.fromObject({
        name: "ProvenanceRegistry creator change",
        description: `Content ownership transferred for ${contentHash.slice(0, 10)}… from ${previousOwner.slice(0, 10)}… to ${newOwner.slice(0, 10)}…. Verify against creator's signed authorization.`,
        alertId: "PRSM-PROVENANCE-OWNERSHIP-TRANSFER",
        severity: FindingSeverity.Medium,
        type: FindingType.Info,
        protocol: "PRSM",
        metadata: {
          contentHash,
          previousOwner,
          newOwner,
          contract: address,
          txHash: txEvent.hash,
        },
      })
    );
  }

  // ── Detector 2: Registration spam ──
  const registrations = txEvent.filterLog(EVENT_SIGNATURES.CONTENT_REGISTERED, address);
  for (const ev of registrations) {
    const contentHash = ev.args.contentHash as string;
    const creator = ev.args.creator as string;
    const royaltyRateBps = Number(ev.args.royaltyRateBps);

    const recentCount = _registrationCounter.record(creator);
    if (recentCount > THRESHOLDS.REGISTRATION_SPAM_PER_HOUR_PER_ADDRESS) {
      findings.push(
        Finding.fromObject({
          name: "ProvenanceRegistry registration spam",
          description: `Address ${creator.slice(0, 10)}… registered ${recentCount} content hashes in the past hour, exceeding threshold of ${THRESHOLDS.REGISTRATION_SPAM_PER_HOUR_PER_ADDRESS}. Possible Sybil attack on creator-pool compensation.`,
          alertId: "PRSM-PROVENANCE-SPAM",
          severity: FindingSeverity.Medium,
          type: FindingType.Suspicious,
          protocol: "PRSM",
          metadata: {
            creator,
            recentCount: recentCount.toString(),
            threshold: THRESHOLDS.REGISTRATION_SPAM_PER_HOUR_PER_ADDRESS.toString(),
            contract: address,
            txHash: txEvent.hash,
          },
        })
      );
    }

    // ── Detector 3: High royalty rate ──
    if (royaltyRateBps > THRESHOLDS.HIGH_ROYALTY_RATE_BPS) {
      findings.push(
        Finding.fromObject({
          name: "ProvenanceRegistry high royalty rate",
          description: `Content registered with royalty rate ${royaltyRateBps} bps (${royaltyRateBps / 100}%), above threshold of ${THRESHOLDS.HIGH_ROYALTY_RATE_BPS / 100}%. Verify legitimate creator-set rate vs. exploit attempt.`,
          alertId: "PRSM-PROVENANCE-HIGH-RATE",
          severity: FindingSeverity.Low,
          type: FindingType.Info,
          protocol: "PRSM",
          metadata: {
            contentHash,
            creator,
            royaltyRateBps: royaltyRateBps.toString(),
            threshold: THRESHOLDS.HIGH_ROYALTY_RATE_BPS.toString(),
            contract: address,
            txHash: txEvent.hash,
          },
        })
      );
    }
  }

  return findings;
}
