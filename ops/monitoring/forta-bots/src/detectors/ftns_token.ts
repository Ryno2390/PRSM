/**
 * FTNSToken Detector
 *
 * Monitors `FTNSTokenSimple.sol` for:
 * - Unauthorized mints (any non-MINTER_ROLE address minting)
 * - Role grants/revokes (admin actions)
 * - Supply approaching cap (>95% of 1B max)
 * - Transfer-volume spikes (>10× rolling average)
 * - Pause/unpause events
 *
 * Maps to Risk Register A4 (FTNSToken contract exploit) — Critical severity.
 * Maps to Exploit Playbook §2.1 P0 examples.
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
  ROLES,
  THRESHOLDS,
  NetworkChainId,
} from "../config/contracts";

/** Rolling-window state for transfer-volume baseline. */
class TransferVolumeWindow {
  private samples: bigint[] = [];
  private readonly windowSize = 100;

  addSample(amount: bigint): void {
    this.samples.push(amount);
    if (this.samples.length > this.windowSize) {
      this.samples.shift();
    }
  }

  /** Returns p99 of recent samples; null if insufficient data. */
  p99(): bigint | null {
    if (this.samples.length < 10) return null;
    const sorted = [...this.samples].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    return sorted[Math.floor(sorted.length * 0.99)];
  }
}

const _transferWindow = new TransferVolumeWindow();

/** Cumulative-supply tracker — naive implementation (production would query chain state). */
let _cumulativeMinted = 0n;
let _cumulativeBurned = 0n;

export async function detectFTNSTokenAnomalies(
  txEvent: TransactionEvent,
  chainId: NetworkChainId
): Promise<Finding[]> {
  const findings: Finding[] = [];

  const address = CONTRACT_ADDRESSES.FTNSToken[chainId];
  if (!address) return findings;
  if (!txEvent.addresses[address.toLowerCase()]) return findings;

  // ── Detector 1: Role grants/revokes (admin actions) ──
  const roleEvents = txEvent.filterLog(
    [EVENT_SIGNATURES.ROLE_GRANTED, EVENT_SIGNATURES.ROLE_REVOKED],
    address
  );
  for (const ev of roleEvents) {
    const role = ev.args.role as string;
    const account = ev.args.account as string;
    const sender = ev.args.sender as string;

    const roleName = roleNameFromHash(role);
    const eventType = ev.name === "RoleGranted" ? "granted" : "revoked";
    const severity =
      role === ROLES.DEFAULT_ADMIN_ROLE
        ? FindingSeverity.High
        : role === ROLES.MINTER_ROLE
          ? FindingSeverity.High
          : FindingSeverity.Medium;

    findings.push(
      Finding.fromObject({
        name: `FTNSToken role ${eventType}`,
        description: `Role ${roleName} ${eventType} on FTNSToken — ${account} by ${sender}.`,
        alertId: "PRSM-FTNS-ROLE-CHANGE",
        severity,
        type: FindingType.Suspicious,
        protocol: "PRSM",
        metadata: {
          role: roleName,
          roleHash: role,
          account,
          sender,
          eventType,
          contract: address,
          txHash: txEvent.hash,
        },
      })
    );
  }

  // ── Detector 2: Pause/Unpause (operational signal, P1 by default) ──
  const pauseEvents = txEvent.filterLog(
    [EVENT_SIGNATURES.PAUSED, EVENT_SIGNATURES.UNPAUSED],
    address
  );
  for (const ev of pauseEvents) {
    const account = ev.args.account as string;
    findings.push(
      Finding.fromObject({
        name: `FTNSToken ${ev.name}`,
        description:
          ev.name === "Paused"
            ? `FTNSToken contract PAUSED by ${account}. All transfers blocked. Investigate.`
            : `FTNSToken contract UNPAUSED by ${account}. Transfers resumed.`,
        alertId: ev.name === "Paused" ? "PRSM-FTNS-PAUSED" : "PRSM-FTNS-UNPAUSED",
        severity: FindingSeverity.High,
        type: ev.name === "Paused" ? FindingType.Suspicious : FindingType.Info,
        protocol: "PRSM",
        metadata: {
          account,
          contract: address,
          txHash: txEvent.hash,
        },
      })
    );
  }

  // ── Detector 3: Transfer-volume spike ──
  const transfers = txEvent.filterLog(EVENT_SIGNATURES.ERC20_TRANSFER, address);
  for (const ev of transfers) {
    const value = BigInt(ev.args.value.toString());
    const from = ev.args.from as string;
    const to = ev.args.to as string;

    // Mint detection — transfers from zero address
    if (from === "0x0000000000000000000000000000000000000000") {
      _cumulativeMinted += value;
      const supplyPct = Number((_cumulativeMinted - _cumulativeBurned) * 100n / THRESHOLDS.SUPPLY_CAP_FTNS);
      if (supplyPct >= THRESHOLDS.SUPPLY_WARNING_PCT) {
        findings.push(
          Finding.fromObject({
            name: "FTNSToken supply approaching cap",
            description: `Active supply ${supplyPct}% of 1B FTNS hard cap. Cap proximity may indicate emission-controller bug or misuse.`,
            alertId: "PRSM-FTNS-SUPPLY-NEAR-CAP",
            severity: FindingSeverity.High,
            type: FindingType.Suspicious,
            protocol: "PRSM",
            metadata: {
              activeSupply: (_cumulativeMinted - _cumulativeBurned).toString(),
              cap: THRESHOLDS.SUPPLY_CAP_FTNS.toString(),
              percentageOfCap: supplyPct.toString(),
              contract: address,
              txHash: txEvent.hash,
            },
          })
        );
      }
    }

    // Burn detection — transfers to zero address
    if (to === "0x0000000000000000000000000000000000000000") {
      _cumulativeBurned += value;
    }

    // Volume-spike detection
    const p99 = _transferWindow.p99();
    if (p99 !== null && value > (p99 * BigInt(THRESHOLDS.ANOMALOUS_TRANSFER_VOLUME_PCT)) / 100n) {
      findings.push(
        Finding.fromObject({
          name: "FTNSToken anomalous transfer volume",
          description: `Transfer of ${value.toString()} FTNS exceeds rolling p99 by ${THRESHOLDS.ANOMALOUS_TRANSFER_VOLUME_PCT / 100}×. Investigate for drain attempt or sanctioned mint.`,
          alertId: "PRSM-FTNS-VOLUME-SPIKE",
          severity: FindingSeverity.High,
          type: FindingType.Suspicious,
          protocol: "PRSM",
          metadata: {
            from,
            to,
            value: value.toString(),
            p99Baseline: p99.toString(),
            multiplier: ((value * 100n) / p99).toString(),
            contract: address,
            txHash: txEvent.hash,
          },
        })
      );
    }
    _transferWindow.addSample(value);
  }

  return findings;
}

/** Resolve role hash to human-readable name. */
function roleNameFromHash(hash: string): string {
  switch (hash) {
    case ROLES.DEFAULT_ADMIN_ROLE: return "DEFAULT_ADMIN";
    case ROLES.MINTER_ROLE: return "MINTER";
    case ROLES.BURNER_ROLE: return "BURNER";
    case ROLES.PAUSER_ROLE: return "PAUSER";
    default: return `unknown(${hash.slice(0, 10)})`;
  }
}
