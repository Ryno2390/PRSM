/**
 * PRSM Forta Detection Bot — Main Entrypoint
 *
 * Wires detectors against TransactionEvent and BlockEvent streams from
 * Forta scanner nodes. Routes findings through AlertRouter to war-room
 * channels per Exploit Response Playbook §3.
 *
 * Architecture:
 *  - handleTransaction(tx)  → all per-tx detectors fan out
 *  - handleBlock(block)      → time-window detectors (stuck escrows, etc.)
 *  - findings  → AlertRouter (Discord / PagerDuty / email)
 */

import {
  BlockEvent,
  Finding,
  HandleBlock,
  HandleTransaction,
  TransactionEvent,
  getEthersProvider,
} from "forta-agent";

import { detectFTNSTokenAnomalies } from "./detectors/ftns_token";
import { detectProvenanceRegistryAnomalies } from "./detectors/provenance_registry";
import { detectRoyaltyDistributorAnomalies } from "./detectors/royalty_distributor";
import { detectBatchSettlementAnomalies } from "./detectors/batch_settlement";
import {
  detectEscrowPoolAnomalies,
  detectStuckEscrows,
} from "./detectors/escrow_pool";
import { getAlertRouter } from "./routing/alert_router";
import { NetworkChainId } from "./config/contracts";

/** Currently-running chain — populated lazily on first handler invocation. */
let _chainId: NetworkChainId | null = null;

async function resolveChainId(): Promise<NetworkChainId> {
  if (_chainId !== null) return _chainId;
  const provider = getEthersProvider();
  const network = await provider.getNetwork();
  _chainId = Number(network.chainId) as NetworkChainId;
  return _chainId;
}

const handleTransaction: HandleTransaction = async (
  txEvent: TransactionEvent
): Promise<Finding[]> => {
  const chainId = await resolveChainId();
  const router = getAlertRouter();

  // Fan-out detectors. Parallel for speed; aggregate findings.
  const detectorResults = await Promise.allSettled([
    detectFTNSTokenAnomalies(txEvent, chainId),
    detectProvenanceRegistryAnomalies(txEvent, chainId),
    detectRoyaltyDistributorAnomalies(txEvent, chainId),
    detectBatchSettlementAnomalies(txEvent, chainId),
    detectEscrowPoolAnomalies(txEvent, chainId),
  ]);

  const findings: Finding[] = [];
  for (const result of detectorResults) {
    if (result.status === "fulfilled") {
      findings.push(...result.value);
    } else {
      console.error("Detector failed:", result.reason);
    }
  }

  // Route findings to alert channels (best-effort, non-blocking on individual failures)
  await Promise.allSettled(findings.map((f) => router.route(f)));

  // Forta network also receives findings via return value
  return findings;
}

const handleBlock: HandleBlock = async (
  _blockEvent: BlockEvent
): Promise<Finding[]> => {
  const router = getAlertRouter();

  // Time-window detectors run per block (not per tx)
  const findings = await detectStuckEscrows();

  await Promise.allSettled(findings.map((f) => router.route(f)));
  return findings;
}

export default {
  handleTransaction,
  handleBlock,
};
