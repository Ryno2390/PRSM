/*
 * A-08 RoyaltyDistributor v2 — post-deploy state verification.
 *
 * Reads a provenance-<network>-*.json manifest produced by
 * deploy-provenance.js (the same script that ships v2 — its
 * 4-arg constructor matches the v2 source). Asserts the v2
 * contract is wired correctly + ownership posture is correct
 * for the current ceremony stage.
 *
 * Companion to:
 *   docs/governance/2026-05-09-A-08-v2-redeploy-ceremony-plan.md
 *   docs/governance/A-08-recoverStranded-design.md
 *
 * The script's assertion set evolves through the ceremony stages:
 *
 *   Stage 1 (post-deploy, pre-transferOwnership):
 *     owner == deployer, pendingOwner == 0, totalClaimable == 0
 *   Stage 2 (post-transferOwnership, pre-acceptOwnership):
 *     EXPECT_PENDING_OWNER set; pendingOwner == that value, owner
 *     still == deployer
 *   Stage 3 (post-acceptOwnership, final state):
 *     EXPECT_FINAL_OWNER set; owner == that value, pendingOwner == 0
 *
 * Required env var:
 *   PROVENANCE_MANIFEST  - path to provenance-<network>-*.json
 *
 * Optional env vars (drive stage-specific assertions):
 *   EXPECT_PENDING_OWNER - assert pendingOwner == this address
 *   EXPECT_FINAL_OWNER   - assert owner == this address (post-accept)
 *
 * If neither EXPECT_* is set, the script runs Stage-1 checks only
 * (post-deploy, pre-transferOwnership).
 *
 * Usage:
 *   PROVENANCE_MANIFEST=contracts/deployments/provenance-base-1234.json \
 *     npx hardhat run scripts/verify-royalty-distributor-v2-deployment.js \
 *     --network base
 *
 * Exit codes:
 *   0 = all on-chain state matches expected ceremony stage
 *   1 = mismatch (caller must investigate before proceeding)
 *
 * R-2026-05-08-1 / A-08-CEREMONY-PLAN-1 §4.3 / §4.6 reference this
 * script. Do NOT proceed with operator-side migration (networks.py
 * commit) until this script exits 0 on Stage 3.
 */
const hre = require("hardhat");
const fs = require("fs");

// Canonical Base mainnet addresses for cross-check (fail loud if
// manifest is wrong). Mirrors the deploy-script's canonical pins.
const CANONICAL_FTNS_BASE = "0x5276a3756C85f2E9e46f6D34386167a209aa16e5";
const CANONICAL_FOUNDATION_SAFE = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791";

async function main() {
  const manifestPath = process.env.PROVENANCE_MANIFEST;
  if (!manifestPath) {
    throw new Error("PROVENANCE_MANIFEST env var required");
  }
  if (!fs.existsSync(manifestPath)) {
    throw new Error(
      `PROVENANCE_MANIFEST points to ${manifestPath} which does not exist`,
    );
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  console.log(`\n=== A-08 v2 RoyaltyDistributor verification ===`);
  console.log(`Manifest: ${manifestPath}`);
  console.log(`Network:  ${manifest.network}`);
  console.log(`ChainId:  ${manifest.chainId}`);
  console.log(`Deployer: ${manifest.deployer}`);

  // Sanity: manifest must agree with --network.
  if (manifest.network !== hre.network.name) {
    throw new Error(
      `manifest.network=${manifest.network} != --network=${hre.network.name}`,
    );
  }

  const expected = manifest.contracts;
  for (const k of [
    "RoyaltyDistributor",
    "FTNSToken",
    "ProvenanceRegistry",
    "NetworkTreasury",
  ]) {
    if (!expected[k]) throw new Error(`manifest missing contracts.${k}`);
  }
  console.log(`\nExpected addresses:`);
  for (const [k, v] of Object.entries(expected)) {
    console.log(`  ${k}: ${v}`);
  }

  // Sanity: chainId on connected RPC must match manifest.
  const onChainChainId = (await hre.ethers.provider.getNetwork()).chainId;
  if (onChainChainId.toString() !== manifest.chainId) {
    throw new Error(
      `RPC chainId=${onChainChainId} != manifest.chainId=${manifest.chainId}`,
    );
  }

  // Cross-check canonical mainnet addresses if the manifest is for
  // base mainnet (chainId 8453). For Sepolia rehearsals
  // (chainId 84532), skip the canonical check — Sepolia uses test
  // tokens.
  if (manifest.chainId === "8453") {
    if (
      expected.FTNSToken.toLowerCase() !== CANONICAL_FTNS_BASE.toLowerCase()
    ) {
      throw new Error(
        `Mainnet manifest FTNSToken=${expected.FTNSToken} != canonical ` +
        `${CANONICAL_FTNS_BASE} — A-08 plan §3 requires canonical pin.`,
      );
    }
    if (
      expected.NetworkTreasury.toLowerCase() !==
      CANONICAL_FOUNDATION_SAFE.toLowerCase()
    ) {
      throw new Error(
        `Mainnet manifest NetworkTreasury=${expected.NetworkTreasury} != ` +
        `canonical Safe ${CANONICAL_FOUNDATION_SAFE} — A-08 plan §3.`,
      );
    }
    console.log(`  ✓ canonical Base mainnet addresses confirmed`);
  } else {
    console.log(`  (Skipping canonical-pin check; chainId=${manifest.chainId} is not Base mainnet)`);
  }

  // 1. v2 RoyaltyDistributor must have bytecode.
  console.log(`\nChecking v2 RoyaltyDistributor…`);
  const distCode = await hre.ethers.provider.getCode(expected.RoyaltyDistributor);
  if (distCode === "0x" || distCode === "0x0") {
    throw new Error(
      `RoyaltyDistributor at ${expected.RoyaltyDistributor} has no bytecode`,
    );
  }
  console.log(`  bytecode: ${(distCode.length / 2 - 1)} bytes ✓`);

  // 2. v2-specific: bytecode must contain the recoverStranded
  // function selector. Without this, we deployed v1 bytecode by
  // mistake — that would silently regress the A-08 fix.
  // recoverStranded(address) selector = first 4 bytes of
  //   keccak256("recoverStranded(address)").
  const recoverStrandedSelector = hre.ethers
    .keccak256(hre.ethers.toUtf8Bytes("recoverStranded(address)"))
    .slice(2, 10); // first 4 bytes (8 hex chars) after "0x"
  if (!distCode.toLowerCase().includes(recoverStrandedSelector.toLowerCase())) {
    throw new Error(
      `RoyaltyDistributor bytecode does NOT contain the recoverStranded ` +
      `selector 0x${recoverStrandedSelector}. This is a v1 deployment, ` +
      `not v2. ABORT — A-08 fix not applied.`,
    );
  }
  console.log(`  ✓ recoverStranded selector 0x${recoverStrandedSelector} present (v2 confirmed)`);

  // Likewise totalClaimable() selector — uint256 getter.
  const totalClaimableSelector = hre.ethers
    .keccak256(hre.ethers.toUtf8Bytes("totalClaimable()"))
    .slice(2, 10);
  if (!distCode.toLowerCase().includes(totalClaimableSelector.toLowerCase())) {
    throw new Error(
      `RoyaltyDistributor bytecode does NOT contain the totalClaimable ` +
      `getter selector 0x${totalClaimableSelector}. This is a v1 ` +
      `deployment. ABORT.`,
    );
  }
  console.log(`  ✓ totalClaimable selector 0x${totalClaimableSelector} present`);

  // 3. Immutable getters match expected.
  const dist = new hre.ethers.Contract(
    expected.RoyaltyDistributor,
    [
      "function ftns() view returns (address)",
      "function registry() view returns (address)",
      "function networkTreasury() view returns (address)",
      "function owner() view returns (address)",
      "function pendingOwner() view returns (address)",
      "function totalClaimable() view returns (uint256)",
    ],
    hre.ethers.provider,
  );

  const ftnsAddr = await dist.ftns();
  if (ftnsAddr.toLowerCase() !== expected.FTNSToken.toLowerCase()) {
    throw new Error(
      `RoyaltyDistributor.ftns()=${ftnsAddr} != expected.FTNSToken=${expected.FTNSToken}`,
    );
  }
  console.log(`  ✓ ftns()           == ${ftnsAddr}`);

  const registryAddr = await dist.registry();
  if (registryAddr.toLowerCase() !== expected.ProvenanceRegistry.toLowerCase()) {
    throw new Error(
      `RoyaltyDistributor.registry()=${registryAddr} != expected.ProvenanceRegistry=${expected.ProvenanceRegistry}`,
    );
  }
  console.log(`  ✓ registry()       == ${registryAddr}`);

  const treasuryAddr = await dist.networkTreasury();
  if (treasuryAddr.toLowerCase() !== expected.NetworkTreasury.toLowerCase()) {
    throw new Error(
      `RoyaltyDistributor.networkTreasury()=${treasuryAddr} != ` +
      `expected.NetworkTreasury=${expected.NetworkTreasury}`,
    );
  }
  console.log(`  ✓ networkTreasury()== ${treasuryAddr}`);

  // 4. Fresh-deploy invariant: totalClaimable must be 0.
  const tc = await dist.totalClaimable();
  if (tc !== 0n) {
    throw new Error(
      `RoyaltyDistributor.totalClaimable()=${tc} != 0. This is NOT a ` +
      `fresh deploy — either the contract has accrued royalties already, ` +
      `or the totalClaimable accumulator was initialized incorrectly.`,
    );
  }
  console.log(`  ✓ totalClaimable() == 0 (fresh deploy)`);

  // 5. Stage-specific ownership assertions.
  const owner = await dist.owner();
  const pendingOwner = await dist.pendingOwner();
  console.log(`\nOwnership state:`);
  console.log(`  owner()        == ${owner}`);
  console.log(`  pendingOwner() == ${pendingOwner}`);

  const expectFinal = process.env.EXPECT_FINAL_OWNER;
  const expectPending = process.env.EXPECT_PENDING_OWNER;

  if (expectFinal) {
    // Stage 3: post-acceptOwnership.
    if (owner.toLowerCase() !== expectFinal.toLowerCase()) {
      throw new Error(
        `Stage 3 assertion failed: owner()=${owner} != ` +
        `EXPECT_FINAL_OWNER=${expectFinal}`,
      );
    }
    if (pendingOwner !== "0x0000000000000000000000000000000000000000") {
      throw new Error(
        `Stage 3 assertion failed: pendingOwner()=${pendingOwner} != 0x0 ` +
        `(acceptOwnership clears pendingOwner; non-zero means accept-tx ` +
        `did not execute or did not target this contract)`,
      );
    }
    console.log(`  ✓ Stage 3 (post-acceptOwnership) ownership posture confirmed`);
  } else if (expectPending) {
    // Stage 2: post-transferOwnership, pre-acceptOwnership.
    if (pendingOwner.toLowerCase() !== expectPending.toLowerCase()) {
      throw new Error(
        `Stage 2 assertion failed: pendingOwner()=${pendingOwner} != ` +
        `EXPECT_PENDING_OWNER=${expectPending}`,
      );
    }
    if (owner.toLowerCase() !== manifest.deployer.toLowerCase()) {
      throw new Error(
        `Stage 2 assertion failed: owner()=${owner} != deployer=${manifest.deployer}. ` +
        `transferOwnership should NOT change current owner; only sets pendingOwner.`,
      );
    }
    console.log(`  ✓ Stage 2 (post-transferOwnership) ownership posture confirmed`);
  } else {
    // Stage 1: post-deploy, pre-transferOwnership.
    if (owner.toLowerCase() !== manifest.deployer.toLowerCase()) {
      throw new Error(
        `Stage 1 assertion failed: owner()=${owner} != deployer=${manifest.deployer}. ` +
        `Post-deploy, owner should equal the address that deployed.`,
      );
    }
    if (pendingOwner !== "0x0000000000000000000000000000000000000000") {
      throw new Error(
        `Stage 1 assertion failed: pendingOwner()=${pendingOwner} != 0x0. ` +
        `Fresh-deploy state should have no pending transfer.`,
      );
    }
    console.log(`  ✓ Stage 1 (post-deploy, pre-transferOwnership) confirmed`);
    console.log(`    Next: deployer must call transferOwnership(<Safe address>) (plan §4.4)`);
  }

  console.log(`\n=== All assertions passed ===\n`);
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(`\n=== VERIFICATION FAILED ===`);
    console.error(err.message || err);
    process.exit(1);
  });
