/*
 * Phase 1.3 Task 8 — post-deploy state verification.
 *
 * Reads a provenance-<network>-*.json manifest produced by
 * deploy-provenance.js, calls the immutable getters on RoyaltyDistributor,
 * and asserts they match the manifest. Fails loudly if any wiring
 * doesn't match — RoyaltyDistributor's constructor args are immutable,
 * so this verification is the LAST chance to catch a bad deploy before
 * the production system points at the wrong contract.
 *
 * Required env var:
 *   PROVENANCE_MANIFEST  - path to provenance-<network>-*.json
 *
 * Usage:
 *   PROVENANCE_MANIFEST=contracts/deployments/provenance-base-1234.json \
 *     npx hardhat run scripts/verify-provenance-deployment.js --network base
 *
 * Exit codes:
 *   0 = all on-chain state matches manifest
 *   1 = mismatch (caller must investigate before trusting the deploy)
 */
const hre = require("hardhat");
const fs = require("fs");

async function main() {
  const manifestPath = process.env.PROVENANCE_MANIFEST;
  if (!manifestPath) {
    throw new Error("PROVENANCE_MANIFEST env var required");
  }
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`PROVENANCE_MANIFEST points to ${manifestPath} which does not exist`);
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  console.log(`\n=== Verifying provenance deployment ===`);
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
  for (const k of ["ProvenanceRegistry", "RoyaltyDistributor", "FTNSToken", "NetworkTreasury"]) {
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

  // 1. ProvenanceRegistry must have bytecode.
  console.log(`\nChecking ProvenanceRegistry…`);
  const regCode = await hre.ethers.provider.getCode(expected.ProvenanceRegistry);
  if (regCode === "0x" || regCode === "0x0") {
    throw new Error(`ProvenanceRegistry at ${expected.ProvenanceRegistry} has no bytecode`);
  }
  console.log(`  bytecode: ${(regCode.length / 2 - 1)} bytes ✓`);

  // 2. RoyaltyDistributor: bytecode + immutable getters.
  console.log(`\nChecking RoyaltyDistributor…`);
  const distCode = await hre.ethers.provider.getCode(expected.RoyaltyDistributor);
  if (distCode === "0x" || distCode === "0x0") {
    throw new Error(`RoyaltyDistributor at ${expected.RoyaltyDistributor} has no bytecode`);
  }
  console.log(`  bytecode: ${(distCode.length / 2 - 1)} bytes ✓`);

  const dist = new hre.ethers.Contract(
    expected.RoyaltyDistributor,
    [
      "function ftns() view returns (address)",
      "function registry() view returns (address)",
      "function networkTreasury() view returns (address)",
      "function NETWORK_FEE_BPS() view returns (uint16)",
    ],
    hre.ethers.provider,
  );

  const checks = [
    { name: "ftns()", actual: await dist.ftns(), expected: expected.FTNSToken },
    { name: "registry()", actual: await dist.registry(), expected: expected.ProvenanceRegistry },
    { name: "networkTreasury()", actual: await dist.networkTreasury(), expected: expected.NetworkTreasury },
  ];

  let mismatches = 0;
  for (const c of checks) {
    const actualLower = c.actual.toLowerCase();
    const expectedLower = c.expected.toLowerCase();
    if (actualLower !== expectedLower) {
      console.error(`  ❌ ${c.name}: on-chain=${c.actual} != manifest=${c.expected}`);
      mismatches += 1;
    } else {
      console.log(`  ✓ ${c.name}: ${c.actual}`);
    }
  }

  // 3. NETWORK_FEE_BPS sanity (constant in source = 200, i.e., 2.00%).
  const feeBps = await dist.NETWORK_FEE_BPS();
  if (feeBps !== 200n) {
    console.error(`  ❌ NETWORK_FEE_BPS: on-chain=${feeBps}, expected 200 (2.00%)`);
    mismatches += 1;
  } else {
    console.log(`  ✓ NETWORK_FEE_BPS: ${feeBps} (2.00%)`);
  }

  // 4. Network treasury must be a contract on mainnet (Safe). Sanity-
  //    repeat the deploy-time check — guards against the rare case
  //    where deploy ran with a Safe that's since been mutated/destroyed
  //    (Safe destruction shouldn't happen but the cost of checking is zero).
  if (manifest.network === "base" || manifest.network === "mainnet") {
    const treasuryCode = await hre.ethers.provider.getCode(expected.NetworkTreasury);
    if (treasuryCode === "0x" || treasuryCode === "0x0") {
      console.error(`  ❌ NetworkTreasury ${expected.NetworkTreasury} has no bytecode on mainnet`);
      mismatches += 1;
    } else {
      console.log(`  ✓ NetworkTreasury bytecode: ${(treasuryCode.length / 2 - 1)} bytes`);
    }
  }

  // 5. FTNS sanity — symbol must match what was used at deploy time.
  console.log(`\nChecking FTNS…`);
  const ftns = new hre.ethers.Contract(
    expected.FTNSToken,
    ["function symbol() view returns (string)"],
    hre.ethers.provider,
  );
  const symbol = await ftns.symbol();
  if (symbol !== "FTNS" && symbol !== "MFTNS") {
    console.error(`  ❌ FTNS.symbol()=${symbol}, expected FTNS or MFTNS`);
    mismatches += 1;
  } else {
    console.log(`  ✓ FTNS.symbol(): ${symbol}`);
  }

  if (mismatches > 0) {
    console.error(`\n❌ VERIFICATION FAILED: ${mismatches} mismatch${mismatches === 1 ? "" : "es"}.`);
    console.error(`   The deployed contracts do NOT match the manifest. Investigate before`);
    console.error(`   trusting this deploy in production. RoyaltyDistributor is immutable —`);
    console.error(`   re-deploy is the only fix for wrong constructor args.`);
    process.exitCode = 1;
    return;
  }

  console.log(`\n✅ All on-chain state matches manifest.`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
