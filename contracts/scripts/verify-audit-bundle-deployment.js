/*
 * Phase 7 Task 9 / Phase 7.1 Task 9 — post-deploy state verification.
 *
 * Reads an audit-bundle-<network>-*.json manifest produced by
 * deploy-audit-bundle.js, calls every cross-wire getter on the deployed
 * contracts, and asserts state matches the manifest. Fails loudly with
 * per-mismatch diagnostics — these contracts are Ownable but cross-wires
 * are NOT immutable; an attacker who briefly held ownership could have
 * mutated them. Verifying before trusting the deploy is the last
 * checkpoint.
 *
 * Mirrors verify-provenance-deployment.js for the audit-bundle stack
 * (EscrowPool + BatchSettlementRegistry + SignatureVerifier + StakeBond).
 *
 * Required env var:
 *   AUDIT_BUNDLE_MANIFEST  - path to audit-bundle-<network>-*.json
 *
 * Optional env vars:
 *   EXPECTED_OWNER         - expected owner() on all 3 Ownable contracts
 *                            (EscrowPool, Registry, StakeBond). If set,
 *                            mismatches fail. If unset, owner is logged
 *                            but not compared (useful pre-handoff).
 *                            Set this to your Foundation Safe address
 *                            after transfer-ownership.js runs.
 *
 * Usage:
 *   AUDIT_BUNDLE_MANIFEST=contracts/deployments/audit-bundle-base-1234.json \
 *     EXPECTED_OWNER=0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 \
 *     npx hardhat run scripts/verify-audit-bundle-deployment.js --network base
 *
 * Exit codes:
 *   0 = all on-chain state matches manifest
 *   1 = mismatch (caller must investigate before trusting the deploy)
 */
const hre = require("hardhat");
const fs = require("fs");

async function main() {
  const manifestPath = process.env.AUDIT_BUNDLE_MANIFEST;
  if (!manifestPath) {
    throw new Error("AUDIT_BUNDLE_MANIFEST env var required");
  }
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`AUDIT_BUNDLE_MANIFEST points to ${manifestPath} which does not exist`);
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  console.log(`\n=== Verifying audit-bundle deployment ===`);
  console.log(`Manifest:  ${manifestPath}`);
  console.log(`Bundle:    ${manifest.bundle || "(no bundle name)"}`);
  console.log(`Phases:    ${(manifest.phases || []).join(", ")}`);
  console.log(`Network:   ${manifest.network}`);
  console.log(`ChainId:   ${manifest.chainId}`);
  console.log(`Deployer:  ${manifest.deployer}`);
  console.log(`Verifier:  ${manifest.params && manifest.params.verifierKind || "(unspecified)"}`);

  // Sanity: manifest must agree with --network.
  if (manifest.network !== hre.network.name) {
    throw new Error(
      `manifest.network=${manifest.network} != --network=${hre.network.name}`,
    );
  }

  const expected = manifest.contracts;
  const required = [
    "EscrowPool",
    "BatchSettlementRegistry",
    "SignatureVerifier",
    "StakeBond",
    "FTNSToken",
    "FoundationReserveWallet",
  ];
  for (const k of required) {
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

  let mismatches = 0;
  const fail = (msg) => { console.error(`  ❌ ${msg}`); mismatches += 1; };
  const ok = (msg) => { console.log(`  ✓ ${msg}`); };

  // ── 1. All 4 contracts have bytecode ───────────────────────────────
  console.log(`\nBytecode presence`);
  for (const name of ["EscrowPool", "BatchSettlementRegistry", "SignatureVerifier", "StakeBond"]) {
    const code = await hre.ethers.provider.getCode(expected[name]);
    if (code === "0x" || code === "0x0") {
      fail(`${name} at ${expected[name]} has no bytecode`);
    } else {
      ok(`${name}: ${(code.length / 2 - 1)} bytes`);
    }
  }

  // FTNS too — sanity-repeat from deploy-time check.
  const ftnsCode = await hre.ethers.provider.getCode(expected.FTNSToken);
  if (ftnsCode === "0x" || ftnsCode === "0x0") {
    fail(`FTNSToken at ${expected.FTNSToken} has no bytecode`);
  } else {
    ok(`FTNSToken: ${(ftnsCode.length / 2 - 1)} bytes`);
  }

  // FoundationReserveWallet — on mainnet should be a contract (Safe).
  // On testnet/local it can be an EOA. Log only.
  const foundationCode = await hre.ethers.provider.getCode(expected.FoundationReserveWallet);
  const foundationIsContract = foundationCode !== "0x" && foundationCode !== "0x0";
  console.log(
    `  ${foundationIsContract ? "✓" : "ℹ"} FoundationReserveWallet: ` +
    `${foundationIsContract ? `${(foundationCode.length / 2 - 1)} bytes (contract — likely Safe)` : "EOA"}`,
  );
  if (manifest.network === "base" || manifest.network === "mainnet") {
    if (!foundationIsContract) {
      fail(
        `FoundationReserveWallet ${expected.FoundationReserveWallet} is an EOA on mainnet — ` +
        `expected a Safe contract. Hot wallets cannot replace 2-of-3 multi-sig.`,
      );
    }
  }

  // ── 2. Cross-wire invariants on EscrowPool + Registry + StakeBond ──
  console.log(`\nCross-wire invariants`);

  const escrow = new hre.ethers.Contract(
    expected.EscrowPool,
    [
      "function settlementRegistry() view returns (address)",
      "function owner() view returns (address)",
    ],
    hre.ethers.provider,
  );
  const registry = new hre.ethers.Contract(
    expected.BatchSettlementRegistry,
    [
      "function escrowPool() view returns (address)",
      "function signatureVerifier() view returns (address)",
      "function stakeBond() view returns (address)",
      "function challengeWindowSeconds() view returns (uint256)",
      "function owner() view returns (address)",
    ],
    hre.ethers.provider,
  );
  const stakeBond = new hre.ethers.Contract(
    expected.StakeBond,
    [
      "function slasher() view returns (address)",
      "function foundationReserveWallet() view returns (address)",
      "function unbondDelaySeconds() view returns (uint256)",
      "function ftns() view returns (address)",
      "function owner() view returns (address)",
    ],
    hre.ethers.provider,
  );

  const compare = async (label, getter, expectedAddr) => {
    let actual;
    try {
      actual = await getter();
    } catch (e) {
      fail(`${label}: getter reverted (${e.message.slice(0, 80)})`);
      return;
    }
    if (typeof actual === "string") {
      if (actual.toLowerCase() === expectedAddr.toLowerCase()) {
        ok(`${label}: ${actual}`);
      } else {
        fail(`${label}: on-chain=${actual} != manifest=${expectedAddr}`);
      }
    } else {
      // numeric (BigInt)
      if (actual.toString() === expectedAddr.toString()) {
        ok(`${label}: ${actual.toString()}`);
      } else {
        fail(`${label}: on-chain=${actual.toString()} != expected=${expectedAddr.toString()}`);
      }
    }
  };

  await compare("escrow.settlementRegistry", () => escrow.settlementRegistry(), expected.BatchSettlementRegistry);
  await compare("registry.escrowPool", () => registry.escrowPool(), expected.EscrowPool);
  await compare("registry.signatureVerifier", () => registry.signatureVerifier(), expected.SignatureVerifier);
  await compare("registry.stakeBond", () => registry.stakeBond(), expected.StakeBond);
  await compare("stakeBond.slasher", () => stakeBond.slasher(), expected.BatchSettlementRegistry);
  await compare("stakeBond.foundationReserveWallet", () => stakeBond.foundationReserveWallet(), expected.FoundationReserveWallet);
  await compare("stakeBond.ftns", () => stakeBond.ftns(), expected.FTNSToken);

  // ── 3. Initialization params match manifest ────────────────────────
  console.log(`\nInitialization params`);
  if (manifest.params && manifest.params.challengeWindowSeconds) {
    await compare(
      "registry.challengeWindowSeconds",
      () => registry.challengeWindowSeconds(),
      BigInt(manifest.params.challengeWindowSeconds),
    );
  }
  if (manifest.params && manifest.params.unbondDelaySeconds) {
    await compare(
      "stakeBond.unbondDelaySeconds",
      () => stakeBond.unbondDelaySeconds(),
      BigInt(manifest.params.unbondDelaySeconds),
    );
  }

  // ── 4. Ownership ───────────────────────────────────────────────────
  // Pre-handoff: owners == deployer.
  // Post-handoff: owners == EXPECTED_OWNER (typically Foundation Safe).
  // If EXPECTED_OWNER is unset, log the values without comparing.
  // L2 audit MEDIUM B-OWNABLE-1: contracts are now Ownable2Step.
  // After transfer-ownership.js runs, owner() is still deployer until
  // the multisig calls acceptOwnership(). pendingOwner() is what's
  // changed mid-ceremony. Surface both fields so operators can tell
  // which stage of the handoff they're verifying.
  console.log(`\nOwnership`);
  const expectedOwner = process.env.EXPECTED_OWNER;
  const ownerAbi = [
    "function owner() view returns (address)",
    "function pendingOwner() view returns (address)",
  ];
  const ownerChecks = [
    { name: "EscrowPool", address: expected.EscrowPool },
    { name: "BatchSettlementRegistry", address: expected.BatchSettlementRegistry },
    { name: "StakeBond", address: expected.StakeBond },
  ];
  for (const c of ownerChecks) {
    let actualOwner, pending;
    try {
      const ctr = new hre.ethers.Contract(c.address, ownerAbi, hre.ethers.provider);
      actualOwner = await ctr.owner();
      try {
        pending = await ctr.pendingOwner();
      } catch (_e) {
        pending = null; // contract is not Ownable2Step
      }
    } catch (e) {
      fail(`${c.name}.owner(): getter reverted`);
      continue;
    }
    if (expectedOwner) {
      if (actualOwner.toLowerCase() === expectedOwner.toLowerCase()) {
        ok(`${c.name}.owner(): ${actualOwner} (matches EXPECTED_OWNER)`);
      } else if (pending && pending.toLowerCase() === expectedOwner.toLowerCase()) {
        console.log(
          `  ⏳ ${c.name}: owner=${actualOwner}, pendingOwner=${pending} ` +
          `(handoff in progress — multisig must acceptOwnership)`,
        );
      } else {
        fail(`${c.name}.owner(): on-chain=${actualOwner} != EXPECTED_OWNER=${expectedOwner}` +
          (pending ? ` (pendingOwner=${pending})` : ""));
      }
    } else {
      console.log(
        `  ℹ ${c.name}.owner(): ${actualOwner}` +
        (pending && pending !== hre.ethers.ZeroAddress ? `, pendingOwner=${pending}` : "") +
        ` (no EXPECTED_OWNER set — pre-handoff is normal; post-handoff set EXPECTED_OWNER=Foundation Safe)`,
      );
    }
  }

  // ── 5. FTNS sanity ─────────────────────────────────────────────────
  console.log(`\nFTNS sanity`);
  try {
    const ftns = new hre.ethers.Contract(
      expected.FTNSToken,
      ["function symbol() view returns (string)"],
      hre.ethers.provider,
    );
    const symbol = await ftns.symbol();
    if (symbol !== "FTNS" && symbol !== "MFTNS") {
      fail(`FTNS.symbol()=${symbol}, expected FTNS or MFTNS`);
    } else {
      ok(`FTNS.symbol(): ${symbol}`);
    }
  } catch (e) {
    fail(`FTNS.symbol(): call failed (${e.message.slice(0, 80)})`);
  }

  // ── Final ──────────────────────────────────────────────────────────
  if (mismatches > 0) {
    console.error(`\n❌ VERIFICATION FAILED: ${mismatches} mismatch${mismatches === 1 ? "" : "es"}.`);
    console.error(`   The deployed contracts do NOT match the manifest. Investigate before`);
    console.error(`   trusting this deploy in production. Cross-wire mismatches indicate`);
    console.error(`   either a bad deploy or post-deploy mutation by an attacker who held`);
    console.error(`   ownership (transfer-ownership.js cuts that risk window — confirm it ran).`);
    process.exitCode = 1;
    return;
  }

  console.log(`\n✅ All on-chain state matches manifest.`);
  if (!process.env.EXPECTED_OWNER) {
    console.log(
      `\nℹ️  Note: EXPECTED_OWNER was unset; ownership was logged but not compared.\n` +
      `   Re-run this script with EXPECTED_OWNER=<Foundation Safe address> after\n` +
      `   transfer-ownership.js completes to assert post-handoff state.`,
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
