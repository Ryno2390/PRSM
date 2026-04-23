/*
 * Bundled deploy for the Phase 3.1 + 7 + 7.1 audit scope.
 *
 * Deploys in the order required by the audit-bundle coordinator
 * (docs/2026-04-21-audit-bundle-coordinator.md §7):
 *
 *   1. EscrowPool                   (Phase 3.1 substrate)
 *   2. BatchSettlementRegistry       (Phase 3.1 + 7 + 7.1 baked in)
 *   3. Signature verifier            (Mock on testnet; production Ed25519 on mainnet)
 *   4. StakeBond                    (Phase 7)
 *   5. Cross-wire (six setter calls)
 *
 * Required env vars:
 *   FTNS_TOKEN_ADDRESS             - ERC20 token for escrow + staking
 *   FOUNDATION_RESERVE_WALLET      - recipient for foundation slash share
 *   SIGNATURE_VERIFIER_ADDRESS     - (optional) pre-deployed verifier; script
 *                                     deploys MockSignatureVerifier if absent
 *
 * Optional env vars:
 *   CHALLENGE_WINDOW_SECONDS       - registry init (default 86400 = 1 day)
 *   UNBOND_DELAY_SECONDS           - stake-bond init (default 604800 = 7 days)
 *   AUTO_VERIFY                    - 1 to auto-verify on Basescan
 *
 * Usage:
 *   npx hardhat run scripts/deploy-audit-bundle.js --network hardhat
 *   npx hardhat run scripts/deploy-audit-bundle.js --network base-sepolia
 *   npx hardhat run scripts/deploy-audit-bundle.js --network base
 *
 * The script produces contracts/deployments/audit-bundle-<network>-<ts>.json
 * with every deployed address + cross-wire tx hash for mainnet-day handoff.
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;
  const ftnsAddress = process.env.FTNS_TOKEN_ADDRESS;
  const foundationWallet = process.env.FOUNDATION_RESERVE_WALLET;
  const preDeployedVerifier = process.env.SIGNATURE_VERIFIER_ADDRESS;
  const challengeWindow = BigInt(process.env.CHALLENGE_WINDOW_SECONDS || "86400");
  const unbondDelay = BigInt(process.env.UNBOND_DELAY_SECONDS || "604800");

  if (!ftnsAddress) throw new Error("FTNS_TOKEN_ADDRESS env var required");
  if (!foundationWallet) throw new Error("FOUNDATION_RESERVE_WALLET env var required");

  console.log(`\n=== Deploying Phase 3.1+7+7.1 audit bundle to ${network} ===`);

  // ── Preflight ───────────────────────────────────────────────────────
  const ftnsChecksum = hre.ethers.getAddress(ftnsAddress);
  const foundationChecksum = hre.ethers.getAddress(foundationWallet);
  if (ftnsChecksum === hre.ethers.ZeroAddress) throw new Error("FTNS_TOKEN_ADDRESS is zero");
  if (foundationChecksum === hre.ethers.ZeroAddress) throw new Error("FOUNDATION_RESERVE_WALLET is zero");

  const code = await hre.ethers.provider.getCode(ftnsChecksum);
  if (code === "0x" || code === "0x0") {
    throw new Error(`no contract at FTNS_TOKEN_ADDRESS ${ftnsChecksum} on ${network}`);
  }

  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;

  console.log(`Deployer:            ${deployer.address}`);
  console.log(`Deployer balance:    ${hre.ethers.formatEther(balance)} ETH`);
  console.log(`Chain id:            ${chainId}`);
  console.log(`FTNS token:          ${ftnsChecksum}`);
  console.log(`Foundation wallet:   ${foundationChecksum}`);
  console.log(`Challenge window:    ${challengeWindow}s`);
  console.log(`Unbond delay:        ${unbondDelay}s`);

  if (balance === 0n) throw new Error("Deployer has zero balance");

  // Mainnet production-verifier guard — refuse to deploy the mock to Base mainnet.
  const isMainnet = network === "base" || network === "mainnet";
  if (isMainnet && !preDeployedVerifier) {
    throw new Error(
      "SIGNATURE_VERIFIER_ADDRESS is required on mainnet. The MockSignatureVerifier " +
      "is test-only (verify() returns a flag set by anyone). Deploy an audited Ed25519 " +
      "implementation first, then pass its address via SIGNATURE_VERIFIER_ADDRESS."
    );
  }

  // Foundation wallet may equal deployer on testnet only. On mainnet the wallet
  // should be the 2-of-3 multi-sig. Belt-and-suspenders check for mainnet.
  if (isMainnet && foundationChecksum.toLowerCase() === deployer.address.toLowerCase()) {
    throw new Error(
      `FOUNDATION_RESERVE_WALLET must not equal deployer on mainnet. ` +
      `Use the Foundation 2-of-3 multi-sig per PRSM-GOV-1 §8.`
    );
  }

  const deployments = {};
  const txHashes = {};

  // ── 1. EscrowPool ───────────────────────────────────────────────────
  console.log("\n[1/5] Deploying EscrowPool…");
  const EscrowPool = await hre.ethers.getContractFactory("EscrowPool");
  // Constructor: (initialOwner, ftnsAddress, initialRegistry). Registry is
  // deployed next; pass address(0) and wire via setSettlementRegistry below.
  const escrow = await EscrowPool.deploy(deployer.address, ftnsChecksum, hre.ethers.ZeroAddress);
  await escrow.waitForDeployment();
  deployments.EscrowPool = await escrow.getAddress();
  console.log(`   EscrowPool:          ${deployments.EscrowPool}`);

  // ── 2. BatchSettlementRegistry ─────────────────────────────────────
  console.log("\n[2/5] Deploying BatchSettlementRegistry…");
  const Registry = await hre.ethers.getContractFactory("BatchSettlementRegistry");
  const registry = await Registry.deploy(deployer.address, challengeWindow);
  await registry.waitForDeployment();
  deployments.BatchSettlementRegistry = await registry.getAddress();
  console.log(`   BatchSettlementRegistry: ${deployments.BatchSettlementRegistry}`);

  // ── 3. Signature verifier ──────────────────────────────────────────
  let verifierAddress;
  if (preDeployedVerifier) {
    verifierAddress = hre.ethers.getAddress(preDeployedVerifier);
    console.log(`\n[3/5] Using pre-deployed verifier at ${verifierAddress}`);
  } else {
    console.log("\n[3/5] Deploying MockSignatureVerifier (TEST-ONLY)…");
    const Verifier = await hre.ethers.getContractFactory("MockSignatureVerifier");
    const verifier = await Verifier.deploy();
    await verifier.waitForDeployment();
    verifierAddress = await verifier.getAddress();
  }
  deployments.SignatureVerifier = verifierAddress;
  console.log(`   SignatureVerifier:   ${verifierAddress}`);

  // ── 4. StakeBond ────────────────────────────────────────────────────
  console.log("\n[4/5] Deploying StakeBond…");
  const StakeBond = await hre.ethers.getContractFactory("StakeBond");
  const stakeBond = await StakeBond.deploy(deployer.address, ftnsChecksum, unbondDelay);
  await stakeBond.waitForDeployment();
  deployments.StakeBond = await stakeBond.getAddress();
  console.log(`   StakeBond:           ${deployments.StakeBond}`);

  // ── 5. Cross-wire ───────────────────────────────────────────────────
  console.log("\n[5/5] Cross-wiring…");

  let tx;

  tx = await escrow.setSettlementRegistry(deployments.BatchSettlementRegistry);
  await tx.wait();
  txHashes.escrow_setSettlementRegistry = tx.hash;
  console.log(`   EscrowPool.setSettlementRegistry → registry (${tx.hash.slice(0, 10)}…)`);

  tx = await registry.setEscrowPool(deployments.EscrowPool);
  await tx.wait();
  txHashes.registry_setEscrowPool = tx.hash;
  console.log(`   Registry.setEscrowPool → escrow (${tx.hash.slice(0, 10)}…)`);

  tx = await registry.setSignatureVerifier(verifierAddress);
  await tx.wait();
  txHashes.registry_setSignatureVerifier = tx.hash;
  console.log(`   Registry.setSignatureVerifier → verifier (${tx.hash.slice(0, 10)}…)`);

  tx = await registry.setStakeBond(deployments.StakeBond);
  await tx.wait();
  txHashes.registry_setStakeBond = tx.hash;
  console.log(`   Registry.setStakeBond → stakeBond (${tx.hash.slice(0, 10)}…)`);

  tx = await stakeBond.setSlasher(deployments.BatchSettlementRegistry);
  await tx.wait();
  txHashes.stakeBond_setSlasher = tx.hash;
  console.log(`   StakeBond.setSlasher → registry (${tx.hash.slice(0, 10)}…)`);

  tx = await stakeBond.setFoundationReserveWallet(foundationChecksum);
  await tx.wait();
  txHashes.stakeBond_setFoundationReserveWallet = tx.hash;
  console.log(`   StakeBond.setFoundationReserveWallet → foundation (${tx.hash.slice(0, 10)}…)`);

  // ── Post-deploy invariant checks ───────────────────────────────────
  console.log("\nPost-deploy invariant checks…");
  const escrowRegistry = await escrow.settlementRegistry();
  const registryEscrow = await registry.escrowPool();
  const registryVerifier = await registry.signatureVerifier();
  const registryBond = await registry.stakeBond();
  const bondSlasher = await stakeBond.slasher();
  const bondFoundation = await stakeBond.foundationReserveWallet();

  const check = (label, got, expected) => {
    const ok = got.toLowerCase() === expected.toLowerCase();
    console.log(`   ${ok ? "✅" : "❌"} ${label}: ${got}`);
    if (!ok) throw new Error(`${label} mismatch: got ${got}, expected ${expected}`);
  };
  check("escrow.settlementRegistry", escrowRegistry, deployments.BatchSettlementRegistry);
  check("registry.escrowPool", registryEscrow, deployments.EscrowPool);
  check("registry.signatureVerifier", registryVerifier, verifierAddress);
  check("registry.stakeBond", registryBond, deployments.StakeBond);
  check("stakeBond.slasher", bondSlasher, deployments.BatchSettlementRegistry);
  check("stakeBond.foundationReserveWallet", bondFoundation, foundationChecksum);

  // ── Manifest ────────────────────────────────────────────────────────
  const manifest = {
    bundle: "audit-bundle",
    phases: ["3.1", "7", "7.1"],
    network,
    chainId: chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    params: {
      challengeWindowSeconds: challengeWindow.toString(),
      unbondDelaySeconds: unbondDelay.toString(),
      verifierIsMock: !preDeployedVerifier,
    },
    contracts: {
      ...deployments,
      FTNSToken: ftnsChecksum,
      FoundationReserveWallet: foundationChecksum,
    },
    crossWireTxHashes: txHashes,
  };

  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `audit-bundle-${network}-${Date.now()}.json`);
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  // ── Optional Basescan verification ─────────────────────────────────
  const verifyEnabled = process.env.AUTO_VERIFY === "1";
  const isBase = network === "base" || network === "base-sepolia";
  if (verifyEnabled && isBase) {
    console.log("\nVerifying on Basescan…");
    const targets = [
      { name: "EscrowPool", address: deployments.EscrowPool, args: [deployer.address, ftnsChecksum, hre.ethers.ZeroAddress] },
      { name: "BatchSettlementRegistry", address: deployments.BatchSettlementRegistry, args: [deployer.address, challengeWindow] },
      { name: "StakeBond", address: deployments.StakeBond, args: [deployer.address, ftnsChecksum, unbondDelay] },
    ];
    if (!preDeployedVerifier) {
      targets.push({ name: "MockSignatureVerifier", address: verifierAddress, args: [] });
    }
    for (const t of targets) {
      try {
        await hre.run("verify:verify", { address: t.address, constructorArguments: t.args });
        console.log(`   ${t.name} verified`);
      } catch (e) {
        console.warn(`   ${t.name} verify failed (non-fatal): ${e.message.split("\n")[0]}`);
      }
    }
  } else if (isBase) {
    console.log("\nSkipping verification (set AUTO_VERIFY=1 to enable).");
  }

  console.log("\n✅ Audit bundle deployment complete.");
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
