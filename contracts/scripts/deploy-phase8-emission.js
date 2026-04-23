/*
 * Phase 8 deploy — EmissionController + CompensationDistributor.
 *
 * Order:
 *   1. EmissionController          (immutable halving curve + mintCap)
 *   2. CompensationDistributor     (pull + weighted split)
 *   3. Cross-wire: EmissionController.setAuthorizedDistributor(distributor)
 *
 * The FTNSToken MINTER_ROLE grant to EmissionController is a separate
 * governance tx that must run AFTER this script (mainnet) and AFTER the
 * Foundation multi-sig has ownership of the token. It is not included
 * here to keep this script idempotent for rehearsal runs.
 *
 * Required env vars:
 *   FTNS_TOKEN_ADDRESS           - ERC20 minter target
 *   CREATOR_POOL                 - creator pool address (multi-sig on mainnet)
 *   OPERATOR_POOL                - operator pool address
 *   GRANT_POOL                   - grant pool address
 *
 * Optional env vars:
 *   BASELINE_RATE_PER_SECOND     - default 1_000_000_000_000_000_000 (1 FTNS/s)
 *   MINT_CAP_WEI                 - default 900_000_000 ether (900M FTNS)
 *   EPOCH_ZERO_START_TIMESTAMP   - default current block timestamp
 *   CREATOR_BPS                  - default 5000 (50%)
 *   OPERATOR_BPS                 - default 3000 (30%)
 *   GRANT_BPS                    - default 2000 (20%)
 *   AUTO_VERIFY                  - 1 to auto-verify on Basescan
 *
 * Usage:
 *   npx hardhat run scripts/deploy-phase8-emission.js --network hardhat
 *   npx hardhat run scripts/deploy-phase8-emission.js --network base-sepolia
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;
  const ftnsAddress = process.env.FTNS_TOKEN_ADDRESS;
  const creatorPool = process.env.CREATOR_POOL;
  const operatorPool = process.env.OPERATOR_POOL;
  const grantPool = process.env.GRANT_POOL;

  if (!ftnsAddress) throw new Error("FTNS_TOKEN_ADDRESS env var required");
  if (!creatorPool) throw new Error("CREATOR_POOL env var required");
  if (!operatorPool) throw new Error("OPERATOR_POOL env var required");
  if (!grantPool) throw new Error("GRANT_POOL env var required");

  const baselineRatePerSecond = BigInt(
    process.env.BASELINE_RATE_PER_SECOND || hre.ethers.parseUnits("1", 18).toString()
  );
  const mintCap = BigInt(
    process.env.MINT_CAP_WEI || hre.ethers.parseUnits("900000000", 18).toString()
  );
  const creatorBps = parseInt(process.env.CREATOR_BPS || "5000", 10);
  const operatorBps = parseInt(process.env.OPERATOR_BPS || "3000", 10);
  const grantBps = parseInt(process.env.GRANT_BPS || "2000", 10);

  if (creatorBps + operatorBps + grantBps !== 10000) {
    throw new Error(`pool BPS must sum to 10000 (got ${creatorBps + operatorBps + grantBps})`);
  }

  console.log(`\n=== Deploying Phase 8 emission layer to ${network} ===`);

  // Preflight
  const ftnsChecksum = hre.ethers.getAddress(ftnsAddress);
  const creatorChecksum = hre.ethers.getAddress(creatorPool);
  const operatorChecksum = hre.ethers.getAddress(operatorPool);
  const grantChecksum = hre.ethers.getAddress(grantPool);

  const code = await hre.ethers.provider.getCode(ftnsChecksum);
  if (code === "0x" || code === "0x0") {
    throw new Error(`no contract at FTNS_TOKEN_ADDRESS ${ftnsChecksum}`);
  }

  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;

  const latestBlock = await hre.ethers.provider.getBlock("latest");
  const epochZeroStart = BigInt(
    process.env.EPOCH_ZERO_START_TIMESTAMP || latestBlock.timestamp.toString()
  );

  console.log(`Deployer:          ${deployer.address}`);
  console.log(`Deployer balance:  ${hre.ethers.formatEther(balance)} ETH`);
  console.log(`Chain id:          ${chainId}`);
  console.log(`FTNS token:        ${ftnsChecksum}`);
  console.log(`Baseline rate:     ${hre.ethers.formatUnits(baselineRatePerSecond, 18)} FTNS/s`);
  console.log(`Mint cap:          ${hre.ethers.formatUnits(mintCap, 18)} FTNS`);
  console.log(`Epoch zero start:  ${epochZeroStart}`);
  console.log(`Pools (${creatorBps}/${operatorBps}/${grantBps} bps):`);
  console.log(`  creator:  ${creatorChecksum}`);
  console.log(`  operator: ${operatorChecksum}`);
  console.log(`  grant:    ${grantChecksum}`);

  if (balance === 0n) throw new Error("Deployer has zero balance");

  const deployments = {};
  const txHashes = {};

  // ── 1. EmissionController ──────────────────────────────────────────
  console.log("\n[1/3] Deploying EmissionController…");
  const Emission = await hre.ethers.getContractFactory("EmissionController");
  const emission = await Emission.deploy(
    ftnsChecksum,
    epochZeroStart,
    baselineRatePerSecond,
    mintCap,
    deployer.address
  );
  await emission.waitForDeployment();
  deployments.EmissionController = await emission.getAddress();
  console.log(`   EmissionController:     ${deployments.EmissionController}`);

  // ── 2. CompensationDistributor ─────────────────────────────────────
  console.log("\n[2/3] Deploying CompensationDistributor…");
  const Distributor = await hre.ethers.getContractFactory("CompensationDistributor");
  const distributor = await Distributor.deploy(
    ftnsChecksum,
    deployments.EmissionController,
    creatorChecksum,
    operatorChecksum,
    grantChecksum,
    { creatorPoolBps: creatorBps, operatorPoolBps: operatorBps, grantPoolBps: grantBps },
    deployer.address
  );
  await distributor.waitForDeployment();
  deployments.CompensationDistributor = await distributor.getAddress();
  console.log(`   CompensationDistributor: ${deployments.CompensationDistributor}`);

  // ── 3. Cross-wire ───────────────────────────────────────────────────
  console.log("\n[3/3] Cross-wiring…");
  let tx = await emission.setAuthorizedDistributor(deployments.CompensationDistributor);
  await tx.wait();
  txHashes.emission_setAuthorizedDistributor = tx.hash;
  console.log(`   EmissionController.setAuthorizedDistributor → distributor (${tx.hash.slice(0, 10)}…)`);

  // ── Invariant checks ──────────────────────────────────────────────
  console.log("\nPost-deploy invariant checks…");
  const immutables = {
    "emission.baselineRatePerSecond": (await emission.baselineRatePerSecond()).toString(),
    "emission.mintCap":              (await emission.mintCap()).toString(),
    "emission.ftnsToken":            await emission.ftnsToken(),
    "emission.epochZeroStart":       (await emission.epochZeroStartTimestamp()).toString(),
    "emission.authorizedDistributor": await emission.authorizedDistributor(),
    "distributor.ftnsToken":         await distributor.ftnsToken(),
    "distributor.emissionController":await distributor.emissionController(),
    "distributor.creatorPool":       await distributor.creatorPool(),
    "distributor.operatorPool":      await distributor.operatorPool(),
    "distributor.grantPool":         await distributor.grantPool(),
  };
  for (const [k, v] of Object.entries(immutables)) {
    console.log(`   ${k}: ${v}`);
  }
  if (immutables["emission.authorizedDistributor"].toLowerCase() !==
      deployments.CompensationDistributor.toLowerCase()) {
    throw new Error("authorizedDistributor wiring mismatch");
  }

  // ── Manifest ────────────────────────────────────────────────────────
  const manifest = {
    bundle: "phase8-emission",
    network,
    chainId: chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    params: {
      baselineRatePerSecond: baselineRatePerSecond.toString(),
      mintCap: mintCap.toString(),
      epochZeroStartTimestamp: epochZeroStart.toString(),
      poolWeightsBps: { creator: creatorBps, operator: operatorBps, grant: grantBps },
    },
    contracts: {
      ...deployments,
      FTNSToken: ftnsChecksum,
      CreatorPool: creatorChecksum,
      OperatorPool: operatorChecksum,
      GrantPool: grantChecksum,
    },
    crossWireTxHashes: txHashes,
    postDeployNotes: [
      "Grant FTNSToken MINTER_ROLE to EmissionController via Foundation multi-sig " +
      "before any mint calls (not executed by this script).",
      "If pool addresses need to change, use CompensationDistributor.setPoolAddresses " +
      "(owner-only; irreversible once rolled).",
    ],
  };
  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `phase8-emission-${network}-${Date.now()}.json`);
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  // ── Optional verification ──────────────────────────────────────────
  const verifyEnabled = process.env.AUTO_VERIFY === "1";
  const isBase = network === "base" || network === "base-sepolia";
  if (verifyEnabled && isBase) {
    console.log("\nVerifying on Basescan…");
    const targets = [
      {
        name: "EmissionController",
        address: deployments.EmissionController,
        args: [ftnsChecksum, epochZeroStart, baselineRatePerSecond, mintCap, deployer.address],
      },
      {
        name: "CompensationDistributor",
        address: deployments.CompensationDistributor,
        args: [
          ftnsChecksum,
          deployments.EmissionController,
          creatorChecksum,
          operatorChecksum,
          grantChecksum,
          { creatorPoolBps: creatorBps, operatorPoolBps: operatorBps, grantPoolBps: grantBps },
          deployer.address,
        ],
      },
    ];
    for (const t of targets) {
      try {
        await hre.run("verify:verify", { address: t.address, constructorArguments: t.args });
        console.log(`   ${t.name} verified`);
      } catch (e) {
        console.warn(`   ${t.name} verify failed (non-fatal): ${e.message.split("\n")[0]}`);
      }
    }
  }

  console.log("\n✅ Phase 8 emission layer deployment complete.");
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
