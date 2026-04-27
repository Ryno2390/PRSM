/*
 * Phase 8 Task 5 — EmissionController + CompensationDistributor deploy script.
 *
 * Sequenced deploy:
 *   1. EmissionController(ftnsToken, epochZeroStart, baselineRate, mintCap, owner)
 *   2. CompensationDistributor(ftnsToken, emissionController, creator, operator,
 *                              grant, initialWeights, owner)
 *
 * Phase 8 design plan §6 Task 5 calls for "deploy Phase 8 contracts to
 * Sepolia" as a precondition for the multi-stakeholder testnet exercise
 * (Foundation multi-sig + 3-5 operator reps + 2-3 investors + auditors).
 * The exercise itself is gated on Foundation participation; this script
 * is the engineering backbone that lets that exercise begin the moment
 * the participants are available.
 *
 * Mainnet deploy ceremony (Base) bundles into the existing Phase 1.3
 * audit clock per the design plan §11. Use --network sepolia for the
 * pre-mainnet exercise; --network base only after Phase 8 audit clears.
 *
 * Required env vars:
 *   FTNS_TOKEN_ADDRESS         — already-deployed FTNS token contract.
 *                                EmissionController gets MINTER_ROLE on
 *                                this token AFTER deploy via a separate
 *                                grant tx (NOT done here — operator
 *                                runbook step 3).
 *   EPOCH_ZERO_START_TIMESTAMP — Unix seconds. Sets epoch boundaries.
 *                                For Sepolia rehearsal, use a recent
 *                                past timestamp so epoch 0 is "active"
 *                                immediately. For mainnet, use the
 *                                ratified launch timestamp from
 *                                PRSM-TOK-1.
 *   BASELINE_RATE_PER_SECOND   — FTNS wei/sec for epoch 0. Halves each
 *                                EPOCH_DURATION_SECONDS (4 years on
 *                                mainnet; configurable on testnets via
 *                                a separate compile-time flag).
 *   MINT_CAP                   — Total FTNS wei the controller may
 *                                ever mint. Hard cap.
 *   EMISSION_OWNER             — Address holding pause + rate-update
 *                                power. Multisig on mainnet.
 *   CREATOR_POOL_ADDRESS       — Pool destination 1 (publishers / model
 *                                authors).
 *   OPERATOR_POOL_ADDRESS      — Pool destination 2 (node operators).
 *   GRANT_POOL_ADDRESS         — Pool destination 3 (Foundation grants).
 *   CREATOR_POOL_BPS           — Initial weight; uint16 in basis points.
 *   OPERATOR_POOL_BPS          — Initial weight; uint16 in basis points.
 *   GRANT_POOL_BPS             — Initial weight; uint16 in basis points.
 *                                Three weights MUST sum to 10_000.
 *   DISTRIBUTOR_OWNER          — Holds weight-update power on the
 *                                distributor. Multisig on mainnet.
 *
 * Optional env vars:
 *   AUTO_VERIFY  — "1" to auto-verify on Etherscan / Basescan after
 *                  deploy. Requires ETHERSCAN_API_KEY (sepolia) or
 *                  BASESCAN_API_KEY (base).
 *
 * Usage:
 *   # Dry-run on local Hardhat node — validates the script + arg shapes
 *   # WITHOUT spending gas. Run BEFORE a real Sepolia deploy. Most env
 *   # vars can be dummy addresses for the dry-run; only weights must be
 *   # consistent (sum to 10_000).
 *   FTNS_TOKEN_ADDRESS=0x000... \
 *   EPOCH_ZERO_START_TIMESTAMP=1714200000 \
 *   BASELINE_RATE_PER_SECOND=1000000000000000000 \
 *   MINT_CAP=1000000000000000000000000 \
 *   EMISSION_OWNER=0x000... \
 *   CREATOR_POOL_ADDRESS=0x000... \
 *   OPERATOR_POOL_ADDRESS=0x000... \
 *   GRANT_POOL_ADDRESS=0x000... \
 *   CREATOR_POOL_BPS=4000 \
 *   OPERATOR_POOL_BPS=4000 \
 *   GRANT_POOL_BPS=2000 \
 *   DISTRIBUTOR_OWNER=0x000... \
 *       npx hardhat run scripts/deploy-phase8-emission-stack.js --network hardhat
 *
 *   # Real Sepolia deploy:
 *   PRIVATE_KEY=0x... \
 *   SEPOLIA_RPC_URL=https://... \
 *   FTNS_TOKEN_ADDRESS=0x... \
 *   EPOCH_ZERO_START_TIMESTAMP=... \
 *   BASELINE_RATE_PER_SECOND=... \
 *   MINT_CAP=... \
 *   EMISSION_OWNER=0x... \
 *   CREATOR_POOL_ADDRESS=0x... \
 *   OPERATOR_POOL_ADDRESS=0x... \
 *   GRANT_POOL_ADDRESS=0x... \
 *   CREATOR_POOL_BPS=... \
 *   OPERATOR_POOL_BPS=... \
 *   GRANT_POOL_BPS=... \
 *   DISTRIBUTOR_OWNER=0x... \
 *   AUTO_VERIFY=1 ETHERSCAN_API_KEY=... \
 *       npx hardhat run scripts/deploy-phase8-emission-stack.js --network sepolia
 *
 * After a successful Sepolia deploy:
 *   1. Copy the addresses printed in the "DEPLOY COMPLETE" block into
 *      prsm/deployments/contract_addresses.json under
 *      sepolia.emission_controller and sepolia.compensation_distributor.
 *   2. Grant MINTER_ROLE on the FTNS token to the deployed
 *      EmissionController (separate operator tx — NOT done here).
 *   3. Run Task 4 (Python EmissionClient) test suite against the
 *      deployed addresses to confirm end-to-end connectivity.
 *   4. Then proceed with Phase 8 §6 Task 5 multi-stakeholder exercise.
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

const BPS_DENOMINATOR = 10_000;

function requireEnv(name) {
  const value = process.env[name];
  if (value === undefined || value === "") {
    throw new Error(
      `${name} env var required — see header comment for full list`
    );
  }
  return value;
}

function requireAddress(name) {
  const raw = requireEnv(name);
  const checksum = hre.ethers.getAddress(raw);
  if (checksum === hre.ethers.ZeroAddress) {
    throw new Error(`${name} cannot be the zero address`);
  }
  return checksum;
}

function requireBigInt(name) {
  const raw = requireEnv(name);
  let parsed;
  try {
    parsed = BigInt(raw);
  } catch (e) {
    throw new Error(`${name} must be an integer, got '${raw}'`);
  }
  if (parsed <= 0n) {
    throw new Error(`${name} must be > 0, got ${parsed}`);
  }
  return parsed;
}

function requireUint16(name) {
  const raw = requireEnv(name);
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 0 || parsed > 0xffff) {
    throw new Error(`${name} must be a uint16 (0..65535), got ${raw}`);
  }
  return parsed;
}

async function main() {
  const network = hre.network.name;
  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;

  // ── Validate inputs (cheap; do BEFORE any deploy tx) ───────────────
  const ftnsToken = requireAddress("FTNS_TOKEN_ADDRESS");
  const epochZeroStart = requireBigInt("EPOCH_ZERO_START_TIMESTAMP");
  const baselineRate = requireBigInt("BASELINE_RATE_PER_SECOND");
  const mintCap = requireBigInt("MINT_CAP");
  const emissionOwner = requireAddress("EMISSION_OWNER");
  const creatorPool = requireAddress("CREATOR_POOL_ADDRESS");
  const operatorPool = requireAddress("OPERATOR_POOL_ADDRESS");
  const grantPool = requireAddress("GRANT_POOL_ADDRESS");
  const creatorBps = requireUint16("CREATOR_POOL_BPS");
  const operatorBps = requireUint16("OPERATOR_POOL_BPS");
  const grantBps = requireUint16("GRANT_POOL_BPS");
  const distributorOwner = requireAddress("DISTRIBUTOR_OWNER");

  // Weight invariant — three pools sum to 10_000 bps. Match the on-chain
  // _validateWeights check exactly, so a misconfigured deploy fails fast
  // here rather than at the constructor revert.
  const weightSum = creatorBps + operatorBps + grantBps;
  if (weightSum !== BPS_DENOMINATOR) {
    throw new Error(
      `Pool weights must sum to ${BPS_DENOMINATOR} bps; ` +
        `got creator=${creatorBps} + operator=${operatorBps} + grant=${grantBps} ` +
        `= ${weightSum}`
    );
  }

  // Three pool addresses MUST be distinct — sharing an address would
  // collapse the weight split. CompensationDistributor doesn't enforce
  // distinctness on-chain (it just sends to whatever addresses you pass),
  // so the safety check belongs here.
  const poolSet = new Set([
    creatorPool.toLowerCase(),
    operatorPool.toLowerCase(),
    grantPool.toLowerCase(),
  ]);
  if (poolSet.size !== 3) {
    throw new Error(
      `Pool addresses must be distinct: creator=${creatorPool}, ` +
        `operator=${operatorPool}, grant=${grantPool}`
    );
  }

  console.log(`\n=== Deploying Phase 8 emission stack to ${network} ===`);
  console.log(`Deployer:                ${deployer.address}`);
  console.log(`Deployer balance:        ${hre.ethers.formatEther(balance)} ETH`);
  console.log(`Chain id:                ${chainId}`);
  console.log(`FTNS token:              ${ftnsToken}`);
  console.log(`Epoch zero start (unix): ${epochZeroStart}`);
  console.log(`Baseline rate (wei/s):   ${baselineRate}`);
  console.log(`Mint cap (wei):          ${mintCap}`);
  console.log(`Emission owner:          ${emissionOwner}`);
  console.log(`Creator pool:            ${creatorPool}  (${creatorBps} bps)`);
  console.log(`Operator pool:           ${operatorPool}  (${operatorBps} bps)`);
  console.log(`Grant pool:              ${grantPool}  (${grantBps} bps)`);
  console.log(`Distributor owner:       ${distributorOwner}`);

  if (balance === 0n && network !== "hardhat") {
    throw new Error("Deployer has zero balance");
  }

  // ── Deploy EmissionController ──────────────────────────────────────
  console.log(`\n[1/2] Deploying EmissionController…`);
  const EmissionController = await hre.ethers.getContractFactory(
    "EmissionController"
  );
  const emission = await EmissionController.deploy(
    ftnsToken,
    epochZeroStart,
    baselineRate,
    mintCap,
    emissionOwner
  );
  await emission.waitForDeployment();
  const emissionAddress = await emission.getAddress();
  console.log(`   EmissionController: ${emissionAddress}`);

  // Post-deploy invariant checks
  const onChainOwner = await emission.owner();
  if (onChainOwner.toLowerCase() !== emissionOwner.toLowerCase()) {
    throw new Error(
      `EmissionController owner mismatch: on-chain=${onChainOwner} expected=${emissionOwner}`
    );
  }
  const onChainRate = await emission.baselineRatePerSecond();
  if (onChainRate !== baselineRate) {
    throw new Error(
      `baselineRatePerSecond mismatch: on-chain=${onChainRate} expected=${baselineRate}`
    );
  }
  const onChainMintCap = await emission.mintCap();
  if (onChainMintCap !== mintCap) {
    throw new Error(
      `mintCap mismatch: on-chain=${onChainMintCap} expected=${mintCap}`
    );
  }
  console.log(`   Post-deploy invariants OK (owner, rate, cap match)`);

  // ── Deploy CompensationDistributor ─────────────────────────────────
  console.log(`\n[2/2] Deploying CompensationDistributor…`);
  const CompensationDistributor = await hre.ethers.getContractFactory(
    "CompensationDistributor"
  );
  const initialWeights = {
    creatorPoolBps: creatorBps,
    operatorPoolBps: operatorBps,
    grantPoolBps: grantBps,
  };
  const distributor = await CompensationDistributor.deploy(
    ftnsToken,
    emissionAddress,
    creatorPool,
    operatorPool,
    grantPool,
    initialWeights,
    distributorOwner
  );
  await distributor.waitForDeployment();
  const distributorAddress = await distributor.getAddress();
  console.log(`   CompensationDistributor: ${distributorAddress}`);

  const onChainDistOwner = await distributor.owner();
  if (onChainDistOwner.toLowerCase() !== distributorOwner.toLowerCase()) {
    throw new Error(
      `CompensationDistributor owner mismatch: on-chain=${onChainDistOwner} expected=${distributorOwner}`
    );
  }
  console.log(`   Post-deploy invariants OK (owner matches)`);

  // ── Manifest ───────────────────────────────────────────────────────
  const manifest = {
    bundle: "phase8-emission-stack",
    network,
    chainId: chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      EmissionController: emissionAddress,
      CompensationDistributor: distributorAddress,
    },
    constructorArgs: {
      EmissionController: {
        ftnsToken,
        epochZeroStartTimestamp: epochZeroStart.toString(),
        baselineRatePerSecond: baselineRate.toString(),
        mintCap: mintCap.toString(),
        initialOwner: emissionOwner,
      },
      CompensationDistributor: {
        ftnsToken,
        emissionController: emissionAddress,
        creatorPool,
        operatorPool,
        grantPool,
        initialWeights: {
          creatorPoolBps: creatorBps,
          operatorPoolBps: operatorBps,
          grantPoolBps: grantBps,
        },
        initialOwner: distributorOwner,
      },
    },
    postDeployNotes: [
      "EmissionController needs MINTER_ROLE on the FTNS token. " +
        "Grant separately via the FTNS token's grantRole — operator " +
        "runbook step 3. Until granted, distribute() reverts at mint().",
      "CompensationDistributor.pullAndDistribute() is permissionless. " +
        "Anyone can poke it; the gas cost is borne by the caller. " +
        "For pre-mainnet exercise, the runbook calls it from a " +
        "scheduled job to keep distribution cadence regular.",
      "Pool weight changes require MIN_WEIGHT_SCHEDULE_DELAY (90 days) " +
        "between scheduling and activation per PRSM-GOV-1 §4.1. The " +
        "distributor owner schedules weights via updateWeights(); " +
        "anyone can activate after the delay via " +
        "_applyScheduledWeightsIfActive (called from pullAndDistribute).",
      "Copy the addresses into prsm/deployments/contract_addresses.json " +
        "under <network>.emission_controller and <network>.compensation_distributor " +
        "before merging.",
    ],
  };
  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(
    outDir,
    `phase8-emission-stack-${network}-${Date.now()}.json`
  );
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  // ── Verification ───────────────────────────────────────────────────
  const verifyEnabled = process.env.AUTO_VERIFY === "1";
  const verifySupported =
    network === "sepolia" ||
    network === "base" ||
    network === "base-sepolia" ||
    network === "mainnet";
  if (verifyEnabled && verifySupported) {
    console.log(`\nVerifying on block explorer…`);
    try {
      await hre.run("verify:verify", {
        address: emissionAddress,
        constructorArguments: [
          ftnsToken,
          epochZeroStart,
          baselineRate,
          mintCap,
          emissionOwner,
        ],
      });
      console.log(`   EmissionController verified`);
    } catch (e) {
      console.warn(
        `   EmissionController verify failed (non-fatal): ${e.message.split("\n")[0]}`
      );
    }
    try {
      await hre.run("verify:verify", {
        address: distributorAddress,
        constructorArguments: [
          ftnsToken,
          emissionAddress,
          creatorPool,
          operatorPool,
          grantPool,
          initialWeights,
          distributorOwner,
        ],
      });
      console.log(`   CompensationDistributor verified`);
    } catch (e) {
      console.warn(
        `   CompensationDistributor verify failed (non-fatal): ${e.message.split("\n")[0]}`
      );
    }
  } else if (verifyEnabled) {
    console.log(
      `\nVerification skipped: network=${network} not in supported list`
    );
  }

  console.log(`\n${"=".repeat(60)}`);
  console.log(`✅ DEPLOY COMPLETE`);
  console.log(`${"=".repeat(60)}`);
  console.log(`Network:                  ${network}`);
  console.log(`EmissionController:       ${emissionAddress}`);
  console.log(`CompensationDistributor:  ${distributorAddress}`);
  console.log(``);
  console.log(`Next steps:`);
  console.log(`  1. Copy the addresses into prsm/deployments/contract_addresses.json:`);
  console.log(`       ${network}.emission_controller       = "${emissionAddress}"`);
  console.log(`       ${network}.compensation_distributor  = "${distributorAddress}"`);
  console.log(`  2. Commit the change.`);
  console.log(`  3. Grant MINTER_ROLE on FTNS token to EmissionController:`);
  console.log(`       ftnsToken.grantRole(MINTER_ROLE, ${emissionAddress})`);
  console.log(`  4. Run Task 4 (Python EmissionClient) test suite against the`);
  console.log(`     deployed addresses to confirm end-to-end connectivity.`);
  console.log(`  5. Schedule the multi-stakeholder testnet exercise per`);
  console.log(`     Phase 8 design plan §6 Task 5.`);
  console.log(`${"=".repeat(60)}\n`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
