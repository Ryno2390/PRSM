/*
 * Phase 7-storage deploy — StorageSlashing + KeyDistribution.
 *
 * Order:
 *   1. StorageSlashing              (challenge-window slashing for shard holders)
 *   2. KeyDistribution              (encrypted key deposit + royalty-gated release)
 *
 * Both contracts take constructor owner; cross-wiring happens off-contract
 * (StorageSlashing uses the StakeBond deployed by the audit-bundle script).
 *
 * Required env vars:
 *   STAKE_BOND_ADDRESS             - deployed StakeBond (from audit-bundle)
 *   AUTHORIZED_VERIFIER            - proof verifier address (Phase 7-storage
 *                                     Python-side verifier EOA on testnet;
 *                                     dedicated verifier contract on mainnet)
 *
 * Optional env vars:
 *   HEARTBEAT_GRACE_SECONDS        - default 86400 (1 day)
 *   AUTO_VERIFY                    - 1 to auto-verify on Basescan
 *
 * Usage:
 *   npx hardhat run scripts/deploy-phase7-storage.js --network hardhat
 *   npx hardhat run scripts/deploy-phase7-storage.js --network base-sepolia
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;
  const stakeBondAddress = process.env.STAKE_BOND_ADDRESS;
  const verifierAddress = process.env.AUTHORIZED_VERIFIER;

  if (!stakeBondAddress) throw new Error("STAKE_BOND_ADDRESS env var required");
  if (!verifierAddress) throw new Error("AUTHORIZED_VERIFIER env var required");

  const heartbeatGrace = BigInt(process.env.HEARTBEAT_GRACE_SECONDS || "86400");

  console.log(`\n=== Deploying Phase 7-storage contracts to ${network} ===`);

  const stakeBondChecksum = hre.ethers.getAddress(stakeBondAddress);
  const verifierChecksum = hre.ethers.getAddress(verifierAddress);

  const stakeBondCode = await hre.ethers.provider.getCode(stakeBondChecksum);
  if (stakeBondCode === "0x" || stakeBondCode === "0x0") {
    throw new Error(`no contract at STAKE_BOND_ADDRESS ${stakeBondChecksum}`);
  }

  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;

  console.log(`Deployer:           ${deployer.address}`);
  console.log(`Deployer balance:   ${hre.ethers.formatEther(balance)} ETH`);
  console.log(`Chain id:           ${chainId}`);
  console.log(`StakeBond:          ${stakeBondChecksum}`);
  console.log(`Authorized verifier:${verifierChecksum}`);
  console.log(`Heartbeat grace:    ${heartbeatGrace}s`);

  if (balance === 0n) throw new Error("Deployer has zero balance");

  const deployments = {};

  // ── 1. StorageSlashing ─────────────────────────────────────────────
  console.log("\n[1/2] Deploying StorageSlashing…");
  const Slashing = await hre.ethers.getContractFactory("StorageSlashing");
  const slashing = await Slashing.deploy(
    stakeBondChecksum,
    verifierChecksum,
    heartbeatGrace,
    deployer.address
  );
  await slashing.waitForDeployment();
  deployments.StorageSlashing = await slashing.getAddress();
  console.log(`   StorageSlashing: ${deployments.StorageSlashing}`);

  // ── 2. KeyDistribution ─────────────────────────────────────────────
  console.log("\n[2/2] Deploying KeyDistribution…");
  const KeyDist = await hre.ethers.getContractFactory("KeyDistribution");
  const keyDist = await KeyDist.deploy(deployer.address);
  await keyDist.waitForDeployment();
  deployments.KeyDistribution = await keyDist.getAddress();
  console.log(`   KeyDistribution: ${deployments.KeyDistribution}`);

  // Invariant check
  const slashingBond = await slashing.stakeBond();
  const slashingVerifier = await slashing.authorizedVerifier();
  const slashingGrace = await slashing.heartbeatGraceSeconds();
  console.log("\nPost-deploy invariant checks…");
  console.log(`   slashing.stakeBond:            ${slashingBond}`);
  console.log(`   slashing.authorizedVerifier:   ${slashingVerifier}`);
  console.log(`   slashing.heartbeatGraceSeconds:${slashingGrace}`);
  if (slashingBond.toLowerCase() !== stakeBondChecksum.toLowerCase()) {
    throw new Error("stakeBond wiring mismatch");
  }
  if (slashingVerifier.toLowerCase() !== verifierChecksum.toLowerCase()) {
    throw new Error("authorizedVerifier wiring mismatch");
  }

  // Manifest
  const manifest = {
    bundle: "phase7-storage",
    network,
    chainId: chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    params: {
      heartbeatGraceSeconds: heartbeatGrace.toString(),
    },
    contracts: {
      ...deployments,
      StakeBond: stakeBondChecksum,
      AuthorizedVerifier: verifierChecksum,
    },
    postDeployNotes: [
      "KeyDistribution.release() is gated on IRoyaltyPaymentVerifier; this must " +
      "be set via KeyDistribution.setRoyaltyVerifier in a follow-up tx.",
      "StorageSlashing.authorizedVerifier can be rotated via setAuthorizedVerifier " +
      "(owner-only) if the off-chain prover migrates.",
    ],
  };
  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `phase7-storage-${network}-${Date.now()}.json`);
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  const verifyEnabled = process.env.AUTO_VERIFY === "1";
  const isBase = network === "base" || network === "base-sepolia";
  if (verifyEnabled && isBase) {
    console.log("\nVerifying on Basescan…");
    const targets = [
      {
        name: "StorageSlashing",
        address: deployments.StorageSlashing,
        args: [stakeBondChecksum, verifierChecksum, heartbeatGrace, deployer.address],
      },
      {
        name: "KeyDistribution",
        address: deployments.KeyDistribution,
        args: [deployer.address],
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

  console.log("\n✅ Phase 7-storage deployment complete.");
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
