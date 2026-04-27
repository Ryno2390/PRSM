/*
 * Phase 3.x.3 Task 2 — PublisherKeyAnchor deploy script.
 *
 * Deploys PublisherKeyAnchor to a target network (typically sepolia
 * for v1; Base mainnet bundles into the existing Phase 1.3 audit
 * clock per the design plan §1.2).
 *
 * Required env var:
 *   ANCHOR_ADMIN_ADDRESS  - admin (multisig) address that holds the
 *                            adminOverride power. For Sepolia v1, this
 *                            can be a designated dev multisig; for
 *                            Base mainnet, MUST be the production
 *                            Foundation multisig per the audit gate.
 *
 * Optional env var:
 *   AUTO_VERIFY           - "1" to auto-verify on Etherscan / Basescan
 *                            after deploy. Requires ETHERSCAN_API_KEY
 *                            (sepolia) or BASESCAN_API_KEY (base).
 *
 * Usage:
 *   # Dry-run on local Hardhat node — validates the script + admin
 *   # address shape WITHOUT spending gas. Use this BEFORE a real
 *   # Sepolia deploy.
 *   ANCHOR_ADMIN_ADDRESS=0x000... npx hardhat run \
 *       scripts/deploy-publisher-key-anchor.js --network hardhat
 *
 *   # Real Sepolia deploy:
 *   PRIVATE_KEY=0x... \
 *   SEPOLIA_RPC_URL=https://... \
 *   ANCHOR_ADMIN_ADDRESS=0x... \
 *   AUTO_VERIFY=1 ETHERSCAN_API_KEY=... \
 *       npx hardhat run scripts/deploy-publisher-key-anchor.js \
 *           --network sepolia
 *
 * After a successful Sepolia deploy, copy the address printed in the
 * "DEPLOY COMPLETE" block into prsm/deployments/contract_addresses.json
 * under sepolia.publisher_key_anchor and commit. The Python anchor
 * client (Task 3) reads from that file at construction time.
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;
  const adminAddress = process.env.ANCHOR_ADMIN_ADDRESS;

  if (!adminAddress) {
    throw new Error(
      "ANCHOR_ADMIN_ADDRESS env var required (multisig admin for adminOverride). " +
      "For Sepolia v1, use a designated dev multisig; for Base mainnet, MUST be " +
      "the production Foundation multisig per Phase 3.x.3 design plan §1.2."
    );
  }

  const adminChecksum = hre.ethers.getAddress(adminAddress);
  if (adminChecksum === hre.ethers.ZeroAddress) {
    throw new Error("ANCHOR_ADMIN_ADDRESS cannot be the zero address");
  }

  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;

  console.log(`\n=== Deploying PublisherKeyAnchor to ${network} ===`);
  console.log(`Deployer:           ${deployer.address}`);
  console.log(`Deployer balance:   ${hre.ethers.formatEther(balance)} ETH`);
  console.log(`Chain id:           ${chainId}`);
  console.log(`Admin (multisig):   ${adminChecksum}`);

  if (balance === 0n && network !== "hardhat") {
    throw new Error("Deployer has zero balance");
  }

  // ── Deploy ─────────────────────────────────────────────────────────
  console.log(`\n[1/1] Deploying PublisherKeyAnchor…`);
  const Anchor = await hre.ethers.getContractFactory("PublisherKeyAnchor");
  const anchor = await Anchor.deploy(adminChecksum);
  await anchor.waitForDeployment();
  const anchorAddress = await anchor.getAddress();
  console.log(`   PublisherKeyAnchor: ${anchorAddress}`);

  // ── Post-deploy invariant check ────────────────────────────────────
  console.log(`\nPost-deploy invariant checks…`);
  const onChainAdmin = await anchor.admin();
  console.log(`   anchor.admin: ${onChainAdmin}`);
  if (onChainAdmin.toLowerCase() !== adminChecksum.toLowerCase()) {
    throw new Error(
      `admin wiring mismatch: on-chain=${onChainAdmin} expected=${adminChecksum}`
    );
  }

  // Smoke-test lookup of an unregistered nodeId returns empty bytes —
  // confirms the contract is responsive at the deployed address.
  const dummyNodeId = "0x" + "00".repeat(16);
  const lookupResult = await anchor.lookup(dummyNodeId);
  if (lookupResult !== "0x") {
    throw new Error(
      `freshly-deployed contract returned non-empty lookup for ${dummyNodeId}: ${lookupResult}`
    );
  }
  console.log(`   anchor.lookup(zero): 0x (empty as expected)`);

  // ── Manifest ───────────────────────────────────────────────────────
  const manifest = {
    bundle: "phase3.x.3-publisher-key-anchor",
    network,
    chainId: chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      PublisherKeyAnchor: anchorAddress,
    },
    constructorArgs: {
      admin: adminChecksum,
    },
    postDeployNotes: [
      "Admin holds adminOverride() power for emergency key revocation. " +
      "On Base mainnet, this MUST be the Foundation multisig.",
      "Admin is immutable at the contract level — rotation requires a " +
      "new contract deploy + migration of every registered publisher.",
      "Copy this address into prsm/deployments/contract_addresses.json " +
      "under <network>.publisher_key_anchor before merging.",
    ],
  };
  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(
    outDir,
    `publisher-key-anchor-${network}-${Date.now()}.json`
  );
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  // ── Etherscan / Basescan verification ──────────────────────────────
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
        address: anchorAddress,
        constructorArguments: [adminChecksum],
      });
      console.log(`   PublisherKeyAnchor verified`);
    } catch (e) {
      console.warn(
        `   PublisherKeyAnchor verify failed (non-fatal): ${e.message.split("\n")[0]}`
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
  console.log(`Network:            ${network}`);
  console.log(`PublisherKeyAnchor: ${anchorAddress}`);
  console.log(`Admin:              ${adminChecksum}`);
  console.log(``);
  console.log(`Next steps:`);
  console.log(`  1. Copy the address into prsm/deployments/contract_addresses.json:`);
  console.log(`       ${network}.publisher_key_anchor = "${anchorAddress}"`);
  console.log(`  2. Commit the change.`);
  console.log(`  3. Run Task 3 (Python client) test suite to confirm`);
  console.log(`     end-to-end connectivity to the deployed address.`);
  console.log(`${"=".repeat(60)}\n`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
