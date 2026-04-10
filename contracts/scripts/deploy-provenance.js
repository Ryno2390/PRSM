/*
 * Deploys ProvenanceRegistry + RoyaltyDistributor.
 *
 * Required env vars:
 *   FTNS_TOKEN_ADDRESS    - existing FTNS ERC20 on the target network
 *   NETWORK_TREASURY      - address that receives the 2% network fee
 *
 * Usage:
 *   npx hardhat run scripts/deploy-provenance.js --network base-sepolia
 *   npx hardhat run scripts/deploy-provenance.js --network base
 *
 * Local smoke test (uses dummy addresses, in-process Hardhat network):
 *   FTNS_TOKEN_ADDRESS=0x0000000000000000000000000000000000000001 \
 *   NETWORK_TREASURY=0x0000000000000000000000000000000000000002 \
 *   npx hardhat run scripts/deploy-provenance.js --network hardhat
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const ftnsAddress = process.env.FTNS_TOKEN_ADDRESS;
  const treasury = process.env.NETWORK_TREASURY;
  if (!ftnsAddress) throw new Error("FTNS_TOKEN_ADDRESS env var required");
  if (!treasury) throw new Error("NETWORK_TREASURY env var required");

  const network = hre.network.name;
  console.log(`\n=== Deploying provenance contracts to ${network} ===`);
  console.log(`FTNS token:       ${ftnsAddress}`);
  console.log(`Network treasury: ${treasury}`);

  const [deployer] = await hre.ethers.getSigners();
  console.log(`Deployer:         ${deployer.address}`);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log(`Deployer balance: ${hre.ethers.formatEther(balance)} ETH`);
  if (balance === 0n) throw new Error("Deployer has zero balance");

  // 1. Registry
  console.log("\nDeploying ProvenanceRegistry…");
  const Registry = await hre.ethers.getContractFactory("ProvenanceRegistry");
  const registry = await Registry.deploy();
  await registry.waitForDeployment();
  const registryAddress = await registry.getAddress();
  console.log(`  ProvenanceRegistry: ${registryAddress}`);

  // 2. Distributor
  console.log("\nDeploying RoyaltyDistributor…");
  const Distributor = await hre.ethers.getContractFactory("RoyaltyDistributor");
  const distributor = await Distributor.deploy(ftnsAddress, registryAddress, treasury);
  await distributor.waitForDeployment();
  const distributorAddress = await distributor.getAddress();
  console.log(`  RoyaltyDistributor: ${distributorAddress}`);

  // 3. Save deployment manifest
  const manifest = {
    network,
    chainId: (await hre.ethers.provider.getNetwork()).chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      ProvenanceRegistry: registryAddress,
      RoyaltyDistributor: distributorAddress,
      FTNSToken: ftnsAddress,
      NetworkTreasury: treasury,
    },
  };

  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `provenance-${network}-${Date.now()}.json`);
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
