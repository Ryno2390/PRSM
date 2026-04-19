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

  // ── Phase 1.1 Task 9: preflight ─────────────────────────────────────
  // Constructor args are immutable on RoyaltyDistributor. A typo here is
  // permanent — re-deploy is the only fix. Catch mistakes BEFORE we burn
  // gas on a bad deploy.
  console.log("\nPreflight…");

  // 1. Checksum validation. ethers v6 throws on bad checksum.
  let ftnsChecksum, treasuryChecksum;
  try {
    ftnsChecksum = hre.ethers.getAddress(ftnsAddress);
    treasuryChecksum = hre.ethers.getAddress(treasury);
  } catch (e) {
    throw new Error(`address checksum failure: ${e.message}`);
  }
  if (ftnsChecksum === hre.ethers.ZeroAddress) throw new Error("FTNS_TOKEN_ADDRESS is zero");
  if (treasuryChecksum === hre.ethers.ZeroAddress) throw new Error("NETWORK_TREASURY is zero");
  console.log(`  FTNS checksum:     ${ftnsChecksum}`);
  console.log(`  Treasury checksum: ${treasuryChecksum}`);

  // 2. Code at FTNS address.
  const code = await hre.ethers.provider.getCode(ftnsChecksum);
  if (code === "0x" || code === "0x0") {
    throw new Error(`no contract at FTNS_TOKEN_ADDRESS ${ftnsChecksum} on network ${network}`);
  }
  console.log(`  FTNS bytecode:     ${(code.length / 2 - 1)} bytes`);

  // 3. Confirm it actually quacks like FTNS. We accept either "FTNS"
  //    (production) or "MFTNS" (the test mock used by the integration
  //    suite) so the script works against both.
  const ftnsAbi = [
    "function symbol() view returns (string)",
    "function name() view returns (string)",
  ];
  const ftns = new hre.ethers.Contract(ftnsChecksum, ftnsAbi, hre.ethers.provider);
  let symbol;
  try {
    symbol = await ftns.symbol();
  } catch (e) {
    throw new Error(`FTNS symbol() call failed: ${e.message}`);
  }
  if (symbol !== "FTNS" && symbol !== "MFTNS") {
    throw new Error(`FTNS_TOKEN_ADDRESS symbol is "${symbol}", expected "FTNS" or "MFTNS"`);
  }
  console.log(`  FTNS symbol:       ${symbol}`);

  // 4. Log chain id (don't enforce — works on hardhat fork too).
  const chainIdActual = (await hre.ethers.provider.getNetwork()).chainId;
  console.log(`  Chain id:          ${chainIdActual}`);

  const [deployer] = await hre.ethers.getSigners();
  console.log(`\nDeployer:         ${deployer.address}`);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log(`Deployer balance: ${hre.ethers.formatEther(balance)} ETH`);
  if (balance === 0n) throw new Error("Deployer has zero balance");

  // 5. Treasury must not equal deployer. networkTreasury is immutable —
  //    the Sepolia bake-in convenience of using deployer-as-treasury is
  //    acceptable on testnet but would permanently route the 2% royalty
  //    fee to the deployer EOA on mainnet with no upgrade path. Belt and
  //    suspenders: operator selects the right value AND the script
  //    refuses if they match.
  if (treasuryChecksum.toLowerCase() === deployer.address.toLowerCase()) {
    throw new Error(
      `NETWORK_TREASURY (${treasuryChecksum}) must not equal deployer (${deployer.address}). ` +
      `Use a dedicated treasury address — a multi-sig or foundation-controlled address, never the deployer EOA.`
    );
  }

  // 1. Registry
  console.log("\nDeploying ProvenanceRegistry…");
  const Registry = await hre.ethers.getContractFactory("ProvenanceRegistry");
  const registry = await Registry.deploy();
  await registry.waitForDeployment();
  const registryAddress = await registry.getAddress();
  console.log(`  ProvenanceRegistry: ${registryAddress}`);

  // 2. Distributor (using checksummed addresses from preflight)
  console.log("\nDeploying RoyaltyDistributor…");
  const Distributor = await hre.ethers.getContractFactory("RoyaltyDistributor");
  const distributor = await Distributor.deploy(ftnsChecksum, registryAddress, treasuryChecksum);
  await distributor.waitForDeployment();
  const distributorAddress = await distributor.getAddress();
  console.log(`  RoyaltyDistributor: ${distributorAddress}`);

  // 3. Save deployment manifest
  const manifest = {
    network,
    chainId: chainIdActual.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      ProvenanceRegistry: registryAddress,
      RoyaltyDistributor: distributorAddress,
      FTNSToken: ftnsChecksum,
      NetworkTreasury: treasuryChecksum,
    },
  };

  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `provenance-${network}-${Date.now()}.json`);
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  // 4. Optional: auto-verify on Basescan (only on real Base networks).
  const verifyEnabled = process.env.AUTO_VERIFY === "1";
  const isBase = network === "base" || network === "base-sepolia";
  if (verifyEnabled && isBase) {
    console.log("\nVerifying on Basescan…");
    const verifyTargets = [
      { name: "ProvenanceRegistry", address: registryAddress, args: [] },
      {
        name: "RoyaltyDistributor",
        address: distributorAddress,
        args: [ftnsChecksum, registryAddress, treasuryChecksum],
      },
    ];
    for (const t of verifyTargets) {
      try {
        await hre.run("verify:verify", {
          address: t.address,
          constructorArguments: t.args,
        });
        console.log(`  ${t.name} verified at ${t.address}`);
      } catch (e) {
        console.warn(`  ${t.name} verification failed (non-fatal): ${e.message}`);
      }
    }
  } else if (isBase) {
    console.log("\nSkipping verification (set AUTO_VERIFY=1 to enable).");
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
