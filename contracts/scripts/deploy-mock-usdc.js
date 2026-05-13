/*
 * Deploy MockUSDC for the Aerodrome pool-seed Sepolia rehearsal.
 *
 * Companion to:
 *   contracts/contracts/test/MockUSDC.sol
 *   docs/governance/2026-06-15-aerodrome-pool-seed-sepolia-rehearsal.md
 *
 * Deploys the 6-decimal mock USDC contract and mints the council-
 * ratified seed amount (default 500K mUSDC) to the Sepolia Safe.
 *
 * Required env vars:
 *   PRIVATE_KEY          - deployer hot key (existing Sepolia deployer)
 *   SEPOLIA_SAFE         - Sepolia Safe address (LP recipient during rehearsal)
 *
 * Optional env vars:
 *   MINT_AMOUNT          - mUSDC amount to mint to Safe in whole dollars
 *                          (script multiplies by 10^6). Default 500000.
 *
 * Usage:
 *   PRIVATE_KEY=0x... \
 *   SEPOLIA_SAFE=0xCb4Bfa18E5B166C2E13c18007b4F4E1b2CE8A889 \
 *   npx hardhat run contracts/scripts/deploy-mock-usdc.js --network base-sepolia
 *
 * Output:
 *   - Contract deployed
 *   - 500K mUSDC minted to SEPOLIA_SAFE
 *   - Deployment manifest written to contracts/deployments/mock-usdc-base-sepolia-<ts>.json
 *
 * Exit codes:
 *   0 = deploy + mint succeeded; manifest written
 *   1 = env var missing
 *   2 = on-chain action failed
 */

const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const safeAddress = process.env.SEPOLIA_SAFE;
  if (!safeAddress) {
    console.error("ERROR: SEPOLIA_SAFE env var required");
    process.exit(1);
  }
  if (!ethers.isAddress(safeAddress)) {
    console.error(`ERROR: SEPOLIA_SAFE is not a valid address: ${safeAddress}`);
    process.exit(1);
  }

  const mintWhole = BigInt(process.env.MINT_AMOUNT || "500000");
  const mintWei = mintWhole * 10n ** 6n;  // USDC convention: 6 decimals

  const [deployer] = await ethers.getSigners();
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log(`Deployer: ${deployer.address}`);
  console.log(`Balance:  ${ethers.formatEther(balance)} ETH`);

  if (balance < ethers.parseEther("0.005")) {
    console.error(
      `ERROR: deployer balance too low for deploy + mint. Need ≥ 0.005 ETH.`,
    );
    process.exit(2);
  }

  console.log(`\nDeploying MockUSDC...`);
  const MockUSDC = await ethers.getContractFactory("MockUSDC");
  const mockUsdc = await MockUSDC.deploy();
  await mockUsdc.waitForDeployment();
  const addr = await mockUsdc.getAddress();
  console.log(`  Deployed at: ${addr}`);

  // Confirm decimals + symbol post-deploy as a sanity check
  const decimals = await mockUsdc.decimals();
  const symbol = await mockUsdc.symbol();
  if (decimals !== 6n) {
    console.error(`ERROR: decimals=${decimals}, expected 6`);
    process.exit(2);
  }
  console.log(`  Decimals: ${decimals} (USDC convention)`);
  console.log(`  Symbol:   ${symbol}`);

  console.log(`\nMinting ${mintWhole} mUSDC to Safe ${safeAddress}...`);
  const tx = await mockUsdc.mint(safeAddress, mintWei);
  const rcpt = await tx.wait();
  if (rcpt.status !== 1) {
    console.error(`ERROR: mint tx reverted`);
    process.exit(2);
  }
  console.log(`  Mint tx: ${rcpt.hash}`);

  const safeBal = await mockUsdc.balanceOf(safeAddress);
  console.log(`  Safe balance: ${safeBal} wei (= ${safeBal / 10n ** 6n} mUSDC)`);

  // Write deployment manifest
  const manifest = {
    bundle: "mock-usdc",
    network: "base-sepolia",
    chainId: "84532",
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      MockUSDC: addr,
    },
    decimals: 6,
    symbol: symbol,
    initialMint: {
      to: safeAddress,
      amountWhole: mintWhole.toString(),
      amountWei: mintWei.toString(),
      txHash: rcpt.hash,
    },
  };
  const outDir = path.join(__dirname, "..", "deployments");
  fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(
    outDir,
    `mock-usdc-base-sepolia-${Date.now()}.json`,
  );
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest: ${outFile}`);
  console.log(`\nNext: update Sepolia rehearsal runbook §2.3 with this address.`);
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error("deploy-mock-usdc.js threw:", err);
    process.exit(2);
  });
