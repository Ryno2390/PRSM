/*
 * Deploys MockERC20 (symbol "MFTNS") as a test FTNS token on Base
 * Sepolia for the Phase 1.3 bake-in. This is NOT the production
 * FTNS token — production uses the existing token on Base mainnet
 * at 0x5276a3756C85f2E9e46f6D34386167a209aa16e5.
 *
 * Use only on testnets. MockERC20 has a public mint() so anyone
 * can mint unlimited supply — deploying this to a real network
 * would be a disaster.
 *
 * Required env vars:
 *   PRIVATE_KEY           - deployer private key (funded on target network)
 *   BASE_SEPOLIA_RPC_URL  - optional; defaults to https://sepolia.base.org
 *   INITIAL_MINT_AMOUNT   - optional; amount to mint to deployer on deploy.
 *                           Defaults to 10000 ether (10000 * 10^18 wei).
 *
 * Usage:
 *   BASE_SEPOLIA_RPC_URL="<rpc>" PRIVATE_KEY="0x..." \
 *   npx hardhat run scripts/deploy-mock-ftns.js --network base-sepolia
 *
 * After deployment, pass the printed MockERC20 address to
 * deploy-provenance.js as FTNS_TOKEN_ADDRESS.
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;
  if (network === "base" || network === "mainnet") {
    throw new Error(
      `refusing to deploy MockERC20 to ${network} — this token has ` +
      `a public mint() and is test-only. Use the existing FTNS ` +
      `contract on Base mainnet instead.`
    );
  }

  console.log(`\n=== Deploying MockERC20 (test FTNS) to ${network} ===`);

  const [deployer] = await hre.ethers.getSigners();
  console.log(`Deployer:         ${deployer.address}`);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log(`Deployer balance: ${hre.ethers.formatEther(balance)} ETH`);
  if (balance === 0n) throw new Error("Deployer has zero balance");

  const chainId = (await hre.ethers.provider.getNetwork()).chainId;
  console.log(`Chain id:         ${chainId}`);

  console.log("\nDeploying MockERC20…");
  const Mock = await hre.ethers.getContractFactory("MockERC20");
  const mock = await Mock.deploy();
  await mock.waitForDeployment();
  const mockAddress = await mock.getAddress();
  console.log(`  MockERC20: ${mockAddress}`);

  const symbol = await mock.symbol();
  const name = await mock.name();
  console.log(`  symbol: ${symbol}`);
  console.log(`  name:   ${name}`);
  if (symbol !== "MFTNS") {
    throw new Error(`unexpected symbol ${symbol}, expected MFTNS`);
  }

  // Mint initial supply to deployer so bake-in workload has balance.
  const mintAmountEnv = process.env.INITIAL_MINT_AMOUNT;
  const mintAmount = mintAmountEnv
    ? hre.ethers.parseUnits(mintAmountEnv, 18)
    : hre.ethers.parseUnits("10000", 18);
  console.log(`\nMinting ${hre.ethers.formatUnits(mintAmount, 18)} MFTNS to deployer…`);
  const mintTx = await mock.mint(deployer.address, mintAmount);
  await mintTx.wait();
  const bal = await mock.balanceOf(deployer.address);
  console.log(`  deployer MFTNS balance: ${hre.ethers.formatUnits(bal, 18)}`);

  // Save deployment manifest alongside deploy-provenance output.
  const manifest = {
    network,
    chainId: chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      MockERC20: mockAddress,
    },
    initialMint: {
      to: deployer.address,
      amount: mintAmount.toString(),
    },
  };
  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `mock-ftns-${network}-${Date.now()}.json`);
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  console.log(
    `\nNext step — export and re-run deploy-provenance.js:\n` +
    `  export FTNS_TOKEN_ADDRESS="${mockAddress}"\n` +
    `  export NETWORK_TREASURY="${deployer.address}"  # or a separate treasury\n` +
    `  npx hardhat run scripts/deploy-provenance.js --network ${network}`
  );
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
