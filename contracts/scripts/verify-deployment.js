const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("🔍 PRSM FTNS Smart Contract Deployment Verification");
  console.log("Network:", hre.network.name);
  
  // Load deployment data
  const deploymentsDir = path.join(__dirname, "../deployments");
  const deploymentFiles = fs.readdirSync(deploymentsDir)
    .filter(file => file.startsWith(hre.network.name))
    .sort()
    .reverse(); // Get most recent deployment
  
  if (deploymentFiles.length === 0) {
    throw new Error(`No deployment files found for network: ${hre.network.name}`);
  }
  
  const latestDeployment = JSON.parse(
    fs.readFileSync(path.join(deploymentsDir, deploymentFiles[0]))
  );
  
  console.log("📄 Using deployment:", deploymentFiles[0]);
  console.log("Deployed by:", latestDeployment.deployer);
  console.log("Deployment time:", latestDeployment.timestamp);
  
  const contracts = latestDeployment.contracts;
  
  // Get contract instances
  const ftnsToken = await ethers.getContractAt("FTNSToken", contracts.ftnsToken);
  const marketplace = await ethers.getContractAt("FTNSMarketplace", contracts.marketplace);
  const governance = await ethers.getContractAt("FTNSGovernance", contracts.governance);
  const timelock = await ethers.getContractAt("TimelockController", contracts.timelock);

  console.log("\n" + "=".repeat(60));
  console.log("🔬 RUNNING VERIFICATION TESTS");
  console.log("=".repeat(60));

  let testsPassed = 0;
  let testsTotal = 0;

  async function test(description, testFn) {
    testsTotal++;
    try {
      await testFn();
      console.log(`✅ ${description}`);
      testsPassed++;
    } catch (error) {
      console.log(`❌ ${description}`);
      console.log(`   Error: ${error.message}`);
    }
  }

  // === FTNS Token Tests ===
  console.log("\n🪙 FTNS Token Verification:");

  await test("Token has correct name and symbol", async () => {
    const name = await ftnsToken.name();
    const symbol = await ftnsToken.symbol();
    if (name !== "PRSM Fungible Tokens for Node Support" || symbol !== "FTNS") {
      throw new Error(`Incorrect name/symbol: ${name}/${symbol}`);
    }
  });

  await test("Token has initial supply", async () => {
    const totalSupply = await ftnsToken.totalSupply();
    const expectedSupply = ethers.utils.parseEther("100000000"); // 100M
    if (!totalSupply.eq(expectedSupply)) {
      throw new Error(`Incorrect supply: ${ethers.utils.formatEther(totalSupply)}`);
    }
  });

  await test("Token supports all required interfaces", async () => {
    const interfaces = [
      "0x01ffc9a7", // ERC165
      "0x36372b07", // ERC20
      "0x7965db0b", // AccessControl
    ];
    
    for (const interfaceId of interfaces) {
      const supported = await ftnsToken.supportsInterface(interfaceId);
      if (!supported) {
        throw new Error(`Interface ${interfaceId} not supported`);
      }
    }
  });

  await test("Token has correct max supply", async () => {
    const maxSupply = await ftnsToken.MAX_SUPPLY();
    const expectedMax = ethers.utils.parseEther("1000000000"); // 1B
    if (!maxSupply.eq(expectedMax)) {
      throw new Error(`Incorrect max supply: ${ethers.utils.formatEther(maxSupply)}`);
    }
  });

  // === Marketplace Tests ===
  console.log("\n🏪 Marketplace Verification:");

  await test("Marketplace has correct FTNS token reference", async () => {
    const tokenAddress = await marketplace.ftnsToken();
    if (tokenAddress.toLowerCase() !== contracts.ftnsToken.toLowerCase()) {
      throw new Error(`Incorrect token reference: ${tokenAddress}`);
    }
  });

  await test("Marketplace has correct platform fee", async () => {
    const feeRate = await marketplace.platformFeeRate();
    if (feeRate.toNumber() !== 250) { // 2.5%
      throw new Error(`Incorrect fee rate: ${feeRate.toNumber()}`);
    }
  });

  await test("Marketplace has correct max fee limit", async () => {
    const maxFee = await marketplace.MAX_FEE_RATE();
    if (maxFee.toNumber() !== 1000) { // 10%
      throw new Error(`Incorrect max fee: ${maxFee.toNumber()}`);
    }
  });

  // === Governance Tests ===
  console.log("\n🗳️  Governance Verification:");

  await test("Governance has correct token reference", async () => {
    const tokenAddress = await governance.token();
    if (tokenAddress.toLowerCase() !== contracts.ftnsToken.toLowerCase()) {
      throw new Error(`Incorrect token reference: ${tokenAddress}`);
    }
  });

  await test("Governance has correct voting delay", async () => {
    const votingDelay = await governance.votingDelay();
    if (votingDelay.toNumber() !== 7200) { // 1 day
      throw new Error(`Incorrect voting delay: ${votingDelay.toNumber()}`);
    }
  });

  await test("Governance has correct voting period", async () => {
    const votingPeriod = await governance.votingPeriod();
    if (votingPeriod.toNumber() !== 50400) { // 7 days
      throw new Error(`Incorrect voting period: ${votingPeriod.toNumber()}`);
    }
  });

  await test("Governance has minimum stake requirement", async () => {
    const minStake = await governance.minimumStakeToPropose();
    const expectedStake = ethers.utils.parseEther("1000");
    if (!minStake.eq(expectedStake)) {
      throw new Error(`Incorrect min stake: ${ethers.utils.formatEther(minStake)}`);
    }
  });

  // === Timelock Tests ===
  console.log("\n🕐 Timelock Verification:");

  await test("Timelock has correct delay", async () => {
    const delay = await timelock.getMinDelay();
    if (delay.toNumber() !== 172800) { // 2 days
      throw new Error(`Incorrect timelock delay: ${delay.toNumber()}`);
    }
  });

  // === Integration Tests ===
  console.log("\n🔗 Integration Verification:");

  await test("FTNS token has minter role for governance", async () => {
    const minterRole = await ftnsToken.MINTER_ROLE();
    const hasRole = await ftnsToken.hasRole(minterRole, contracts.governance);
    if (!hasRole) {
      throw new Error("Governance doesn't have minter role");
    }
  });

  await test("FTNS token has minter role for marketplace", async () => {
    const minterRole = await ftnsToken.MINTER_ROLE();
    const hasRole = await ftnsToken.hasRole(minterRole, contracts.marketplace);
    if (!hasRole) {
      throw new Error("Marketplace doesn't have minter role");
    }
  });

  await test("Timelock has proposer role for governance", async () => {
    const proposerRole = await timelock.PROPOSER_ROLE();
    const hasRole = await timelock.hasRole(proposerRole, contracts.governance);
    if (!hasRole) {
      throw new Error("Governance doesn't have proposer role on timelock");
    }
  });

  await test("Timelock has executor role for governance", async () => {
    const executorRole = await timelock.EXECUTOR_ROLE();
    const hasRole = await timelock.hasRole(executorRole, contracts.governance);
    if (!hasRole) {
      throw new Error("Governance doesn't have executor role on timelock");
    }
  });

  // === Functionality Tests ===
  console.log("\n⚙️  Functionality Verification:");

  const [deployer] = await ethers.getSigners();

  await test("Can get account info from token", async () => {
    const accountInfo = await ftnsToken.getAccountInfo(deployer.address);
    // Should have structure with liquid, locked, staked, etc.
    if (typeof accountInfo.liquid === 'undefined') {
      throw new Error("Account info structure incorrect");
    }
  });

  await test("Can get marketplace stats", async () => {
    const stats = await marketplace.getMarketplaceStats();
    // Should have structure with totalListings, activeListings, etc.
    if (typeof stats.totalListings === 'undefined') {
      throw new Error("Marketplace stats structure incorrect");
    }
  });

  await test("Can check proposal creation capability", async () => {
    const canCreate = await governance.canCreateProposal(deployer.address);
    // Should return boolean
    if (typeof canCreate !== 'boolean') {
      throw new Error("canCreateProposal should return boolean");
    }
  });

  // === Contract Size Verification ===
  console.log("\n📏 Contract Size Verification:");

  await test("Contracts are within size limits", async () => {
    for (const [name, address] of Object.entries(contracts)) {
      const code = await ethers.provider.getCode(address);
      const sizeKB = (code.length - 2) / 2 / 1024; // Remove 0x prefix, convert to KB
      
      console.log(`   ${name}: ${sizeKB.toFixed(1)} KB`);
      
      if (sizeKB > 24) { // Ethereum contract size limit
        throw new Error(`${name} exceeds size limit: ${sizeKB.toFixed(1)} KB`);
      }
    }
  });

  // === Final Summary ===
  console.log("\n" + "=".repeat(60));
  console.log("📊 VERIFICATION SUMMARY");
  console.log("=".repeat(60));
  console.log(`Tests Passed: ${testsPassed}/${testsTotal}`);
  console.log(`Success Rate: ${((testsPassed / testsTotal) * 100).toFixed(1)}%`);
  
  if (testsPassed === testsTotal) {
    console.log("\n🎉 ALL TESTS PASSED - DEPLOYMENT VERIFIED!");
    console.log("\n🔗 Contract Addresses:");
    for (const [name, address] of Object.entries(contracts)) {
      console.log(`${name}: ${address}`);
    }
    
    console.log("\n📋 Next Steps:");
    console.log("1. Verify contracts on PolygonScan");
    console.log("2. Update frontend configuration");
    console.log("3. Create initial marketplace listings");
    console.log("4. Begin user onboarding");
    
    if (hre.network.name === "polygon-mumbai") {
      console.log("\n🧪 Testnet Verification Commands:");
      console.log(`npx hardhat verify --network polygon-mumbai ${contracts.ftnsToken}`);
      console.log(`npx hardhat verify --network polygon-mumbai ${contracts.marketplace}`);
      console.log(`npx hardhat verify --network polygon-mumbai ${contracts.governance}`);
      console.log(`npx hardhat verify --network polygon-mumbai ${contracts.timelock}`);
    }
    
  } else {
    console.log("\n❌ VERIFICATION FAILED!");
    console.log("Please review the failed tests above before proceeding.");
    process.exit(1);
  }
}

if (require.main === module) {
  main()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}

module.exports = main;