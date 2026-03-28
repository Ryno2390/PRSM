const { ethers, upgrades } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("🚀 Starting PRSM FTNS Simple Token Deployment");
  console.log("Network:", hre.network.name);
  
  // Get deployer account
  const [deployer] = await ethers.getSigners();
  console.log("Deployer address:", deployer.address);
  console.log("Deployer balance:", ethers.formatEther(await deployer.provider.getBalance(deployer.address)), "MATIC");

  const deployedContracts = {};

  try {
    // Deploy FTNS Token (Simple)
    console.log("\n📄 Deploying FTNS Token (Simple)...");
    const FTNSTokenSimple = await ethers.getContractFactory("FTNSTokenSimple");
    
    const ftnsToken = await upgrades.deployProxy(
      FTNSTokenSimple,
      [deployer.address, deployer.address], // owner and treasury
      { 
        initializer: "initialize",
        kind: "uups"
      }
    );
    
    
    deployedContracts.ftnsToken = await ftnsToken.getAddress();
    console.log("✅ FTNS Token deployed to:", await ftnsToken.getAddress());
    // implementation address lookup skipped (not needed for basic deploy)

    // Wait for deployment to be mined
    console.log("\n⏳ Waiting for deployment confirmation...");
    await ftnsToken.waitForDeployment();
    const tokenAddress = await ftnsToken.getAddress();
    console.log(`✅ FTNS Token confirmed at: ${tokenAddress}`);

    // Save deployment data
    const deploymentData = {
      network: hre.network.name,
      chainId: (await deployer.provider.getNetwork()).chainId,
      deployer: deployer.address,
      timestamp: new Date().toISOString(),
      contracts: deployedContracts,
      gasUsed: {}
    };
    
    const deploymentsDir = path.join(__dirname, "../deployments");
    if (!fs.existsSync(deploymentsDir)) {
      fs.mkdirSync(deploymentsDir, { recursive: true });
    }
    
    const deploymentFile = path.join(
      deploymentsDir,
      `${hre.network.name}-simple-${Date.now()}.json`
    );
    
    fs.writeFileSync(deploymentFile, JSON.stringify(deploymentData, (_, v) => typeof v === "bigint" ? v.toString() : v, 2));
    console.log(`\n📄 Deployment data saved to: ${deploymentFile}`);

    // Display summary
    console.log("\n" + "=".repeat(60));
    console.log("🎉 PRSM FTNS SIMPLE DEPLOYMENT COMPLETE!");
    console.log("=".repeat(60));
    console.log(`Network: ${hre.network.name}`);
    console.log(`Deployer: ${deployer.address}`);
    console.log(`\n📄 Contract Address:`);
    console.log(`FTNS Token: ${deployedContracts.ftnsToken}`);
    
    if (hre.network.name === "polygon-mumbai") {
      console.log("\n🧪 Testnet Verification Command:");
      console.log(`npx hardhat verify --network polygon-mumbai ${deployedContracts.ftnsToken}`);
    }
    
    return deployedContracts;

  } catch (error) {
    console.error("\n❌ Deployment failed:", error);
    throw error;
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