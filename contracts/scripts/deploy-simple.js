const { ethers, upgrades } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("🚀 Starting PRSM FTNS Simple Token Deployment");
  console.log("Network:", hre.network.name);
  
  // Get deployer account
  const [deployer] = await ethers.getSigners();
  console.log("Deployer address:", deployer.address);
  console.log("Deployer balance:", ethers.utils.formatEther(await deployer.getBalance()), "MATIC");

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
    await ftnsToken.deployed();
    
    deployedContracts.ftnsToken = ftnsToken.address;
    console.log("✅ FTNS Token deployed to:", ftnsToken.address);
    console.log("   Implementation:", await upgrades.erc1967.getImplementationAddress(ftnsToken.address));

    // Verify deployment
    console.log("\n🔍 Verifying deployment...");
    
    const tokenName = await ftnsToken.name();
    const tokenSymbol = await ftnsToken.symbol();
    const tokenDecimals = await ftnsToken.decimals();
    const totalSupply = await ftnsToken.totalSupply();
    
    console.log(`- Token: ${tokenName} (${tokenSymbol}) with ${tokenDecimals} decimals`);
    console.log(`- Total Supply: ${ethers.utils.formatEther(totalSupply)} FTNS`);
    console.log(`- Deployer Balance: ${ethers.utils.formatEther(await ftnsToken.balanceOf(deployer.address))} FTNS`);

    // Save deployment data
    const deploymentData = {
      network: hre.network.name,
      chainId: await deployer.getChainId(),
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
    
    fs.writeFileSync(deploymentFile, JSON.stringify(deploymentData, null, 2));
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