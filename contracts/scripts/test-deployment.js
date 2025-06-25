const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

/**
 * Test deployment script to validate contract deployment setup
 * This script simulates deployment without actually deploying to save gas
 */

async function main() {
  console.log("🧪 Testing PRSM FTNS Deployment Setup");
  console.log("Network:", hre.network.name);
  
  try {
    // Get deployer account
    const [deployer] = await ethers.getSigners();
    console.log("Deployer address:", deployer.address);
    
    // Check if on testnet
    const chainId = await deployer.getChainId();
    console.log("Chain ID:", chainId);
    
    if (hre.network.name !== "hardhat" && hre.network.name !== "localhost") {
      const balance = await deployer.getBalance();
      console.log("Deployer balance:", ethers.utils.formatEther(balance), "MATIC");
      
      // Check minimum balance for deployment
      const minBalance = ethers.utils.parseEther("0.01"); // 0.01 MATIC minimum
      if (balance.lt(minBalance)) {
        console.warn("⚠️  Low balance detected. Get MATIC from faucet:");
        console.warn("   https://faucet.polygon.technology/");
      }
    }

    // Test contract compilation
    console.log("\n📋 Testing contract compilation...");
    const FTNSTokenSimple = await ethers.getContractFactory("FTNSTokenSimple");
    console.log("✅ FTNSTokenSimple contract compiled successfully");
    
    // Estimate deployment gas
    console.log("\n⛽ Estimating deployment gas...");
    const deploymentData = FTNSTokenSimple.getDeployTransaction(
      deployer.address,
      deployer.address
    );
    
    if (hre.network.name !== "hardhat") {
      try {
        const gasEstimate = await deployer.estimateGas(deploymentData);
        const gasPrice = await deployer.getGasPrice();
        const estimatedCost = gasEstimate.mul(gasPrice);
        
        console.log("Estimated gas:", gasEstimate.toString());
        console.log("Gas price:", ethers.utils.formatUnits(gasPrice, "gwei"), "gwei");
        console.log("Estimated cost:", ethers.utils.formatEther(estimatedCost), "MATIC");
      } catch (e) {
        console.warn("Could not estimate gas:", e.message);
      }
    }
    
    // Test proxy deployment setup
    console.log("\n🔄 Testing upgradeable proxy setup...");
    try {
      const { upgrades } = require("@openzeppelin/hardhat-upgrades");
      console.log("✅ OpenZeppelin upgrades plugin available");
      
      // This would be the actual deployment on testnet
      if (process.env.DEPLOY_FOR_REAL === "true") {
        console.log("\n🚀 DEPLOYING TO TESTNET...");
        
        const ftnsToken = await upgrades.deployProxy(
          FTNSTokenSimple,
          [deployer.address, deployer.address],
          { 
            initializer: "initialize",
            kind: "uups"
          }
        );
        await ftnsToken.deployed();
        
        console.log("✅ FTNS Token deployed to:", ftnsToken.address);
        console.log("   Implementation:", await upgrades.erc1967.getImplementationAddress(ftnsToken.address));
        
        // Verify contract functionality
        const name = await ftnsToken.name();
        const symbol = await ftnsToken.symbol();
        const totalSupply = await ftnsToken.totalSupply();
        
        console.log("\n🔍 Contract verification:");
        console.log("- Name:", name);
        console.log("- Symbol:", symbol);
        console.log("- Total Supply:", ethers.utils.formatEther(totalSupply), "FTNS");
        
        // Save deployment info
        const deploymentData = {
          network: hre.network.name,
          chainId: chainId,
          deployer: deployer.address,
          timestamp: new Date().toISOString(),
          contracts: {
            ftnsToken: ftnsToken.address
          },
          implementation: await upgrades.erc1967.getImplementationAddress(ftnsToken.address),
          verification: {
            name: name,
            symbol: symbol,
            totalSupply: totalSupply.toString()
          }
        };
        
        const deploymentsDir = path.join(__dirname, "../deployments");
        if (!fs.existsSync(deploymentsDir)) {
          fs.mkdirSync(deploymentsDir, { recursive: true });
        }
        
        const deploymentFile = path.join(
          deploymentsDir,
          `${hre.network.name}-${Date.now()}.json`
        );
        
        fs.writeFileSync(deploymentFile, JSON.stringify(deploymentData, null, 2));
        
        console.log("\n📄 Deployment saved to:", deploymentFile);
        console.log("\n🎉 DEPLOYMENT COMPLETE!");
        console.log("\n📋 Next steps:");
        console.log("1. Update PRSM .env with contract address:");
        console.log(`   FTNS_TOKEN_ADDRESS=${ftnsToken.address}`);
        console.log("2. Verify contract on PolygonScan:");
        console.log(`   npx hardhat verify --network ${hre.network.name} ${ftnsToken.address}`);
        console.log("3. Test Web3 integration");
        
        return deploymentData;
      } else {
        console.log("✅ Proxy deployment setup validated");
        console.log("ℹ️  To deploy for real, set DEPLOY_FOR_REAL=true");
      }
    } catch (e) {
      console.error("❌ Proxy setup error:", e.message);
    }
    
    // Test network connection
    console.log("\n🌐 Testing network connection...");
    try {
      const blockNumber = await deployer.provider.getBlockNumber();
      console.log("✅ Connected to network, current block:", blockNumber);
    } catch (e) {
      console.error("❌ Network connection failed:", e.message);
    }
    
    // Validate configuration
    console.log("\n⚙️  Configuration validation:");
    const requiredEnvVars = ["POLYGON_MUMBAI_RPC_URL"];
    for (const envVar of requiredEnvVars) {
      if (process.env[envVar]) {
        console.log(`✅ ${envVar}: configured`);
      } else {
        console.log(`⚠️  ${envVar}: not configured`);
      }
    }
    
    if (process.env.PRIVATE_KEY) {
      console.log("✅ PRIVATE_KEY: configured");
    } else {
      console.log("⚠️  PRIVATE_KEY: not configured (required for deployment)");
    }
    
    console.log("\n✅ Deployment setup validation complete!");
    
    if (hre.network.name === "polygon-mumbai") {
      console.log("\n🎯 Ready for testnet deployment!");
      console.log("Run with DEPLOY_FOR_REAL=true to deploy:");
      console.log("DEPLOY_FOR_REAL=true npx hardhat run scripts/test-deployment.js --network polygon-mumbai");
    }
    
  } catch (error) {
    console.error("\n❌ Validation failed:", error);
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