const { ethers, upgrades } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("🚀 Starting PRSM FTNS Smart Contract Deployment");
  console.log("Network:", hre.network.name);
  
  // Get deployer account
  const [deployer] = await ethers.getSigners();
  console.log("Deployer address:", deployer.address);
  console.log("Deployer balance:", ethers.utils.formatEther(await deployer.getBalance()), "MATIC");

  // Deployment configuration
  const config = {
    // FTNS Token
    ftnsToken: {
      initialOwner: deployer.address,
      treasuryAddress: deployer.address, // In production, this would be a multisig
    },
    
    // Marketplace
    marketplace: {
      platformFeeRate: 250, // 2.5% platform fee
      feeRecipient: deployer.address, // In production, this would be treasury
    },
    
    // Governance
    governance: {
      minimumStakeToPropose: ethers.utils.parseEther("1000"), // 1000 FTNS to create proposal
      votingDelay: 7200,    // 1 day (assuming 12s blocks)
      votingPeriod: 50400,  // 7 days
      quorumFraction: 10,   // 10% quorum
      timelockDelay: 172800 // 2 days timelock
    }
  };

  const deployedContracts = {};

  try {
    // 1. Deploy FTNS Token
    console.log("\n📄 Deploying FTNS Token...");
    const FTNSToken = await ethers.getContractFactory("FTNSToken");
    
    const ftnsToken = await upgrades.deployProxy(
      FTNSToken,
      [config.ftnsToken.initialOwner, config.ftnsToken.treasuryAddress],
      { 
        initializer: "initialize",
        kind: "uups"
      }
    );
    await ftnsToken.deployed();
    
    deployedContracts.ftnsToken = ftnsToken.address;
    console.log("✅ FTNS Token deployed to:", ftnsToken.address);
    console.log("   Implementation:", await upgrades.erc1967.getImplementationAddress(ftnsToken.address));

    // 2. Deploy Timelock Controller for Governance
    console.log("\n🕐 Deploying Timelock Controller...");
    const TimelockController = await ethers.getContractFactory("TimelockController");
    
    const timelock = await TimelockController.deploy(
      config.governance.timelockDelay,
      [], // proposers (will be set to governance contract)
      [], // executors (will be set to governance contract)
      deployer.address // admin (will be renounced after setup)
    );
    await timelock.deployed();
    
    deployedContracts.timelock = timelock.address;
    console.log("✅ Timelock Controller deployed to:", timelock.address);

    // 3. Deploy Governance
    console.log("\n🗳️  Deploying FTNS Governance...");
    const FTNSGovernance = await ethers.getContractFactory("FTNSGovernance");
    
    const governance = await upgrades.deployProxy(
      FTNSGovernance,
      [
        ftnsToken.address,
        timelock.address,
        config.governance.minimumStakeToPropose
      ],
      {
        initializer: "initialize",
        kind: "uups"
      }
    );
    await governance.deployed();
    
    deployedContracts.governance = governance.address;
    console.log("✅ FTNS Governance deployed to:", governance.address);
    console.log("   Implementation:", await upgrades.erc1967.getImplementationAddress(governance.address));

    // 4. Deploy Marketplace
    console.log("\n🏪 Deploying FTNS Marketplace...");
    const FTNSMarketplace = await ethers.getContractFactory("FTNSMarketplace");
    
    const marketplace = await upgrades.deployProxy(
      FTNSMarketplace,
      [
        ftnsToken.address,
        config.marketplace.feeRecipient,
        config.marketplace.platformFeeRate
      ],
      {
        initializer: "initialize",
        kind: "uups"
      }
    );
    await marketplace.deployed();
    
    deployedContracts.marketplace = marketplace.address;
    console.log("✅ FTNS Marketplace deployed to:", marketplace.address);
    console.log("   Implementation:", await upgrades.erc1967.getImplementationAddress(marketplace.address));

    // 5. Setup roles and permissions
    console.log("\n🔧 Setting up roles and permissions...");
    
    // Grant MINTER_ROLE to governance for rewards
    console.log("- Granting MINTER_ROLE to governance contract...");
    const minterRole = await ftnsToken.MINTER_ROLE();
    await ftnsToken.grantRole(minterRole, governance.address);
    
    // Grant MINTER_ROLE to marketplace for operations  
    console.log("- Granting MINTER_ROLE to marketplace contract...");
    await ftnsToken.grantRole(minterRole, marketplace.address);
    
    // Setup timelock roles
    console.log("- Setting up timelock permissions...");
    const proposerRole = await timelock.PROPOSER_ROLE();
    const executorRole = await timelock.EXECUTOR_ROLE();
    const timelockAdminRole = await timelock.TIMELOCK_ADMIN_ROLE();
    
    // Grant proposer and executor roles to governance
    await timelock.grantRole(proposerRole, governance.address);
    await timelock.grantRole(executorRole, governance.address);
    
    // Grant executor role to deployer temporarily (for emergency)
    await timelock.grantRole(executorRole, deployer.address);
    
    // Revoke admin role from deployer (governance takes control)
    await timelock.renounceRole(timelockAdminRole, deployer.address);
    
    console.log("✅ Roles and permissions configured");

    // 6. Initial token distribution
    console.log("\n💰 Configuring initial token distribution...");
    
    // Check initial supply
    const initialSupply = await ftnsToken.totalSupply();
    console.log("Initial supply:", ethers.utils.formatEther(initialSupply), "FTNS");
    
    // Transfer some tokens to governance for rewards
    const governanceAllocation = ethers.utils.parseEther("10000000"); // 10M FTNS
    console.log("- Transferring", ethers.utils.formatEther(governanceAllocation), "FTNS to governance treasury");
    await ftnsToken.transfer(governance.address, governanceAllocation);
    
    console.log("✅ Initial distribution complete");

    // 7. Verify deployments
    console.log("\n🔍 Verifying deployments...");
    
    // Test FTNS token basic functions
    const tokenName = await ftnsToken.name();
    const tokenSymbol = await ftnsToken.symbol();
    const tokenDecimals = await ftnsToken.decimals();
    console.log(`- FTNS Token: ${tokenName} (${tokenSymbol}) with ${tokenDecimals} decimals`);
    
    // Test marketplace basic functions
    const marketplaceFeeRate = await marketplace.platformFeeRate();
    console.log(`- Marketplace fee rate: ${marketplaceFeeRate / 100}%`);
    
    // Test governance basic functions
    const minStake = await governance.minimumStakeToPropose();
    console.log(`- Minimum stake to propose: ${ethers.utils.formatEther(minStake)} FTNS`);
    
    console.log("✅ All deployments verified");

    // 8. Save deployment addresses
    const deploymentData = {
      network: hre.network.name,
      chainId: await deployer.getChainId(),
      deployer: deployer.address,
      timestamp: new Date().toISOString(),
      contracts: deployedContracts,
      config: config,
      gasUsed: {
        // Gas usage would be tracked during deployment
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
    console.log(`\n📄 Deployment data saved to: ${deploymentFile}`);

    // 9. Display summary
    console.log("\n" + "=".repeat(60));
    console.log("🎉 PRSM FTNS DEPLOYMENT COMPLETE!");
    console.log("=".repeat(60));
    console.log(`Network: ${hre.network.name}`);
    console.log(`Deployer: ${deployer.address}`);
    console.log("\n📄 Contract Addresses:");
    console.log(`FTNS Token:      ${deployedContracts.ftnsToken}`);
    console.log(`Timelock:        ${deployedContracts.timelock}`);
    console.log(`Governance:      ${deployedContracts.governance}`);
    console.log(`Marketplace:     ${deployedContracts.marketplace}`);
    
    console.log("\n🔗 Next Steps:");
    console.log("1. Verify contracts on PolygonScan");
    console.log("2. Update frontend configuration with contract addresses");
    console.log("3. Create initial marketplace listings");
    console.log("4. Setup governance proposals for parameter tuning");
    console.log("5. Begin community onboarding");
    
    console.log("\n💡 Useful Commands:");
    console.log(`npx hardhat verify --network ${hre.network.name} ${deployedContracts.ftnsToken}`);
    console.log(`npx hardhat verify --network ${hre.network.name} ${deployedContracts.marketplace}`);
    
    return deployedContracts;

  } catch (error) {
    console.error("\n❌ Deployment failed:", error);
    throw error;
  }
}

// Execute deployment
if (require.main === module) {
  main()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}

module.exports = main;