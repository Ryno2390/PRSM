/**
 * PRSM Smart Contract Deployment Script
 * ====================================
 * 
 * Comprehensive deployment script for all PRSM smart contracts
 * Supports multiple networks and environments
 * Includes verification and configuration
 */

const { ethers, upgrades } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Deployment configuration
const DEPLOYMENT_CONFIG = {
    polygon_mainnet: {
        name: "Polygon Mainnet",
        chainId: 137,
        confirmations: 5,
        gasPrice: "30000000000", // 30 gwei
        initialSupply: ethers.utils.parseEther("100000000"), // 100M tokens
        votingDelay: 1, // 1 block
        votingPeriod: 50400, // 1 week
        proposalThreshold: ethers.utils.parseEther("1000"), // 1000 tokens
        quorumFraction: 4, // 4%
        timelockDelay: 172800, // 2 days
    },
    polygon_mumbai: {
        name: "Polygon Mumbai Testnet",
        chainId: 80001,
        confirmations: 2,
        gasPrice: "20000000000", // 20 gwei
        initialSupply: ethers.utils.parseEther("10000000"), // 10M tokens for testing
        votingDelay: 1,
        votingPeriod: 100, // Shorter for testing
        proposalThreshold: ethers.utils.parseEther("100"), // 100 tokens
        quorumFraction: 2, // 2% for testing
        timelockDelay: 3600, // 1 hour for testing
    }
};

class ContractDeployer {
    constructor(network) {
        this.network = network;
        this.config = DEPLOYMENT_CONFIG[network];
        this.deployedContracts = {};
        this.deploymentLog = [];
        
        if (!this.config) {
            throw new Error(`Unsupported network: ${network}`);
        }
        
        console.log(`🚀 Deploying to ${this.config.name} (Chain ID: ${this.config.chainId})`);
    }
    
    async deploy() {
        try {
            console.log("📋 Starting PRSM contract deployment...\n");
            
            // Get deployer account
            const [deployer] = await ethers.getSigners();
            console.log(`👤 Deploying with account: ${deployer.address}`);
            console.log(`💰 Account balance: ${ethers.utils.formatEther(await deployer.getBalance())} MATIC\n`);
            
            // Deploy contracts in order
            await this.deployTimelockController(deployer);
            await this.deployFTNSToken(deployer);
            await this.deployFTNSGovernance(deployer);
            await this.deployFTNSMarketplace(deployer);
            
            // Configure contracts
            await this.configureContracts();
            
            // Save deployment info
            await this.saveDeploymentInfo();
            
            // Verify contracts if on mainnet/testnet
            if (this.network !== "hardhat" && this.network !== "localhost") {
                await this.verifyContracts();
            }
            
            console.log("✅ Deployment completed successfully!");
            this.printDeploymentSummary();
            
        } catch (error) {
            console.error("❌ Deployment failed:", error);
            throw error;
        }
    }
    
    async deployTimelockController(deployer) {
        console.log("📦 Deploying TimelockController...");
        
        const TimelockController = await ethers.getContractFactory("TimelockController");
        const timelock = await TimelockController.deploy(
            this.config.timelockDelay,
            [], // proposers (will be set to governance)
            [], // executors (will be set to governance)
            deployer.address // admin (will be renounced later)
        );
        
        await timelock.deployed();
        await this.waitForConfirmations(timelock);
        
        this.deployedContracts.timelock = timelock;
        this.logDeployment("TimelockController", timelock.address);
        
        console.log(`✅ TimelockController deployed at: ${timelock.address}\n`);
    }
    
    async deployFTNSToken(deployer) {
        console.log("📦 Deploying FTNS Token...");
        
        const FTNSToken = await ethers.getContractFactory("FTNSToken");
        const token = await FTNSToken.deploy(deployer.address);
        
        await token.deployed();
        await this.waitForConfirmations(token);
        
        this.deployedContracts.token = token;
        this.logDeployment("FTNSToken", token.address);
        
        console.log(`✅ FTNS Token deployed at: ${token.address}`);
        console.log(`   Initial supply: ${ethers.utils.formatEther(await token.totalSupply())} FTNS\n`);
    }
    
    async deployFTNSGovernance(deployer) {
        console.log("📦 Deploying FTNS Governance...");
        
        const FTNSGovernance = await ethers.getContractFactory("FTNSGovernance");
        const governance = await FTNSGovernance.deploy(
            this.deployedContracts.token.address,
            this.deployedContracts.timelock.address,
            deployer.address
        );
        
        await governance.deployed();
        await this.waitForConfirmations(governance);
        
        this.deployedContracts.governance = governance;
        this.logDeployment("FTNSGovernance", governance.address);
        
        console.log(`✅ FTNS Governance deployed at: ${governance.address}\n`);
    }
    
    async deployFTNSMarketplace(deployer) {
        console.log("📦 Deploying FTNS Marketplace...");
        
        const FTNSMarketplace = await ethers.getContractFactory("FTNSMarketplace");
        const marketplace = await FTNSMarketplace.deploy(
            this.deployedContracts.token.address,
            deployer.address, // Fee recipient
            deployer.address  // Initial owner
        );
        
        await marketplace.deployed();
        await this.waitForConfirmations(marketplace);
        
        this.deployedContracts.marketplace = marketplace;
        this.logDeployment("FTNSMarketplace", marketplace.address);
        
        console.log(`✅ FTNS Marketplace deployed at: ${marketplace.address}\n`);
    }
    
    async configureContracts() {
        console.log("⚙️  Configuring contracts...\n");
        
        const [deployer] = await ethers.getSigners();
        
        // Configure timelock roles
        console.log("🔧 Configuring Timelock roles...");
        const PROPOSER_ROLE = await this.deployedContracts.timelock.PROPOSER_ROLE();
        const EXECUTOR_ROLE = await this.deployedContracts.timelock.EXECUTOR_ROLE();
        
        // Grant roles to governance contract
        await this.deployedContracts.timelock.grantRole(PROPOSER_ROLE, this.deployedContracts.governance.address);
        await this.deployedContracts.timelock.grantRole(EXECUTOR_ROLE, this.deployedContracts.governance.address);
        
        // Configure token permissions
        console.log("🔧 Configuring Token permissions...");
        await this.deployedContracts.token.setAuthorizedMinter(this.deployedContracts.marketplace.address, true);
        await this.deployedContracts.token.setAuthorizedMinter(this.deployedContracts.governance.address, true);
        
        // Add initial arbitrator to marketplace
        console.log("🔧 Configuring Marketplace...");
        await this.deployedContracts.marketplace.addArbitrator(deployer.address);
        
        console.log("✅ Contract configuration completed\n");
    }
    
    async waitForConfirmations(contract) {
        if (this.config.confirmations > 0) {
            console.log(`   ⏳ Waiting for ${this.config.confirmations} confirmations...`);
            await contract.deployTransaction.wait(this.config.confirmations);
        }
    }
    
    logDeployment(contractName, address) {
        this.deploymentLog.push({
            contract: contractName,
            address: address,
            network: this.network,
            timestamp: new Date().toISOString(),
            blockNumber: null // Will be filled later
        });
    }
    
    async saveDeploymentInfo() {
        console.log("💾 Saving deployment information...");
        
        const deploymentInfo = {
            network: this.network,
            chainId: this.config.chainId,
            timestamp: new Date().toISOString(),
            contracts: {
                FTNSToken: this.deployedContracts.token.address,
                FTNSGovernance: this.deployedContracts.governance.address,
                FTNSMarketplace: this.deployedContracts.marketplace.address,
                TimelockController: this.deployedContracts.timelock.address,
            },
            config: this.config,
            deploymentLog: this.deploymentLog
        };
        
        // Create deployments directory if it doesn't exist
        const deploymentsDir = path.join(__dirname, "../deployments");
        if (!fs.existsSync(deploymentsDir)) {
            fs.mkdirSync(deploymentsDir, { recursive: true });
        }
        
        // Save to network-specific file
        const filePath = path.join(deploymentsDir, `${this.network}.json`);
        fs.writeFileSync(filePath, JSON.stringify(deploymentInfo, null, 2));
        
        // Also save a general deployments file
        const allDeployments = this.loadAllDeployments();
        allDeployments[this.network] = deploymentInfo;
        
        const allDeploymentsPath = path.join(deploymentsDir, "deployments.json");
        fs.writeFileSync(allDeploymentsPath, JSON.stringify(allDeployments, null, 2));
        
        console.log(`✅ Deployment info saved to ${filePath}\n`);
    }
    
    loadAllDeployments() {
        const deploymentsPath = path.join(__dirname, "../deployments/deployments.json");
        if (fs.existsSync(deploymentsPath)) {
            return JSON.parse(fs.readFileSync(deploymentsPath, "utf8"));
        }
        return {};
    }
    
    async verifyContracts() {
        console.log("🔍 Verifying contracts on block explorer...\n");
        
        try {
            // Verify TimelockController
            await this.verifyContract(
                "TimelockController",
                this.deployedContracts.timelock.address,
                [
                    this.config.timelockDelay,
                    [],
                    [],
                    (await ethers.getSigners())[0].address
                ]
            );
            
            // Verify FTNSToken
            await this.verifyContract(
                "FTNSToken",
                this.deployedContracts.token.address,
                [(await ethers.getSigners())[0].address]
            );
            
            // Verify FTNSGovernance
            await this.verifyContract(
                "FTNSGovernance", 
                this.deployedContracts.governance.address,
                [
                    this.deployedContracts.token.address,
                    this.deployedContracts.timelock.address,
                    (await ethers.getSigners())[0].address
                ]
            );
            
            // Verify FTNSMarketplace
            await this.verifyContract(
                "FTNSMarketplace",
                this.deployedContracts.marketplace.address,
                [
                    this.deployedContracts.token.address,
                    (await ethers.getSigners())[0].address,
                    (await ethers.getSigners())[0].address
                ]
            );
            
        } catch (error) {
            console.log("⚠️  Contract verification failed (this is non-critical):", error.message);
        }
    }
    
    async verifyContract(name, address, constructorArguments) {
        try {
            console.log(`🔍 Verifying ${name}...`);
            await hre.run("verify:verify", {
                address: address,
                constructorArguments: constructorArguments,
            });
            console.log(`✅ ${name} verified`);
        } catch (error) {
            console.log(`⚠️  ${name} verification failed:`, error.message);
        }
    }
    
    printDeploymentSummary() {
        console.log("\n" + "=".repeat(60));
        console.log("🎉 PRSM DEPLOYMENT SUMMARY");
        console.log("=".repeat(60));
        console.log(`Network: ${this.config.name}`);
        console.log(`Chain ID: ${this.config.chainId}`);
        console.log(`Timestamp: ${new Date().toISOString()}\n`);
        
        console.log("📋 Deployed Contracts:");
        console.log(`   🪙  FTNS Token:        ${this.deployedContracts.token.address}`);
        console.log(`   🗳️   FTNS Governance:   ${this.deployedContracts.governance.address}`);
        console.log(`   🏪  FTNS Marketplace:  ${this.deployedContracts.marketplace.address}`);
        console.log(`   ⏰  Timelock:          ${this.deployedContracts.timelock.address}\n`);
        
        console.log("⚙️  Configuration:");
        console.log(`   Initial Supply:       ${ethers.utils.formatEther(this.config.initialSupply)} FTNS`);
        console.log(`   Voting Delay:         ${this.config.votingDelay} blocks`);
        console.log(`   Voting Period:        ${this.config.votingPeriod} blocks`);
        console.log(`   Proposal Threshold:   ${ethers.utils.formatEther(this.config.proposalThreshold)} FTNS`);
        console.log(`   Quorum Fraction:      ${this.config.quorumFraction}%`);
        console.log(`   Timelock Delay:       ${this.config.timelockDelay} seconds\n`);
        
        console.log("🔗 Integration URLs:");
        console.log(`   Add to MetaMask:      https://github.com/prsm-ai/smart-contracts`);
        console.log(`   Block Explorer:       https://polygonscan.com/address/${this.deployedContracts.token.address}`);
        console.log(`   Governance UI:        https://tally.xyz/governance/YOUR_GOVERNANCE_ID\n`);
        
        console.log("📖 Next Steps:");
        console.log("   1. Update frontend configuration with new contract addresses");
        console.log("   2. Verify contracts on block explorer if not done automatically");
        console.log("   3. Set up governance proposals for initial configuration");
        console.log("   4. Configure monitoring and alerting");
        console.log("   5. Run integration tests against deployed contracts");
        
        console.log("=".repeat(60));
    }
}

// Main deployment function
async function main() {
    const network = hre.network.name;
    
    console.log("🚀 PRSM Smart Contract Deployment");
    console.log("==================================\n");
    
    const deployer = new ContractDeployer(network);
    await deployer.deploy();
}

// Error handling
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("❌ Deployment failed:", error);
        process.exit(1);
    });

module.exports = { ContractDeployer, DEPLOYMENT_CONFIG };