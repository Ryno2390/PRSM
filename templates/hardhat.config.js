require("@nomicfoundation/hardhat-toolbox");
require("@openzeppelin/hardhat-upgrades");
require("hardhat-gas-reporter");
require("solidity-coverage");
require("dotenv").config();

// Configuration
const POLYGON_MAINNET_RPC_URL = process.env.POLYGON_MAINNET_RPC_URL || "https://polygon-rpc.com";
const POLYGON_MUMBAI_RPC_URL = process.env.POLYGON_MUMBAI_RPC_URL || "https://rpc-mumbai.maticvigil.com";
const PRIVATE_KEY = process.env.PRIVATE_KEY || "0x0000000000000000000000000000000000000000000000000000000000000001";
const POLYGONSCAN_API_KEY = process.env.POLYGONSCAN_API_KEY || "";
const COINMARKETCAP_API_KEY = process.env.COINMARKETCAP_API_KEY || "";

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
      viaIR: true,
    },
  },
  
  networks: {
    hardhat: {
      chainId: 31337,
      gas: 12000000,
      blockGasLimit: 12000000,
      allowUnlimitedContractSize: true,
      accounts: {
        mnemonic: "test test test test test test test test test test test junk",
        count: 20,
        accountsBalance: "10000000000000000000000", // 10,000 ETH
      },
    },
    
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 31337,
    },
    
    polygon_mainnet: {
      url: POLYGON_MAINNET_RPC_URL,
      chainId: 137,
      accounts: PRIVATE_KEY !== "0x0000000000000000000000000000000000000000000000000000000000000001" 
        ? [PRIVATE_KEY] 
        : [],
      gasPrice: 30000000000, // 30 gwei
      gas: 2100000,
      confirmations: 5,
      timeoutBlocks: 200,
      skipDryRun: false
    },
    
    polygon_mumbai: {
      url: POLYGON_MUMBAI_RPC_URL,
      chainId: 80001,
      accounts: PRIVATE_KEY !== "0x0000000000000000000000000000000000000000000000000000000000000001" 
        ? [PRIVATE_KEY] 
        : [],
      gasPrice: 20000000000, // 20 gwei
      gas: 2100000,
      confirmations: 2,
      timeoutBlocks: 200,
      skipDryRun: true
    },
  },
  
  etherscan: {
    apiKey: {
      polygon: POLYGONSCAN_API_KEY,
      polygonMumbai: POLYGONSCAN_API_KEY,
    },
  },
  
  gasReporter: {
    enabled: process.env.REPORT_GAS !== undefined,
    currency: "USD",
    coinmarketcap: COINMARKETCAP_API_KEY,
    token: "MATIC",
    gasPrice: 30,
    showTimeSpent: true,
    showMethodSig: true,
    maxMethodDiff: 10,
  },
  
  mocha: {
    timeout: 300000, // 5 minutes
  },
  
  paths: {
    sources: "./templates/smart_contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  },
};