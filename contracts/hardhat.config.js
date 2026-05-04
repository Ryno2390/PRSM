require("@nomicfoundation/hardhat-toolbox");
require("@nomicfoundation/hardhat-verify");
require("@openzeppelin/hardhat-upgrades");
require("dotenv").config();

// Validate PRIVATE_KEY format before handing it to hardhat's network
// configs. The previous `process.env.PRIVATE_KEY ? [...] : []` guard
// only filtered empty/undefined; a placeholder like
// `your_private_key_here` still got passed through and tripped
// hardhat's "private key too short" config validator (blocks ALL
// network targets, including hardhat-local). Filter to the canonical
// 0x-prefixed 64-hex-char (32-byte) format.
const PK_RE = /^0x[0-9a-fA-F]{64}$/;
let _malformedPkWarned = false;  // dedup across N network configs
function pkAccounts() {
  const pk = process.env.PRIVATE_KEY;
  if (!pk) return [];
  if (!PK_RE.test(pk)) {
    // 2026-05-04 ceremony lesson L1: silently returning [] when PRIVATE_KEY
    // is set-but-malformed leaves the operator with `TypeError: Cannot read
    // properties of undefined (reading 'address')` deep inside the deploy
    // script — opaque + costs ~5 min of debug. A single warn here turns
    // that into a 5-second debug.
    //
    // Most common cause caught today: eth_account's `acct.key.hex()` in
    // some library versions returns the key WITHOUT the `0x` prefix.
    // Fix: `export PRIVATE_KEY="0x$PRIVATE_KEY"`.
    //
    // We continue returning [] (not throwing) so hardhat-local rehearsals
    // that don't need PRIVATE_KEY still work. Dedup'd across calls so the
    // warn fires once even though hardhat enumerates ~16 network configs.
    if (!_malformedPkWarned) {
      const len = pk.length;
      const hint = !pk.startsWith("0x")
        ? `missing 0x prefix; try: export PRIVATE_KEY="0x$PRIVATE_KEY"`
        : len !== 66
        ? `wrong length (got ${len}, expected 66 = 0x + 64 hex)`
        : `non-hex characters in key body`;
      console.warn(
        `[hardhat.config:pkAccounts] PRIVATE_KEY set but malformed; ` +
        `skipping all networks that require it (${hint})`
      );
      _malformedPkWarned = true;
    }
    return [];
  }
  return [pk];
}

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.22",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    // Local development network
    hardhat: {
      chainId: 31337,
      gas: 12000000,
      blockGasLimit: 12000000,
      allowUnlimitedContractSize: true
    },
    
    // Sepolia Testnet
    "sepolia": {
      url: process.env.SEPOLIA_RPC_URL || `https://sepolia.infura.io/v3/${process.env.INFURA_KEY}` || "",
      accounts: pkAccounts(),
      chainId: 11155111,
      gas: 6000000,
      gasPrice: 20000000000, // 20 gwei
      confirmations: 2,
      timeoutBlocks: 200,
      skipDryRun: true
    },
    
    // Polygon Mumbai Testnet
    "polygon-mumbai": {
      url: process.env.POLYGON_MUMBAI_RPC_URL || `https://polygon-mumbai.infura.io/v3/${process.env.INFURA_KEY}` || "https://rpc-mumbai.maticvigil.com",
      accounts: pkAccounts(),
      chainId: 80001,
      gas: 6000000,
      gasPrice: 10000000000, // 10 gwei
      confirmations: 2,
      timeoutBlocks: 200,
      skipDryRun: true
    },
    
    // Base Mainnet
    "base": {
      url: process.env.BASE_RPC_URL || "https://mainnet.base.org",
      accounts: pkAccounts(),
      chainId: 8453,
      gas: "auto",
      gasPrice: "auto",
      confirmations: 3,
      timeoutBlocks: 200,
      skipDryRun: false
    },

    // Base Sepolia Testnet
    "base-sepolia": {
      url: process.env.BASE_SEPOLIA_RPC_URL || "https://sepolia.base.org",
      accounts: pkAccounts(),
      chainId: 84532,
      gas: "auto",
      gasPrice: "auto",
      confirmations: 2,
      timeoutBlocks: 200,
      skipDryRun: false
    },

    // Base Fork (dry-run)
    "base-fork": {
      url: process.env.BASE_RPC_URL || "https://mainnet.base.org",
      forking: {
        url: process.env.BASE_RPC_URL || "https://mainnet.base.org",
      },
      accounts: pkAccounts(),
      chainId: 8453,
    },

    // Ethereum Mainnet Fork (dry-run)
    "mainnet-fork": {
      url: process.env.MAINNET_RPC_URL || "",
      forking: {
        url: process.env.MAINNET_RPC_URL || "",
      },
      accounts: pkAccounts(),
      chainId: 1,
    },

    // Ethereum Mainnet
    "mainnet": {
      url: process.env.MAINNET_RPC_URL || "",
      accounts: pkAccounts(),
      chainId: 1,
      gas: 6000000,
      gasPrice: "auto",
      confirmations: 3,
      timeoutBlocks: 200,
      skipDryRun: false
    },

    // Polygon Mainnet
    "polygon-mainnet": {
      url: process.env.POLYGON_MAINNET_RPC_URL || "https://polygon-rpc.com",
      accounts: pkAccounts(),
      chainId: 137,
      gas: 6000000,
      gasPrice: 30000000000, // 30 gwei
      confirmations: 2,
      timeoutBlocks: 200,
      skipDryRun: true
    }
  },
  
  // Contract verification
  etherscan: {
    // Etherscan V2 unified multichain API — one key covers all networks including Base
    apiKey: process.env.ETHERSCAN_API_KEY || "",
    customChains: [
      {
        network: "base",
        chainId: 8453,
        urls: {
          apiURL: "https://api.etherscan.io/v2/api",
          browserURL: "https://basescan.org"
        }
      },
      {
        network: "base-sepolia",
        chainId: 84532,
        urls: {
          apiURL: "https://api.etherscan.io/v2/api",
          browserURL: "https://sepolia.basescan.org"
        }
      }
    ]
  },
  
  // Gas reporting
  gasReporter: {
    enabled: process.env.REPORT_GAS !== undefined,
    currency: "USD",
    gasPrice: 30, // gwei
    coinmarketcap: process.env.COINMARKETCAP_API_KEY
  },
  
  // Solidity coverage configuration
  solidity_coverage: {
    skipFiles: ['Migrations.sol']
  },
  
  // Path configuration
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  },
  
  // Mocha test configuration
  mocha: {
    timeout: 60000
  }
};