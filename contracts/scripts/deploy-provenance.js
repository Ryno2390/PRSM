/*
 * Deploys ProvenanceRegistry + RoyaltyDistributor.
 *
 * Required env vars:
 *   FTNS_TOKEN_ADDRESS    - existing FTNS ERC20 on the target network
 *   NETWORK_TREASURY      - address that receives the 2% network fee
 *
 * Optional env vars:
 *   ROYALTY_DISTRIBUTOR_OWNER - initial owner of RoyaltyDistributor.
 *                                Defaults to the deployer EOA. On mainnet,
 *                                set this to the Foundation Safe to skip
 *                                the post-deploy transferOwnership step
 *                                (otherwise hand-off via Ownable2Step is
 *                                required before the Safe can call
 *                                recoverStranded — A-08 fix).
 *
 * Usage:
 *   npx hardhat run scripts/deploy-provenance.js --network base-sepolia
 *   npx hardhat run scripts/deploy-provenance.js --network base
 *
 * Local smoke test (uses dummy addresses, in-process Hardhat network):
 *   FTNS_TOKEN_ADDRESS=0x0000000000000000000000000000000000000001 \
 *   NETWORK_TREASURY=0x0000000000000000000000000000000000000002 \
 *   npx hardhat run scripts/deploy-provenance.js --network hardhat
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

// Canonical Base mainnet FTNS address. Pinned here so a typo'd
// FTNS_TOKEN_ADDRESS gets caught BEFORE we deploy RoyaltyDistributor
// with a permanent (immutable constructor arg) link to a wrong token.
// Verified 2026-04-30 via direct Base RPC: symbol=FTNS,
// totalSupply=100M, name="PRSM Fungible Tokens for Node Support".
const CANONICAL_FTNS_BASE_MAINNET = "0x5276a3756C85f2E9e46f6D34386167a209aa16e5";

// L2 audit MEDIUM B-TREASURY-1 fix: pin the canonical Foundation Safe
// address. Pre-fix, the script verified treasury was *a* contract on
// mainnet but accepted ANY contract — a typo to a different deployed
// contract (random ERC-20, random Safe) would permanently route the
// 2% network fee elsewhere since networkTreasury is immutable on
// RoyaltyDistributor. Belt-and-suspenders mirror of the FTNS pin
// above. Foundation Safe deployed 2026-05-04 (Phase 1.3 Task 8).
const CANONICAL_FOUNDATION_SAFE_BASE_MAINNET = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791";

const BASE_MAINNET_CHAIN_ID = 8453n;

async function main() {
  const ftnsAddress = process.env.FTNS_TOKEN_ADDRESS;
  const treasury = process.env.NETWORK_TREASURY;
  if (!ftnsAddress) throw new Error("FTNS_TOKEN_ADDRESS env var required");
  if (!treasury) throw new Error("NETWORK_TREASURY env var required");

  const network = hre.network.name;
  const isMainnet = network === "base" || network === "mainnet";
  console.log(`\n=== Deploying provenance contracts to ${network} ===`);
  console.log(`FTNS token:       ${ftnsAddress}`);
  console.log(`Network treasury: ${treasury}`);

  // 2026-05-04 ceremony lesson: the operator hit two issues (PRIVATE_KEY
  // missing 0x prefix; BASE_RPC_URL pointing at ETH mainnet not Base) that
  // pre-task8-checklist.sh would have caught in 2 seconds. Print a banner
  // here so future operators are reminded.
  if (isMainnet) {
    console.log("");
    console.log("⚠️  Did you run scripts/pre-task8-checklist.sh first?");
    console.log("   It catches PRIVATE_KEY format / BASE_RPC_URL chainId /");
    console.log("   FTNS bytecode / treasury-bytecode issues BEFORE gas burns.");
    console.log("   If not run yet, abort with Ctrl+C and run it now.");
    console.log("");
  }

  // ── Phase 1.1 Task 9: preflight ─────────────────────────────────────
  // Constructor args are immutable on RoyaltyDistributor. A typo here is
  // permanent — re-deploy is the only fix. Catch mistakes BEFORE we burn
  // gas on a bad deploy.
  console.log("\nPreflight…");

  // 1. Checksum validation. ethers v6 throws on bad checksum.
  let ftnsChecksum, treasuryChecksum;
  try {
    ftnsChecksum = hre.ethers.getAddress(ftnsAddress);
    treasuryChecksum = hre.ethers.getAddress(treasury);
  } catch (e) {
    throw new Error(`address checksum failure: ${e.message}`);
  }
  if (ftnsChecksum === hre.ethers.ZeroAddress) throw new Error("FTNS_TOKEN_ADDRESS is zero");
  if (treasuryChecksum === hre.ethers.ZeroAddress) throw new Error("NETWORK_TREASURY is zero");
  console.log(`  FTNS checksum:     ${ftnsChecksum}`);
  console.log(`  Treasury checksum: ${treasuryChecksum}`);

  // 2. Code at FTNS address.
  const code = await hre.ethers.provider.getCode(ftnsChecksum);
  if (code === "0x" || code === "0x0") {
    throw new Error(`no contract at FTNS_TOKEN_ADDRESS ${ftnsChecksum} on network ${network}`);
  }
  console.log(`  FTNS bytecode:     ${(code.length / 2 - 1)} bytes`);

  // 3. Confirm it actually quacks like FTNS. We accept either "FTNS"
  //    (production) or "MFTNS" (the test mock used by the integration
  //    suite) so the script works against both.
  const ftnsAbi = [
    "function symbol() view returns (string)",
    "function name() view returns (string)",
  ];
  const ftns = new hre.ethers.Contract(ftnsChecksum, ftnsAbi, hre.ethers.provider);
  let symbol;
  try {
    symbol = await ftns.symbol();
  } catch (e) {
    throw new Error(`FTNS symbol() call failed: ${e.message}`);
  }
  if (symbol !== "FTNS" && symbol !== "MFTNS") {
    throw new Error(`FTNS_TOKEN_ADDRESS symbol is "${symbol}", expected "FTNS" or "MFTNS"`);
  }
  console.log(`  FTNS symbol:       ${symbol}`);

  // 4. Log chain id, and on mainnet HARD-FAIL if it doesn't match Base.
  //    --network flag → hardhat config → connected RPC. If the RPC
  //    URL points at the wrong chain (e.g. operator pasted a polygon
  //    or sepolia URL into BASE_RPC_URL), the chainId mismatch catches
  //    it BEFORE any tx is submitted.
  const chainIdActual = (await hre.ethers.provider.getNetwork()).chainId;
  console.log(`  Chain id:          ${chainIdActual}`);
  if (network === "base" && chainIdActual !== BASE_MAINNET_CHAIN_ID) {
    throw new Error(
      `--network=base but RPC reports chainId=${chainIdActual}; ` +
      `expected ${BASE_MAINNET_CHAIN_ID}. Check BASE_RPC_URL — likely ` +
      `points at the wrong network. ABORT before deploy.`
    );
  }

  // 5. On mainnet ONLY: refuse FTNS_TOKEN_ADDRESS that doesn't match
  //    the canonical pinned production address. RoyaltyDistributor's
  //    constructor wires this in immutably; a typo here is permanent.
  //    Operator can override via FORCE_NONCANONICAL_FTNS=1 if/when a
  //    new production token is intentionally deployed.
  if (isMainnet && ftnsChecksum.toLowerCase() !== CANONICAL_FTNS_BASE_MAINNET.toLowerCase()) {
    if (process.env.FORCE_NONCANONICAL_FTNS !== "1") {
      throw new Error(
        `FTNS_TOKEN_ADDRESS=${ftnsChecksum} does not match the canonical ` +
        `Base mainnet FTNS at ${CANONICAL_FTNS_BASE_MAINNET}. If this is ` +
        `intentional (new production token), set FORCE_NONCANONICAL_FTNS=1 ` +
        `to override. Otherwise, FIX THE TYPO before proceeding.`
      );
    }
    console.log(`  ⚠️  using non-canonical FTNS (FORCE_NONCANONICAL_FTNS=1 set)`);
  }

  const [deployer] = await hre.ethers.getSigners();
  console.log(`\nDeployer:         ${deployer.address}`);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log(`Deployer balance: ${hre.ethers.formatEther(balance)} ETH`);
  if (balance === 0n) throw new Error("Deployer has zero balance");

  // 6. Treasury must not equal deployer. networkTreasury is immutable —
  //    the Sepolia bake-in convenience of using deployer-as-treasury is
  //    acceptable on testnet but would permanently route the 2% royalty
  //    fee to the deployer EOA on mainnet with no upgrade path. Belt and
  //    suspenders: operator selects the right value AND the script
  //    refuses if they match.
  if (treasuryChecksum.toLowerCase() === deployer.address.toLowerCase()) {
    throw new Error(
      `NETWORK_TREASURY (${treasuryChecksum}) must not equal deployer (${deployer.address}). ` +
      `Use a dedicated treasury address — a multi-sig or foundation-controlled address, never the deployer EOA.`
    );
  }

  // 7. On mainnet ONLY: NETWORK_TREASURY must be a contract (Safe),
  //    not an EOA. An EOA treasury defeats the multi-sig safety property
  //    — losing a single private key would compromise the entire 2%
  //    royalty stream forever, since networkTreasury is immutable.
  if (isMainnet) {
    const treasuryCode = await hre.ethers.provider.getCode(treasuryChecksum);
    if (treasuryCode === "0x" || treasuryCode === "0x0") {
      throw new Error(
        `NETWORK_TREASURY ${treasuryChecksum} is an EOA on ${network}. ` +
        `Expected a deployed multi-sig contract (Safe). Hot wallets cannot ` +
        `replace 2-of-3 multi-sig safety for the immutable treasury role.`
      );
    }
    console.log(`  Treasury bytecode: ${(treasuryCode.length / 2 - 1)} bytes (contract ✓)`);

    // 7b. L2 audit MEDIUM B-TREASURY-1 fix: refuse NETWORK_TREASURY that
    //     doesn't match the canonical pinned Foundation Safe. The
    //     bytecode-only check above accepts ANY contract (random ERC-20,
    //     a different Safe, etc.) — without this pin, a typo to such an
    //     address would permanently route the 2% network fee elsewhere
    //     since networkTreasury is immutable on RoyaltyDistributor.
    //     Operator can override via FORCE_NONCANONICAL_TREASURY=1 if/when
    //     the Foundation Safe is intentionally rotated (re-deploy of
    //     RoyaltyDistributor required, since networkTreasury is
    //     immutable).
    if (treasuryChecksum.toLowerCase() !== CANONICAL_FOUNDATION_SAFE_BASE_MAINNET.toLowerCase()) {
      if (process.env.FORCE_NONCANONICAL_TREASURY !== "1") {
        throw new Error(
          `NETWORK_TREASURY=${treasuryChecksum} does not match the canonical ` +
          `Base mainnet Foundation Safe at ${CANONICAL_FOUNDATION_SAFE_BASE_MAINNET}. ` +
          `If this is intentional (rotated Safe), set FORCE_NONCANONICAL_TREASURY=1 ` +
          `to override. Otherwise, FIX THE TYPO before proceeding.`
        );
      }
      console.log(`  ⚠️  using non-canonical treasury (FORCE_NONCANONICAL_TREASURY=1 set)`);
    } else {
      console.log(`  Treasury matches canonical Foundation Safe ✓`);
    }
  }

  // 1. Registry
  console.log("\nDeploying ProvenanceRegistry…");
  const Registry = await hre.ethers.getContractFactory("ProvenanceRegistry");
  const registry = await Registry.deploy();
  await registry.waitForDeployment();
  const registryAddress = await registry.getAddress();
  console.log(`  ProvenanceRegistry: ${registryAddress}`);

  // 2. Distributor (using checksummed addresses from preflight)
  // L4 self-audit A-08: 4-arg constructor — initial owner defaults to
  // deployer; operator can transferOwnership to Foundation Safe via
  // Ownable2Step after deploy, OR pre-set ROYALTY_DISTRIBUTOR_OWNER to
  // the Safe to skip the hand-off (Safe must already be deployed).
  const ownerOverride = process.env.ROYALTY_DISTRIBUTOR_OWNER;
  const initialOwner = ownerOverride
    ? hre.ethers.getAddress(ownerOverride)
    : deployer.address;
  console.log(`  Initial owner:     ${initialOwner}${ownerOverride ? " (override)" : " (deployer)"}`);
  console.log("\nDeploying RoyaltyDistributor…");
  const Distributor = await hre.ethers.getContractFactory("RoyaltyDistributor");
  const distributor = await Distributor.deploy(
    ftnsChecksum,
    registryAddress,
    treasuryChecksum,
    initialOwner,
  );
  await distributor.waitForDeployment();
  const distributorAddress = await distributor.getAddress();
  console.log(`  RoyaltyDistributor: ${distributorAddress}`);

  // 3. Save deployment manifest
  const manifest = {
    network,
    chainId: chainIdActual.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      ProvenanceRegistry: registryAddress,
      RoyaltyDistributor: distributorAddress,
      FTNSToken: ftnsChecksum,
      NetworkTreasury: treasuryChecksum,
    },
  };

  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `provenance-${network}-${Date.now()}.json`);
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  // 4. Optional: auto-verify on Basescan (only on real Base networks).
  const verifyEnabled = process.env.AUTO_VERIFY === "1";
  const isBase = network === "base" || network === "base-sepolia";
  if (verifyEnabled && isBase) {
    console.log("\nVerifying on Basescan…");
    const verifyTargets = [
      { name: "ProvenanceRegistry", address: registryAddress, args: [] },
      {
        name: "RoyaltyDistributor",
        address: distributorAddress,
        args: [ftnsChecksum, registryAddress, treasuryChecksum, initialOwner],
      },
    ];
    for (const t of verifyTargets) {
      try {
        await hre.run("verify:verify", {
          address: t.address,
          constructorArguments: t.args,
        });
        console.log(`  ${t.name} verified at ${t.address}`);
      } catch (e) {
        console.warn(`  ${t.name} verification failed (non-fatal): ${e.message}`);
      }
    }
  } else if (isBase) {
    console.log("\nSkipping verification (set AUTO_VERIFY=1 to enable).");
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
