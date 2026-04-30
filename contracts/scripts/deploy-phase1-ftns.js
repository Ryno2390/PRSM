/*
 * Phase 1.3 — FTNSTokenSimple UUPS proxy deployment.
 *
 * Deploys the production FTNS token via OpenZeppelin upgrades plugin.
 * The token grants four roles to the deployer at init time
 * (DEFAULT_ADMIN_ROLE, MINTER_ROLE, PAUSER_ROLE, BURNER_ROLE) — those
 * must subsequently be transferred to the Foundation 2-of-3 multi-sig
 * via transfer-ftns-roles.js BEFORE the rest of the audit-bundle +
 * Phase 8 + Phase 7-storage ceremony.
 *
 * Two-phase deploy model parallels the Ownable contracts:
 *   1. Hot deployer deploys + holds admin keys.
 *   2. Hot deployer hands off to multi-sig at end of ceremony.
 * MINTER_ROLE → EmissionController is a separate multi-sig-signed tx
 * AFTER ownership has transferred (per deploy-phase8-emission.js
 * docstring); not handled here.
 *
 * Honest scope: this script intentionally deploys ONLY the token. The
 * old contracts/scripts/deploy.js is stale (ethers v5; bundles
 * Timelock/Governance/Marketplace which are out of scope for the
 * current mainnet ceremony). Do not use deploy.js.
 *
 * Required env vars:
 *   PRIVATE_KEY              - deployer key (funded on target network)
 *   BASE_RPC_URL             - mainnet RPC (only needed for NETWORK=base)
 *   BASE_SEPOLIA_RPC_URL     - testnet RPC (only for NETWORK=base-sepolia)
 *
 * Optional env vars:
 *   TREASURY_ADDRESS         - receives INITIAL_SUPPLY (100M FTNS).
 *                              Defaults to deployer for hardhat-local;
 *                              REQUIRED on testnet/mainnet (silent
 *                              fallback to deployer would route 100M
 *                              FTNS to a hot key on mainnet).
 *
 * Output:
 *   contracts/deployments/phase1-ftns-<network>-<ts>.json
 *   STDOUT line "FTNSTokenSimple: 0x..." for orchestrator parsing.
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;
  const isMainnet = network === "base" || network === "mainnet";
  const isTestnet =
    network === "base-sepolia"
    || network === "sepolia"
    || network === "polygon-mumbai";

  console.log(`\n=== Phase 1.3 — FTNSTokenSimple UUPS deployment ===`);
  console.log(`Network: ${network}`);

  const [deployer] = await hre.ethers.getSigners();
  console.log(`Deployer: ${deployer.address}`);
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log(`Deployer balance: ${hre.ethers.formatEther(balance)} ETH`);
  if (balance === 0n) throw new Error("Deployer has zero balance");

  // ── Treasury validation ──────────────────────────────────────────────
  // Must be explicit on testnet/mainnet — silent fallback to deployer on
  // mainnet would mint 100M FTNS to a hot key.
  let treasury = process.env.TREASURY_ADDRESS;
  if (!treasury) {
    if (isMainnet || isTestnet) {
      throw new Error(
        `TREASURY_ADDRESS required on ${network}. The initial 100M FTNS ` +
        `supply is minted to this address; defaulting to deployer would ` +
        `route the entire pre-mine to a hot key. Set explicitly to the ` +
        `Foundation multi-sig.`
      );
    }
    treasury = deployer.address;
    console.log(`   TREASURY_ADDRESS unset; defaulting to deployer for ${network}.`);
  }
  treasury = hre.ethers.getAddress(treasury);
  console.log(`Treasury (initial-supply recipient): ${treasury}`);

  // ── Deploy ────────────────────────────────────────────────────────────
  console.log(`\nDeploying FTNSTokenSimple as UUPS proxy…`);
  const FTNS = await hre.ethers.getContractFactory("FTNSTokenSimple");
  const proxy = await hre.upgrades.deployProxy(
    FTNS,
    [deployer.address, treasury],
    { initializer: "initialize", kind: "uups" },
  );
  await proxy.waitForDeployment();
  const proxyAddress = await proxy.getAddress();
  const implAddress = await hre.upgrades.erc1967.getImplementationAddress(
    proxyAddress,
  );

  console.log(`  FTNSTokenSimple: ${proxyAddress}`);
  console.log(`  implementation:  ${implAddress}`);

  // ── Post-deploy invariants ───────────────────────────────────────────
  console.log(`\nPost-deploy invariant checks…`);
  const name = await proxy.name();
  const symbol = await proxy.symbol();
  const totalSupply = await proxy.totalSupply();
  console.log(`  name:         ${name}`);
  console.log(`  symbol:       ${symbol}`);
  console.log(`  totalSupply:  ${hre.ethers.formatUnits(totalSupply, 18)} FTNS`);
  if (symbol !== "FTNS") {
    throw new Error(`unexpected symbol ${symbol}, expected FTNS`);
  }

  const expectedInitial = hre.ethers.parseUnits("100000000", 18);
  if (totalSupply !== expectedInitial) {
    throw new Error(
      `initial supply ${totalSupply} != expected ${expectedInitial} (100M)`,
    );
  }
  const treasuryBal = await proxy.balanceOf(treasury);
  if (treasuryBal !== expectedInitial) {
    throw new Error(
      `treasury balance ${treasuryBal} != initial supply ${expectedInitial}`,
    );
  }
  console.log(`  treasury balance: ${hre.ethers.formatUnits(treasuryBal, 18)} FTNS ✓`);

  // Sanity: deployer holds the four post-init roles.
  const ROLES = {
    DEFAULT_ADMIN_ROLE:
      "0x0000000000000000000000000000000000000000000000000000000000000000",
    MINTER_ROLE: hre.ethers.id("MINTER_ROLE"),
    PAUSER_ROLE: hre.ethers.id("PAUSER_ROLE"),
    BURNER_ROLE: hre.ethers.id("BURNER_ROLE"),
  };
  for (const [label, roleHash] of Object.entries(ROLES)) {
    const has = await proxy.hasRole(roleHash, deployer.address);
    if (!has) {
      throw new Error(`deployer is missing ${label} post-init`);
    }
    console.log(`  deployer has ${label} ✓`);
  }

  // ── Manifest ──────────────────────────────────────────────────────────
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;
  const manifest = {
    bundle: "phase1-ftns",
    network,
    chainId: chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      FTNSTokenSimple: proxyAddress,
    },
    proxy: {
      kind: "uups",
      implementation: implAddress,
    },
    initialSupply: {
      to: treasury,
      amount: totalSupply.toString(),
    },
    deployerRoles: Object.keys(ROLES),
  };

  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `phase1-ftns-${network}-${Date.now()}.json`);
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  console.log(`\n✅ Phase 1.3 FTNS deployment complete.`);
  console.log(
    `\nNext steps:\n` +
    `   export FTNS_TOKEN_ADDRESS="${proxyAddress}"\n` +
    `   # Run rest of audit-bundle / Phase 8 / Phase 7-storage,\n` +
    `   # THEN run transfer-ftns-roles.js to hand off all 4 roles to\n` +
    `   # the Foundation multi-sig.`
  );
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
