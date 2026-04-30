/*
 * Phase 1.3 — FTNSTokenSimple AccessControl role handoff.
 *
 * FTNSTokenSimple is AccessControl-based, NOT Ownable, so transfer-
 * ownership.js explicitly skips it (per its docstring). This script
 * does the parallel role-handoff ceremony:
 *
 *   1. Grant DEFAULT_ADMIN_ROLE to FOUNDATION_MULTISIG (multi-sig can
 *      then grant/revoke any other role).
 *   2. Renounce DEFAULT_ADMIN_ROLE on deployer.
 *   3. Renounce MINTER_ROLE on deployer.
 *   4. Renounce PAUSER_ROLE on deployer.
 *   5. Renounce BURNER_ROLE on deployer.
 *
 * After this runs, the deployer hot key has zero authority on
 * FTNSToken. The multi-sig must then (in a separate, multi-sig-signed
 * tx) grant MINTER_ROLE to EmissionController for emission to flow —
 * that is intentionally NOT in this script because it requires the
 * multi-sig signature, not the deployer's.
 *
 * Two-phase parity with transfer-ownership.js:
 *   - Phase 1: hot deployer wires + holds keys (Phase 1.3 init grants
 *     deployer all four roles).
 *   - Phase 2: this script hands the lock to the multi-sig at end of
 *     ceremony, BEFORE any non-rehearsal usage.
 *
 * Required env vars:
 *   FOUNDATION_MULTISIG  - the 2-of-3 Safe that takes DEFAULT_ADMIN_ROLE
 *   PHASE1_FTNS_MANIFEST - path to phase1-ftns-<network>-*.json
 *                          (alternative: FTNS_TOKEN_ADDRESS direct)
 *
 * Optional env vars:
 *   FTNS_TOKEN_ADDRESS   - skip manifest read; use this address directly
 *
 * Usage:
 *   FOUNDATION_MULTISIG=0x... \
 *     PHASE1_FTNS_MANIFEST=contracts/deployments/phase1-ftns-base-1234.json \
 *     npx hardhat run scripts/transfer-ftns-roles.js --network base
 *
 * Idempotent: re-running after a successful handoff is a no-op (every
 * renounce step early-returns if the deployer no longer holds the role,
 * and the grant step early-returns if the multi-sig already has admin).
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

const ROLE_HASH = {
  DEFAULT_ADMIN_ROLE:
    "0x0000000000000000000000000000000000000000000000000000000000000000",
  MINTER_ROLE: null,  // computed below via hre.ethers.id()
  PAUSER_ROLE: null,
  BURNER_ROLE: null,
};

function loadManifest() {
  const p = process.env.PHASE1_FTNS_MANIFEST;
  if (!p) return null;
  if (!fs.existsSync(p)) {
    throw new Error(`PHASE1_FTNS_MANIFEST points to ${p} which does not exist`);
  }
  const manifest = JSON.parse(fs.readFileSync(p, "utf8"));
  console.log(`Loaded Phase 1 FTNS manifest: ${p}`);
  return manifest;
}

async function main() {
  const network = hre.network.name;
  const isMainnet = network === "base" || network === "mainnet";

  // ── Resolve role hashes ──────────────────────────────────────────────
  ROLE_HASH.MINTER_ROLE = hre.ethers.id("MINTER_ROLE");
  ROLE_HASH.PAUSER_ROLE = hre.ethers.id("PAUSER_ROLE");
  ROLE_HASH.BURNER_ROLE = hre.ethers.id("BURNER_ROLE");

  const multisig = process.env.FOUNDATION_MULTISIG;
  if (!multisig) throw new Error("FOUNDATION_MULTISIG env var required");
  const multisigChecksum = hre.ethers.getAddress(multisig);

  console.log(`\n=== Phase 1.3 FTNS role handoff to multi-sig ===`);
  console.log(`Network:           ${network}`);
  console.log(`Foundation multi-sig: ${multisigChecksum}`);

  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log(`Deployer:          ${deployer.address}`);
  console.log(`Deployer balance:  ${hre.ethers.formatEther(balance)} ETH`);

  if (
    isMainnet
    && multisigChecksum.toLowerCase() === deployer.address.toLowerCase()
  ) {
    throw new Error(
      "FOUNDATION_MULTISIG must not equal deployer on mainnet."
    );
  }
  if (multisigChecksum === hre.ethers.ZeroAddress) {
    throw new Error("FOUNDATION_MULTISIG is the zero address");
  }
  if (isMainnet) {
    const code = await hre.ethers.provider.getCode(multisigChecksum);
    if (code === "0x" || code === "0x0") {
      throw new Error(
        `FOUNDATION_MULTISIG ${multisigChecksum} is an EOA on mainnet — ` +
        `expected a deployed multi-sig contract (Safe).`
      );
    }
  }

  // ── Resolve token address ─────────────────────────────────────────────
  let tokenAddress = process.env.FTNS_TOKEN_ADDRESS;
  if (!tokenAddress) {
    const manifest = loadManifest();
    if (!manifest || !manifest.contracts || !manifest.contracts.FTNSTokenSimple) {
      throw new Error(
        "FTNS_TOKEN_ADDRESS or PHASE1_FTNS_MANIFEST required"
      );
    }
    if (manifest.network !== network) {
      throw new Error(
        `manifest network=${manifest.network} != --network=${network}`
      );
    }
    tokenAddress = manifest.contracts.FTNSTokenSimple;
  }
  tokenAddress = hre.ethers.getAddress(tokenAddress);
  console.log(`FTNS token:        ${tokenAddress}`);

  const code = await hre.ethers.provider.getCode(tokenAddress);
  if (code === "0x" || code === "0x0") {
    throw new Error(`FTNS token at ${tokenAddress} has no contract code`);
  }

  const token = new hre.ethers.Contract(
    tokenAddress,
    [
      "function hasRole(bytes32 role, address account) view returns (bool)",
      "function grantRole(bytes32 role, address account)",
      "function renounceRole(bytes32 role, address callerConfirmation)",
    ],
    deployer,
  );

  const txs = [];
  const skipped = [];

  // ── Step 1: grant DEFAULT_ADMIN_ROLE to multi-sig ───────────────────
  console.log(`\nStep 1 — Grant DEFAULT_ADMIN_ROLE to multi-sig`);
  const adminAlready = await token.hasRole(
    ROLE_HASH.DEFAULT_ADMIN_ROLE,
    multisigChecksum,
  );
  if (adminAlready) {
    console.log(`   ⏭  multi-sig already holds DEFAULT_ADMIN_ROLE; skipping grant`);
    skipped.push({ role: "DEFAULT_ADMIN_ROLE", action: "grant", account: multisigChecksum });
  } else {
    const deployerHasAdmin = await token.hasRole(
      ROLE_HASH.DEFAULT_ADMIN_ROLE,
      deployer.address,
    );
    if (!deployerHasAdmin) {
      throw new Error(
        `deployer ${deployer.address} does not hold DEFAULT_ADMIN_ROLE on ` +
        `${tokenAddress} — cannot grant to multi-sig. Either ceremony has ` +
        `already run (multi-sig must use its own signers) or this is the ` +
        `wrong token.`
      );
    }
    const tx = await token.grantRole(ROLE_HASH.DEFAULT_ADMIN_ROLE, multisigChecksum);
    const rcpt = await tx.wait();
    console.log(`     ✅ tx ${tx.hash} (block ${rcpt.blockNumber})`);
    txs.push({ role: "DEFAULT_ADMIN_ROLE", action: "grant", account: multisigChecksum, txHash: tx.hash });
    const verify = await token.hasRole(ROLE_HASH.DEFAULT_ADMIN_ROLE, multisigChecksum);
    if (!verify) throw new Error(`grant did not stick: multi-sig still lacks DEFAULT_ADMIN_ROLE`);
  }

  // ── Steps 2-5: renounce all four roles on deployer ──────────────────
  for (const roleName of ["DEFAULT_ADMIN_ROLE", "MINTER_ROLE", "PAUSER_ROLE", "BURNER_ROLE"]) {
    console.log(`\nStep — Renounce ${roleName} on deployer`);
    const roleHash = ROLE_HASH[roleName];
    const has = await token.hasRole(roleHash, deployer.address);
    if (!has) {
      console.log(`   ⏭  deployer no longer has ${roleName}; skipping`);
      skipped.push({ role: roleName, action: "renounce", account: deployer.address });
      continue;
    }
    // Belt-and-braces: don't strand the contract by renouncing admin if
    // the multi-sig hasn't actually received it. (Step 1 invariants
    // already guarantee this for DEFAULT_ADMIN_ROLE; the other three
    // roles are not tied to the same risk but the check is cheap.)
    if (roleName === "DEFAULT_ADMIN_ROLE") {
      const multisigHasAdmin = await token.hasRole(roleHash, multisigChecksum);
      if (!multisigHasAdmin) {
        throw new Error(
          `refusing to renounce DEFAULT_ADMIN_ROLE on deployer because ` +
          `multi-sig ${multisigChecksum} does not yet hold it — would ` +
          `permanently strand the contract. Re-run Step 1 first.`
        );
      }
    }
    const tx = await token.renounceRole(roleHash, deployer.address);
    const rcpt = await tx.wait();
    console.log(`     ✅ tx ${tx.hash} (block ${rcpt.blockNumber})`);
    txs.push({ role: roleName, action: "renounce", account: deployer.address, txHash: tx.hash });
    const verify = await token.hasRole(roleHash, deployer.address);
    if (verify) throw new Error(`renounce did not stick: deployer still holds ${roleName}`);
  }

  // ── Final invariant ──────────────────────────────────────────────────
  console.log(`\nFinal invariant — deployer holds zero roles on FTNS:`);
  for (const [label, roleHash] of Object.entries(ROLE_HASH)) {
    const has = await token.hasRole(roleHash, deployer.address);
    if (has) {
      throw new Error(`deployer still holds ${label} after ceremony`);
    }
    console.log(`  deployer.${label}: false ✓`);
  }
  const multisigAdmin = await token.hasRole(
    ROLE_HASH.DEFAULT_ADMIN_ROLE,
    multisigChecksum,
  );
  if (!multisigAdmin) {
    throw new Error(`multi-sig does not hold DEFAULT_ADMIN_ROLE after ceremony`);
  }
  console.log(`  multisig.DEFAULT_ADMIN_ROLE: true ✓`);

  // ── Manifest ──────────────────────────────────────────────────────────
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;
  const manifest = {
    bundle: "ftns-role-transfer",
    network,
    chainId: chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    foundationMultisig: multisigChecksum,
    ftnsToken: tokenAddress,
    transactions: txs,
    skipped,
  };

  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(
    outDir,
    `ftns-role-transfer-${network}-${Date.now()}.json`,
  );
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);
  console.log(
    `\n✅ FTNS roles handed off: ${txs.length} txs; skipped ` +
    `(idempotent): ${skipped.length}.`
  );
  console.log(
    `\nReminder: multi-sig must now grant MINTER_ROLE to EmissionController ` +
    `via 2-of-3 governance tx for emission to flow.`
  );
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
