/*
 * PRSM-PROV-1 Item 7 (T7.7) — ProvenanceRegistryV2 deploy script.
 *
 * Deploys ProvenanceRegistryV2 to a target network. Sepolia is the
 * intended v1 target — it lets us exercise the dispute round-trip
 * against a real chain without committing to the mainnet upgrade,
 * which is gated behind L4 audit firm review per PRSM-PROV-1 plan
 * §4.5 + PRSM-POL-1 §5.
 *
 * The contract has NO constructor arguments. Storage shape is
 * separate from v1 (deployed at Base mainnet 0xdF47...9915) — new
 * uploads register here; legacy uploads remain in v1. Off-chain
 * dispute resolution (RoyaltyDistributor + V2Client) reads from
 * both, so this deploy doesn't change v1 behavior in any way.
 *
 * Optional env var:
 *   AUTO_VERIFY  - "1" to auto-verify on Etherscan / Basescan after
 *                  deploy. Requires ETHERSCAN_API_KEY (sepolia) or
 *                  BASESCAN_API_KEY (base / base-sepolia).
 *
 * Usage:
 *   # Dry-run on local Hardhat node — validates the script + smoke-
 *   # tests the post-deploy invariants WITHOUT spending gas.
 *   npx hardhat run scripts/deploy-provenance-registry-v2.js \
 *       --network hardhat
 *
 *   # Real Sepolia deploy:
 *   PRIVATE_KEY=0x... \
 *   SEPOLIA_RPC_URL=https://... \
 *   AUTO_VERIFY=1 ETHERSCAN_API_KEY=... \
 *       npx hardhat run scripts/deploy-provenance-registry-v2.js \
 *           --network sepolia
 *
 *   # Base mainnet deploy (authorized 2026-05-06 by PRSM-CR-2026-05-06-2):
 *   PRIVATE_KEY=0x... \
 *   BASE_RPC_URL=https://... \
 *   AUTO_VERIFY=1 ETHERSCAN_API_KEY=... \
 *       npx hardhat run scripts/deploy-provenance-registry-v2.js \
 *           --network base
 *
 * After a successful deploy, copy the address printed in the
 * "DEPLOY COMPLETE" block into prsm/deployments/contract_addresses.json
 * under <network>.provenance_registry_v2 and commit. The Python V2 client
 * (prsm/economy/web3/provenance_registry_v2.py) reads the address from
 * the env var PRSM_PROVENANCE_REGISTRY_V2_ADDRESS at construction
 * time; the deployments file is the source-of-truth that operators
 * copy into their env.
 *
 * Authorization for mainnet deploy: PRSM-CR-2026-05-06-2 ratified the
 * V2 mainnet deploy on existing evidence (Hardhat 15-test suite +
 * Python 33-unit-test suite + Sepolia live exercise on
 * 0xe75F0c24a9e63B63456d170d99F03Ab7fC3450A7). The L4 audit firm gate
 * was discharged by council resolution; see docs/governance/.
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;

  // Mainnet authorization: PRSM-CR-2026-05-06-2 ratified the V2
  // mainnet deploy on existing evidence (Hardhat + Python + Sepolia
  // live exercise). The previous unconditional block on
  // "base" / "mainnet" is removed per that resolution.
  //
  // Ethereum mainnet (chain 1) was never an intended target for V2 —
  // PRSM is Base-native — so it is still fail-fast'd here. Operators
  // who actually want to deploy V2 to Ethereum mainnet must remove
  // this guard in a separate, council-ratified PR.
  if (network === "mainnet") {
    throw new Error(
      `Direct deploy to ${network} (Ethereum mainnet, chain 1) is ` +
      `BLOCKED. PRSM is Base-native; V2 mainnet authorization in ` +
      `PRSM-CR-2026-05-06-2 covers Base mainnet ("base", chain 8453) ` +
      `only. Either use --network base for the intended target, or ` +
      `obtain a separate council resolution for Ethereum mainnet ` +
      `before removing this guard.`
    );
  }

  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  const chainId = (await hre.ethers.provider.getNetwork()).chainId;

  console.log(`\n=== Deploying ProvenanceRegistryV2 to ${network} ===`);
  console.log(`Deployer:           ${deployer.address}`);
  console.log(`Deployer balance:   ${hre.ethers.formatEther(balance)} ETH`);
  console.log(`Chain id:           ${chainId}`);

  if (balance === 0n && network !== "hardhat") {
    throw new Error("Deployer has zero balance");
  }

  // ── Deploy ─────────────────────────────────────────────────────────
  console.log(`\n[1/1] Deploying ProvenanceRegistryV2…`);
  const Registry = await hre.ethers.getContractFactory(
    "ProvenanceRegistryV2"
  );
  const registry = await Registry.deploy();
  await registry.waitForDeployment();
  const registryAddress = await registry.getAddress();
  console.log(`   ProvenanceRegistryV2: ${registryAddress}`);

  // ── Post-deploy invariant checks ───────────────────────────────────
  console.log(`\nPost-deploy invariant checks…`);

  // Public RPCs (Alchemy, Infura) load-balance reads across replicas
  // that may lag the write-side replica by 1-2 seconds even after a
  // confirmed deploy. The contract IS deployed by the time getAddress()
  // returns, but the read replica we hit next may not yet see the
  // bytecode. Poll until either getCode returns non-empty or the
  // budget expires.
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let codeBytes = "0x";
  for (let attempt = 0; attempt < 15; attempt++) {
    codeBytes = await hre.ethers.provider.getCode(registryAddress);
    if (codeBytes !== "0x") {
      if (attempt > 0) {
        console.log(
          `   read replica caught up after ${attempt + 1} attempts ` +
          `(${(attempt + 1) * 1000}ms)`
        );
      }
      break;
    }
    await sleep(1000);
  }
  if (codeBytes === "0x") {
    throw new Error(
      `getCode(${registryAddress}) returned empty after 15s — ` +
      `replica may be unrecoverably lagging. Re-verify with ` +
      `${network} block explorer manually before proceeding.`
    );
  }

  // MAX_ROYALTY_RATE_BPS must equal 10000 - NETWORK_FEE_BPS (200) = 9800.
  // Mismatch here would indicate a constant-drift bug that has to be
  // caught at deploy-time — the off-chain royalty split assumes this.
  const maxBps = await registry.MAX_ROYALTY_RATE_BPS();
  console.log(`   MAX_ROYALTY_RATE_BPS: ${maxBps}`);
  if (maxBps !== 9800n) {
    throw new Error(
      `MAX_ROYALTY_RATE_BPS=${maxBps}; expected 9800 ` +
      `(10000 - 200 NETWORK_FEE_BPS). Constant drift — abort.`
    );
  }

  // Smoke-test: an unregistered contentHash returns the zero Content
  // struct. Confirms the contract is responsive at the deployed
  // address and storage layout matches the ABI we're calling against.
  const dummyHash = "0x" + "00".repeat(32);
  const empty = await registry.contents(dummyHash);
  if (empty.creator !== hre.ethers.ZeroAddress) {
    throw new Error(
      `freshly-deployed contract returned non-zero creator for ` +
      `${dummyHash}: ${empty.creator}`
    );
  }
  console.log(
    `   contents(zero).creator: ${hre.ethers.ZeroAddress} (empty as expected)`
  );

  // verifyEmbeddingCommitment(zero, zero) must return false because
  // the anti-zero-forgery short-circuit is the load-bearing safety
  // property of the V2 dispute path. Catching its absence at deploy
  // time prevents shipping a contract that would let an attacker win
  // disputes against legacy byte-hash-only content by submitting a
  // zero claim.
  const zeroVerify = await registry.verifyEmbeddingCommitment(
    dummyHash, "0x" + "00".repeat(32),
  );
  if (zeroVerify !== false) {
    throw new Error(
      `verifyEmbeddingCommitment(zero, zero) returned true — ` +
      `anti-zero-forgery guard is broken. ABORT.`
    );
  }
  console.log(`   verifyEmbeddingCommitment(zero, zero): false (guard active)`);

  // ── Manifest ───────────────────────────────────────────────────────
  const manifest = {
    bundle: "prsm-prov-1-item-7-provenance-registry-v2",
    network,
    chainId: chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      ProvenanceRegistryV2: registryAddress,
    },
    constructorArgs: {},
    invariants: {
      MAX_ROYALTY_RATE_BPS: 9800,
      anti_zero_forgery_guard: "verifyEmbeddingCommitment(_, zero) === false",
    },
    postDeployNotes: [
      "ProvenanceRegistryV2 has no admin role and no constructor " +
      "args — it is a pure storage contract for content provenance " +
      "+ embedding commitment.",
      "v1 (Base mainnet 0xdF47...9915) is unaffected by this deploy. " +
      "Both registries co-exist; off-chain readers consult both.",
      "Base mainnet deploy of v2 authorized 2026-05-06 by " +
      "PRSM-CR-2026-05-06-2 (council ratification on existing " +
      "evidence). Ethereum mainnet is still blocked unless a separate " +
      "resolution authorizes it.",
      "Copy this address into prsm/deployments/contract_addresses.json " +
      "under <network>.provenance_registry_v2 before merging.",
      "Operators who run V2Client set " +
      "PRSM_PROVENANCE_REGISTRY_V2_ADDRESS to this address.",
    ],
  };
  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(
    outDir,
    `provenance-registry-v2-${network}-${Date.now()}.json`
  );
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);

  // ── Etherscan / Basescan verification ──────────────────────────────
  const verifyEnabled = process.env.AUTO_VERIFY === "1";
  const verifySupported =
    network === "sepolia" ||
    network === "base-sepolia" ||
    network === "base";
  if (verifyEnabled && verifySupported) {
    console.log(`\nVerifying on block explorer…`);
    try {
      await hre.run("verify:verify", {
        address: registryAddress,
        constructorArguments: [],
      });
      console.log(`   ProvenanceRegistryV2 verified`);
    } catch (e) {
      console.warn(
        `   ProvenanceRegistryV2 verify failed (non-fatal): ` +
        `${e.message.split("\n")[0]}`
      );
    }
  } else if (verifyEnabled) {
    console.log(
      `\nVerification skipped: network=${network} not in supported ` +
      `list (sepolia, base-sepolia, base).`
    );
  }

  console.log(`\n${"=".repeat(60)}`);
  console.log(`✅ DEPLOY COMPLETE`);
  console.log(`${"=".repeat(60)}`);
  console.log(`Network:                ${network}`);
  console.log(`ProvenanceRegistryV2:   ${registryAddress}`);
  console.log(``);
  console.log(`Next steps:`);
  console.log(`  1. Copy the address into prsm/deployments/contract_addresses.json:`);
  console.log(`       ${network}.provenance_registry_v2 = "${registryAddress}"`);
  console.log(`  2. Commit the change.`);
  console.log(`  3. Set PRSM_PROVENANCE_REGISTRY_V2_ADDRESS env var to`);
  console.log(`     this address before exercising the V2Client.`);
  console.log(`  4. Run scripts/exercise_provenance_registry_v2_sepolia.py`);
  console.log(`     (T7.7 follow-on) to confirm the dispute round-trip`);
  console.log(`     against the deployed address.`);
  console.log(`${"=".repeat(60)}\n`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
