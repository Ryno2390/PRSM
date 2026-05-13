/*
 * Aerodrome USDC-FTNS pool-seed transaction bundle builder.
 *
 * Companion to:
 *   docs/governance/2026-06-15-aerodrome-pool-seed-ceremony-plan.md
 *   docs/governance/2026-06-15-aerodrome-pool-seed-sepolia-rehearsal.md
 *
 * Emits a Safe{Wallet} Transaction Builder JSON bundle that, when
 * imported into the Foundation Safe{Wallet} UI, queues the 3-tx
 * sequence required to seed the Aerodrome USDC-FTNS pool:
 *
 *   TX 1: FTNS.approve(Router, ftns_amount_wei)
 *   TX 2: USDC.approve(Router, usdc_amount_wei)
 *   TX 3: Router.addLiquidity(FTNS, USDC, false,
 *                              ftns_amount, usdc_amount,
 *                              ftns_amount, usdc_amount,  // min == desired (first liquidity)
 *                              Safe, deadline)
 *
 * The bundle file is plain JSON; Safe UI Transaction Builder app
 * accepts it via the Import button. After import, the founder
 * council signs each tx serially with 2-of-3 hardware wallets.
 *
 * Required CLI args (all):
 *   --network            mainnet network identifier (base-mainnet | sepolia)
 *   --ftns-token         FTNS token address (sprint-A-08-canonical for mainnet)
 *   --usdc-token         USDC address (Circle canonical for mainnet)
 *   --router             Aerodrome Router address (verify per Aerodrome docs)
 *   --safe-address       Foundation Safe address (LP recipient)
 *   --ftns-amount        FTNS seed amount in WHOLE tokens (script multiplies by 10^18)
 *   --usdc-amount        USDC seed amount in WHOLE dollars (script multiplies by 10^6)
 *   --out                output file path
 *
 * Optional:
 *   --deadline-minutes   tx deadline window from block.timestamp;
 *                        default 30 (Safe UI typically queues ~5 min;
 *                        sign + execute takes 10-20 min via hardware
 *                        wallets; 30-min buffer is conservative)
 *   --description        free-form description for the Safe bundle metadata
 *
 * Usage:
 *   node contracts/scripts/build-aerodrome-pool-seed-tx.js \
 *     --network base-mainnet \
 *     --ftns-token 0x5276a3756C85f2E9e46f6D34386167a209aa16e5 \
 *     --usdc-token 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 \
 *     --router 0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43 \
 *     --safe-address 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 \
 *     --ftns-amount 1000000 \
 *     --usdc-amount 250000 \
 *     --out /tmp/aerodrome-seed-bundle.json
 *
 * Exit codes:
 *   0 = bundle written
 *   1 = argument validation failed
 *   2 = address checksum / chainid validation failed
 */

const { ethers } = require("ethers");
const fs = require("fs");

// ── Aerodrome canonical (Base mainnet, 2024) — verify per docs ────
const KNOWN_AERODROME_ROUTER = "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43";
const KNOWN_USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913";
const KNOWN_FTNS_BASE = "0x5276a3756C85f2E9e46f6D34386167a209aa16e5";

// ── Chain ID mapping ─────────────────────────────────────────────
const CHAIN_IDS = {
  "base-mainnet": 8453,
  "base-sepolia": 84532,
  "sepolia": 11155111,
};

function parseArgs() {
  const args = {};
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      const key = argv[i].slice(2);
      const val = argv[i + 1];
      if (!val || val.startsWith("--")) {
        console.error(`ERROR: --${key} requires a value`);
        process.exit(1);
      }
      args[key] = val;
      i++;
    }
  }
  return args;
}

function validateAddress(label, addr) {
  if (!ethers.isAddress(addr)) {
    console.error(`ERROR: --${label} is not a valid address: ${addr}`);
    process.exit(2);
  }
  return ethers.getAddress(addr); // checksums
}

function main() {
  const args = parseArgs();
  const required = [
    "network", "ftns-token", "usdc-token", "router",
    "safe-address", "ftns-amount", "usdc-amount", "out",
  ];
  for (const r of required) {
    if (!args[r]) {
      console.error(`ERROR: missing required arg --${r}`);
      console.error("Run with no args for usage.");
      process.exit(1);
    }
  }

  const network = args["network"];
  const chainId = CHAIN_IDS[network];
  if (!chainId) {
    console.error(
      `ERROR: unknown --network ${network}; valid: ${Object.keys(CHAIN_IDS).join(", ")}`,
    );
    process.exit(1);
  }

  const ftnsToken = validateAddress("ftns-token", args["ftns-token"]);
  const usdcToken = validateAddress("usdc-token", args["usdc-token"]);
  const router = validateAddress("router", args["router"]);
  const safeAddress = validateAddress("safe-address", args["safe-address"]);

  // Warn (not fail) if the user-supplied addresses don't match
  // our embedded "known canonical" values for mainnet. These
  // CAN drift if Aerodrome / Circle redeploy; the warning is a
  // signal to re-verify rather than a hard rejection.
  if (network === "base-mainnet") {
    const warnings = [];
    if (router.toLowerCase() !== KNOWN_AERODROME_ROUTER.toLowerCase()) {
      warnings.push(
        `  --router ${router} != known canonical ${KNOWN_AERODROME_ROUTER}`,
      );
    }
    if (usdcToken.toLowerCase() !== KNOWN_USDC_BASE.toLowerCase()) {
      warnings.push(
        `  --usdc-token ${usdcToken} != known canonical ${KNOWN_USDC_BASE}`,
      );
    }
    if (ftnsToken.toLowerCase() !== KNOWN_FTNS_BASE.toLowerCase()) {
      warnings.push(
        `  --ftns-token ${ftnsToken} != known canonical ${KNOWN_FTNS_BASE}`,
      );
    }
    if (warnings.length > 0) {
      console.error(
        "WARNING: address mismatch vs script's embedded canonical:\n" +
          warnings.join("\n") +
          "\n  → re-verify via Aerodrome docs / Basescan before signing.",
      );
    }
  }

  const ftnsAmountWhole = BigInt(args["ftns-amount"]);
  const usdcAmountWhole = BigInt(args["usdc-amount"]);
  const ftnsDecimals = 18n;
  const usdcDecimals = 6n;
  const ftnsAmountWei = ftnsAmountWhole * 10n ** ftnsDecimals;
  const usdcAmountWei = usdcAmountWhole * 10n ** usdcDecimals;

  // Deadline (block.timestamp + N min). The Safe UI tx submission
  // includes its own timestamp; we anchor the deadline to "now"
  // + N min as a defensive ceiling. If signing takes longer, the
  // operator regenerates the bundle.
  const deadlineMinutes = parseInt(args["deadline-minutes"] || "30");
  const deadline = Math.floor(Date.now() / 1000) + deadlineMinutes * 60;

  const description =
    args["description"] ||
    `Aerodrome USDC-FTNS pool seed: ${ftnsAmountWhole} FTNS + ${usdcAmountWhole} USDC, recipient ${safeAddress}, deadline +${deadlineMinutes}min`;

  // ── Build txs ──────────────────────────────────────────────────

  // approve(address,uint256)
  const erc20Iface = new ethers.Interface([
    "function approve(address spender, uint256 amount) external returns (bool)",
  ]);

  // addLiquidity(address,address,bool,uint256,uint256,uint256,uint256,address,uint256)
  const routerIface = new ethers.Interface([
    "function addLiquidity(address tokenA,address tokenB,bool stable,uint256 amountADesired,uint256 amountBDesired,uint256 amountAMin,uint256 amountBMin,address to,uint256 deadline) external returns (uint256 amountA, uint256 amountB, uint256 liquidity)",
  ]);

  const tx1Data = erc20Iface.encodeFunctionData("approve", [
    router, ftnsAmountWei,
  ]);
  const tx2Data = erc20Iface.encodeFunctionData("approve", [
    router, usdcAmountWei,
  ]);
  const tx3Data = routerIface.encodeFunctionData("addLiquidity", [
    ftnsToken, usdcToken, false,        // FTNS, USDC, volatile pool
    ftnsAmountWei, usdcAmountWei,        // desired
    ftnsAmountWei, usdcAmountWei,        // min == desired (first liquidity)
    safeAddress, deadline,
  ]);

  // ── Safe Transaction Builder format ────────────────────────────
  //
  // Spec: https://help.safe.global/en/articles/40818-transaction-builder
  //
  // Minimal viable shape: top-level metadata + transactions array.
  // Each tx must have `to`, `value`, `data`, `contractMethod` (or null).
  // We omit `contractMethod` since we provide pre-encoded `data`.

  const bundle = {
    version: "1.0",
    chainId: chainId.toString(),
    createdAt: Math.floor(Date.now() / 1000) * 1000,
    meta: {
      name: "Aerodrome USDC-FTNS pool seed",
      description: description,
      txBuilderVersion: "1.16.5",
      createdFromSafeAddress: safeAddress,
      createdFromOwnerAddress: "",
      checksum: "0x0",
    },
    transactions: [
      {
        to: ftnsToken,
        value: "0",
        data: tx1Data,
        contractMethod: null,
        contractInputsValues: null,
      },
      {
        to: usdcToken,
        value: "0",
        data: tx2Data,
        contractMethod: null,
        contractInputsValues: null,
      },
      {
        to: router,
        value: "0",
        data: tx3Data,
        contractMethod: null,
        contractInputsValues: null,
      },
    ],
  };

  fs.writeFileSync(args["out"], JSON.stringify(bundle, null, 2));
  console.log(`Bundle written to ${args["out"]}`);
  console.log(`  Network:     ${network} (chainId ${chainId})`);
  console.log(`  FTNS:        ${ftnsAmountWhole} (= ${ftnsAmountWei} wei)`);
  console.log(`  USDC:        ${usdcAmountWhole} (= ${usdcAmountWei} wei)`);
  console.log(`  Implied $/FTNS: ${Number(usdcAmountWhole) / Number(ftnsAmountWhole)}`);
  console.log(`  Router:      ${router}`);
  console.log(`  Safe:        ${safeAddress}`);
  console.log(`  Deadline:    ${new Date(deadline * 1000).toISOString()} (now +${deadlineMinutes}min)`);
  console.log("");
  console.log("Next steps:");
  console.log("  1. Open Safe{Wallet} UI on " + network);
  console.log("  2. Open Transaction Builder app");
  console.log("  3. Import " + args["out"]);
  console.log("  4. Verify all 3 txs match this script's output (data + to)");
  console.log("  5. Queue + sign with 2-of-3 hardware wallets");
  console.log("  6. Execute serially (TX 1 → TX 2 → TX 3)");
  console.log("  7. Run verify-aerodrome-pool-seed.js post-execution");
}

main();
