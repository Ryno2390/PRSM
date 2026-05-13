/*
 * Aerodrome USDC-FTNS pool-seed — post-ceremony state verification.
 *
 * Companion to:
 *   docs/governance/2026-06-15-aerodrome-pool-seed-ceremony-plan.md §6
 *
 * After the Foundation Safe executes the 3-tx pool-seed sequence,
 * this script asserts the resulting on-chain state matches the
 * ceremony's intent. If ANY assertion fails, the operator must
 * STOP and file an incident — the pool may still exist, but the
 * seed may not have completed as intended.
 *
 * Six assertions per ceremony-plan §6:
 *   A. Pool exists at PoolFactory.getPool(FTNS, USDC, false), non-zero
 *   B. Pool's getReserves() matches the seeded amounts within rounding
 *   C. Pool's token0 / token1 resolve to {FTNS, USDC}
 *   D. LP_token.balanceOf(FoundationSafe) > 0
 *   E. Safe's FTNS balance decreased by exactly the seeded amount
 *   F. Safe's USDC balance decreased by exactly the seeded amount
 *
 * Assertions E + F require knowing the Safe's pre-ceremony balances.
 * The script reads them from the env vars below (operator must
 * snapshot before running the ceremony).
 *
 * Required CLI args:
 *   --network            base-mainnet | base-sepolia | sepolia
 *   --safe-address       Foundation Safe address
 *   --ftns-token         FTNS token address
 *   --usdc-token         USDC token address
 *   --factory            Aerodrome PoolFactory address
 *   --seeded-ftns        FTNS seeded (whole tokens, script multiplies 10^18)
 *   --seeded-usdc        USDC seeded (whole dollars, script multiplies 10^6)
 *   --pre-safe-ftns      Safe's FTNS balance before the ceremony (wei)
 *   --pre-safe-usdc      Safe's USDC balance before the ceremony (wei)
 *
 * Usage:
 *   PRSM_BASE_RPC_URL=https://mainnet.base.org \
 *   npx hardhat run contracts/scripts/verify-aerodrome-pool-seed.js --network base \
 *     -- --network base-mainnet \
 *     --safe-address 0x91b0000000000000000000000000000000005791 \
 *     --ftns-token 0x5276a3756C85f2E9e46f6D34386167a209aa16e5 \
 *     --usdc-token 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 \
 *     --factory 0x420DD381b31aEf6683db6B902084cB0FFECe40Da \
 *     --seeded-ftns 1000000 \
 *     --seeded-usdc 250000 \
 *     --pre-safe-ftns 100000000000000000000000000 \
 *     --pre-safe-usdc 250000000000
 *
 * Exit codes:
 *   0  all 6 assertions pass — pool seed verified
 *   1  argument validation failed
 *   2  one or more assertions failed — STOP, file incident
 */

const { ethers } = require("hardhat");

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

async function main() {
  const args = parseArgs();
  const required = [
    "network", "safe-address", "ftns-token", "usdc-token",
    "factory", "seeded-ftns", "seeded-usdc",
    "pre-safe-ftns", "pre-safe-usdc",
  ];
  for (const r of required) {
    if (!args[r]) {
      console.error(`ERROR: missing required arg --${r}`);
      process.exit(1);
    }
  }

  const provider = ethers.provider;
  const block = await provider.getBlockNumber();
  console.log(`Verifying at block ${block} on ${args["network"]}\n`);

  const safe = ethers.getAddress(args["safe-address"]);
  const ftns = ethers.getAddress(args["ftns-token"]);
  const usdc = ethers.getAddress(args["usdc-token"]);
  const factory = ethers.getAddress(args["factory"]);

  const seededFtnsWei = BigInt(args["seeded-ftns"]) * 10n ** 18n;
  const seededUsdcWei = BigInt(args["seeded-usdc"]) * 10n ** 6n;
  const preSafeFtns = BigInt(args["pre-safe-ftns"]);
  const preSafeUsdc = BigInt(args["pre-safe-usdc"]);

  const errors = [];
  const passes = [];

  // ── A. Pool exists ────────────────────────────────────────────
  const factoryAbi = [
    "function getPool(address tokenA, address tokenB, bool stable) external view returns (address)",
  ];
  const factoryC = new ethers.Contract(factory, factoryAbi, provider);
  let poolAddr;
  try {
    poolAddr = await factoryC.getPool(ftns, usdc, false);
  } catch (e) {
    poolAddr = ethers.ZeroAddress;
  }
  if (poolAddr === ethers.ZeroAddress) {
    errors.push(
      `A. Pool does NOT exist at PoolFactory.getPool(${ftns}, ${usdc}, false). Seed ceremony failed or wrong factory.`,
    );
  } else {
    passes.push(`A. Pool exists at ${poolAddr}`);
  }

  if (poolAddr !== ethers.ZeroAddress) {
    const poolAbi = [
      "function getReserves() external view returns (uint256 _reserve0, uint256 _reserve1, uint256 _blockTimestampLast)",
      "function token0() external view returns (address)",
      "function token1() external view returns (address)",
      "function balanceOf(address owner) external view returns (uint256)",
    ];
    const pool = new ethers.Contract(poolAddr, poolAbi, provider);

    // ── B + C. Reserves match + tokens are {FTNS, USDC} ─────────
    const token0 = ethers.getAddress(await pool.token0());
    const token1 = ethers.getAddress(await pool.token1());
    const tokens = new Set([token0, token1]);
    const expected = new Set([ftns, usdc]);
    if (
      tokens.size === 2 && tokens.has(ftns) && tokens.has(usdc)
    ) {
      passes.push(`C. token0 + token1 resolve to {FTNS, USDC}`);
    } else {
      errors.push(
        `C. token0/token1 mismatch: pool has {${token0}, ${token1}}, expected {${ftns}, ${usdc}}`,
      );
    }

    const reserves = await pool.getReserves();
    const reserve0 = reserves[0];
    const reserve1 = reserves[1];

    // Map reserves to FTNS/USDC slots by token0 identity
    const ftnsReserve = token0 === ftns ? reserve0 : reserve1;
    const usdcReserve = token0 === usdc ? reserve0 : reserve1;

    // Tolerance for rounding: 1 wei on FTNS side, 1 microunit on USDC.
    // First liquidity has zero slippage so amounts SHOULD be exact.
    if (ftnsReserve === seededFtnsWei) {
      passes.push(`B(FTNS). reserve matches seed: ${ftnsReserve} == ${seededFtnsWei}`);
    } else {
      const diff = ftnsReserve - seededFtnsWei;
      errors.push(
        `B(FTNS). reserve != seed: ${ftnsReserve} vs ${seededFtnsWei} (diff ${diff})`,
      );
    }
    if (usdcReserve === seededUsdcWei) {
      passes.push(`B(USDC). reserve matches seed: ${usdcReserve} == ${seededUsdcWei}`);
    } else {
      const diff = usdcReserve - seededUsdcWei;
      errors.push(
        `B(USDC). reserve != seed: ${usdcReserve} vs ${seededUsdcWei} (diff ${diff})`,
      );
    }

    // ── D. Safe holds LP tokens ─────────────────────────────────
    const lpBal = await pool.balanceOf(safe);
    if (lpBal > 0n) {
      passes.push(`D. Foundation Safe holds ${lpBal} LP tokens`);
    } else {
      errors.push(
        `D. Foundation Safe LP balance is ZERO. Liquidity routed to a different address.`,
      );
    }
  }

  // ── E. Safe's FTNS balance dropped by exactly seeded ─────────
  const erc20Abi = [
    "function balanceOf(address owner) external view returns (uint256)",
    "function allowance(address owner, address spender) external view returns (uint256)",
  ];
  const ftnsC = new ethers.Contract(ftns, erc20Abi, provider);
  const usdcC = new ethers.Contract(usdc, erc20Abi, provider);
  const postSafeFtns = BigInt(await ftnsC.balanceOf(safe));
  const postSafeUsdc = BigInt(await usdcC.balanceOf(safe));

  const ftnsDelta = preSafeFtns - postSafeFtns;
  const usdcDelta = preSafeUsdc - postSafeUsdc;

  if (ftnsDelta === seededFtnsWei) {
    passes.push(`E. Safe's FTNS balance dropped by exactly seeded amount`);
  } else {
    errors.push(
      `E. Safe FTNS delta != seeded: pre=${preSafeFtns} post=${postSafeFtns} delta=${ftnsDelta} expected=${seededFtnsWei}`,
    );
  }
  if (usdcDelta === seededUsdcWei) {
    passes.push(`F. Safe's USDC balance dropped by exactly seeded amount`);
  } else {
    errors.push(
      `F. Safe USDC delta != seeded: pre=${preSafeUsdc} post=${postSafeUsdc} delta=${usdcDelta} expected=${seededUsdcWei}`,
    );
  }

  // ── Report ────────────────────────────────────────────────────
  console.log("─── Passes ───");
  for (const p of passes) console.log(`  ✓ ${p}`);
  if (errors.length > 0) {
    console.log("");
    console.log("─── FAILURES ───");
    for (const e of errors) console.log(`  ✗ ${e}`);
    console.log("");
    console.log("STOP. Do NOT update PRSM_FTNS_USD_RATE or AERODROME_USDC_FTNS_POOL_ADDRESS env.");
    console.log("File incident per docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md.");
    process.exit(2);
  }

  console.log("");
  console.log("All 6 assertions pass. Pool seed verified.");
  console.log("");
  console.log(`Pool address: ${poolAddr}`);
  console.log("");
  console.log("Next steps (operator):");
  console.log("  1. Record pool address in prsm/config/networks.py MAINNET");
  console.log("  2. Set AERODROME_USDC_FTNS_POOL_ADDRESS env on all operator nodes");
  console.log("  3. Update PRSM_FTNS_USD_RATE to implied seed price");
  console.log("  4. Notify operators via Foundation broadcast");
  console.log("  5. Draft PRSM-CR-2026-06-XX-2 (post-ceremony ratification)");
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error("verify-aerodrome-pool-seed.js threw:", err);
    process.exit(1);
  });
