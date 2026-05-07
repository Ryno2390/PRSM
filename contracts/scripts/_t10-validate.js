/*
 * T10 live validation on Base Sepolia.
 *
 * Reads EmissionController + CompensationDistributor state, optionally
 * calls pullAndDistribute(), and prints a labelled snapshot. Used to
 * confirm on-chain that:
 *   - Pre-boundary (t < epochZeroStart + 3600): rate = 1 FTNS/s, epoch = 0
 *   - Post-boundary (t >= epochZeroStart + 3600): rate = 0.5 FTNS/s, epoch = 1
 *
 * Env:
 *   EMISSION_CONTROLLER       (default: T10 Sepolia address)
 *   COMPENSATION_DISTRIBUTOR  (default: T10 Sepolia address)
 *   FTNS_TOKEN                (default: testnet FTNS)
 *   LABEL                     (string for snapshot, e.g. "pre-boundary")
 *   CALL_PULL                 ("1" to invoke pullAndDistribute)
 */
const hre = require("hardhat");

const DEFAULTS = {
  EMISSION_CONTROLLER: "0x1478F8f5F13a5BDeBc2a0b7C185D19BEE15f312e",
  COMPENSATION_DISTRIBUTOR: "0xFd730f8E513eD184F255cb1a62791e711B2e81b9",
  FTNS_TOKEN: "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a",
};

async function main() {
  const ec = process.env.EMISSION_CONTROLLER || DEFAULTS.EMISSION_CONTROLLER;
  const cd = process.env.COMPENSATION_DISTRIBUTOR || DEFAULTS.COMPENSATION_DISTRIBUTOR;
  const ftns = process.env.FTNS_TOKEN || DEFAULTS.FTNS_TOKEN;
  const label = process.env.LABEL || "snapshot";
  const callPull = process.env.CALL_PULL === "1";

  const [signer] = await hre.ethers.getSigners();
  const provider = hre.ethers.provider;
  const block = await provider.getBlock("latest");

  const Emission = await hre.ethers.getContractFactory("EmissionController");
  const Distrib = await hre.ethers.getContractFactory("CompensationDistributor");
  const FTNS = await hre.ethers.getContractFactory("FTNSTokenSimple");

  const emission = Emission.attach(ec);
  const distributor = Distrib.attach(cd);
  const token = FTNS.attach(ftns);

  // -- Pre-state --
  const epochZero = await emission.epochZeroStartTimestamp();
  const epochDur = await emission.EPOCH_DURATION_SECONDS();
  const baselineRate = await emission.baselineRatePerSecond();
  const currentEpoch = await emission.currentEpoch();
  const currentRate = await emission.currentEpochRate();
  const lastMint = await emission.lastMintTimestamp();
  const mintedToDate = await emission.mintedToDate();
  const distribBal = await token.balanceOf(cd);
  const creatorPool = await distributor.creatorPool();
  const creatorBal = await token.balanceOf(creatorPool);

  console.log(`\n=== ${label} (block ${block.number}, ts ${block.timestamp}) ===`);
  console.log(`  epochZeroStart    : ${epochZero}`);
  console.log(`  EPOCH_DURATION_S  : ${epochDur}  (${Number(epochDur) / 3600} hours)`);
  console.log(`  baselineRate      : ${hre.ethers.formatUnits(baselineRate, 18)} FTNS/s`);
  console.log(`  currentEpoch      : ${currentEpoch}`);
  console.log(`  currentEpochRate  : ${hre.ethers.formatUnits(currentRate, 18)} FTNS/s`);
  console.log(`  lastMintTimestamp : ${lastMint}`);
  console.log(`  secs since lastMint: ${BigInt(block.timestamp) - lastMint}`);
  console.log(`  mintedToDate      : ${hre.ethers.formatUnits(mintedToDate, 18)} FTNS`);
  console.log(`  distributor bal   : ${hre.ethers.formatUnits(distribBal, 18)} FTNS`);
  console.log(`  creatorPool bal   : ${hre.ethers.formatUnits(creatorBal, 18)} FTNS`);

  if (!callPull) return;

  // -- Mint --
  console.log(`\n→ pullAndDistribute() …`);
  const tx = await distributor.connect(signer).pullAndDistribute();
  const receipt = await tx.wait();
  console.log(`  tx ${tx.hash}  (block ${receipt.blockNumber}, gas ${receipt.gasUsed})`);

  // -- Post-state --
  const newMinted = await emission.mintedToDate();
  const newLastMint = await emission.lastMintTimestamp();
  const newRate = await emission.currentEpochRate();
  const newEpoch = await emission.currentEpoch();
  const newCreatorBal = await token.balanceOf(creatorPool);

  console.log(`\n--- post-mint ---`);
  console.log(`  mintedToDate      : ${hre.ethers.formatUnits(newMinted, 18)} FTNS  (Δ = ${hre.ethers.formatUnits(newMinted - mintedToDate, 18)})`);
  console.log(`  lastMintTimestamp : ${newLastMint}  (Δ = ${newLastMint - lastMint})`);
  console.log(`  currentEpoch      : ${newEpoch}`);
  console.log(`  currentEpochRate  : ${hre.ethers.formatUnits(newRate, 18)} FTNS/s`);
  console.log(`  creatorPool bal   : ${hre.ethers.formatUnits(newCreatorBal, 18)} FTNS  (Δ = ${hre.ethers.formatUnits(newCreatorBal - creatorBal, 18)})`);
}

main().catch((e) => { console.error(e); process.exit(1); });
