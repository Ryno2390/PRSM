/*
 * One-shot helper: grant MINTER_ROLE on the FTNS testnet token to a new
 * EmissionController, and (optionally) revoke it from the previous one.
 * Used during T10 (accelerated halving) testnet redeploy.
 *
 * Env vars:
 *   FTNS_TOKEN_ADDRESS       — testnet FTNSTokenSimple
 *   NEW_EMISSION_CONTROLLER  — address to grant
 *   OLD_EMISSION_CONTROLLER  — (optional) address to revoke
 */
const hre = require("hardhat");

async function main() {
  const ftnsAddr = process.env.FTNS_TOKEN_ADDRESS;
  const newCtrl = process.env.NEW_EMISSION_CONTROLLER;
  const oldCtrl = process.env.OLD_EMISSION_CONTROLLER;
  if (!ftnsAddr || !newCtrl) throw new Error("FTNS_TOKEN_ADDRESS and NEW_EMISSION_CONTROLLER required");

  const [signer] = await hre.ethers.getSigners();
  console.log(`Signer: ${signer.address}`);

  const FTNS = await hre.ethers.getContractFactory("FTNSTokenSimple");
  const token = FTNS.attach(ftnsAddr);

  const minterRole = await token.MINTER_ROLE();
  console.log(`MINTER_ROLE = ${minterRole}`);

  if (oldCtrl) {
    const has = await token.hasRole(minterRole, oldCtrl);
    if (has) {
      console.log(`Revoking MINTER_ROLE from old controller ${oldCtrl}…`);
      const tx = await token.revokeRole(minterRole, oldCtrl);
      await tx.wait();
      console.log(`  ✔ tx ${tx.hash}`);
    } else {
      console.log(`Old controller ${oldCtrl} did not hold MINTER_ROLE; skipping revoke.`);
    }
  }

  const alreadyHas = await token.hasRole(minterRole, newCtrl);
  if (alreadyHas) {
    console.log(`New controller ${newCtrl} already has MINTER_ROLE; skipping.`);
  } else {
    console.log(`Granting MINTER_ROLE to new controller ${newCtrl}…`);
    const tx = await token.grantRole(minterRole, newCtrl);
    await tx.wait();
    console.log(`  ✔ tx ${tx.hash}`);
  }

  console.log("Done.");
}

main().catch((e) => { console.error(e); process.exit(1); });
