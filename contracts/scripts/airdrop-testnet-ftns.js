/**
 * Manual testnet-FTNS airdrop (Option A from
 * docs/2026-05-05-public-testnet-deploy-plan.md §5).
 *
 * Sends a fixed amount of testnet-FTNS to a recipient address. Intended
 * for first ~20 testnet users; promote to a web faucet endpoint
 * (Option B) when usage exceeds 20.
 *
 * Required env vars:
 *   FTNS_TOKEN_ADDRESS — testnet FTNS contract (from
 *                        contracts/deployments/mock-ftns-base-sepolia-*.json
 *                        or prsm/config/networks.py TESTNET.ftns_token).
 *   RECIPIENT          — 0x address to airdrop to.
 *
 * Optional env vars:
 *   AMOUNT_FTNS — human-readable amount (default: 1000). Multiplied by
 *                 1e18 for transfer.
 *
 * Pre-reqs:
 *   - Faucet wallet (the signer) holds testnet-FTNS. Top up via the
 *     deployer's MockFTNS mint function if balance runs low.
 *   - Faucet wallet holds Sepolia ETH for gas. Top up via Coinbase
 *     faucet at https://www.coinbase.com/faucets/base-sepolia-faucet.
 *
 * Usage:
 *   PRIVATE_KEY=0x<faucet-key> \
 *   FTNS_TOKEN_ADDRESS=0x<testnet-ftns> \
 *   RECIPIENT=0x<user-address> \
 *   AMOUNT_FTNS=1000 \
 *       npx hardhat run scripts/airdrop-testnet-ftns.js --network base-sepolia
 *
 * Output:
 *   Prints tx hash + Basescan link. Logs the airdrop to
 *   contracts/deployments/airdrops-base-sepolia.log for tracking.
 */

const fs = require("fs");
const path = require("path");
const hre = require("hardhat");

async function main() {
  const ftnsAddress = process.env.FTNS_TOKEN_ADDRESS;
  const recipient = process.env.RECIPIENT;
  const amountStr = process.env.AMOUNT_FTNS || "1000";

  if (!ftnsAddress) throw new Error("FTNS_TOKEN_ADDRESS env var required");
  if (!recipient) throw new Error("RECIPIENT env var required");

  // Validate addresses (throws on bad checksum)
  const ftnsChecksum = hre.ethers.getAddress(ftnsAddress);
  const recipientChecksum = hre.ethers.getAddress(recipient);

  const network = hre.network.name;
  if (network !== "base-sepolia" && network !== "hardhat" && network !== "localhost") {
    throw new Error(
      `Refusing to run on network '${network}'. This script is testnet-only. ` +
      `Use base-sepolia or a local hardhat fork.`
    );
  }

  const [signer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(signer.address);
  console.log(`\n=== Testnet-FTNS airdrop on ${network} ===`);
  console.log(`Faucet wallet:    ${signer.address}`);
  console.log(`Faucet ETH bal:   ${hre.ethers.formatEther(balance)} ETH`);

  // Use the standard ERC20 ABI subset we need. MockFTNS / FTNSTokenSimple
  // both expose the standard ERC20 surface.
  const erc20Abi = [
    "function balanceOf(address owner) view returns (uint256)",
    "function transfer(address to, uint256 amount) returns (bool)",
    "function symbol() view returns (string)",
    "function decimals() view returns (uint8)",
  ];
  const ftns = new hre.ethers.Contract(ftnsChecksum, erc20Abi, signer);

  const decimals = await ftns.decimals();
  const symbol = await ftns.symbol();
  const faucetBal = await ftns.balanceOf(signer.address);

  // Parse amount with the token's decimal precision (typically 18)
  const amountWei = hre.ethers.parseUnits(amountStr, decimals);

  console.log(`Token:            ${ftnsChecksum} (${symbol})`);
  console.log(`Faucet ${symbol} bal:  ${hre.ethers.formatUnits(faucetBal, decimals)} ${symbol}`);
  console.log(`Recipient:        ${recipientChecksum}`);
  console.log(`Amount:           ${amountStr} ${symbol}`);

  if (faucetBal < amountWei) {
    throw new Error(
      `Faucet wallet has only ${hre.ethers.formatUnits(faucetBal, decimals)} ${symbol}; ` +
      `requested ${amountStr}. Top up the faucet wallet first.`
    );
  }

  // Pre-flight: don't airdrop if recipient already has a meaningful balance.
  // This guards against accidentally re-airdropping to the same address.
  const recipientBalBefore = await ftns.balanceOf(recipientChecksum);
  if (recipientBalBefore > amountWei * 5n) {
    console.warn(
      `\n⚠️  Recipient already has ${hre.ethers.formatUnits(recipientBalBefore, decimals)} ` +
      `${symbol} (>5x airdrop amount). Skipping; pass FORCE=1 to override.`
    );
    if (process.env.FORCE !== "1") {
      console.log("Aborting — set FORCE=1 to airdrop anyway.");
      process.exit(0);
    }
    console.log("FORCE=1 set; proceeding with airdrop despite high recipient balance.");
  }

  console.log(`\nSending transfer…`);
  const tx = await ftns.transfer(recipientChecksum, amountWei);
  console.log(`  Tx hash:        ${tx.hash}`);
  console.log(`  Basescan:       https://sepolia.basescan.org/tx/${tx.hash}`);
  const receipt = await tx.wait();
  console.log(`  Confirmed in block ${receipt.blockNumber} (gas used: ${receipt.gasUsed.toString()})`);

  const recipientBalAfter = await ftns.balanceOf(recipientChecksum);
  console.log(`\nRecipient balance now: ${hre.ethers.formatUnits(recipientBalAfter, decimals)} ${symbol}`);

  // Append to airdrop log for tracking
  const logPath = path.join(__dirname, "..", "deployments", "airdrops-base-sepolia.log");
  const logLine = JSON.stringify({
    timestamp: new Date().toISOString(),
    network,
    faucet: signer.address,
    recipient: recipientChecksum,
    amount: amountStr,
    symbol,
    tx_hash: tx.hash,
    block: receipt.blockNumber,
  }) + "\n";
  fs.appendFileSync(logPath, logLine);
  console.log(`Logged to: ${logPath}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ airdrop failed:");
    console.error(error);
    process.exit(1);
  });
