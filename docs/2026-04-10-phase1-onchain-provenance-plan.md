# Phase 1: On-Chain Provenance & Royalty Distribution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move PRSM royalty distribution from a local SQLite ledger onto Base mainnet smart contracts so any third party can independently verify a creator earned FTNS for content they registered.

**Architecture:**
- Two new Solidity contracts on Base: `ProvenanceRegistry` (immutable creator/rate map) and `RoyaltyDistributor` (atomic 3-way FTNS split).
- Python Web3 clients in `prsm/economy/web3/` mirror existing patterns from `ftns_onchain.py`.
- Wired into `prsm/node/content_economy.py` behind feature flag `PRSM_ONCHAIN_PROVENANCE=1` so existing flows keep working during rollout.
- TDD throughout: failing test → minimal impl → green → commit.

**Tech Stack:** Solidity 0.8.22, OpenZeppelin upgradeable contracts, Hardhat, Web3.py 6.x, eth-account, pytest, pytest-asyncio.

**Pre-existing context the implementer needs:**
- FTNS token already deployed on Base mainnet at `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` — see `contracts/contracts/FTNSTokenSimple.sol`.
- Existing on-chain payment path: `prsm/economy/ftns_onchain.py::OnChainFTNSLedger` + `prsm/economy/batch_settlement.py::BatchSettlementManager`.
- Existing royalty calc (off-chain, current): `prsm/node/content_economy.py` — constants `ORIGINAL_CREATOR_ROYALTY_RATE = 0.08`, `DERIVATIVE_CREATOR_ROYALTY_RATE = 0.01`, `NETWORK_FEE_RATE = 0.02`.
- Hardhat config at `contracts/hardhat.config.js`. Networks `base`, `base-fork`, `sepolia` already set up. Etherscan V2 multichain key reused for Basescan verification.

**Out of scope for Phase 1:**
- Marketplace listings (Phase 3).
- Derivative royalty chain (>1 hop) — Phase 1 supports only original creator + serving node + treasury. Multi-hop derivative can be added later via mapping additions without contract redeploy.
- NFT/ERC-721 content wrappers — content addressed by `bytes32` hash only.
- Permit/EIP-2612 gasless approval — use plain `approve` + `transferFrom`. Permit is a future optimization.

---

## File Structure

| File | Status | Purpose |
|---|---|---|
| `contracts/contracts/ProvenanceRegistry.sol` | **Create** | Maps `bytes32 contentHash → (creator, royaltyRateBps, registeredAt, metadataUri)`. Immutable after registration except for `transferOwnership`. |
| `contracts/contracts/RoyaltyDistributor.sol` | **Create** | Stateless splitter. `distributeRoyalty(contentHash, servingNode, gross)` pulls FTNS from msg.sender via `transferFrom`, splits creator/serving-node/treasury, emits `RoyaltyPaid`. |
| `contracts/test/ProvenanceRegistry.test.js` | **Create** | Hardhat/Mocha tests for registry. |
| `contracts/test/RoyaltyDistributor.test.js` | **Create** | Hardhat/Mocha tests for distributor (uses mock ERC-20 + mock registry). |
| `contracts/scripts/deploy-provenance.js` | **Create** | Hardhat script: deploy registry, deploy distributor pointing at registry + FTNS, save addresses to `contracts/deployments/`. |
| `prsm/economy/web3/provenance_registry.py` | **Create** | Async Web3.py client. Methods: `register_content`, `get_content`, `is_registered`, `transfer_ownership`. |
| `prsm/economy/web3/royalty_distributor.py` | **Create** | Async Web3.py client. Methods: `distribute_royalty`, `preview_split`. Handles ERC-20 approval flow. |
| `prsm/economy/web3/__init__.py` | **Modify** | Export new clients. |
| `prsm/node/content_economy.py` | **Modify** | When `PRSM_ONCHAIN_PROVENANCE=1`, route royalty calc + payment through new clients instead of local ledger. |
| `prsm/cli/provenance.py` | **Create** | Click subcommands `register`, `info`, `transfer`. |
| `prsm/cli/__init__.py` | **Modify** | Register `provenance` subcommand group. |
| `tests/contracts/test_provenance_registry_client.py` | **Create** | Pytest unit tests for Python client (mocks Web3). |
| `tests/contracts/test_royalty_distributor_client.py` | **Create** | Pytest unit tests for Python client (mocks Web3). |
| `tests/integration/test_onchain_provenance_e2e.py` | **Create** | End-to-end test using Hardhat local node: deploy contracts, register content, distribute royalty, assert on-chain event. |
| `CHANGELOG.md` | **Modify** | Add Phase 1 entry. |
| `docs/ONCHAIN_PROVENANCE.md` | **Create** | User-facing doc explaining the new flow. |

---

## Task 1: ProvenanceRegistry.sol Contract & Tests

**Files:**
- Create: `contracts/contracts/ProvenanceRegistry.sol`
- Create: `contracts/test/ProvenanceRegistry.test.js`

- [ ] **Step 1: Write the failing Hardhat test**

Create `contracts/test/ProvenanceRegistry.test.js`:

```javascript
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ProvenanceRegistry", function () {
  let registry;
  let owner, creator, otherCreator;

  beforeEach(async function () {
    [owner, creator, otherCreator] = await ethers.getSigners();
    const Registry = await ethers.getContractFactory("ProvenanceRegistry");
    registry = await Registry.deploy();
    await registry.waitForDeployment();
  });

  describe("registerContent", function () {
    it("registers new content with creator address as msg.sender", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("test-content"));
      const royaltyRateBps = 800; // 8%
      const metadataUri = "ipfs://QmTest";

      await expect(
        registry.connect(creator).registerContent(contentHash, royaltyRateBps, metadataUri)
      )
        .to.emit(registry, "ContentRegistered")
        .withArgs(contentHash, creator.address, royaltyRateBps, metadataUri);

      const content = await registry.contents(contentHash);
      expect(content.creator).to.equal(creator.address);
      expect(content.royaltyRateBps).to.equal(royaltyRateBps);
      expect(content.metadataUri).to.equal(metadataUri);
      expect(content.registeredAt).to.be.gt(0);
    });

    it("rejects duplicate registration", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("dup"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://A");
      await expect(
        registry.connect(otherCreator).registerContent(contentHash, 500, "ipfs://B")
      ).to.be.revertedWith("Already registered");
    });

    it("rejects royalty rate above 100%", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("high"));
      await expect(
        registry.connect(creator).registerContent(contentHash, 10001, "ipfs://X")
      ).to.be.revertedWith("Invalid rate");
    });

    it("accepts zero royalty rate (free content)", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("free"));
      await registry.connect(creator).registerContent(contentHash, 0, "ipfs://free");
      const content = await registry.contents(contentHash);
      expect(content.royaltyRateBps).to.equal(0);
    });
  });

  describe("isRegistered", function () {
    it("returns false for unregistered content", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("none"));
      expect(await registry.isRegistered(contentHash)).to.equal(false);
    });

    it("returns true after registration", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("yes"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://Y");
      expect(await registry.isRegistered(contentHash)).to.equal(true);
    });
  });

  describe("transferOwnership", function () {
    it("allows current creator to transfer", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("xfer"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");

      await expect(
        registry.connect(creator).transferContentOwnership(contentHash, otherCreator.address)
      )
        .to.emit(registry, "OwnershipTransferred")
        .withArgs(contentHash, creator.address, otherCreator.address);

      const content = await registry.contents(contentHash);
      expect(content.creator).to.equal(otherCreator.address);
    });

    it("rejects transfer by non-owner", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("xfer2"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");
      await expect(
        registry.connect(otherCreator).transferContentOwnership(contentHash, otherCreator.address)
      ).to.be.revertedWith("Not creator");
    });

    it("rejects transfer to zero address", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("xfer3"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");
      await expect(
        registry.connect(creator).transferContentOwnership(contentHash, ethers.ZeroAddress)
      ).to.be.revertedWith("Zero address");
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd contracts && npx hardhat test test/ProvenanceRegistry.test.js
```

Expected: FAIL with "ContractFactory.getContractFactory: ProvenanceRegistry — Contract source not found".

- [ ] **Step 3: Write the Solidity contract**

Create `contracts/contracts/ProvenanceRegistry.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/**
 * @title ProvenanceRegistry
 * @notice On-chain registry of PRSM content provenance — content hash to creator
 *         address and royalty rate. Source of truth for RoyaltyDistributor.
 * @dev Content is identified by an opaque bytes32 hash (keccak256 of canonical
 *      content bytes, set by the off-chain client). Royalty rates are basis
 *      points (1 bp = 0.01%, max 10000 = 100%). Records are immutable except
 *      for ownership transfer.
 */
contract ProvenanceRegistry {
    struct Content {
        address creator;        // Receives royalties
        uint16 royaltyRateBps;  // 0..10000
        uint64 registeredAt;    // unix seconds
        string metadataUri;     // ipfs://, https://, etc.
    }

    mapping(bytes32 => Content) public contents;

    event ContentRegistered(
        bytes32 indexed contentHash,
        address indexed creator,
        uint16 royaltyRateBps,
        string metadataUri
    );

    event OwnershipTransferred(
        bytes32 indexed contentHash,
        address indexed previousCreator,
        address indexed newCreator
    );

    /**
     * @notice Register a new piece of content. Caller becomes the creator.
     * @param contentHash keccak256 hash of canonical content bytes
     * @param royaltyRateBps royalty rate in basis points (max 10000)
     * @param metadataUri off-chain pointer to descriptive metadata
     */
    function registerContent(
        bytes32 contentHash,
        uint16 royaltyRateBps,
        string calldata metadataUri
    ) external {
        require(contents[contentHash].creator == address(0), "Already registered");
        require(royaltyRateBps <= 10000, "Invalid rate");

        contents[contentHash] = Content({
            creator: msg.sender,
            royaltyRateBps: royaltyRateBps,
            registeredAt: uint64(block.timestamp),
            metadataUri: metadataUri
        });

        emit ContentRegistered(contentHash, msg.sender, royaltyRateBps, metadataUri);
    }

    /**
     * @notice Transfer creator role for an existing piece of content.
     * @dev Only the current creator can transfer. New creator receives all
     *      future royalties. Royalty rate is preserved.
     */
    function transferContentOwnership(bytes32 contentHash, address newCreator) external {
        Content storage c = contents[contentHash];
        require(c.creator == msg.sender, "Not creator");
        require(newCreator != address(0), "Zero address");

        address previous = c.creator;
        c.creator = newCreator;

        emit OwnershipTransferred(contentHash, previous, newCreator);
    }

    /**
     * @notice Convenience accessor — true iff contentHash has a creator.
     */
    function isRegistered(bytes32 contentHash) external view returns (bool) {
        return contents[contentHash].creator != address(0);
    }
}
```

- [ ] **Step 4: Compile and run tests to verify they pass**

```bash
cd contracts && npx hardhat compile && npx hardhat test test/ProvenanceRegistry.test.js
```

Expected: All 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add contracts/contracts/ProvenanceRegistry.sol contracts/test/ProvenanceRegistry.test.js
git commit -m "feat(contracts): add ProvenanceRegistry contract for on-chain content ownership"
```

---

## Task 2: RoyaltyDistributor.sol Contract & Tests

**Files:**
- Create: `contracts/contracts/RoyaltyDistributor.sol`
- Create: `contracts/test/RoyaltyDistributor.test.js`
- Create: `contracts/contracts/test/MockERC20.sol` (test helper)

- [ ] **Step 1: Write the test helper MockERC20**

Create `contracts/contracts/test/MockERC20.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/// @dev Test-only ERC20. Anyone can mint. Not for production.
contract MockERC20 is ERC20 {
    constructor() ERC20("MockFTNS", "MFTNS") {}

    function mint(address to, uint256 amount) external {
        _mint(to, amount);
    }
}
```

- [ ] **Step 2: Write the failing Hardhat test**

Create `contracts/test/RoyaltyDistributor.test.js`:

```javascript
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("RoyaltyDistributor", function () {
  let registry, token, distributor;
  let admin, treasury, creator, servingNode, payer;
  const ONE = ethers.parseUnits("1", 18);

  beforeEach(async function () {
    [admin, treasury, creator, servingNode, payer] = await ethers.getSigners();

    const Registry = await ethers.getContractFactory("ProvenanceRegistry");
    registry = await Registry.deploy();
    await registry.waitForDeployment();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Distributor = await ethers.getContractFactory("RoyaltyDistributor");
    distributor = await Distributor.deploy(
      await token.getAddress(),
      await registry.getAddress(),
      treasury.address
    );
    await distributor.waitForDeployment();

    // Mint payer some tokens and approve distributor
    await token.mint(payer.address, ONE * 1000n);
    await token.connect(payer).approve(await distributor.getAddress(), ONE * 1000n);
  });

  describe("distributeRoyalty", function () {
    it("splits payment 8% creator / 2% network / 90% serving node", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("split-test"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");

      const gross = ONE * 100n; // 100 FTNS
      const expectedCreator = (gross * 800n) / 10000n; // 8 FTNS
      const expectedNetwork = (gross * 200n) / 10000n; // 2 FTNS
      const expectedNode = gross - expectedCreator - expectedNetwork; // 90 FTNS

      await expect(
        distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross)
      )
        .to.emit(distributor, "RoyaltyPaid")
        .withArgs(
          contentHash,
          payer.address,
          creator.address,
          servingNode.address,
          expectedCreator,
          expectedNetwork,
          expectedNode
        );

      expect(await token.balanceOf(creator.address)).to.equal(expectedCreator);
      expect(await token.balanceOf(treasury.address)).to.equal(expectedNetwork);
      expect(await token.balanceOf(servingNode.address)).to.equal(expectedNode);
    });

    it("reverts if content not registered", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("missing"));
      await expect(
        distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, ONE * 10n)
      ).to.be.revertedWith("Not registered");
    });

    it("reverts if gross is zero", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("zero"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");
      await expect(
        distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, 0)
      ).to.be.revertedWith("Zero amount");
    });

    it("reverts if serving node is zero address", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("zeronode"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");
      await expect(
        distributor.connect(payer).distributeRoyalty(contentHash, ethers.ZeroAddress, ONE * 10n)
      ).to.be.revertedWith("Zero serving node");
    });

    it("handles 0% royalty (creator gets nothing, all to node + treasury)", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("free"));
      await registry.connect(creator).registerContent(contentHash, 0, "ipfs://F");

      const gross = ONE * 100n;
      await distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross);

      expect(await token.balanceOf(creator.address)).to.equal(0);
      expect(await token.balanceOf(treasury.address)).to.equal((gross * 200n) / 10000n);
      expect(await token.balanceOf(servingNode.address)).to.equal(gross - (gross * 200n) / 10000n);
    });

    it("handles 100% royalty (creator gets everything except network fee)", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("max"));
      await registry.connect(creator).registerContent(contentHash, 9800, "ipfs://M");

      const gross = ONE * 100n;
      await distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross);

      expect(await token.balanceOf(creator.address)).to.equal((gross * 9800n) / 10000n);
      expect(await token.balanceOf(treasury.address)).to.equal((gross * 200n) / 10000n);
      expect(await token.balanceOf(servingNode.address)).to.equal(0);
    });

    it("reverts if creator + network fee exceed 100%", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("over"));
      await registry.connect(creator).registerContent(contentHash, 9900, "ipfs://O");

      const gross = ONE * 100n;
      // creator share = 99, network = 2, sum = 101 > 100 → underflow → revert
      await expect(
        distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross)
      ).to.be.revertedWith("Rate plus fee exceeds 100%");
    });
  });

  describe("preview", function () {
    it("returns the same split distributeRoyalty would apply", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("preview"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://P");

      const gross = ONE * 50n;
      const [c, n, s] = await distributor.preview(contentHash, gross);
      expect(c).to.equal((gross * 800n) / 10000n);
      expect(n).to.equal((gross * 200n) / 10000n);
      expect(s).to.equal(gross - c - n);
    });
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd contracts && npx hardhat test test/RoyaltyDistributor.test.js
```

Expected: FAIL — RoyaltyDistributor not found.

- [ ] **Step 4: Write the Solidity contract**

Create `contracts/contracts/RoyaltyDistributor.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

interface IProvenanceRegistry {
    function contents(bytes32 contentHash)
        external
        view
        returns (address creator, uint16 royaltyRateBps, uint64 registeredAt, string memory metadataUri);
}

/**
 * @title RoyaltyDistributor
 * @notice Atomic three-way splitter for FTNS payments tied to registered content.
 * @dev Stateless. Pulls FTNS from msg.sender via transferFrom, so the payer must
 *      have approved this contract for at least `gross` first. Splits:
 *        creator share = gross * royaltyRateBps / 10000  (from registry)
 *        network share = gross * NETWORK_FEE_BPS / 10000 (constant)
 *        serving node share = gross - creator - network
 *      Reverts if creator + network share exceed gross (i.e. royaltyRateBps + 200 > 10000).
 */
contract RoyaltyDistributor is ReentrancyGuard {
    IERC20 public immutable ftns;
    IProvenanceRegistry public immutable registry;
    address public immutable networkTreasury;

    /// @dev 2% network fee, basis points
    uint16 public constant NETWORK_FEE_BPS = 200;

    event RoyaltyPaid(
        bytes32 indexed contentHash,
        address indexed payer,
        address indexed creator,
        address servingNode,
        uint256 creatorAmount,
        uint256 networkAmount,
        uint256 servingNodeAmount
    );

    constructor(address _ftns, address _registry, address _networkTreasury) {
        require(_ftns != address(0), "Zero ftns");
        require(_registry != address(0), "Zero registry");
        require(_networkTreasury != address(0), "Zero treasury");
        ftns = IERC20(_ftns);
        registry = IProvenanceRegistry(_registry);
        networkTreasury = _networkTreasury;
    }

    /**
     * @notice Pull `gross` FTNS from msg.sender and split it 3 ways.
     * @param contentHash content registered in ProvenanceRegistry
     * @param servingNode address of node that served the content
     * @param gross total FTNS amount being distributed (in token base units)
     */
    function distributeRoyalty(
        bytes32 contentHash,
        address servingNode,
        uint256 gross
    ) external nonReentrant {
        require(gross > 0, "Zero amount");
        require(servingNode != address(0), "Zero serving node");

        (address creator, uint16 rateBps, , ) = registry.contents(contentHash);
        require(creator != address(0), "Not registered");

        uint256 creatorAmt = (gross * rateBps) / 10000;
        uint256 networkAmt = (gross * NETWORK_FEE_BPS) / 10000;
        require(creatorAmt + networkAmt <= gross, "Rate plus fee exceeds 100%");
        uint256 nodeAmt = gross - creatorAmt - networkAmt;

        // Pull full amount once, then push to recipients.
        require(ftns.transferFrom(msg.sender, address(this), gross), "Pull failed");

        if (creatorAmt > 0) {
            require(ftns.transfer(creator, creatorAmt), "Creator xfer failed");
        }
        if (networkAmt > 0) {
            require(ftns.transfer(networkTreasury, networkAmt), "Network xfer failed");
        }
        if (nodeAmt > 0) {
            require(ftns.transfer(servingNode, nodeAmt), "Node xfer failed");
        }

        emit RoyaltyPaid(
            contentHash,
            msg.sender,
            creator,
            servingNode,
            creatorAmt,
            networkAmt,
            nodeAmt
        );
    }

    /**
     * @notice Read-only preview of how `gross` would be split for `contentHash`.
     * @return creatorAmount, networkAmount, servingNodeAmount
     */
    function preview(bytes32 contentHash, uint256 gross)
        external
        view
        returns (uint256 creatorAmount, uint256 networkAmount, uint256 servingNodeAmount)
    {
        (address creator, uint16 rateBps, , ) = registry.contents(contentHash);
        require(creator != address(0), "Not registered");

        creatorAmount = (gross * rateBps) / 10000;
        networkAmount = (gross * NETWORK_FEE_BPS) / 10000;
        require(creatorAmount + networkAmount <= gross, "Rate plus fee exceeds 100%");
        servingNodeAmount = gross - creatorAmount - networkAmount;
    }
}
```

- [ ] **Step 5: Compile and run tests**

```bash
cd contracts && npx hardhat compile && npx hardhat test test/RoyaltyDistributor.test.js
```

Expected: All 8 tests pass.

- [ ] **Step 6: Run full contracts test suite to make sure nothing else broke**

```bash
cd contracts && npx hardhat test
```

Expected: All tests in the suite still pass.

- [ ] **Step 7: Commit**

```bash
git add contracts/contracts/RoyaltyDistributor.sol contracts/contracts/test/MockERC20.sol contracts/test/RoyaltyDistributor.test.js
git commit -m "feat(contracts): add RoyaltyDistributor for atomic FTNS three-way split"
```

---

## Task 3: Hardhat Deploy Script + Local Hardhat-Node Smoke Test

**Files:**
- Create: `contracts/scripts/deploy-provenance.js`

- [ ] **Step 1: Write the deploy script**

Create `contracts/scripts/deploy-provenance.js`:

```javascript
/*
 * Deploys ProvenanceRegistry + RoyaltyDistributor.
 *
 * Required env vars:
 *   FTNS_TOKEN_ADDRESS    - existing FTNS ERC20 on the target network
 *   NETWORK_TREASURY      - address that receives the 2% network fee
 *
 * Usage:
 *   npx hardhat run scripts/deploy-provenance.js --network base-sepolia
 *   npx hardhat run scripts/deploy-provenance.js --network base
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const ftnsAddress = process.env.FTNS_TOKEN_ADDRESS;
  const treasury = process.env.NETWORK_TREASURY;
  if (!ftnsAddress) throw new Error("FTNS_TOKEN_ADDRESS env var required");
  if (!treasury) throw new Error("NETWORK_TREASURY env var required");

  const network = hre.network.name;
  console.log(`\n=== Deploying provenance contracts to ${network} ===`);
  console.log(`FTNS token:       ${ftnsAddress}`);
  console.log(`Network treasury: ${treasury}`);

  const [deployer] = await hre.ethers.getSigners();
  console.log(`Deployer:         ${deployer.address}`);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log(`Deployer balance: ${hre.ethers.formatEther(balance)} ETH`);
  if (balance === 0n) throw new Error("Deployer has zero balance");

  // 1. Registry
  console.log("\nDeploying ProvenanceRegistry…");
  const Registry = await hre.ethers.getContractFactory("ProvenanceRegistry");
  const registry = await Registry.deploy();
  await registry.waitForDeployment();
  const registryAddress = await registry.getAddress();
  console.log(`  ProvenanceRegistry: ${registryAddress}`);

  // 2. Distributor
  console.log("\nDeploying RoyaltyDistributor…");
  const Distributor = await hre.ethers.getContractFactory("RoyaltyDistributor");
  const distributor = await Distributor.deploy(ftnsAddress, registryAddress, treasury);
  await distributor.waitForDeployment();
  const distributorAddress = await distributor.getAddress();
  console.log(`  RoyaltyDistributor: ${distributorAddress}`);

  // 3. Save deployment manifest
  const manifest = {
    network,
    chainId: (await hre.ethers.provider.getNetwork()).chainId.toString(),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      ProvenanceRegistry: registryAddress,
      RoyaltyDistributor: distributorAddress,
      FTNSToken: ftnsAddress,
      NetworkTreasury: treasury,
    },
  };

  const outDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `provenance-${network}-${Date.now()}.json`);
  fs.writeFileSync(outFile, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved → ${outFile}`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
```

- [ ] **Step 2: Add a `base-sepolia` network entry to `hardhat.config.js` if missing**

Inspect `contracts/hardhat.config.js`. If `base-sepolia` is missing, add it inside the `networks: { … }` block:

```javascript
"base-sepolia": {
  url: process.env.BASE_SEPOLIA_RPC_URL || "https://sepolia.base.org",
  accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
  chainId: 84532,
  gas: "auto",
  gasPrice: "auto",
  confirmations: 2,
  timeoutBlocks: 200,
  skipDryRun: false,
},
```

Also extend `etherscan.customChains` with:

```javascript
{
  network: "base-sepolia",
  chainId: 84532,
  urls: {
    apiURL: "https://api.etherscan.io/v2/api",
    browserURL: "https://sepolia.basescan.org",
  },
},
```

- [ ] **Step 3: Local smoke test against `hardhat` in-process network**

Run a one-shot deploy against the in-process Hardhat network using a mock FTNS as the token, to validate the script wires up correctly:

```bash
cd contracts && FTNS_TOKEN_ADDRESS=0x0000000000000000000000000000000000000001 NETWORK_TREASURY=0x0000000000000000000000000000000000000002 npx hardhat run scripts/deploy-provenance.js --network hardhat
```

Expected: prints deployed addresses, writes a manifest under `contracts/deployments/`. (The dummy FTNS address is fine because the script does not call into FTNS — the constructor just stores the address.)

- [ ] **Step 4: Commit**

```bash
git add contracts/scripts/deploy-provenance.js contracts/hardhat.config.js
git commit -m "feat(contracts): add deploy-provenance script and base-sepolia network config"
```

> **Note:** Live testnet/mainnet deployment is **deferred to Task 10** so the Python integration can be validated first. Do not deploy to Base Sepolia or Base mainnet here.

---

## Task 4: Python Web3 Client — ProvenanceRegistryClient

**Files:**
- Create: `prsm/economy/web3/provenance_registry.py`
- Create: `tests/contracts/__init__.py` (if missing)
- Create: `tests/contracts/test_provenance_registry_client.py`

- [ ] **Step 1: Write the failing pytest**

Create `tests/contracts/test_provenance_registry_client.py`:

```python
"""Unit tests for ProvenanceRegistryClient.

These mock Web3 entirely — they validate the client wraps the contract
ABI correctly and handles common error cases. End-to-end happy-path
testing against a real Hardhat node lives in
tests/integration/test_onchain_provenance_e2e.py.
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.provenance_registry import (
    ProvenanceRegistryClient,
    ContentRecord,
)


def _hash(s: str) -> bytes:
    return hashlib.sha3_256(s.encode()).digest()


@pytest.fixture
def mock_web3():
    with patch("prsm.economy.web3.provenance_registry.Web3") as MockWeb3:
        w3_instance = MagicMock()
        MockWeb3.return_value = w3_instance
        MockWeb3.HTTPProvider.return_value = MagicMock()
        yield w3_instance, MockWeb3


def _make_client(mock_web3, deploy_address="0xRegistry"):
    w3_instance, _ = mock_web3
    contract = MagicMock()
    w3_instance.eth.contract.return_value = contract

    account = MagicMock()
    account.address = "0xCreator"
    with patch(
        "prsm.economy.web3.provenance_registry.Account.from_key",
        return_value=account,
    ):
        client = ProvenanceRegistryClient(
            rpc_url="http://localhost:8545",
            contract_address=deploy_address,
            private_key="0x" + "11" * 32,
        )
    return client, contract, w3_instance


def test_register_content_builds_and_sends_tx(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.registerContent.return_value.build_transaction.return_value = {
        "to": "0xRegistry",
        "data": "0x",
        "gas": 100000,
        "gasPrice": 1,
        "nonce": 0,
        "chainId": 8453,
    }
    w3.eth.get_transaction_count.return_value = 0
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453

    signed = MagicMock()
    signed.raw_transaction = b"raw"
    w3.eth.account.sign_transaction.return_value = signed
    w3.eth.send_raw_transaction.return_value = b"\xab" * 32
    w3.eth.wait_for_transaction_receipt.return_value = MagicMock(status=1, transactionHash=b"\xab" * 32)

    content_hash = _hash("hello")
    tx_hash = client.register_content(content_hash, royalty_rate_bps=800, metadata_uri="ipfs://X")

    contract.functions.registerContent.assert_called_once_with(content_hash, 800, "ipfs://X")
    assert tx_hash.startswith("0x")


def test_register_content_rejects_invalid_rate(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    with pytest.raises(ValueError, match="royalty_rate_bps"):
        client.register_content(_hash("x"), royalty_rate_bps=10001, metadata_uri="ipfs://X")


def test_register_content_rejects_wrong_hash_length(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    with pytest.raises(ValueError, match="32 bytes"):
        client.register_content(b"short", royalty_rate_bps=800, metadata_uri="ipfs://X")


def test_get_content_returns_record(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.contents.return_value.call.return_value = (
        "0xCreator",
        800,
        1700000000,
        "ipfs://X",
    )
    rec = client.get_content(_hash("hello"))
    assert isinstance(rec, ContentRecord)
    assert rec.creator == "0xCreator"
    assert rec.royalty_rate_bps == 800
    assert rec.registered_at == 1700000000
    assert rec.metadata_uri == "ipfs://X"


def test_get_content_returns_none_for_unregistered(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.contents.return_value.call.return_value = (
        "0x0000000000000000000000000000000000000000",
        0,
        0,
        "",
    )
    rec = client.get_content(_hash("missing"))
    assert rec is None


def test_is_registered_returns_bool(mock_web3):
    client, contract, w3 = _make_client(mock_web3)
    contract.functions.isRegistered.return_value.call.return_value = True
    assert client.is_registered(_hash("yes")) is True
    contract.functions.isRegistered.return_value.call.return_value = False
    assert client.is_registered(_hash("no")) is False
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM && python -m pytest tests/contracts/test_provenance_registry_client.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'prsm.economy.web3.provenance_registry'`.

- [ ] **Step 3: Write the client implementation**

Create `prsm/economy/web3/provenance_registry.py`:

```python
"""ProvenanceRegistry Web3 Client.

Thin wrapper around the on-chain ProvenanceRegistry contract. Mirrors the
patterns established by prsm/economy/ftns_onchain.py — Web3.py 6.x,
synchronous calls (the rest of the on-chain stack is sync), and explicit
private-key signing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:  # pragma: no cover
    HAS_WEB3 = False
    Web3 = None  # type: ignore
    Account = None  # type: ignore

logger = logging.getLogger(__name__)

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

PROVENANCE_REGISTRY_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "uint16", "name": "royaltyRateBps", "type": "uint16"},
            {"internalType": "string", "name": "metadataUri", "type": "string"},
        ],
        "name": "registerContent",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "address", "name": "newCreator", "type": "address"},
        ],
        "name": "transferContentOwnership",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "name": "contents",
        "outputs": [
            {"internalType": "address", "name": "creator", "type": "address"},
            {"internalType": "uint16", "name": "royaltyRateBps", "type": "uint16"},
            {"internalType": "uint64", "name": "registeredAt", "type": "uint64"},
            {"internalType": "string", "name": "metadataUri", "type": "string"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "contentHash", "type": "bytes32"}],
        "name": "isRegistered",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "creator", "type": "address"},
            {"indexed": False, "internalType": "uint16", "name": "royaltyRateBps", "type": "uint16"},
            {"indexed": False, "internalType": "string", "name": "metadataUri", "type": "string"},
        ],
        "name": "ContentRegistered",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "previousCreator", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "newCreator", "type": "address"},
        ],
        "name": "OwnershipTransferred",
        "type": "event",
    },
]


@dataclass
class ContentRecord:
    creator: str
    royalty_rate_bps: int
    registered_at: int
    metadata_uri: str


class ProvenanceRegistryClient:
    """Sync Web3 client for ProvenanceRegistry."""

    def __init__(
        self,
        rpc_url: str,
        contract_address: str,
        private_key: Optional[str] = None,
    ) -> None:
        if not HAS_WEB3:
            raise RuntimeError("web3 package is required (pip install web3 eth-account)")

        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract = self.web3.eth.contract(
            address=self.contract_address,
            abi=PROVENANCE_REGISTRY_ABI,
        )

        self._account = None
        if private_key:
            self._account = Account.from_key(private_key)

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Writes ─────────────────────────────────────────────────

    def register_content(
        self,
        content_hash: bytes,
        royalty_rate_bps: int,
        metadata_uri: str,
    ) -> str:
        """Register `content_hash` with caller as creator. Returns tx hash hex."""
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        if not (0 <= royalty_rate_bps <= 10000):
            raise ValueError("royalty_rate_bps must be in [0, 10000]")

        tx = self.contract.functions.registerContent(
            content_hash, royalty_rate_bps, metadata_uri
        ).build_transaction(self._tx_overrides())
        return self._sign_and_send(tx)

    def transfer_ownership(self, content_hash: bytes, new_creator: str) -> str:
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")

        tx = self.contract.functions.transferContentOwnership(
            content_hash, Web3.to_checksum_address(new_creator)
        ).build_transaction(self._tx_overrides())
        return self._sign_and_send(tx)

    # ── Reads ──────────────────────────────────────────────────

    def get_content(self, content_hash: bytes) -> Optional[ContentRecord]:
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        creator, rate_bps, registered_at, metadata_uri = (
            self.contract.functions.contents(content_hash).call()
        )
        if creator == ZERO_ADDRESS:
            return None
        return ContentRecord(
            creator=creator,
            royalty_rate_bps=int(rate_bps),
            registered_at=int(registered_at),
            metadata_uri=metadata_uri,
        )

    def is_registered(self, content_hash: bytes) -> bool:
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        return bool(self.contract.functions.isRegistered(content_hash).call())

    # ── Internals ──────────────────────────────────────────────

    def _tx_overrides(self) -> dict:
        return {
            "from": self._account.address,
            "nonce": self.web3.eth.get_transaction_count(self._account.address),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
        }

    def _sign_and_send(self, tx: dict) -> str:
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            raise RuntimeError(f"Transaction failed: 0x{tx_hash.hex()}")
        return "0x" + tx_hash.hex()
```

- [ ] **Step 4: Make sure `tests/contracts/__init__.py` exists**

Check, and create as empty file if absent:

```bash
ls /Users/ryneschultz/Documents/GitHub/PRSM/tests/contracts/__init__.py 2>/dev/null || touch /Users/ryneschultz/Documents/GitHub/PRSM/tests/contracts/__init__.py
```

(Use Bash tool with the command above as a single command — `touch` is OK here as a one-shot file create.)

- [ ] **Step 5: Run the unit tests**

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM && python -m pytest tests/contracts/test_provenance_registry_client.py -v
```

Expected: All 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add prsm/economy/web3/provenance_registry.py tests/contracts/test_provenance_registry_client.py tests/contracts/__init__.py
git commit -m "feat(economy): add ProvenanceRegistryClient Web3 wrapper"
```

---

## Task 5: Python Web3 Client — RoyaltyDistributorClient

**Files:**
- Create: `prsm/economy/web3/royalty_distributor.py`
- Create: `tests/contracts/test_royalty_distributor_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/contracts/test_royalty_distributor_client.py`:

```python
"""Unit tests for RoyaltyDistributorClient (mocked Web3)."""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.royalty_distributor import (
    RoyaltyDistributorClient,
    SplitPreview,
)


def _hash(s: str) -> bytes:
    return hashlib.sha3_256(s.encode()).digest()


@pytest.fixture
def mock_web3():
    with patch("prsm.economy.web3.royalty_distributor.Web3") as MockWeb3:
        w3 = MagicMock()
        MockWeb3.return_value = w3
        MockWeb3.HTTPProvider.return_value = MagicMock()
        MockWeb3.to_checksum_address.side_effect = lambda x: x
        yield w3, MockWeb3


def _make_client(mock_web3):
    w3, _ = mock_web3
    distributor_contract = MagicMock()
    token_contract = MagicMock()
    # eth.contract is called twice — first for distributor, then for token
    w3.eth.contract.side_effect = [distributor_contract, token_contract]

    account = MagicMock()
    account.address = "0xPayer"
    with patch(
        "prsm.economy.web3.royalty_distributor.Account.from_key",
        return_value=account,
    ):
        client = RoyaltyDistributorClient(
            rpc_url="http://localhost:8545",
            distributor_address="0xDistributor",
            ftns_token_address="0xFTNS",
            private_key="0x" + "22" * 32,
        )
    return client, distributor_contract, token_contract, w3


def test_preview_returns_split(mock_web3):
    client, distributor, token, w3 = _make_client(mock_web3)
    distributor.functions.preview.return_value.call.return_value = (8, 2, 90)
    preview = client.preview_split(_hash("x"), 100)
    assert isinstance(preview, SplitPreview)
    assert preview.creator_amount == 8
    assert preview.network_amount == 2
    assert preview.serving_node_amount == 90


def test_distribute_royalty_approves_then_distributes(mock_web3):
    client, distributor, token, w3 = _make_client(mock_web3)

    # current allowance is 0 → approval needed
    token.functions.allowance.return_value.call.return_value = 0
    token.functions.approve.return_value.build_transaction.return_value = {
        "to": "0xFTNS", "data": "0x", "gas": 60000, "gasPrice": 1, "nonce": 0, "chainId": 8453,
    }
    distributor.functions.distributeRoyalty.return_value.build_transaction.return_value = {
        "to": "0xDistributor", "data": "0x", "gas": 200000, "gasPrice": 1, "nonce": 1, "chainId": 8453,
    }

    w3.eth.get_transaction_count.side_effect = [0, 1]
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453

    signed = MagicMock()
    signed.raw_transaction = b"raw"
    w3.eth.account.sign_transaction.return_value = signed
    w3.eth.send_raw_transaction.side_effect = [b"\xa1" * 32, b"\xa2" * 32]
    w3.eth.wait_for_transaction_receipt.return_value = MagicMock(status=1)

    tx_hash = client.distribute_royalty(_hash("c"), "0xNode", gross=100)

    token.functions.approve.assert_called_once()
    distributor.functions.distributeRoyalty.assert_called_once_with(_hash("c"), "0xNode", 100)
    assert tx_hash.startswith("0x")


def test_distribute_royalty_skips_approval_when_allowance_sufficient(mock_web3):
    client, distributor, token, w3 = _make_client(mock_web3)
    token.functions.allowance.return_value.call.return_value = 10**30  # huge
    distributor.functions.distributeRoyalty.return_value.build_transaction.return_value = {
        "to": "0xDistributor", "data": "0x", "gas": 200000, "gasPrice": 1, "nonce": 0, "chainId": 8453,
    }

    w3.eth.get_transaction_count.return_value = 0
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453

    signed = MagicMock()
    signed.raw_transaction = b"raw"
    w3.eth.account.sign_transaction.return_value = signed
    w3.eth.send_raw_transaction.return_value = b"\xb1" * 32
    w3.eth.wait_for_transaction_receipt.return_value = MagicMock(status=1)

    client.distribute_royalty(_hash("c"), "0xNode", gross=100)

    token.functions.approve.assert_not_called()


def test_distribute_royalty_rejects_zero_gross(mock_web3):
    client, distributor, token, w3 = _make_client(mock_web3)
    with pytest.raises(ValueError, match="gross"):
        client.distribute_royalty(_hash("c"), "0xNode", gross=0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM && python -m pytest tests/contracts/test_royalty_distributor_client.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Write the client**

Create `prsm/economy/web3/royalty_distributor.py`:

```python
"""RoyaltyDistributor Web3 Client.

Wraps the RoyaltyDistributor contract and the FTNS ERC-20 approval flow:
when a payer wants to distribute `gross` FTNS for content X, this client
first checks the existing allowance and only sends an `approve` tx when
needed, then calls `distributeRoyalty`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:  # pragma: no cover
    HAS_WEB3 = False
    Web3 = None  # type: ignore
    Account = None  # type: ignore

logger = logging.getLogger(__name__)


ROYALTY_DISTRIBUTOR_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "address", "name": "servingNode", "type": "address"},
            {"internalType": "uint256", "name": "gross", "type": "uint256"},
        ],
        "name": "distributeRoyalty",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "uint256", "name": "gross", "type": "uint256"},
        ],
        "name": "preview",
        "outputs": [
            {"internalType": "uint256", "name": "creatorAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "networkAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "servingNodeAmount", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "payer", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "creator", "type": "address"},
            {"indexed": False, "internalType": "address", "name": "servingNode", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "creatorAmount", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "networkAmount", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "servingNodeAmount", "type": "uint256"},
        ],
        "name": "RoyaltyPaid",
        "type": "event",
    },
]

# Minimal ERC-20 ABI subset we need (allowance + approve)
_ERC20_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "owner", "type": "address"},
            {"internalType": "address", "name": "spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "spender", "type": "address"},
            {"internalType": "uint256", "name": "value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


@dataclass
class SplitPreview:
    creator_amount: int
    network_amount: int
    serving_node_amount: int


class RoyaltyDistributorClient:
    """Sync Web3 client for RoyaltyDistributor + FTNS approval flow."""

    def __init__(
        self,
        rpc_url: str,
        distributor_address: str,
        ftns_token_address: str,
        private_key: Optional[str] = None,
    ) -> None:
        if not HAS_WEB3:
            raise RuntimeError("web3 package required")

        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.distributor_address = Web3.to_checksum_address(distributor_address)
        self.ftns_address = Web3.to_checksum_address(ftns_token_address)

        self.distributor = self.web3.eth.contract(
            address=self.distributor_address, abi=ROYALTY_DISTRIBUTOR_ABI
        )
        self.token = self.web3.eth.contract(
            address=self.ftns_address, abi=_ERC20_ABI
        )

        self._account = None
        if private_key:
            self._account = Account.from_key(private_key)

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Reads ──────────────────────────────────────────────────

    def preview_split(self, content_hash: bytes, gross: int) -> SplitPreview:
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        creator_amt, network_amt, node_amt = (
            self.distributor.functions.preview(content_hash, gross).call()
        )
        return SplitPreview(
            creator_amount=int(creator_amt),
            network_amount=int(network_amt),
            serving_node_amount=int(node_amt),
        )

    def allowance(self) -> int:
        if not self._account:
            raise RuntimeError("private_key required")
        return int(
            self.token.functions.allowance(
                self._account.address, self.distributor_address
            ).call()
        )

    # ── Writes ─────────────────────────────────────────────────

    def distribute_royalty(
        self, content_hash: bytes, serving_node: str, gross: int
    ) -> str:
        if not self._account:
            raise RuntimeError("private_key required")
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        if gross <= 0:
            raise ValueError("gross must be positive")

        # Approve only if needed
        current_allowance = int(
            self.token.functions.allowance(
                self._account.address, self.distributor_address
            ).call()
        )
        if current_allowance < gross:
            approve_tx = self.token.functions.approve(
                self.distributor_address, gross
            ).build_transaction(self._tx_overrides())
            self._sign_and_send(approve_tx)

        tx = self.distributor.functions.distributeRoyalty(
            content_hash, Web3.to_checksum_address(serving_node), gross
        ).build_transaction(self._tx_overrides())
        return self._sign_and_send(tx)

    # ── Internals ──────────────────────────────────────────────

    def _tx_overrides(self) -> dict:
        return {
            "from": self._account.address,
            "nonce": self.web3.eth.get_transaction_count(self._account.address),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
        }

    def _sign_and_send(self, tx: dict) -> str:
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            raise RuntimeError(f"Transaction failed: 0x{tx_hash.hex()}")
        return "0x" + tx_hash.hex()
```

- [ ] **Step 4: Run unit tests**

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM && python -m pytest tests/contracts/test_royalty_distributor_client.py -v
```

Expected: All 4 tests pass.

- [ ] **Step 5: Update `prsm/economy/web3/__init__.py` to export new clients**

Open `prsm/economy/web3/__init__.py` and add at the bottom (preserve existing exports):

```python
from prsm.economy.web3.provenance_registry import (
    ProvenanceRegistryClient,
    ContentRecord,
)
from prsm.economy.web3.royalty_distributor import (
    RoyaltyDistributorClient,
    SplitPreview,
)

__all__ = [
    *globals().get("__all__", []),
    "ProvenanceRegistryClient",
    "ContentRecord",
    "RoyaltyDistributorClient",
    "SplitPreview",
]
```

- [ ] **Step 6: Commit**

```bash
git add prsm/economy/web3/royalty_distributor.py prsm/economy/web3/__init__.py tests/contracts/test_royalty_distributor_client.py
git commit -m "feat(economy): add RoyaltyDistributorClient with allowance-aware distribute flow"
```

---

## Task 6: Wire Clients into `content_economy.py` Behind Feature Flag

**Files:**
- Modify: `prsm/node/content_economy.py`

**Goal:** When env var `PRSM_ONCHAIN_PROVENANCE=1`, content uploads call `ProvenanceRegistryClient.register_content` and content access payments route through `RoyaltyDistributorClient.distribute_royalty` instead of the local-ledger split. When the flag is unset, the existing path is unchanged.

- [ ] **Step 1: Read the existing payment hook in content_economy.py**

Open `prsm/node/content_economy.py` and locate the function(s) that perform the royalty split — search for `ORIGINAL_CREATOR_ROYALTY_RATE` and `NETWORK_FEE_RATE`. Identify the single function that currently distributes payment (likely something like `_distribute_royalties`, `_pay_creator`, or `_handle_content_payment`). Note the exact name and its call sites.

- [ ] **Step 2: Add the feature flag scaffolding at the top of the file**

Below the existing `ROYALTY_MODEL` enum and constants, add:

```python
import os

# ── On-chain provenance feature flag (Phase 1) ────────────────────────
ONCHAIN_PROVENANCE_ENABLED = os.getenv("PRSM_ONCHAIN_PROVENANCE", "").lower() in ("1", "true", "yes")
PROVENANCE_REGISTRY_ADDRESS = os.getenv("PRSM_PROVENANCE_REGISTRY_ADDRESS", "")
ROYALTY_DISTRIBUTOR_ADDRESS = os.getenv("PRSM_ROYALTY_DISTRIBUTOR_ADDRESS", "")
```

- [ ] **Step 3: Add lazy on-chain client construction**

Inside the main `ContentEconomy` (or equivalent) class — its name is whatever owns the royalty distribution method you found in Step 1 — add:

```python
def _get_provenance_client(self):
    """Lazy-init ProvenanceRegistryClient. Returns None if disabled or unconfigured."""
    if not ONCHAIN_PROVENANCE_ENABLED:
        return None
    if not PROVENANCE_REGISTRY_ADDRESS:
        logger.warning("PRSM_ONCHAIN_PROVENANCE=1 but PRSM_PROVENANCE_REGISTRY_ADDRESS not set")
        return None
    if getattr(self, "_provenance_client", None) is None:
        from prsm.economy.web3.provenance_registry import ProvenanceRegistryClient
        rpc_url = os.getenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
        pk = os.getenv("FTNS_WALLET_PRIVATE_KEY")
        self._provenance_client = ProvenanceRegistryClient(
            rpc_url=rpc_url,
            contract_address=PROVENANCE_REGISTRY_ADDRESS,
            private_key=pk,
        )
    return self._provenance_client

def _get_royalty_distributor(self):
    """Lazy-init RoyaltyDistributorClient. Returns None if disabled or unconfigured."""
    if not ONCHAIN_PROVENANCE_ENABLED:
        return None
    if not ROYALTY_DISTRIBUTOR_ADDRESS:
        logger.warning("PRSM_ONCHAIN_PROVENANCE=1 but PRSM_ROYALTY_DISTRIBUTOR_ADDRESS not set")
        return None
    if getattr(self, "_royalty_client", None) is None:
        from prsm.economy.web3.royalty_distributor import RoyaltyDistributorClient
        rpc_url = os.getenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
        pk = os.getenv("FTNS_WALLET_PRIVATE_KEY")
        ftns_addr = os.getenv("FTNS_TOKEN_ADDRESS", "0x5276a3756C85f2E9e46f6D34386167a209aa16e5")
        self._royalty_client = RoyaltyDistributorClient(
            rpc_url=rpc_url,
            distributor_address=ROYALTY_DISTRIBUTOR_ADDRESS,
            ftns_token_address=ftns_addr,
            private_key=pk,
        )
    return self._royalty_client
```

- [ ] **Step 4: Branch the existing royalty distribution function**

In the function identified in Step 1, at the very start of the body (before the existing local-ledger logic), insert:

```python
# Phase 1: route through on-chain RoyaltyDistributor when enabled.
distributor = self._get_royalty_distributor()
if distributor is not None:
    try:
        content_hash_bytes = bytes.fromhex(content_hash) if isinstance(content_hash, str) else content_hash
        if len(content_hash_bytes) != 32:
            # Hash content_hash to 32 bytes if it's not already
            import hashlib
            content_hash_bytes = hashlib.sha3_256(content_hash_bytes).digest()
        gross_wei = int(Decimal(str(amount_ftns)) * Decimal(10**18))
        tx_hash = distributor.distribute_royalty(
            content_hash=content_hash_bytes,
            serving_node=serving_node_address,
            gross=gross_wei,
        )
        logger.info(
            f"on-chain royalty distributed: content={content_hash_bytes.hex()[:12]}… "
            f"gross={amount_ftns} tx={tx_hash[:16]}…"
        )
        return  # On-chain path completed; skip local fallback
    except Exception as exc:
        logger.error(f"on-chain royalty distribution failed, falling back to local: {exc}")
        # Intentional fall-through to local path so we never lose a payment
```

> **Notes for the implementer:**
> - `amount_ftns`, `content_hash`, and `serving_node_address` are placeholder names. Use the variable names that exist in the function you're modifying. If the existing function takes a different shape (e.g. it splits inside the function instead of being given a single gross), adapt the call accordingly — the goal is: pass the *gross* amount and let the contract do the split.
> - The fall-through to the existing local logic is intentional: until we have full confidence in the on-chain path, a chain failure must not drop the payment.

- [ ] **Step 5: Add a content registration hook on first upload**

Search `content_economy.py` for the function that handles content uploads (likely `_handle_upload`, `register_content`, `add_content`, or similar). At the end of that function, after the existing local registration succeeds, add:

```python
# Phase 1: mirror to on-chain provenance registry when enabled.
provenance = self._get_provenance_client()
if provenance is not None:
    try:
        import hashlib
        content_hash_bytes = bytes.fromhex(content_hash) if isinstance(content_hash, str) else content_hash
        if len(content_hash_bytes) != 32:
            content_hash_bytes = hashlib.sha3_256(content_hash_bytes).digest()
        if not provenance.is_registered(content_hash_bytes):
            royalty_bps = int(ORIGINAL_CREATOR_ROYALTY_RATE * 10000)  # 0.08 → 800
            tx = provenance.register_content(
                content_hash=content_hash_bytes,
                royalty_rate_bps=royalty_bps,
                metadata_uri=metadata_uri or "",
            )
            logger.info(
                f"on-chain provenance registered: content={content_hash_bytes.hex()[:12]}… tx={tx[:16]}…"
            )
    except Exception as exc:
        logger.error(f"on-chain provenance registration failed (non-fatal): {exc}")
```

> Same caveat: variable names depend on the exact function found.

- [ ] **Step 6: Run the existing content_economy unit tests to confirm no regression with the flag OFF**

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM && python -m pytest tests/unit/test_content_economy*.py -v 2>&1 | tail -50
```

Expected: All previously passing tests still pass (since `PRSM_ONCHAIN_PROVENANCE` is unset, the new branches are skipped).

If no `test_content_economy*` tests exist, run the broader `tests/unit/test_royalty*.py`:

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM && python -m pytest tests/unit/ -k "royalty or content_economy or provenance" -v 2>&1 | tail -50
```

Expected: pre-existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add prsm/node/content_economy.py
git commit -m "feat(node): wire on-chain provenance + royalty distributor behind PRSM_ONCHAIN_PROVENANCE flag"
```

---

## Task 7: CLI Subcommands `prsm provenance register|info|transfer`

**Files:**
- Create: `prsm/cli/provenance.py`
- Modify: `prsm/cli/__init__.py` (or wherever the click group is registered)

- [ ] **Step 1: Inspect existing CLI structure**

Run:

```bash
ls /Users/ryneschultz/Documents/GitHub/PRSM/prsm/cli/ 2>/dev/null
```

Then read whichever file defines the top-level `click` group (likely `prsm/cli/__init__.py` or `prsm/cli/main.py`). Note how other subcommand groups are added (e.g., `cli.add_command(...)`).

- [ ] **Step 2: Write the subcommand module**

Create `prsm/cli/provenance.py`:

```python
"""CLI: prsm provenance register|info|transfer

Phase 1 on-chain provenance commands. Requires:
  PRSM_PROVENANCE_REGISTRY_ADDRESS
  PRSM_BASE_RPC_URL              (defaults to https://mainnet.base.org)
  FTNS_WALLET_PRIVATE_KEY        (for register/transfer; not for info)
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import click


def _content_hash_for(path: Path) -> bytes:
    """Canonical content hash: keccak-256 of file bytes (matches contract bytes32)."""
    h = hashlib.sha3_256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.digest()


def _make_client():
    from prsm.economy.web3.provenance_registry import ProvenanceRegistryClient

    addr = os.getenv("PRSM_PROVENANCE_REGISTRY_ADDRESS")
    if not addr:
        click.echo("error: PRSM_PROVENANCE_REGISTRY_ADDRESS not set", err=True)
        sys.exit(1)
    rpc = os.getenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
    pk = os.getenv("FTNS_WALLET_PRIVATE_KEY")
    return ProvenanceRegistryClient(rpc_url=rpc, contract_address=addr, private_key=pk)


@click.group("provenance")
def provenance() -> None:
    """On-chain provenance registry commands."""


@provenance.command("register")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--royalty-bps",
    type=click.IntRange(0, 10000),
    default=800,
    show_default=True,
    help="Royalty rate in basis points (800 = 8%).",
)
@click.option(
    "--metadata-uri",
    type=str,
    default="",
    help="Off-chain metadata URI (ipfs://, https://, etc.).",
)
def register(file: Path, royalty_bps: int, metadata_uri: str) -> None:
    """Register FILE in the on-chain provenance registry."""
    client = _make_client()
    if client.address is None:
        click.echo("error: FTNS_WALLET_PRIVATE_KEY required to register", err=True)
        sys.exit(1)

    content_hash = _content_hash_for(file)
    click.echo(f"file:        {file}")
    click.echo(f"hash:        0x{content_hash.hex()}")
    click.echo(f"creator:     {client.address}")
    click.echo(f"royalty:     {royalty_bps} bps ({royalty_bps / 100:.2f}%)")
    click.echo(f"metadata:    {metadata_uri or '(none)'}")

    if client.is_registered(content_hash):
        click.echo("note: already registered, no-op", err=True)
        sys.exit(0)

    tx = client.register_content(content_hash, royalty_bps, metadata_uri)
    click.echo(f"tx:          {tx}")


@provenance.command("info")
@click.argument("hash_or_file")
def info(hash_or_file: str) -> None:
    """Show provenance record for a content hash (0x… 32 bytes) or file path."""
    client = _make_client()

    if hash_or_file.startswith("0x") and len(hash_or_file) == 66:
        content_hash = bytes.fromhex(hash_or_file[2:])
    else:
        p = Path(hash_or_file)
        if not p.exists():
            click.echo(f"error: not a hash and not an existing file: {hash_or_file}", err=True)
            sys.exit(1)
        content_hash = _content_hash_for(p)

    rec = client.get_content(content_hash)
    if rec is None:
        click.echo(f"hash 0x{content_hash.hex()} is NOT registered")
        sys.exit(0)

    click.echo(f"hash:        0x{content_hash.hex()}")
    click.echo(f"creator:     {rec.creator}")
    click.echo(f"royalty:     {rec.royalty_rate_bps} bps ({rec.royalty_rate_bps / 100:.2f}%)")
    click.echo(f"registered:  {rec.registered_at} (unix)")
    click.echo(f"metadata:    {rec.metadata_uri or '(none)'}")


@provenance.command("transfer")
@click.argument("content_hash")
@click.argument("new_creator")
def transfer(content_hash: str, new_creator: str) -> None:
    """Transfer creator role for CONTENT_HASH (0x…) to NEW_CREATOR (0x…)."""
    client = _make_client()
    if client.address is None:
        click.echo("error: FTNS_WALLET_PRIVATE_KEY required to transfer", err=True)
        sys.exit(1)

    if not content_hash.startswith("0x") or len(content_hash) != 66:
        click.echo("error: content_hash must be 0x followed by 64 hex chars", err=True)
        sys.exit(1)
    if not new_creator.startswith("0x") or len(new_creator) != 42:
        click.echo("error: new_creator must be a 0x address", err=True)
        sys.exit(1)

    tx = client.transfer_ownership(bytes.fromhex(content_hash[2:]), new_creator)
    click.echo(f"tx: {tx}")
```

- [ ] **Step 3: Register the subcommand group**

Open the file from Step 1 (the file that defines the root `cli` click group). After the existing `cli.add_command(...)` calls, add:

```python
from prsm.cli.provenance import provenance as provenance_cmd
cli.add_command(provenance_cmd)
```

(If the root group is named differently — e.g. `prsm` or `app` — use that name instead of `cli`.)

- [ ] **Step 4: Smoke test the help output**

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM && python -m prsm.cli provenance --help
```

Expected: shows the three subcommands `register`, `info`, `transfer`.

If the entry point is different (e.g. `python -m prsm` or a `prsm` script), adapt the command. Find the right entry point with:

```bash
grep -r "click.group" /Users/ryneschultz/Documents/GitHub/PRSM/prsm/cli/ 2>/dev/null | head -5
```

- [ ] **Step 5: Commit**

```bash
git add prsm/cli/provenance.py prsm/cli/__init__.py
git commit -m "feat(cli): add prsm provenance register/info/transfer subcommands"
```

---

## Task 8: End-to-End Integration Test on a Local Hardhat Node

**Files:**
- Create: `tests/integration/test_onchain_provenance_e2e.py`

This test boots a local Hardhat node (`npx hardhat node`), deploys a MockERC20 + ProvenanceRegistry + RoyaltyDistributor, then exercises the full Python client flow against that real local chain. It validates: register → preview → approve → distribute → assert balances + event.

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_onchain_provenance_e2e.py`:

```python
"""End-to-end integration test for Phase 1 on-chain provenance.

Boots a local Hardhat node, deploys MockERC20 + ProvenanceRegistry +
RoyaltyDistributor, exercises the full Python client flow, and asserts
on-chain state matches expectations.

Skipped automatically if `npx hardhat` is not on PATH.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "contracts"


def _hardhat_available() -> bool:
    return shutil.which("npx") is not None and (CONTRACTS_DIR / "node_modules").exists()


pytestmark = pytest.mark.skipif(
    not _hardhat_available(), reason="Hardhat / npx not available"
)


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect((host, port))
                return
            except OSError:
                time.sleep(0.3)
    raise TimeoutError(f"timed out waiting for {host}:{port}")


@pytest.fixture(scope="module")
def hardhat_node():
    """Start a local hardhat node for the duration of the module."""
    proc = subprocess.Popen(
        ["npx", "hardhat", "node", "--port", "8545"],
        cwd=str(CONTRACTS_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_port("127.0.0.1", 8545, timeout=60.0)
        yield "http://127.0.0.1:8545"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def deployed(hardhat_node):
    """Compile + deploy the three contracts using a small inline JS deploy script."""
    deploy_js = """
const hre = require("hardhat");
const fs = require("fs");

async function main() {
  const [deployer, treasury] = await hre.ethers.getSigners();

  const Token = await hre.ethers.getContractFactory("MockERC20");
  const token = await Token.deploy();
  await token.waitForDeployment();

  const Registry = await hre.ethers.getContractFactory("ProvenanceRegistry");
  const registry = await Registry.deploy();
  await registry.waitForDeployment();

  const Distributor = await hre.ethers.getContractFactory("RoyaltyDistributor");
  const distributor = await Distributor.deploy(
    await token.getAddress(),
    await registry.getAddress(),
    treasury.address
  );
  await distributor.waitForDeployment();

  // Mint some tokens to deployer for the test
  await token.mint(deployer.address, hre.ethers.parseUnits("10000", 18));

  const out = {
    token: await token.getAddress(),
    registry: await registry.getAddress(),
    distributor: await distributor.getAddress(),
    treasury: treasury.address,
    deployer: deployer.address,
    deployerKey: "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",  // Hardhat default account #0
  };
  fs.writeFileSync("/tmp/prsm_e2e_deploy.json", JSON.stringify(out));
}
main().catch((e) => { console.error(e); process.exit(1); });
"""
    script = CONTRACTS_DIR / "scripts" / "_e2e_deploy.js"
    script.write_text(deploy_js)
    try:
        subprocess.run(
            ["npx", "hardhat", "compile"],
            cwd=str(CONTRACTS_DIR),
            check=True,
        )
        subprocess.run(
            ["npx", "hardhat", "run", "scripts/_e2e_deploy.js", "--network", "localhost"],
            cwd=str(CONTRACTS_DIR),
            check=True,
        )
        with open("/tmp/prsm_e2e_deploy.json") as f:
            return json.load(f)
    finally:
        if script.exists():
            script.unlink()


def _content_hash(s: str) -> bytes:
    return hashlib.sha3_256(s.encode()).digest()


def test_register_then_distribute_e2e(hardhat_node, deployed):
    from prsm.economy.web3.provenance_registry import ProvenanceRegistryClient
    from prsm.economy.web3.royalty_distributor import RoyaltyDistributorClient

    registry_client = ProvenanceRegistryClient(
        rpc_url=hardhat_node,
        contract_address=deployed["registry"],
        private_key=deployed["deployerKey"],
    )

    distributor_client = RoyaltyDistributorClient(
        rpc_url=hardhat_node,
        distributor_address=deployed["distributor"],
        ftns_token_address=deployed["token"],
        private_key=deployed["deployerKey"],
    )

    content_hash = _content_hash("e2e-test-content")

    # 1. Register
    assert not registry_client.is_registered(content_hash)
    tx_register = registry_client.register_content(
        content_hash, royalty_rate_bps=800, metadata_uri="ipfs://e2e"
    )
    assert tx_register.startswith("0x")
    assert registry_client.is_registered(content_hash)

    record = registry_client.get_content(content_hash)
    assert record is not None
    assert record.royalty_rate_bps == 800
    assert record.metadata_uri == "ipfs://e2e"
    assert record.creator.lower() == deployed["deployer"].lower()

    # 2. Preview
    gross_wei = 100 * 10**18  # 100 FTNS
    preview = distributor_client.preview_split(content_hash, gross_wei)
    assert preview.creator_amount == 8 * 10**18
    assert preview.network_amount == 2 * 10**18
    assert preview.serving_node_amount == 90 * 10**18

    # 3. Distribute. The deployer is also payer; serving node = a fresh address.
    serving_node = "0x000000000000000000000000000000000000dEaD"
    tx_dist = distributor_client.distribute_royalty(
        content_hash, serving_node, gross_wei
    )
    assert tx_dist.startswith("0x")

    # 4. Verify balances on-chain via direct token contract reads
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(hardhat_node))
    erc20_abi = [
        {
            "inputs": [{"name": "owner", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        }
    ]
    token_contract = w3.eth.contract(
        address=Web3.to_checksum_address(deployed["token"]), abi=erc20_abi
    )
    treasury_balance = token_contract.functions.balanceOf(
        Web3.to_checksum_address(deployed["treasury"])
    ).call()
    serving_balance = token_contract.functions.balanceOf(
        Web3.to_checksum_address(serving_node)
    ).call()
    creator_balance = token_contract.functions.balanceOf(
        Web3.to_checksum_address(deployed["deployer"])
    ).call()

    assert treasury_balance == 2 * 10**18
    assert serving_balance == 90 * 10**18
    # Creator started with 10_000 FTNS, paid 100 (gross), received 8 back
    # Net: 10_000 - 100 + 8 = 9908
    assert creator_balance == (10_000 - 100 + 8) * 10**18
```

- [ ] **Step 2: Run the integration test**

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM && python -m pytest tests/integration/test_onchain_provenance_e2e.py -v -s
```

Expected: test passes. Hardhat node spins up, deploys, the Python clients exercise the full register → preview → distribute flow, and balances verify.

If `npx hardhat` is not installed, the test will skip — that's OK in CI without contracts toolchain. To fully validate, run:

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM/contracts && npm install && cd .. && python -m pytest tests/integration/test_onchain_provenance_e2e.py -v -s
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_onchain_provenance_e2e.py
git commit -m "test(integration): end-to-end on-chain provenance flow against local hardhat node"
```

---

## Task 9: Documentation + CHANGELOG Update

**Files:**
- Create: `docs/ONCHAIN_PROVENANCE.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the user-facing doc**

Create `docs/ONCHAIN_PROVENANCE.md`:

```markdown
# On-Chain Provenance & Royalty Distribution (Phase 1)

> Status: shipped in v1.8.0, opt-in via feature flag.

## What this is

PRSM v1.8.0 adds two smart contracts on Base mainnet that move royalty
distribution from the local node ledger to the chain:

- **ProvenanceRegistry** — maps a 32-byte content hash to a creator address
  and a royalty rate (basis points). Anyone can register content; the
  registrant becomes the creator and earns royalties on use.
- **RoyaltyDistributor** — atomic three-way splitter. Pulls FTNS from a
  payer, looks up the creator and royalty rate from the registry, and
  splits the gross amount into creator / network treasury / serving node
  shares.

## Why

Before v1.8.0, royalty distribution was computed in a local SQLite ledger
on each node. Trust depended on the operator. With v1.8.0, anyone can
audit royalty payments by reading Base mainnet event logs.

## Splits

For a `gross` payment on content with rate `R` basis points:

| Recipient | Amount |
|---|---|
| Creator | `gross * R / 10000` |
| Network treasury | `gross * 200 / 10000` (2% fixed) |
| Serving node | `gross - creator - network` |

The contract reverts if `R + 200 > 10000`. The maximum royalty rate is
9800 bps (98%) so the network fee always fits.

## Enabling on a node

Set these env vars before starting your node:

```bash
export PRSM_ONCHAIN_PROVENANCE=1
export PRSM_PROVENANCE_REGISTRY_ADDRESS=0x…   # from contracts/deployments/
export PRSM_ROYALTY_DISTRIBUTOR_ADDRESS=0x…   # from contracts/deployments/
export PRSM_BASE_RPC_URL=https://mainnet.base.org
export FTNS_TOKEN_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5
export FTNS_WALLET_PRIVATE_KEY=0x…            # creator/payer wallet
```

When the flag is set, content uploads automatically register on-chain
and royalty payments route through `RoyaltyDistributor`. Local-ledger
fallback remains active so payments are never lost during a chain
outage.

## CLI commands

```bash
# Register a file
prsm provenance register ./mydata.csv --royalty-bps 800 --metadata-uri ipfs://Qm…

# Look up a record
prsm provenance info 0x<64 hex chars>
prsm provenance info ./mydata.csv

# Transfer ownership
prsm provenance transfer 0x<hash> 0x<new_creator>
```

## Verifying a royalty payment on Basescan

1. Find the `RoyaltyPaid` event on the RoyaltyDistributor contract.
2. The `contentHash` topic identifies which piece of content was paid.
3. The `creator` topic confirms who received the royalty.
4. The `creatorAmount`, `networkAmount`, and `servingNodeAmount` data
   fields show the exact split.
5. Cross-reference the `creator` address against
   `ProvenanceRegistry.contents(contentHash)` — they must match.

## Limitations (Phase 1)

- Only original creator + serving node + treasury — derivative chains are
  not yet supported on-chain.
- Approval flow is plain ERC-20 `approve` + `transferFrom` (no permit).
  Each new payer pays one extra approve transaction the first time.
- Content addressed by `bytes32` only — no NFT wrapper.

These will be addressed in subsequent phases.
```

- [ ] **Step 2: Update the CHANGELOG**

Open `CHANGELOG.md`. Add a new top entry above the most recent release:

```markdown
## [Unreleased] — Phase 1: On-Chain Provenance

### Added
- `contracts/contracts/ProvenanceRegistry.sol` — on-chain content provenance registry. Maps content hash to creator address and royalty rate.
- `contracts/contracts/RoyaltyDistributor.sol` — atomic three-way FTNS splitter (creator / network treasury / serving node).
- `prsm/economy/web3/provenance_registry.py` — Python Web3 client for the registry.
- `prsm/economy/web3/royalty_distributor.py` — Python Web3 client for the distributor with allowance-aware approval flow.
- `prsm provenance register|info|transfer` CLI subcommands.
- Feature flag `PRSM_ONCHAIN_PROVENANCE=1` to opt-in `prsm/node/content_economy.py` to on-chain royalty routing.
- End-to-end integration test against a local Hardhat node.
- `docs/ONCHAIN_PROVENANCE.md` — user-facing documentation.

### Why
Closes Phase 1 of the audit-gap remediation roadmap (`docs/2026-04-10-audit-gap-roadmap.md`). Moves royalty distribution off the local SQLite ledger and onto Base mainnet so anyone can independently verify creator earnings.

### Notes
- Feature is opt-in. With the flag unset, behavior is unchanged.
- Local-ledger fallback remains in place: if an on-chain distribution fails, the local path still records the payment.
- Live deployment to Base Sepolia and Base mainnet is the final step in Task 10 (manual, gated on operator review).
```

- [ ] **Step 3: Commit**

```bash
git add docs/ONCHAIN_PROVENANCE.md CHANGELOG.md
git commit -m "docs(phase1): on-chain provenance user docs and changelog entry"
```

---

## Task 10: Live Deployment to Base Sepolia (manual, operator-gated)

**This task is intentionally manual.** Mainnet/testnet deployments must be operator-reviewed. The agent should pause here and prompt the operator to perform these steps, then run the smoke test below afterward.

- [ ] **Step 1: Operator: confirm prerequisites**

Operator must have:
- `PRIVATE_KEY` env var set to a Base Sepolia funded wallet
- `BASE_SEPOLIA_RPC_URL` set (e.g. https://sepolia.base.org)
- `ETHERSCAN_API_KEY` set
- A small amount of Sepolia ETH on the deployer address (~0.01 ETH)
- A test FTNS token deployed on Base Sepolia (or use a standard test ERC-20 — record its address as `FTNS_TOKEN_ADDRESS`)

- [ ] **Step 2: Operator: deploy**

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM/contracts
FTNS_TOKEN_ADDRESS=0x… NETWORK_TREASURY=0x… npx hardhat run scripts/deploy-provenance.js --network base-sepolia
```

Expected output: prints both contract addresses, writes a manifest to `contracts/deployments/provenance-base-sepolia-<timestamp>.json`.

- [ ] **Step 3: Operator: verify on Basescan**

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM/contracts
npx hardhat verify --network base-sepolia <REGISTRY_ADDRESS>
npx hardhat verify --network base-sepolia <DISTRIBUTOR_ADDRESS> <FTNS_TOKEN_ADDRESS> <REGISTRY_ADDRESS> <NETWORK_TREASURY>
```

- [ ] **Step 4: Operator: smoke test the live deployment**

```bash
export PRSM_PROVENANCE_REGISTRY_ADDRESS=<REGISTRY_ADDRESS>
export PRSM_BASE_RPC_URL=https://sepolia.base.org
export FTNS_WALLET_PRIVATE_KEY=$PRIVATE_KEY

prsm provenance register ./README.md --royalty-bps 800 --metadata-uri "ipfs://test"
prsm provenance info ./README.md
```

Expected: register prints a tx hash; info shows creator + 800 bps + metadata.

- [ ] **Step 5: Operator: commit deployment manifest**

```bash
git add contracts/deployments/provenance-base-sepolia-*.json
git commit -m "chore(deploy): record Base Sepolia provenance contract addresses"
```

- [ ] **Step 6: Mainnet deployment**

Mainnet deployment is **not part of this plan**. After Sepolia bake-in, a separate operator decision triggers mainnet deployment using the same script with `--network base`. Document the decision and resulting addresses in a follow-up commit.

---

## Self-Review Checklist

After all tasks complete, verify:

- [ ] Both contracts compile under Solidity 0.8.22 with the existing Hardhat config.
- [ ] All Hardhat tests pass: `cd contracts && npx hardhat test`.
- [ ] All Python unit tests pass: `python -m pytest tests/contracts/ -v`.
- [ ] End-to-end integration test passes when Hardhat is available, or skips cleanly when not.
- [ ] `prsm provenance --help` shows all three subcommands.
- [ ] With `PRSM_ONCHAIN_PROVENANCE` unset, all existing `tests/unit/test_*content_economy*` and `test_*royalty*` tests still pass (no regression).
- [ ] `docs/ONCHAIN_PROVENANCE.md` exists and documents the env vars + CLI.
- [ ] `CHANGELOG.md` has the Phase 1 entry.
- [ ] No `TODO`, `FIXME`, or `pass  # implement later` markers in any of the new files.
- [ ] No new files in repo root — everything in `contracts/`, `prsm/`, `tests/`, `docs/`.
