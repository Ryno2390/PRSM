/**
 * Team B — Access Control & Ownership PoC suite.
 *
 * Each `it()` is a finding. PoCs that exit cleanly with the asserted
 * outcome demonstrate a confirmed attack/defect path. PoCs marked
 * "(defense)" prove a stated invariant holds (no exploit found).
 *
 * Scope coverage:
 *   B1  pre-handoff backdoor      — defense (no API to insert hidden grants)
 *   B2  one-step transferOwnership — confirmed brick path on typo
 *   B3  role-graph cycles          — defense (admin-of-admin chain bottoms out)
 *   B4  renounceRole reachability  — confirmed: admin can renounce + brick UUPS
 *   B5  cross-wire mutability      — confirmed: owner can re-point cross-wires
 *   B6  pauser locking             — defense (any PAUSER unpauses)
 *   B7  initializer re-entry       — defense (`_disableInitializers` in ctor)
 *   B8  slasher acceptance         — confirmed: StakeBond.setSlasher accepts EOA
 *   B9  Foundation Safe owners     — out-of-scope for hardhat (on-chain check)
 *   B10 constructor poisoning      — defense (zero-address checks)
 *
 * Plus VERIFIER-ABI: post-handoff verifier script uses wrong getter names.
 */
const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");

describe("Team B — Access Control PoC", function () {
  // ── B2 ──────────────────────────────────────────────────────────────
  describe("B2 — single-step transferOwnership brick risk", function () {
    it("CONFIRMED: typo in transferOwnership target permanently bricks Ownable contract", async function () {
      const [deployer, attacker] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const ftns = await MockERC20.deploy();

      const Pool = await ethers.getContractFactory("EscrowPool");
      const pool = await Pool.deploy(deployer.address, await ftns.getAddress(), ethers.ZeroAddress);

      // Operator typos one hex digit. This address has no signer in scope.
      // ethers v6 still validates checksum, so we use a checksummed address
      // that simply isn't controlled by anyone in this fixture.
      const typoTarget = "0x000000000000000000000000000000000000dEaD";

      // Single-step Ownable accepts any non-zero target with no acceptance step.
      await pool.transferOwnership(typoTarget);
      expect(await pool.owner()).to.equal(typoTarget);

      // Original deployer can no longer perform owner-only ops.
      await expect(
        pool.connect(deployer).setSettlementRegistry(attacker.address),
      ).to.be.revertedWithCustomError(pool, "OwnableUnauthorizedAccount");

      // Contract is permanently bricked: no one can set the registry,
      // settle from requesters, or update the FTNS token escape hatch.
      // Ownable2Step would have prevented this by requiring acceptOwnership.
    });
  });

  // ── B4 ──────────────────────────────────────────────────────────────
  describe("B4 — renounceRole reachability on FTNSTokenSimple", function () {
    it("REGRESSION: post-HIGH-5, renounceRole(DEFAULT_ADMIN_ROLE) reverts on the contract layer", async function () {
      const [admin, treasury] = await ethers.getSigners();
      const FTNS = await ethers.getContractFactory("FTNSTokenSimple");
      const ftns = await upgrades.deployProxy(
        FTNS,
        [admin.address, treasury.address],
        { initializer: "initialize", kind: "uups" },
      );

      const ADMIN = ethers.ZeroHash;
      const MINTER = ethers.id("MINTER_ROLE");

      // Admin holds DEFAULT_ADMIN_ROLE per initialize().
      expect(await ftns.hasRole(ADMIN, admin.address)).to.equal(true);

      // Post-fix: renounceRole(DEFAULT_ADMIN_ROLE) reverts. The script-
      // layer guard in transfer-ftns-roles.js is now backed by a
      // contract-layer guard, eliminating (a) script-bypass, (b) direct
      // call to renounceRole, and (c) multi-sig-self-renounce.
      await expect(
        ftns.connect(admin).renounceRole(ADMIN, admin.address),
      ).to.be.revertedWith("DEFAULT_ADMIN_ROLE renounce disabled - use grantRole(new) + revokeRole(old)");

      // Admin still holds the role; can still grant other roles.
      expect(await ftns.hasRole(ADMIN, admin.address)).to.equal(true);
      await expect(
        ftns.connect(admin).grantRole(MINTER, treasury.address),
      ).to.not.be.reverted;
    });
  });

  // ── B5 ──────────────────────────────────────────────────────────────
  describe("B5 — cross-wire mutability after handoff", function () {
    it("CONFIRMED: owner can re-point EscrowPool.settlementRegistry to attacker contract", async function () {
      const [owner, foundation, attacker] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const ftns = await MockERC20.deploy();

      const Pool = await ethers.getContractFactory("EscrowPool");
      const pool = await Pool.deploy(owner.address, await ftns.getAddress(), ethers.ZeroAddress);

      // Simulate handoff: pretend `foundation` is the multi-sig.
      await pool.transferOwnership(foundation.address);
      expect(await pool.owner()).to.equal(foundation.address);

      // A compromised multi-sig (or one signer + bribed second) can
      // re-point the registry to an attacker-controlled contract.
      await pool.connect(foundation).setSettlementRegistry(attacker.address);
      expect(await pool.settlementRegistry()).to.equal(attacker.address);

      // From this point, ANY attacker.address-initiated tx to
      // pool.settleFromRequester drains a requester's escrow to any
      // recipient. Funds previously deposited by anyone are at risk.
      const TEN = ethers.parseUnits("10", 18);
      await ftns.mint(owner.address, TEN);
      await ftns.connect(owner).approve(await pool.getAddress(), TEN);
      await pool.connect(owner).deposit(TEN);

      await pool
        .connect(attacker)
        .settleFromRequester(owner.address, attacker.address, TEN);

      expect(await ftns.balanceOf(attacker.address)).to.equal(TEN);
      expect(await pool.balances(owner.address)).to.equal(0);
    });

    it("CONFIRMED: owner can replace FTNS token via setFtnsToken, stranding all balances", async function () {
      const [owner, requester] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const realFtns = await MockERC20.deploy();
      const fakeFtns = await MockERC20.deploy();

      const Pool = await ethers.getContractFactory("EscrowPool");
      const pool = await Pool.deploy(owner.address, await realFtns.getAddress(), ethers.ZeroAddress);

      // Requester deposits 100 real FTNS.
      const HUNDRED = ethers.parseUnits("100", 18);
      await realFtns.mint(requester.address, HUNDRED);
      await realFtns.connect(requester).approve(await pool.getAddress(), HUNDRED);
      await pool.connect(requester).deposit(HUNDRED);
      expect(await pool.balances(requester.address)).to.equal(HUNDRED);

      // Owner swaps to fakeFtns. Operational policy says "drain first";
      // contract does NOT enforce.
      await pool.setFtnsToken(await fakeFtns.getAddress());

      // Real FTNS still in pool's address; pool now points at fake.
      // Requester withdrawing pulls FAKE tokens (pool has zero fake-FTNS
      // balance → underlying ERC20 revert with insufficient balance).
      // Real funds stranded.
      await expect(pool.connect(requester).withdraw(HUNDRED)).to.be.reverted;
      // Real funds still in pool address, unrecoverable through the
      // contract's interface.
      expect(await realFtns.balanceOf(await pool.getAddress())).to.equal(HUNDRED);
      expect(await fakeFtns.balanceOf(await pool.getAddress())).to.equal(0);
    });

    it("CONFIRMED: owner can re-point StakeBond.slasher to attacker, slashing arbitrary providers", async function () {
      const [owner, provider, foundation, attacker] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const ftns = await MockERC20.deploy();

      const Bond = await ethers.getContractFactory("StakeBond");
      const bond = await Bond.deploy(owner.address, await ftns.getAddress(), 7 * 24 * 60 * 60);

      // Provider stakes 10K FTNS at standard tier (50% slash).
      const STAKE = ethers.parseUnits("10000", 18);
      await ftns.mint(provider.address, STAKE);
      await ftns.connect(provider).approve(await bond.getAddress(), STAKE);
      await bond.connect(provider).bond(STAKE, 5000);

      // Owner sets slasher to attacker EOA. There is no contract-side
      // sanity check that the slasher is actually a Registry contract.
      await bond.setSlasher(attacker.address);
      expect(await bond.slasher()).to.equal(attacker.address);

      // Attacker calls slash() with arbitrary provider + claims bounty.
      const reasonId = ethers.id("fabricated-batch-id");
      await bond.connect(attacker).slash(provider.address, attacker.address, reasonId);

      // 50% of 10K = 5K slashed; attacker gets 70% bounty = 3.5K claimable.
      const bounty = await bond.slashedBountyPayable(attacker.address);
      expect(bounty).to.equal(ethers.parseUnits("3500", 18));

      // The slasher cross-wire HAS NO ALLOW-LIST and NO INTERFACE CHECK.
      // After handoff, this is a single multi-sig call away from
      // arbitrary slashing. The whole stake-bond honesty model rests on
      // multi-sig integrity at this point.
    });
  });

  // ── B6 ──────────────────────────────────────────────────────────────
  describe("B6 — pauser locking", function () {
    it("DEFENSE: any PAUSER can unpause; pause is reversible while ≥1 PAUSER exists", async function () {
      const [admin, treasury, pauserA, pauserB] = await ethers.getSigners();
      const FTNS = await ethers.getContractFactory("FTNSTokenSimple");
      const ftns = await upgrades.deployProxy(
        FTNS,
        [admin.address, treasury.address],
        { initializer: "initialize", kind: "uups" },
      );
      const PAUSER = ethers.id("PAUSER_ROLE");
      await ftns.connect(admin).grantRole(PAUSER, pauserA.address);
      await ftns.connect(admin).grantRole(PAUSER, pauserB.address);

      // Malicious pauserA pauses.
      await ftns.connect(pauserA).pause();
      expect(await ftns.paused()).to.equal(true);

      // Any other PAUSER (or admin via re-grant) can unpause.
      await ftns.connect(pauserB).unpause();
      expect(await ftns.paused()).to.equal(false);
    });

    it("REGRESSION (B-PAUSE-1, transitive via HIGH-5): admin cannot renounce DEFAULT_ADMIN_ROLE, so all-PAUSERs-renounce-while-paused does NOT permanently freeze transfers", async function () {
      const [admin, treasury, freshPauser] = await ethers.getSigners();
      const FTNS = await ethers.getContractFactory("FTNSTokenSimple");
      const ftns = await upgrades.deployProxy(
        FTNS,
        [admin.address, treasury.address],
        { initializer: "initialize", kind: "uups" },
      );
      const PAUSER = ethers.id("PAUSER_ROLE");
      const ADMIN = ethers.ZeroHash;

      await ftns.connect(admin).pause();

      // Admin renounces PAUSER (themselves). Allowed — only the dangerous
      // ADMIN renounce is blocked.
      await ftns.connect(admin).renounceRole(PAUSER, admin.address);
      expect(await ftns.hasRole(PAUSER, admin.address)).to.equal(false);

      // Admin attempts to renounce DEFAULT_ADMIN_ROLE. Post-HIGH-5 this
      // reverts — closing the precondition that made B-PAUSE-1 dangerous.
      await expect(
        ftns.connect(admin).renounceRole(ADMIN, admin.address),
      ).to.be.revertedWith("DEFAULT_ADMIN_ROLE renounce disabled - use grantRole(new) + revokeRole(old)");

      // Admin still holds DEFAULT_ADMIN_ROLE → can grant fresh PAUSER.
      await ftns.connect(admin).grantRole(PAUSER, freshPauser.address);

      // Fresh PAUSER unpauses. Transfers resume.
      await ftns.connect(freshPauser).unpause();
      expect(await ftns.paused()).to.equal(false);

      const ONE = ethers.parseUnits("1", 18);
      await expect(
        ftns.connect(treasury).transfer(admin.address, ONE),
      ).to.not.be.reverted;
    });
  });

  // ── B7 ──────────────────────────────────────────────────────────────
  describe("B7 — initializer re-entry on FTNSTokenSimple", function () {
    it("DEFENSE: implementation has _disableInitializers in constructor; proxy initializer cannot be re-called", async function () {
      const [admin, treasury, attacker] = await ethers.getSigners();
      const FTNS = await ethers.getContractFactory("FTNSTokenSimple");

      // Direct implementation: try calling initialize on the deployed
      // implementation contract. _disableInitializers locked it.
      const implOnly = await FTNS.deploy();
      await expect(
        implOnly.initialize(attacker.address, attacker.address),
      ).to.be.revertedWithCustomError(implOnly, "InvalidInitialization");

      // Proxy: initialize-once enforced by `initializer` modifier.
      const proxy = await upgrades.deployProxy(
        FTNS,
        [admin.address, treasury.address],
        { initializer: "initialize", kind: "uups" },
      );
      await expect(
        proxy.initialize(attacker.address, attacker.address),
      ).to.be.revertedWithCustomError(proxy, "InvalidInitialization");
    });
  });

  // ── B8 ──────────────────────────────────────────────────────────────
  describe("B8 — slasher acceptance pattern", function () {
    it("CONFIRMED: StakeBond.slasher accepts any address — no IBatchSettlementRegistry interface check", async function () {
      const [owner, eoa] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const ftns = await MockERC20.deploy();

      const Bond = await ethers.getContractFactory("StakeBond");
      const bond = await Bond.deploy(owner.address, await ftns.getAddress(), 7 * 24 * 60 * 60);

      // Pure EOA — no contract code at all. setSlasher accepts.
      await bond.setSlasher(eoa.address);
      expect(await bond.slasher()).to.equal(eoa.address);

      // Even zero-address (slashing-disabled mode) is accepted by the
      // setter — but the comment says "Address(0) disables slashing"
      // which IS intentional. Documenting that the only validation is
      // implicit; setter is permissive by design.
      await bond.setSlasher(ethers.ZeroAddress);
      expect(await bond.slasher()).to.equal(ethers.ZeroAddress);
    });
  });

  // ── B10 ─────────────────────────────────────────────────────────────
  describe("B10 — constructor argument poisoning", function () {
    it("DEFENSE: RoyaltyDistributor constructor rejects zero addresses for ftns/registry/networkTreasury", async function () {
      const Distributor = await ethers.getContractFactory("RoyaltyDistributor");
      const dummy = "0x000000000000000000000000000000000000bEEF";

      await expect(
        Distributor.deploy(ethers.ZeroAddress, dummy, dummy),
      ).to.be.revertedWith("Zero ftns");
      await expect(
        Distributor.deploy(dummy, ethers.ZeroAddress, dummy),
      ).to.be.revertedWith("Zero registry");
      await expect(
        Distributor.deploy(dummy, dummy, ethers.ZeroAddress),
      ).to.be.revertedWith("Zero treasury");
    });

    it("CONFIRMED secondary: networkTreasury IS immutable, but a wrong-but-nonzero address at deploy is permanent", async function () {
      // The on-chain check in deploy-provenance.js requires treasury
      // to be a contract on mainnet — but ANY contract qualifies, not
      // specifically the Foundation Safe. A typo to a different deployed
      // contract address (random ERC-20, random Safe, etc.) would
      // permanently route the 2% network fee elsewhere.
      const [deployer, attackerEOA] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const ftns = await MockERC20.deploy();
      const Registry = await ethers.getContractFactory("ProvenanceRegistry");
      const registry = await Registry.deploy();

      // Deploy a dummy contract that happens to be at an arbitrary
      // address. Nothing validates "this is actually the Safe."
      const wrongTreasury = await MockERC20.deploy();
      const Distributor = await ethers.getContractFactory("RoyaltyDistributor");
      const dist = await Distributor.deploy(
        await ftns.getAddress(),
        await registry.getAddress(),
        await wrongTreasury.getAddress(),
      );

      // networkTreasury is immutable — no way to fix.
      expect(await dist.networkTreasury()).to.equal(await wrongTreasury.getAddress());
      // The fix-after path requires re-deploying RoyaltyDistributor
      // and migrating all integrations. Operational defense is the
      // checksum review at signing time.
    });
  });

  // ── VERIFIER-ABI ────────────────────────────────────────────────────
  describe("VERIFIER-ABI — verify-audit-bundle-deployment.js getter mismatch", function () {
    it("CONFIRMED: deployed contracts do NOT expose challengeWindow() / unbondDelay() / ftnsToken() — verifier compares against missing getters", async function () {
      const [deployer] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const ftns = await MockERC20.deploy();

      const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
      const registry = await Registry.deploy(deployer.address, 86400);

      const Bond = await ethers.getContractFactory("StakeBond");
      const bond = await Bond.deploy(deployer.address, await ftns.getAddress(), 7 * 24 * 60 * 60);

      // The verifier script's ABI claims these getters. Build a Contract
      // with that exact ABI and confirm calls revert (no such function).
      const verifierAbiRegistry = new ethers.Contract(
        await registry.getAddress(),
        [
          "function challengeWindow() view returns (uint256)", // WRONG
          "function challengeWindowSeconds() view returns (uint256)", // RIGHT
        ],
        deployer,
      );
      await expect(verifierAbiRegistry.challengeWindow())
        .to.be.reverted;
      // Correct getter works:
      expect(await verifierAbiRegistry.challengeWindowSeconds()).to.equal(86400n);

      const verifierAbiBond = new ethers.Contract(
        await bond.getAddress(),
        [
          "function unbondDelay() view returns (uint256)", // WRONG
          "function unbondDelaySeconds() view returns (uint256)", // RIGHT
          "function ftnsToken() view returns (address)", // WRONG
          "function ftns() view returns (address)", // RIGHT
        ],
        deployer,
      );
      await expect(verifierAbiBond.unbondDelay()).to.be.reverted;
      await expect(verifierAbiBond.ftnsToken()).to.be.reverted;
      // Correct getters work:
      expect(await verifierAbiBond.unbondDelaySeconds()).to.equal(BigInt(7 * 24 * 60 * 60));
      expect(await verifierAbiBond.ftns()).to.equal(await ftns.getAddress());
    });
  });

  // ── Bonus: ProvenanceRegistry transferContentOwnership single-step ──
  describe("Bonus — ProvenanceRegistry.transferContentOwnership zero-target check", function () {
    it("DEFENSE: zero-address transfer reverts; single-step but the only loss is per-content royalty stream, not contract control", async function () {
      const Registry = await ethers.getContractFactory("ProvenanceRegistry");
      const registry = await Registry.deploy();
      const [creator] = await ethers.getSigners();

      const hash = ethers.id("content");
      await registry.connect(creator).registerContent(hash, 500, "ipfs://x");

      await expect(
        registry.connect(creator).transferContentOwnership(hash, ethers.ZeroAddress),
      ).to.be.revertedWith("Zero address");

      // No two-step semantic, but loss surface is bounded: a typo
      // re-routes future royalties for one content hash, not all of
      // them. Original creator can still register new hashes.
    });
  });
});
