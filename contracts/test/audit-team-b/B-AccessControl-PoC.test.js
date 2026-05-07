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
 *   B5  cross-wire mutability      — mostly closed: settlementRegistry
 *                                     (HIGH-6) + StakeBond.slasher
 *                                     (HIGH-7) NOW immutable; only
 *                                     FTNS-token swap remains mutable
 *                                     (defensive escape hatch by design)
 *   B6  pauser locking             — defense (any PAUSER unpauses)
 *   B7  initializer re-entry       — defense (`_disableInitializers` in ctor)
 *   B8  slasher acceptance         — closed (HIGH-7): setSlasher REMOVED;
 *                                     constructor still accepts any
 *                                     non-zero address with no interface
 *                                     check, but rotation is no longer
 *                                     possible after deployment
 *   B9  Foundation Safe owners     — out-of-scope for hardhat (on-chain check)
 *   B10 constructor poisoning      — defense (zero-address checks)
 *
 * Plus VERIFIER-ABI: post-handoff verifier script uses wrong getter names.
 */
const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");

describe("Team B — Access Control PoC", function () {
  // ── B2 ──────────────────────────────────────────────────────────────
  describe("B2 — Ownable2Step prevents transferOwnership-typo brick (post-MEDIUM B-OWNABLE-1)", function () {
    it("REGRESSION (MEDIUM B-OWNABLE-1): typo in transferOwnership target only sets pendingOwner; original owner retains control until acceptOwnership", async function () {
      // L2 audit MEDIUM B-OWNABLE-1 fix: Ownable → Ownable2Step across
      // all 7 contracts. transferOwnership(target) now only sets the
      // pendingOwner; ownership transfer completes only when target
      // calls acceptOwnership. A typo to a dead address simply sets
      // pendingOwner to that address — the original owner retains
      // control and can re-call transferOwnership to fix the typo.
      const [deployer, attacker] = await ethers.getSigners();

      // Need a real registry address for EscrowPool's immutable
      // settlementRegistry constructor arg (post-HIGH-6). Deploy a
      // throwaway registry first.
      const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
      const registry = await Registry.deploy(deployer.address, 86400);

      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const ftns = await MockERC20.deploy();
      const Pool = await ethers.getContractFactory("EscrowPool");
      const pool = await Pool.deploy(
        deployer.address, await ftns.getAddress(), await registry.getAddress(),
      );

      const typoTarget = "0x000000000000000000000000000000000000dEaD";

      // transferOwnership now sets pendingOwner ONLY. Owner is unchanged.
      await pool.transferOwnership(typoTarget);
      expect(await pool.owner()).to.equal(deployer.address);
      expect(await pool.pendingOwner()).to.equal(typoTarget);

      // Original deployer still controls the contract — can still
      // perform owner-only ops while the typo'd handoff is pending.
      const Token2 = await ethers.getContractFactory("MockERC20");
      const altToken = await Token2.deploy();
      await expect(
        pool.connect(deployer).setFtnsToken(await altToken.getAddress()),
      ).to.not.be.reverted;

      // And — critically — deployer can RE-CALL transferOwnership to
      // overwrite the typo'd pendingOwner with the correct multisig.
      await pool.transferOwnership(attacker.address);  // attacker stands in for "correct multisig"
      expect(await pool.pendingOwner()).to.equal(attacker.address);

      // Pre-fix Ownable would have permanently bricked at the typo
      // step. Post-fix, the brick path requires the operator to ALSO
      // get the new owner to call acceptOwnership() — a 2-step
      // confirmation that surfaces the typo before it's permanent.
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
  describe("B5 — cross-wire mutability after handoff (post-HIGH-6: settlementRegistry now immutable)", function () {
    it("REGRESSION (HIGH-6): owner CANNOT re-point EscrowPool.settlementRegistry — field is now immutable, no setter exposed", async function () {
      const [owner, foundation, attacker, registry] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const ftns = await MockERC20.deploy();

      // Constructor now requires a real registry address — the immutable
      // field is set once and cannot change. Use `registry` signer as a
      // stand-in EOA registry so we can demonstrate post-fix surface.
      const Pool = await ethers.getContractFactory("EscrowPool");
      const pool = await Pool.deploy(
        owner.address,
        await ftns.getAddress(),
        registry.address,
      );

      // Simulate handoff: pretend `foundation` is the multi-sig.
      // Post-MEDIUM B-OWNABLE-1: Ownable2Step requires acceptOwnership.
      await pool.transferOwnership(foundation.address);
      await pool.connect(foundation).acceptOwnership();
      expect(await pool.owner()).to.equal(foundation.address);

      // Post-fix: setSettlementRegistry no longer exists at the high-
      // level binding. A compromised multi-sig CANNOT re-point.
      expect(pool.setSettlementRegistry).to.be.undefined;

      // Low-level call attempting the old selector reverts because the
      // function is not in the deployed bytecode.
      const nukedSelector = ethers.id("setSettlementRegistry(address)").slice(0, 10);
      const data =
        nukedSelector + "000000000000000000000000" + attacker.address.slice(2);
      await expect(
        foundation.sendTransaction({ to: await pool.getAddress(), data }),
      ).to.be.reverted;

      // settlementRegistry value is unchanged — still the constructor arg.
      expect(await pool.settlementRegistry()).to.equal(registry.address);

      // attacker.address cannot trigger settleFromRequester even with
      // funds available, because they are not the immutable registry.
      const TEN = ethers.parseUnits("10", 18);
      await ftns.mint(owner.address, TEN);
      await ftns.connect(owner).approve(await pool.getAddress(), TEN);
      await pool.connect(owner).deposit(TEN);

      await expect(
        pool
          .connect(attacker)
          .settleFromRequester(owner.address, attacker.address, TEN),
      ).to.be.revertedWithCustomError(pool, "CallerNotRegistry");

      // Funds untouched.
      expect(await ftns.balanceOf(attacker.address)).to.equal(0n);
      expect(await pool.balances(owner.address)).to.equal(TEN);
    });

    it("REGRESSION (MEDIUM B-CROSS-2): owner CANNOT setFtnsToken while pending balances are non-zero — strand attack vector closed", async function () {
      const [owner, requester] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const realFtns = await MockERC20.deploy();
      const fakeFtns = await MockERC20.deploy();

      // L4 self-audit MED-6: initialRegistry must be non-zero. Deploy a
      // throwaway registry; this test exercises FTNS token swap, not
      // settlement plumbing — any real registry suffices.
      const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
      const registry = await Registry.deploy(owner.address, 86400);
      const Pool = await ethers.getContractFactory("EscrowPool");
      const pool = await Pool.deploy(owner.address, await realFtns.getAddress(), await registry.getAddress());

      // Requester deposits 100 real FTNS — totalEscrowedBalance now = 100.
      const HUNDRED = ethers.parseUnits("100", 18);
      await realFtns.mint(requester.address, HUNDRED);
      await realFtns.connect(requester).approve(await pool.getAddress(), HUNDRED);
      await pool.connect(requester).deposit(HUNDRED);
      expect(await pool.balances(requester.address)).to.equal(HUNDRED);
      expect(await pool.totalEscrowedBalance()).to.equal(HUNDRED);

      // Pre-fix: owner could swap mid-flight and strand real FTNS.
      // Post-fix: setFtnsToken reverts because totalEscrowedBalance > 0.
      await expect(
        pool.setFtnsToken(await fakeFtns.getAddress()),
      ).to.be.revertedWithCustomError(pool, "PendingBalancesNonZero");

      // ftns reference unchanged — still real.
      expect(await pool.ftns()).to.equal(await realFtns.getAddress());

      // Requester can drain via withdraw (totalEscrowedBalance → 0),
      // and AFTER full unwind the swap is allowed.
      await pool.connect(requester).withdraw(HUNDRED);
      expect(await pool.totalEscrowedBalance()).to.equal(0);
      await expect(
        pool.setFtnsToken(await fakeFtns.getAddress()),
      ).to.not.be.reverted;
      expect(await pool.ftns()).to.equal(await fakeFtns.getAddress());
    });

    it("REGRESSION (HIGH-7): owner CANNOT re-point StakeBond.slasher — field is now immutable, no setter exposed", async function () {
      const [owner, provider, foundation, attacker, legitSlasher] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const ftns = await MockERC20.deploy();

      // Constructor now requires a slasher arg. Use legitSlasher signer
      // as a stand-in EOA "registry" so we can demonstrate post-fix
      // surface — the slash() path itself reverts on EOA caller-not-
      // slasher mismatch, but the immutability of the field is the
      // load-bearing assertion.
      const Bond = await ethers.getContractFactory("StakeBond");
      const bond = await Bond.deploy(
        owner.address, await ftns.getAddress(), 7 * 24 * 60 * 60, legitSlasher.address
      );

      // Provider stakes 10K FTNS at standard tier (50% slash).
      const STAKE = ethers.parseUnits("10000", 18);
      await ftns.mint(provider.address, STAKE);
      await ftns.connect(provider).approve(await bond.getAddress(), STAKE);
      await bond.connect(provider).bond(STAKE, 5000);

      // Post-fix: setSlasher no longer exists at the high-level binding.
      expect(bond.setSlasher).to.be.undefined;

      // Low-level call attempting the old selector reverts because the
      // function is not in the deployed bytecode.
      const nukedSelector = ethers.id("setSlasher(address)").slice(0, 10);
      const data = nukedSelector + "000000000000000000000000" + attacker.address.slice(2);
      await expect(
        owner.sendTransaction({ to: await bond.getAddress(), data }),
      ).to.be.reverted;

      // slasher value is unchanged — still legitSlasher.
      expect(await bond.slasher()).to.equal(legitSlasher.address);

      // Attacker calling slash() reverts CallerNotSlasher.
      const reasonId = ethers.id("fabricated-batch-id");
      await expect(
        bond.connect(attacker).slash(provider.address, attacker.address, reasonId),
      ).to.be.revertedWithCustomError(bond, "CallerNotSlasher");

      // Provider's stake untouched; attacker bounty unaccrued.
      const stake = await bond.stakeOf(provider.address);
      expect(stake.amount).to.equal(STAKE);
      expect(await bond.slashedBountyPayable(attacker.address)).to.equal(0n);
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
  describe("B8 — slasher acceptance pattern (post-HIGH-7: setter REMOVED)", function () {
    it("REGRESSION (HIGH-7): no setSlasher exists; constructor accepts any address but rotation is impossible after deploy", async function () {
      const [owner, eoa] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const ftns = await MockERC20.deploy();

      // Constructor accepts an EOA (no interface check) — this is the
      // residual permissiveness, but rotation is now impossible. The
      // operational defense moves entirely to checksum review at signing
      // time, just as for RoyaltyDistributor.networkTreasury.
      const Bond = await ethers.getContractFactory("StakeBond");
      const bondWithEOA = await Bond.deploy(
        owner.address, await ftns.getAddress(), 7 * 24 * 60 * 60, eoa.address
      );
      expect(await bondWithEOA.slasher()).to.equal(eoa.address);

      // L4 self-audit MED-6 (post-fix): constructor REJECTS address(0)
      // for slasher. The "slashing-disabled mode" is no longer reachable
      // via the bond constructor — operators must wire a real BSR
      // address (which itself can have stakeBond=0 to disable slashing).
      await expect(
        Bond.deploy(owner.address, await ftns.getAddress(), 7 * 24 * 60 * 60, ethers.ZeroAddress)
      ).to.be.revertedWithCustomError(Bond, "ZeroAddress");

      // Post-fix: high-level binding has no setter; low-level call to
      // the removed selector reverts.
      expect(bondWithEOA.setSlasher).to.be.undefined;
      const nukedSelector = ethers.id("setSlasher(address)").slice(0, 10);
      const data = nukedSelector + "000000000000000000000000" + owner.address.slice(2);
      await expect(
        owner.sendTransaction({ to: await bondWithEOA.getAddress(), data })
      ).to.be.reverted;
    });
  });

  // ── B10 ─────────────────────────────────────────────────────────────
  describe("B10 — constructor argument poisoning", function () {
    it("DEFENSE: RoyaltyDistributor constructor rejects zero addresses for ftns/registry/networkTreasury", async function () {
      const Distributor = await ethers.getContractFactory("RoyaltyDistributor");
      const dummy = "0x000000000000000000000000000000000000bEEF";

      // L4 self-audit A-08: constructor now takes a 4th arg `_initialOwner`.
      const owner = (await ethers.getSigners())[0].address;
      await expect(
        Distributor.deploy(ethers.ZeroAddress, dummy, dummy, owner),
      ).to.be.revertedWith("Zero ftns");
      await expect(
        Distributor.deploy(dummy, ethers.ZeroAddress, dummy, owner),
      ).to.be.revertedWith("Zero registry");
      await expect(
        Distributor.deploy(dummy, dummy, ethers.ZeroAddress, owner),
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
        deployer.address,
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
      const bond = await Bond.deploy(deployer.address, await ftns.getAddress(), 7 * 24 * 60 * 60, await registry.getAddress());

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
