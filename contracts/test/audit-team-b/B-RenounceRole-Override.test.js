/*
 * Team B — REGRESSION GUARD for B-RENOUNCE-1 (HIGH) + B-PAUSE-1 (MEDIUM, bundled).
 *
 * Original findings:
 *   - B-RENOUNCE-1: OZ AccessControl.renounceRole reachable on the
 *     sole DEFAULT_ADMIN_ROLE holder. One bad multi-sig tx → permanent
 *     UUPS upgrade brick + total role-rotation lockout.
 *   - B-PAUSE-1: All-PAUSER renounce while paused → permanent freeze
 *     (preconditioned on admin already renounced).
 *
 * REMEDIATION SHIPPED 2026-05-05:
 *   FTNSTokenSimple.renounceRole overridden to revert if role ==
 *   DEFAULT_ADMIN_ROLE. Genuine handoff still possible via
 *   grantRole(newAdmin) + revokeRole(oldAdmin). PAUSER renounce
 *   remains allowed; B-PAUSE-1 is closed transitively because the
 *   "admin renounced" precondition is now blocked — even all-PAUSERs-
 *   renouncing-while-paused leaves admin able to grant a fresh
 *   PAUSER who can unpause.
 *
 * NOTE: Live mainnet FTNSTokenSimple at 0x5276...16e5 still runs the
 * pre-fix bytecode. This fix becomes effective only after the existing
 * admin (currently an out-of-Safe hot key per CRIT-2) executes a UUPS
 * upgrade to the new implementation. Sequence:
 *   1. Deploy new impl (this commit).
 *   2. Existing admin upgrades the proxy to point at new impl.
 *   3. Then run transfer-ftns-roles.js to hand admin to the Safe.
 * Order matters: handing off admin BEFORE the upgrade leaves the Safe
 * still vulnerable to a one-tx renounce.
 */
const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");
const { loadFixture } = require("@nomicfoundation/hardhat-network-helpers");

describe("Audit Team B — B-RENOUNCE-1 regression: renounceRole override", function () {
  async function deployFixture() {
    const [owner, treasury, newAdmin, otherPauser, attacker] =
      await ethers.getSigners();

    const FTNSTokenSimple = await ethers.getContractFactory("FTNSTokenSimple");
    const ftns = await upgrades.deployProxy(
      FTNSTokenSimple,
      [owner.address, treasury.address],
      { initializer: "initialize", kind: "uups" }
    );
    return { ftns, owner, treasury, newAdmin, otherPauser, attacker };
  }

  it("REGRESSION: renounceRole(DEFAULT_ADMIN_ROLE) reverts", async function () {
    const { ftns, owner } = await loadFixture(deployFixture);
    const DEFAULT_ADMIN_ROLE = ethers.ZeroHash;
    expect(await ftns.hasRole(DEFAULT_ADMIN_ROLE, owner.address)).to.equal(true);
    await expect(
      ftns.connect(owner).renounceRole(DEFAULT_ADMIN_ROLE, owner.address)
    ).to.be.revertedWith("DEFAULT_ADMIN_ROLE renounce disabled - use grantRole(new) + revokeRole(old)");
    expect(await ftns.hasRole(DEFAULT_ADMIN_ROLE, owner.address)).to.equal(true);
  });

  it("Other roles can still be renounced normally (MINTER_ROLE)", async function () {
    const { ftns, owner } = await loadFixture(deployFixture);
    const MINTER_ROLE = await ftns.MINTER_ROLE();
    expect(await ftns.hasRole(MINTER_ROLE, owner.address)).to.equal(true);
    await expect(
      ftns.connect(owner).renounceRole(MINTER_ROLE, owner.address)
    ).to.not.be.reverted;
    expect(await ftns.hasRole(MINTER_ROLE, owner.address)).to.equal(false);
  });

  it("Other roles can still be renounced normally (BURNER_ROLE)", async function () {
    const { ftns, owner } = await loadFixture(deployFixture);
    const BURNER_ROLE = await ftns.BURNER_ROLE();
    expect(await ftns.hasRole(BURNER_ROLE, owner.address)).to.equal(true);
    await expect(
      ftns.connect(owner).renounceRole(BURNER_ROLE, owner.address)
    ).to.not.be.reverted;
    expect(await ftns.hasRole(BURNER_ROLE, owner.address)).to.equal(false);
  });

  it("Other roles can still be renounced normally (PAUSER_ROLE)", async function () {
    const { ftns, owner } = await loadFixture(deployFixture);
    const PAUSER_ROLE = await ftns.PAUSER_ROLE();
    expect(await ftns.hasRole(PAUSER_ROLE, owner.address)).to.equal(true);
    await expect(
      ftns.connect(owner).renounceRole(PAUSER_ROLE, owner.address)
    ).to.not.be.reverted;
    expect(await ftns.hasRole(PAUSER_ROLE, owner.address)).to.equal(false);
  });

  it("Genuine admin handoff still works via grantRole + revokeRole", async function () {
    const { ftns, owner, newAdmin } = await loadFixture(deployFixture);
    const DEFAULT_ADMIN_ROLE = ethers.ZeroHash;

    // Stage 1: existing admin grants admin to new admin.
    await ftns.connect(owner).grantRole(DEFAULT_ADMIN_ROLE, newAdmin.address);
    expect(await ftns.hasRole(DEFAULT_ADMIN_ROLE, newAdmin.address)).to.equal(true);
    expect(await ftns.hasRole(DEFAULT_ADMIN_ROLE, owner.address)).to.equal(true);

    // Stage 2: existing admin revokes old admin.
    await ftns.connect(owner).revokeRole(DEFAULT_ADMIN_ROLE, owner.address);
    expect(await ftns.hasRole(DEFAULT_ADMIN_ROLE, owner.address)).to.equal(false);
    expect(await ftns.hasRole(DEFAULT_ADMIN_ROLE, newAdmin.address)).to.equal(true);

    // Stage 3: new admin can grant other roles, demonstrating control.
    const MINTER_ROLE = await ftns.MINTER_ROLE();
    await ftns.connect(newAdmin).grantRole(MINTER_ROLE, owner.address);
    expect(await ftns.hasRole(MINTER_ROLE, owner.address)).to.equal(true);

    // Stage 4: new admin also cannot self-renounce admin (foot-gun
    // remains blocked for the new admin too).
    await expect(
      ftns.connect(newAdmin).renounceRole(DEFAULT_ADMIN_ROLE, newAdmin.address)
    ).to.be.revertedWith("DEFAULT_ADMIN_ROLE renounce disabled - use grantRole(new) + revokeRole(old)");
  });

  it("B-PAUSE-1 closed transitively: even if all PAUSERs renounce while paused, admin can grant fresh PAUSER who unpauses", async function () {
    const { ftns, owner, otherPauser } = await loadFixture(deployFixture);
    const PAUSER_ROLE = await ftns.PAUSER_ROLE();

    // Owner pauses.
    await ftns.connect(owner).pause();
    expect(await ftns.paused()).to.equal(true);

    // Owner (the only PAUSER) renounces PAUSER_ROLE.
    await ftns.connect(owner).renounceRole(PAUSER_ROLE, owner.address);
    expect(await ftns.hasRole(PAUSER_ROLE, owner.address)).to.equal(false);

    // Pre-fix scenario said: "no one can unpause now". But owner still
    // holds DEFAULT_ADMIN_ROLE (because that renounce is blocked), so
    // owner grants PAUSER to a fresh address.
    await ftns.connect(owner).grantRole(PAUSER_ROLE, otherPauser.address);
    expect(await ftns.hasRole(PAUSER_ROLE, otherPauser.address)).to.equal(true);

    // Fresh pauser unpauses.
    await ftns.connect(otherPauser).unpause();
    expect(await ftns.paused()).to.equal(false);
  });

  it("Random caller cannot renounce another address's roles (OZ default behavior preserved)", async function () {
    const { ftns, owner, attacker } = await loadFixture(deployFixture);
    const MINTER_ROLE = await ftns.MINTER_ROLE();
    // OZ requires callerConfirmation == msg.sender; attacker can't
    // renounce owner's MINTER role.
    await expect(
      ftns.connect(attacker).renounceRole(MINTER_ROLE, owner.address)
    ).to.be.revertedWithCustomError(ftns, "AccessControlBadConfirmation");
  });
});
