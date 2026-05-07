INFO:Slither:slither.config.json has an unknown key: solc_version : 0.8.22
'npx hardhat clean' running (wd: /Users/ryneschultz/Documents/GitHub/PRSM/contracts)
'npx hardhat clean --global' running (wd: /Users/ryneschultz/Documents/GitHub/PRSM/contracts)
Problem executing hardhat: WARNING: You are currently using Node.js v25.9.0, which is not supported by Hardhat. This can lead to unexpected behavior. See https://v2.hardhat.org/nodejs-versions

'npx hardhat compile --force' running (wd: /Users/ryneschultz/Documents/GitHub/PRSM/contracts)

Detector: reentrancy-balance
Reentrancy in CompensationDistributor._distribute() (contracts/CompensationDistributor.sol#184-206):
	External call allowing reentrancy:
	- toCreator > 0 && ! ftnsToken.transfer(creatorPool,toCreator) (contracts/CompensationDistributor.sol#194)
	Balance read before the call:
	- available = ftnsToken.balanceOf(address(this)) (contracts/CompensationDistributor.sol#185)
	Possible stale balance used after the call in a condition:
	- toCreator > 0 && ! ftnsToken.transfer(creatorPool,toCreator) (contracts/CompensationDistributor.sol#194)
		- stale variable `toCreator`
	- toGrant > 0 && ! ftnsToken.transfer(grantPool,toGrant) (contracts/CompensationDistributor.sol#200)
		- stale variable `toGrant`
	- toOperator > 0 && ! ftnsToken.transfer(operatorPool,toOperator) (contracts/CompensationDistributor.sol#197)
		- stale variable `toOperator`
Reentrancy in CompensationDistributor._distribute() (contracts/CompensationDistributor.sol#184-206):
	External call allowing reentrancy:
	- toOperator > 0 && ! ftnsToken.transfer(operatorPool,toOperator) (contracts/CompensationDistributor.sol#197)
	Balance read before the call:
	- available = ftnsToken.balanceOf(address(this)) (contracts/CompensationDistributor.sol#185)
	Possible stale balance used after the call in a condition:
	- toOperator > 0 && ! ftnsToken.transfer(operatorPool,toOperator) (contracts/CompensationDistributor.sol#197)
		- stale variable `toOperator`
Reentrancy in CompensationDistributor._distribute() (contracts/CompensationDistributor.sol#184-206):
	External call allowing reentrancy:
	- toGrant > 0 && ! ftnsToken.transfer(grantPool,toGrant) (contracts/CompensationDistributor.sol#200)
	Balance read before the call:
	- available = ftnsToken.balanceOf(address(this)) (contracts/CompensationDistributor.sol#185)
	Possible stale balance used after the call in a condition:
	- toGrant > 0 && ! ftnsToken.transfer(grantPool,toGrant) (contracts/CompensationDistributor.sol#200)
		- stale variable `toGrant`
Reference: https://github.com/crytic/slither/wiki/Detector-Documentation#reentrancy-vulnerabilities

Detector: unchecked-transfer
FTNSBridge.bridgeOut(uint256,uint256) (contracts/FTNSBridge.sol#203-276) ignores return value by ftnsToken.transferFrom(msg.sender,address(this),amount) (contracts/FTNSBridge.sol#251)
Reference: https://github.com/crytic/slither/wiki/Detector-Documentation#unchecked-transfer

Detector: divide-before-multiply
StakeBond.slash(address,address,bytes32) (contracts/StakeBond.sol#561-603) performs a multiplication on the result of a division:
	- slashAmount = (uint256(s.amount) * uint256(s.tier_slash_rate_bps)) / 10000 (contracts/StakeBond.sol#576)
	- challengerShare = (slashAmount * uint256(CHALLENGER_BOUNTY_BPS)) / 10000 (contracts/StakeBond.sol#593)
Reference: https://github.com/crytic/slither/wiki/Detector-Documentation#divide-before-multiply

Detector: incorrect-equality
CompensationDistributor._distribute() (contracts/CompensationDistributor.sol#184-206) uses a dangerous strict equality:
	- available == 0 (contracts/CompensationDistributor.sol#186)
CompensationDistributor._tryPull() (contracts/CompensationDistributor.sol#162-182) uses a dangerous strict equality:
	- toMint == 0 (contracts/CompensationDistributor.sol#179)
ProvenanceRegistryV2.verifyEmbeddingCommitment(bytes32,bytes32) (contracts/ProvenanceRegistryV2.sol#156-166) uses a dangerous strict equality:
	- onChain == bytes32(0) (contracts/ProvenanceRegistryV2.sol#162)
ProvenanceRegistryV2.verifyEmbeddingCommitment(bytes32,bytes32) (contracts/ProvenanceRegistryV2.sol#156-166) uses a dangerous strict equality:
	- onChain == claimed (contracts/ProvenanceRegistryV2.sol#165)
Reference: https://github.com/crytic/slither/wiki/Detector-Documentation#dangerous-strict-equalities

Detector: reentrancy-no-eth
Reentrancy in FTNSBridge.bridgeIn(BridgeSecurity.BridgeMessage,BridgeSecurity.ValidatorSignature[]) (contracts/FTNSBridge.sol#286-332):
	External calls:
	- (messageHash,isValid) = bridgeSecurity.verifyBridgeSignatures(message,signatures) (contracts/FTNSBridge.sol#297-300)
	State variables written after the call(s):
	- processedSourceTx[message.sourceTxId] = true (contracts/FTNSBridge.sol#305)
	FTNSBridge.processedSourceTx (contracts/FTNSBridge.sol#64) can be used in cross function reentrancies:
	- FTNSBridge.checkBridgeIn(BridgeSecurity.BridgeMessage,BridgeSecurity.ValidatorSignature[]) (contracts/FTNSBridge.sol#341-358)
	- FTNSBridge.isSourceTxProcessed(bytes32) (contracts/FTNSBridge.sol#467-469)
	- FTNSBridge.processedSourceTx (contracts/FTNSBridge.sol#64)
Reference: https://github.com/crytic/slither/wiki/Detector-Documentation#reentrancy-vulnerabilities-2

Detector: uninitialized-local
Sha512.bytesToBytes8(bytes,uint256).out (contracts/lib/Sha512.sol#45) is a local variable never initialized
Sha512.cutBlock(bytes,uint256).result (contracts/lib/Sha512.sol#59) is a local variable never initialized
BatchSettlementRegistry.challengeReceipt(bytes32,ReceiptLeaf,bytes32[],BatchSettlementRegistry.ReasonCode,bytes).proven (contracts/BatchSettlementRegistry.sol#665) is a local variable never initialized
Sha512.hash(bytes).fvar (contracts/lib/Sha512.sol#187) is a local variable never initialized
Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).tables (contracts/lib/Ed25519Lib.sol#424) is a local variable never initialized
Sha512.hash(bytes).W (contracts/lib/Sha512.sol#186) is a local variable never initialized
Reference: https://github.com/crytic/slither/wiki/Detector-Documentation#uninitialized-local-variables

Detector: unused-return
FTNSBridge.checkBridgeIn(BridgeSecurity.BridgeMessage,BridgeSecurity.ValidatorSignature[]) (contracts/FTNSBridge.sol#341-358) ignores return value by bridgeSecurity.checkBridgeSignatures(message,signatures) (contracts/FTNSBridge.sol#357)
Reference: https://github.com/crytic/slither/wiki/Detector-Documentation#unused-return

Detector: write-after-write
Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).kkx (contracts/lib/Ed25519Lib.sol#435) is written in both
	kkx = xxyy + xxyy (contracts/lib/Ed25519Lib.sol#453)
	kkx = xxyy_scope_7 + xxyy_scope_7 (contracts/lib/Ed25519Lib.sol#477)
Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).kky (contracts/lib/Ed25519Lib.sol#434) is written in both
	kky = xx2 + yy2 (contracts/lib/Ed25519Lib.sol#455)
	kky = xx2_scope_5 + yy2_scope_6 (contracts/lib/Ed25519Lib.sol#479)
Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).kkv (contracts/lib/Ed25519Lib.sol#437) is written in both
	kkv = addmod(uint256,uint256,uint256)(zz2 + zz2,0xffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffda - kku,0x7fffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffed) (contracts/lib/Ed25519Lib.sol#456-460)
	kkv = addmod(uint256,uint256,uint256)(zz2_scope_8 + zz2_scope_8,0xffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffda - kku,0x7fffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffed) (contracts/lib/Ed25519Lib.sol#480-484)
Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).kkx (contracts/lib/Ed25519Lib.sol#435) is written in both
	kkx = xxyy_scope_7 + xxyy_scope_7 (contracts/lib/Ed25519Lib.sol#477)
	kkx = xxyy_scope_14 + xxyy_scope_14 (contracts/lib/Ed25519Lib.sol#501)
Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).kky (contracts/lib/Ed25519Lib.sol#434) is written in both
	kky = xx2_scope_5 + yy2_scope_6 (contracts/lib/Ed25519Lib.sol#479)
	kky = xx2_scope_12 + yy2_scope_13 (contracts/lib/Ed25519Lib.sol#503)
Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).kkv (contracts/lib/Ed25519Lib.sol#437) is written in both
	kkv = addmod(uint256,uint256,uint256)(zz2_scope_8 + zz2_scope_8,0xffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffda - kku,0x7fffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffed) (contracts/lib/Ed25519Lib.sol#480-484)
	kkv = addmod(uint256,uint256,uint256)(zz2_scope_15 + zz2_scope_15,0xffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffda - kku,0x7fffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffed) (contracts/lib/Ed25519Lib.sol#504-508)
Reference: https://github.com/crytic/slither/wiki/Detector-Documentation#write-after-write
. analyzed (76 contracts with 58 detectors), 23 result(s) found
**THIS CHECKLIST IS NOT COMPLETE**. Use `--show-ignored-findings` to show all the results.
Summary
 - [reentrancy-balance](#reentrancy-balance) (3 results) (High)
 - [unchecked-transfer](#unchecked-transfer) (1 results) (High)
 - [divide-before-multiply](#divide-before-multiply) (1 results) (Medium)
 - [incorrect-equality](#incorrect-equality) (4 results) (Medium)
 - [reentrancy-no-eth](#reentrancy-no-eth) (1 results) (Medium)
 - [uninitialized-local](#uninitialized-local) (6 results) (Medium)
 - [unused-return](#unused-return) (1 results) (Medium)
 - [write-after-write](#write-after-write) (6 results) (Medium)
## reentrancy-balance
Impact: High
Confidence: Medium
 - [ ] ID-0
Reentrancy in [CompensationDistributor._distribute()](contracts/CompensationDistributor.sol#L184-L206):
	External call allowing reentrancy:
	- [toCreator > 0 && ! ftnsToken.transfer(creatorPool,toCreator)](contracts/CompensationDistributor.sol#L194)
	Balance read before the call:
	- [available = ftnsToken.balanceOf(address(this))](contracts/CompensationDistributor.sol#L185)
	Possible stale balance used after the call in a condition:
	- [toCreator > 0 && ! ftnsToken.transfer(creatorPool,toCreator)](contracts/CompensationDistributor.sol#L194)
		- stale variable `toCreator`
	- [toGrant > 0 && ! ftnsToken.transfer(grantPool,toGrant)](contracts/CompensationDistributor.sol#L200)
		- stale variable `toGrant`
	- [toOperator > 0 && ! ftnsToken.transfer(operatorPool,toOperator)](contracts/CompensationDistributor.sol#L197)
		- stale variable `toOperator`

contracts/CompensationDistributor.sol#L184-L206


 - [ ] ID-1
Reentrancy in [CompensationDistributor._distribute()](contracts/CompensationDistributor.sol#L184-L206):
	External call allowing reentrancy:
	- [toGrant > 0 && ! ftnsToken.transfer(grantPool,toGrant)](contracts/CompensationDistributor.sol#L200)
	Balance read before the call:
	- [available = ftnsToken.balanceOf(address(this))](contracts/CompensationDistributor.sol#L185)
	Possible stale balance used after the call in a condition:
	- [toGrant > 0 && ! ftnsToken.transfer(grantPool,toGrant)](contracts/CompensationDistributor.sol#L200)
		- stale variable `toGrant`

contracts/CompensationDistributor.sol#L184-L206


 - [ ] ID-2
Reentrancy in [CompensationDistributor._distribute()](contracts/CompensationDistributor.sol#L184-L206):
	External call allowing reentrancy:
	- [toOperator > 0 && ! ftnsToken.transfer(operatorPool,toOperator)](contracts/CompensationDistributor.sol#L197)
	Balance read before the call:
	- [available = ftnsToken.balanceOf(address(this))](contracts/CompensationDistributor.sol#L185)
	Possible stale balance used after the call in a condition:
	- [toOperator > 0 && ! ftnsToken.transfer(operatorPool,toOperator)](contracts/CompensationDistributor.sol#L197)
		- stale variable `toOperator`

contracts/CompensationDistributor.sol#L184-L206


## unchecked-transfer
Impact: High
Confidence: Medium
 - [ ] ID-3
[FTNSBridge.bridgeOut(uint256,uint256)](contracts/FTNSBridge.sol#L203-L276) ignores return value by [ftnsToken.transferFrom(msg.sender,address(this),amount)](contracts/FTNSBridge.sol#L251)

contracts/FTNSBridge.sol#L203-L276


## divide-before-multiply
Impact: Medium
Confidence: Medium
 - [ ] ID-4
[StakeBond.slash(address,address,bytes32)](contracts/StakeBond.sol#L561-L603) performs a multiplication on the result of a division:
	- [slashAmount = (uint256(s.amount) * uint256(s.tier_slash_rate_bps)) / 10000](contracts/StakeBond.sol#L576)
	- [challengerShare = (slashAmount * uint256(CHALLENGER_BOUNTY_BPS)) / 10000](contracts/StakeBond.sol#L593)

contracts/StakeBond.sol#L561-L603


## incorrect-equality
Impact: Medium
Confidence: High
 - [ ] ID-5
[ProvenanceRegistryV2.verifyEmbeddingCommitment(bytes32,bytes32)](contracts/ProvenanceRegistryV2.sol#L156-L166) uses a dangerous strict equality:
	- [onChain == claimed](contracts/ProvenanceRegistryV2.sol#L165)

contracts/ProvenanceRegistryV2.sol#L156-L166


 - [ ] ID-6
[CompensationDistributor._distribute()](contracts/CompensationDistributor.sol#L184-L206) uses a dangerous strict equality:
	- [available == 0](contracts/CompensationDistributor.sol#L186)

contracts/CompensationDistributor.sol#L184-L206


 - [ ] ID-7
[CompensationDistributor._tryPull()](contracts/CompensationDistributor.sol#L162-L182) uses a dangerous strict equality:
	- [toMint == 0](contracts/CompensationDistributor.sol#L179)

contracts/CompensationDistributor.sol#L162-L182


 - [ ] ID-8
[ProvenanceRegistryV2.verifyEmbeddingCommitment(bytes32,bytes32)](contracts/ProvenanceRegistryV2.sol#L156-L166) uses a dangerous strict equality:
	- [onChain == bytes32(0)](contracts/ProvenanceRegistryV2.sol#L162)

contracts/ProvenanceRegistryV2.sol#L156-L166


## reentrancy-no-eth
Impact: Medium
Confidence: Medium
 - [ ] ID-9
Reentrancy in [FTNSBridge.bridgeIn(BridgeSecurity.BridgeMessage,BridgeSecurity.ValidatorSignature[])](contracts/FTNSBridge.sol#L286-L332):
	External calls:
	- [(messageHash,isValid) = bridgeSecurity.verifyBridgeSignatures(message,signatures)](contracts/FTNSBridge.sol#L297-L300)
	State variables written after the call(s):
	- [processedSourceTx[message.sourceTxId] = true](contracts/FTNSBridge.sol#L305)
	[FTNSBridge.processedSourceTx](contracts/FTNSBridge.sol#L64) can be used in cross function reentrancies:
	- [FTNSBridge.checkBridgeIn(BridgeSecurity.BridgeMessage,BridgeSecurity.ValidatorSignature[])](contracts/FTNSBridge.sol#L341-L358)
	- [FTNSBridge.isSourceTxProcessed(bytes32)](contracts/FTNSBridge.sol#L467-L469)
	- [FTNSBridge.processedSourceTx](contracts/FTNSBridge.sol#L64)

contracts/FTNSBridge.sol#L286-L332


## uninitialized-local
Impact: Medium
Confidence: Medium
 - [ ] ID-10
[BatchSettlementRegistry.challengeReceipt(bytes32,ReceiptLeaf,bytes32[],BatchSettlementRegistry.ReasonCode,bytes).proven](contracts/BatchSettlementRegistry.sol#L665) is a local variable never initialized

contracts/BatchSettlementRegistry.sol#L665


 - [ ] ID-11
[Sha512.hash(bytes).fvar](contracts/lib/Sha512.sol#L187) is a local variable never initialized

contracts/lib/Sha512.sol#L187


 - [ ] ID-12
[Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).tables](contracts/lib/Ed25519Lib.sol#L424) is a local variable never initialized

contracts/lib/Ed25519Lib.sol#L424


 - [ ] ID-13
[Sha512.hash(bytes).W](contracts/lib/Sha512.sol#L186) is a local variable never initialized

contracts/lib/Sha512.sol#L186


 - [ ] ID-14
[Sha512.bytesToBytes8(bytes,uint256).out](contracts/lib/Sha512.sol#L45) is a local variable never initialized

contracts/lib/Sha512.sol#L45


 - [ ] ID-15
[Sha512.cutBlock(bytes,uint256).result](contracts/lib/Sha512.sol#L59) is a local variable never initialized

contracts/lib/Sha512.sol#L59


## unused-return
Impact: Medium
Confidence: Medium
 - [ ] ID-16
[FTNSBridge.checkBridgeIn(BridgeSecurity.BridgeMessage,BridgeSecurity.ValidatorSignature[])](contracts/FTNSBridge.sol#L341-L358) ignores return value by [bridgeSecurity.checkBridgeSignatures(message,signatures)](contracts/FTNSBridge.sol#L357)

contracts/FTNSBridge.sol#L341-L358


## write-after-write
Impact: Medium
Confidence: High
 - [ ] ID-17
[Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).kkx](contracts/lib/Ed25519Lib.sol#L435) is written in both
	[kkx = xxyy + xxyy](contracts/lib/Ed25519Lib.sol#L453)
	[kkx = xxyy_scope_7 + xxyy_scope_7](contracts/lib/Ed25519Lib.sol#L477)

contracts/lib/Ed25519Lib.sol#L435


 - [ ] ID-18
[Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).kkx](contracts/lib/Ed25519Lib.sol#L435) is written in both
	[kkx = xxyy_scope_7 + xxyy_scope_7](contracts/lib/Ed25519Lib.sol#L477)
	[kkx = xxyy_scope_14 + xxyy_scope_14](contracts/lib/Ed25519Lib.sol#L501)

contracts/lib/Ed25519Lib.sol#L435


 - [ ] ID-19
[Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).kky](contracts/lib/Ed25519Lib.sol#L434) is written in both
	[kky = xx2_scope_5 + yy2_scope_6](contracts/lib/Ed25519Lib.sol#L479)
	[kky = xx2_scope_12 + yy2_scope_13](contracts/lib/Ed25519Lib.sol#L503)

contracts/lib/Ed25519Lib.sol#L434


 - [ ] ID-20
[Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).kky](contracts/lib/Ed25519Lib.sol#L434) is written in both
	[kky = xx2 + yy2](contracts/lib/Ed25519Lib.sol#L455)
	[kky = xx2_scope_5 + yy2_scope_6](contracts/lib/Ed25519Lib.sol#L479)

contracts/lib/Ed25519Lib.sol#L434


 - [ ] ID-21
[Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).kkv](contracts/lib/Ed25519Lib.sol#L437) is written in both
	[kkv = addmod(uint256,uint256,uint256)(zz2_scope_8 + zz2_scope_8,0xffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffda - kku,0x7fffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffed)](contracts/lib/Ed25519Lib.sol#L480-L484)
	[kkv = addmod(uint256,uint256,uint256)(zz2_scope_15 + zz2_scope_15,0xffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffda - kku,0x7fffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffed)](contracts/lib/Ed25519Lib.sol#L504-L508)

contracts/lib/Ed25519Lib.sol#L437


 - [ ] ID-22
[Ed25519Lib.verify(bytes32,bytes32,bytes32,bytes).kkv](contracts/lib/Ed25519Lib.sol#L437) is written in both
	[kkv = addmod(uint256,uint256,uint256)(zz2 + zz2,0xffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffda - kku,0x7fffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffed)](contracts/lib/Ed25519Lib.sol#L456-L460)
	[kkv = addmod(uint256,uint256,uint256)(zz2_scope_8 + zz2_scope_8,0xffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffda - kku,0x7fffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffff_ffffffed)](contracts/lib/Ed25519Lib.sol#L480-L484)

contracts/lib/Ed25519Lib.sol#L437


