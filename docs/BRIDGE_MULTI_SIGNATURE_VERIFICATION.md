# Bridge Multi-Signature Verification

## Overview

PRSM's cross-chain bridge implements production-grade multi-signature verification using EIP-712 typed structured data signing. This document describes the signing format, verification process, and integration guidelines.

## Architecture

### Components

1. **BridgeSecurity.sol** - Core signature verification contract
   - Validator registry management
   - EIP-712 typed message hashing
   - Multi-signature threshold verification
   - Replay attack protection

2. **FTNSBridge.sol** - Bridge operations contract
   - Bridge out (lock/burn tokens)
   - Bridge in (mint tokens with signature verification)
   - Fee management
   - Chain support configuration

3. **bridge_security.py** - Python client library
   - Message construction and signing
   - Signature aggregation
   - Contract interaction helpers

### M-of-N Threshold Scheme

The bridge uses a configurable M-of-N signature threshold:
- **N** = Total number of registered validators
- **M** = Minimum signatures required (threshold)

Example: 3-of-5 means 3 signatures required from 5 registered validators.

## EIP-712 Signing Format

### Domain Separator

```javascript
{
  name: "PRSM Bridge Security",
  version: "1",
  chainId: <chain_id>,
  verifyingContract: <bridge_security_address>
}
```

### BridgeMessage Type

```solidity
struct BridgeMessage {
    address recipient;      // Token recipient on destination chain
    uint256 amount;         // Amount of tokens (in wei)
    uint256 sourceChainId;  // Chain ID where tokens were locked
    bytes32 sourceTxId;     // Transaction ID on source chain
    uint256 nonce;          // Unique nonce for replay protection
}
```

### Type Hash

```
BridgeMessage(address recipient,uint256 amount,uint256 sourceChainId,bytes32 sourceTxId,uint256 nonce)
```

**Typehash**: `keccak256("BridgeMessage(address recipient,uint256 amount,uint256 sourceChainId,bytes32 sourceTxId,uint256 nonce)")`

### Message Hash Computation

1. **Struct Hash**:
   ```
   structHash = keccak256(abi.encode(
       TYPEHASH,
       recipient,
       amount,
       sourceChainId,
       sourceTxId,
       nonce
   ))
   ```

2. **EIP-712 Typed Hash**:
   ```
   messageHash = keccak256(abi.encodePacked(
       "\x19\x01",
       domainSeparator,
       structHash
   ))
   ```

## Signature Format

### ValidatorSignature Structure

```solidity
struct ValidatorSignature {
    address validator;  // Address of the signing validator
    bytes32 r;          // ECDSA signature r component
    bytes32 s;          // ECDSA signature s component
    uint8 v;            // ECDSA recovery id (27 or 28)
}
```

### Signing Process

1. **Construct the BridgeMessage** with all required fields
2. **Compute the EIP-712 typed hash** using the domain separator
3. **Sign the hash** with the validator's private key
4. **Extract r, s, v** from the 65-byte signature
5. **Package** with the validator's address

### Example (JavaScript/ethers.js)

```javascript
const { ethers } = require('ethers');

// Create the message
const message = {
    recipient: "0x1234...5678",
    amount: ethers.parseEther("100"),
    sourceChainId: 1,
    sourceTxId: ethers.id("source-tx-123"),
    nonce: 1
};

// Define EIP-712 types
const types = {
    BridgeMessage: [
        { name: "recipient", type: "address" },
        { name: "amount", type: "uint256" },
        { name: "sourceChainId", type: "uint256" },
        { name: "sourceTxId", type: "bytes32" },
        { name: "nonce", type: "uint256" }
    ]
};

// Get domain from contract
const domain = {
    name: "PRSM Bridge Security",
    version: "1",
    chainId: await provider.getNetwork().then(n => n.chainId),
    verifyingContract: bridgeSecurityAddress
};

// Sign the message
const signature = await validator.signTypedData(domain, types, message);

// Parse signature components
const { r, s, v } = ethers.Signature.from(signature);

// Create ValidatorSignature struct
const validatorSig = {
    validator: validator.address,
    r,
    s,
    v
};
```

## Verification Process

### On-Chain Verification

The `verifyBridgeSignatures` function:

1. **Computes the message hash** using EIP-712
2. **Checks if already processed** (replay protection)
3. **Validates nonce uniqueness** per source chain
4. **Verifies each signature**:
   - Recovers signer from signature
   - Confirms signer matches claimed validator
   - Checks validator is registered
   - Detects duplicate signatures
5. **Counts valid signatures** from distinct validators
6. **Validates threshold is met**
7. **Marks transaction as processed**

### View Function (No State Change)

Use `checkBridgeSignatures` to verify without state changes:
- Returns `(isValid, validCount)`
- Useful for pre-verification before submitting transaction

## Replay Protection

### Multi-Layer Protection

1. **Source Transaction ID**: Unique identifier from source chain
2. **Nonce**: Unique per bridge operation
3. **Source Chain ID**: Prevents cross-chain replay
4. **Domain Separator**: Chain-specific signing domain

### Processed Transaction Tracking

```solidity
mapping(bytes32 => bool) public processedBridgeTx;
mapping(uint256 => mapping(uint256 => bool)) public usedNonces;
```

## Integration Guide

### For Validators

1. **Monitor bridge events** on source chain
2. **Validate the bridge-out transaction**:
   - Verify transaction exists and is confirmed
   - Check amount and recipient
   - Validate chain IDs
3. **Sign the BridgeMessage**:
   - Use EIP-712 typed data signing
   - Include all fields accurately
4. **Submit signature** to aggregation service

### For Bridge Operators

1. **Collect signatures** from validators
2. **Aggregate signatures** into array
3. **Submit bridge-in transaction**:
   ```solidity
   bridge.bridgeIn(message, signatures);
   ```

### For Python Integration

```python
from prsm.economy.blockchain.bridge_security import (
    BridgeSecurityClient,
    BridgeMessage,
    ValidatorSignature,
    SignatureAggregator
)

# Initialize client
client = BridgeSecurityClient(web3, bridge_security_address, bridge_address)
await client.connect()

# Create message
message = client.create_bridge_message(
    recipient="0x...",
    amount=1000000000000000000,  # 1 token
    source_chain_id=1,
    source_tx_id=bytes32_hash,
    nonce=1
)

# Sign message (validator side)
signature = client.sign_bridge_message_eip712(message, private_key)

# Aggregate signatures
aggregator = SignatureAggregator(client)
aggregator.add_signature(signature1)
aggregator.add_signature(signature2)
aggregator.add_signature(signature3)

# Check if threshold met
if await aggregator.has_threshold():
    signatures = aggregator.get_signatures()
    # Submit to bridge
```

## Security Considerations

### Validator Key Management

- **Use HSM or secure key storage** for validator private keys
- **Implement key rotation** procedures
- **Monitor for key compromise** indicators

### Threshold Selection

- **Minimum 3-of-5** recommended for production
- **Higher thresholds** for larger amounts
- **Consider geographic distribution** of validators

### Signature Collection

- **Verify message accuracy** before signing
- **Never sign arbitrary hashes**
- **Log all signing operations** for audit

### Replay Attack Prevention

- **Always include unique nonce**
- **Verify sourceTxId uniqueness**
- **Check processed status** before submission

## Events

### BridgeTransactionVerified

```solidity
event BridgeTransactionVerified(
    bytes32 indexed messageHash,
    address indexed recipient,
    uint256 amount,
    uint256 sourceChainId,
    bytes32 indexed sourceTxId,
    uint256 nonce,
    uint256 signatureCount
);
```

### SignatureVerificationFailed

```solidity
event SignatureVerificationFailed(
    bytes32 indexed messageHash,
    address indexed validator,
    string reason
);
```

### DuplicateSignatureDetected

```solidity
event DuplicateSignatureDetected(
    bytes32 indexed messageHash,
    address indexed validator
);
```

## Error Codes

| Error | Description |
|-------|-------------|
| `InvalidSignatureThreshold` | Threshold is 0 or exceeds validator count |
| `InvalidValidatorAddress` | Validator address is zero |
| `ValidatorAlreadyExists` | Address is already a validator |
| `ValidatorNotFound` | Address is not a registered validator |
| `InsufficientSignatures` | Not enough valid signatures |
| `InvalidSignature` | Signature does not match claimed validator |
| `DuplicateSignature` | Same validator signed twice |
| `BridgeTransactionAlreadyProcessed` | Transaction was already verified |
| `NonceAlreadyUsed` | Nonce has been used for this source chain |

## Gas Optimization

### Signature Verification

- **Batch verification** reduces per-signature gas cost
- **Early exit** on threshold met
- **Efficient duplicate detection** using mapping

### Estimated Gas Costs

| Operation | Gas (approx) |
|-----------|--------------|
| Verify 3 signatures | ~120,000 |
| Verify 5 signatures | ~180,000 |
| Add validator | ~45,000 |
| Remove validator | ~35,000 |
| Update threshold | ~30,000 |

## Upgrade Path

The contracts use UUPS proxy pattern for upgradeability:

1. **Deploy new implementation**
2. **Call `upgradeTo(newImplementation)`**
3. **State is preserved** automatically

Only the admin role can authorize upgrades.

## Testing

Run the test suite:

```bash
cd contracts
npm test test/BridgeSecurity.test.js
```

Test coverage includes:
- Signature verification (valid/invalid)
- Threshold validation
- Replay protection
- Duplicate detection
- Validator management
- Upgrade functionality

## References

- [EIP-712: Typed Structured Data Signing](https://eips.ethereum.org/EIPS/eip-712)
- [EIP-191: Signed Data Standard](https://eips.ethereum.org/EIPS/eip-191)
- [OpenZeppelin ECDSA Library](https://docs.openzeppelin.com/contracts/4.x/api/utils#ECDSA)
