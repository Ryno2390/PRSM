# DAG Transaction Cryptographic Signatures

## Overview

PRSM's DAG ledger implements Ed25519 cryptographic signature verification for all transactions to ensure:
- **Authentication**: Verify the identity of transaction senders
- **Non-repudiation**: Prevent senders from denying their transactions
- **Integrity**: Detect any tampering with transaction data
- **Security**: Prevent impersonation and fraud

## Signature Algorithm

### Ed25519 (Edwards-curve Digital Signature Algorithm)

We use Ed25519 for transaction signatures due to its:
- **Fast signing and verification**: Optimized for performance
- **Small key sizes**: 32-byte private and public keys
- **Compact signatures**: 64-byte signatures
- **Strong security**: 128-bit security level, resistant to side-channel attacks
- **Deterministic signatures**: Same message and key always produce the same signature

### Technical Specifications

| Property | Value |
|----------|-------|
| Algorithm | Ed25519 (EdDSA on Curve25519) |
| Private Key Size | 32 bytes (256 bits) |
| Public Key Size | 32 bytes (256 bits) |
| Signature Size | 64 bytes (512 bits) |
| Hash Function | SHA-256 (for transaction data) |
| Encoding | Base64 (signatures), Hex (public keys) |

## Architecture

### Components

1. **`DAGSignatureManager`** ([`prsm/core/cryptography/dag_signatures.py`](prsm/core/cryptography/dag_signatures.py))
   - Key pair generation
   - Transaction signing
   - Signature verification
   - Key serialization/deserialization

2. **`DAGLedger`** ([`prsm/node/dag_ledger.py`](prsm/node/dag_ledger.py))
   - Signature verification on transaction submission
   - Public key registry for wallets
   - Configurable verification (can be disabled for testing)

3. **`KeyPair`** dataclass
   - Holds Ed25519 private and public keys
   - Provides serialization methods

### Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client/Wallet │     │   DAG Ledger     │     │  Signature      │
│                 │     │                  │     │  Manager        │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                        │
         │ 1. Generate KeyPair   │                        │
         │───────────────────────────────────────────────>│
         │                       │                        │
         │ 2. Create Transaction │                        │
         │                       │                        │
         │ 3. Sign Transaction   │                        │
         │───────────────────────────────────────────────>│
         │                       │                        │
         │ 4. Submit Transaction │                        │
         │──────────────────────>│                        │
         │                       │                        │
         │                       │ 5. Verify Signature    │
         │                       │───────────────────────>│
         │                       │                        │
         │                       │ 6. Return Result       │
         │                       │<───────────────────────│
         │                       │                        │
         │ 7. Transaction        │                        │
         │    Accepted/Rejected  │                        │
         │<──────────────────────│                        │
```

## Usage

### 1. Generate a Key Pair

```python
from prsm.core.cryptography import create_signing_key_pair

# Generate a new Ed25519 key pair
key_pair = create_signing_key_pair()

# Get public key for registration
public_key_hex = key_pair.get_public_key_hex()
private_key_hex = key_pair.get_private_key_hex()  # Store securely!
```

### 2. Register Public Key with Wallet

```python
from prsm.node.dag_ledger import DAGLedger

ledger = DAGLedger(db_path=":memory:", verify_signatures=True)
await ledger.initialize()

# Create wallet with public key
await ledger.create_wallet(
    wallet_id="user_wallet_123",
    display_name="User Wallet",
    public_key=public_key_hex
)

# Or register public key for existing wallet
await ledger.register_wallet_public_key("user_wallet_123", public_key_hex)
```

### 3. Sign a Transaction

```python
from prsm.core.cryptography import sign_hash

# Create transaction data
tx_data = {
    "tx_type": "transfer",
    "amount": 100.0,
    "from_wallet": "user_wallet_123",
    "to_wallet": "recipient_wallet",
    "timestamp": 1234567890.0,
    "parent_ids": ["parent_tx_1", "parent_tx_2"],
}

# Calculate transaction hash
import hashlib
import json
tx_hash = hashlib.sha256(
    json.dumps(tx_data, sort_keys=True).encode()
).hexdigest()

# Sign the hash
signature = sign_hash(tx_hash, key_pair.private_key)
```

### 4. Submit Signed Transaction

```python
# Submit transaction with signature
tx = await ledger.transfer(
    from_wallet="user_wallet_123",
    to_wallet="recipient_wallet",
    amount=100.0,
    signature=signature,
    public_key=public_key_hex,  # Optional if already registered
)
```

### 5. Verify a Signature (Manual)

```python
from prsm.core.cryptography import verify_hash_signature

# Verify signature manually
is_valid = verify_hash_signature(tx_hash, signature, public_key)
print(f"Signature valid: {is_valid}")
```

## Transaction Signing Process

### What Gets Signed

The transaction hash is computed from the following fields:
- `tx_id`: Unique transaction identifier
- `tx_type`: Type of transaction (TRANSFER, COMPUTE_PAYMENT, etc.)
- `amount`: Transaction amount
- `from_wallet`: Sender wallet ID
- `to_wallet`: Recipient wallet ID
- `timestamp`: Unix timestamp
- `parent_ids`: List of parent transaction IDs

```python
def hash(self) -> str:
    data = {
        "tx_id": self.tx_id,
        "tx_type": self.tx_type.value,
        "amount": self.amount,
        "from_wallet": self.from_wallet,
        "to_wallet": self.to_wallet,
        "timestamp": self.timestamp,
        "parent_ids": self.parent_ids,
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()
```

### Signature Verification Rules

| Transaction Type | from_wallet | Signature Required |
|-----------------|-------------|-------------------|
| GENESIS | None | No |
| WELCOME_GRANT | None | No |
| COMPUTE_EARNING | None | No |
| STORAGE_REWARD | None | No |
| TRANSFER | Set | **Yes** |
| COMPUTE_PAYMENT | Set | **Yes** |
| CONTENT_ROYALTY | Set | **Yes** |
| APPROVAL | Set | **Yes** |

## Security Considerations

### Private Key Security

1. **Never share private keys**: Private keys must be kept secret
2. **Secure storage**: Store private keys encrypted at rest
3. **Key rotation**: Consider implementing key rotation policies
4. **Backup**: Securely backup private keys to prevent loss

### Signature Verification

1. **Always verify**: Never accept transactions without signature verification in production
2. **Public key validation**: Ensure public keys are properly registered
3. **Replay protection**: Transaction IDs prevent replay attacks

### Testing Mode

For testing purposes, signature verification can be disabled:

```python
# Production configuration
ledger = DAGLedger(verify_signatures=True)

# Testing configuration (NOT for production!)
ledger = DAGLedger(verify_signatures=False)
```

## Error Handling

### Exception Types

| Exception | Description |
|-----------|-------------|
| `SignatureError` | Base class for signature errors |
| `InvalidSignatureError` | Signature verification failed |
| `MissingSignatureError` | Required signature not provided |
| `MissingPublicKeyError` | Public key not available for verification |

### Example Error Handling

```python
from prsm.core.cryptography import (
    InvalidSignatureError,
    MissingSignatureError,
)

try:
    tx = await ledger.transfer(
        from_wallet="sender",
        to_wallet="receiver",
        amount=100.0,
        signature=signature,
    )
except MissingSignatureError:
    print("Transaction requires a signature")
except InvalidSignatureError:
    print("Signature verification failed - possible fraud attempt")
```

## Database Schema

### Wallets Table

```sql
CREATE TABLE wallets (
    wallet_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    public_key TEXT,  -- Ed25519 public key (hex)
    created_at REAL NOT NULL
);
```

### Transactions Table

```sql
CREATE TABLE dag_transactions (
    tx_id TEXT PRIMARY KEY,
    tx_type TEXT NOT NULL,
    amount REAL NOT NULL,
    from_wallet TEXT,
    to_wallet TEXT NOT NULL,
    timestamp REAL NOT NULL,
    signature TEXT,    -- Base64-encoded Ed25519 signature
    public_key TEXT,   -- Ed25519 public key (hex)
    description TEXT DEFAULT '',
    parent_ids TEXT NOT NULL,
    cumulative_weight INTEGER DEFAULT 1,
    confirmation_level REAL DEFAULT 0.0,
    hash TEXT NOT NULL
);
```

## API Reference

### DAGSignatureManager

```python
class DAGSignatureManager:
    @staticmethod
    def generate_key_pair() -> KeyPair:
        """Generate a new Ed25519 key pair."""
    
    @staticmethod
    def sign_transaction_hash(tx_hash: str, private_key: Ed25519PrivateKey) -> str:
        """Sign a transaction hash, returns base64-encoded signature."""
    
    @staticmethod
    def verify_signature(tx_hash: str, signature_b64: str, public_key: Ed25519PublicKey) -> bool:
        """Verify a transaction signature."""
    
    @staticmethod
    def load_public_key_from_hex(hex_string: str) -> Ed25519PublicKey:
        """Load public key from hex string."""
    
    @staticmethod
    def load_private_key_from_hex(hex_string: str) -> Ed25519PrivateKey:
        """Load private key from hex string."""
```

### KeyPair

```python
@dataclass
class KeyPair:
    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey
    
    @classmethod
    def generate(cls) -> KeyPair:
        """Generate a new key pair."""
    
    def get_private_key_hex(self) -> str:
        """Get private key as hex string."""
    
    def get_public_key_hex(self) -> str:
        """Get public key as hex string."""
    
    def get_private_key_base64(self) -> str:
        """Get private key as base64 string."""
    
    def get_public_key_base64(self) -> str:
        """Get public key as base64 string."""
```

### DAGLedger Signature Methods

```python
class DAGLedger:
    def __init__(self, db_path: str = ":memory:", verify_signatures: bool = True):
        """Initialize ledger with optional signature verification."""
    
    async def create_wallet(
        self, 
        wallet_id: str, 
        display_name: str = "",
        public_key: Optional[str] = None
    ) -> None:
        """Create wallet with optional public key."""
    
    async def register_wallet_public_key(self, wallet_id: str, public_key: str) -> None:
        """Register public key for existing wallet."""
    
    def get_wallet_public_key(self, wallet_id: str) -> Optional[str]:
        """Get registered public key for wallet."""
    
    async def transfer(
        self,
        from_wallet: str,
        to_wallet: str,
        amount: float,
        description: str = "",
        signature: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> DAGTransaction:
        """Transfer with signature verification."""
```

## Testing

Run the signature verification tests:

```bash
pytest tests/unit/node/test_dag_ledger.py -v
```

Test categories:
- `TestDAGSignatureManager`: Core signature operations
- `TestKeyPair`: Key pair serialization
- `TestConvenienceFunctions`: Module-level functions
- `TestDAGTransactionSigning`: Transaction hash and signing
- `TestDAGLedgerSignatureVerification`: Full ledger integration
- `TestSignatureErrorHandling`: Error cases

## Future Enhancements

1. **Batch Signature Verification**: Verify multiple signatures efficiently
2. **Multi-signature Support**: Require multiple signatures for high-value transactions
3. **Key Rotation**: Automated key rotation with history
4. **Hardware Security Module (HSM) Integration**: Store keys in HSM
5. **Threshold Signatures**: Distribute signing authority
6. **Post-Quantum Signatures**: Migrate to quantum-resistant algorithms

## References

- [Ed25519 RFC 8032](https://datatracker.ietf.org/doc/html/rfc8032)
- [cryptography.io Ed25519 Documentation](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/)
- [IOTA Tangle Whitepaper](https://iota.org/foundation/whitepapers)
