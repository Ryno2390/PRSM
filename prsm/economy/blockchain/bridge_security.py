"""
Bridge Security Client
======================

Python client for interacting with PRSM's multi-signature bridge verification system.
Provides utilities for EIP-712 message signing and verification.

Features:
- EIP-712 typed structured data signing
- Multi-signature coordination
- Validator signature aggregation
- Bridge message construction and verification

Usage:
    from prsm.economy.blockchain.bridge_security import BridgeSecurityClient
    
    client = BridgeSecurityClient(web3_provider, bridge_security_address)
    
    # Create a bridge message
    message = client.create_bridge_message(
        recipient="0x...",
        amount=1000000000000000000,  # 1 token in wei
        source_chain_id=1,
        source_tx_id=bytes32_hash,
        nonce=1
    )
    
    # Sign the message
    signature = client.sign_bridge_message(message, private_key)
    
    # Verify signatures on-chain
    result = await client.verify_signatures(message, [signature])
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import hashlib
import structlog

# Web3 imports - handle gracefully if not installed
try:
    from web3 import Web3
    from web3.contract import Contract
    from eth_account import Account
    from eth_account.messages import encode_typed_data
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False
    Web3 = None
    Contract = None
    Account = None

logger = structlog.get_logger(__name__)


# ============ Constants ============

# EIP-712 Domain Typehash
DOMAIN_TYPEHASH = bytes.fromhex(
    "8b73c3c69bb8fe3d512ecc4cf759cc79239f7b179b0ffacaa9a75d522b39400f"
)

# BridgeMessage Typehash (keccak256 of the type string)
BRIDGE_MESSAGE_TYPEHASH = bytes.fromhex(
    # keccak256("BridgeMessage(address recipient,uint256 amount,uint256 sourceChainId,bytes32 sourceTxId,uint256 nonce)")
    "a8b6c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
)

# Compute actual typehash
import hashlib
BRIDGE_MESSAGE_TYPE_STRING = "BridgeMessage(address recipient,uint256 amount,uint256 sourceChainId,bytes32 sourceTxId,uint256 nonce)"
BRIDGE_MESSAGE_TYPEHASH = Web3.keccak(text=BRIDGE_MESSAGE_TYPE_STRING) if HAS_WEB3 else hashlib.sha256(BRIDGE_MESSAGE_TYPE_STRING.encode()).digest()


class BridgeMessageField(Enum):
    """Fields in a BridgeMessage struct"""
    RECIPIENT = "recipient"
    AMOUNT = "amount"
    SOURCE_CHAIN_ID = "sourceChainId"
    SOURCE_TX_ID = "sourceTxId"
    NONCE = "nonce"


@dataclass
class BridgeMessage:
    """
    Bridge message structure for cross-chain transfers.
    
    Attributes:
        recipient: Address to receive tokens on destination chain
        amount: Amount of tokens to bridge (in wei)
        source_chain_id: Chain ID where tokens were locked/burned
        source_tx_id: Transaction ID on source chain
        nonce: Unique nonce for replay protection
    """
    recipient: str
    amount: int
    source_chain_id: int
    source_tx_id: bytes
    nonce: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for EIP-712 encoding"""
        return {
            "recipient": self.recipient,
            "amount": self.amount,
            "sourceChainId": self.source_chain_id,
            "sourceTxId": self.source_tx_id,
            "nonce": self.nonce,
        }
    
    def to_tuple(self) -> Tuple[str, int, int, bytes, int]:
        """Convert to tuple for ABI encoding"""
        return (
            self.recipient,
            self.amount,
            self.source_chain_id,
            self.source_tx_id,
            self.nonce,
        )


@dataclass
class ValidatorSignature:
    """
    Validator signature with metadata.
    
    Attributes:
        validator: Address of the validator who signed
        r: ECDSA signature r component
        s: ECDSA signature s component
        v: ECDSA recovery id
    """
    validator: str
    r: bytes
    s: bytes
    v: int
    
    @classmethod
    def from_bytes(cls, validator: str, signature: bytes) -> "ValidatorSignature":
        """
        Create from raw signature bytes.
        
        Args:
            validator: Address of the validator
            signature: 65-byte signature (r: 32, s: 32, v: 1)
            
        Returns:
            ValidatorSignature instance
        """
        if len(signature) != 65:
            raise ValueError(f"Invalid signature length: {len(signature)}, expected 65")
        
        r = signature[:32]
        s = signature[32:64]
        v = signature[64]
        
        # Handle Ethereum's v value (27 or 28)
        if v < 27:
            v += 27
        
        return cls(validator=validator, r=r, s=s, v=v)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for contract call"""
        return {
            "validator": self.validator,
            "r": self.r,
            "s": self.s,
            "v": self.v,
        }
    
    def to_tuple(self) -> Tuple[str, bytes, bytes, int]:
        """Convert to tuple for ABI encoding"""
        return (self.validator, self.r, self.s, self.v)


class BridgeSecurityClient:
    """
    Client for interacting with PRSM's Bridge Security contract.
    
    Provides methods for:
    - Creating and signing bridge messages
    - Aggregating validator signatures
    - Verifying signatures on-chain
    - Managing validator registry
    """
    
    def __init__(
        self,
        web3: "Web3",
        bridge_security_address: str,
        bridge_address: Optional[str] = None,
    ):
        """
        Initialize the bridge security client.
        
        Args:
            web3: Web3 instance connected to the network
            bridge_security_address: Address of the BridgeSecurity contract
            bridge_address: Optional address of the FTNSBridge contract
        """
        if not HAS_WEB3:
            raise ImportError("web3 is required. Install with: pip install web3")
        
        self.web3 = web3
        self.bridge_security_address = bridge_security_address
        self.bridge_address = bridge_address
        
        # Load contract ABIs
        self._bridge_security_abi = self._get_bridge_security_abi()
        self._bridge_abi = self._get_bridge_abi()
        
        # Contract instances
        self.bridge_security: Optional[Contract] = None
        self.bridge: Optional[Contract] = None
        
        # Domain separator cache
        self._domain_separator: Optional[bytes] = None
        self._chain_id: Optional[int] = None
    
    def _get_bridge_security_abi(self) -> List[Dict]:
        """Get the BridgeSecurity contract ABI"""
        return [
            # Read functions
            {
                "inputs": [],
                "name": "signatureThreshold",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "totalValidators",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "address", "name": "", "type": "address"}],
                "name": "isValidator",
                "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
                "name": "processedBridgeTx",
                "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "domainSeparator",
                "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "BRIDGE_MESSAGE_TYPEHASH_V2",
                "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "components": [
                            {"internalType": "address", "name": "recipient", "type": "address"},
                            {"internalType": "uint256", "name": "amount", "type": "uint256"},
                            {"internalType": "uint256", "name": "sourceChainId", "type": "uint256"},
                            {"internalType": "bytes32", "name": "sourceTxId", "type": "bytes32"},
                            {"internalType": "uint256", "name": "nonce", "type": "uint256"}
                        ],
                        "internalType": "struct BridgeSecurity.BridgeMessage",
                        "name": "message",
                        "type": "tuple"
                    },
                    {
                        "components": [
                            {"internalType": "address", "name": "validator", "type": "address"},
                            {"internalType": "bytes32", "name": "r", "type": "bytes32"},
                            {"internalType": "bytes32", "name": "s", "type": "bytes32"},
                            {"internalType": "uint8", "name": "v", "type": "uint8"}
                        ],
                        "internalType": "struct BridgeSecurity.ValidatorSignature[]",
                        "name": "signatures",
                        "type": "tuple[]"
                    }
                ],
                "name": "verifyBridgeSignatures",
                "outputs": [
                    {"internalType": "bytes32", "name": "messageHash", "type": "bytes32"},
                    {"internalType": "bool", "name": "isValid", "type": "bool"}
                ],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "components": [
                            {"internalType": "address", "name": "recipient", "type": "address"},
                            {"internalType": "uint256", "name": "amount", "type": "uint256"},
                            {"internalType": "uint256", "name": "sourceChainId", "type": "uint256"},
                            {"internalType": "bytes32", "name": "sourceTxId", "type": "bytes32"},
                            {"internalType": "uint256", "name": "nonce", "type": "uint256"}
                        ],
                        "internalType": "struct BridgeSecurity.BridgeMessage",
                        "name": "message",
                        "type": "tuple"
                    },
                    {
                        "components": [
                            {"internalType": "address", "name": "validator", "type": "address"},
                            {"internalType": "bytes32", "name": "r", "type": "bytes32"},
                            {"internalType": "bytes32", "name": "s", "type": "bytes32"},
                            {"internalType": "uint8", "name": "v", "type": "uint8"}
                        ],
                        "internalType": "struct BridgeSecurity.ValidatorSignature[]",
                        "name": "signatures",
                        "type": "tuple[]"
                    }
                ],
                "name": "checkBridgeSignatures",
                "outputs": [
                    {"internalType": "bool", "name": "isValid", "type": "bool"},
                    {"internalType": "uint256", "name": "validCount", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "components": [
                            {"internalType": "address", "name": "recipient", "type": "address"},
                            {"internalType": "uint256", "name": "amount", "type": "uint256"},
                            {"internalType": "uint256", "name": "sourceChainId", "type": "uint256"},
                            {"internalType": "bytes32", "name": "sourceTxId", "type": "bytes32"},
                            {"internalType": "uint256", "name": "nonce", "type": "uint256"}
                        ],
                        "internalType": "struct BridgeSecurity.BridgeMessage",
                        "name": "message",
                        "type": "tuple"
                    }
                ],
                "name": "hashBridgeMessage",
                "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
                "stateMutability": "view",
                "type": "function"
            },
            # Events
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "messageHash", "type": "bytes32"},
                    {"indexed": True, "name": "recipient", "type": "address"},
                    {"indexed": False, "name": "amount", "type": "uint256"},
                    {"indexed": False, "name": "sourceChainId", "type": "uint256"},
                    {"indexed": True, "name": "sourceTxId", "type": "bytes32"},
                    {"indexed": False, "name": "nonce", "type": "uint256"},
                    {"indexed": False, "name": "signatureCount", "type": "uint256"}
                ],
                "name": "BridgeTransactionVerified",
                "type": "event"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "messageHash", "type": "bytes32"},
                    {"indexed": True, "name": "validator", "type": "address"},
                    {"indexed": False, "name": "reason", "type": "string"}
                ],
                "name": "SignatureVerificationFailed",
                "type": "event"
            }
        ]
    
    def _get_bridge_abi(self) -> List[Dict]:
        """Get the FTNSBridge contract ABI"""
        return [
            {
                "inputs": [
                    {
                        "components": [
                            {"internalType": "address", "name": "recipient", "type": "address"},
                            {"internalType": "uint256", "name": "amount", "type": "uint256"},
                            {"internalType": "uint256", "name": "sourceChainId", "type": "uint256"},
                            {"internalType": "bytes32", "name": "sourceTxId", "type": "bytes32"},
                            {"internalType": "uint256", "name": "nonce", "type": "uint256"}
                        ],
                        "internalType": "struct BridgeSecurity.BridgeMessage",
                        "name": "message",
                        "type": "tuple"
                    },
                    {
                        "components": [
                            {"internalType": "address", "name": "validator", "type": "address"},
                            {"internalType": "bytes32", "name": "r", "type": "bytes32"},
                            {"internalType": "bytes32", "name": "s", "type": "bytes32"},
                            {"internalType": "uint8", "name": "v", "type": "uint8"}
                        ],
                        "internalType": "struct BridgeSecurity.ValidatorSignature[]",
                        "name": "signatures",
                        "type": "tuple[]"
                    }
                ],
                "name": "bridgeIn",
                "outputs": [{"internalType": "bool", "name": "success", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amount", "type": "uint256"},
                    {"internalType": "uint256", "name": "destinationChain", "type": "uint256"}
                ],
                "name": "bridgeOut",
                "outputs": [{"internalType": "bytes32", "name": "transactionId", "type": "bytes32"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
    
    async def connect(self) -> None:
        """Connect to the contracts"""
        self.bridge_security = self.web3.eth.contract(
            address=self.bridge_security_address,
            abi=self._bridge_security_abi
        )
        
        if self.bridge_address:
            self.bridge = self.web3.eth.contract(
                address=self.bridge_address,
                abi=self._bridge_abi
            )
        
        # Cache domain separator
        self._domain_separator = await self.get_domain_separator()
        self._chain_id = self.web3.eth.chain_id
        
        logger.info(
            "Connected to BridgeSecurity contract",
            address=self.bridge_security_address,
            chain_id=self._chain_id
        )
    
    # ============ Read Operations ============
    
    async def get_signature_threshold(self) -> int:
        """Get the current signature threshold (M in M-of-N)"""
        return await self.bridge_security.functions.signatureThreshold().call()
    
    async def get_total_validators(self) -> int:
        """Get the total number of registered validators (N in M-of-N)"""
        return await self.bridge_security.functions.totalValidators().call()
    
    async def is_validator(self, address: str) -> bool:
        """Check if an address is a registered validator"""
        return await self.bridge_security.functions.isValidator(address).call()
    
    async def is_processed(self, message_hash: bytes) -> bool:
        """Check if a bridge transaction has been processed"""
        return await self.bridge_security.functions.processedBridgeTx(message_hash).call()
    
    async def get_domain_separator(self) -> bytes:
        """Get the EIP-712 domain separator"""
        return await self.bridge_security.functions.domainSeparator().call()
    
    async def get_typehash(self) -> bytes:
        """Get the BridgeMessage typehash"""
        return await self.bridge_security.functions.BRIDGE_MESSAGE_TYPEHASH_V2().call()
    
    async def get_configuration(self) -> Tuple[int, int]:
        """
        Get the current bridge security configuration.
        
        Returns:
            Tuple of (threshold, total_validators)
        """
        threshold, total = await self.bridge_security.functions.getConfiguration().call()
        return threshold, total
    
    # ============ Message Operations ============
    
    def create_bridge_message(
        self,
        recipient: str,
        amount: int,
        source_chain_id: int,
        source_tx_id: bytes,
        nonce: int
    ) -> BridgeMessage:
        """
        Create a bridge message.
        
        Args:
            recipient: Address to receive tokens
            amount: Amount of tokens (in wei)
            source_chain_id: Chain ID where tokens were locked
            source_tx_id: Transaction ID on source chain
            nonce: Unique nonce for replay protection
            
        Returns:
            BridgeMessage instance
        """
        # Validate inputs
        if not self.web3.is_address(recipient):
            raise ValueError(f"Invalid recipient address: {recipient}")
        
        if amount <= 0:
            raise ValueError(f"Amount must be positive: {amount}")
        
        if len(source_tx_id) != 32:
            raise ValueError(f"source_tx_id must be 32 bytes, got {len(source_tx_id)}")
        
        return BridgeMessage(
            recipient=self.web3.to_checksum_address(recipient),
            amount=amount,
            source_chain_id=source_chain_id,
            source_tx_id=source_tx_id,
            nonce=nonce
        )
    
    async def hash_bridge_message(self, message: BridgeMessage) -> bytes:
        """
        Compute the EIP-712 hash of a bridge message.
        
        Args:
            message: Bridge message to hash
            
        Returns:
            32-byte EIP-712 typed hash
        """
        return await self.bridge_security.functions.hashBridgeMessage(
            message.to_tuple()
        ).call()
    
    def compute_struct_hash(self, message: BridgeMessage) -> bytes:
        """
        Compute the struct hash for a bridge message (without domain separator).
        
        This is useful for off-chain signature verification.
        
        Args:
            message: Bridge message
            
        Returns:
            32-byte struct hash
        """
        # Encode the struct according to EIP-712
        typehash = Web3.keccak(text=BRIDGE_MESSAGE_TYPE_STRING)
        
        encoded = self.web3.codec.encode(
            ['bytes32', 'address', 'uint256', 'uint256', 'bytes32', 'uint256'],
            [
                typehash,
                self.web3.to_checksum_address(message.recipient),
                message.amount,
                message.source_chain_id,
                message.source_tx_id,
                message.nonce
            ]
        )
        
        return Web3.keccak(encoded)
    
    def compute_typed_hash(self, message: BridgeMessage, domain_separator: Optional[bytes] = None) -> bytes:
        """
        Compute the full EIP-712 typed hash.
        
        Args:
            message: Bridge message
            domain_separator: Optional domain separator (uses cached if not provided)
            
        Returns:
            32-byte EIP-712 typed hash
        """
        if domain_separator is None:
            if self._domain_separator is None:
                raise ValueError("Domain separator not cached. Call connect() first.")
            domain_separator = self._domain_separator
        
        struct_hash = self.compute_struct_hash(message)
        
        # EIP-712: keccak256("\x19\x01" ‖ domainSeparator ‖ structHash)
        return Web3.keccak(
            b'\x19\x01' + domain_separator + struct_hash
        )
    
    # ============ Signing Operations ============
    
    def sign_bridge_message(
        self,
        message: BridgeMessage,
        private_key: str,
        domain_separator: Optional[bytes] = None
    ) -> ValidatorSignature:
        """
        Sign a bridge message with a private key.
        
        Args:
            message: Bridge message to sign
            private_key: Private key to sign with (hex string)
            domain_separator: Optional domain separator
            
        Returns:
            ValidatorSignature with the signature components
        """
        # Get the typed hash
        typed_hash = self.compute_typed_hash(message, domain_separator)
        
        # Sign the hash
        account = Account.from_key(private_key)
        signature = account.sign_message(encode_typed_data(typed_hash))
        
        # Create ValidatorSignature
        return ValidatorSignature(
            validator=account.address,
            r=signature.r.to_bytes(32, 'big'),
            s=signature.s.to_bytes(32, 'big'),
            v=signature.v
        )
    
    def sign_bridge_message_eip712(
        self,
        message: BridgeMessage,
        private_key: str
    ) -> ValidatorSignature:
        """
        Sign a bridge message using full EIP-712 typed data.
        
        This method uses the full EIP-712 typed data structure for signing,
        which is compatible with wallets like MetaMask.
        
        Args:
            message: Bridge message to sign
            private_key: Private key to sign with
            
        Returns:
            ValidatorSignature with the signature components
        """
        # Build the typed data structure
        typed_data = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"}
                ],
                "BridgeMessage": [
                    {"name": "recipient", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "sourceChainId", "type": "uint256"},
                    {"name": "sourceTxId", "type": "bytes32"},
                    {"name": "nonce", "type": "uint256"}
                ]
            },
            "primaryType": "BridgeMessage",
            "domain": {
                "name": "PRSM Bridge Security",
                "version": "1",
                "chainId": self._chain_id or 1,
                "verifyingContract": self.bridge_security_address
            },
            "message": message.to_dict()
        }
        
        # Sign the typed data
        account = Account.from_key(private_key)
        signed = account.sign_message(encode_typed_data(typed_data))
        
        return ValidatorSignature(
            validator=account.address,
            r=signed.r.to_bytes(32, 'big'),
            s=signed.s.to_bytes(32, 'big'),
            v=signed.v
        )
    
    # ============ Verification Operations ============
    
    async def verify_signatures(
        self,
        message: BridgeMessage,
        signatures: List[ValidatorSignature]
    ) -> Tuple[bytes, bool]:
        """
        Verify multi-signatures for a bridge message on-chain.
        
        This calls the contract's verifyBridgeSignatures function which:
        1. Verifies each signature cryptographically
        2. Checks that signers are registered validators
        3. Ensures no duplicate signatures
        4. Validates the threshold is met
        5. Marks the transaction as processed
        
        Args:
            message: Bridge message to verify
            signatures: List of validator signatures
            
        Returns:
            Tuple of (message_hash, is_valid)
        """
        # Convert signatures to contract format
        sig_tuples = [sig.to_tuple() for sig in signatures]
        
        # Call the contract
        message_hash, is_valid = await self.bridge_security.functions.verifyBridgeSignatures(
            message.to_tuple(),
            sig_tuples
        ).call()
        
        return message_hash, is_valid
    
    async def check_signatures(
        self,
        message: BridgeMessage,
        signatures: List[ValidatorSignature]
    ) -> Tuple[bool, int]:
        """
        Check signatures without state changes (view function).
        
        Use this to verify signatures before submitting a transaction.
        
        Args:
            message: Bridge message to check
            signatures: List of validator signatures
            
        Returns:
            Tuple of (is_valid, valid_count)
        """
        sig_tuples = [sig.to_tuple() for sig in signatures]
        
        is_valid, valid_count = await self.bridge_security.functions.checkBridgeSignatures(
            message.to_tuple(),
            sig_tuples
        ).call()
        
        return is_valid, valid_count
    
    # ============ Bridge Operations ============
    
    async def bridge_in(
        self,
        message: BridgeMessage,
        signatures: List[ValidatorSignature],
        from_address: str
    ) -> str:
        """
        Execute a bridge-in transaction.
        
        Args:
            message: Bridge message
            signatures: Validator signatures
            from_address: Address to send the transaction from
            
        Returns:
            Transaction hash
        """
        if not self.bridge:
            raise ValueError("Bridge contract not configured")
        
        sig_tuples = [sig.to_tuple() for sig in signatures]
        
        # Build transaction
        tx = await self.bridge.functions.bridgeIn(
            message.to_tuple(),
            sig_tuples
        ).build_transaction({
            'from': from_address,
            'gas': 500000,  # Estimate gas
        })
        
        return tx
    
    # ============ Utility Functions ============
    
    def recover_signer(
        self,
        message: BridgeMessage,
        signature: ValidatorSignature,
        domain_separator: Optional[bytes] = None
    ) -> str:
        """
        Recover the signer address from a signature.
        
        Args:
            message: Bridge message that was signed
            signature: The signature
            domain_separator: Optional domain separator
            
        Returns:
            Recovered address
        """
        typed_hash = self.compute_typed_hash(message, domain_separator)
        
        # Reconstruct signature bytes
        sig_bytes = signature.r + signature.s + bytes([signature.v])
        
        # Recover address
        recovered = self.web3.eth.account.recover_message(
            encode_typed_data(typed_hash),
            signature=sig_bytes
        )
        
        return recovered
    
    def validate_signature(
        self,
        message: BridgeMessage,
        signature: ValidatorSignature,
        domain_separator: Optional[bytes] = None
    ) -> bool:
        """
        Validate that a signature matches the claimed validator.
        
        Args:
            message: Bridge message
            signature: Signature to validate
            domain_separator: Optional domain separator
            
        Returns:
            True if signature is valid for the claimed validator
        """
        recovered = self.recover_signer(message, signature, domain_separator)
        return recovered.lower() == signature.validator.lower()


class SignatureAggregator:
    """
    Utility class for aggregating signatures from multiple validators.
    
    Use this to collect signatures off-chain before submitting to the bridge.
    """
    
    def __init__(self, client: BridgeSecurityClient):
        self.client = client
        self._signatures: Dict[str, ValidatorSignature] = {}  # validator -> signature
    
    def add_signature(self, signature: ValidatorSignature) -> bool:
        """
        Add a signature to the aggregator.
        
        Args:
            signature: Validator signature to add
            
        Returns:
            True if signature was added (not duplicate)
        """
        validator_lower = signature.validator.lower()
        
        if validator_lower in self._signatures:
            logger.warning(
                "Duplicate signature ignored",
                validator=signature.validator
            )
            return False
        
        self._signatures[validator_lower] = signature
        return True
    
    def get_signatures(self) -> List[ValidatorSignature]:
        """Get all collected signatures"""
        return list(self._signatures.values())
    
    def get_signature_count(self) -> int:
        """Get the number of collected signatures"""
        return len(self._signatures)
    
    async def has_threshold(self) -> bool:
        """Check if we have enough signatures to meet the threshold"""
        threshold = await self.client.get_signature_threshold()
        return len(self._signatures) >= threshold
    
    def clear(self) -> None:
        """Clear all collected signatures"""
        self._signatures.clear()


# ============ Helper Functions ============

def generate_nonce() -> int:
    """Generate a unique nonce for a bridge message"""
    import time
    import random
    
    # Combine timestamp with random component
    timestamp = int(time.time() * 1000)
    random_component = random.randint(0, 2**32 - 1)
    
    return (timestamp << 32) | random_component


def bytes32_from_hex(hex_string: str) -> bytes:
    """Convert a hex string to a 32-byte value"""
    # Remove 0x prefix if present
    if hex_string.startswith('0x'):
        hex_string = hex_string[2:]
    
    # Pad to 64 characters (32 bytes)
    hex_string = hex_string.zfill(64)
    
    return bytes.fromhex(hex_string)


def bytes32_from_string(string: str) -> bytes:
    """Convert a string to a 32-byte hash"""
    return Web3.keccak(text=string) if HAS_WEB3 else hashlib.sha256(string.encode()).digest()
