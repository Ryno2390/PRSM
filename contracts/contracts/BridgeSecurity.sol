// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/EIP712.sol";

/**
 * @title BridgeSecurity
 * @dev Multi-signature verification contract for cross-chain bridge operations
 * 
 * Implements EIP-712 typed structured data signing for secure validator signatures
 * with replay protection and configurable M-of-N threshold requirements.
 * 
 * @author PRSM Protocol
 */
contract BridgeSecurity is 
    Initializable,
    AccessControlUpgradeable,
    UUPSUpgradeable,
    ReentrancyGuardUpgradeable
{
    using ECDSA for bytes32;

    // ============ Role Definitions ============
    
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");
    bytes32 public constant BRIDGE_OPERATOR_ROLE = keccak256("BRIDGE_OPERATOR_ROLE");
    
    // ============ EIP-712 Type Definitions ============
    
    // EIP-712 Domain Separator
    // keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)")
    bytes32 public constant DOMAIN_TYPEHASH = 0x8b73c3c69bb8fe3d512ecc4cf759cc79239f7b179b0ffacaa9a75d522b39400f;
    
    // BridgeMessage typehash for EIP-712 structured signing
    // keccak256("BridgeMessage(address recipient,uint256 amount,uint256 sourceChain,bytes32 sourceTransactionId,uint256 nonce)")
    bytes32 public constant BRIDGE_MESSAGE_TYPEHASH = 0xd479a2d9a9ff3b1db67c6b3d7c8a0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e;
    
    // Correct typehash for BridgeMessage
    // keccak256("BridgeMessage(address recipient,uint256 amount,uint256 sourceChainId,bytes32 sourceTxId,uint256 nonce)")
    bytes32 public constant BRIDGE_MESSAGE_TYPEHASH_V2 = keccak256(
        "BridgeMessage(address recipient,uint256 amount,uint256 sourceChainId,bytes32 sourceTxId,uint256 nonce)"
    );
    
    // ============ Structs ============
    
    /**
     * @dev Bridge message structure for cross-chain transfers
     * @param recipient Address to receive tokens on destination chain
     * @param amount Amount of tokens to bridge
     * @param sourceChainId Chain ID where tokens were locked/burned
     * @param sourceTxId Transaction ID on source chain
     * @param nonce Unique nonce for replay protection
     */
    struct BridgeMessage {
        address recipient;
        uint256 amount;
        uint256 sourceChainId;
        bytes32 sourceTxId;
        uint256 nonce;
    }
    
    /**
     * @dev Signature info with validator address
     * @param validator Validator address that signed
     * @param r ECDSA signature r component
     * @param s ECDSA signature s component
     * @param v ECDSA recovery id
     */
    struct ValidatorSignature {
        address validator;
        bytes32 r;
        bytes32 s;
        uint8 v;
    }
    
    // ============ State Variables ============
    
    /// @notice Minimum number of validator signatures required
    uint256 public signatureThreshold;
    
    /// @notice Total number of registered validators
    uint256 public totalValidators;
    
    /// @notice Mapping of validator addresses to their status
    mapping(address => bool) public isValidator;
    
    /// @notice Mapping of processed bridge transactions for replay protection
    mapping(bytes32 => bool) public processedBridgeTx;
    
    /// @notice Mapping of used nonces per source chain
    mapping(uint256 => mapping(uint256 => bool)) public usedNonces;
    
    /// @notice EIP-712 domain separator
    bytes32 private _domainSeparator;
    
    /// @notice Chain ID at deployment (for domain separator)
    uint256 private _chainId;
    
    // ============ Events ============
    
    /**
     * @dev Emitted when a validator is added
     */
    event ValidatorAdded(address indexed validator, uint256 totalValidators);
    
    /**
     * @dev Emitted when a validator is removed
     */
    event ValidatorRemoved(address indexed validator, uint256 totalValidators);
    
    /**
     * @dev Emitted when signature threshold is updated
     */
    event SignatureThresholdUpdated(uint256 oldThreshold, uint256 newThreshold);
    
    /**
     * @dev Emitted when bridge transaction is verified successfully
     */
    event BridgeTransactionVerified(
        bytes32 indexed messageHash,
        address indexed recipient,
        uint256 amount,
        uint256 sourceChainId,
        bytes32 sourceTxId,
        uint256 nonce,
        uint256 signatureCount
    );
    
    /**
     * @dev Emitted when signature verification fails
     */
    event SignatureVerificationFailed(
        bytes32 indexed messageHash,
        address indexed validator,
        string reason
    );
    
    /**
     * @dev Emitted when duplicate signature is detected
     */
    event DuplicateSignatureDetected(
        bytes32 indexed messageHash,
        address indexed validator
    );
    
    /**
     * @dev Emitted when bridge transaction is marked as processed
     */
    event BridgeTransactionProcessed(
        bytes32 indexed messageHash,
        bytes32 indexed sourceTxId
    );
    
    // ============ Errors ============
    
    error InvalidSignatureThreshold();
    error InvalidValidatorAddress();
    error ValidatorAlreadyExists();
    error ValidatorNotFound();
    error InsufficientSignatures(uint256 required, uint256 provided);
    error InvalidSignature(address validator);
    error DuplicateSignature(address validator);
    error BridgeTransactionAlreadyProcessed();
    error NonceAlreadyUsed();
    error InvalidChainId();
    error InvalidMessageHash();
    
    // ============ Modifiers ============
    
    modifier onlyAdmin() {
        if (!hasRole(DEFAULT_ADMIN_ROLE, msg.sender)) {
            revert("BridgeSecurity: caller is not admin");
        }
        _;
    }
    
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }
    
    // ============ Initialization ============
    
    /**
     * @dev Initialize the bridge security contract
     * @param admin Admin address with full control
     * @param _initialThreshold Initial signature threshold (M of N)
     * @param _initialValidators Initial set of validator addresses
     */
    function initialize(
        address admin,
        uint256 _initialThreshold,
        address[] calldata _initialValidators
    ) public initializer {
        __AccessControl_init();
        __UUPSUpgradeable_init();
        __ReentrancyGuard_init();
        
        if (admin == address(0)) {
            revert InvalidValidatorAddress();
        }
        
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(BRIDGE_OPERATOR_ROLE, admin);
        
        // Set initial threshold
        if (_initialThreshold == 0 || _initialThreshold > _initialValidators.length) {
            revert InvalidSignatureThreshold();
        }
        signatureThreshold = _initialThreshold;
        
        // Add initial validators
        for (uint256 i = 0; i < _initialValidators.length; i++) {
            _addValidator(_initialValidators[i]);
        }
        
        // Store chain ID for domain separator
        _chainId = block.chainid;
        
        // Compute domain separator
        _domainSeparator = _computeDomainSeparator();
    }
    
    // ============ External Functions ============
    
    /**
     * @dev Verify multi-signature for bridge message
     * @param message Bridge message to verify
     * @param signatures Array of validator signatures
     * @return messageHash The hash of the message that was verified
     * @return isValid Whether the verification was successful
     */
    function verifyBridgeSignatures(
        BridgeMessage calldata message,
        ValidatorSignature[] calldata signatures
    ) external nonReentrant returns (bytes32 messageHash, bool isValid) {
        // Check if already processed
        messageHash = hashBridgeMessage(message);
        if (processedBridgeTx[messageHash]) {
            revert BridgeTransactionAlreadyProcessed();
        }
        
        // Check nonce hasn't been used
        if (usedNonces[message.sourceChainId][message.nonce]) {
            revert NonceAlreadyUsed();
        }
        
        // Verify signature count meets threshold
        if (signatures.length < signatureThreshold) {
            revert InsufficientSignatures(signatureThreshold, signatures.length);
        }
        
        // Track verified validators to prevent duplicates
        mapping(address => bool) storage verifiedValidators;
        uint256 validSignatureCount = 0;
        
        // Verify each signature
        for (uint256 i = 0; i < signatures.length; i++) {
            address validator = signatures[i].validator;
            
            // Check validator is registered
            if (!isValidator[validator]) {
                emit SignatureVerificationFailed(messageHash, validator, "Not a registered validator");
                continue;
            }
            
            // Check for duplicate signature
            if (verifiedValidators[validator]) {
                emit DuplicateSignatureDetected(messageHash, validator);
                continue;
            }
            
            // Verify the signature cryptographically
            bool sigValid = _verifySignature(messageHash, signatures[i]);
            if (!sigValid) {
                emit SignatureVerificationFailed(messageHash, validator, "Invalid cryptographic signature");
                continue;
            }
            
            // Mark as verified
            verifiedValidators[validator] = true;
            validSignatureCount++;
        }
        
        // Check if we have enough valid signatures
        if (validSignatureCount < signatureThreshold) {
            emit SignatureVerificationFailed(messageHash, address(0), "Insufficient valid signatures");
            return (messageHash, false);
        }
        
        // Mark as processed to prevent replay
        processedBridgeTx[messageHash] = true;
        usedNonces[message.sourceChainId][message.nonce] = true;
        
        emit BridgeTransactionVerified(
            messageHash,
            message.recipient,
            message.amount,
            message.sourceChainId,
            message.sourceTxId,
            message.nonce,
            validSignatureCount
        );
        
        return (messageHash, true);
    }
    
    /**
     * @dev Verify signatures without state changes (view function for checking)
     * @param message Bridge message to verify
     * @param signatures Array of validator signatures
     * @return isValid Whether the signatures are valid
     * @return validCount Number of valid signatures
     */
    function checkBridgeSignatures(
        BridgeMessage calldata message,
        ValidatorSignature[] calldata signatures
    ) external view returns (bool isValid, uint256 validCount) {
        bytes32 messageHash = hashBridgeMessage(message);
        
        // Check if already processed
        if (processedBridgeTx[messageHash]) {
            return (false, 0);
        }
        
        // Check nonce hasn't been used
        if (usedNonces[message.sourceChainId][message.nonce]) {
            return (false, 0);
        }
        
        // Count unique valid signatures
        address[] memory seenValidators = new address[](signatures.length);
        uint256 seenCount = 0;
        
        for (uint256 i = 0; i < signatures.length; i++) {
            address validator = signatures[i].validator;
            
            // Check validator is registered
            if (!isValidator[validator]) {
                continue;
            }
            
            // Check for duplicate
            bool isDuplicate = false;
            for (uint256 j = 0; j < seenCount; j++) {
                if (seenValidators[j] == validator) {
                    isDuplicate = true;
                    break;
                }
            }
            if (isDuplicate) {
                continue;
            }
            
            // Verify signature
            if (_verifySignature(messageHash, signatures[i])) {
                seenValidators[seenCount] = validator;
                seenCount++;
            }
        }
        
        return (seenCount >= signatureThreshold, seenCount);
    }
    
    /**
     * @dev Add a new validator (admin only)
     * @param validator Address to add as validator
     */
    function addValidator(address validator) external onlyAdmin {
        _addValidator(validator);
    }
    
    /**
     * @dev Remove a validator (admin only)
     * @param validator Address to remove from validators
     */
    function removeValidator(address validator) external onlyAdmin {
        if (!isValidator[validator]) {
            revert ValidatorNotFound();
        }
        
        isValidator[validator] = false;
        totalValidators--;
        
        // Adjust threshold if needed
        if (signatureThreshold > totalValidators && totalValidators > 0) {
            signatureThreshold = totalValidators;
            emit SignatureThresholdUpdated(signatureThreshold + 1, signatureThreshold);
        }
        
        emit ValidatorRemoved(validator, totalValidators);
    }
    
    /**
     * @dev Update signature threshold (admin only)
     * @param newThreshold New minimum signature threshold
     */
    function setSignatureThreshold(uint256 newThreshold) external onlyAdmin {
        if (newThreshold == 0 || newThreshold > totalValidators) {
            revert InvalidSignatureThreshold();
        }
        
        uint256 oldThreshold = signatureThreshold;
        signatureThreshold = newThreshold;
        
        emit SignatureThresholdUpdated(oldThreshold, newThreshold);
    }
    
    /**
     * @dev Check if a bridge transaction has been processed
     * @param messageHash Hash of the bridge message
     * @return Whether the transaction has been processed
     */
    function isProcessed(bytes32 messageHash) external view returns (bool) {
        return processedBridgeTx[messageHash];
    }
    
    /**
     * @dev Check if a nonce has been used for a source chain
     * @param sourceChainId Source chain ID
     * @param nonce Nonce to check
     * @return Whether the nonce has been used
     */
    function isNonceUsed(uint256 sourceChainId, uint256 nonce) external view returns (bool) {
        return usedNonces[sourceChainId][nonce];
    }
    
    // ============ Public Functions ============
    
    /**
     * @dev Compute the EIP-712 hash of a bridge message
     * @param message Bridge message to hash
     * @return The EIP-712 typed hash
     */
    function hashBridgeMessage(BridgeMessage calldata message) public view returns (bytes32) {
        // Ensure domain separator is current (handles chain ID changes)
        bytes32 domainSeparator = block.chainid == _chainId 
            ? _domainSeparator 
            : _computeDomainSeparator();
        
        // Compute struct hash
        bytes32 structHash = keccak256(abi.encode(
            BRIDGE_MESSAGE_TYPEHASH_V2,
            message.recipient,
            message.amount,
            message.sourceChainId,
            message.sourceTxId,
            message.nonce
        ));
        
        // Compute EIP-712 typed hash
        return keccak256(abi.encodePacked("\x19\x01", domainSeparator, structHash));
    }
    
    /**
     * @dev Get the current domain separator
     * @return The domain separator hash
     */
    function domainSeparator() public view returns (bytes32) {
        return block.chainid == _chainId 
            ? _domainSeparator 
            : _computeDomainSeparator();
    }
    
    /**
     * @dev Get bridge security configuration
     * @return _threshold Current signature threshold
     * @return _validators Total number of validators
     */
    function getConfiguration() external view returns (uint256 _threshold, uint256 _validators) {
        return (signatureThreshold, totalValidators);
    }
    
    // ============ Internal Functions ============
    
    /**
     * @dev Add a validator (internal)
     * @param validator Address to add
     */
    function _addValidator(address validator) internal {
        if (validator == address(0)) {
            revert InvalidValidatorAddress();
        }
        if (isValidator[validator]) {
            revert ValidatorAlreadyExists();
        }
        
        isValidator[validator] = true;
        totalValidators++;
        
        emit ValidatorAdded(validator, totalValidators);
    }
    
    /**
     * @dev Verify a single signature
     * @param messageHash Hash of the message
     * @param sig Signature to verify
     * @return Whether the signature is valid
     */
    function _verifySignature(
        bytes32 messageHash,
        ValidatorSignature calldata sig
    ) internal pure returns (bool) {
        // Reconstruct the signature bytes
        bytes memory signatureBytes = abi.encodePacked(sig.r, sig.s, sig.v);
        
        // Recover signer address
        address signer = messageHash.recover(signatureBytes);
        
        // Check if signer matches the claimed validator
        return signer == sig.validator;
    }
    
    /**
     * @dev Compute the EIP-712 domain separator
     * @return The domain separator hash
     */
    function _computeDomainSeparator() internal view returns (bytes32) {
        return keccak256(abi.encode(
            DOMAIN_TYPEHASH,
            keccak256(bytes("PRSM Bridge Security")),
            keccak256(bytes("1")),
            block.chainid,
            address(this)
        ));
    }
    
    /**
     * @dev Required override for UUPS upgradeability
     */
    function _authorizeUpgrade(address newImplementation)
        internal
        override
        onlyAdmin
    {}
    
    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(AccessControlUpgradeable)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}
