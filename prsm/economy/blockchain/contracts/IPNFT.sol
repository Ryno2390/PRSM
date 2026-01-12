// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title PRSM IP-NFT (Intellectual Property NFT)
 * @dev Represents scientific IP on the PRSM network. 
 * Supports cross-chain bridging and royalty routing.
 */
contract IPNFT is ERC721Enumerable, ERC721URIStorage, AccessControl {
    using Counters for Counters.Counter;
    
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant BRIDGE_ROLE = keccak256("BRIDGE_ROLE");
    
    Counters.Counter private _tokenIdCounter;
    
    struct IPMetadata {
        string provenanceHash;
        string dataVersion;
        address originalCreator;
        uint256 royaltyBasisPoints; // e.g. 500 = 5%
    }
    
    mapping(uint256 => IPMetadata) public ipDetails;
    
    event IPNFTMinted(address indexed creator, uint256 indexed tokenId, string provenanceHash);
    
    constructor() ERC721("PRSM Scientific IP", "PRSM-IP") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
    }
    
    function mintIP(
        address to, 
        string memory uri, 
        string memory provenanceHash, 
        string memory dataVersion,
        uint256 royaltyBP
    ) public onlyRole(MINTER_ROLE) returns (uint256) {
        uint256 tokenId = _tokenIdCounter.current();
        _tokenIdCounter.increment();
        
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, uri);
        
        ipDetails[tokenId] = IPMetadata({
            provenanceHash: provenanceHash,
            dataVersion: dataVersion,
            originalCreator: to,
            royaltyBasisPoints: royaltyBP
        });
        
        emit IPNFTMinted(to, tokenId, provenanceHash);
        return tokenId;
    }
    
    // Bridge functions for Cross-Chain Interoperability
    function bridgeMint(address to, uint256 tokenId, string memory uri, string memory provenanceHash) external onlyRole(BRIDGE_ROLE) {
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, uri);
        ipDetails[tokenId].provenanceHash = provenanceHash;
    }

    function bridgeBurn(uint256 tokenId) external onlyRole(BRIDGE_ROLE) {
        _burn(tokenId);
    }

    // Required overrides
    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }

    function tokenURI(uint256 tokenId) public view override(ERC721, ERC721URIStorage) returns (string memory) {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(bytes4 interfaceId) public view override(ERC721Enumerable, ERC721URIStorage, AccessControl) returns (bool) {
        return super.supportsInterface(interfaceId);
    }
    
    function _beforeTokenTransfer(address from, address to, uint256 tokenId, uint256 batchSize) internal override(ERC721, ERC721Enumerable) {
        super._beforeTokenTransfer(from, to, tokenId, batchSize);
    }
}
