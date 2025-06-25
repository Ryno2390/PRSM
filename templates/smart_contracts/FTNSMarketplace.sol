// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title FTNS Marketplace
 * @dev Decentralized marketplace for AI services and compute resources in PRSM
 * 
 * Features:
 * - Service listings with FTNS token payments
 * - Compute resource trading
 * - Context-based service discovery
 * - Reputation and rating system
 * - Escrow and dispute resolution
 * - Revenue sharing mechanisms
 */
contract FTNSMarketplace is Ownable, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;
    
    // Events
    event ServiceListed(
        uint256 indexed serviceId,
        address indexed provider,
        string context,
        uint256 price,
        string serviceType
    );
    
    event ServicePurchased(
        uint256 indexed serviceId,
        address indexed buyer,
        address indexed provider,
        uint256 amount,
        string context
    );
    
    event ComputeResourceListed(
        uint256 indexed resourceId,
        address indexed provider,
        uint256 computeUnits,
        uint256 pricePerUnit
    );
    
    event ComputeResourcePurchased(
        uint256 indexed resourceId,
        address indexed buyer,
        uint256 computeUnits,
        uint256 totalPrice
    );
    
    event ReviewSubmitted(
        uint256 indexed serviceId,
        address indexed reviewer,
        uint8 rating,
        string comment
    );
    
    event DisputeCreated(
        uint256 indexed disputeId,
        uint256 indexed serviceId,
        address indexed buyer,
        address provider
    );
    
    event DisputeResolved(
        uint256 indexed disputeId,
        address winner,
        uint256 amount
    );
    
    // Enums
    enum ServiceStatus { Active, Paused, Completed, Disputed }
    enum DisputeStatus { Pending, Resolved, Cancelled }
    enum ServiceType { AI_MODEL, COMPUTE, DATA_PROCESSING, CUSTOM }
    
    // Structs
    struct Service {
        uint256 id;
        address provider;
        string context;
        string name;
        string description;
        ServiceType serviceType;
        uint256 price;
        uint256 minStake;
        ServiceStatus status;
        uint256 totalSales;
        uint256 rating; // Out of 100
        uint256 reviewCount;
        uint256 createdAt;
        bool exists;
    }
    
    struct ComputeResource {
        uint256 id;
        address provider;
        uint256 totalComputeUnits;
        uint256 availableComputeUnits;
        uint256 pricePerUnit;
        string specifications;
        ServiceStatus status;
        uint256 createdAt;
        bool exists;
    }
    
    struct Purchase {
        uint256 id;
        uint256 serviceId;
        address buyer;
        address provider;
        uint256 amount;
        uint256 timestamp;
        ServiceStatus status;
        bool escrowReleased;
    }
    
    struct Review {
        address reviewer;
        uint256 serviceId;
        uint8 rating; // 1-5 stars
        string comment;
        uint256 timestamp;
    }
    
    struct Dispute {
        uint256 id;
        uint256 serviceId;
        uint256 purchaseId;
        address buyer;
        address provider;
        string reason;
        DisputeStatus status;
        address winner;
        uint256 amount;
        uint256 createdAt;
        uint256 resolvedAt;
    }
    
    // State variables
    IERC20 public immutable ftnsToken;
    
    mapping(uint256 => Service) public services;
    mapping(uint256 => ComputeResource) public computeResources;
    mapping(uint256 => Purchase) public purchases;
    mapping(uint256 => Review[]) public serviceReviews;
    mapping(uint256 => Dispute) public disputes;
    mapping(address => uint256[]) public providerServices;
    mapping(address => uint256[]) public buyerPurchases;
    mapping(string => uint256[]) public contextServices;
    
    uint256 public nextServiceId = 1;
    uint256 public nextResourceId = 1;
    uint256 public nextPurchaseId = 1;
    uint256 public nextDisputeId = 1;
    
    uint256 public marketplaceFee = 250; // 2.5% in basis points
    uint256 public escrowPeriod = 7 days;
    uint256 public disputePeriod = 14 days;
    
    address public feeRecipient;
    address[] public arbitrators;
    mapping(address => bool) public isArbitrator;
    
    constructor(
        address _ftnsToken,
        address _feeRecipient,
        address initialOwner
    ) Ownable(initialOwner) {
        ftnsToken = IERC20(_ftnsToken);
        feeRecipient = _feeRecipient;
    }
    
    /**
     * @dev List a service on the marketplace
     */
    function listService(
        string memory context,
        string memory name,
        string memory description,
        ServiceType serviceType,
        uint256 price,
        uint256 minStake
    ) public whenNotPaused returns (uint256) {
        require(price > 0, "Price must be greater than 0");
        
        uint256 serviceId = nextServiceId++;
        
        services[serviceId] = Service({
            id: serviceId,
            provider: msg.sender,
            context: context,
            name: name,
            description: description,
            serviceType: serviceType,
            price: price,
            minStake: minStake,
            status: ServiceStatus.Active,
            totalSales: 0,
            rating: 0,
            reviewCount: 0,
            createdAt: block.timestamp,
            exists: true
        });
        
        providerServices[msg.sender].push(serviceId);
        contextServices[context].push(serviceId);
        
        emit ServiceListed(serviceId, msg.sender, context, price, _serviceTypeToString(serviceType));
        
        return serviceId;
    }
    
    /**
     * @dev List compute resources
     */
    function listComputeResource(
        uint256 computeUnits,
        uint256 pricePerUnit,
        string memory specifications
    ) public whenNotPaused returns (uint256) {
        require(computeUnits > 0, "Compute units must be greater than 0");
        require(pricePerUnit > 0, "Price per unit must be greater than 0");
        
        uint256 resourceId = nextResourceId++;
        
        computeResources[resourceId] = ComputeResource({
            id: resourceId,
            provider: msg.sender,
            totalComputeUnits: computeUnits,
            availableComputeUnits: computeUnits,
            pricePerUnit: pricePerUnit,
            specifications: specifications,
            status: ServiceStatus.Active,
            createdAt: block.timestamp,
            exists: true
        });
        
        emit ComputeResourceListed(resourceId, msg.sender, computeUnits, pricePerUnit);
        
        return resourceId;
    }
    
    /**
     * @dev Purchase a service
     */
    function purchaseService(uint256 serviceId) 
        public 
        nonReentrant 
        whenNotPaused 
        returns (uint256) 
    {
        Service storage service = services[serviceId];
        require(service.exists, "Service does not exist");
        require(service.status == ServiceStatus.Active, "Service not available");
        require(service.provider != msg.sender, "Cannot purchase own service");
        
        uint256 totalAmount = service.price;
        uint256 fee = (totalAmount * marketplaceFee) / 10000;
        uint256 providerAmount = totalAmount - fee;
        
        // Transfer tokens to escrow (this contract)
        ftnsToken.safeTransferFrom(msg.sender, address(this), totalAmount);
        
        uint256 purchaseId = nextPurchaseId++;
        
        purchases[purchaseId] = Purchase({
            id: purchaseId,
            serviceId: serviceId,
            buyer: msg.sender,
            provider: service.provider,
            amount: totalAmount,
            timestamp: block.timestamp,
            status: ServiceStatus.Active,
            escrowReleased: false
        });
        
        buyerPurchases[msg.sender].push(purchaseId);
        service.totalSales += totalAmount;
        
        emit ServicePurchased(serviceId, msg.sender, service.provider, totalAmount, service.context);
        
        return purchaseId;
    }
    
    /**
     * @dev Purchase compute resources
     */
    function purchaseComputeResource(uint256 resourceId, uint256 computeUnits) 
        public 
        nonReentrant 
        whenNotPaused 
    {
        ComputeResource storage resource = computeResources[resourceId];
        require(resource.exists, "Resource does not exist");
        require(resource.status == ServiceStatus.Active, "Resource not available");
        require(resource.availableComputeUnits >= computeUnits, "Insufficient compute units");
        require(resource.provider != msg.sender, "Cannot purchase own resource");
        
        uint256 totalPrice = computeUnits * resource.pricePerUnit;
        uint256 fee = (totalPrice * marketplaceFee) / 10000;
        uint256 providerAmount = totalPrice - fee;
        
        // Transfer tokens
        ftnsToken.safeTransferFrom(msg.sender, resource.provider, providerAmount);
        ftnsToken.safeTransferFrom(msg.sender, feeRecipient, fee);
        
        // Update available compute units
        resource.availableComputeUnits -= computeUnits;
        
        emit ComputeResourcePurchased(resourceId, msg.sender, computeUnits, totalPrice);
    }
    
    /**
     * @dev Release escrow after service completion
     */
    function releaseEscrow(uint256 purchaseId) public nonReentrant {
        Purchase storage purchase = purchases[purchaseId];
        require(purchase.buyer == msg.sender || purchase.provider == msg.sender, "Not authorized");
        require(!purchase.escrowReleased, "Escrow already released");
        require(
            block.timestamp >= purchase.timestamp + escrowPeriod || 
            purchase.buyer == msg.sender,
            "Escrow period not elapsed"
        );
        
        purchase.escrowReleased = true;
        purchase.status = ServiceStatus.Completed;
        
        uint256 fee = (purchase.amount * marketplaceFee) / 10000;
        uint256 providerAmount = purchase.amount - fee;
        
        ftnsToken.safeTransfer(purchase.provider, providerAmount);
        ftnsToken.safeTransfer(feeRecipient, fee);
    }
    
    /**
     * @dev Submit a review for a service
     */
    function submitReview(
        uint256 serviceId,
        uint8 rating,
        string memory comment
    ) public {
        require(rating >= 1 && rating <= 5, "Rating must be between 1 and 5");
        
        Service storage service = services[serviceId];
        require(service.exists, "Service does not exist");
        
        // Check if user has purchased this service
        bool hasPurchased = false;
        uint256[] memory userPurchases = buyerPurchases[msg.sender];
        for (uint i = 0; i < userPurchases.length; i++) {
            if (purchases[userPurchases[i]].serviceId == serviceId) {
                hasPurchased = true;
                break;
            }
        }
        require(hasPurchased, "Must purchase service to review");
        
        serviceReviews[serviceId].push(Review({
            reviewer: msg.sender,
            serviceId: serviceId,
            rating: rating,
            comment: comment,
            timestamp: block.timestamp
        }));
        
        // Update service rating
        uint256 totalRating = service.rating * service.reviewCount + (rating * 20); // Convert to 100-point scale
        service.reviewCount++;
        service.rating = totalRating / service.reviewCount;
        
        emit ReviewSubmitted(serviceId, msg.sender, rating, comment);
    }
    
    /**
     * @dev Create a dispute
     */
    function createDispute(uint256 purchaseId, string memory reason) 
        public 
        returns (uint256) 
    {
        Purchase storage purchase = purchases[purchaseId];
        require(purchase.buyer == msg.sender, "Only buyer can create dispute");
        require(!purchase.escrowReleased, "Escrow already released");
        require(
            block.timestamp <= purchase.timestamp + disputePeriod,
            "Dispute period expired"
        );
        
        uint256 disputeId = nextDisputeId++;
        
        disputes[disputeId] = Dispute({
            id: disputeId,
            serviceId: purchase.serviceId,
            purchaseId: purchaseId,
            buyer: purchase.buyer,
            provider: purchase.provider,
            reason: reason,
            status: DisputeStatus.Pending,
            winner: address(0),
            amount: purchase.amount,
            createdAt: block.timestamp,
            resolvedAt: 0
        });
        
        purchase.status = ServiceStatus.Disputed;
        
        emit DisputeCreated(disputeId, purchase.serviceId, purchase.buyer, purchase.provider);
        
        return disputeId;
    }
    
    /**
     * @dev Resolve a dispute (arbitrators only)
     */
    function resolveDispute(uint256 disputeId, address winner) public {
        require(isArbitrator[msg.sender], "Only arbitrators can resolve disputes");
        
        Dispute storage dispute = disputes[disputeId];
        require(dispute.status == DisputeStatus.Pending, "Dispute not pending");
        require(winner == dispute.buyer || winner == dispute.provider, "Invalid winner");
        
        dispute.status = DisputeStatus.Resolved;
        dispute.winner = winner;
        dispute.resolvedAt = block.timestamp;
        
        Purchase storage purchase = purchases[dispute.purchaseId];
        purchase.escrowReleased = true;
        
        if (winner == dispute.buyer) {
            // Refund to buyer
            ftnsToken.safeTransfer(dispute.buyer, dispute.amount);
        } else {
            // Pay provider
            uint256 fee = (dispute.amount * marketplaceFee) / 10000;
            uint256 providerAmount = dispute.amount - fee;
            
            ftnsToken.safeTransfer(dispute.provider, providerAmount);
            ftnsToken.safeTransfer(feeRecipient, fee);
        }
        
        emit DisputeResolved(disputeId, winner, dispute.amount);
    }
    
    // View functions
    function getService(uint256 serviceId) public view returns (Service memory) {
        return services[serviceId];
    }
    
    function getServicesByContext(string memory context) 
        public 
        view 
        returns (uint256[] memory) 
    {
        return contextServices[context];
    }
    
    function getServicesByProvider(address provider) 
        public 
        view 
        returns (uint256[] memory) 
    {
        return providerServices[provider];
    }
    
    function getServiceReviews(uint256 serviceId) 
        public 
        view 
        returns (Review[] memory) 
    {
        return serviceReviews[serviceId];
    }
    
    function getPurchasesByBuyer(address buyer) 
        public 
        view 
        returns (uint256[] memory) 
    {
        return buyerPurchases[buyer];
    }
    
    // Admin functions
    function setMarketplaceFee(uint256 newFee) public onlyOwner {
        require(newFee <= 1000, "Fee cannot exceed 10%");
        marketplaceFee = newFee;
    }
    
    function setFeeRecipient(address newRecipient) public onlyOwner {
        feeRecipient = newRecipient;
    }
    
    function addArbitrator(address arbitrator) public onlyOwner {
        if (!isArbitrator[arbitrator]) {
            arbitrators.push(arbitrator);
            isArbitrator[arbitrator] = true;
        }
    }
    
    function removeArbitrator(address arbitrator) public onlyOwner {
        isArbitrator[arbitrator] = false;
        // Remove from array
        for (uint i = 0; i < arbitrators.length; i++) {
            if (arbitrators[i] == arbitrator) {
                arbitrators[i] = arbitrators[arbitrators.length - 1];
                arbitrators.pop();
                break;
            }
        }
    }
    
    function pause() public onlyOwner {
        _pause();
    }
    
    function unpause() public onlyOwner {
        _unpause();
    }
    
    // Internal functions
    function _serviceTypeToString(ServiceType serviceType) 
        internal 
        pure 
        returns (string memory) 
    {
        if (serviceType == ServiceType.AI_MODEL) return "AI_MODEL";
        if (serviceType == ServiceType.COMPUTE) return "COMPUTE";
        if (serviceType == ServiceType.DATA_PROCESSING) return "DATA_PROCESSING";
        return "CUSTOM";
    }
}