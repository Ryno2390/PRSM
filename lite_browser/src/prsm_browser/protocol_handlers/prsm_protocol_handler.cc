// PRSM Protocol Handler Implementation
#include "prsm_browser/protocol_handlers/prsm_protocol_handler.h"

#include <iostream>
#include <sstream>
#include <random>
#include <algorithm>
#include <regex>

namespace prsm_browser {

// PRSMUrl Implementation
PRSMUrl::PRSMUrl(const std::string& url) : is_valid_(false) {
    ParseUrl(url);
}

void PRSMUrl::ParseUrl(const std::string& url) {
    // Parse URL format: prsm://institution.edu/research-area/specific-content
    std::regex prsm_regex(R"(^prsm://([^/]+)(?:/([^/]+))?(?:/(.+))?$)");
    std::smatch matches;
    
    if (std::regex_match(url, matches, prsm_regex)) {
        is_valid_ = true;
        institution_ = matches[1].str();
        research_area_ = matches.size() > 2 ? matches[2].str() : "";
        specific_content_ = matches.size() > 3 ? matches[3].str() : "";
        
        std::cout << "✅ Parsed PRSM URL:" << std::endl;
        std::cout << "   Institution: " << institution_ << std::endl;
        std::cout << "   Research Area: " << research_area_ << std::endl;
        std::cout << "   Content: " << specific_content_ << std::endl;
    } else {
        std::cout << "❌ Invalid PRSM URL format: " << url << std::endl;
    }
}

std::string PRSMUrl::GetDescription() const {
    if (!is_valid_) return "Invalid PRSM URL";
    
    std::ostringstream desc;
    desc << "Research at " << institution_;
    if (!research_area_.empty()) {
        desc << " in " << research_area_;
    }
    if (!specific_content_.empty()) {
        desc << " - " << specific_content_;
    }
    return desc.str();
}

// PRSMProtocolHandler Implementation
PRSMProtocolHandler::PRSMProtocolHandler() 
    : debug_mode_(true), user_institution_("unc.edu"), user_credentials_("test_user") {
    
    std::cout << "🌐 Initializing PRSM Protocol Handler..." << std::endl;
    std::cout << "   User Institution: " << user_institution_ << std::endl;
    std::cout << "   Debug Mode: " << (debug_mode_ ? "ON" : "OFF") << std::endl;
    
    // Initialize with some known research institutions
    known_peers_.push_back(CreateMockPeer("mit.edu"));
    known_peers_.push_back(CreateMockPeer("stanford.edu"));
    known_peers_.push_back(CreateMockPeer("duke.edu"));
    known_peers_.push_back(CreateMockPeer("ncsu.edu"));
    
    std::cout << "   Loaded " << known_peers_.size() << " known research institutions" << std::endl;
}

PRSMProtocolHandler::~PRSMProtocolHandler() {
    std::cout << "🔌 Shutting down PRSM Protocol Handler..." << std::endl;
    
    // Close all active connections
    for (auto& connection : active_connections_) {
        if (connection && connection->is_connected) {
            std::cout << "   Closing connection: " << connection->connection_id << std::endl;
        }
    }
    active_connections_.clear();
}

bool PRSMProtocolHandler::CanHandle(const std::string& url) const {
    return url.substr(0, 7) == "prsm://";
}

std::unique_ptr<NetworkConnection> PRSMProtocolHandler::HandleRequest(const PRSMUrl& prsm_url) {
    if (!prsm_url.IsValid()) {
        std::cout << "❌ Cannot handle invalid PRSM URL" << std::endl;
        return nullptr;
    }
    
    std::cout << "🔗 Handling PRSM request: " << prsm_url.GetDescription() << std::endl;
    
    // Step 1: Discover peers at the institution
    auto peers = DiscoverPeers(prsm_url.GetInstitution());
    if (peers.empty()) {
        std::cout << "❌ No peers found at " << prsm_url.GetInstitution() << std::endl;
        return nullptr;
    }
    
    // Step 2: Find peers with relevant research area
    auto research_peers = FindResearchPeers(prsm_url.GetResearchArea());
    
    // Step 3: Connect to the best peer
    PeerInfo best_peer = peers[0]; // Simple selection for now
    auto connection = ConnectToPeer(best_peer);
    
    if (connection && connection->is_connected) {
        std::cout << "✅ Successfully connected to " << best_peer.institution << std::endl;
        
        // Step 4: Resolve the specific research content
        std::string content = ResolveResearchContent(prsm_url, *connection);
        std::cout << "📄 Retrieved content: " << content.substr(0, 100) << "..." << std::endl;
    }
    
    return connection;
}

std::vector<PeerInfo> PRSMProtocolHandler::DiscoverPeers(const std::string& institution) {
    std::cout << "🔍 Discovering peers at " << institution << "..." << std::endl;
    
    std::vector<PeerInfo> institution_peers;
    
    // Find peers from the specified institution
    for (const auto& peer : known_peers_) {
        if (peer.institution == institution) {
            institution_peers.push_back(peer);
        }
    }
    
    // If no exact match, create a mock peer for demonstration
    if (institution_peers.empty() && debug_mode_) {
        std::cout << "   Creating mock peer for " << institution << std::endl;
        institution_peers.push_back(CreateMockPeer(institution));
    }
    
    std::cout << "   Found " << institution_peers.size() << " peers at " << institution << std::endl;
    return institution_peers;
}

std::vector<PeerInfo> PRSMProtocolHandler::FindResearchPeers(const std::string& research_area) {
    if (research_area.empty()) {
        return known_peers_; // Return all peers if no specific area
    }
    
    std::cout << "🔬 Finding peers with research area: " << research_area << std::endl;
    
    std::vector<PeerInfo> research_peers;
    
    for (const auto& peer : known_peers_) {
        // Simple substring matching for research areas
        for (const auto& area : peer.research_areas) {
            if (area.find(research_area) != std::string::npos) {
                research_peers.push_back(peer);
                break;
            }
        }
    }
    
    std::cout << "   Found " << research_peers.size() << " peers with " << research_area << " research" << std::endl;
    return research_peers;
}

std::unique_ptr<NetworkConnection> PRSMProtocolHandler::ConnectToPeer(const PeerInfo& peer) {
    std::cout << "🤝 Connecting to peer: " << peer.institution << " (" << peer.peer_id << ")" << std::endl;
    
    // Create connection
    auto connection = std::make_unique<NetworkConnection>();
    connection->connection_id = GenerateConnectionId();
    connection->peer = peer;
    connection->is_encrypted = true;
    connection->encryption_protocol = "post-quantum-TLS";
    connection->is_connected = true; // Mock successful connection
    
    std::cout << "   Connection ID: " << connection->connection_id << std::endl;
    std::cout << "   Encryption: " << connection->encryption_protocol << std::endl;
    std::cout << "   Status: " << (connection->is_connected ? "Connected" : "Failed") << std::endl;
    
    // Store the connection
    active_connections_.push_back(std::make_unique<NetworkConnection>(*connection));
    
    return connection;
}

std::string PRSMProtocolHandler::ResolveResearchContent(const PRSMUrl& prsm_url, 
                                                       const NetworkConnection& connection) {
    std::cout << "📄 Resolving research content..." << std::endl;
    std::cout << "   Institution: " << prsm_url.GetInstitution() << std::endl;
    std::cout << "   Research Area: " << prsm_url.GetResearchArea() << std::endl;
    std::cout << "   Content: " << prsm_url.GetSpecificContent() << std::endl;
    
    // Mock content resolution
    std::ostringstream content;
    content << "PRSM Research Content from " << connection.peer.institution << "\\n\\n";
    content << "Research Area: " << prsm_url.GetResearchArea() << "\\n";
    content << "Content: " << prsm_url.GetSpecificContent() << "\\n\\n";
    content << "This is mock research content retrieved through the PRSM P2P protocol.\\n";
    content << "In a real implementation, this would be:n";
    content << "- Cryptographically verified content\\n";
    content << "- Retrieved from distributed shards\\n";
    content << "- Post-quantum encrypted\\n";
    content << "- Peer-reviewed and authenticated\\n\\n";
    content << "Connection Details:\\n";
    content << "- Peer ID: " << connection.peer.peer_id << "\\n";
    content << "- Institution: " << connection.peer.institution << "\\n";
    content << "- Encryption: " << connection.encryption_protocol << "\\n";
    content << "- Reputation: " << connection.peer.reputation_score << "\\n";
    
    return content.str();
}

bool PRSMProtocolHandler::AuthenticateWithInstitution(const std::string& institution) {
    std::cout << "🔐 Authenticating with " << institution << "..." << std::endl;
    
    // Mock authentication logic
    bool is_trusted = IsInstitutionTrusted(institution);
    
    std::cout << "   Authentication: " << (is_trusted ? "SUCCESS" : "FAILED") << std::endl;
    return is_trusted;
}

bool PRSMProtocolHandler::VerifyPeerCredentials(const PeerInfo& peer) {
    std::cout << "✅ Verifying credentials for " << peer.peer_id << std::endl;
    
    // Mock credential verification
    bool is_valid = peer.is_trusted && peer.reputation_score > 0.5;
    
    std::cout << "   Trusted: " << (peer.is_trusted ? "YES" : "NO") << std::endl;
    std::cout << "   Reputation: " << peer.reputation_score << std::endl;
    std::cout << "   Verification: " << (is_valid ? "PASSED" : "FAILED") << std::endl;
    
    return is_valid;
}

// Private helper methods
bool PRSMProtocolHandler::IsInstitutionTrusted(const std::string& institution) const {
    // List of trusted research institutions
    std::vector<std::string> trusted_institutions = {
        "mit.edu", "stanford.edu", "harvard.edu", "unc.edu", "duke.edu",
        "ncsu.edu", "berkeley.edu", "caltech.edu", "cmu.edu", "cornell.edu"
    };
    
    return std::find(trusted_institutions.begin(), trusted_institutions.end(), 
                    institution) != trusted_institutions.end();
}

PeerInfo PRSMProtocolHandler::CreateMockPeer(const std::string& institution) const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> reputation_dist(0.7, 0.99);
    static std::uniform_int_distribution<> port_dist(8000, 9000);
    
    PeerInfo peer;
    peer.peer_id = "peer_" + institution + "_" + std::to_string(gen() % 1000);
    peer.institution = institution;
    peer.ip_address = "192.168.1." + std::to_string(gen() % 255);
    peer.port = port_dist(gen);
    peer.is_trusted = IsInstitutionTrusted(institution);
    peer.reputation_score = reputation_dist(gen);
    
    // Add relevant research areas based on institution
    if (institution.find("mit") != std::string::npos) {
        peer.research_areas = {"artificial-intelligence", "quantum-computing", "robotics"};
    } else if (institution.find("stanford") != std::string::npos) {
        peer.research_areas = {"machine-learning", "computer-vision", "nlp"};
    } else if (institution.find("unc") != std::string::npos) {
        peer.research_areas = {"quantum-algorithms", "cryptography", "bioinformatics"};
    } else {
        peer.research_areas = {"general-research", "collaboration", "data-science"};
    }
    
    return peer;
}

std::string PRSMProtocolHandler::GenerateConnectionId() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dist(10000, 99999);
    
    return "conn_" + std::to_string(dist(gen));
}

} // namespace prsm_browser