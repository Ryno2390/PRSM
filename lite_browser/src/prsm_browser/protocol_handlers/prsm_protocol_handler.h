// PRSM Protocol Handler
// Handles prsm:// URLs for native P2P research collaboration

#ifndef PRSM_BROWSER_PROTOCOL_HANDLERS_PRSM_PROTOCOL_HANDLER_H_
#define PRSM_BROWSER_PROTOCOL_HANDLERS_PRSM_PROTOCOL_HANDLER_H_

#include <string>
#include <memory>
#include <vector>

namespace prsm_browser {

// Forward declarations
struct PeerInfo;
struct NetworkConnection;

// PRSM URL structure: prsm://institution.edu/research-area/specific-content
class PRSMUrl {
public:
    explicit PRSMUrl(const std::string& url);
    
    // Parse PRSM URL components
    bool IsValid() const { return is_valid_; }
    std::string GetInstitution() const { return institution_; }
    std::string GetResearchArea() const { return research_area_; }
    std::string GetSpecificContent() const { return specific_content_; }
    
    // Generate human-readable description
    std::string GetDescription() const;

private:
    bool is_valid_;
    std::string institution_;
    std::string research_area_;
    std::string specific_content_;
    
    void ParseUrl(const std::string& url);
};

// Peer information for P2P connections
struct PeerInfo {
    std::string peer_id;
    std::string institution;
    std::string ip_address;
    uint16_t port;
    std::vector<std::string> research_areas;
    bool is_trusted;
    double reputation_score;
};

// Network connection details
struct NetworkConnection {
    std::string connection_id;
    PeerInfo peer;
    bool is_encrypted;
    std::string encryption_protocol;
    bool is_connected;
};

// Main PRSM protocol handler class
class PRSMProtocolHandler {
public:
    PRSMProtocolHandler();
    ~PRSMProtocolHandler();
    
    // Handle PRSM URL requests
    bool CanHandle(const std::string& url) const;
    std::unique_ptr<NetworkConnection> HandleRequest(const PRSMUrl& prsm_url);
    
    // Peer discovery and connection management
    std::vector<PeerInfo> DiscoverPeers(const std::string& institution);
    std::vector<PeerInfo> FindResearchPeers(const std::string& research_area);
    std::unique_ptr<NetworkConnection> ConnectToPeer(const PeerInfo& peer);
    
    // Research content resolution
    std::string ResolveResearchContent(const PRSMUrl& prsm_url, 
                                     const NetworkConnection& connection);
    
    // Security and authentication
    bool AuthenticateWithInstitution(const std::string& institution);
    bool VerifyPeerCredentials(const PeerInfo& peer);

private:
    // Internal state
    std::vector<PeerInfo> known_peers_;
    std::vector<std::unique_ptr<NetworkConnection>> active_connections_;
    
    // Configuration
    bool debug_mode_;
    std::string user_institution_;
    std::string user_credentials_;
    
    // Helper methods
    bool IsInstitutionTrusted(const std::string& institution) const;
    PeerInfo CreateMockPeer(const std::string& institution) const;
    std::string GenerateConnectionId() const;
};

} // namespace prsm_browser

#endif // PRSM_BROWSER_PROTOCOL_HANDLERS_PRSM_PROTOCOL_HANDLER_H_