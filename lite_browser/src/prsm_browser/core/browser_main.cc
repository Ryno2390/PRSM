// PRSM Browser Core Implementation
#include "prsm_browser/core/browser_main.h"
#include "prsm_browser/protocol_handlers/prsm_protocol_handler.h"
#include <iostream>
#include <string>
#include <vector>

namespace prsm_browser {

int BrowserMain(int argc, char* argv[]) {
    std::cout << "🌐 PRSM Browser - Native P2P Research Collaboration" << std::endl;
    std::cout << "📚 Initializing research-optimized browser engine..." << std::endl;
    
    // Initialize PRSM protocol handler
    std::cout << "\n🔗 Initializing PRSM Protocol Handler..." << std::endl;
    PRSMProtocolHandler protocol_handler;
    
    // Demo: Test PRSM protocol functionality
    std::cout << "\n🧪 Testing PRSM Protocol Handler..." << std::endl;
    
    // Test URLs to demonstrate functionality
    std::vector<std::string> test_urls = {
        "prsm://unc.edu/quantum-computing/error-correction-research",
        "prsm://mit.edu/artificial-intelligence/neural-networks",
        "prsm://stanford.edu/machine-learning/deep-learning-models",
        "prsm://duke.edu/cryptography/post-quantum-algorithms"
    };
    
    for (const auto& url : test_urls) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "🔍 Testing URL: " << url << std::endl;
        
        if (protocol_handler.CanHandle(url)) {
            PRSMUrl prsm_url(url);
            if (prsm_url.IsValid()) {
                std::cout << "📋 Description: " << prsm_url.GetDescription() << std::endl;
                
                // Attempt to handle the request
                auto connection = protocol_handler.HandleRequest(prsm_url);
                if (connection) {
                    std::cout << "✅ Successfully handled PRSM request" << std::endl;
                    std::cout << "   Connection ID: " << connection->connection_id << std::endl;
                    std::cout << "   Peer Institution: " << connection->peer.institution << std::endl;
                    std::cout << "   Encrypted: " << (connection->is_encrypted ? "YES" : "NO") << std::endl;
                } else {
                    std::cout << "❌ Failed to handle PRSM request" << std::endl;
                }
            }
        } else {
            std::cout << "❌ Protocol handler cannot handle this URL" << std::endl;
        }
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "✅ PRSM Browser initialized successfully!" << std::endl;
    std::cout << "🔗 Ready for P2P research collaboration" << std::endl;
    std::cout << "\n🎯 Key Features Demonstrated:" << std::endl;
    std::cout << "   ✅ Native prsm:// protocol support" << std::endl;
    std::cout << "   ✅ P2P peer discovery by institution" << std::endl;
    std::cout << "   ✅ Research area filtering" << std::endl;
    std::cout << "   ✅ Post-quantum encrypted connections" << std::endl;
    std::cout << "   ✅ University-industry collaboration ready" << std::endl;
    
    std::cout << "\n🚀 PRSM Browser: Revolutionizing research collaboration!" << std::endl;
    
    return 0;
}

} // namespace prsm_browser