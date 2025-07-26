// Shard Protocol Handler
// Handles shard:// URLs for distributed file reconstruction

#ifndef LITE_BROWSER_PROTOCOL_HANDLERS_SHARD_PROTOCOL_HANDLER_H_
#define LITE_BROWSER_PROTOCOL_HANDLERS_SHARD_PROTOCOL_HANDLER_H_

#include <string>
#include <memory>
#include <vector>
#include <map>

namespace lite_browser {

// Shard URL structure: shard://distributed-file/filename.pdf
class ShardUrl {
public:
    explicit ShardUrl(const std::string& url);
    
    bool IsValid() const { return is_valid_; }
    std::string GetFileId() const { return file_id_; }
    std::string GetFilename() const { return filename_; }
    std::string GetDescription() const;

private:
    bool is_valid_;
    std::string file_id_;
    std::string filename_;
    void ParseUrl(const std::string& url);
};

// Shard information
struct ShardInfo {
    std::string shard_id;
    std::string peer_id;
    std::string institution;
    std::vector<uint8_t> shard_data;
    std::string integrity_hash;
    bool is_verified;
};

// File manifest describing shard distribution
struct ShardManifest {
    std::string file_id;
    std::string filename;
    size_t total_size;
    int total_shards;
    int required_shards;
    std::string file_hash;
    std::map<int, ShardInfo> shard_locations;
    std::string security_level;
};

// Main shard protocol handler
class ShardProtocolHandler {
public:
    ShardProtocolHandler();
    ~ShardProtocolHandler();
    
    // Handle shard URL requests
    bool CanHandle(const std::string& url) const;
    std::vector<uint8_t> HandleRequest(const ShardUrl& shard_url);
    
    // Shard discovery and reconstruction
    ShardManifest DiscoverShardManifest(const std::string& file_id);
    std::vector<ShardInfo> CollectRequiredShards(const ShardManifest& manifest);
    std::vector<uint8_t> ReconstructFile(const ShardManifest& manifest, 
                                        const std::vector<ShardInfo>& shards);
    
    // Security and verification
    bool VerifyShardIntegrity(const ShardInfo& shard);
    bool VerifyFileIntegrity(const std::vector<uint8_t>& file_data, 
                           const std::string& expected_hash);

private:
    std::map<std::string, ShardManifest> cached_manifests_;
    bool debug_mode_;
    
    // Helper methods
    ShardManifest CreateMockManifest(const std::string& file_id, const std::string& filename);
    std::vector<uint8_t> GenerateMockFileContent(const std::string& filename);
    std::string GenerateShardId() const;
};

} // namespace lite_browser

#endif // LITE_BROWSER_PROTOCOL_HANDLERS_SHARD_PROTOCOL_HANDLER_H_