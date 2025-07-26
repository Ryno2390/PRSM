// Collaboration Protocol Handler
// Handles collab:// URLs for real-time multi-institutional collaboration

#ifndef LITE_BROWSER_PROTOCOL_HANDLERS_COLLAB_PROTOCOL_HANDLER_H_
#define LITE_BROWSER_PROTOCOL_HANDLERS_COLLAB_PROTOCOL_HANDLER_H_

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <chrono>

namespace lite_browser {

// Collaboration URL structure: collab://session-type/session-name
class CollabUrl {
public:
    explicit CollabUrl(const std::string& url);
    
    bool IsValid() const { return is_valid_; }
    std::string GetSessionType() const { return session_type_; }
    std::string GetSessionName() const { return session_name_; }
    std::string GetDescription() const;

private:
    bool is_valid_;
    std::string session_type_;
    std::string session_name_;
    void ParseUrl(const std::string& url);
};

// Participant information
struct Participant {
    std::string user_id;
    std::string institution;
    std::string role;
    bool is_online;
    std::chrono::system_clock::time_point last_seen;
    std::vector<std::string> permissions;
};

// Collaboration session
struct CollaborationSession {
    std::string session_id;
    std::string session_name;
    std::string session_type;
    std::string owner;
    std::chrono::system_clock::time_point created_at;
    
    // Participants and permissions
    std::map<std::string, Participant> participants;
    std::vector<std::string> required_approvers;
    
    // Session configuration
    bool multi_signature_required;
    std::string security_level;
    bool real_time_editing;
    
    // Current state
    std::string current_document;
    std::vector<std::string> pending_approvals;
    bool is_active;
};

// Approval workflow
struct ApprovalRequest {
    std::string request_id;
    std::string session_id;
    std::string requester;
    std::string action_type;
    std::string description;
    std::vector<std::string> required_approvers;
    std::map<std::string, bool> approvals;
    std::chrono::system_clock::time_point created_at;
    bool is_approved;
};

// Main collaboration protocol handler
class CollabProtocolHandler {
public:
    CollabProtocolHandler();
    ~CollabProtocolHandler();
    
    // Handle collaboration URL requests
    bool CanHandle(const std::string& url) const;
    std::unique_ptr<CollaborationSession> HandleRequest(const CollabUrl& collab_url);
    
    // Session management
    std::unique_ptr<CollaborationSession> CreateSession(const std::string& session_type,
                                                       const std::string& session_name,
                                                       const std::string& owner);
    std::unique_ptr<CollaborationSession> JoinSession(const std::string& session_id,
                                                     const std::string& user_id);
    bool LeaveSession(const std::string& session_id, const std::string& user_id);
    
    // Multi-signature approval workflows
    std::string RequestApproval(const std::string& session_id,
                               const std::string& action_type,
                               const std::string& description,
                               const std::vector<std::string>& required_approvers);
    bool SubmitApproval(const std::string& request_id,
                       const std::string& approver,
                       bool approved);
    
    // Real-time collaboration
    bool SendCollaborativeEdit(const std::string& session_id,
                              const std::string& user_id,
                              const std::string& edit_data);
    std::vector<std::string> GetRecentEdits(const std::string& session_id);

private:
    std::map<std::string, std::unique_ptr<CollaborationSession>> active_sessions_;
    std::map<std::string, ApprovalRequest> pending_approvals_;
    bool debug_mode_;
    std::string user_institution_;
    
    // Helper methods
    std::string GenerateSessionId() const;
    std::string GenerateRequestId() const;
    std::unique_ptr<CollaborationSession> CreateMockSession(const std::string& session_type,
                                                           const std::string& session_name);
    bool CheckPermissions(const CollaborationSession& session,
                         const std::string& user_id,
                         const std::string& action) const;
};

} // namespace lite_browser

#endif // LITE_BROWSER_PROTOCOL_HANDLERS_COLLAB_PROTOCOL_HANDLER_H_