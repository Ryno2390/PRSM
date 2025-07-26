// Collaboration Protocol Handler Implementation
#include "lite_browser/protocol_handlers/collab_protocol_handler.h"

#include <iostream>
#include <sstream>
#include <random>
#include <regex>
#include <algorithm>

namespace lite_browser {

// CollabUrl Implementation
CollabUrl::CollabUrl(const std::string& url) : is_valid_(false) {
    ParseUrl(url);
}

void CollabUrl::ParseUrl(const std::string& url) {
    // Parse URL format: collab://session-type/session-name
    std::regex collab_regex(R"(^collab://([^/]+)/(.+)$)");
    std::smatch matches;
    
    if (std::regex_match(url, matches, collab_regex)) {
        is_valid_ = true;
        session_type_ = matches[1].str();
        session_name_ = matches[2].str();
        
        std::cout << "✅ Parsed Collaboration URL:" << std::endl;
        std::cout << "   Session Type: " << session_type_ << std::endl;
        std::cout << "   Session Name: " << session_name_ << std::endl;
    } else {
        std::cout << "❌ Invalid Collaboration URL format: " << url << std::endl;
    }
}

std::string CollabUrl::GetDescription() const {
    if (!is_valid_) return "Invalid Collaboration URL";
    return "Collaboration session: " + session_name_ + " (Type: " + session_type_ + ")";
}

// CollabProtocolHandler Implementation
CollabProtocolHandler::CollabProtocolHandler() 
    : debug_mode_(true), user_institution_("unc.edu") {
    
    std::cout << "🤝 Initializing Collaboration Protocol Handler..." << std::endl;
    std::cout << "   🔗 Real-Time Multi-Institutional Collaboration Engine" << std::endl;
    std::cout << "   User Institution: " << user_institution_ << std::endl;
    std::cout << "   Debug Mode: " << (debug_mode_ ? "ON" : "OFF") << std::endl;
}

CollabProtocolHandler::~CollabProtocolHandler() {
    std::cout << "🤝 Shutting down Collaboration Protocol Handler..." << std::endl;
    
    // Clean up active sessions
    for (auto& [session_id, session] : active_sessions_) {
        std::cout << "   Closing session: " << session->session_name << std::endl;
    }
    active_sessions_.clear();
}

bool CollabProtocolHandler::CanHandle(const std::string& url) const {
    return url.substr(0, 9) == "collab://";
}

std::unique_ptr<CollaborationSession> CollabProtocolHandler::HandleRequest(const CollabUrl& collab_url) {
    if (!collab_url.IsValid()) {
        std::cout << "❌ Cannot handle invalid Collaboration URL" << std::endl;
        return nullptr;
    }
    
    std::cout << "🤝 Handling Collaboration request: " << collab_url.GetDescription() << std::endl;
    
    // Check if session already exists
    for (auto& [session_id, session] : active_sessions_) {
        if (session->session_name == collab_url.GetSessionName() && 
            session->session_type == collab_url.GetSessionType()) {
            
            std::cout << "   Joining existing session: " << session_id << std::endl;
            return JoinSession(session_id, "current_user@" + user_institution_);
        }
    }
    
    // Create new session
    std::cout << "   Creating new collaboration session..." << std::endl;
    return CreateSession(collab_url.GetSessionType(), 
                        collab_url.GetSessionName(),
                        "current_user@" + user_institution_);
}

std::unique_ptr<CollaborationSession> CollabProtocolHandler::CreateSession(
    const std::string& session_type,
    const std::string& session_name,
    const std::string& owner) {
    
    std::cout << "📝 Creating collaboration session: " << session_name << std::endl;
    
    auto session = CreateMockSession(session_type, session_name);
    session->owner = owner;
    
    // Add owner as first participant
    Participant owner_participant;
    owner_participant.user_id = owner;
    owner_participant.institution = user_institution_;
    owner_participant.role = "owner";
    owner_participant.is_online = true;
    owner_participant.last_seen = std::chrono::system_clock::now();
    owner_participant.permissions = {"read", "write", "approve", "invite"};
    
    session->participants[owner] = owner_participant;
    
    // Store in active sessions
    std::string session_id = session->session_id;
    active_sessions_[session_id] = std::make_unique<CollaborationSession>(*session);
    
    std::cout << "✅ Session created successfully:" << std::endl;
    std::cout << "   Session ID: " << session_id << std::endl;
    std::cout << "   Type: " << session->session_type << std::endl;
    std::cout << "   Owner: " << session->owner << std::endl;
    std::cout << "   Multi-sig required: " << (session->multi_signature_required ? "YES" : "NO") << std::endl;
    
    return session;
}

std::unique_ptr<CollaborationSession> CollabProtocolHandler::JoinSession(
    const std::string& session_id,
    const std::string& user_id) {
    
    std::cout << "👥 User joining session: " << user_id << " -> " << session_id << std::endl;
    
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        std::cout << "❌ Session not found: " << session_id << std::endl;
        return nullptr;
    }
    
    auto& session = it->second;
    
    // Check if user is already a participant
    if (session->participants.find(user_id) != session->participants.end()) {
        std::cout << "   User already in session, updating status..." << std::endl;
        session->participants[user_id].is_online = true;
        session->participants[user_id].last_seen = std::chrono::system_clock::now();
    } else {
        // Add new participant
        Participant new_participant;
        new_participant.user_id = user_id;
        new_participant.institution = user_id.substr(user_id.find('@') + 1);
        new_participant.role = "collaborator";
        new_participant.is_online = true;
        new_participant.last_seen = std::chrono::system_clock::now();
        new_participant.permissions = {"read", "write"};
        
        session->participants[user_id] = new_participant;
        
        std::cout << "   New participant added:" << std::endl;
        std::cout << "     User: " << user_id << std::endl;
        std::cout << "     Institution: " << new_participant.institution << std::endl;
        std::cout << "     Role: " << new_participant.role << std::endl;
    }
    
    std::cout << "✅ Successfully joined session" << std::endl;
    std::cout << "   Active participants: " << session->participants.size() << std::endl;
    
    return std::make_unique<CollaborationSession>(*session);
}

bool CollabProtocolHandler::LeaveSession(const std::string& session_id, const std::string& user_id) {
    std::cout << "👋 User leaving session: " << user_id << " <- " << session_id << std::endl;
    
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        return false;
    }
    
    auto& session = it->second;
    auto participant_it = session->participants.find(user_id);
    if (participant_it != session->participants.end()) {
        participant_it->second.is_online = false;
        participant_it->second.last_seen = std::chrono::system_clock::now();
        
        std::cout << "✅ User marked as offline" << std::endl;
        return true;
    }
    
    return false;
}

std::string CollabProtocolHandler::RequestApproval(
    const std::string& session_id,
    const std::string& action_type,
    const std::string& description,
    const std::vector<std::string>& required_approvers) {
    
    std::cout << "📋 Requesting approval for session: " << session_id << std::endl;
    std::cout << "   Action: " << action_type << std::endl;
    std::cout << "   Description: " << description << std::endl;
    
    ApprovalRequest request;
    request.request_id = GenerateRequestId();
    request.session_id = session_id;
    request.requester = "current_user@" + user_institution_;
    request.action_type = action_type;
    request.description = description;
    request.required_approvers = required_approvers;
    request.created_at = std::chrono::system_clock::now();
    request.is_approved = false;
    
    // Initialize approval status
    for (const auto& approver : required_approvers) {
        request.approvals[approver] = false;
    }
    
    pending_approvals_[request.request_id] = request;
    
    std::cout << "✅ Approval request created:" << std::endl;
    std::cout << "   Request ID: " << request.request_id << std::endl;
    std::cout << "   Required approvers: " << required_approvers.size() << std::endl;
    
    return request.request_id;
}

bool CollabProtocolHandler::SubmitApproval(const std::string& request_id,
                                          const std::string& approver,
                                          bool approved) {
    
    std::cout << "✅ Submitting approval: " << request_id << std::endl;
    std::cout << "   Approver: " << approver << std::endl;
    std::cout << "   Decision: " << (approved ? "APPROVED" : "REJECTED") << std::endl;
    
    auto it = pending_approvals_.find(request_id);
    if (it == pending_approvals_.end()) {
        std::cout << "❌ Approval request not found" << std::endl;
        return false;
    }
    
    auto& request = it->second;
    
    // Check if approver is authorized
    auto approver_it = std::find(request.required_approvers.begin(), 
                                request.required_approvers.end(), 
                                approver);
    if (approver_it == request.required_approvers.end()) {
        std::cout << "❌ Approver not authorized for this request" << std::endl;
        return false;
    }
    
    // Record approval
    request.approvals[approver] = approved;
    
    // Check if all approvals are received
    bool all_approved = true;
    int approvals_received = 0;
    
    for (const auto& [approver_id, approval_status] : request.approvals) {
        if (approval_status) {
            approvals_received++;
        } else {
            all_approved = false;
        }
    }
    
    if (all_approved && approvals_received == static_cast<int>(request.required_approvers.size())) {
        request.is_approved = true;
        std::cout << "🎉 All approvals received! Action approved." << std::endl;
    } else if (!approved) {
        std::cout << "❌ Action rejected by " << approver << std::endl;
    } else {
        std::cout << "⏳ Waiting for more approvals: " << approvals_received 
                  << "/" << request.required_approvers.size() << std::endl;
    }
    
    return true;
}

bool CollabProtocolHandler::SendCollaborativeEdit(const std::string& session_id,
                                                 const std::string& user_id,
                                                 const std::string& edit_data) {
    
    std::cout << "✏️  Collaborative edit in session: " << session_id << std::endl;
    std::cout << "   User: " << user_id << std::endl;
    std::cout << "   Edit size: " << edit_data.size() << " bytes" << std::endl;
    
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        std::cout << "❌ Session not found" << std::endl;
        return false;
    }
    
    auto& session = it->second;
    
    // Check permissions
    if (!CheckPermissions(*session, user_id, "write")) {
        std::cout << "❌ User does not have write permissions" << std::endl;
        return false;
    }
    
    // Mock processing the edit
    std::cout << "✅ Edit processed and synchronized across all participants" << std::endl;
    
    // Notify other participants
    int active_participants = 0;
    for (const auto& [participant_id, participant] : session->participants) {
        if (participant.is_online && participant_id != user_id) {
            active_participants++;
            std::cout << "   Notified: " << participant_id << std::endl;
        }
    }
    
    std::cout << "   Synchronized with " << active_participants << " other participants" << std::endl;
    
    return true;
}

std::vector<std::string> CollabProtocolHandler::GetRecentEdits(const std::string& session_id) {
    std::cout << "📜 Retrieving recent edits for session: " << session_id << std::endl;
    
    // Mock recent edits
    std::vector<std::string> recent_edits = {
        "User alice@mit.edu added section 'Introduction'",
        "User bob@stanford.edu edited 'Methodology' section",
        "User charlie@unc.edu added references",
        "User alice@mit.edu approved final draft"
    };
    
    std::cout << "   Found " << recent_edits.size() << " recent edits" << std::endl;
    
    return recent_edits;
}

// Private helper methods
std::string CollabProtocolHandler::GenerateSessionId() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dist(100000, 999999);
    
    return "collab_session_" + std::to_string(dist(gen));
}

std::string CollabProtocolHandler::GenerateRequestId() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dist(100000, 999999);
    
    return "approval_req_" + std::to_string(dist(gen));
}

std::unique_ptr<CollaborationSession> CollabProtocolHandler::CreateMockSession(
    const std::string& session_type,
    const std::string& session_name) {
    
    auto session = std::make_unique<CollaborationSession>();
    session->session_id = GenerateSessionId();
    session->session_name = session_name;
    session->session_type = session_type;
    session->created_at = std::chrono::system_clock::now();
    session->is_active = true;
    session->real_time_editing = true;
    
    // Configure based on session type
    if (session_type == "grant-proposal") {
        session->multi_signature_required = true;
        session->security_level = "high";
        session->required_approvers = {"tech.transfer@unc.edu", "grants.office@unc.edu"};
        session->current_document = "NSF Quantum Computing Grant Proposal";
    } else if (session_type == "research-paper") {
        session->multi_signature_required = false;
        session->security_level = "medium";
        session->current_document = "Collaborative Research Paper";
    } else if (session_type == "industry-evaluation") {
        session->multi_signature_required = true;
        session->security_level = "maximum";
        session->required_approvers = {"legal@unc.edu", "tech.transfer@unc.edu"};
        session->current_document = "Industry Partnership Evaluation";
    } else {
        session->multi_signature_required = false;
        session->security_level = "standard";
        session->current_document = "General Collaboration Session";
    }
    
    return session;
}

bool CollabProtocolHandler::CheckPermissions(const CollaborationSession& session,
                                           const std::string& user_id,
                                           const std::string& action) const {
    
    auto participant_it = session.participants.find(user_id);
    if (participant_it == session.participants.end()) {
        return false; // User not in session
    }
    
    const auto& permissions = participant_it->second.permissions;
    return std::find(permissions.begin(), permissions.end(), action) != permissions.end();
}

} // namespace lite_browser