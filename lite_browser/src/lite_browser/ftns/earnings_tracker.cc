// FTNS Earnings Tracker Implementation
#include "lite_browser/ftns/earnings_tracker.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <filesystem>

namespace lite_browser {

FTNSEarningsTracker::FTNSEarningsTracker() 
    : current_balance_(0.0),
      pending_validation_(0.0),
      is_contributing_(false),
      minimum_payout_threshold_(100.0),
      current_cpu_usage_(0.0),
      current_storage_usage_(0.0),
      current_bandwidth_usage_(0.0) {
    
    std::cout << "💰 Initializing FTNS Earnings Tracker..." << std::endl;
    
    // Enable all contribution types by default
    enabled_contributions_[ContributionType::STORAGE] = true;
    enabled_contributions_[ContributionType::COMPUTE] = true;
    enabled_contributions_[ContributionType::BANDWIDTH] = true;
    enabled_contributions_[ContributionType::RESEARCH_VALIDATION] = false; // Requires approval
    enabled_contributions_[ContributionType::PEER_DISCOVERY] = true;
    enabled_contributions_[ContributionType::COLLABORATION_HOSTING] = true;
    
    // Load existing earning history
    LoadEarningHistory();
    
    std::cout << "   Current balance: " << current_balance_ << " FTNS" << std::endl;
    std::cout << "   Pending validation: " << pending_validation_ << " FTNS" << std::endl;
    std::cout << "   Minimum payout: " << minimum_payout_threshold_ << " FTNS" << std::endl;
}

FTNSEarningsTracker::~FTNSEarningsTracker() {
    std::cout << "💰 Shutting down FTNS Earnings Tracker..." << std::endl;
    SaveEarningHistory();
}

std::string FTNSEarningsTracker::RecordEarning(ContributionType type,
                                              double ftns_amount,
                                              const std::string& task_id,
                                              const std::string& description) {
    
    if (!enabled_contributions_[type]) {
        std::cout << "⚠️  Contribution type " << ContributionTypeToString(type) 
                  << " is disabled" << std::endl;
        return "";
    }
    
    EarningEvent event;
    event.event_id = GenerateEventId();
    event.type = type;
    event.ftns_earned = ftns_amount;
    event.timestamp = std::chrono::system_clock::now();
    event.description = description;
    event.task_id = task_id;
    event.validated = false;
    event.paid_out = false;
    
    // Record current resource usage
    event.cpu_hours_used = current_cpu_usage_;
    event.storage_gb_hours = current_storage_usage_;
    event.bandwidth_gb = current_bandwidth_usage_;
    
    earning_history_.push_back(event);
    pending_validation_ += ftns_amount;
    
    std::cout << "💰 FTNS Earning Recorded:" << std::endl;
    std::cout << "   Event ID: " << event.event_id << std::endl;
    std::cout << "   Type: " << ContributionTypeToString(type) << std::endl;
    std::cout << "   Amount: " << ftns_amount << " FTNS" << std::endl;
    std::cout << "   Description: " << description << std::endl;
    std::cout << "   Status: Pending validation" << std::endl;
    
    return event.event_id;
}

bool FTNSEarningsTracker::ValidateEarning(const std::string& event_id) {
    auto it = std::find_if(earning_history_.begin(), earning_history_.end(),
                          [&event_id](const EarningEvent& event) {
                              return event.event_id == event_id;
                          });
    
    if (it == earning_history_.end()) {
        std::cout << "❌ Earning event not found: " << event_id << std::endl;
        return false;
    }
    
    if (it->validated) {
        std::cout << "ℹ️  Event already validated: " << event_id << std::endl;
        return true;
    }
    
    // Mock validation process
    std::cout << "🔍 Validating earning event: " << event_id << std::endl;
    std::cout << "   Amount: " << it->ftns_earned << " FTNS" << std::endl;
    std::cout << "   Type: " << ContributionTypeToString(it->type) << std::endl;
    
    // Simulate validation (always succeeds in demo)
    it->validated = true;
    current_balance_ += it->ftns_earned;
    pending_validation_ -= it->ftns_earned;
    
    std::cout << "✅ Earning validated and added to balance" << std::endl;
    std::cout << "   New balance: " << current_balance_ << " FTNS" << std::endl;
    
    // Check if payout threshold is reached
    if (current_balance_ >= minimum_payout_threshold_) {
        std::cout << "🎉 Payout threshold reached! Consider processing payout." << std::endl;
    }
    
    return true;
}

bool FTNSEarningsTracker::ProcessPayout(double minimum_threshold) {
    if (current_balance_ < minimum_threshold) {
        std::cout << "💸 Insufficient balance for payout:" << std::endl;
        std::cout << "   Current: " << current_balance_ << " FTNS" << std::endl;
        std::cout << "   Required: " << minimum_threshold << " FTNS" << std::endl;
        return false;
    }
    
    std::cout << "💸 Processing FTNS payout:" << std::endl;
    std::cout << "   Amount: " << current_balance_ << " FTNS" << std::endl;
    std::cout << "   Address: " << (payout_address_.empty() ? "institutional_default" : payout_address_) << std::endl;
    
    // Mock payout process
    double payout_amount = current_balance_;
    current_balance_ = 0.0;
    
    // Mark relevant events as paid out
    for (auto& event : earning_history_) {
        if (event.validated && !event.paid_out) {
            event.paid_out = true;
        }
    }
    
    std::cout << "✅ Payout processed successfully!" << std::endl;
    std::cout << "   " << payout_amount << " FTNS transferred" << std::endl;
    std::cout << "   Remaining balance: " << current_balance_ << " FTNS" << std::endl;
    
    return true;
}

EarningStatus FTNSEarningsTracker::GetCurrentStatus() {
    EarningStatus status;
    
    // Calculate lifetime earnings
    status.total_lifetime_earnings = 0.0;
    for (const auto& event : earning_history_) {
        if (event.validated) {
            status.total_lifetime_earnings += event.ftns_earned;
        }
    }
    
    status.current_balance = current_balance_;
    status.pending_validation = pending_validation_;
    
    // Calculate monthly earnings (last 30 days)
    auto now = std::chrono::system_clock::now();
    auto thirty_days_ago = now - std::chrono::hours(24 * 30);
    
    status.monthly_earnings = 0.0;
    for (const auto& event : earning_history_) {
        if (event.timestamp >= thirty_days_ago && event.validated) {
            status.monthly_earnings += event.ftns_earned;
        }
    }
    
    // Estimate monthly potential (based on current settings)
    status.estimated_monthly_potential = status.monthly_earnings * 1.2; // 20% growth estimate
    
    // Current contribution status
    status.currently_contributing_compute = is_contributing_ && enabled_contributions_[ContributionType::COMPUTE];
    status.currently_contributing_storage = is_contributing_ && enabled_contributions_[ContributionType::STORAGE];
    status.currently_contributing_bandwidth = is_contributing_ && enabled_contributions_[ContributionType::BANDWIDTH];
    
    // Next payout info
    status.next_payout_amount = current_balance_;
    status.days_until_payout = (minimum_payout_threshold_ - current_balance_) / (status.monthly_earnings / 30.0);
    if (current_balance_ >= minimum_payout_threshold_) {
        status.days_until_payout = 0;
        status.next_payout_date = "Available now";
    } else {
        status.next_payout_date = "Est. " + std::to_string(static_cast<int>(status.days_until_payout)) + " days";
    }
    
    return status;
}

std::vector<FTNSTask> FTNSEarningsTracker::GetAvailableTasks() {
    std::cout << "🔍 Fetching available FTNS tasks..." << std::endl;
    
    // Mock available tasks
    std::vector<FTNSTask> tasks;
    
    // Storage task
    FTNSTask storage_task;
    storage_task.task_id = "storage_task_001";
    storage_task.task_type = "Distributed Storage";
    storage_task.requester_institution = "mit.edu";
    storage_task.ftns_rate_per_hour = 0.5;
    storage_task.estimated_duration_hours = 168; // 1 week
    storage_task.total_ftns_potential = 84.0;
    storage_task.task_description = "Store research dataset shards for quantum computing project";
    storage_task.required_resources = "10GB storage, stable connection";
    storage_task.requires_validation = false;
    storage_task.deadline = std::chrono::system_clock::now() + std::chrono::hours(24 * 7);
    tasks.push_back(storage_task);
    
    // Compute task
    FTNSTask compute_task;
    compute_task.task_id = "compute_task_002";
    compute_task.task_type = "ML Training";
    compute_task.requester_institution = "stanford.edu";
    compute_task.ftns_rate_per_hour = 5.0;
    compute_task.estimated_duration_hours = 12;
    compute_task.total_ftns_potential = 60.0;
    compute_task.task_description = "Train neural network model for climate research";
    compute_task.required_resources = "25% CPU, 4GB RAM";
    compute_task.requires_validation = true;
    compute_task.deadline = std::chrono::system_clock::now() + std::chrono::hours(48);
    tasks.push_back(compute_task);
    
    // Research validation task
    FTNSTask validation_task;
    validation_task.task_id = "validation_task_003";
    validation_task.task_type = "Research Validation";
    validation_task.requester_institution = "unc.edu";
    validation_task.ftns_rate_per_hour = 15.0;
    validation_task.estimated_duration_hours = 4;
    validation_task.total_ftns_potential = 60.0;
    validation_task.task_description = "Validate quantum algorithm implementation";
    validation_task.required_resources = "Domain expertise in quantum computing";
    validation_task.requires_validation = true;
    validation_task.deadline = std::chrono::system_clock::now() + std::chrono::hours(72);
    tasks.push_back(validation_task);
    
    std::cout << "   Found " << tasks.size() << " available tasks" << std::endl;
    
    for (const auto& task : tasks) {
        std::cout << "   • " << task.task_type << " (" << task.requester_institution 
                  << "): " << task.total_ftns_potential << " FTNS potential" << std::endl;
    }
    
    return tasks;
}

void FTNSEarningsTracker::StartComputeContribution() {
    if (is_contributing_) {
        std::cout << "ℹ️  Compute contribution already active" << std::endl;
        return;
    }
    
    std::cout << "🚀 Starting compute contribution..." << std::endl;
    is_contributing_ = true;
    last_contribution_time_ = std::chrono::system_clock::now();
    
    // Start earning from background contributions
    if (enabled_contributions_[ContributionType::STORAGE]) {
        RecordEarning(ContributionType::STORAGE, 0.1, "background_storage", 
                     "Background storage contribution started");
    }
    
    if (enabled_contributions_[ContributionType::PEER_DISCOVERY]) {
        RecordEarning(ContributionType::PEER_DISCOVERY, 0.05, "peer_discovery",
                     "Helping peers discover research networks");
    }
    
    std::cout << "✅ Compute contribution active - earning FTNS in background" << std::endl;
}

void FTNSEarningsTracker::StopComputeContribution() {
    if (!is_contributing_) {
        std::cout << "ℹ️  Compute contribution not active" << std::endl;
        return;
    }
    
    std::cout << "🛑 Stopping compute contribution..." << std::endl;
    
    // Calculate final earnings for this session
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::hours>(now - last_contribution_time_);
    
    if (duration.count() > 0) {
        double session_earnings = duration.count() * 0.2; // Base rate
        RecordEarning(ContributionType::COMPUTE, session_earnings, "session_complete",
                     "Contribution session completed: " + std::to_string(duration.count()) + " hours");
    }
    
    is_contributing_ = false;
    
    std::cout << "✅ Compute contribution stopped" << std::endl;
    std::cout << "   Session duration: " << duration.count() << " hours" << std::endl;
}

void FTNSEarningsTracker::UpdateResourceUsage(double cpu_percent, double storage_gb, double bandwidth_mbps) {
    current_cpu_usage_ = cpu_percent;
    current_storage_usage_ = storage_gb;
    current_bandwidth_usage_ = bandwidth_mbps;
    
    // Award micro-earnings for active resource usage
    if (is_contributing_) {
        double earnings = (cpu_percent / 100.0) * 0.01; // Small ongoing earnings
        
        if (earnings > 0.001) { // Only record if meaningful
            RecordEarning(ContributionType::COMPUTE, earnings, "resource_usage",
                         "Resource utilization: " + std::to_string(static_cast<int>(cpu_percent)) + "% CPU");
        }
    }
}

double FTNSEarningsTracker::GetAverageHourlyRate() {
    if (earning_history_.empty()) return 0.0;
    
    double total_earnings = 0.0;
    double total_hours = 0.0;
    
    for (const auto& event : earning_history_) {
        if (event.validated) {
            total_earnings += event.ftns_earned;
            total_hours += event.cpu_hours_used;
        }
    }
    
    return total_hours > 0 ? total_earnings / total_hours : 0.0;
}

std::map<ContributionType, double> FTNSEarningsTracker::GetEarningsByType() {
    std::map<ContributionType, double> earnings_by_type;
    
    for (const auto& event : earning_history_) {
        if (event.validated) {
            earnings_by_type[event.type] += event.ftns_earned;
        }
    }
    
    return earnings_by_type;
}

// Helper method implementations
std::string FTNSEarningsTracker::GenerateEventId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dist(100000, 999999);
    
    return "ftns_event_" + std::to_string(dist(gen));
}

double FTNSEarningsTracker::CalculateEarningRate(ContributionType type) {
    switch (type) {
        case ContributionType::STORAGE:
            return 0.3; // FTNS per GB per hour
        case ContributionType::COMPUTE:
            return 2.0; // FTNS per CPU hour
        case ContributionType::BANDWIDTH:
            return 0.1; // FTNS per Mbps hour
        case ContributionType::RESEARCH_VALIDATION:
            return 15.0; // High rate for expert validation
        case ContributionType::PEER_DISCOVERY:
            return 0.5; // Medium rate for network services
        case ContributionType::COLLABORATION_HOSTING:
            return 1.0; // Medium rate for hosting
        default:
            return 1.0;
    }
}

bool FTNSEarningsTracker::LoadEarningHistory() {
    std::string data_file = GetDataFilePath();
    std::ifstream file(data_file);
    
    if (!file.is_open()) {
        std::cout << "ℹ️  No existing earning history found" << std::endl;
        return true; // Not an error for first run
    }
    
    // Mock loading - in real implementation would parse JSON
    std::string line;
    int event_count = 0;
    
    while (std::getline(file, line) && event_count < 5) {
        // Create mock historical events
        EarningEvent event;
        event.event_id = GenerateEventId();
        event.type = ContributionType::STORAGE;
        event.ftns_earned = 1.0 + (event_count * 0.5);
        event.timestamp = std::chrono::system_clock::now() - std::chrono::hours(24 * event_count);
        event.description = "Historical storage contribution";
        event.validated = true;
        event.paid_out = false;
        
        earning_history_.push_back(event);
        current_balance_ += event.ftns_earned;
        event_count++;
    }
    
    file.close();
    
    std::cout << "📁 Loaded " << event_count << " historical earning events" << std::endl;
    return true;
}

bool FTNSEarningsTracker::SaveEarningHistory() {
    std::string data_file = GetDataFilePath();
    std::ofstream file(data_file);
    
    if (!file.is_open()) {
        std::cout << "❌ Failed to save earning history" << std::endl;
        return false;
    }
    
    // Mock saving - in real implementation would write JSON
    file << "# FTNS Earning History" << std::endl;
    file << "# Events: " << earning_history_.size() << std::endl;
    file << "# Balance: " << current_balance_ << std::endl;
    file << "# Pending: " << pending_validation_ << std::endl;
    
    for (const auto& event : earning_history_) {
        file << event.event_id << "," << static_cast<int>(event.type) 
             << "," << event.ftns_earned << "," << event.validated << std::endl;
    }
    
    file.close();
    
    std::cout << "💾 Earning history saved: " << earning_history_.size() << " events" << std::endl;
    return true;
}

std::string FTNSEarningsTracker::GetDataFilePath() {
    std::string home_dir = std::getenv("HOME") ? std::getenv("HOME") : ".";
    std::string data_dir = home_dir + "/.config/lite_browser/ftns";
    
    std::filesystem::create_directories(data_dir);
    return data_dir + "/earnings.dat";
}

// Utility function implementations
std::string ContributionTypeToString(ContributionType type) {
    switch (type) {
        case ContributionType::STORAGE: return "Storage";
        case ContributionType::COMPUTE: return "Compute";
        case ContributionType::BANDWIDTH: return "Bandwidth";
        case ContributionType::RESEARCH_VALIDATION: return "Research Validation";
        case ContributionType::PEER_DISCOVERY: return "Peer Discovery";
        case ContributionType::COLLABORATION_HOSTING: return "Collaboration Hosting";
        default: return "Unknown";
    }
}

ContributionType StringToContributionType(const std::string& type_str) {
    if (type_str == "Storage") return ContributionType::STORAGE;
    if (type_str == "Compute") return ContributionType::COMPUTE;
    if (type_str == "Bandwidth") return ContributionType::BANDWIDTH;
    if (type_str == "Research Validation") return ContributionType::RESEARCH_VALIDATION;
    if (type_str == "Peer Discovery") return ContributionType::PEER_DISCOVERY;
    if (type_str == "Collaboration Hosting") return ContributionType::COLLABORATION_HOSTING;
    return ContributionType::STORAGE; // Default
}

} // namespace lite_browser