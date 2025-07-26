// FTNS Earnings Tracker
// Tracks and manages FTNS token earnings from compute contributions

#ifndef LITE_BROWSER_FTNS_EARNINGS_TRACKER_H_
#define LITE_BROWSER_FTNS_EARNINGS_TRACKER_H_

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <memory>

namespace lite_browser {

// Types of contributions that earn FTNS
enum class ContributionType {
    STORAGE,
    COMPUTE,
    BANDWIDTH,
    RESEARCH_VALIDATION,
    PEER_DISCOVERY,
    COLLABORATION_HOSTING
};

// Individual earning event
struct EarningEvent {
    std::string event_id;
    ContributionType type;
    double ftns_earned;
    std::chrono::system_clock::time_point timestamp;
    std::string description;
    std::string task_id;
    bool validated;
    bool paid_out;
    
    // Resource usage details
    double cpu_hours_used;
    double storage_gb_hours;
    double bandwidth_gb;
    std::string peer_institution;
};

// Daily earnings summary
struct DailyEarnings {
    std::string date;
    double total_ftns;
    int contribution_count;
    std::map<ContributionType, double> earnings_by_type;
    double cpu_utilization_avg;
    double storage_utilization_avg;
};

// Monthly earnings report
struct MonthlyReport {
    int year;
    int month;
    double total_ftns_earned;
    double total_ftns_paid_out;
    double pending_payout;
    std::vector<DailyEarnings> daily_breakdown;
    
    // Performance metrics
    double avg_daily_earnings;
    double peak_daily_earnings;
    int active_contribution_days;
    
    // Resource efficiency
    double ftns_per_cpu_hour;
    double ftns_per_gb_storage;
    double ftns_per_gb_bandwidth;
};

// Current earning status
struct EarningStatus {
    double total_lifetime_earnings;
    double current_balance;
    double pending_validation;
    double monthly_earnings;
    double estimated_monthly_potential;
    
    // Current contributions
    bool currently_contributing_compute;
    bool currently_contributing_storage;
    bool currently_contributing_bandwidth;
    
    // Next payout info
    double next_payout_amount;
    std::string next_payout_date;
    int days_until_payout;
};

// FTNS task information
struct FTNSTask {
    std::string task_id;
    std::string task_type;
    std::string requester_institution;
    double ftns_rate_per_hour;
    double estimated_duration_hours;
    double total_ftns_potential;
    std::string task_description;
    std::string required_resources;
    bool requires_validation;
    std::chrono::system_clock::time_point deadline;
};

// Main FTNS earnings tracker
class FTNSEarningsTracker {
public:
    FTNSEarningsTracker();
    ~FTNSEarningsTracker();
    
    // Earning tracking
    std::string RecordEarning(ContributionType type, 
                             double ftns_amount,
                             const std::string& task_id,
                             const std::string& description);
    
    bool ValidateEarning(const std::string& event_id);
    bool ProcessPayout(double minimum_threshold);
    
    // Status and reporting
    EarningStatus GetCurrentStatus();
    MonthlyReport GenerateMonthlyReport(int year, int month);
    std::vector<DailyEarnings> GetRecentEarnings(int days);
    std::vector<EarningEvent> GetEarningHistory(int limit = 100);
    
    // Task management
    std::vector<FTNSTask> GetAvailableTasks();
    bool AcceptTask(const std::string& task_id);
    bool CompleteTask(const std::string& task_id, const std::string& result_data);
    
    // Contribution monitoring
    void StartComputeContribution();
    void StopComputeContribution();
    void UpdateResourceUsage(double cpu_percent, double storage_gb, double bandwidth_mbps);
    
    // Settings
    void SetMinimumPayoutThreshold(double threshold);
    void SetPayoutAddress(const std::string& address);
    void EnableContributionType(ContributionType type, bool enabled);
    
    // Statistics
    double GetAverageHourlyRate();
    double GetEfficiencyScore();
    std::map<ContributionType, double> GetEarningsByType();

private:
    std::vector<EarningEvent> earning_history_;
    std::map<std::string, FTNSTask> active_tasks_;
    
    // Current status
    double current_balance_;
    double pending_validation_;
    bool is_contributing_;
    std::chrono::system_clock::time_point last_contribution_time_;
    
    // Settings
    double minimum_payout_threshold_;
    std::string payout_address_;
    std::map<ContributionType, bool> enabled_contributions_;
    
    // Resource monitoring
    double current_cpu_usage_;
    double current_storage_usage_;
    double current_bandwidth_usage_;
    
    // Helper methods
    std::string GenerateEventId();
    double CalculateEarningRate(ContributionType type);
    bool ShouldAcceptTask(const FTNSTask& task);
    void UpdateEarningStatistics();
    
    // Storage
    bool SaveEarningHistory();
    bool LoadEarningHistory();
    std::string GetDataFilePath();
    
    // Validation
    bool ValidateTaskCompletion(const std::string& task_id, const std::string& result);
    bool RequestInstitutionalApproval(const std::string& event_id);
};

// Utility functions
std::string ContributionTypeToString(ContributionType type);
ContributionType StringToContributionType(const std::string& type_str);
double CalculateOptimalResourceAllocation(double available_cpu, double available_storage);

} // namespace lite_browser

#endif // LITE_BROWSER_FTNS_EARNINGS_TRACKER_H_