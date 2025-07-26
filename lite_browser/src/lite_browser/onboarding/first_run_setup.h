// First Run Setup Interface
// Handles user onboarding, profile creation, and FTNS configuration

#ifndef LITE_BROWSER_ONBOARDING_FIRST_RUN_SETUP_H_
#define LITE_BROWSER_ONBOARDING_FIRST_RUN_SETUP_H_

#include <string>
#include <vector>
#include <map>

namespace lite_browser {

// User profile configuration
struct UserProfile {
    std::string full_name;
    std::string email;
    std::string institution;
    std::string institution_domain;
    std::string orcid_id;
    std::vector<std::string> research_areas;
    std::string academic_title;
    std::string department;
    
    // Collaboration preferences
    bool allow_industry_collaboration;
    bool require_multi_sig_approval;
    bool auto_accept_institutional_invites;
    std::vector<std::string> trusted_institutions;
};

// Compute contribution settings
struct ComputeContribution {
    bool enabled;
    int max_cpu_usage_percent;      // 0-100%
    int max_storage_gb;             // Available storage in GB
    int max_bandwidth_mbps;         // Max bandwidth contribution
    bool contribute_during_idle_only;
    bool exclude_sensitive_files;
    std::vector<std::string> contribution_types; // "storage", "compute", "bandwidth"
    
    // Advanced settings
    bool allow_ml_training;
    bool allow_data_processing;
    std::vector<std::string> excluded_file_types;
};

// FTNS (Federated Token Network System) configuration
struct FTNSConfiguration {
    bool auto_earn_enabled;
    int minimum_payout_threshold;   // Minimum FTNS before payout
    std::string payout_address;     // Wallet address for payouts
    std::vector<std::string> preferred_contribution_types;
    bool institutional_validation_required;
    
    // Earning preferences
    bool prioritize_research_tasks;
    bool accept_industry_compute_tasks;
    double min_hourly_rate_ftns;    // Minimum FTNS per hour for tasks
};

// Network and security preferences
struct NetworkSettings {
    bool enable_ipv6;
    int max_peer_connections;
    std::vector<std::string> bootstrap_nodes;
    bool auto_discover_institutional_peers;
    bool restrict_to_academic_networks;
};

struct SecuritySettings {
    bool post_quantum_enabled;
    bool strict_certificate_validation;
    bool encrypt_local_storage;
    bool auto_update_trust_anchors;
    std::string encryption_level; // "standard", "high", "maximum"
};

// Complete onboarding configuration
struct OnboardingConfig {
    UserProfile user_profile;
    ComputeContribution compute_settings;
    FTNSConfiguration ftns_settings;
    NetworkSettings network_settings;
    SecuritySettings security_settings;
    
    bool setup_completed;
    std::string setup_completion_time;
};

// First run setup handler
class FirstRunSetup {
public:
    FirstRunSetup();
    ~FirstRunSetup();
    
    // Main setup flow
    bool RunFirstTimeSetup();
    bool IsFirstRun() const;
    
    // Setup steps
    bool WelcomeAndIntroduction();
    bool CollectUserProfile(UserProfile& profile);
    bool ConfigureComputeContribution(ComputeContribution& compute);
    bool ConfigureFTNSSettings(FTNSConfiguration& ftns);
    bool ConfigureNetworkSettings(NetworkSettings& network);
    bool ConfigureSecuritySettings(SecuritySettings& security);
    bool FinalizeSetup(const OnboardingConfig& config);
    
    // Institutional verification
    bool VerifyInstitutionalAffiliation(const std::string& email, const std::string& domain);
    std::vector<std::string> GetSupportedInstitutions();
    
    // Configuration management
    bool SaveConfiguration(const OnboardingConfig& config);
    OnboardingConfig LoadConfiguration();
    
    // Resource estimation
    int EstimateMonthlyFTNSEarnings(const ComputeContribution& compute);
    std::string GetResourceUsageSummary(const ComputeContribution& compute);

private:
    std::string config_file_path_;
    bool debug_mode_;
    
    // Helper methods
    std::string PromptUserInput(const std::string& prompt, bool required = true);
    bool PromptYesNo(const std::string& question, bool default_yes = false);
    int PromptInteger(const std::string& prompt, int min_val, int max_val, int default_val);
    std::vector<std::string> PromptMultiSelect(const std::string& prompt, 
                                              const std::vector<std::string>& options);
    
    void DisplayWelcomeMessage();
    void DisplayFTNSInformation();
    void DisplaySecurityInformation();
    void DisplaySetupSummary(const OnboardingConfig& config);
    
    // Validation
    bool ValidateEmail(const std::string& email);
    bool ValidateInstitutionDomain(const std::string& domain);
    bool ValidateORCID(const std::string& orcid);
    
    // Resource calculations
    int CalculateStorageContribution(int available_gb);
    int CalculateComputeContribution(int cpu_percent);
    double EstimateFTNSRate(const std::string& contribution_type);
};

} // namespace lite_browser

#endif // LITE_BROWSER_ONBOARDING_FIRST_RUN_SETUP_H_