// LITE Browser Main Entry Point
// Linked Information Transfer Engine - Native P2P Research Browser

#include <iostream>
#include <string>
#include "lite_browser/core/browser_main.h"
#include "lite_browser/onboarding/first_run_setup.h"

int main(int argc, char* argv[]) {
    // Check for first-run setup flag
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--first-run" || arg == "--setup") {
            lite_browser::FirstRunSetup setup;
            if (setup.RunFirstTimeSetup()) {
                std::cout << "\n🚀 Launching LITE Browser with new configuration..." << std::endl;
                // Continue to normal startup after setup
            } else {
                std::cout << "Setup cancelled or failed. Exiting." << std::endl;
                return 1;
            }
            break;
        }
    }
    
    // Check if this is truly a first run (no config exists)
    lite_browser::FirstRunSetup setup_checker;
    if (setup_checker.IsFirstRun()) {
        std::cout << "💡 Welcome to LITE Browser - First Time Setup Required" << std::endl;
        std::cout << "🔗 Linked Information Transfer Engine" << std::endl;
        
        if (setup_checker.RunFirstTimeSetup()) {
            std::cout << "\n🚀 Setup complete! Starting LITE Browser..." << std::endl;
        } else {
            std::cout << "Setup is required to use LITE Browser. Exiting." << std::endl;
            return 1;
        }
    } else {
        std::cout << "💡 Starting LITE Browser..." << std::endl;
        std::cout << "🔗 Linked Information Transfer Engine" << std::endl;
    }
    
    // Initialize LITE Browser
    return lite_browser::BrowserMain(argc, argv);
}