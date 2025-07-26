// PRSM Browser Main Entry Point
// This is the main entry point for the PRSM Browser application

#include <iostream>
#include "prsm_browser/core/browser_main.h"

int main(int argc, char* argv[]) {
    std::cout << "🚀 Starting PRSM Browser..." << std::endl;
    
    // Initialize PRSM Browser
    return prsm_browser::BrowserMain(argc, argv);
}