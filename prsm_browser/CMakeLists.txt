cmake_minimum_required(VERSION 3.16)
project(PRSMBrowser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# For now, we'll build without external dependencies for demo purposes
# find_package(OpenSSL REQUIRED)
# find_package(Boost REQUIRED COMPONENTS system filesystem thread)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# PRSM Browser executable
add_executable(prsm_browser
    src/prsm_browser/prsm_browser_main.cc
    src/prsm_browser/core/browser_main.cc
    src/prsm_browser/protocol_handlers/prsm_protocol_handler.cc
)

# Link libraries (simplified for demo)
# target_link_libraries(prsm_browser
#     ${Boost_LIBRARIES}
#     OpenSSL::SSL
#     OpenSSL::Crypto
#     oqs
#     p2p
# )

# Compiler flags
target_compile_options(prsm_browser PRIVATE
    -Wall
    -Wextra
    -O2
    -DPRSM_BROWSER_VERSION="0.1.0"
)

# Installation
install(TARGETS prsm_browser DESTINATION bin)