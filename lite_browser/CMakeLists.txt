cmake_minimum_required(VERSION 3.16)
project(LITEBrowser)

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

# LITE Browser executable
add_executable(lite_browser
    src/lite_browser/lite_browser_main.cc
    src/lite_browser/core/browser_main.cc
    src/lite_browser/protocol_handlers/lite_protocol_handler.cc
    src/lite_browser/protocol_handlers/shard_protocol_handler.cc
    src/lite_browser/protocol_handlers/collab_protocol_handler.cc
    src/lite_browser/onboarding/first_run_setup.cc
    src/lite_browser/ftns/earnings_tracker.cc
)

# Link libraries (simplified for demo)
# target_link_libraries(lite_browser
#     ${Boost_LIBRARIES}
#     OpenSSL::SSL
#     OpenSSL::Crypto
#     oqs
#     p2p
# )

# Compiler flags
target_compile_options(lite_browser PRIVATE
    -Wall
    -Wextra
    -O2
    -DLITE_BROWSER_VERSION="0.1.0"
)

# Installation
install(TARGETS lite_browser DESTINATION bin)