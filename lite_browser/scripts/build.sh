#!/bin/bash

# PRSM Browser Build Script
# Compiles the PRSM Browser from source

set -e

echo "🔨 Building PRSM Browser..."

# Check if dependencies are installed
if [ ! -d "../third_party/liboqs" ]; then
    echo "❌ Dependencies not found. Please run ./install_deps.sh first"
    exit 1
fi

# Create build directory
mkdir -p ../build
cd ../build

# Configure build with CMake
echo "⚙️ Configuring build with CMake..."
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    ..

# Build PRSM Browser
echo "🏗️ Compiling PRSM Browser..."
ninja

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ PRSM Browser build successful!"
    echo ""
    echo "📍 Executable location: ../build/prsm_browser"
    echo ""
    echo "🚀 Run with: ./run_dev.sh"
else
    echo "❌ Build failed!"
    exit 1
fi