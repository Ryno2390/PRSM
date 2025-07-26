#!/bin/bash

# PRSM Browser Development Runner
# Starts the PRSM Browser in development mode

set -e

echo "🚀 Starting PRSM Browser (Development Mode)..."

# Check if browser is built
if [ ! -f "../build/prsm_browser" ]; then
    echo "❌ PRSM Browser not found. Please run ./build.sh first"
    exit 1
fi

# Set development environment variables
export PRSM_BROWSER_MODE="development"
export PRSM_BROWSER_LOG_LEVEL="debug"
export PRSM_BROWSER_DATA_DIR="./dev_data"

# Create development data directory
mkdir -p ./dev_data

# Start PRSM Browser
echo "🌐 Launching PRSM Browser..."
echo "📊 Development mode: ON"
echo "🔍 Debug logging: ON"
echo "📁 Data directory: ./dev_data"
echo ""

../build/prsm_browser --dev-mode --log-level=debug

echo ""
echo "👋 PRSM Browser development session ended"