#!/bin/bash

# PRSM Unified Dashboard Launcher
# ==========================================
# This script handles environment setup and launches both the 
# PRSM API backend and the High-Fidelity Streamlit frontend.

# 1. Setup Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set PYTHONPATH so the 'prsm' module can be found
export PYTHONPATH=$PYTHONPATH:.

# Use the project's virtual environment
VENV_PATH="$SCRIPT_DIR/.venv"
PYTHON_EXE="$VENV_PATH/bin/python3"
STREAMLIT_EXE="$VENV_PATH/bin/streamlit"

if [ ! -f "$PYTHON_EXE" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# 2. Cleanup
echo "🧹 Cleaning up existing processes on ports 8000 and 8501..."
lsof -ti:8000,8501 | xargs kill -9 2>/dev/null

# 3. Launch API Server (Backend)
echo "📡 Starting PRSM API Server (Backend)..."
# We run uvicorn through python -m to ensure the environment is correct
$PYTHON_EXE -m uvicorn prsm.interface.api.main:app --host 127.0.0.1 --port 8000 > logs/api_startup.log 2>&1 &
API_PID=$!

# Give the API a few seconds to initialize
sleep 3

# 4. Launch Streamlit (Frontend)
echo "🎨 Starting PRSM Command Center (Frontend)..."
DASHBOARD_PATH="prsm/interface/dashboard/streamlit_app.py"

# Function to handle shutdown
cleanup() {
    echo -e "
👋 Shutting down PRSM Dashboard..."
    kill $API_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run Streamlit
$STREAMLIT_EXE run "$DASHBOARD_PATH" --server.port 8501 --server.headless false
