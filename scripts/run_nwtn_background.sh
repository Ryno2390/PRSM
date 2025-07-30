#!/bin/bash

# NWTN Background Test Runner
# ===========================
# Runs the full NWTN Prompt #1 test as a caffeinated background process
# that continues even when the laptop is closed.

echo "🚀 Starting NWTN Background Test Runner"
echo "========================================"
echo "📝 Test: Quantum Gravity Unification (Prompt #1)"
echo "📊 Corpus: 116,051 NWTN-ready papers"
echo "🎯 Modes: CONSERVATIVE vs REVOLUTIONARY"
echo "🔬 Process: 5,040 iterations deep reasoning each"
echo "⏰ Started: $(date)"
echo ""

# Set up environment
# API key should be set in environment or loaded from secure config
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  ANTHROPIC_API_KEY not set. Please set it in your environment."
    echo "   export ANTHROPIC_API_KEY='your-key-here'"
    exit 1
fi

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="nwtn_background_test_${TIMESTAMP}.log"
PROGRESS_FILE="nwtn_background_progress_${TIMESTAMP}.txt"

echo "📁 Log file: ${LOG_FILE}"
echo "📈 Progress file: ${PROGRESS_FILE}"
echo "💡 Use 'tail -f ${LOG_FILE}' to monitor progress"
echo ""

# Create progress tracking
echo "NWTN Background Test Progress" > "${PROGRESS_FILE}"
echo "============================" >> "${PROGRESS_FILE}"
echo "Started: $(date)" >> "${PROGRESS_FILE}"
echo "Test: Quantum Gravity Unification" >> "${PROGRESS_FILE}"
echo "Corpus: 116,051 NWTN-ready papers" >> "${PROGRESS_FILE}"
echo "Process ID: $$" >> "${PROGRESS_FILE}"
echo "" >> "${PROGRESS_FILE}"

# Function to update progress
update_progress() {
    echo "[$(date)] $1" >> "${PROGRESS_FILE}"
    echo "[$(date)] $1"
}

# Start caffeinate to prevent sleep and run the test
update_progress "🔋 Starting caffeinate to prevent system sleep..."
update_progress "🧠 Launching NWTN direct test with full 5,040-iteration reasoning..."

# Run with caffeinate, nohup, and background process
nohup caffeinate -i -m -d -s python3 test_nwtn_direct_prompt_1.py > "${LOG_FILE}" 2>&1 &

# Get the process ID
PID=$!
echo "🎯 Background process started with PID: ${PID}"
echo "Process ID: ${PID}" >> "${PROGRESS_FILE}"

# Save PID to file for easy killing later if needed
echo "${PID}" > "nwtn_background_${TIMESTAMP}.pid"
update_progress "📋 Process ID ${PID} saved to nwtn_background_${TIMESTAMP}.pid"

echo ""
echo "✅ NWTN test is now running in the background!"
echo ""
echo "📊 Monitoring commands:"
echo "   - Monitor progress: tail -f ${LOG_FILE}"
echo "   - Check progress: cat ${PROGRESS_FILE}"
echo "   - Kill process: kill ${PID}"
echo "   - Or use: kill \$(cat nwtn_background_${TIMESTAMP}.pid)"
echo ""
echo "🌙 Your laptop can now be closed - the test will continue running"
echo "⏰ Expected completion: Several hours (depends on reasoning complexity)"
echo ""
echo "🎉 Background test launched successfully!"

# Optional: Wait a few seconds to make sure it started properly
sleep 5
if ps -p $PID > /dev/null; then
    update_progress "✅ Process ${PID} confirmed running successfully"
    echo "✅ Process confirmed running - safe to close laptop"
else
    update_progress "❌ Process ${PID} failed to start - check ${LOG_FILE}"
    echo "❌ Process failed to start - check ${LOG_FILE}"
fi