#!/bin/bash

# NWTN Fixed Pipeline Test Runner
# ==============================
# Test the fundamentally fixed NWTN pipeline in background

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROGRESS_FILE="nwtn_fixed_pipeline_progress_${TIMESTAMP}.txt"
LOG_FILE="nwtn_fixed_pipeline_test_${TIMESTAMP}.log"
PID_FILE="nwtn_fixed_pipeline_${TIMESTAMP}.pid"

echo "🔧 NWTN Fixed Pipeline Test Runner" | tee "$PROGRESS_FILE"
echo "==================================" | tee -a "$PROGRESS_FILE"
echo "Started: $(date)" | tee -a "$PROGRESS_FILE"
echo "Bug Fixed: estimated_time None value in progress visualizer" | tee -a "$PROGRESS_FILE"
echo "Expected: Complete synthesis extraction with NO errors" | tee -a "$PROGRESS_FILE"
echo "Process ID: $$" | tee -a "$PROGRESS_FILE"
echo "" | tee -a "$PROGRESS_FILE"

# Start caffeinate to prevent system sleep
echo "[$(date)] 🔋 Starting caffeinate to prevent system sleep..." | tee -a "$PROGRESS_FILE"
caffeinate -i -s &
CAFFEINATE_PID=$!

# Launch the fixed pipeline test
echo "[$(date)] 🔧 Launching FIXED NWTN pipeline test..." | tee -a "$PROGRESS_FILE"
nohup python3 test_nwtn_final_fixed.py > "$LOG_FILE" 2>&1 &
TEST_PID=$!

echo "Process ID: $TEST_PID" | tee -a "$PROGRESS_FILE"
echo "[$(date)] 📋 Process ID $TEST_PID saved to $PID_FILE" | tee -a "$PROGRESS_FILE"
echo "$TEST_PID" > "$PID_FILE"

# Wait a few seconds and confirm the process started
sleep 5
if ps -p $TEST_PID > /dev/null 2>&1; then
    echo "[$(date)] ✅ Process $TEST_PID confirmed running successfully" | tee -a "$PROGRESS_FILE"
else
    echo "[$(date)] ❌ Process $TEST_PID failed to start" | tee -a "$PROGRESS_FILE"
    kill $CAFFEINATE_PID 2>/dev/null
    exit 1
fi

echo "" | tee -a "$PROGRESS_FILE"
echo "📊 Monitoring Commands:" | tee -a "$PROGRESS_FILE"
echo "   - Check progress: cat $PROGRESS_FILE" | tee -a "$PROGRESS_FILE"
echo "   - Monitor live log: tail -f $LOG_FILE" | tee -a "$PROGRESS_FILE"
echo "   - Check if running: ps -p $TEST_PID" | tee -a "$PROGRESS_FILE"
echo "   - Kill process: kill $TEST_PID" | tee -a "$PROGRESS_FILE"
echo "" | tee -a "$PROGRESS_FILE"

# Wait for the process to complete
wait $TEST_PID
EXIT_CODE=$?

# Stop caffeinate
kill $CAFFEINATE_PID 2>/dev/null

echo "[$(date)] 🏁 Fixed pipeline test completed with exit code: $EXIT_CODE" | tee -a "$PROGRESS_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] 🎉 PIPELINE FULLY FUNCTIONAL - No errors!" | tee -a "$PROGRESS_FILE"
    echo "[$(date)] 📄 Check results in nwtn_fixed_complete_results_*.json" | tee -a "$PROGRESS_FILE"
else
    echo "[$(date)] ❌ Test encountered issues - check $LOG_FILE" | tee -a "$PROGRESS_FILE"
fi

echo "[$(date)] 📊 Final summary:" | tee -a "$PROGRESS_FILE"
echo "   - Full log: $LOG_FILE" | tee -a "$PROGRESS_FILE"
echo "   - Progress: $PROGRESS_FILE" | tee -a "$PROGRESS_FILE"
echo "   - Results: nwtn_fixed_complete_results_*.json" | tee -a "$PROGRESS_FILE"
echo "   - Status: FUNDAMENTAL BUG FIX TESTED" | tee -a "$PROGRESS_FILE"