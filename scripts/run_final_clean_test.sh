#!/bin/bash

# NWTN Final Clean Test Background Runner
# ======================================
# Run the bulletproof NWTN test with complete error handling

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROGRESS_FILE="nwtn_final_clean_progress_${TIMESTAMP}.txt"
LOG_FILE="nwtn_final_clean_test_${TIMESTAMP}.log"
PID_FILE="nwtn_final_clean_${TIMESTAMP}.pid"

echo "🧠 NWTN Final Clean Test Runner" | tee "$PROGRESS_FILE"
echo "===============================" | tee -a "$PROGRESS_FILE"
echo "Started: $(date)" | tee -a "$PROGRESS_FILE"
echo "Test: Quantum Gravity Unification (BULLETPROOF VERSION)" | tee -a "$PROGRESS_FILE"
echo "Corpus: 116,051 NWTN-ready papers" | tee -a "$PROGRESS_FILE"
echo "Error Handling: Complete with bulletproof timing" | tee -a "$PROGRESS_FILE"
echo "Process ID: $$" | tee -a "$PROGRESS_FILE"
echo "" | tee -a "$PROGRESS_FILE"

# Start caffeinate to prevent system sleep
echo "[$(date)] 🔋 Starting caffeinate to prevent system sleep..." | tee -a "$PROGRESS_FILE"
caffeinate -i -s &
CAFFEINATE_PID=$!

# Launch the bulletproof NWTN test
echo "[$(date)] 🧠 Launching BULLETPROOF NWTN test..." | tee -a "$PROGRESS_FILE"
nohup python3 test_nwtn_final_clean.py > "$LOG_FILE" 2>&1 &
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

echo "[$(date)] 🏁 BULLETPROOF NWTN test completed with exit code: $EXIT_CODE" | tee -a "$PROGRESS_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] ✅ Test completed successfully with NO ERRORS" | tee -a "$PROGRESS_FILE"
    echo "[$(date)] 📄 Check results in nwtn_final_clean_results_*.json" | tee -a "$PROGRESS_FILE"
else
    echo "[$(date)] ❌ Test failed - check $LOG_FILE for details" | tee -a "$PROGRESS_FILE"
fi

echo "[$(date)] 📊 Final summary:" | tee -a "$PROGRESS_FILE"
echo "   - Full log: $LOG_FILE" | tee -a "$PROGRESS_FILE"
echo "   - Progress: $PROGRESS_FILE" | tee -a "$PROGRESS_FILE"
echo "   - Results: nwtn_final_clean_results_*.json" | tee -a "$PROGRESS_FILE"
echo "   - Status: BULLETPROOF TEST COMPLETE" | tee -a "$PROGRESS_FILE"