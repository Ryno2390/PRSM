#!/bin/bash

# NWTN Progress Checker
# ====================
# Check the status of the background NWTN test

echo "📊 NWTN Background Test Progress Checker"
echo "========================================"
echo ""

# Find the most recent progress file
PROGRESS_FILE=$(ls -t nwtn_background_progress_*.txt 2>/dev/null | head -n1)
PID_FILE=$(ls -t nwtn_background_*.pid 2>/dev/null | head -n1)
LOG_FILE=$(ls -t nwtn_background_test_*.log 2>/dev/null | head -n1)

if [ -z "$PROGRESS_FILE" ]; then
    echo "❌ No NWTN background test found"
    echo "💡 Run ./run_nwtn_background.sh to start a test"
    exit 1
fi

echo "📁 Files found:"
echo "   - Progress: $PROGRESS_FILE"
echo "   - Log: $LOG_FILE"  
echo "   - PID: $PID_FILE"
echo ""

# Check if process is still running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Process $PID is running"
        
        # Show CPU and memory usage
        echo "📈 Resource usage:"
        ps -p $PID -o pid,ppid,%cpu,%mem,etime,command
        echo ""
    else
        echo "❌ Process $PID is not running (may have completed or failed)"
    fi
else
    echo "⚠️  No PID file found"
fi

# Show recent progress
echo "📋 Recent Progress:"
echo "=================="
if [ -f "$PROGRESS_FILE" ]; then
    tail -n 10 "$PROGRESS_FILE"
else
    echo "No progress file found"
fi

echo ""

# Show recent log entries
echo "📝 Recent Log Entries:"
echo "====================="
if [ -f "$LOG_FILE" ]; then
    tail -n 20 "$LOG_FILE"
else
    echo "No log file found"
fi

echo ""
echo "💡 Commands:"
echo "   - Live log monitoring: tail -f $LOG_FILE"
echo "   - Kill process: kill \$(cat $PID_FILE)"
echo "   - Full progress: cat $PROGRESS_FILE"