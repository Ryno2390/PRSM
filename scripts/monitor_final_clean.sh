#!/bin/bash

# NWTN Final Clean Test Monitor
# ============================
# Monitor the bulletproof NWTN test with complete error handling

echo "📊 NWTN Final Clean Test Monitor"
echo "================================"
echo ""

# Find the most recent final clean test files
PROGRESS_FILE=$(ls -t nwtn_final_clean_progress_*.txt 2>/dev/null | head -n1)
PID_FILE=$(ls -t nwtn_final_clean_*.pid 2>/dev/null | head -n1)
LOG_FILE=$(ls -t nwtn_final_clean_test_*.log 2>/dev/null | head -n1)

if [ -z "$PROGRESS_FILE" ]; then
    echo "❌ No NWTN final clean test found"
    echo "💡 Run ./run_final_clean_test.sh to start a test"
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
        
        # Show what stage we're in based on log content
        if grep -q "🎯 TEST 1: CONSERVATIVE MODE" "$LOG_FILE" 2>/dev/null; then
            if grep -q "✅ CONSERVATIVE processing complete" "$LOG_FILE" 2>/dev/null; then
                if grep -q "🎯 TEST 2: REVOLUTIONARY MODE" "$LOG_FILE" 2>/dev/null; then
                    if grep -q "✅ REVOLUTIONARY processing complete" "$LOG_FILE" 2>/dev/null; then
                        if grep -q "📊 COMPARATIVE ANALYSIS" "$LOG_FILE" 2>/dev/null; then
                            echo "🎯 Status: Finalizing analysis and saving results"
                        else
                            echo "🎯 Status: Processing comparative analysis"
                        fi
                    else
                        echo "🎯 Status: Running REVOLUTIONARY mode (5,040 iterations)"
                    fi
                else
                    echo "🎯 Status: Transitioning to REVOLUTIONARY mode"
                fi
            else
                echo "🎯 Status: Running CONSERVATIVE mode (5,040 iterations)"
            fi
        else
            echo "🎯 Status: Initializing NWTN system and knowledge base"
        fi
        
    else
        echo "❌ Process $PID is not running (completed or failed)"
        
        # Check if we have results
        RESULTS_FILE=$(ls -t nwtn_final_clean_results_*.json 2>/dev/null | head -n1)
        if [ -n "$RESULTS_FILE" ]; then
            echo "🎉 Results file found: $RESULTS_FILE"
            
            # Quick preview of success status
            if grep -q '"success": true' "$RESULTS_FILE" 2>/dev/null; then
                echo "✅ Test appears to have succeeded!"
            fi
        else
            echo "⚠️  No results file found - check log for issues"
        fi
    fi
else
    echo "⚠️  No PID file found"
fi

echo ""
echo "📋 Recent Progress:"
echo "=================="
if [ -f "$PROGRESS_FILE" ]; then
    tail -n 15 "$PROGRESS_FILE"
else
    echo "No progress file found"
fi

echo ""
echo "📝 Recent Log Entries (last 15 lines):"
echo "======================================"
if [ -f "$LOG_FILE" ]; then
    tail -n 15 "$LOG_FILE"
else
    echo "No log file found"
fi

echo ""
echo "💡 Commands:"
echo "   - Live log monitoring: tail -f $LOG_FILE"
echo "   - Kill process: kill \$(cat $PID_FILE)"
echo "   - Full progress: cat $PROGRESS_FILE"
echo "   - Check results: ls -la nwtn_final_clean_results_*.json"
echo "   - View results: cat nwtn_final_clean_results_*.json | jq ."