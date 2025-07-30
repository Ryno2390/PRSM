#!/bin/bash
# Keep External Drive Awake Script
# Prevents external drive from sleeping during PDF downloads

DRIVE_PATH="/Volumes/My Passport/PRSM_Storage"
KEEP_ALIVE_FILE="$DRIVE_PATH/.keep_alive"
LOG_FILE="drive_keep_alive.log"

echo "🔄 Starting external drive keep-alive process..." | tee -a "$LOG_FILE"
echo "📁 Monitoring: $DRIVE_PATH" | tee -a "$LOG_FILE"
echo "⏰ Started at: $(date)" | tee -a "$LOG_FILE"

# Function to keep drive awake
keep_drive_awake() {
    while true; do
        if [ -d "$DRIVE_PATH" ]; then
            # Touch a small file to create disk activity
            echo "$(date): Drive activity" > "$KEEP_ALIVE_FILE"
            
            # Also check database accessibility 
            if sqlite3 "$DRIVE_PATH/storage.db" "SELECT 1;" >/dev/null 2>&1; then
                echo "$(date '+%H:%M:%S'): ✅ Drive and database accessible" >> "$LOG_FILE"
            else
                echo "$(date '+%H:%M:%S'): ❌ Database not accessible!" >> "$LOG_FILE"
            fi
        else
            echo "$(date '+%H:%M:%S'): ❌ Drive not mounted!" >> "$LOG_FILE"
        fi
        
        # Wait 5 minutes between touches (well under 10-minute sleep timeout)
        sleep 300
    done
}

# Start the keep-alive process in background
keep_drive_awake &
KEEP_ALIVE_PID=$!

echo "🚀 Keep-alive process started with PID: $KEEP_ALIVE_PID" | tee -a "$LOG_FILE"
echo "$KEEP_ALIVE_PID" > drive_keep_alive.pid

# Keep this script running
wait $KEEP_ALIVE_PID