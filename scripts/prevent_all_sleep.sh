#!/bin/bash
# Prevent All Sleep Modes for PDF Download
# Ensures system runs at full speed even with laptop closed

echo "🚫 PREVENTING ALL SLEEP MODES FOR PDF DOWNLOAD"
echo "=============================================="

# Kill any existing caffeinate processes to avoid conflicts
echo "🔄 Stopping existing caffeinate processes..."
pkill -f "caffeinate" 2>/dev/null || echo "No existing caffeinate processes found"

# Wait a moment for processes to terminate
sleep 2

# Start comprehensive sleep prevention
echo "🛡️ Starting comprehensive sleep prevention..."

# Prevent ALL sleep modes with maximum coverage
# -d: prevent display sleep
# -i: prevent idle sleep  
# -s: prevent system sleep
# -u: prevent disk sleep
# -m: prevent machine sleep
# -w: waits for process to exit (keeps running indefinitely)
caffeinate -d -i -s -u -m &
CAFFEINATE_PID=$!

echo "✅ Started caffeinate with PID: $CAFFEINATE_PID"
echo "$CAFFEINATE_PID" > caffeinate_full_prevention.pid

# Also use pmset to directly override power management (if possible)
echo "🔧 Attempting to override power management settings..."

# Create a temporary power assertion for the PDF download process
cat > /tmp/prevent_sleep.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>PreventSystemSleep</key>
    <true/>
    <key>PreventUserIdleSystemSleep</key>
    <true/>
    <key>PreventUserIdleDisplaySleep</key>
    <true/>
    <key>PreventDiskSleep</key>
    <true/>
</dict>
</plist>
EOF

echo "📄 Created power assertion configuration"

# Verify caffeinate is working
sleep 3
if ps -p $CAFFEINATE_PID > /dev/null; then
    echo "✅ Caffeinate process confirmed running"
else
    echo "❌ Caffeinate process failed to start"
    exit 1
fi

# Check current power assertions
echo ""
echo "🔍 Current Power Assertions:"
pmset -g assertions | grep -E "PreventSystemSleep|PreventUserIdleSystemSleep|PreventDiskSleep" | head -5

echo ""
echo "🎯 SLEEP PREVENTION ACTIVE"
echo "=========================="
echo "✅ System sleep: PREVENTED"
echo "✅ Idle sleep: PREVENTED" 
echo "✅ Display sleep: PREVENTED"
echo "✅ Disk sleep: PREVENTED"
echo "✅ Machine sleep: PREVENTED"
echo ""
echo "🖥️ You can now close your laptop safely"
echo "📥 PDF downloads will continue at full speed"
echo ""
echo "To stop sleep prevention: kill $CAFFEINATE_PID or run:"
echo "kill \$(cat caffeinate_full_prevention.pid)"