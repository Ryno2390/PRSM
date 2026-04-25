#!/bin/bash
# PRSM Performance Monitoring Launcher Script

set -e

echo "🚀 PRSM Performance Monitoring Dashboard"
echo "========================================"

# Check if Python and required packages are available
python3 -c "import sqlite3, structlog, psutil" 2>/dev/null || {
    echo "❌ Missing required packages. Installing..."
    pip install structlog psutil
}

# Set up environment
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Parse command line arguments
MODE="${1:-single}"
INTERVAL="${2:-300}"

case "$MODE" in
    "single")
        echo "🎯 Running single performance test..."
        python3 scripts/performance_monitoring_dashboard.py --mode single --output "performance_report_$(date +%Y%m%d_%H%M%S).md"
        ;;
    "continuous")
        echo "🔄 Starting continuous monitoring (interval: ${INTERVAL}s)"
        echo "Press Ctrl+C to stop monitoring"
        python3 scripts/performance_monitoring_dashboard.py --mode continuous --interval "$INTERVAL"
        ;;
    "help")
        echo "Usage: $0 [MODE] [INTERVAL]"
        echo ""
        echo "Modes:"
        echo "  single     - Run a single performance test (default)"
        echo "  continuous - Run continuous monitoring"
        echo "  help       - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 single                    # Run single test"
        echo "  $0 continuous 600           # Monitor every 10 minutes"
        echo "  $0 continuous 1800          # Monitor every 30 minutes"
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac