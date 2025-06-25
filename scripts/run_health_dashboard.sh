#!/bin/bash

# PRSM System Health Dashboard Launcher
# Provides easy access to different health monitoring modes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

echo "🏥 PRSM System Health Dashboard"
echo "=============================="

# Check if psutil is available
if ! python3 -c "import psutil" 2>/dev/null; then
    echo "📦 Installing required dependency: psutil"
    pip3 install psutil
fi

# Parse command line arguments
MODE="single"
INTERVAL=60
PORT=8080
OUTPUT="health_report.md"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --help|-h)
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE       Health monitoring mode: single, continuous, server (default: single)"
            echo "  --interval SEC    Monitoring interval in seconds (default: 60)"
            echo "  --port PORT       Web server port (default: 8080)"
            echo "  --output FILE     Output file for single mode (default: health_report.md)"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Single health check"
            echo "  $0 --mode continuous --interval 30   # Continuous monitoring every 30s"
            echo "  $0 --mode server --port 9090         # Web dashboard on port 9090"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Mode: $MODE"
if [[ "$MODE" == "continuous" ]]; then
    echo "Interval: ${INTERVAL}s"
elif [[ "$MODE" == "server" ]]; then
    echo "Port: $PORT"
    echo "Interval: ${INTERVAL}s"
elif [[ "$MODE" == "single" ]]; then
    echo "Output: $OUTPUT"
fi
echo ""

# Run the health dashboard
python3 scripts/system_health_dashboard.py \
    --mode "$MODE" \
    --interval "$INTERVAL" \
    --port "$PORT" \
    --output "$OUTPUT"