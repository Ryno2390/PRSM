#!/bin/bash
# Keep System Awake for PDF Processing
# This script prevents macOS sleep while PDF processing continues

echo "🔋 NWTN PDF Processing - Keep Alive System"
echo "=========================================="
echo "This will prevent your MacBook from sleeping during PDF processing"
echo "Safe to close lid - processing will continue in background"
echo ""

# Get the PDF process PID
PDF_PID=$(ps aux | grep "download_full_pdfs.py" | grep -v grep | awk '{print $2}')

if [ -z "$PDF_PID" ]; then
    echo "❌ PDF download process not found. Please start it first:"
    echo "   python download_full_pdfs.py"
    exit 1
fi

echo "✅ Found PDF process (PID: $PDF_PID)"
echo "🔋 Starting system keep-alive..."

# Prevent system sleep while process runs
caffeinate -i -w $PDF_PID &
CAFFEINATE_PID=$!

echo "✅ System keep-alive active (Caffeinate PID: $CAFFEINATE_PID)"
echo ""
echo "💡 Your MacBook will now stay awake even with the lid closed"
echo "📊 To monitor progress: python monitor_pdf_download.py"
echo "🛑 To stop keep-alive: kill $CAFFEINATE_PID"
echo ""
echo "🎯 Safe to close your MacBook lid now!"
echo "   PDF processing will continue in the background"

# Wait for the process to complete
wait $PDF_PID
echo "🎉 PDF processing completed! System sleep restrictions removed."