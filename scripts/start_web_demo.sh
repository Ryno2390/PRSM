#!/bin/bash

# PRSM Web Demo Startup Script
# 
# This script starts the complete PRSM web demonstration including:
# - PostgreSQL + pgvector database
# - FastAPI backend server
# - Integrated UI for investor presentations

set -e  # Exit on any error

echo "🚀 Starting PRSM Web Demo"
echo "========================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "web_demo_server.py" ]; then
    echo "❌ Please run this script from the PRSM root directory"
    exit 1
fi

# Start PostgreSQL + pgvector database
echo "📦 Starting PostgreSQL + pgvector database..."
docker-compose -f docker-compose.vector.yml up -d postgres-vector

# Wait for database to be ready
echo "⏳ Waiting for database to initialize..."
sleep 10

# Check if database is ready
max_attempts=30
attempts=0
while ! docker exec prsm_postgres_vector pg_isready -U postgres -d prsm_vector_dev > /dev/null 2>&1; do
    if [ $attempts -ge $max_attempts ]; then
        echo "❌ Database failed to start within expected time"
        echo "Check database logs: docker logs prsm_postgres_vector"
        exit 1
    fi
    echo "   Still waiting for database... ($((attempts + 1))/$max_attempts)"
    sleep 2
    attempts=$((attempts + 1))
done

echo "✅ Database is ready!"

# Install Python dependencies if needed
if [ ! -f ".venv/bin/activate" ]; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements-web-demo.txt
else
    echo "📦 Using existing Python environment"
fi

# Start the FastAPI server
echo "🌐 Starting PRSM Web Demo Server..."
echo ""
echo "🎯 Demo will be available at:"
echo "   📍 Main UI:      http://localhost:8000"
echo "   📊 API Docs:     http://localhost:8000/docs"
echo "   ❤️  Health Check: http://localhost:8000/health"
echo ""
echo "🔧 For real AI embeddings, set environment variables:"
echo "   export OPENAI_API_KEY=your_openai_key"
echo "   export ANTHROPIC_API_KEY=your_anthropic_key"
echo ""
echo "Press Ctrl+C to stop the demo"
echo "========================="

# Start the server
python web_demo_server.py