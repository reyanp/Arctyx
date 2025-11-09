#!/bin/bash

# DataFoundry Startup Script
# This script starts all services: Frontend, Flask API, and Orchestrator

set -e

PROJECT_ROOT="/home/reyan/Documents/HackUTDNVIDIA"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "DataFoundry - Starting All Services"
echo "=================================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: .env file not found!"
    echo "   Agent Mode requires NVIDIA_API_KEY"
    echo "   Create .env file with: NVIDIA_API_KEY=your_key"
    echo ""
fi

# Check/create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install backend dependencies
echo ""
echo "📦 Installing backend dependencies..."
pip install -q -r backend/requirements.txt

# Install frontend dependencies
echo ""
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "=================================================="
echo "Starting Services..."
echo "=================================================="

# Start Flask API (Port 5000)
echo ""
echo "🚀 Starting Flask API on port 5000..."
cd backend
python flask_api.py --port 5000 > ../logs/flask.log 2>&1 &
FLASK_PID=$!
echo "   PID: $FLASK_PID"
cd ..

# Wait a moment for Flask to start
sleep 2

# Start Orchestrator (Port 8000)
echo ""
echo "🤖 Starting Orchestrator on port 8000..."
cd backend
python agents/serve_orchestrator.py > ../logs/orchestrator.log 2>&1 &
ORCHESTRATOR_PID=$!
echo "   PID: $ORCHESTRATOR_PID"
cd ..

# Wait a moment for Orchestrator to start
sleep 2

# Start Frontend (Port 5173)
echo ""
echo "🎨 Starting Frontend on port 8080..."
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   PID: $FRONTEND_PID"
cd ..

# Save PIDs to file for cleanup
mkdir -p logs
echo "$FLASK_PID" > logs/flask.pid
echo "$ORCHESTRATOR_PID" > logs/orchestrator.pid
echo "$FRONTEND_PID" > logs/frontend.pid

echo ""
echo "=================================================="
echo "✅ All Services Started!"
echo "=================================================="
echo ""
echo "Services:"
echo "  🌐 Frontend:     http://localhost:8080"
echo "  🔧 Flask API:    http://localhost:5000"
echo "  🤖 Orchestrator: http://localhost:8000"
echo ""
echo "Logs:"
echo "  Flask API:    tail -f logs/flask.log"
echo "  Orchestrator: tail -f logs/orchestrator.log"
echo "  Frontend:     tail -f logs/frontend.log"
echo ""
echo "To stop all services: ./stop.sh"
echo "=================================================="
echo ""

# Keep script running and show combined logs
echo "📊 Showing combined logs (Ctrl+C to stop viewing, services will keep running):"
echo ""
sleep 1
tail -f logs/*.log

