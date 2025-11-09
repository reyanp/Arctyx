#!/bin/bash

# DataFoundry Stop Script
# Stops all running services

PROJECT_ROOT="/home/reyan/Documents/HackUTDNVIDIA"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "DataFoundry - Stopping All Services"
echo "=================================================="

# Stop services from PID files
if [ -f "logs/flask.pid" ]; then
    FLASK_PID=$(cat logs/flask.pid)
    echo "🛑 Stopping Flask API (PID: $FLASK_PID)..."
    kill $FLASK_PID 2>/dev/null || echo "   Already stopped"
    rm logs/flask.pid
fi

if [ -f "logs/orchestrator.pid" ]; then
    ORCHESTRATOR_PID=$(cat logs/orchestrator.pid)
    echo "🛑 Stopping Orchestrator (PID: $ORCHESTRATOR_PID)..."
    kill $ORCHESTRATOR_PID 2>/dev/null || echo "   Already stopped"
    rm logs/orchestrator.pid
fi

if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    echo "🛑 Stopping Frontend (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID 2>/dev/null || echo "   Already stopped"
    rm logs/frontend.pid
fi

# Also kill any remaining processes on those ports
echo ""
echo "🧹 Cleaning up any remaining processes..."
pkill -f "flask_api.py" 2>/dev/null || true
pkill -f "serve_orchestrator.py" 2>/dev/null || true
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

echo ""
echo "✅ All services stopped!"
echo "=================================================="

