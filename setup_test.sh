#!/bin/bash

# Simple startup script to test backend services
# This checks dependencies and starts Flask API and Orchestrator

set -e

PROJECT_ROOT="/home/justin/Projects/HackUTD2025"
cd "$PROJECT_ROOT"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  DataFoundry Backend Testing Setup"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  ✓ Python $python_version"

# Check/create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
echo "  ✓ Virtual environment activated"

# Check/install dependencies
echo ""
echo "Checking dependencies..."
if ! python -c "import flask" 2>/dev/null; then
    echo "  Installing backend dependencies..."
    pip install -q -r backend/requirements.txt
    echo "  ✓ Dependencies installed"
else
    echo "  ✓ Dependencies already installed"
fi

# Check for .env file
echo ""
echo "Checking environment configuration..."
if [ ! -f ".env" ]; then
    echo "  ⚠️  WARNING: .env file not found!"
    echo "     The orchestrator requires NVIDIA_API_KEY"
    echo "     Create .env file with:"
    echo "       NVIDIA_API_KEY=nvapi-your-key-here"
    echo ""
    read -p "Do you want to create .env file now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your NVIDIA API key: " api_key
        echo "NVIDIA_API_KEY=$api_key" > .env
        echo "  ✓ .env file created"
    else
        echo "  ⚠️  Continuing without .env (orchestrator may fail)"
    fi
else
    echo "  ✓ .env file exists"
fi

# Check test data
echo ""
echo "Checking test data..."
if [ ! -f "backend/testing_data/adult.csv" ]; then
    echo "  ⚠️  WARNING: Test data not found at backend/testing_data/adult.csv"
    echo "     Upload a CSV file to test with"
else
    test_rows=$(wc -l < "backend/testing_data/adult.csv")
    echo "  ✓ Test data found ($test_rows rows)"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Setup Complete!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Now open 3 terminals and run:"
echo ""
echo "Terminal 1 (Flask API):"
echo "  cd $PROJECT_ROOT"
echo "  source venv/bin/activate"
echo "  python backend/flask_api.py"
echo ""
echo "Terminal 2 (Orchestrator):"
echo "  cd $PROJECT_ROOT"
echo "  source venv/bin/activate"
echo "  python backend/agents/serve_orchestrator.py"
echo ""
echo "Terminal 3 (Test Script):"
echo "  cd $PROJECT_ROOT"
echo "  source venv/bin/activate"
echo "  python backend/test_agent_frontend.py"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
