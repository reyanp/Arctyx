#!/bin/bash

# DataFoundry Test Runner
# Convenient script to run all tests with proper setup

PROJECT_ROOT="/home/justin/Projects/HackUTD2025"
cd "$PROJECT_ROOT"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    DataFoundry Backend Test Runner                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found. Run ./setup_test.sh first"
    exit 1
fi

# Check if services are running
echo ""
echo "Checking services..."

flask_running=false
orch_running=false

if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✓ Flask API is running on port 5000"
    flask_running=true
else
    echo "✗ Flask API is NOT running on port 5000"
fi

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Orchestrator is running on port 8000"
    orch_running=true
else
    echo "✗ Orchestrator is NOT running on port 8000"
fi

if [ "$flask_running" = false ] || [ "$orch_running" = false ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  ⚠️  SERVICES NOT RUNNING                                                      ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Please start the services in separate terminals:"
    echo ""
    
    if [ "$flask_running" = false ]; then
        echo "Terminal 1 (Flask API):"
        echo "  cd $PROJECT_ROOT"
        echo "  source venv/bin/activate"
        echo "  python backend/flask_api.py"
        echo ""
    fi
    
    if [ "$orch_running" = false ]; then
        echo "Terminal 2 (Orchestrator):"
        echo "  cd $PROJECT_ROOT"
        echo "  source venv/bin/activate"
        echo "  python backend/agents/serve_orchestrator.py"
        echo ""
    fi
    
    echo "Then run this script again."
    echo ""
    exit 1
fi

# Run tests
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Running Tests                                          ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""

python backend/test_agent_frontend.py

test_exit_code=$?

echo ""
if [ $test_exit_code -eq 0 ]; then
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  ✓ TESTS COMPLETED SUCCESSFULLY                                               ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Your backend is working correctly! You can now:"
    echo "  1. Test with the frontend"
    echo "  2. Deploy to production"
    echo "  3. Run ./run_all_services.sh to start everything"
else
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  ✗ TESTS FAILED                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Check the error output above for details."
    echo "See QUICK_START_TEST.md for troubleshooting tips."
fi
echo ""
