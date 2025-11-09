#!/bin/bash

echo "=============================================="
echo "Testing Agent CSV Conversion with Verbose Logging"
echo "=============================================="
echo ""
echo "This script will help identify the CSV to Parquet conversion issue."
echo ""
echo "Instructions:"
echo "1. Make sure both services are running:"
echo "   - Flask API on port 5000"
echo "   - Orchestrator on port 8000"
echo "2. Check logs/flask.log and logs/orchestrator.log for detailed output"
echo ""
echo "Starting test in 3 seconds..."
sleep 3

# Test with the adult.csv dataset
echo ""
echo "Testing agent with adult.csv..."
echo ""

curl -X POST http://localhost:5000/api/agent/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_message": "Generate 100 synthetic samples",
    "dataset_path": "backend/testing_data/adult.csv"
  }' \
  -w "\n\nHTTP Status: %{http_code}\n"

echo ""
echo "=============================================="
echo "Test complete. Check the logs for details:"
echo "  - Flask logs: logs/flask.log"
echo "  - Orchestrator logs: logs/orchestrator.log"
echo "=============================================="
