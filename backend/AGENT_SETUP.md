# Agent Mode Setup Guide

This guide explains how to set up and use the AI Agent/Orchestrator mode in DataFoundry.

## Prerequisites

### 1. Install Dependencies

Make sure all required packages are installed:

```bash
cd /home/reyan/Documents/HackUTDNVIDIA
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Set Up NVIDIA API Key

The agent uses NVIDIA's LLM services. You need an API key:

1. Go to [https://build.nvidia.com/](https://build.nvidia.com/)
2. Sign up/login and generate an API key
3. Create a `.env` file in the project root:

```bash
cd /home/reyan/Documents/HackUTDNVIDIA
nano .env
```

4. Add your API key:

```env
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxx
```

5. Save and exit (Ctrl+X, Y, Enter)

## Running the Services

You need to run **BOTH** services for Agent Mode to work:

### Terminal 1: Flask API (Port 5000)

```bash
cd /home/reyan/Documents/HackUTDNVIDIA/backend
source ../venv/bin/activate
python flask_api.py --port 5000
```

### Terminal 2: Orchestrator (Port 8000)

```bash
cd /home/reyan/Documents/HackUTDNVIDIA/backend
source ../venv/bin/activate
python agents/serve_orchestrator.py
```

## Using Agent Mode

1. Start both services (Flask + Orchestrator)
2. Open the frontend in your browser
3. Upload a dataset on the Upload page
4. Go to the Schema page
5. Click the "Agent Mode" button (top-right of the workflow card)
6. Describe your goal in natural language, for example:
   - "Generate 1000 synthetic samples using weak supervision labeling"
   - "Label the data and train a model with 50 epochs"
   - "Create synthetic data for high-income individuals"
7. Click "Run Agent"
8. The agent will orchestrate the entire workflow automatically

## How It Works

The agent uses:
- **NVIDIA Nemotron Ultra 253B** - A powerful LLM for understanding your goals
- **LangGraph** - For orchestrating multi-step workflows
- **ReAct Pattern** - Reasoning and acting in a loop
- **Custom Tools** - Labeling, training, generation, and anomaly detection

The agent:
1. Parses your natural language goal
2. Plans a sequence of steps
3. Executes tools in the correct order
4. Manages file paths between steps
5. Returns structured results with all output files

### Response Format

The agent always returns a consistent JSON structure:

```json
{
  "output": "Natural language description of what was done",
  "error": "Error message if workflow failed (optional)",
  "file_paths": {
    "labeled_output_path": null,
    "config_path": "/path/to/config.json",
    "model_path": "/path/to/model.pth",
    "preprocessor_path": "/path/to/preprocessor.joblib",
    "synthetic_output_path": "/path/to/synthetic.csv",
    "anomaly_report_path": null
  },
  "steps_completed": ["training", "generation"]
}
```

- **output**: Human-readable description of results
- **error**: Only present if workflow failed
- **file_paths**: All possible output files (null if not created)
- **steps_completed**: List of workflow steps that ran

### Automatic CSV Conversion

CSV files are automatically converted to Parquet format before being passed to the agent, since the agent pipelines require Parquet. The conversion happens transparently in the Flask proxy endpoint.

## Troubleshooting

### "Orchestrator service not available"
- Make sure the orchestrator is running on port 8000
- Check that your NVIDIA_API_KEY is set in `.env`

### "Agent request timed out"
- The workflow is taking longer than 5 minutes
- This is normal for large datasets or many epochs
- The backend is still processing - check the orchestrator logs

### Connection errors
- Ensure both Flask (5000) and Orchestrator (8000) are running
- Check that ports aren't blocked by firewall

## Required Dependencies

All dependencies are in `requirements.txt`:
- `fastapi` - Web framework for orchestrator
- `uvicorn` - ASGI server for FastAPI
- `langchain` - LLM framework
- `langchain-nvidia-ai-endpoints` - NVIDIA LLM integration
- `langchain-core` - Core LangChain functionality
- `langgraph` - Agent graph orchestration
- `pydantic` - Data validation
- `python-dotenv` - Environment variable loading
- `requests` - HTTP client (for Flask → Orchestrator communication)

Plus all the existing DataFoundry dependencies (torch, pandas, snorkel, etc.)

