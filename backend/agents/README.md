# DataFoundry Orchestrator

A hierarchical, multi-agent AI system for automated data labeling, model training, synthetic data generation, and anomaly detection.

## Quick Start

### 1. Start the Orchestrator Server

```bash
cd /home/justin/Projects/HackUTD2025/backend/agents
source ../../.venv/bin/activate
python serve_orchestrator.py
```

Server will run on `http://localhost:8000`

### 2. Run Tests

```bash
cd /home/justin/Projects/HackUTD2025/backend

# Test individual pipelines
python -m agents.test_agents --test labeling
python -m agents.test_agents --test training
python -m agents.test_agents --test generation
python -m agents.test_agents --test anomaly

# Test orchestrator (requires server running)
python -m agents.test_agents --test orchestrator-single
python -m agents.test_agents --test orchestrator-multi

# Run all tests
python -m agents.test_agents --test all
```

## Architecture

- **serve_orchestrator.py** - Main orchestrator server (LangGraph + FastAPI)
- **config.py** - Shared NVIDIA NIM clients
- **labeling_pipeline/** - Self-correcting weak supervision pipeline
- **training_pipeline/** - Self-correcting model training pipeline
- **generation_pipeline/** - Synthetic data generation
- **anomaly_pipeline/** - Anomaly detection
- **test_agents.py** - Test suite

## Environment

Requires `NVIDIA_API_KEY` in `.env` file at project root.

## Usage

Send POST request to `/generate` endpoint:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "Label this dataset: /path/to/data.parquet"}'
```

Or use the Python test suite for examples.

