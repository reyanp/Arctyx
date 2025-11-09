# DataFoundry Backend - Quick Start Guide

DataFoundry provides two complementary APIs for data processing:

## 🚀 Two Server Options

### Option 1: Flask API (Direct Tool Access) - Port 5000
**Best for:** Frontend integration, fine-grained control, direct access to tools

```bash
cd /home/justin/Projects/HackUTD2025/backend
python flask_api.py
```

Access at: `http://localhost:5000`

**Features:**
- ✅ Data Labeling (weak supervision)
- ✅ Model Training (generative models)
- ✅ Data Generation (synthetic data)
- ✅ Anomaly Detection (data cleaning)
- ✅ RESTful JSON API
- ✅ Synchronous responses
- ✅ Full control over parameters

### Option 2: Orchestrator (AI Agent) - Port 8000
**Best for:** Natural language workflows, automated decision-making

```bash
cd /home/justin/Projects/HackUTD2025/backend/agents
python serve_orchestrator.py
```

Access at: `http://localhost:8000`

**Features:**
- ✅ Natural language interface
- ✅ Self-correcting workflows
- ✅ Automated hyperparameter tuning
- ✅ Multi-step task orchestration
- ✅ LangGraph-based agent system

## 📋 Prerequisites

```bash
# 1. Activate virtual environment
cd /home/justin/Projects/HackUTD2025
source .venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Set up environment variables
# Create .env file in project root with:
# NVIDIA_API_KEY=your_api_key_here
```

## 🎯 Quick Test

### Test Flask API

```bash
# Terminal 1: Start Flask API
python backend/flask_api.py

# Terminal 2: Test it
curl http://localhost:5000/health
```

### Run Complete Example

```bash
# Make sure Flask API is running, then:
python backend/example_flask_usage.py
```

## 📚 Documentation

- **Flask API**: See `backend/FLASK_API_README.md` for complete API reference
- **Orchestrator**: See `backend/agents/README.md` for agent system details
- **DataFoundry Core**: See `backend/DataFoundry/` for library documentation

## 🔄 Typical Workflow

### Using Flask API (Recommended for Frontend)

```python
import requests

BASE_URL = "http://localhost:5000"

# 1. Label data
response = requests.post(f"{BASE_URL}/api/labeling/create-labels", json={
    "data_path": "data.csv",
    "output_path": "labeled.parquet",
    "labeling_functions": [...]
})

# 2. Train model
response = requests.post(f"{BASE_URL}/api/training/create-config", json={
    "data_path": "labeled.parquet",
    "output_dir": "models/"
})
config = response.json()['config_path']

response = requests.post(f"{BASE_URL}/api/training/train-model", json={
    "config_path": config
})

# 3. Generate synthetic data
response = requests.post(f"{BASE_URL}/api/generation/generate-data", json={
    "model_path": "models/model.pth",
    "config_path": config,
    "label": 1.0,
    "num_to_generate": 1000
})

# 4. Detect anomalies
response = requests.post(f"{BASE_URL}/api/cleaning/detect-anomalies", json={
    "config_path": config,
    "model_path": "models/model.pth",
    "preprocessor_path": "models/preprocessor.joblib",
    "data_to_scan_path": "data.csv"
})
```

### Using Orchestrator (Natural Language)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "Label the adult dataset and generate 1000 synthetic high-income samples"}'
```

## 🎨 Frontend Integration

The Flask API is designed for easy frontend integration:

```javascript
// React/Next.js example
const labelData = async (dataPath, labelingFunctions) => {
  const response = await fetch('http://localhost:5000/api/labeling/create-labels', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data_path: dataPath,
      output_path: 'output/labeled.parquet',
      labeling_functions: labelingFunctions
    })
  });
  return response.json();
};

const trainModel = async (configPath) => {
  const response = await fetch('http://localhost:5000/api/training/train-model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_path: configPath })
  });
  return response.json();
};

// Use in component
const handleTraining = async () => {
  setLoading(true);
  const result = await trainModel(configPath);
  setModelPath(result.model_path);
  setLoading(false);
};
```

## 🔧 Development

### Running Tests

```bash
# Test individual components
python -m pytest backend/DataFoundry/test_datafoundry.py
python -m pytest backend/agents/test_agents.py

# Or use the agent tester
cd backend
python -m agents.test_agents --test all
```

### Project Structure

```
backend/
├── flask_api.py              # Flask REST API (Port 5000)
├── FLASK_API_README.md       # Flask API documentation
├── example_flask_usage.py    # Example usage script
│
├── agents/                   # Agent system
│   ├── serve_orchestrator.py  # Orchestrator server (Port 8000)
│   ├── labeling_pipeline/
│   ├── training_pipeline/
│   ├── generation_pipeline/
│   └── anomaly_pipeline/
│
└── DataFoundry/              # Core library
    ├── labeler.py            # Weak supervision
    ├── trainer.py            # Model training
    ├── generator.py          # Synthetic data
    └── evaluator.py          # Anomaly detection
```

## ❓ FAQ

**Q: Which API should I use for my frontend?**  
A: Use the Flask API (port 5000) for direct control and synchronous responses.

**Q: Can I run both servers at the same time?**  
A: Yes! They run on different ports (5000 and 8000) and can be used together.

**Q: How do I handle long-running training jobs?**  
A: The Flask API is synchronous. For production, consider adding Celery or similar task queue.

**Q: Can I upload files via the API?**  
A: Currently, you need to provide file paths. File upload endpoints can be added if needed.

**Q: How do I authenticate requests?**  
A: The current API has no authentication. Add it for production deployment.

## 🐛 Troubleshooting

### Port already in use
```bash
# Find and kill process
lsof -i :5000
kill -9 <PID>

# Or use different port
python flask_api.py --port 5001
```

### Import errors
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Reinstall dependencies
pip install -r backend/requirements.txt
```

### CUDA/GPU issues
```bash
# Check PyTorch GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode if needed
export CUDA_VISIBLE_DEVICES=-1
```

## 📞 Support

- Check the documentation: `backend/FLASK_API_README.md`
- Review examples: `backend/example_flask_usage.py`
- Test components: `python -m agents.test_agents --test all`

