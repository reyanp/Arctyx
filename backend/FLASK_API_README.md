# DataFoundry Flask API

A REST API that provides direct access to all DataFoundry tools for data labeling, model training, synthetic data generation, and data cleaning/anomaly detection.

## Quick Start

### 1. Install Dependencies

Make sure you have all required packages installed:

```bash
cd /home/justin/Projects/HackUTD2025
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Start the API Server

```bash
cd /home/justin/Projects/HackUTD2025/backend
python flask_api.py --port 5000
```

The server will start on `http://localhost:5000`

### 3. Test the API

```bash
# Health check
curl http://localhost:5000/health
```

## API Endpoints

### Utility Endpoints

#### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "features": {
    "data_labeling": true,
    "model_training": true,
    "data_generation": true,
    "anomaly_detection": true,
    "agent_pipelines": true
  }
}
```

#### Get Dataset Info
```bash
POST /api/dataset/info
Content-Type: application/json

{
  "data_path": "path/to/dataset.parquet"
}
```

---

## Data Labeling

### 1. Create Labels (Direct Method)

Apply weak supervision to label a dataset using Snorkel labeling functions.

```bash
POST /api/labeling/create-labels
Content-Type: application/json

{
  "data_path": "path/to/unlabeled.csv",
  "output_path": "path/to/labeled.parquet",
  "labeling_functions": [
    {
      "name": "lf_high_income",
      "code": "def lf_high_income(x):\n    return 1 if x['capital-gain'] > 5000 else 0"
    },
    {
      "name": "lf_educated",
      "code": "def lf_educated(x):\n    return 1 if x['education-num'] >= 13 else 0"
    }
  ]
}
```

**Python Example:**
```python
import requests

url = "http://localhost:5000/api/labeling/create-labels"
data = {
    "data_path": "backend/testing_data/adult.csv",
    "output_path": "backend/output_data/adult_labeled.parquet",
    "labeling_functions": [
        {
            "name": "lf_high_capital_gain",
            "code": """def lf_high_capital_gain(x):
    return 1 if x.get('capital-gain', 0) > 5000 else 0"""
        },
        {
            "name": "lf_educated",
            "code": """def lf_educated(x):
    return 1 if x.get('education-num', 0) >= 13 else 0"""
        }
    ]
}

response = requests.post(url, json=data)
print(response.json())
```

### 2. Run Labeling Pipeline (Agent Method)

Use the self-correcting AI agent pipeline for automated labeling with quality control.

```bash
POST /api/labeling/run-pipeline
Content-Type: application/json

{
  "user_goal": "Label whether income is >50K based on census features",
  "raw_data_path": "path/to/unlabeled.parquet",
  "hand_labeled_examples_path": "path/to/ground_truth.parquet",
  "target_auc_score": 0.85,
  "max_attempts": 3
}
```

---

## Model Training

### 1. Create Training Config

Generate a training configuration file with sensible defaults.

```bash
POST /api/training/create-config
Content-Type: application/json

{
  "data_path": "path/to/labeled_data.parquet",
  "output_dir": "path/to/output/",
  "model_type": "tabular_cvae",
  "model_params": {
    "latent_dim": 64,
    "encoder_hidden_layers": [128, 64],
    "decoder_hidden_layers": [64, 128]
  },
  "training_params": {
    "batch_size": 128,
    "learning_rate": 0.001,
    "epochs": 50
  }
}
```

**Model Types:**
- `tabular_cvae` - Conditional Variational Autoencoder (recommended for most cases)
- `mixed_data_cvae` - For mixed numerical/categorical data
- `tabular_vae_gmm` - VAE with Gaussian Mixture Model
- `tabular_ctgan` - Conditional Tabular GAN

**Python Example:**
```python
import requests

url = "http://localhost:5000/api/training/create-config"
data = {
    "data_path": "backend/output_data/adult_labeled.parquet",
    "output_dir": "backend/output_data/models/",
    "model_type": "tabular_cvae",
    "training_params": {
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 0.001
    }
}

response = requests.post(url, json=data)
config_info = response.json()
print(f"Config created: {config_info['config_path']}")
```

### 2. Train Model

Train a generative model using a configuration file.

```bash
POST /api/training/train-model
Content-Type: application/json

{
  "config_path": "path/to/config.json"
}
```

**Python Example:**
```python
import requests

url = "http://localhost:5000/api/training/train-model"
data = {
    "config_path": "backend/output_data/models/config.json"
}

response = requests.post(url, json=data)
training_result = response.json()
print(f"Model trained: {training_result['model_path']}")
```

### 3. Run Training Pipeline (Agent Method)

Use the self-correcting training pipeline for automated hyperparameter tuning.

```bash
POST /api/training/run-pipeline
Content-Type: application/json

{
  "labeled_data_path": "path/to/labeled_data.parquet",
  "holdout_test_path": "path/to/test_data.parquet",
  "target_utility_pct": 0.85,
  "max_attempts": 3
}
```

---

## Data Generation

### Generate Synthetic Data

Generate synthetic samples from a trained model.

```bash
POST /api/generation/generate-data
Content-Type: application/json

{
  "model_path": "path/to/model.pth",
  "config_path": "path/to/config.json",
  "label": 1.0,
  "num_to_generate": 1000,
  "output_path": "path/to/synthetic_data.parquet",
  "output_format": "parquet"
}
```

**Parameters:**
- `label`: The condition to generate for (e.g., 1.0 for high income, 0.0 for low income)
- `num_to_generate`: Number of synthetic samples to create
- `output_format`: `"parquet"` (recommended), `"csv"`, or `"pt"` (PyTorch tensors)

**Python Example:**
```python
import requests

url = "http://localhost:5000/api/generation/generate-data"

# Generate 1000 high-income samples
data = {
    "model_path": "backend/output_data/models/model.pth",
    "config_path": "backend/output_data/models/config.json",
    "label": 1.0,
    "num_to_generate": 1000,
    "output_path": "backend/output_data/synthetic_high_income.parquet",
    "output_format": "parquet"
}

response = requests.post(url, json=data)
print(response.json())

# Generate 1000 low-income samples
data["label"] = 0.0
data["output_path"] = "backend/output_data/synthetic_low_income.parquet"
response = requests.post(url, json=data)
print(response.json())
```

---

## Data Cleaning / Anomaly Detection

### Detect Anomalies

Find anomalies in a dataset using reconstruction error from a trained model.

```bash
POST /api/cleaning/detect-anomalies
Content-Type: application/json

{
  "config_path": "path/to/config.json",
  "model_path": "path/to/model.pth",
  "preprocessor_path": "path/to/preprocessor.joblib",
  "data_to_scan_path": "path/to/data.parquet",
  "output_path": "path/to/anomaly_report.parquet"
}
```

**Response includes:**
- `num_samples_scanned`: Number of samples analyzed
- `anomaly_score_range`: [min, max] anomaly scores
- `mean_anomaly_score`: Average anomaly score
- `output_path`: Path to the detailed report (sorted by anomaly score, highest first)

**Python Example:**
```python
import requests
import pandas as pd

url = "http://localhost:5000/api/cleaning/detect-anomalies"
data = {
    "config_path": "backend/output_data/models/config.json",
    "model_path": "backend/output_data/models/model.pth",
    "preprocessor_path": "backend/output_data/models/preprocessor.joblib",
    "data_to_scan_path": "backend/testing_data/adult.csv",
    "output_path": "backend/output_data/anomalies.parquet"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Scanned {result['num_samples_scanned']} samples")
print(f"Anomaly scores: {result['anomaly_score_range']}")

# Load detailed report
df = pd.read_parquet(result['output_path'])
print("\nTop 10 anomalies:")
print(df.head(10))
```

---

## File Management

### List Files
```bash
GET /api/files/list
```

### Download File
```bash
GET /api/files/download?path=path/to/file
```

---

## Complete Workflow Example

Here's a complete Python script demonstrating the full DataFoundry workflow:

```python
import requests
import time

BASE_URL = "http://localhost:5000"

# 1. LABEL DATA
print("Step 1: Labeling data...")
response = requests.post(f"{BASE_URL}/api/labeling/create-labels", json={
    "data_path": "backend/testing_data/adult.csv",
    "output_path": "backend/output_data/adult_labeled.parquet",
    "labeling_functions": [
        {
            "name": "lf_high_capital",
            "code": "def lf_high_capital(x):\n    return 1 if x.get('capital-gain', 0) > 5000 else 0"
        },
        {
            "name": "lf_educated",
            "code": "def lf_educated(x):\n    return 1 if x.get('education-num', 0) >= 13 else 0"
        }
    ]
})
labeled_path = response.json()['output_path']
print(f"✓ Labeled data saved to: {labeled_path}")

# 2. CREATE TRAINING CONFIG
print("\nStep 2: Creating training config...")
response = requests.post(f"{BASE_URL}/api/training/create-config", json={
    "data_path": labeled_path,
    "output_dir": "backend/output_data/models/",
    "model_type": "tabular_cvae",
    "training_params": {
        "epochs": 30,
        "batch_size": 128,
        "learning_rate": 0.001
    }
})
config_path = response.json()['config_path']
print(f"✓ Config created: {config_path}")

# 3. TRAIN MODEL
print("\nStep 3: Training model (this may take a few minutes)...")
response = requests.post(f"{BASE_URL}/api/training/train-model", json={
    "config_path": config_path
})
model_info = response.json()
model_path = model_info['model_path']
preprocessor_path = model_info['preprocessor_path']
print(f"✓ Model trained: {model_path}")

# 4. GENERATE SYNTHETIC DATA
print("\nStep 4: Generating synthetic data...")
response = requests.post(f"{BASE_URL}/api/generation/generate-data", json={
    "model_path": model_path,
    "config_path": config_path,
    "label": 1.0,
    "num_to_generate": 1000,
    "output_path": "backend/output_data/synthetic_data.parquet",
    "output_format": "parquet"
})
synthetic_path = response.json()['output_path']
print(f"✓ Generated 1000 synthetic samples: {synthetic_path}")

# 5. DETECT ANOMALIES
print("\nStep 5: Detecting anomalies...")
response = requests.post(f"{BASE_URL}/api/cleaning/detect-anomalies", json={
    "config_path": config_path,
    "model_path": model_path,
    "preprocessor_path": preprocessor_path,
    "data_to_scan_path": "backend/testing_data/adult.csv",
    "output_path": "backend/output_data/anomalies.parquet"
})
anomaly_info = response.json()
print(f"✓ Scanned {anomaly_info['num_samples_scanned']} samples")
print(f"  Anomaly score range: {anomaly_info['anomaly_score_range']}")
print(f"  Report saved to: {anomaly_info['output_path']}")

print("\n✅ Complete workflow finished successfully!")
```

---

## Integration with Orchestrator

The Flask API runs independently from the LangGraph orchestrator (port 8000). You can:

1. **Direct Control**: Use the Flask API endpoints directly for fine-grained control
2. **Agent Automation**: Call the orchestrator at `http://localhost:8000/generate` for natural language workflows
3. **Hybrid Approach**: Use Flask API for specific steps, orchestrator for complex multi-step tasks

### Example: Using Both APIs

```python
import requests

# Option 1: Direct control via Flask API (port 5000)
flask_url = "http://localhost:5000/api/labeling/create-labels"
requests.post(flask_url, json={...})

# Option 2: Natural language via Orchestrator (port 8000)
orchestrator_url = "http://localhost:8000/generate"
requests.post(orchestrator_url, json={
    "input_message": "Label the adult dataset and train a model"
})
```

---

## Error Handling

All endpoints return JSON responses with the following structure:

**Success:**
```json
{
  "success": true,
  "output_path": "path/to/output",
  "message": "Operation completed successfully"
}
```

**Error:**
```json
{
  "error": "Error message",
  "traceback": "Detailed stack trace (in debug mode)"
}
```

---

## Command Line Options

```bash
python flask_api.py --help

Options:
  --host HOST      Host to bind to (default: 0.0.0.0)
  --port PORT      Port to bind to (default: 5000)
  --debug          Enable debug mode
```

---

## Architecture

```
Backend Architecture
├── flask_api.py (Port 5000)         ← Direct tool access
│   ├── /api/labeling/*
│   ├── /api/training/*
│   ├── /api/generation/*
│   └── /api/cleaning/*
│
├── agents/serve_orchestrator.py (Port 8000)  ← AI orchestration
│   └── /generate (natural language)
│
└── DataFoundry/                      ← Core library
    ├── labeler.py
    ├── trainer.py
    ├── generator.py
    └── evaluator.py
```

---

## Notes

- The Flask API provides **synchronous** endpoints - long-running operations will block until complete
- For production use, consider adding:
  - Authentication/authorization
  - Rate limiting
  - Async task queue (Celery)
  - File upload endpoints
  - WebSocket for progress updates
- All file paths can be absolute or relative to the backend directory
- Parquet format is recommended for all data files (faster, more robust than CSV)

