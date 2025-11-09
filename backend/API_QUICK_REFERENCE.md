# DataFoundry API Quick Reference

## 🚀 Start Server

```bash
cd /home/justin/Projects/HackUTD2025/backend
python flask_api.py
```

Server: `http://localhost:5000`

## 🔍 Test Server

```bash
# Quick test
curl http://localhost:5000/health

# Run full test suite
python test_flask_api.py

# Run example workflow
python example_flask_usage.py
```

## 📋 Endpoints Cheat Sheet

### 1. Data Labeling

```python
import requests

response = requests.post('http://localhost:5000/api/labeling/create-labels', json={
    'data_path': 'testing_data/adult.csv',
    'output_path': 'output_data/labeled.parquet',
    'labeling_functions': [
        {
            'name': 'lf_example',
            'code': 'def lf_example(x):\n    return 1 if x["col"] > 0 else 0'
        }
    ]
})
```

### 2. Create Training Config

```python
response = requests.post('http://localhost:5000/api/training/create-config', json={
    'data_path': 'output_data/labeled.parquet',
    'output_dir': 'output_data/models/',
    'model_type': 'tabular_cvae',  # or mixed_data_cvae, tabular_vae_gmm
    'training_params': {
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 0.001
    }
})
config_path = response.json()['config_path']
```

### 3. Train Model

```python
response = requests.post('http://localhost:5000/api/training/train-model', json={
    'config_path': config_path
})
result = response.json()
model_path = result['model_path']
preprocessor_path = result['preprocessor_path']
```

### 4. Generate Synthetic Data

```python
response = requests.post('http://localhost:5000/api/generation/generate-data', json={
    'model_path': model_path,
    'config_path': config_path,
    'label': 1.0,  # Condition: 1.0 for high income, 0.0 for low income
    'num_to_generate': 1000,
    'output_path': 'output_data/synthetic.parquet',
    'output_format': 'parquet'  # or 'csv', 'pt'
})
```

### 5. Detect Anomalies

```python
response = requests.post('http://localhost:5000/api/cleaning/detect-anomalies', json={
    'config_path': config_path,
    'model_path': model_path,
    'preprocessor_path': preprocessor_path,
    'data_to_scan_path': 'testing_data/adult.csv',
    'output_path': 'output_data/anomalies.parquet'
})
```

## 🎯 Complete Workflow (Copy & Paste)

```python
import requests

BASE_URL = "http://localhost:5000"

# 1. Label
resp1 = requests.post(f"{BASE_URL}/api/labeling/create-labels", json={
    "data_path": "testing_data/adult.csv",
    "output_path": "output_data/labeled.parquet",
    "labeling_functions": [{
        "name": "lf_high_capital",
        "code": "def lf_high_capital(x):\n    return 1 if x.get('capital-gain', 0) > 5000 else 0"
    }]
})
labeled_path = resp1.json()['output_path']

# 2. Create Config
resp2 = requests.post(f"{BASE_URL}/api/training/create-config", json={
    "data_path": labeled_path,
    "output_dir": "output_data/models/"
})
config_path = resp2.json()['config_path']

# 3. Train
resp3 = requests.post(f"{BASE_URL}/api/training/train-model", json={
    "config_path": config_path
})
model_path = resp3.json()['model_path']
preprocessor_path = resp3.json()['preprocessor_path']

# 4. Generate
resp4 = requests.post(f"{BASE_URL}/api/generation/generate-data", json={
    "model_path": model_path,
    "config_path": config_path,
    "label": 1.0,
    "num_to_generate": 1000
})
synthetic_path = resp4.json()['output_path']

# 5. Detect Anomalies
resp5 = requests.post(f"{BASE_URL}/api/cleaning/detect-anomalies", json={
    "config_path": config_path,
    "model_path": model_path,
    "preprocessor_path": preprocessor_path,
    "data_to_scan_path": "testing_data/adult.csv"
})
anomaly_path = resp5.json()['output_path']

print(f"✅ Complete! Results in: {synthetic_path}, {anomaly_path}")
```

## 🌐 JavaScript/Frontend

```javascript
const BASE_URL = 'http://localhost:5000';

// Label data
const labelData = async (dataPath, labelingFunctions) => {
  const res = await fetch(`${BASE_URL}/api/labeling/create-labels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data_path: dataPath,
      output_path: 'output_data/labeled.parquet',
      labeling_functions: labelingFunctions
    })
  });
  return res.json();
};

// Create config
const createConfig = async (dataPath) => {
  const res = await fetch(`${BASE_URL}/api/training/create-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data_path: dataPath,
      output_dir: 'output_data/models/',
      model_type: 'tabular_cvae'
    })
  });
  return res.json();
};

// Train model
const trainModel = async (configPath) => {
  const res = await fetch(`${BASE_URL}/api/training/train-model`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_path: configPath })
  });
  return res.json();
};

// Generate data
const generateData = async (modelPath, configPath, label, count) => {
  const res = await fetch(`${BASE_URL}/api/generation/generate-data`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_path: modelPath,
      config_path: configPath,
      label: label,
      num_to_generate: count,
      output_format: 'parquet'
    })
  });
  return res.json();
};

// Detect anomalies
const detectAnomalies = async (configPath, modelPath, preprocessorPath, dataPath) => {
  const res = await fetch(`${BASE_URL}/api/cleaning/detect-anomalies`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      config_path: configPath,
      model_path: modelPath,
      preprocessor_path: preprocessorPath,
      data_to_scan_path: dataPath
    })
  });
  return res.json();
};
```

## 📦 Response Format

All endpoints return JSON:

**Success:**
```json
{
  "success": true,
  "output_path": "/path/to/file",
  "message": "Operation completed"
}
```

**Error:**
```json
{
  "error": "Error message",
  "traceback": "Stack trace (debug mode)"
}
```

## 🔧 Model Types

- `tabular_cvae` - Conditional VAE (recommended)
- `mixed_data_cvae` - For mixed numerical/categorical
- `tabular_vae_gmm` - VAE with Gaussian Mixture Model
- `tabular_ctgan` - Conditional Tabular GAN

## 📁 File Paths

All paths relative to: `/home/justin/Projects/HackUTD2025/backend/`

**Input:**
- `testing_data/adult.csv` - Sample dataset

**Output:**
- `output_data/` - All generated files
- `output_data/models/` - Model files
- `uploaded_data/` - User uploads

## ⚡ Tips

1. **File Format**: Use Parquet over CSV (faster, more robust)
2. **Paths**: Can be absolute or relative to backend directory
3. **Long Operations**: Training can take minutes - handle async in frontend
4. **Error Handling**: Check `response.status_code` and `error` field
5. **CORS**: Already enabled for cross-origin requests

## 📚 Documentation

- **Full API Docs**: `backend/FLASK_API_README.md`
- **Quick Start**: `backend/QUICKSTART.md`
- **Examples**: `backend/example_flask_usage.py`
- **Tests**: `backend/test_flask_api.py`

## 🎨 Example UI Flow

```
1. User uploads CSV
   └─> Validate with /api/dataset/info

2. User defines labeling rules in UI
   └─> Submit to /api/labeling/create-labels

3. User adjusts training params with sliders
   └─> Create config with /api/training/create-config
   └─> Train with /api/training/train-model

4. User generates synthetic data
   └─> Use /api/generation/generate-data

5. User cleans data
   └─> Scan with /api/cleaning/detect-anomalies
   └─> Show top anomalies in UI

6. User downloads results
   └─> Use /api/files/download
```

## 🐛 Troubleshooting

**Can't connect to API:**
```bash
# Check if running
curl http://localhost:5000/health

# Start server
python backend/flask_api.py

# Use different port
python backend/flask_api.py --port 5001
```

**Import errors:**
```bash
source .venv/bin/activate
pip install -r backend/requirements.txt
```

**Path not found:**
- Use absolute paths: `/home/justin/Projects/HackUTD2025/backend/testing_data/adult.csv`
- Or relative to backend: `testing_data/adult.csv`

## 🌐 Two APIs

| Feature | Flask API (5000) | Orchestrator (8000) |
|---------|-----------------|---------------------|
| Control | Direct/Manual | AI-Driven |
| Interface | REST JSON | Natural Language |
| Speed | Fast | Slower (LLM calls) |
| Use Case | Frontend | Automation |

Both can run together! Use Flask for direct control, Orchestrator for automation.

---

**Quick Start**: `python flask_api.py` → `python example_flask_usage.py` → Done! 🚀

