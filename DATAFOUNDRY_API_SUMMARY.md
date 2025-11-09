# DataFoundry API Summary

## 🎉 What Was Created

A complete Flask REST API for DataFoundry that provides direct access to all data processing tools:

1. **Data Labeling** - Weak supervision with Snorkel
2. **Model Training** - Train generative models (cVAE, VAE-GMM, etc.)
3. **Data Generation** - Generate synthetic data from trained models
4. **Data Cleaning** - Anomaly detection and data quality analysis

## 📁 New Files

### Core API
- `backend/flask_api.py` - Main Flask API server (Port 5000)
- `backend/FLASK_API_README.md` - Complete API documentation with examples
- `backend/QUICKSTART.md` - Quick start guide for both APIs

### Testing & Examples
- `backend/test_flask_api.py` - Quick test script to verify API endpoints
- `backend/example_flask_usage.py` - Complete workflow example

## 🚀 Quick Start

### 1. Start the Flask API Server

```bash
cd /home/justin/Projects/HackUTD2025/backend
source ../.venv/bin/activate
python flask_api.py
```

Server will be available at: `http://localhost:5000`

### 2. Test the API

In another terminal:

```bash
# Quick test
python test_flask_api.py

# Full workflow example
python example_flask_usage.py
```

## 🔌 API Endpoints Overview

### Utility
- `GET  /health` - Health check
- `POST /api/dataset/info` - Get dataset information

### Data Labeling
- `POST /api/labeling/create-labels` - Label data using weak supervision
- `POST /api/labeling/run-pipeline` - Run self-correcting labeling pipeline

### Model Training
- `POST /api/training/create-config` - Generate training configuration
- `POST /api/training/train-model` - Train a model
- `POST /api/training/run-pipeline` - Run self-correcting training pipeline

### Data Generation
- `POST /api/generation/generate-data` - Generate synthetic data
- `POST /api/generation/run-pipeline` - Run generation pipeline

### Data Cleaning / Anomaly Detection
- `POST /api/cleaning/detect-anomalies` - Detect anomalies in data
- `POST /api/cleaning/run-pipeline` - Run anomaly detection pipeline

### File Management
- `GET  /api/files/list` - List available files
- `GET  /api/files/download` - Download a file

## 💡 Usage Examples

### Python

```python
import requests

# Label data
response = requests.post('http://localhost:5000/api/labeling/create-labels', json={
    'data_path': 'testing_data/adult.csv',
    'output_path': 'output_data/labeled.parquet',
    'labeling_functions': [
        {
            'name': 'lf_high_income',
            'code': 'def lf_high_income(x):\\n    return 1 if x["capital-gain"] > 5000 else 0'
        }
    ]
})
print(response.json())
```

### JavaScript/TypeScript (Frontend)

```javascript
// React example
const labelData = async () => {
  const response = await fetch('http://localhost:5000/api/labeling/create-labels', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data_path: 'data.csv',
      output_path: 'labeled.parquet',
      labeling_functions: [...]
    })
  });
  const result = await response.json();
  console.log(result);
};
```

### cURL

```bash
# Health check
curl http://localhost:5000/health

# Label data
curl -X POST http://localhost:5000/api/labeling/create-labels \
  -H "Content-Type: application/json" \
  -d '{"data_path": "testing_data/adult.csv", "output_path": "output_data/labeled.parquet", "labeling_functions": [...]}'
```

## 🏗️ Architecture

```
DataFoundry Backend
│
├── Flask API (Port 5000) ← NEW!
│   ├── Direct tool access
│   ├── RESTful JSON API
│   ├── Synchronous responses
│   └── Full parameter control
│
├── Orchestrator (Port 8000)
│   ├── Natural language interface
│   ├── AI-driven workflows
│   ├── Self-correcting pipelines
│   └── LangGraph agents
│
└── DataFoundry Core Library
    ├── labeler.py - Weak supervision
    ├── trainer.py - Model training
    ├── generator.py - Synthetic data
    └── evaluator.py - Anomaly detection
```

## 🎯 Use Cases

### For Frontend Developers
Use the Flask API to:
- Let users create custom labeling functions
- Train models with user-specified parameters
- Generate synthetic data on demand
- Detect anomalies in uploaded datasets

### For Data Scientists
Use the Flask API to:
- Programmatically control the entire pipeline
- Integrate with Jupyter notebooks
- Batch process multiple datasets
- Fine-tune hyperparameters

### For End Users (via Frontend)
- Upload unlabeled data
- Define labeling rules in a UI
- Train models with simple sliders/dropdowns
- Generate synthetic data with one click
- Clean data by detecting anomalies

## 🔄 Complete Workflow

```
1. Upload Data
   └─> POST /api/dataset/info (validate)

2. Label Data
   └─> POST /api/labeling/create-labels
       (User defines labeling functions in UI)

3. Configure Training
   └─> POST /api/training/create-config
       (User adjusts model parameters)

4. Train Model
   └─> POST /api/training/train-model
       (Show progress to user)

5. Generate Data
   └─> POST /api/generation/generate-data
       (User specifies label & count)

6. Detect Anomalies
   └─> POST /api/cleaning/detect-anomalies
       (User reviews anomaly report)

7. Download Results
   └─> GET /api/files/download?path=...
```

## 🌐 Frontend Integration Tips

### State Management
```javascript
// Example state for a data pipeline
const [pipelineState, setPipelineState] = useState({
  step: 'upload',        // upload, label, train, generate, clean
  dataPath: null,
  labeledPath: null,
  configPath: null,
  modelPath: null,
  syntheticPath: null,
  loading: false,
  error: null
});
```

### Progress Tracking
```javascript
// For long-running operations
const trainModel = async (configPath) => {
  setLoading(true);
  setProgress('Training model... This may take a few minutes');
  
  try {
    const response = await fetch('/api/training/train-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config_path: configPath })
    });
    
    const result = await response.json();
    setModelPath(result.model_path);
    setProgress('Training complete!');
  } catch (error) {
    setError(error.message);
  } finally {
    setLoading(false);
  }
};
```

### Error Handling
```javascript
const handleApiError = (error) => {
  if (error.traceback) {
    console.error('Server error:', error.traceback);
  }
  
  // User-friendly error messages
  const friendlyMessages = {
    'Invalid or missing data_path': 'Please select a valid data file',
    'No valid labeling functions': 'Please define at least one labeling rule',
    'Model file not found': 'Model training may have failed'
  };
  
  return friendlyMessages[error.error] || error.error;
};
```

## 📊 Example UI Components

### Labeling Function Builder
```javascript
const LabelingFunctionBuilder = () => {
  const [functions, setFunctions] = useState([]);
  
  const addFunction = () => {
    setFunctions([...functions, {
      name: '',
      code: 'def my_function(x):\n    return 1 if x["column"] > value else 0'
    }]);
  };
  
  const submitLabeling = async () => {
    const response = await fetch('/api/labeling/create-labels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        data_path: dataPath,
        output_path: 'output/labeled.parquet',
        labeling_functions: functions
      })
    });
    // Handle response...
  };
  
  return (
    <div>
      {functions.map((fn, idx) => (
        <div key={idx}>
          <input 
            value={fn.name} 
            onChange={(e) => updateFunction(idx, 'name', e.target.value)}
            placeholder="Function name"
          />
          <textarea 
            value={fn.code}
            onChange={(e) => updateFunction(idx, 'code', e.target.value)}
          />
        </div>
      ))}
      <button onClick={addFunction}>Add Function</button>
      <button onClick={submitLabeling}>Label Data</button>
    </div>
  );
};
```

### Model Training Configuration
```javascript
const TrainingConfig = () => {
  const [config, setConfig] = useState({
    model_type: 'tabular_cvae',
    latent_dim: 64,
    epochs: 50,
    batch_size: 128,
    learning_rate: 0.001
  });
  
  const createAndTrain = async () => {
    // Create config
    const configResponse = await fetch('/api/training/create-config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        data_path: labeledPath,
        output_dir: 'models/',
        model_type: config.model_type,
        training_params: {
          epochs: config.epochs,
          batch_size: config.batch_size,
          learning_rate: config.learning_rate
        }
      })
    });
    
    const { config_path } = await configResponse.json();
    
    // Train model
    const trainResponse = await fetch('/api/training/train-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config_path })
    });
    
    const result = await trainResponse.json();
    console.log('Model trained:', result.model_path);
  };
  
  return (
    <div>
      <label>Model Type</label>
      <select value={config.model_type} onChange={(e) => setConfig({...config, model_type: e.target.value})}>
        <option value="tabular_cvae">Conditional VAE (Recommended)</option>
        <option value="mixed_data_cvae">Mixed Data cVAE</option>
        <option value="tabular_vae_gmm">VAE with GMM</option>
      </select>
      
      <label>Epochs: {config.epochs}</label>
      <input type="range" min="10" max="200" value={config.epochs} 
             onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})} />
      
      <label>Learning Rate: {config.learning_rate}</label>
      <input type="range" min="0.0001" max="0.01" step="0.0001" value={config.learning_rate}
             onChange={(e) => setConfig({...config, learning_rate: parseFloat(e.target.value)})} />
      
      <button onClick={createAndTrain}>Train Model</button>
    </div>
  );
};
```

## 📝 Notes

- The Flask API runs independently on port 5000
- The Orchestrator (AI agent) runs on port 8000
- Both can run simultaneously
- Flask API is synchronous - consider adding async/websockets for production
- All responses are JSON format
- File paths can be absolute or relative to backend directory
- Parquet format is recommended over CSV for all data files

## 🔒 Production Considerations

For production deployment, consider adding:
- ✅ Authentication/Authorization (JWT, OAuth, etc.)
- ✅ Rate limiting
- ✅ Request validation schemas (Pydantic)
- ✅ Async task queue (Celery, Redis)
- ✅ WebSocket for progress updates
- ✅ File upload endpoints
- ✅ HTTPS/TLS
- ✅ CORS configuration
- ✅ Logging and monitoring
- ✅ Database for job tracking

## 📚 Documentation

1. **Flask API Reference**: `backend/FLASK_API_README.md`
2. **Quick Start Guide**: `backend/QUICKSTART.md`
3. **Orchestrator Guide**: `backend/agents/README.md`
4. **Example Usage**: `backend/example_flask_usage.py`
5. **API Tests**: `backend/test_flask_api.py`

## 🤝 Integration with Orchestrator

The two APIs complement each other:

```python
# Direct control via Flask API
flask_api = "http://localhost:5000"
requests.post(f"{flask_api}/api/labeling/create-labels", json={...})

# Natural language via Orchestrator
orchestrator = "http://localhost:8000"
requests.post(f"{orchestrator}/generate", json={
    "input_message": "Label the adult dataset for income prediction"
})
```

Use Flask API when you want:
- Fine-grained control
- Frontend integration
- Synchronous operations
- Custom parameters

Use Orchestrator when you want:
- Natural language interface
- Automated decision-making
- Self-correcting workflows
- Multi-step orchestration

## ✅ Next Steps

1. Start the Flask API: `python flask_api.py`
2. Test the endpoints: `python test_flask_api.py`
3. Run the example: `python example_flask_usage.py`
4. Integrate with your frontend
5. Customize for your use case

## 📞 Support

- See examples in `backend/example_flask_usage.py`
- Read full docs in `backend/FLASK_API_README.md`
- Check quick start in `backend/QUICKSTART.md`
- Test with `backend/test_flask_api.py`

---

**Created for HackUTD 2025** 🚀

