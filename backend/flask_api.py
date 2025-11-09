"""
DataFoundry Flask API

A REST API that exposes direct access to DataFoundry tools:
- Data Labeling: Weak supervision and automated labeling
- Model Training: Train generative models on labeled data
- Data Generation: Generate synthetic data from trained models
- Data Cleaning: Anomaly detection and data quality analysis

This API allows frontend users to directly tune and control the data pipeline,
or integrate with the orchestrator agent for automated workflows.
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd

# Add DataFoundry to path
backend_dir = Path(__file__).parent
datafoundry_path = backend_dir / "DataFoundry"
agents_path = backend_dir / "agents"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(datafoundry_path))

# Import DataFoundry core modules
import DataFoundry.labeler as labeler
import DataFoundry.trainer as trainer
import DataFoundry.generator as generator
import DataFoundry.evaluator as evaluator

# Import agent pipelines (optional - for self-correcting workflows)
try:
    from agents.labeling_pipeline.labeling_tool import run_labeling_pipeline
    from agents.training_pipeline.training_tool import run_training_pipeline
    from agents.generation_pipeline.generation_tool import run_generation_pipeline
    from agents.anomaly_pipeline.anomaly_tool import run_anomaly_pipeline
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    print("Warning: Agent pipelines not available. Only core DataFoundry functions will be exposed.")

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Configuration
UPLOAD_FOLDER = backend_dir / "uploaded_data"
OUTPUT_FOLDER = backend_dir / "output_data"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['OUTPUT_FOLDER'] = str(OUTPUT_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Check if the API is running and show available features."""
    # Check if orchestrator is running
    orchestrator_available = False
    try:
        import requests
        resp = requests.get('http://localhost:8000/health', timeout=2)
        orchestrator_available = resp.status_code == 200
    except:
        pass
    
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'features': {
            'data_labeling': True,
            'model_training': True,
            'data_generation': True,
            'anomaly_detection': True,
            'agent_pipelines': AGENTS_AVAILABLE,
            'orchestrator': orchestrator_available
        }
    })


@app.route('/api/agent/generate', methods=['POST'])
def agent_generate():
    """
    Proxy endpoint to the orchestrator service.
    Automatically converts CSV files to Parquet format for agent compatibility.
    
    Request body:
    {
        "input_message": "Generate 1000 samples of high-income individuals...",
        "dataset_path": "path/to/dataset.csv"  // optional, auto-converted to Parquet if CSV
    }
    
    Returns:
    {
        "output": "Natural language response from the agent",
        "error": "Error message if workflow failed" // optional
        "file_paths": {
            "labeled_output_path": null,  // or string path if labeling was performed
            "config_path": null,           // or string path if training was performed
            "model_path": null,            // or string path if training was performed
            "preprocessor_path": null,     // or string path if training was performed
            "synthetic_output_path": null, // or string path if generation was performed
            "anomaly_report_path": null    // or string path if anomaly detection was performed
        },
        "steps_completed": ["training", "generation"]  // list of completed steps
    }
    
    All error cases return the same structure with null file paths and an error field.
    """
    try:
        import requests
        
        data = request.get_json()
        input_message = data.get('input_message')
        dataset_path = data.get('dataset_path')
        
        if not input_message:
            return jsonify({'error': 'Missing input_message'}), 400
        
        # Convert CSV to Parquet if needed (agent pipelines require Parquet)
        if dataset_path:
            print("\n" + "="*80)
            print("[AGENT CSV CONVERSION] Starting conversion process")
            print("="*80)
            print(f"[AGENT] Received dataset_path from frontend: {dataset_path}")
            print(f"[AGENT] Path is absolute: {os.path.isabs(dataset_path)}")
            print(f"[AGENT] Backend directory: {backend_dir}")
            
            # Check if file is CSV
            if dataset_path.lower().endswith('.csv'):
                print(f"[AGENT] File extension is .csv, conversion needed")
                
                # Convert CSV to Parquet for agent
                try:
                    print(f"\n[AGENT CONVERSION STEP 1] Resolving path...")
                    
                    # Ensure we have the absolute path
                    original_path = dataset_path
                    if not os.path.isabs(dataset_path):
                        # Strip 'backend/' prefix if present to avoid double backend/backend
                        if dataset_path.startswith('backend/'):
                            dataset_path = dataset_path[len('backend/'):]
                            print(f"[AGENT] Stripped 'backend/' prefix: {original_path} -> {dataset_path}")
                        
                        dataset_path = os.path.join(backend_dir, dataset_path)
                        print(f"[AGENT] Converted relative to absolute: {original_path} -> {dataset_path}")
                    else:
                        print(f"[AGENT] Path is already absolute: {dataset_path}")
                    
                    print(f"\n[AGENT CONVERSION STEP 2] Checking file existence...")
                    print(f"[AGENT] CSV path: {dataset_path}")
                    print(f"[AGENT] File exists: {os.path.exists(dataset_path)}")
                    
                    if not os.path.exists(dataset_path):
                        error_msg = f"CSV file not found at: {dataset_path}"
                        print(f"[AGENT ERROR] {error_msg}")
                        return jsonify({
                            'output': error_msg,
                            'error': error_msg,
                            'file_paths': {
                                'labeled_output_path': None,
                                'config_path': None,
                                'model_path': None,
                                'preprocessor_path': None,
                                'synthetic_output_path': None,
                                'anomaly_report_path': None
                            },
                            'steps_completed': []
                        }), 400
                    
                    print(f"[AGENT] File size: {os.path.getsize(dataset_path)} bytes")
                    
                    print(f"\n[AGENT CONVERSION STEP 3] Reading CSV file...")
                    # Read CSV
                    df = pd.read_csv(dataset_path)
                    print(f"[AGENT] ✓ CSV loaded successfully")
                    print(f"[AGENT]   Rows: {len(df)}")
                    print(f"[AGENT]   Columns: {len(df.columns)}")
                    print(f"[AGENT]   Column names: {df.columns.tolist()}")
                    print(f"[AGENT]   Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                    
                    print(f"\n[AGENT CONVERSION STEP 4] Preparing Parquet path...")
                    # Create parquet path (same directory, add _agent suffix)
                    parquet_path = dataset_path.rsplit('.csv', 1)[0] + '_agent.parquet'
                    print(f"[AGENT] Target Parquet path: {parquet_path}")
                    
                    # Ensure directory exists
                    parquet_dir = os.path.dirname(parquet_path)
                    print(f"[AGENT] Target directory: {parquet_dir}")
                    print(f"[AGENT] Directory exists: {os.path.exists(parquet_dir)}")
                    
                    if not os.path.exists(parquet_dir):
                        os.makedirs(parquet_dir, exist_ok=True)
                        print(f"[AGENT] Created directory: {parquet_dir}")
                    
                    print(f"\n[AGENT CONVERSION STEP 5] Writing Parquet file...")
                    # Save as parquet with explicit engine
                    df.to_parquet(parquet_path, index=False, engine='pyarrow')
                    print(f"[AGENT] ✓ Parquet write command completed")
                    
                    print(f"\n[AGENT CONVERSION STEP 6] Verifying Parquet file...")
                    if os.path.exists(parquet_path):
                        file_size = os.path.getsize(parquet_path)
                        print(f"[AGENT] ✓ Parquet file exists")
                        print(f"[AGENT]   Path: {parquet_path}")
                        print(f"[AGENT]   Size: {file_size} bytes")
                        
                        # Verify it's a valid parquet file by trying to read it
                        print(f"[AGENT] Attempting to read back Parquet file for verification...")
                        try:
                            test_df = pd.read_parquet(parquet_path)
                            print(f"[AGENT] ✓ Parquet file verified successfully")
                            print(f"[AGENT]   Read {len(test_df)} rows, {len(test_df.columns)} columns")
                            print(f"[AGENT]   Column names match: {test_df.columns.tolist() == df.columns.tolist()}")
                        except Exception as verify_error:
                            error_msg = f"Parquet file created but failed verification: {verify_error}"
                            print(f"[AGENT ERROR] {error_msg}")
                            import traceback
                            traceback.print_exc()
                            return jsonify({
                                'output': error_msg,
                                'error': error_msg,
                                'file_paths': {
                                    'labeled_output_path': None,
                                    'config_path': None,
                                    'model_path': None,
                                    'preprocessor_path': None,
                                    'synthetic_output_path': None,
                                    'anomaly_report_path': None
                                },
                                'steps_completed': []
                            }), 400
                    else:
                        error_msg = f"Parquet file not found after write: {parquet_path}"
                        print(f"[AGENT ERROR] {error_msg}")
                        return jsonify({
                            'output': error_msg,
                            'error': error_msg,
                            'file_paths': {
                                'labeled_output_path': None,
                                'config_path': None,
                                'model_path': None,
                                'preprocessor_path': None,
                                'synthetic_output_path': None,
                                'anomaly_report_path': None
                            },
                            'steps_completed': []
                        }), 400
                    
                    # Update dataset_path to use the parquet file
                    dataset_path = parquet_path
                    print(f"\n[AGENT] ✓ Conversion complete")
                    print(f"[AGENT] Using Parquet file: {dataset_path}")
                    print("="*80 + "\n")
                    
                except Exception as conv_error:
                    error_msg = f'Failed to convert CSV to Parquet: {str(conv_error)}'
                    print(f"\n[AGENT ERROR] {error_msg}")
                    import traceback
                    traceback.print_exc()
                    print("="*80 + "\n")
                    return jsonify({
                        'output': error_msg,
                        'error': error_msg,
                        'file_paths': {
                            'labeled_output_path': None,
                            'config_path': None,
                            'model_path': None,
                            'preprocessor_path': None,
                            'synthetic_output_path': None,
                            'anomaly_report_path': None
                        },
                        'steps_completed': []
                    }), 400
            else:
                print(f"[AGENT] File is not CSV (extension: {dataset_path.split('.')[-1]})")
                print(f"[AGENT] Using original path: {dataset_path}")
            
            # Enhance the message with dataset path
            print(f"\n[AGENT] Final dataset path to send to orchestrator: {dataset_path}")
            print(f"[AGENT] File exists: {os.path.exists(dataset_path)}")
            if os.path.exists(dataset_path):
                print(f"[AGENT] File size: {os.path.getsize(dataset_path)} bytes")
            input_message = f"{input_message}\n\nDataset path: {dataset_path}"
        
        # Call orchestrator
        print("\n" + "="*80)
        print("[AGENT] Calling orchestrator service")
        print("="*80)
        orchestrator_url = 'http://localhost:8000/generate'
        print(f"[AGENT] Orchestrator URL: {orchestrator_url}")
        print(f"[AGENT] Request payload:")
        print(f"  - input_message length: {len(input_message)} chars")
        print(f"  - input_message content:\n{input_message}")
        print("="*80 + "\n")
        
        response = requests.post(
            orchestrator_url,
            json={'input_message': input_message},
            timeout=300  # 5 minute timeout for long-running operations
        )
        
        print("\n" + "="*80)
        print("[AGENT] Orchestrator response received")
        print("="*80)
        print(f"[AGENT] Status code: {response.status_code}")
        print(f"[AGENT] Response preview: {response.text[:500]}...")
        print("="*80 + "\n")
        
        if response.status_code != 200:
            return jsonify({
                'output': f'Orchestrator error: {response.text}',
                'error': 'Orchestrator returned an error',
                'file_paths': {
                    'labeled_output_path': None,
                    'config_path': None,
                    'model_path': None,
                    'preprocessor_path': None,
                    'synthetic_output_path': None,
                    'anomaly_report_path': None
                },
                'steps_completed': []
            }), response.status_code
        
        # Parse orchestrator response
        result = response.json()
        
        # Log the orchestrator response for debugging
        print(f"Orchestrator response: {result}")
        
        # Ensure response has all required fields
        if 'output' not in result:
            result['output'] = 'No output from orchestrator'
        
        if 'file_paths' not in result:
            result['file_paths'] = {
                'labeled_output_path': None,
                'config_path': None,
                'model_path': None,
                'preprocessor_path': None,
                'synthetic_output_path': None,
                'anomaly_report_path': None
            }
        else:
            # Ensure all file_path keys exist
            for key in ['labeled_output_path', 'config_path', 'model_path', 
                       'preprocessor_path', 'synthetic_output_path', 'anomaly_report_path']:
                if key not in result['file_paths']:
                    result['file_paths'][key] = None
        
        if 'steps_completed' not in result:
            result['steps_completed'] = []
        
        # Check if there's an error in the output text
        if 'error' in result.get('output', '').lower() or 'failed' in result.get('output', '').lower():
            result['error'] = result['output']
        
        return jsonify(result)
        
    except requests.exceptions.ConnectionError:
        return jsonify({
            'output': 'Orchestrator service not available. Make sure the orchestrator is running on port 8000.',
            'error': 'Orchestrator service not available',
            'file_paths': {
                'labeled_output_path': None,
                'config_path': None,
                'model_path': None,
                'preprocessor_path': None,
                'synthetic_output_path': None,
                'anomaly_report_path': None
            },
            'steps_completed': []
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({
            'output': 'Orchestrator request timed out. The operation took longer than 5 minutes.',
            'error': 'Orchestrator request timed out',
            'file_paths': {
                'labeled_output_path': None,
                'config_path': None,
                'model_path': None,
                'preprocessor_path': None,
                'synthetic_output_path': None,
                'anomaly_report_path': None
            },
            'steps_completed': []
        }), 504
    except Exception as e:
        return jsonify({
            'output': f'Error: {str(e)}',
            'error': str(e),
            'file_paths': {
                'labeled_output_path': None,
                'config_path': None,
                'model_path': None,
                'preprocessor_path': None,
                'synthetic_output_path': None,
                'anomaly_report_path': None
            },
            'steps_completed': []
        }), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload a dataset file (CSV or Parquet) to the server.
    
    Form data:
    - file: The file to upload (multipart/form-data)
    
    Returns:
    {
        "file_path": "uploaded_data/filename.csv",
        "filename": "filename.csv",
        "size_bytes": 12345
    }
    """
    try:
        # Check if file was included in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not (file.filename.endswith('.csv') or file.filename.endswith('.parquet')):
            return jsonify({'error': 'Only .csv and .parquet files are supported'}), 400
        
        # Generate safe filename (prevent path traversal)
        from werkzeug.utils import secure_filename
        import time
        
        original_filename = secure_filename(file.filename)
        # Add timestamp to prevent overwrites
        timestamp = int(time.time() * 1000)
        name, ext = os.path.splitext(original_filename)
        safe_filename = f"{name}_{timestamp}{ext}"
        
        # Save file to upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Return absolute path for consistency with other endpoints
        # Also return relative path for convenience
        absolute_path = str(file_path)
        relative_path = f"uploaded_data/{safe_filename}"
        
        return jsonify({
            'success': True,
            'file_path': absolute_path,  # Absolute path (preferred for API use)
            'relative_path': relative_path,  # Relative path (for display)
            'filename': safe_filename,
            'original_filename': original_filename,
            'size_bytes': file_size,
            'message': f'File uploaded successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/dataset/info', methods=['POST'])
def get_dataset_info():
    """
    Get information about a dataset.
    
    Request body:
    {
        "data_path": "path/to/dataset.parquet"
    }
    """
    try:
        data = request.get_json()
        data_path = data.get('data_path')
        
        if not data_path:
            return jsonify({'error': 'Invalid or missing data_path'}), 400
        
        # Handle both relative and absolute paths
        # If relative, try resolving from backend directory or current working directory
        if not os.path.isabs(data_path):
            # Try relative to backend directory first
            backend_path = os.path.join(backend_dir, data_path)
            if os.path.exists(backend_path):
                data_path = backend_path
            # Otherwise, try current working directory (might already be absolute after join)
            elif not os.path.exists(data_path):
                return jsonify({'error': f'File not found: {data_path}'}), 404
        elif not os.path.exists(data_path):
            return jsonify({'error': f'File not found: {data_path}'}), 404
        
        # Load dataset
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            return jsonify({'error': 'Unsupported file format. Use .csv or .parquet'}), 400
        
        # Get info
        # Convert sample data to dict and replace NaN with None for valid JSON
        sample_records = df.head(5).to_dict(orient='records')
        
        # Replace NaN values with None (which becomes null in JSON)
        import math
        for record in sample_records:
            for key, value in record.items():
                if isinstance(value, float) and math.isnan(value):
                    record[key] = None
        
        info = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': sample_records
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================================
# DATA LABELING ENDPOINTS
# ============================================================================

@app.route('/api/labeling/create-labels', methods=['POST'])
def create_labels_endpoint():
    """
    Label a dataset using weak supervision with Snorkel.
    
    Request body:
    {
        "data_path": "path/to/unlabeled.csv",
        "output_path": "path/to/labeled.parquet",
        "labeling_functions": [
            {
                "name": "lf_high_income",
                "code": "def lf_high_income(x):\\n    return 1 if x['capital-gain'] > 5000 else 0"
            }
        ]
    }
    
    Note: labeling_functions should be Python code strings that define functions
    taking a row (x) and returning 0, 1, or -1 (abstain).
    """
    try:
        from snorkel.labeling import labeling_function
        
        data = request.get_json()
        data_path = data.get('data_path')
        output_path = data.get('output_path', str(OUTPUT_FOLDER / 'labeled_data.parquet'))
        lf_defs = data.get('labeling_functions', [])
        
        if not data_path or not os.path.exists(data_path):
            return jsonify({'error': 'Invalid or missing data_path'}), 400
        
        if not lf_defs:
            return jsonify({'error': 'No labeling functions provided'}), 400
        
        # Parse and compile labeling functions
        labeling_functions_list = []
        for lf_def in lf_defs:
            name = lf_def.get('name', f'lf_{len(labeling_functions_list)}')
            code = lf_def.get('code')
            
            if not code:
                continue
            
            # Execute the function definition to get the plain function
            local_scope = {}
            exec(code, {}, local_scope)
            
            # Get the plain function (not yet decorated)
            plain_func = None
            for key, value in local_scope.items():
                if callable(value) and not key.startswith('_'):
                    plain_func = value
                    break
            
            if plain_func:
                # Now apply the Snorkel decorator to the plain function
                decorated_func = labeling_function(name=name)(plain_func)
                labeling_functions_list.append(decorated_func)
        
        if not labeling_functions_list:
            return jsonify({'error': 'No valid labeling functions could be parsed'}), 400
        
        # Run labeling
        result_path = labeler.create_labels(
            data_path=data_path,
            labeling_functions_list=labeling_functions_list,
            output_path=output_path
        )
        
        return jsonify({
            'success': True,
            'output_path': result_path,
            'message': f'Successfully labeled dataset with {len(labeling_functions_list)} labeling functions'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/labeling/run-pipeline', methods=['POST'])
def run_labeling_pipeline_endpoint():
    """
    Run the self-correcting labeling pipeline (requires agents).
    
    Request body:
    {
        "user_goal": "Label whether income is >50K based on census features",
        "raw_data_path": "path/to/unlabeled.parquet",
        "hand_labeled_examples_path": "path/to/ground_truth.parquet",
        "target_auc_score": 0.85,
        "max_attempts": 3
    }
    """
    if not AGENTS_AVAILABLE:
        return jsonify({'error': 'Agent pipelines not available'}), 503
    
    try:
        data = request.get_json()
        user_goal = data.get('user_goal')
        raw_data_path = data.get('raw_data_path')
        hand_labeled_examples_path = data.get('hand_labeled_examples_path')
        target_auc_score = data.get('target_auc_score', 0.85)
        max_attempts = data.get('max_attempts', 3)
        
        if not all([user_goal, raw_data_path, hand_labeled_examples_path]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Run pipeline
        result_path = run_labeling_pipeline(
            user_goal=user_goal,
            raw_data_path=raw_data_path,
            hand_labeled_examples_path=hand_labeled_examples_path,
            target_auc_score=target_auc_score,
            max_attempts=max_attempts
        )
        
        return jsonify({
            'success': True,
            'output_path': result_path,
            'message': 'Labeling pipeline completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================================
# MODEL TRAINING ENDPOINTS
# ============================================================================

@app.route('/api/training/train-model', methods=['POST'])
def train_model_endpoint():
    """
    Train a generative model on labeled data.
    
    Request body:
    {
        "config_path": "path/to/config.json"
    }
    
    The config file should follow the DataFoundry config format with:
    - data_path: Path to labeled training data
    - output_model_path: Where to save the trained model
    - preprocessor_path: Where to save the preprocessor
    - model_params: Model architecture configuration
    - training_params: Training hyperparameters
    """
    try:
        data = request.get_json()
        config_path = data.get('config_path')
        
        if not config_path or not os.path.exists(config_path):
            return jsonify({'error': 'Invalid or missing config_path'}), 400
        
        # Run training
        model_path = trainer.train_model(config_path=config_path)
        
        # Load config to get preprocessor path
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        preprocessor_path = config.get('preprocessor_path', '')
        
        return jsonify({
            'success': True,
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'config_path': config_path,
            'message': 'Model training completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/training/create-config', methods=['POST'])
def create_training_config():
    """
    Create a training configuration file with sensible defaults.
    
    Request body:
    {
        "data_path": "path/to/labeled_data.parquet",
        "output_dir": "path/to/output/",
        "model_type": "tabular_cvae",  // or "mixed_data_cvae", "tabular_vae_gmm"
        "model_params": {
            "latent_dim": 64,
            "condition_dim": 1,
            "encoder_hidden_layers": [128, 64],
            "decoder_hidden_layers": [64, 128]
        },
        "training_params": {
            "batch_size": 128,
            "learning_rate": 0.001,
            "epochs": 50
        }
    }
    """
    try:
        data = request.get_json()
        data_path = data.get('data_path')
        output_dir = data.get('output_dir', str(OUTPUT_FOLDER))
        model_type = data.get('model_type', 'tabular_cvae')
        
        if not data_path or not os.path.exists(data_path):
            return jsonify({'error': 'Invalid or missing data_path'}), 400
        
        # Load data to infer schema
        df = pd.read_parquet(data_path)
        
        # Infer columns - separate numerical and categorical
        all_cols = df.columns.tolist()
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Categorical columns are non-numerical columns that aren't label columns
        label_cols = ['label', 'label_probability', 'income_binary', 'income']
        categorical_cols = [col for col in all_cols 
                           if col not in numerical_cols and col not in label_cols]
        
        # Remove label columns from numerical features
        condition_col = None
        for col in label_cols:
            if col in numerical_cols:
                condition_col = col
                numerical_cols.remove(col)
                break
        
        if not condition_col:
            # Try to find label_probability or income_binary
            if 'label_probability' in all_cols:
                condition_col = 'label_probability'
            elif 'income_binary' in all_cols:
                condition_col = 'income_binary'
            elif numerical_cols:
                condition_col = numerical_cols[-1]  # Use last numerical column as fallback
                numerical_cols = numerical_cols[:-1]
            else:
                return jsonify({'error': 'No suitable condition column found'}), 400
        
        # Map model type to template
        model_map = {
            'tabular_cvae': ('tabular_cvae.py', 'TabularCVAE'),
            'mixed_data_cvae': ('mixed_data_cvae.py', 'MixedDataCVAE'),
            'tabular_vae_gmm': ('tabular_vae_gmm.py', 'TabularVAEGMM'),
            'tabular_ctgan': ('tabular_ctgan.py', 'TabularCTGAN')
        }
        
        # Auto-select model type if categorical columns exist but user didn't specify mixed_data_cvae
        if categorical_cols and model_type != 'mixed_data_cvae':
            # Automatically switch to MixedDataCVAE to handle categorical columns
            model_template, model_class = model_map['mixed_data_cvae']
            print(f"Note: Categorical columns detected. Automatically using MixedDataCVAE instead of {model_type}")
        else:
            model_template, model_class = model_map.get(model_type, ('tabular_cvae.py', 'TabularCVAE'))
        
        # Build feature_cols dict - include both numerical and categorical if they exist
        feature_cols = {}
        if numerical_cols:
            feature_cols['numerical_cols'] = numerical_cols
        if categorical_cols:
            feature_cols['categorical_cols'] = categorical_cols
        
        # For MixedDataCVAE, we need categorical_embed_dims if categorical columns exist
        model_params = {
            'model_template': model_template,
            'model_class_name': model_class,
            'latent_dim': data.get('model_params', {}).get('latent_dim', 64),
            'condition_dim': data.get('model_params', {}).get('condition_dim', 1),
            'encoder_hidden_layers': data.get('model_params', {}).get('encoder_hidden_layers', [128, 64]),
            'decoder_hidden_layers': data.get('model_params', {}).get('decoder_hidden_layers', [64, 128]),
            'feature_cols': feature_cols,
            'condition_cols': [condition_col]
        }
        
        # If using MixedDataCVAE with categorical columns, calculate embedding dimensions
        if model_class == 'MixedDataCVAE' and categorical_cols:
            categorical_embed_dims = []
            for col in categorical_cols:
                n_categories = df[col].nunique()
                embed_dim = min(8, max(4, n_categories // 2))  # Embedding dim between 4-8
                categorical_embed_dims.append((n_categories, embed_dim))
            model_params['categorical_embed_dims'] = categorical_embed_dims
            model_params['numerical_dim'] = len(numerical_cols) if numerical_cols else 0
        
        # Build config
        os.makedirs(output_dir, exist_ok=True)
        
        config = {
            'data_path': data_path,
            'output_model_path': os.path.join(output_dir, 'model.pth'),
            'preprocessor_path': os.path.join(output_dir, 'preprocessor.joblib'),
            'model_params': model_params,
            'training_params': {
                'batch_size': data.get('training_params', {}).get('batch_size', 128),
                'learning_rate': data.get('training_params', {}).get('learning_rate', 0.001),
                'epochs': data.get('training_params', {}).get('epochs', 3)  # Reduced for faster testing
            }
        }
        
        # Save config
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return jsonify({
            'success': True,
            'config_path': config_path,
            'config': config,
            'message': 'Training configuration created successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/training/run-pipeline', methods=['POST'])
def run_training_pipeline_endpoint():
    """
    Run the self-correcting training pipeline (requires agents).
    
    Request body:
    {
        "labeled_data_path": "path/to/labeled_data.parquet",
        "holdout_test_path": "path/to/test_data.parquet",
        "target_utility_pct": 0.85,
        "max_attempts": 3
    }
    """
    if not AGENTS_AVAILABLE:
        return jsonify({'error': 'Agent pipelines not available'}), 503
    
    try:
        data = request.get_json()
        labeled_data_path = data.get('labeled_data_path')
        holdout_test_path = data.get('holdout_test_path')
        target_utility_pct = data.get('target_utility_pct', 0.85)
        max_attempts = data.get('max_attempts', 3)
        
        if not all([labeled_data_path, holdout_test_path]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Run pipeline
        config_path, model_path, preprocessor_path = run_training_pipeline(
            labeled_data_path=labeled_data_path,
            holdout_test_path=holdout_test_path,
            target_utility_pct=target_utility_pct,
            max_attempts=max_attempts
        )
        
        return jsonify({
            'success': True,
            'config_path': config_path,
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'message': 'Training pipeline completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================================
# DATA GENERATION ENDPOINTS
# ============================================================================

@app.route('/api/generation/generate-data', methods=['POST'])
def generate_data_endpoint():
    """
    Generate synthetic data from a trained model.
    
    Request body:
    {
        "model_path": "path/to/model.pth",
        "config_path": "path/to/config.json",
        "label": 1.0,
        "num_to_generate": 100,
        "output_path": "path/to/output.parquet",
        "output_format": "parquet"  // or "csv", "pt"
    }
    """
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        config_path = data.get('config_path')
        label = data.get('label', 1.0)
        num_to_generate = data.get('num_to_generate', 100)
        output_path = data.get('output_path', str(OUTPUT_FOLDER / 'synthetic_data.parquet'))
        output_format = data.get('output_format', 'parquet')
        
        if not all([model_path, config_path]):
            return jsonify({'error': 'Missing required parameters: model_path, config_path'}), 400
        
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model file not found: {model_path}'}), 400
        
        if not os.path.exists(config_path):
            return jsonify({'error': f'Config file not found: {config_path}'}), 400
        
        # Generate data
        result_path = generator.generate_data(
            model_path=model_path,
            config_path=config_path,
            label=label,
            num_to_generate=num_to_generate,
            output_path=output_path,
            output_format=output_format
        )
        
        return jsonify({
            'success': True,
            'output_path': result_path,
            'num_generated': num_to_generate,
            'label': label,
            'message': f'Successfully generated {num_to_generate} synthetic samples'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/generation/run-pipeline', methods=['POST'])
def run_generation_pipeline_endpoint():
    """
    Run the generation pipeline (wrapper for consistency).
    
    Request body:
    {
        "config_path": "path/to/config.json",
        "model_path": "path/to/model.pth",
        "preprocessor_path": "path/to/preprocessor.joblib",
        "label": 1.0,
        "num_to_generate": 100,
        "output_format": "parquet"
    }
    """
    if not AGENTS_AVAILABLE:
        return jsonify({'error': 'Agent pipelines not available'}), 503
    
    try:
        data = request.get_json()
        config_path = data.get('config_path')
        model_path = data.get('model_path')
        preprocessor_path = data.get('preprocessor_path')
        label = data.get('label', 1.0)
        num_to_generate = data.get('num_to_generate', 100)
        output_format = data.get('output_format', 'parquet')
        
        if not all([config_path, model_path, preprocessor_path]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Run pipeline
        result_path = run_generation_pipeline(
            config_path=config_path,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            label=label,
            num_to_generate=num_to_generate,
            output_format=output_format
        )
        
        return jsonify({
            'success': True,
            'output_path': result_path,
            'message': 'Generation pipeline completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================================
# DATA CLEANING / ANOMALY DETECTION ENDPOINTS
# ============================================================================

@app.route('/api/cleaning/detect-anomalies', methods=['POST'])
def detect_anomalies_endpoint():
    """
    Detect anomalies in a dataset using a trained model.
    
    Request body:
    {
        "config_path": "path/to/config.json",
        "model_path": "path/to/model.pth",
        "preprocessor_path": "path/to/preprocessor.joblib",
        "data_to_scan_path": "path/to/data_to_scan.parquet",
        "output_path": "path/to/anomaly_report.parquet"
    }
    """
    try:
        data = request.get_json()
        config_path = data.get('config_path')
        model_path = data.get('model_path')
        preprocessor_path = data.get('preprocessor_path')
        data_to_scan_path = data.get('data_to_scan_path')
        output_path = data.get('output_path', str(OUTPUT_FOLDER / 'anomaly_report.parquet'))
        
        if not all([config_path, model_path, preprocessor_path, data_to_scan_path]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        for path_name, path_value in [
            ('config', config_path),
            ('model', model_path),
            ('preprocessor', preprocessor_path),
            ('data_to_scan', data_to_scan_path)
        ]:
            if not os.path.exists(path_value):
                return jsonify({'error': f'{path_name} file not found: {path_value}'}), 400
        
        # Detect anomalies
        result_path = evaluator.find_anomalies(
            config_path=config_path,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            data_path_to_scan=data_to_scan_path,
            output_path=output_path
        )
        
        # Load results to provide summary
        result_df = pd.read_parquet(result_path)
        
        return jsonify({
            'success': True,
            'output_path': result_path,
            'num_samples_scanned': len(result_df),
            'anomaly_score_range': [
                float(result_df['anomaly_score'].min()),
                float(result_df['anomaly_score'].max())
            ],
            'mean_anomaly_score': float(result_df['anomaly_score'].mean()),
            'message': 'Anomaly detection completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/cleaning/run-pipeline', methods=['POST'])
def run_anomaly_pipeline_endpoint():
    """
    Run the anomaly detection pipeline.
    
    Request body:
    {
        "config_path": "path/to/config.json",
        "model_path": "path/to/model.pth",
        "preprocessor_path": "path/to/preprocessor.joblib",
        "data_to_scan_path": "path/to/data_to_scan.parquet"
    }
    """
    if not AGENTS_AVAILABLE:
        return jsonify({'error': 'Agent pipelines not available'}), 503
    
    try:
        data = request.get_json()
        config_path = data.get('config_path')
        model_path = data.get('model_path')
        preprocessor_path = data.get('preprocessor_path')
        data_to_scan_path = data.get('data_to_scan_path')
        
        if not all([config_path, model_path, preprocessor_path, data_to_scan_path]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Run pipeline
        result_path = run_anomaly_pipeline(
            config_path=config_path,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            data_to_scan_path=data_to_scan_path
        )
        
        return jsonify({
            'success': True,
            'output_path': result_path,
            'message': 'Anomaly detection pipeline completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================================
# FILE MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/api/files/list', methods=['GET'])
def list_files():
    """List available files in output directory."""
    try:
        output_dir = Path(app.config['OUTPUT_FOLDER'])
        files = []
        
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                files.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'modified': file_path.stat().st_mtime
                })
        
        return jsonify({'files': files})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/files/download', methods=['GET'])
def download_file():
    """
    Download a file.
    
    Query params:
    - path: Path to the file to download (must be within OUTPUT_FOLDER or UPLOAD_FOLDER)
    """
    try:
        file_path = request.args.get('path')
        
        if not file_path:
            return jsonify({'error': 'Missing path parameter'}), 400
        
        # Convert to absolute path and resolve any .. or symlinks
        file_path = os.path.abspath(os.path.normpath(file_path))
        
        # Security: Ensure file is within allowed directories
        output_folder = os.path.abspath(app.config['OUTPUT_FOLDER'])
        upload_folder = os.path.abspath(app.config['UPLOAD_FOLDER'])
        
        if not (file_path.startswith(output_folder) or file_path.startswith(upload_folder)):
            return jsonify({
                'error': 'File path must be within OUTPUT_FOLDER or UPLOAD_FOLDER',
                'allowed_folders': [output_folder, upload_folder]
            }), 403
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        if not os.path.isfile(file_path):
            return jsonify({'error': 'Path is not a file'}), 400
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/files/convert-to-csv', methods=['POST'])
def convert_to_csv():
    """
    Convert a parquet file to CSV format.
    
    Request body:
    {
        "parquet_path": "path/to/file.parquet",
        "output_path": "path/to/output.csv"  // optional, defaults to same directory with .csv extension
    }
    
    Returns:
    {
        "success": true,
        "csv_path": "path/to/output.csv",
        "message": "Successfully converted parquet to CSV"
    }
    """
    try:
        data = request.get_json()
        parquet_path = data.get('parquet_path')
        
        if not parquet_path:
            return jsonify({'error': 'Missing required parameter: parquet_path'}), 400
        
        if not os.path.exists(parquet_path):
            return jsonify({'error': f'Parquet file not found: {parquet_path}'}), 404
        
        if not parquet_path.endswith('.parquet'):
            return jsonify({'error': 'File must be a .parquet file'}), 400
        
        # Security: Validate input path is within allowed directories
        parquet_path = os.path.abspath(os.path.normpath(parquet_path))
        output_folder = os.path.abspath(app.config['OUTPUT_FOLDER'])
        upload_folder = os.path.abspath(app.config['UPLOAD_FOLDER'])
        
        if not (parquet_path.startswith(output_folder) or parquet_path.startswith(upload_folder)):
            return jsonify({
                'error': 'Input file path must be within OUTPUT_FOLDER or UPLOAD_FOLDER',
                'allowed_folders': [output_folder, upload_folder]
            }), 403
        
        # Determine output path
        output_path = data.get('output_path')
        if not output_path:
            # Default: same directory, replace .parquet with .csv
            output_path = parquet_path.rsplit('.parquet', 1)[0] + '.csv'
        else:
            # Validate output path is also within allowed directories
            output_path = os.path.abspath(os.path.normpath(output_path))
            if not (output_path.startswith(output_folder) or output_path.startswith(upload_folder)):
                return jsonify({
                    'error': 'Output file path must be within OUTPUT_FOLDER or UPLOAD_FOLDER'
                }), 403
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Load parquet and convert to CSV
        df = pd.read_parquet(parquet_path)
        df.to_csv(output_path, index=False)
        
        return jsonify({
            'success': True,
            'csv_path': output_path,
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'message': f'Successfully converted {len(df)} rows to CSV format'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/files/convert-to-parquet', methods=['POST'])
def convert_to_parquet():
    """
    Convert a CSV file to Parquet format.
    
    Request body:
    {
        "csv_path": "path/to/file.csv",
        "output_path": "path/to/output.parquet"  // optional, defaults to same directory with .parquet extension
    }
    
    Returns:
    {
        "success": true,
        "parquet_path": "path/to/output.parquet",
        "message": "Successfully converted CSV to Parquet"
    }
    """
    try:
        data = request.get_json()
        csv_path = data.get('csv_path')
        
        if not csv_path:
            return jsonify({'error': 'Missing required parameter: csv_path'}), 400
        
        if not os.path.exists(csv_path):
            return jsonify({'error': f'CSV file not found: {csv_path}'}), 404
        
        if not csv_path.endswith('.csv'):
            return jsonify({'error': 'File must be a .csv file'}), 400
        
        # Security: Validate input path is within allowed directories
        csv_path = os.path.abspath(os.path.normpath(csv_path))
        output_folder = os.path.abspath(app.config['OUTPUT_FOLDER'])
        upload_folder = os.path.abspath(app.config['UPLOAD_FOLDER'])
        testing_folder = os.path.abspath(backend_dir / "testing_data")
        
        if not (csv_path.startswith(output_folder) or csv_path.startswith(upload_folder) or csv_path.startswith(testing_folder)):
            return jsonify({
                'error': 'Input file path must be within OUTPUT_FOLDER, UPLOAD_FOLDER, or testing_data',
                'allowed_folders': [output_folder, upload_folder, testing_folder]
            }), 403
        
        # Determine output path
        output_path = data.get('output_path')
        if not output_path:
            # Default: same directory, replace .csv with .parquet
            output_path = csv_path.rsplit('.csv', 1)[0] + '.parquet'
        else:
            # Validate output path is also within allowed directories
            output_path = os.path.abspath(os.path.normpath(output_path))
            if not (output_path.startswith(output_folder) or output_path.startswith(upload_folder)):
                return jsonify({
                    'error': 'Output file path must be within OUTPUT_FOLDER or UPLOAD_FOLDER'
                }), 403
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Load CSV and convert to Parquet
        df = pd.read_csv(csv_path)
        df.to_parquet(output_path, index=False)
        
        return jsonify({
            'success': True,
            'parquet_path': output_path,
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'message': f'Successfully converted {len(df)} rows to Parquet format'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DataFoundry Flask API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', default=5000, type=int, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║                  DataFoundry Flask API Server                  ║
╚════════════════════════════════════════════════════════════════╝

🚀 Server starting on http://{args.host}:{args.port}

📊 Available Features:
  ✓ Data Labeling      - Weak supervision & automated labeling
  ✓ Model Training     - Train generative models
  ✓ Data Generation    - Generate synthetic data
  ✓ Data Cleaning      - Anomaly detection & quality analysis
  {'✓' if AGENTS_AVAILABLE else '✗'} Agent Pipelines    - Self-correcting workflows

📖 API Documentation:
  Health Check:        GET  /health
  Upload File:         POST /api/upload
  Dataset Info:        POST /api/dataset/info
  
  Labeling:           POST /api/labeling/create-labels
  Labeling Pipeline:  POST /api/labeling/run-pipeline
  
  Training:           POST /api/training/train-model
  Create Config:      POST /api/training/create-config
  Training Pipeline:  POST /api/training/run-pipeline
  
  Generation:         POST /api/generation/generate-data
  Generation Pipeline: POST /api/generation/run-pipeline
  
  Anomaly Detection:  POST /api/cleaning/detect-anomalies
  Cleaning Pipeline:  POST /api/cleaning/run-pipeline
  
  List Files:         GET  /api/files/list
  Download File:      GET  /api/files/download?path=<path>
  Convert to CSV:     POST /api/files/convert-to-csv

Press Ctrl+C to stop the server.
════════════════════════════════════════════════════════════════
""")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

