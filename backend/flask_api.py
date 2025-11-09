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
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'features': {
            'data_labeling': True,
            'model_training': True,
            'data_generation': True,
            'anomaly_detection': True,
            'agent_pipelines': AGENTS_AVAILABLE
        }
    })


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
        
        if not data_path or not os.path.exists(data_path):
            return jsonify({'error': 'Invalid or missing data_path'}), 400
        
        # Load dataset
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            return jsonify({'error': 'Unsupported file format. Use .csv or .parquet'}), 400
        
        # Get info
        info = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': df.head(5).to_dict(orient='records')
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
        
        # Infer columns
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove label columns from features
        label_cols = ['label', 'label_probability', 'income_binary']
        condition_col = None
        for col in label_cols:
            if col in numerical_cols:
                condition_col = col
                numerical_cols.remove(col)
                break
        
        if not condition_col:
            condition_col = numerical_cols[-1]  # Use last numerical column
            numerical_cols = numerical_cols[:-1]
        
        # Map model type to template
        model_map = {
            'tabular_cvae': ('tabular_cvae.py', 'TabularCVAE'),
            'mixed_data_cvae': ('mixed_data_cvae.py', 'MixedDataCVAE'),
            'tabular_vae_gmm': ('tabular_vae_gmm.py', 'TabularVAEGMM'),
            'tabular_ctgan': ('tabular_ctgan.py', 'TabularCTGAN')
        }
        
        model_template, model_class = model_map.get(model_type, ('tabular_cvae.py', 'TabularCVAE'))
        
        # Build config
        os.makedirs(output_dir, exist_ok=True)
        
        config = {
            'data_path': data_path,
            'output_model_path': os.path.join(output_dir, 'model.pth'),
            'preprocessor_path': os.path.join(output_dir, 'preprocessor.joblib'),
            'model_params': {
                'model_template': model_template,
                'model_class_name': model_class,
                'latent_dim': data.get('model_params', {}).get('latent_dim', 64),
                'condition_dim': data.get('model_params', {}).get('condition_dim', 1),
                'encoder_hidden_layers': data.get('model_params', {}).get('encoder_hidden_layers', [128, 64]),
                'decoder_hidden_layers': data.get('model_params', {}).get('decoder_hidden_layers', [64, 128]),
                'feature_cols': {
                    'numerical_cols': numerical_cols
                },
                'condition_cols': [condition_col]
            },
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
    - path: Path to the file to download
    """
    try:
        file_path = request.args.get('path')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

Press Ctrl+C to stop the server.
════════════════════════════════════════════════════════════════
""")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

