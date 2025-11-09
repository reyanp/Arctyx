"""
Example script demonstrating DataFoundry Flask API usage.

This script shows how to:
1. Label data using weak supervision
2. Create a training configuration
3. Train a generative model
4. Generate synthetic data
5. Detect anomalies

Before running, make sure the Flask API server is running:
    python flask_api.py
"""

import requests
import time
import pandas as pd

BASE_URL = "http://localhost:5000"


def check_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            info = response.json()
            print("✓ API is running")
            print(f"  Features: {info['features']}")
            return True
        return False
    except Exception as e:
        print(f"✗ API is not running: {e}")
        print("  Please start the server with: python flask_api.py")
        return False


def label_data():
    """Step 1: Label unlabeled data using weak supervision."""
    print("\n" + "="*60)
    print("STEP 1: Labeling Data")
    print("="*60)
    
    # First, create a smaller sample dataset for faster testing
    print("Creating small sample dataset for faster testing...")
    full_df = pd.read_csv("testing_data/adult.csv")
    sample_df = full_df.sample(n=min(500, len(full_df)), random_state=42)  # Only 500 samples
    sample_path = "output_data/adult_sample.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"  Sampled {len(sample_df)} rows from {len(full_df)} total")
    
    url = f"{BASE_URL}/api/labeling/create-labels"
    
    # Define labeling functions as code strings
    labeling_functions = [
        {
            "name": "lf_high_capital_gain",
            "code": """def lf_high_capital_gain(x):
    '''Label as high income if capital gain is high'''
    return 1 if x.get('capital-gain', 0) > 5000 else 0"""
        },
        {
            "name": "lf_educated",
            "code": """def lf_educated(x):
    '''Label as high income if highly educated'''
    return 1 if x.get('education-num', 0) >= 13 else 0"""
        },
        {
            "name": "lf_professional",
            "code": """def lf_professional(x):
    '''Label as high income if professional occupation'''
    occupation = x.get('occupation', '')
    return 1 if 'Prof-specialty' in str(occupation) or 'Exec-managerial' in str(occupation) else 0"""
        }
    ]
    
    data = {
        "data_path": sample_path,  # Use sampled data
        "output_path": "output_data/adult_labeled.parquet",
        "labeling_functions": labeling_functions
    }
    
    print(f"Sending request to: {url}")
    print(f"Labeling functions: {len(labeling_functions)}")
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ {result['message']}")
        print(f"  Output: {result['output_path']}")
        
        # Show sample of labeled data
        df = pd.read_parquet(result['output_path'])
        print(f"\n  Labeled {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Label probability stats: min={df['label_probability'].min():.3f}, "
              f"max={df['label_probability'].max():.3f}, "
              f"mean={df['label_probability'].mean():.3f}")
        
        return result['output_path']
    else:
        print(f"✗ Error: {response.json().get('error', 'Unknown error')}")
        return None


def create_config(labeled_path):
    """Step 2: Create a training configuration."""
    print("\n" + "="*60)
    print("STEP 2: Creating Training Configuration")
    print("="*60)
    
    url = f"{BASE_URL}/api/training/create-config"
    
    data = {
        "data_path": labeled_path,
        "output_dir": "output_data/models/",
        "model_type": "tabular_cvae",
        "model_params": {
            "latent_dim": 64,
            "encoder_hidden_layers": [128, 64],
            "decoder_hidden_layers": [64, 128]
        },
        "training_params": {
            "epochs": 3,  # Minimal for quick testing
            "batch_size": 128,
            "learning_rate": 0.001
        }
    }
    
    print(f"Sending request to: {url}")
    print(f"Model type: {data['model_type']}")
    print(f"Training params: {data['training_params']}")
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ {result['message']}")
        print(f"  Config: {result['config_path']}")
        
        # Show config details
        config = result['config']
        print(f"\n  Model: {config['model_params']['model_class_name']}")
        print(f"  Latent dim: {config['model_params']['latent_dim']}")
        print(f"  Features: {len(config['model_params']['feature_cols']['numerical_cols'])} numerical")
        print(f"  Epochs: {config['training_params']['epochs']}")
        
        return result['config_path']
    else:
        print(f"✗ Error: {response.json().get('error', 'Unknown error')}")
        return None


def train_model(config_path):
    """Step 3: Train a generative model."""
    print("\n" + "="*60)
    print("STEP 3: Training Model")
    print("="*60)
    print("⏳ This may take a few minutes...")
    
    url = f"{BASE_URL}/api/training/train-model"
    
    data = {
        "config_path": config_path
    }
    
    start_time = time.time()
    response = requests.post(url, json=data)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ {result['message']}")
        print(f"  Model: {result['model_path']}")
        print(f"  Preprocessor: {result['preprocessor_path']}")
        print(f"  Training time: {elapsed:.1f} seconds")
        
        return result['model_path'], result['preprocessor_path'], config_path
    else:
        print(f"✗ Error: {response.json().get('error', 'Unknown error')}")
        return None, None, None


def generate_synthetic_data(model_path, config_path):
    """Step 4: Generate synthetic data."""
    print("\n" + "="*60)
    print("STEP 4: Generating Synthetic Data")
    print("="*60)
    
    url = f"{BASE_URL}/api/generation/generate-data"
    
    # Generate high-income samples
    data = {
        "model_path": model_path,
        "config_path": config_path,
        "label": 1.0,
        "num_to_generate": 50,  # Reduced for faster testing
        "output_path": "output_data/synthetic_high_income.parquet",
        "output_format": "parquet"
    }
    
    print(f"Generating {data['num_to_generate']} high-income samples (label={data['label']})...")
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Generated high-income data: {result['output_path']}")
        
        # Show sample
        df = pd.read_parquet(result['output_path'])
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)[:5]}... ({len(df.columns)} total)")
        
        high_income_path = result['output_path']
    else:
        print(f"✗ Error: {response.json().get('error', 'Unknown error')}")
        high_income_path = None
    
    # Generate low-income samples
    data["label"] = 0.0
    data["num_to_generate"] = 50  # Reduced for faster testing
    data["output_path"] = "output_data/synthetic_low_income.parquet"
    
    print(f"\nGenerating {data['num_to_generate']} low-income samples (label={data['label']})...")
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Generated low-income data: {result['output_path']}")
        
        df = pd.read_parquet(result['output_path'])
        print(f"  Shape: {df.shape}")
        
        low_income_path = result['output_path']
    else:
        print(f"✗ Error: {response.json().get('error', 'Unknown error')}")
        low_income_path = None
    
    return high_income_path, low_income_path


def detect_anomalies(model_path, config_path, preprocessor_path):
    """Step 5: Detect anomalies in data."""
    print("\n" + "="*60)
    print("STEP 5: Detecting Anomalies")
    print("="*60)
    
    url = f"{BASE_URL}/api/cleaning/detect-anomalies"
    
    data = {
        "config_path": config_path,
        "model_path": model_path,
        "preprocessor_path": preprocessor_path,
        "data_to_scan_path": "output_data/adult_sample.csv",  # Use sampled data for faster testing
        "output_path": "output_data/anomaly_report.parquet"
    }
    
    print(f"Scanning data for anomalies...")
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ {result['message']}")
        print(f"  Scanned: {result['num_samples_scanned']} samples")
        print(f"  Score range: [{result['anomaly_score_range'][0]:.4f}, {result['anomaly_score_range'][1]:.4f}]")
        print(f"  Mean score: {result['mean_anomaly_score']:.4f}")
        print(f"  Report: {result['output_path']}")
        
        # Show top anomalies
        df = pd.read_parquet(result['output_path'])
        print(f"\n  Top 5 anomalies:")
        for i, (idx, row) in enumerate(df.head(5).iterrows(), 1):
            print(f"    {i}. Score: {row['anomaly_score']:.4f}")
        
        return result['output_path']
    else:
        print(f"✗ Error: {response.json().get('error', 'Unknown error')}")
        return None


def main():
    """Run the complete DataFoundry workflow."""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        DataFoundry Flask API - Complete Workflow           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Check if API is running
    if not check_health():
        return
    
    try:
        # Step 1: Label data
        labeled_path = label_data()
        if not labeled_path:
            return
        
        # Step 2: Create config
        config_path = create_config(labeled_path)
        if not config_path:
            return
        
        # Step 3: Train model
        model_path, preprocessor_path, config_path = train_model(config_path)
        if not model_path:
            return
        
        # Step 4: Generate synthetic data
        high_path, low_path = generate_synthetic_data(model_path, config_path)
        
        # Step 5: Detect anomalies
        anomaly_path = detect_anomalies(model_path, config_path, preprocessor_path)
        
        print("\n" + "="*60)
        print("✅ WORKFLOW COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print(f"  1. Labeled data:        {labeled_path}")
        print(f"  2. Training config:     {config_path}")
        print(f"  3. Trained model:       {model_path}")
        print(f"  4. Preprocessor:        {preprocessor_path}")
        if high_path:
            print(f"  5. Synthetic (high):    {high_path}")
        if low_path:
            print(f"  6. Synthetic (low):     {low_path}")
        if anomaly_path:
            print(f"  7. Anomaly report:      {anomaly_path}")
        
        print("\n💡 Tip: You can now use these files in your frontend application!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Workflow interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

