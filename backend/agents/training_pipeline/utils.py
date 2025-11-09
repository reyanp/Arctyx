"""
Utility functions for the training pipeline.
"""

import warnings
import json
import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')

# Add DataFoundry to path for model loading
import sys
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))
from DataFoundry.utils import Preprocessor, CustomDataset
from torch.utils.data import DataLoader


def evaluate_model_quality(
    config_path: str,
    model_path: str,
    preprocessor_path: str,
    holdout_test_path: str
) -> dict:
    """
    Evaluates model quality using reconstruction error on holdout data.
    This is more appropriate for generative models than classification-based metrics.
    
    Args:
        config_path: Path to the model configuration JSON
        model_path: Path to the trained model
        preprocessor_path: Path to the preprocessor
        holdout_test_path: Path to the holdout test data (real data)
    
    Returns:
        Dictionary with 'reconstruction_error' and 'quality_score' (0-1, higher is better)
    """
    print("--- Running Model Quality Evaluation ---")
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        params = config['model_params']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Calculate input_dim automatically if not specified (for tabular models)
        # This matches the logic in trainer.py
        if 'input_dim' not in params and 'text_model' not in params:
            num_numerical = len(params.get('feature_cols', {}).get('numerical_cols', []))
            num_categorical = len(params.get('feature_cols', {}).get('categorical_cols', []))
            params['input_dim'] = num_numerical + num_categorical
            print(f"Auto-calculated input_dim: {params['input_dim']} (numerical: {num_numerical}, categorical: {num_categorical})")
        
        # Load holdout test data
        test_df = pd.read_parquet(holdout_test_path)
        print(f"Evaluating on {len(test_df)} holdout samples")
        
        # Get condition column from config
        condition_cols = params.get('condition_cols', [])
        if not condition_cols:
            condition_cols = ['label_probability']  # Default fallback
        
        # Ensure holdout test set has the required condition column(s)
        # If missing, add it using available label columns or default values
        for condition_col in condition_cols:
            if condition_col not in test_df.columns:
                # Try to find a suitable replacement
                if 'income_binary' in test_df.columns:
                    # Use income_binary as a proxy for label_probability
                    test_df[condition_col] = test_df['income_binary'].astype(float)
                    print(f"  Added missing condition column '{condition_col}' using 'income_binary'")
                elif 'label' in test_df.columns:
                    test_df[condition_col] = test_df['label'].astype(float)
                    print(f"  Added missing condition column '{condition_col}' using 'label'")
                else:
                    # Use default value (0.5) if no label column available
                    test_df[condition_col] = 0.5
                    print(f"  Added missing condition column '{condition_col}' with default value 0.5")
        
        # Load preprocessor
        preprocessor = joblib.load(preprocessor_path)
        
        # Create dataset and dataloader
        test_dataset = CustomDataset(test_df, preprocessor)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Load model
        import importlib
        model_name = params['model_template'].replace('.py', '')
        model_module = importlib.import_module(f'DataFoundry.models.{model_name}')
        ModelClass = getattr(model_module, params['model_class_name'])
        model = ModelClass(params).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Calculate reconstruction error
        total_error = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                condition = batch['condition'].to(device)
                
                # Handle different feature types
                if 'numerical_features' in batch or 'categorical_features' in batch:
                    # Check if this is MixedDataCVAE (needs special handling)
                    is_mixed_cvae = params.get('model_class_name') == 'MixedDataCVAE'
                    
                    if is_mixed_cvae:
                        # MixedDataCVAE requires separate numerical and categorical inputs
                        x_num = batch['numerical_features'].to(device)
                        x_cat = batch['categorical_features'].to(device).long()
                        recon_num, recon_cat_logits, mu, logvar = model(x_num, x_cat, condition)
                        
                        # Calculate error: MSE for numerical, CrossEntropy for categorical
                        num_error = torch.nn.functional.mse_loss(recon_num, x_num, reduction='sum')
                        cat_error = 0.0
                        for i, cat_logits in enumerate(recon_cat_logits):
                            cat_error += torch.nn.functional.cross_entropy(
                                cat_logits, 
                                x_cat[:, i], 
                                reduction='sum'
                            )
                        error = num_error + cat_error
                        total_error += error.item()
                        num_samples += x_num.size(0)
                    else:
                        # Standard tabular model
                        feature_list = []
                        if 'numerical_features' in batch:
                            feature_list.append(batch['numerical_features'].to(device))
                        if 'categorical_features' in batch:
                            cat_features = batch['categorical_features'].to(device).float()
                            feature_list.append(cat_features)
                        
                        x = torch.cat(feature_list, dim=1) if len(feature_list) > 1 else feature_list[0]
                        
                        # Forward pass
                        recon_x, mu, logvar = model(x, condition)
                        
                        # Calculate MSE reconstruction error
                        error = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
                        total_error += error.item()
                        num_samples += x.size(0)
        
        # Average reconstruction error per sample
        avg_reconstruction_error = total_error / num_samples if num_samples > 0 else float('inf')
        
        # Convert to quality score (0-1, lower error = higher quality)
        # Normalize: assume good models have error < 1.0, scale accordingly
        quality_score = max(0.0, min(1.0, 1.0 / (1.0 + avg_reconstruction_error)))
        
        print(f"Average Reconstruction Error: {avg_reconstruction_error:.6f}")
        print(f"Quality Score: {quality_score:.4f} (higher is better, 0-1 scale)")
        
        return {
            "reconstruction_error": float(avg_reconstruction_error),
            "quality_score": float(quality_score),
            "num_samples_evaluated": num_samples
        }
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()
        # Return default scores if evaluation fails
        return {
            "reconstruction_error": float('inf'),
            "quality_score": 0.0,
            "num_samples_evaluated": 0
        }

