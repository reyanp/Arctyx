import warnings
import importlib
import json
import os
import tempfile

import joblib
import pandas as pd
import torch
from torch.utils.data import Dataset

from DataFoundry.utils import Preprocessor

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')


def generate_data(model_path, config_path, label, num_to_generate, output_path, output_format='parquet'):
    """
    Generates synthetic data from a trained decoder.
    
    Args:
        model_path: Path to saved full model state dict (.pth file)
        config_path: Path to JSON config file
        label: Desired label/condition value (e.g., 1 for positive class)
        num_to_generate: Number of samples to generate
        output_path: Path where generated data will be saved
        output_format: 'parquet' for Parquet (default, recommended for internal use),
                      'pt' for PyTorch tensors, or 'csv' for CSV (not recommended for internal use)
    
    Returns:
        output_path: String path to the generated file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    params = config['model_params']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Calculate input_dim automatically if not specified (for tabular models)
    if 'input_dim' not in params and 'text_model' not in params:
        num_numerical = len(params.get('feature_cols', {}).get('numerical_cols', []))
        num_categorical = len(params.get('feature_cols', {}).get('categorical_cols', []))
        params['input_dim'] = num_numerical + num_categorical
        print(f"Auto-calculated input_dim: {params['input_dim']} (numerical: {num_numerical}, categorical: {num_categorical})")
    
    print(f"Using device: {device}")
    print(f"Generating {num_to_generate} samples with label {label}")
    
    # Dynamically Load Model (Structure only)
    model_name = params['model_template'].replace('.py', '')
    model_module = importlib.import_module(f'DataFoundry.models.{model_name}')
    ModelClass = getattr(model_module, params['model_class_name'])
    model = ModelClass(params)
    
    # Load Trained Weights (full model, then use decoder)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    print(f"Loaded full model from {model_path}")
    
    # Load Preprocessor
    preprocessor = joblib.load(config['preprocessor_path'])
    print(f"Loaded preprocessor from {config['preprocessor_path']}")
    
    # Generate Inputs
    z = torch.randn(num_to_generate, params['latent_dim']).to(device)
    # Create condition vector - handle both single value and multi-dimensional conditions
    condition_dim = params.get('condition_dim', 1)
    if condition_dim == 1:
        c = torch.tensor([label] * num_to_generate, dtype=torch.float32).unsqueeze(1).to(device)
    else:
        # For multi-dimensional conditions, repeat the label across all dimensions
        c = torch.tensor([[label] * condition_dim] * num_to_generate, dtype=torch.float32).to(device)
    
    # Generate Data
    with torch.no_grad():
        # Check if this is MixedDataCVAE (has different decode signature)
        is_mixed_cvae = params.get('model_class_name') == 'MixedDataCVAE' or 'categorical_embed_dims' in params
        
        # For text models, need to specify sequence length
        if 'text_model' in params:
            seq_len = params.get('max_length', 128)
            generated_tensors = model.decode(z, c, seq_len=seq_len)
        elif is_mixed_cvae:
            # MixedDataCVAE returns (recon_num, recon_cat_logits)
            recon_num, recon_cat_logits = model.decode(z, c)
            generated_tensors = (recon_num, recon_cat_logits)
        else:
            generated_tensors = model.decode(z, c)
    
    # Handle Output
    if output_format == 'parquet':
        # Convert to DataFrame using inverse_transform and save as Parquet (recommended for internal use)
        tensors_dict = {}
        
        # Handle text vs tabular models differently
        if 'text_model' in params:
            # For text models: convert logits to token IDs
            # generated_tensors shape: [batch_size, seq_len, vocab_size]
            token_ids = torch.argmax(generated_tensors, dim=-1)  # [batch_size, seq_len]
            tensors_dict['output_text_ids'] = token_ids.cpu()
        elif is_mixed_cvae:
            # MixedDataCVAE returns separate numerical and categorical outputs
            recon_num, recon_cat_logits = generated_tensors
            tensors_dict['output_numerical'] = recon_num.cpu()
            # Convert categorical logits to indices
            cat_indices = torch.stack([
                torch.argmax(cat_logits, dim=-1) for cat_logits in recon_cat_logits
            ], dim=1)
            tensors_dict['output_categorical'] = cat_indices.cpu()
        else:
            # For standard tabular models: split into numerical and categorical features
            # The model outputs combined features, need to split them
            num_numerical = len(params.get('feature_cols', {}).get('numerical_cols', []))
            num_categorical = len(params.get('feature_cols', {}).get('categorical_cols', []))
            
            generated_cpu = generated_tensors.cpu()
            
            # Split the output tensor
            if num_numerical > 0 and num_categorical > 0:
                # Both types exist - split the tensor
                tensors_dict['output_numerical'] = generated_cpu[:, :num_numerical]
                tensors_dict['output_categorical'] = generated_cpu[:, num_numerical:num_numerical + num_categorical]
            elif num_numerical > 0:
                # Only numerical
                tensors_dict['output_numerical'] = generated_cpu
            elif num_categorical > 0:
                # Only categorical
                tensors_dict['output_categorical'] = generated_cpu
            else:
                # Fallback: assume all numerical
                tensors_dict['output_numerical'] = generated_cpu
        
        df = preprocessor.inverse_transform(tensors_dict)
        
        # Add condition column back so the data can be used with the preprocessor again
        # Get condition columns from preprocessor (it stores them from config)
        condition_cols = preprocessor.condition_cols
        if condition_cols:
            # Handle both single value and multi-dimensional conditions
            if condition_dim == 1:
                # For single dimension, assign label to all condition columns
                if isinstance(condition_cols, list):
                    for col in condition_cols:
                        df[col] = label
                else:
                    df[condition_cols] = label
            else:
                # For multi-dimensional, repeat label for each column
                if isinstance(condition_cols, list):
                    for col in condition_cols:
                        df[col] = label  # For now, use same label for all dimensions
                else:
                    df[condition_cols] = label
        
        df.to_parquet(output_path, index=False)
        print(f"Saved generated data to {output_path} (Parquet format)")
    elif output_format == 'pt':
        # Save as PyTorch tensor format (for direct model use)
        torch.save(generated_tensors, output_path)
        print(f"Saved generated tensors to {output_path} (PyTorch format)")
    elif output_format == 'csv':
        # Convert to DataFrame using inverse_transform and save as CSV (not recommended)
        tensors_dict = {}
        
        # Handle text vs tabular models differently
        if 'text_model' in params:
            # For text models: convert logits to token IDs
            token_ids = torch.argmax(generated_tensors, dim=-1)  # [batch_size, seq_len]
            tensors_dict['output_text_ids'] = token_ids.cpu()
        elif is_mixed_cvae:
            # MixedDataCVAE returns separate numerical and categorical outputs
            recon_num, recon_cat_logits = generated_tensors
            tensors_dict['output_numerical'] = recon_num.cpu()
            # Convert categorical logits to indices
            cat_indices = torch.stack([
                torch.argmax(cat_logits, dim=-1) for cat_logits in recon_cat_logits
            ], dim=1)
            tensors_dict['output_categorical'] = cat_indices.cpu()
        else:
            # For standard tabular models: split into numerical and categorical features
            # The model outputs combined features, need to split them
            num_numerical = len(params.get('feature_cols', {}).get('numerical_cols', []))
            num_categorical = len(params.get('feature_cols', {}).get('categorical_cols', []))
            
            generated_cpu = generated_tensors.cpu()
            
            # Split the output tensor
            if num_numerical > 0 and num_categorical > 0:
                # Both types exist - split the tensor
                tensors_dict['output_numerical'] = generated_cpu[:, :num_numerical]
                tensors_dict['output_categorical'] = generated_cpu[:, num_numerical:num_numerical + num_categorical]
            elif num_numerical > 0:
                # Only numerical
                tensors_dict['output_numerical'] = generated_cpu
            elif num_categorical > 0:
                # Only categorical
                tensors_dict['output_categorical'] = generated_cpu
            else:
                # Fallback: assume all numerical
                tensors_dict['output_numerical'] = generated_cpu
        
        df = preprocessor.inverse_transform(tensors_dict)
        
        # Add condition column back so the data can be used with the preprocessor again
        # Get condition columns from preprocessor (it stores them from config)
        condition_cols = preprocessor.condition_cols
        if condition_cols:
            # Handle both single value and multi-dimensional conditions
            if condition_dim == 1:
                # For single dimension, assign label to all condition columns
                if isinstance(condition_cols, list):
                    for col in condition_cols:
                        df[col] = label
                else:
                    df[condition_cols] = label
            else:
                # For multi-dimensional, repeat label for each column
                if isinstance(condition_cols, list):
                    for col in condition_cols:
                        df[col] = label  # For now, use same label for all dimensions
                else:
                    df[condition_cols] = label
        
        df.to_csv(output_path, index=False)
        print(f"Saved generated data to {output_path} (CSV format - not recommended for internal use)")
    else:
        raise ValueError(f"Unsupported output_format: {output_format}. Must be 'parquet', 'pt', or 'csv'")
    
    return output_path


class GeneratedTensorDataset(Dataset):
    """
    PyTorch Dataset wrapper for generated tensors.
    Useful for directly using generated data in PyTorch training pipelines.
    """
    
    def __init__(self, tensor_path):
        """
        Args:
            tensor_path: Path to saved PyTorch tensor file (.pt)
        """
        self.tensors = torch.load(tensor_path, map_location='cpu')
    
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        return self.tensors[idx]


def generate_data_as_dataset(model_path, config_path, label, num_to_generate):
    """
    Generate synthetic data and return as a PyTorch Dataset.
    
    Args:
        model_path: Path to saved full model state dict (.pth file)
        config_path: Path to JSON config file
        label: Desired label/condition value (e.g., 1 for positive class)
        num_to_generate: Number of samples to generate
    
    Returns:
        GeneratedTensorDataset: PyTorch Dataset containing generated tensors
    """
    # Generate data to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Generate data
        generate_data(
            model_path=model_path,
            config_path=config_path,
            label=label,
            num_to_generate=num_to_generate,
            output_path=tmp_path,
            output_format='pt'
        )
        
        # Return as Dataset
        return GeneratedTensorDataset(tmp_path)
    except Exception as e:
        # Clean up on error
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_parquet_as_dataset(parquet_path, preprocessor_path):
    """
    Load a Parquet file and convert it to a PyTorch Dataset using the preprocessor.
    This ensures consistency with the training pipeline.
    
    Args:
        parquet_path: Path to Parquet file
        preprocessor_path: Path to saved preprocessor (joblib file)
    
    Returns:
        CustomDataset: PyTorch Dataset ready for use with DataLoader
    """
    from DataFoundry.utils import CustomDataset
    
    # Load data and preprocessor
    df = pd.read_parquet(parquet_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Create dataset (same as used in training)
    return CustomDataset(df, preprocessor)

