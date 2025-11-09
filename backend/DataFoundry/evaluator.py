import warnings
import json
import importlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import joblib
from DataFoundry.utils import Preprocessor, CustomDataset

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')


def find_anomalies(config_path, model_path, preprocessor_path, data_path_to_scan, output_path):
    """
    Uses a trained cVAE model to score a dataset for anomalies.
    An anomaly is a data point that the cVAE is bad at reconstructing.
    Measures this by calculating reconstruction error (MSE) for each data point.
    
    Args:
        config_path: Path to the config.json file
        model_path: Path to the saved full model (.pth) file
        preprocessor_path: Path to the saved preprocessor.joblib file
        data_path_to_scan: Path to the .parquet file to be scored
        output_path: Path to save the resulting .parquet file with anomaly scores
    
    Returns:
        output_path: String path to the output file with anomaly scores
    """
    # Setup
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    params = config['model_params']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Calculate input_dim automatically if not specified (for tabular models)
    if 'input_dim' not in params and 'text_model' not in params:
        num_numerical = len(params.get('feature_cols', {}).get('numerical_cols', []))
        num_categorical = len(params.get('feature_cols', {}).get('categorical_cols', []))
        params['input_dim'] = num_numerical + num_categorical
        print(f"Auto-calculated input_dim: {params['input_dim']} (numerical: {num_numerical}, categorical: {num_categorical})")
    
    # Load Tools
    preprocessor = joblib.load(preprocessor_path)
    print(f"Loaded preprocessor from {preprocessor_path}")
    
    # Dynamically load the cVAE model structure
    model_name = params['model_template'].replace('.py', '')
    model_module = importlib.import_module(f'DataFoundry.models.{model_name}')
    ModelClass = getattr(model_module, params['model_class_name'])
    
    # Initialize the model
    model = ModelClass(params).to(device)
    
    # Load the full model's weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Load Data
    df = pd.read_parquet(data_path_to_scan)
    print(f"Loaded {len(df)} rows to scan for anomalies")
    
    # Use CustomDataset to wrap the DataFrame
    dataset = CustomDataset(df, preprocessor)
    
    # Create DataLoader (shuffle=False to map scores back to original DataFrame)
    batch_size = config['training_params']['batch_size']
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate Anomaly Scores
    all_scores = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move all tensors in the batch to device
            condition = batch['condition'].to(device)
            
            # Extract features to be reconstructed
            if 'numerical_features' in batch or 'categorical_features' in batch:
                # Tabular model - combine numerical and categorical features
                feature_list = []
                
                if 'numerical_features' in batch:
                    feature_list.append(batch['numerical_features'].to(device))
                
                if 'categorical_features' in batch:
                    cat_features = batch['categorical_features'].to(device).float()
                    feature_list.append(cat_features)
                
                # Concatenate all features
                x = torch.cat(feature_list, dim=1) if len(feature_list) > 1 else feature_list[0]
                
                # Run the full model
                recon_x, mu, logvar = model(x, condition)
                
                # Calculate reconstruction error per sample
                # MSE loss with reduction='none' gives error for each feature
                recon_error = F.mse_loss(recon_x, x, reduction='none')
                # Average across features to get per-sample score
                sample_scores = torch.mean(recon_error, dim=1)
                
            elif 'input_ids' in batch:
                # Text model
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                condition = batch['condition'].to(device)
                
                # Run the full model
                recon_logits, mu, logvar = model(input_ids, attention_mask, condition)
                
                # Calculate reconstruction error (cross-entropy per sample)
                # Reshape for per-sample calculation
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                vocab_size = recon_logits.size(-1)
                
                # Calculate cross-entropy per position
                recon_logits_flat = recon_logits.view(-1, vocab_size)
                input_ids_flat = input_ids.view(-1)
                ce_per_position = F.cross_entropy(
                    recon_logits_flat,
                    input_ids_flat,
                    reduction='none'
                )
                # Reshape back and average over sequence length
                ce_per_sample = ce_per_position.view(batch_size, seq_len)
                sample_scores = torch.mean(ce_per_sample, dim=1)
            else:
                raise ValueError("No recognized features in batch")
            
            # Append scores to list
            all_scores.extend(sample_scores.cpu().numpy())
    
    # Save Results
    df['anomaly_score'] = all_scores
    
    # Sort DataFrame to show highest anomalies first
    df = df.sort_values(by='anomaly_score', ascending=False)
    
    # Save the sorted DataFrame
    df.to_parquet(output_path, index=False)
    print(f"Saved anomaly scores to {output_path}")
    print(f"  - Anomaly score range: [{df['anomaly_score'].min():.4f}, {df['anomaly_score'].max():.4f}]")
    print(f"  - Mean anomaly score: {df['anomaly_score'].mean():.4f}")
    
    return output_path

