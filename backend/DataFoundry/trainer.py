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


def _vae_loss_function(recon_x, x, mu, logvar):
    """
    Private helper to calculate VAE loss (Reconstruction + KL Divergence).
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
    
    Returns:
        Total VAE loss
    """
    RECON_LOSS = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return RECON_LOSS + KLD


def train_model(config_path):
    """
    Main function to run the entire training process from a config file.
    
    Args:
        config_path: Path to JSON config file
    
    Returns:
        model_path: Path to saved full model state dict
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    params = config['model_params']
    train_params = config['training_params']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load data
    df = pd.read_parquet(config['data_path'])
    print(f"Loaded data with {len(df)} rows")
    
    # Initialize & Fit Preprocessor
    preprocessor = Preprocessor(config)
    preprocessor.fit(df)
    
    # Calculate input_dim automatically if not specified (for tabular models)
    if 'input_dim' not in params and 'text_model' not in params:
        num_numerical = len(params.get('feature_cols', {}).get('numerical_cols', []))
        num_categorical = len(params.get('feature_cols', {}).get('categorical_cols', []))
        params['input_dim'] = num_numerical + num_categorical
        print(f"Auto-calculated input_dim: {params['input_dim']} (numerical: {num_numerical}, categorical: {num_categorical})")
    
    # Save Preprocessor
    joblib.dump(preprocessor, config['preprocessor_path'])
    print(f"Saved preprocessor to {config['preprocessor_path']}")
    
    # Create PyTorch Dataset
    dataset = CustomDataset(df, preprocessor)
    data_loader = DataLoader(
        dataset,
        batch_size=train_params['batch_size'],
        shuffle=True
    )
    
    # Dynamically Load Model
    model_name = params['model_template'].replace('.py', '')
    model_module = importlib.import_module(f'DataFoundry.models.{model_name}')
    ModelClass = getattr(model_module, params['model_class_name'])
    model = ModelClass(params).to(device)
    print(f"Loaded model: {params['model_class_name']} from {params['model_template']}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params['learning_rate']
    )
    
    # Training Loop
    model.train()
    for epoch in range(train_params['epochs']):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            # Move tensors to device
            condition = batch['condition'].to(device)
            
            # Handle different feature types
            if 'numerical_features' in batch or 'categorical_features' in batch:
                # Check if this is MixedDataCVAE (needs special handling)
                if params.get('model_class_name') == 'MixedDataCVAE' or 'MixedDataCVAE' in str(type(model)):
                    # MixedDataCVAE requires separate numerical and categorical inputs
                    x_num = batch['numerical_features'].to(device)
                    x_cat = batch['categorical_features'].to(device).long()  # Keep as long for embedding
                    recon_num, recon_cat_logits, mu, logvar = model(x_num, x_cat, condition)
                    
                    # Loss: MSE for numerical, CrossEntropy for categorical
                    num_loss = F.mse_loss(recon_num, x_num, reduction='sum')
                    cat_loss = 0
                    for i, cat_logits in enumerate(recon_cat_logits):
                        cat_loss += F.cross_entropy(
                            cat_logits, 
                            x_cat[:, i], 
                            reduction='sum'
                        )
                    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = num_loss + cat_loss + KLD
                else:
                    # Standard tabular model - combine numerical and categorical features
                    feature_list = []
                    
                    if 'numerical_features' in batch:
                        feature_list.append(batch['numerical_features'].to(device))
                    
                    if 'categorical_features' in batch:
                        # Convert categorical indices to float and add to features
                        # Categorical features are indices, convert to float for model
                        cat_features = batch['categorical_features'].to(device).float()
                        feature_list.append(cat_features)
                    
                    # Concatenate all features
                    x = torch.cat(feature_list, dim=1) if len(feature_list) > 1 else feature_list[0]
                    
                    recon_x, mu, logvar = model(x, condition)
                    loss = _vae_loss_function(recon_x, x, mu, logvar)
            elif 'input_ids' in batch:
                # Text model
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                recon_logits, mu, logvar = model(input_ids, attention_mask, condition)
                # For text, we use cross-entropy loss on the logits
                # Reshape for cross-entropy: [batch_size * seq_len, vocab_size]
                # Note: This is a simplified version - full implementation would handle sequence properly
                RECON_LOSS = F.cross_entropy(
                    recon_logits.view(-1, recon_logits.size(-1)),
                    input_ids.view(-1),
                    reduction='sum'
                )
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = RECON_LOSS + KLD
            else:
                raise ValueError("No recognized features in batch")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f'Epoch {epoch + 1}/{train_params["epochs"]}, Average Loss: {avg_loss:.4f}')
    
    # Save Full Model (needed for anomaly detection)
    model_path = config['output_model_path']
    torch.save(model.state_dict(), model_path)
    print(f"Saved full model to {model_path}")
    
    return model_path

