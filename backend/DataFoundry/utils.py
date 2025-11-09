import warnings
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import AutoTokenizer
from torch.utils.data import Dataset

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')


class Preprocessor:
    """
    Centralizes all data transformation logic for consistency between training and generation.
    Converts pandas DataFrame rows to PyTorch tensors and vice versa.
    """
    
    def __init__(self, config):
        """
        Initializes all pre-processing objects based on the config.
        
        Args:
            config: The config dictionary. Expected to have 'model_params' key.
        """
        self.config = config["model_params"]
        
        # Initialize tokenizer if text model is specified
        if "text_model" in self.config:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["text_model"])
        
        # Initialize scaler if numerical columns are specified
        if "numerical_cols" in self.config.get("feature_cols", {}):
            self.scaler = StandardScaler()
        
        # Initialize label encoders for categorical columns if specified
        if "categorical_cols" in self.config.get("feature_cols", {}):
            categorical_cols = self.config["feature_cols"]["categorical_cols"]
            self.label_encoders = {col: LabelEncoder() for col in categorical_cols}
        else:
            self.label_encoders = {}
        
        # Store condition and feature column configurations
        self.condition_cols = self.config["condition_cols"]
        self.feature_cols = self.config["feature_cols"]
    
    def fit(self, dataframe):
        """
        Fits pre-processors that require fitting (e.g., StandardScaler, LabelEncoder).
        
        Args:
            dataframe: pandas DataFrame to fit the preprocessors on.
        """
        if hasattr(self, 'scaler'):
            nums_data = dataframe[self.feature_cols["numerical_cols"]]
            self.scaler.fit(nums_data)
        
        # Fit label encoders for categorical columns
        if hasattr(self, 'label_encoders') and self.label_encoders:
            for col, encoder in self.label_encoders.items():
                encoder.fit(dataframe[col].astype(str))  # Convert to string to handle any type
    
    def transform(self, dataframe_row):
        """
        Converts a single row of a pandas DataFrame into a dictionary of PyTorch tensors.
        This is used by the PyTorch Dataset in trainer.py.
        
        Args:
            dataframe_row: A single row (pandas.Series) from a DataFrame.
            
        Returns:
            Dictionary of torch.Tensor objects with keys like 'input_ids', 'attention_mask',
            'numerical_features', and 'condition'.
        """
        tensors = {}
        
        # Handle text features if tokenizer exists
        if hasattr(self, 'tokenizer'):
            text = dataframe_row[self.feature_cols["text_col"]]
            tokenized = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.config.get('max_length', 128),
                return_tensors='pt'
            )
            tensors['input_ids'] = tokenized['input_ids'].squeeze(0)
            tensors['attention_mask'] = tokenized['attention_mask'].squeeze(0)
        
        # Handle numerical features if scaler exists
        if hasattr(self, 'scaler'):
            nums = dataframe_row[self.feature_cols["numerical_cols"]].values.astype(float)
            scaled_nums = self.scaler.transform([nums])
            tensors['numerical_features'] = torch.tensor(scaled_nums, dtype=torch.float32).squeeze(0)
        
        # Handle categorical features if label encoders exist
        if hasattr(self, 'label_encoders') and self.label_encoders:
            categorical_indices = []
            for col in self.feature_cols.get("categorical_cols", []):
                if col in self.label_encoders:
                    # Get the encoded index for this category
                    category_value = str(dataframe_row[col])
                    # Handle unseen categories by using the most common category or 0
                    try:
                        encoded_idx = self.label_encoders[col].transform([category_value])[0]
                    except ValueError:
                        # Unseen category - use 0 as default
                        encoded_idx = 0
                    categorical_indices.append(encoded_idx)
            
            if categorical_indices:
                tensors['categorical_features'] = torch.tensor(
                    categorical_indices, 
                    dtype=torch.long
                )
        
        # Handle condition (label) data
        condition_data = dataframe_row[self.condition_cols].values.astype(float)
        tensors['condition'] = torch.tensor(condition_data, dtype=torch.float32)
        
        return tensors
    
    def inverse_transform(self, tensors):
        """
        Converts the output tensors from the decoder back into a human-readable format.
        
        Args:
            tensors: A dictionary of output tensors from the decoder.
                    Expected keys: 'output_text_ids' (optional), 'output_numerical' (optional),
                    'output_categorical' (optional)
        
        Returns:
            pandas.DataFrame of the synthetic data.
        """
        df = pd.DataFrame()
        
        # Handle numerical features
        if hasattr(self, 'scaler') and 'output_numerical' in tensors:
            scaled_nums = tensors['output_numerical'].cpu().numpy()
            nums = self.scaler.inverse_transform(scaled_nums)
            df[self.feature_cols["numerical_cols"]] = nums
        
        # Handle categorical features
        if hasattr(self, 'label_encoders') and self.label_encoders and 'output_categorical' in tensors:
            categorical_indices = tensors['output_categorical'].cpu().numpy()
            categorical_cols = self.feature_cols.get("categorical_cols", [])
            
            # Handle both 1D and 2D arrays
            if len(categorical_indices.shape) == 1:
                # Single sample
                categorical_indices = categorical_indices.reshape(1, -1)
            
            for i, col in enumerate(categorical_cols):
                if col in self.label_encoders:
                    # Convert indices back to categories
                    # Round and clamp indices to valid range
                    indices = categorical_indices[:, i]
                    indices = np.round(indices).astype(int)
                    indices = np.clip(indices, 0, len(self.label_encoders[col].classes_) - 1)
                    categories = self.label_encoders[col].inverse_transform(indices)
                    df[col] = categories
        
        # Handle text features
        if hasattr(self, 'tokenizer') and 'output_text_ids' in tensors:
            df['text'] = self.tokenizer.batch_decode(
                tensors['output_text_ids'],
                skip_special_tokens=True
            )
        
        return df


class CustomDataset(Dataset):
    """
    PyTorch Dataset that uses Preprocessor to transform DataFrame rows.
    Reusable across trainer, generator, and evaluator modules.
    """
    
    def __init__(self, df, preprocessor):
        """
        Args:
            df: pandas DataFrame
            preprocessor: Preprocessor instance
        """
        self.df = df
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.preprocessor.transform(self.df.iloc[idx])