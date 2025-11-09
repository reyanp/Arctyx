import os
import json
import tempfile
import shutil
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from snorkel.labeling import labeling_function

# Suppress sklearn warnings about feature names
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# Suppress other common warnings
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

try:
    # Try absolute imports (when running from project root)
    from DataFoundry.labeler import create_labels
    from DataFoundry.trainer import train_model
    from DataFoundry.generator import generate_data, generate_data_as_dataset, GeneratedTensorDataset
    from DataFoundry.evaluator import find_anomalies
    from DataFoundry.utils import Preprocessor, CustomDataset
except ImportError:
    # Fall back to relative imports (when running from within DataFoundry directory)
    from labeler import create_labels
    from trainer import train_model
    from generator import generate_data, generate_data_as_dataset, GeneratedTensorDataset
    from evaluator import find_anomalies
    from utils import Preprocessor, CustomDataset


class DataFoundryTester:
    """
    Comprehensive test class for DataFoundry's main functionalities:
    1. Data labeling (weak supervision with Snorkel)
    2. Semi-supervised training (cVAE training)
    3. Data synthesis (generating synthetic data)
    4. Anomaly detection (using trained cVAE models)
    """
    
    def __init__(self, test_dir=None):
        """
        Initialize the tester with a temporary directory for test files.
        
        Args:
            test_dir: Optional directory for test files. If None, creates a temp directory.
        """
        if test_dir is None:
            self.test_dir = tempfile.mkdtemp(prefix='datafoundry_test_')
        else:
            self.test_dir = test_dir
            os.makedirs(self.test_dir, exist_ok=True)
        
        self.test_results = {}
        print(f"Test directory: {self.test_dir}")
    
    def cleanup(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"Cleaned up test directory: {self.test_dir}")
    
    def create_sample_tabular_data(self, num_samples=100, output_path=None, include_categorical=False):
        """
        Create sample tabular data for testing.
        
        Args:
            num_samples: Number of samples to generate
            output_path: Path to save the data. If None, uses test_dir.
            include_categorical: If True, includes categorical columns
        
        Returns:
            Path to created data file
        """
        if output_path is None:
            suffix = '_with_categorical' if include_categorical else ''
            output_path = os.path.join(self.test_dir, f'sample_tabular_data{suffix}.parquet')
        
        # Generate synthetic tabular data
        np.random.seed(42)
        data = {
            'feature1': np.random.randn(num_samples),
            'feature2': np.random.randn(num_samples),
            'feature3': np.random.randn(num_samples),
            'condition': np.random.choice([0, 1], size=num_samples)
        }
        
        # Add categorical columns if requested
        if include_categorical:
            data['category1'] = np.random.choice(['A', 'B', 'C'], size=num_samples)
            data['category2'] = np.random.choice(['X', 'Y'], size=num_samples)
        
        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
        print(f"Created sample tabular data: {output_path} ({len(df)} samples, {len(df.columns)} columns)")
        return output_path
    
    def create_sample_text_data(self, num_samples=50, output_path=None):
        """
        Create sample text data for testing.
        
        Args:
            num_samples: Number of samples to generate
            output_path: Path to save the data. If None, uses test_dir.
        
        Returns:
            Path to created data file
        """
        if output_path is None:
            output_path = os.path.join(self.test_dir, 'sample_text_data.parquet')
        
        # Generate synthetic text data
        np.random.seed(42)
        texts = [
            f"This is sample text {i} with some content about topic {np.random.choice(['A', 'B'])}"
            for i in range(num_samples)
        ]
        
        data = {
            'text': texts,
            'condition': np.random.choice([0, 1], size=num_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
        print(f"Created sample text data: {output_path} ({len(df)} samples)")
        return output_path
    
    def create_labeling_functions(self, include_categorical=False):
        """
        Create sample labeling functions for testing.
        
        Args:
            include_categorical: If True, includes labeling functions for categorical data
        
        Returns:
            List of LabelingFunction objects
        """
        @labeling_function()
        def lf_positive_feature1(x):
            """Label positive if feature1 > 0.5"""
            return 1 if x['feature1'] > 0.5 else 0
        
        @labeling_function()
        def lf_negative_feature1(x):
            """Label negative if feature1 < -0.5"""
            return 0 if x['feature1'] < -0.5 else 1
        
        @labeling_function()
        def lf_positive_feature2(x):
            """Label positive if feature2 > 0"""
            return 1 if x['feature2'] > 0 else 0
        
        lfs = [lf_positive_feature1, lf_negative_feature1, lf_positive_feature2]
        
        # Add categorical labeling functions if requested
        if include_categorical:
            @labeling_function()
            def lf_category_a(x):
                """Label positive if category1 is 'A'"""
                return 1 if x.get('category1') == 'A' else 0
            
            @labeling_function()
            def lf_category_x(x):
                """Label positive if category2 is 'X'"""
                return 1 if x.get('category2') == 'X' else 0
            
            lfs.extend([lf_category_a, lf_category_x])
        
        return lfs
    
    def create_text_labeling_functions(self):
        """
        Create sample labeling functions for text data testing.
        
        Returns:
            List of LabelingFunction objects
        """
        @labeling_function()
        def lf_contains_topic_a(x):
            """Label positive if text contains 'topic A'"""
            return 1 if 'topic A' in str(x.get('text', '')).lower() else 0
        
        @labeling_function()
        def lf_contains_topic_b(x):
            """Label positive if text contains 'topic B'"""
            return 1 if 'topic B' in str(x.get('text', '')).lower() else 0
        
        @labeling_function()
        def lf_has_sample(x):
            """Label positive if text contains 'sample'"""
            return 1 if 'sample' in str(x.get('text', '')).lower() else 0
        
        return [lf_contains_topic_a, lf_contains_topic_b, lf_has_sample]
    
    def test_data_labeling(self, data_path=None, is_text=False, include_categorical=False):
        """
        Test 1: Data labeling with weak supervision (Snorkel).
        
        Args:
            data_path: Path to unlabeled data. If None, creates sample data.
            is_text: Whether the data is text data (uses different labeling functions)
            include_categorical: Whether to include categorical columns in tabular data
        
        Returns:
            Path to labeled data file
        """
        print("\n" + "="*60)
        print("TEST 1: Data Labeling (Weak Supervision)")
        print("="*60)
        
        if data_path is None:
            if is_text:
                data_path = self.create_sample_text_data(num_samples=30)
            else:
                data_path = self.create_sample_tabular_data(num_samples=100, include_categorical=include_categorical)
        
        if is_text:
            labeling_functions = self.create_text_labeling_functions()
        else:
            labeling_functions = self.create_labeling_functions(include_categorical=include_categorical)
        output_path = os.path.join(self.test_dir, 'labeled_data.parquet')
        
        try:
            labeled_path = create_labels(
                data_path=data_path,
                labeling_functions_list=labeling_functions,
                output_path=output_path
            )
            
            # Verify output
            df_labeled = pd.read_parquet(labeled_path)
            assert 'label_probability' in df_labeled.columns, "Missing label_probability column"
            assert len(df_labeled) > 0, "Labeled data is empty"
            assert df_labeled['label_probability'].between(0, 1).all(), "Invalid probabilities"
            
            print(f"✓ Data labeling successful!")
            print(f"  - Labeled {len(df_labeled)} samples")
            print(f"  - Probability range: [{df_labeled['label_probability'].min():.3f}, {df_labeled['label_probability'].max():.3f}]")
            
            self.test_results['labeling'] = {'success': True, 'output_path': labeled_path}
            return labeled_path
            
        except Exception as e:
            print(f"✗ Data labeling failed: {e}")
            self.test_results['labeling'] = {'success': False, 'error': str(e)}
            raise
    
    def test_semi_supervised_training(self, labeled_data_path=None, model_type='tabular'):
        """
        Test 2: Semi-supervised training (cVAE training).
        
        Args:
            labeled_data_path: Path to labeled data. If None, creates and labels sample data.
            model_type: 'tabular' or 'text'
        
        Returns:
            Path to trained decoder
        """
        print("\n" + "="*60)
        print(f"TEST 2: Semi-Supervised Training ({model_type.upper()} cVAE)")
        print("="*60)
        
        # Create labeled data if not provided
        if labeled_data_path is None:
            labeled_data_path = self.test_data_labeling()
        
        # Create config for training
        if model_type == 'tabular':
            # Check if data has categorical columns by reading a sample
            df_sample = pd.read_parquet(labeled_data_path).head(1)
            has_categorical = 'category1' in df_sample.columns and 'category2' in df_sample.columns
            
            feature_cols = {
                'numerical_cols': ['feature1', 'feature2', 'feature3']
            }
            
            # Add categorical columns if they exist
            if has_categorical:
                feature_cols['categorical_cols'] = ['category1', 'category2']
                # input_dim will be auto-calculated (3 numerical + 2 categorical = 5)
                input_dim = None  # Let it auto-calculate
            else:
                input_dim = 3  # Only numerical
            
            config = {
                'data_path': labeled_data_path,
                'preprocessor_path': os.path.join(self.test_dir, 'preprocessor.pkl'),
                'output_model_path': os.path.join(self.test_dir, 'decoder_tabular.pth'),
                'model_params': {
                    'model_template': 'tabular_cvae.py',
                    'model_class_name': 'TabularCVAE',
                    'latent_dim': 8,
                    'condition_dim': 1,
                    'encoder_hidden_layers': [16, 12],
                    'decoder_hidden_layers': [12, 16],
                    'feature_cols': feature_cols,
                    'condition_cols': ['condition']
                },
                'training_params': {
                    'batch_size': 16,
                    'learning_rate': 0.001,
                    'epochs': 5  # Small number for testing
                }
            }
            
            # Only set input_dim if not auto-calculating
            if input_dim is not None:
                config['model_params']['input_dim'] = input_dim
        else:  # text
            config = {
                'data_path': labeled_data_path,
                'preprocessor_path': os.path.join(self.test_dir, 'preprocessor_text.pkl'),
                'output_model_path': os.path.join(self.test_dir, 'decoder_text.pth'),
                'model_params': {
                    'model_template': 'text_cvae.py',
                    'model_class_name': 'TextCVAE',
                    'text_model': 'distilbert-base-uncased',  # Smaller model
                    'latent_dim': 8,
                    'condition_dim': 1,
                    'max_length': 64,
                    'decoder_hidden_layers': [256, 128],
                    'feature_cols': {
                        'text_col': 'text'
                    },
                    'condition_cols': ['condition']
                },
                'training_params': {
                    'batch_size': 8,
                    'learning_rate': 0.0001,
                    'epochs': 3  # Small number for testing
                }
            }
        
        config_path = os.path.join(self.test_dir, f'config_{model_type}.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        try:
            # Test PyTorch Dataset compatibility
            df = pd.read_parquet(labeled_data_path)
            preprocessor = Preprocessor(config)
            preprocessor.fit(df)
            
            dataset = CustomDataset(df, preprocessor)
            assert len(dataset) == len(df), "Dataset length mismatch"
            
            # Test DataLoader
            data_loader = DataLoader(dataset, batch_size=config['training_params']['batch_size'], shuffle=True)
            sample_batch = next(iter(data_loader))
            assert 'condition' in sample_batch, "Missing condition in batch"
            print(f"✓ PyTorch Dataset compatibility verified")
            print(f"  - Dataset size: {len(dataset)}")
            print(f"  - Batch keys: {list(sample_batch.keys())}")
            
            # Train model
            model_path = train_model(config_path)
            
            # Verify model was saved
            assert os.path.exists(model_path), "Model file not created"
            model_state = torch.load(model_path, map_location='cpu')
            assert len(model_state) > 0, "Model state is empty"
            
            print(f"✓ Training successful!")
            print(f"  - Model saved to: {model_path}")
            
            self.test_results[f'training_{model_type}'] = {
                'success': True,
                'model_path': model_path,
                'config_path': config_path
            }
            return model_path, config_path
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results[f'training_{model_type}'] = {'success': False, 'error': str(e)}
            raise
    
    def test_data_synthesis(self, model_path=None, config_path=None, model_type='tabular', 
                           num_samples=20, output_format='parquet'):
        """
        Test 3: Data synthesis (generating synthetic data).
        
        Args:
            model_path: Path to trained model. If None, trains a model first.
            config_path: Path to config file. If None, trains a model first.
            model_type: 'tabular' or 'text'
            num_samples: Number of samples to generate
            output_format: 'parquet' (default, recommended), 'pt', or 'csv'
        
        Returns:
            Path to generated data file
        """
        print("\n" + "="*60)
        print(f"TEST 3: Data Synthesis ({model_type.upper()} model, {output_format.upper()} format)")
        print("="*60)
        
        # Train model if not provided
        if model_path is None or config_path is None:
            labeled_path = self.test_data_labeling()
            model_path, config_path = self.test_semi_supervised_training(
                labeled_data_path=labeled_path,
                model_type=model_type
            )
        
        # Determine file extension
        ext_map = {'parquet': 'parquet', 'pt': 'pt', 'csv': 'csv'}
        ext = ext_map.get(output_format, 'parquet')
        output_path = os.path.join(
            self.test_dir,
            f'generated_{model_type}_{output_format}.{ext}'
        )
        
        try:
            generated_path = generate_data(
                model_path=model_path,
                config_path=config_path,
                label=1,  # Generate samples with label 1
                num_to_generate=num_samples,
                output_path=output_path,
                output_format=output_format
            )
            
            # Verify output based on format
            if output_format == 'parquet':
                # Load Parquet (recommended format for internal use)
                df_generated = pd.read_parquet(generated_path)
                assert len(df_generated) == num_samples, "Wrong number of samples in Parquet"
                assert len(df_generated.columns) > 0, "Parquet has no columns"
                print(f"✓ Parquet output verified")
                print(f"  - Rows: {len(df_generated)}")
                print(f"  - Columns: {list(df_generated.columns)}")
                
                # Verify categorical columns if they exist
                categorical_cols = [col for col in df_generated.columns if col.startswith('category')]
                if categorical_cols:
                    print(f"  - Categorical columns: {categorical_cols}")
                    # Verify categorical values are valid strings (not indices)
                    for col in categorical_cols:
                        assert df_generated[col].dtype == 'object' or df_generated[col].dtype.name == 'category', \
                            f"Categorical column {col} should be string/category type, got {df_generated[col].dtype}"
                        print(f"    - {col} values: {df_generated[col].unique()[:5]}")  # Show first 5 unique values
                
                # Test loading Parquet as PyTorch Dataset (using preprocessor)
                # Load config to get preprocessor path
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                from DataFoundry.generator import load_parquet_as_dataset
                gen_dataset = load_parquet_as_dataset(
                    generated_path,
                    config_dict['preprocessor_path']
                )
                gen_loader = DataLoader(gen_dataset, batch_size=8)
                sample = next(iter(gen_loader))
                print(f"✓ Parquet data works with PyTorch DataLoader (via preprocessor)")
                print(f"  - Batch keys: {list(sample.keys())}")
                
            elif output_format == 'pt':
                # Load PyTorch tensor
                generated_tensors = torch.load(generated_path, map_location='cpu')
                assert isinstance(generated_tensors, torch.Tensor), "Output is not a tensor"
                assert generated_tensors.shape[0] == num_samples, "Wrong number of samples"
                print(f"✓ PyTorch tensor output verified")
                print(f"  - Shape: {generated_tensors.shape}")
                print(f"  - Dtype: {generated_tensors.dtype}")
                
                # Test PyTorch Dataset compatibility with generated data
                gen_dataset = GeneratedTensorDataset(generated_path)
                gen_loader = DataLoader(gen_dataset, batch_size=8)
                sample = next(iter(gen_loader))
                print(f"✓ Generated data works with PyTorch DataLoader")
                print(f"  - Batch shape: {sample.shape}")
                
                # Test generate_data_as_dataset function
                gen_dataset_direct = generate_data_as_dataset(
                    model_path=model_path,
                    config_path=config_path,
                    label=0,  # Generate with different label
                    num_to_generate=10
                )
                assert len(gen_dataset_direct) == 10, "Dataset length mismatch"
                print(f"✓ Direct dataset generation works")
                print(f"  - Dataset size: {len(gen_dataset_direct)}")
                
            else:  # csv
                # Load CSV (not recommended for internal use)
                df_generated = pd.read_csv(generated_path)
                assert len(df_generated) == num_samples, "Wrong number of samples in CSV"
                assert len(df_generated.columns) > 0, "CSV has no columns"
                print(f"✓ CSV output verified (not recommended for internal use)")
                print(f"  - Rows: {len(df_generated)}")
                print(f"  - Columns: {list(df_generated.columns)}")
            
            print(f"✓ Data synthesis successful!")
            print(f"  - Generated {num_samples} samples")
            print(f"  - Output: {generated_path}")
            
            self.test_results[f'synthesis_{model_type}_{output_format}'] = {
                'success': True,
                'output_path': generated_path
            }
            return generated_path
            
        except Exception as e:
            print(f"✗ Data synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results[f'synthesis_{model_type}_{output_format}'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    def test_anomaly_detection(self, model_path=None, config_path=None, data_path_to_scan=None, 
                               model_type='tabular'):
        """
        Test 4: Anomaly detection using trained cVAE model.
        
        Args:
            model_path: Path to trained model. If None, trains a model first.
            config_path: Path to config file. If None, trains a model first.
            data_path_to_scan: Path to data to scan for anomalies. If None, uses labeled training data.
            model_type: 'tabular' or 'text'
        
        Returns:
            Path to output file with anomaly scores
        """
        print("\n" + "="*60)
        print(f"TEST 4: Anomaly Detection ({model_type.upper()} model)")
        print("="*60)
        
        # Train model if not provided
        if model_path is None or config_path is None:
            labeled_path = self.test_data_labeling()
            model_path, config_path = self.test_semi_supervised_training(
                labeled_data_path=labeled_path,
                model_type=model_type
            )
        
        # Use provided data or training data for scanning
        if data_path_to_scan is None:
            # Load config to get training data path
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            data_path_to_scan = config_dict['data_path']
        
        # Load config to get preprocessor path
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        output_path = os.path.join(
            self.test_dir,
            f'anomaly_scores_{model_type}.parquet'
        )
        
        try:
            anomaly_path = find_anomalies(
                config_path=config_path,
                model_path=model_path,
                preprocessor_path=config_dict['preprocessor_path'],
                data_path_to_scan=data_path_to_scan,
                output_path=output_path
            )
            
            # Verify output
            df_anomalies = pd.read_parquet(anomaly_path)
            assert 'anomaly_score' in df_anomalies.columns, "Missing anomaly_score column"
            assert len(df_anomalies) > 0, "Anomaly scores DataFrame is empty"
            
            # Verify scores are valid (non-negative)
            assert (df_anomalies['anomaly_score'] >= 0).all(), "Anomaly scores should be non-negative"
            
            # Verify DataFrame is sorted by anomaly_score (descending)
            assert df_anomalies['anomaly_score'].is_monotonic_decreasing, \
                "Anomaly scores should be sorted in descending order"
            
            # Verify all original columns are preserved
            original_df = pd.read_parquet(data_path_to_scan)
            for col in original_df.columns:
                assert col in df_anomalies.columns, f"Missing original column: {col}"
            
            print(f"✓ Anomaly detection successful!")
            print(f"  - Scanned {len(df_anomalies)} samples")
            print(f"  - Anomaly score range: [{df_anomalies['anomaly_score'].min():.4f}, {df_anomalies['anomaly_score'].max():.4f}]")
            print(f"  - Mean anomaly score: {df_anomalies['anomaly_score'].mean():.4f}")
            print(f"  - Top 5 anomalies:")
            for idx, row in df_anomalies.head(5).iterrows():
                print(f"    - Score: {row['anomaly_score']:.4f}")
            
            # Verify we can load the results as a PyTorch Dataset
            from DataFoundry.generator import load_parquet_as_dataset
            anomaly_dataset = load_parquet_as_dataset(
                anomaly_path,
                config_dict['preprocessor_path']
            )
            assert len(anomaly_dataset) == len(df_anomalies), "Dataset length mismatch"
            print(f"✓ Anomaly results work with PyTorch DataLoader")
            
            self.test_results[f'anomaly_{model_type}'] = {
                'success': True,
                'output_path': anomaly_path,
                'max_score': float(df_anomalies['anomaly_score'].max()),
                'min_score': float(df_anomalies['anomaly_score'].min()),
                'mean_score': float(df_anomalies['anomaly_score'].mean())
            }
            return anomaly_path
            
        except Exception as e:
            print(f"✗ Anomaly detection failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results[f'anomaly_{model_type}'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    def test_model_zoo_models(self):
        """
        Test 5: Model Zoo - Test advanced generative models.
        Tests model instantiation and forward passes for:
        - MixedDataCVAE
        - CTGAN (Generator and Discriminator)
        - TabularVAE_GMM
        """
        print("\n" + "="*60)
        print("TEST 5: Model Zoo - Advanced Generative Models")
        print("="*60)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 8
        
        try:
            # Test 1: MixedDataCVAE
            print("\n--- Testing MixedDataCVAE ---")
            import importlib
            mixed_model_module = importlib.import_module('DataFoundry.models.mixed_data_cvae')
            MixedDataCVAE = getattr(mixed_model_module, 'MixedDataCVAE')
            
            mixed_config = {
                'numerical_dim': 3,
                'categorical_embed_dims': [(3, 4), (2, 3)],  # 2 categorical features
                'latent_dim': 8,
                'condition_dim': 1,
                'encoder_hidden_layers': [16, 12],
                'decoder_hidden_layers': [12, 16]
            }
            
            mixed_model = MixedDataCVAE(mixed_config).to(device)
            
            # Test forward pass
            x_num = torch.randn(batch_size, 3).to(device)
            # Create categorical indices: first feature has 3 categories (0-2), second has 2 categories (0-1)
            x_cat = torch.stack([
                torch.randint(0, 3, (batch_size,)),  # First categorical: 3 categories
                torch.randint(0, 2, (batch_size,))   # Second categorical: 2 categories
            ], dim=1).to(device)
            c = torch.randn(batch_size, 1).to(device)
            
            recon_num, recon_cat_logits, mu, logvar = mixed_model(x_num, x_cat, c)
            
            assert recon_num.shape == (batch_size, 3), "Wrong numerical reconstruction shape"
            assert len(recon_cat_logits) == 2, "Wrong number of categorical logits"
            assert recon_cat_logits[0].shape == (batch_size, 3), "Wrong first categorical logits shape"
            assert recon_cat_logits[1].shape == (batch_size, 2), "Wrong second categorical logits shape"
            assert mu.shape == (batch_size, 8), "Wrong mu shape"
            assert logvar.shape == (batch_size, 8), "Wrong logvar shape"
            
            print(f"✓ MixedDataCVAE instantiated and forward pass successful")
            print(f"  - Numerical reconstruction: {recon_num.shape}")
            print(f"  - Categorical logits: {[logits.shape for logits in recon_cat_logits]}")
            
            self.test_results['model_zoo_mixed_data_cvae'] = {'success': True}
            
            # Test 2: CTGAN Generator and Discriminator
            print("\n--- Testing CTGAN (Generator & Discriminator) ---")
            ctgan_module = importlib.import_module('DataFoundry.models.tabular_ctgan')
            CTGAN_Generator = getattr(ctgan_module, 'CTGAN_Generator')
            CTGAN_Discriminator = getattr(ctgan_module, 'CTGAN_Discriminator')
            
            gen_config = {
                'latent_dim': 8,
                'condition_dim': 1,
                'output_dim': 5,
                'gen_hidden_layers': [16, 16]
            }
            
            disc_config = {
                'input_dim': 5,
                'condition_dim': 1,
                'disc_hidden_layers': [16, 16]
            }
            
            generator = CTGAN_Generator(gen_config).to(device)
            discriminator = CTGAN_Discriminator(disc_config).to(device)
            
            # Test Generator forward pass
            z = torch.randn(batch_size, 8).to(device)
            c = torch.randn(batch_size, 1).to(device)
            fake_data = generator(z, c)
            
            assert fake_data.shape == (batch_size, 5), "Wrong generator output shape"
            assert torch.all(fake_data >= -1) and torch.all(fake_data <= 1), "Generator output not in [-1, 1] range"
            
            # Test Discriminator forward pass
            real_data = torch.randn(batch_size, 5).to(device)
            real_scores = discriminator(real_data, c)
            fake_scores = discriminator(fake_data, c)
            
            assert real_scores.shape == (batch_size, 1), "Wrong discriminator output shape for real data"
            assert fake_scores.shape == (batch_size, 1), "Wrong discriminator output shape for fake data"
            
            print(f"✓ CTGAN Generator and Discriminator instantiated and forward pass successful")
            print(f"  - Generator output: {fake_data.shape}")
            print(f"  - Discriminator real scores: {real_scores.shape}")
            print(f"  - Discriminator fake scores: {fake_scores.shape}")
            
            self.test_results['model_zoo_ctgan'] = {'success': True}
            
            # Test 3: TabularVAE_GMM
            print("\n--- Testing TabularVAE_GMM ---")
            gmm_module = importlib.import_module('DataFoundry.models.tabular_vae_gmm')
            TabularVAE_GMM = getattr(gmm_module, 'TabularVAE_GMM')
            
            gmm_config = {
                'input_dim': 5,
                'latent_dim': 8,
                'condition_dim': 1,
                'n_clusters': 3,
                'encoder_hidden_layers': [16, 12],
                'decoder_hidden_layers': [12, 16]
            }
            
            gmm_model = TabularVAE_GMM(gmm_config).to(device)
            
            # Test forward pass
            x = torch.randn(batch_size, 5).to(device)
            c = torch.randn(batch_size, 1).to(device)
            recon_x, z_mu, z_logvar = gmm_model(x, c)
            
            assert recon_x.shape == (batch_size, 5), "Wrong reconstruction shape"
            assert z_mu.shape == (batch_size, 8), "Wrong z_mu shape"
            assert z_logvar.shape == (batch_size, 8), "Wrong z_logvar shape"
            
            # Test GMM parameters
            gmm_pi, gmm_means, gmm_logvars = gmm_model.get_gmm_params()
            assert gmm_pi.shape == (3,), "Wrong GMM pi shape"
            assert gmm_means.shape == (3, 8), "Wrong GMM means shape"
            assert gmm_logvars.shape == (3, 8), "Wrong GMM logvars shape"
            assert torch.allclose(gmm_pi.sum(), torch.tensor(1.0)), "GMM pi should sum to 1"
            
            print(f"✓ TabularVAE_GMM instantiated and forward pass successful")
            print(f"  - Reconstruction: {recon_x.shape}")
            print(f"  - GMM clusters: {gmm_pi.shape[0]}")
            print(f"  - GMM means: {gmm_means.shape}")
            
            self.test_results['model_zoo_vae_gmm'] = {'success': True}
            
            print(f"\n✓ All Model Zoo tests passed!")
            return True
            
        except Exception as e:
            print(f"✗ Model Zoo test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['model_zoo'] = {'success': False, 'error': str(e)}
            raise
    
    def run_all_tests(self):
        """
        Run all main functionality tests with all available models:
        1. Data labeling (weak supervision)
        2. Semi-supervised training (cVAE)
        3. Data synthesis (generating synthetic data)
        4. Anomaly detection (scoring data for anomalies)
        5. Model Zoo - Advanced generative models (MixedDataCVAE, CTGAN, TabularVAE_GMM)
        
        Returns:
            Dictionary of test results
        """
        print("\n" + "="*60)
        print("RUNNING ALL DATAFOUNDRY TESTS")
        print("="*60)
        
        try:
            # Test 1: Data Labeling (Numerical only)
            labeled_path = self.test_data_labeling()
            
            # Test 2: Semi-Supervised Training (Tabular - Numerical only)
            model_tabular, config_tabular = self.test_semi_supervised_training(
                labeled_data_path=labeled_path,
                model_type='tabular'
            )
            
            # Test 3: Data Synthesis (Tabular, all formats - Numerical only)
            # Parquet (default, recommended for internal use)
            self.test_data_synthesis(
                model_path=model_tabular,
                config_path=config_tabular,
                model_type='tabular',
                num_samples=20,
                output_format='parquet'
            )
            
            # PyTorch tensor format
            self.test_data_synthesis(
                model_path=model_tabular,
                config_path=config_tabular,
                model_type='tabular',
                num_samples=20,
                output_format='pt'
            )
            
            # CSV (not recommended, but tested for compatibility)
            self.test_data_synthesis(
                model_path=model_tabular,
                config_path=config_tabular,
                model_type='tabular',
                num_samples=20,
                output_format='csv'
            )
            
            # Test 4: Anomaly Detection (Tabular - Numerical only)
            self.test_anomaly_detection(
                model_path=model_tabular,
                config_path=config_tabular,
                model_type='tabular'
            )
            
            # Test with Categorical data
            print("\n" + "="*60)
            print("TESTING WITH CATEGORICAL DATA")
            print("="*60)
            
            # Test 1: Data Labeling (with categorical)
            labeled_path_categorical = self.test_data_labeling(include_categorical=True)
            
            # Test 2: Semi-Supervised Training (Tabular - with categorical)
            model_tabular_cat, config_tabular_cat = self.test_semi_supervised_training(
                labeled_data_path=labeled_path_categorical,
                model_type='tabular'
            )
            
            # Test 3: Data Synthesis (Tabular with categorical, Parquet format)
            self.test_data_synthesis(
                model_path=model_tabular_cat,
                config_path=config_tabular_cat,
                model_type='tabular',
                num_samples=20,
                output_format='parquet'
            )
            
            # Test 4: Anomaly Detection (Tabular with categorical)
            self.test_anomaly_detection(
                model_path=model_tabular_cat,
                config_path=config_tabular_cat,
                model_type='tabular'
            )
            
            # Test with Text model (if transformers available)
            try:
                # Test 2: Semi-Supervised Training (Text)
                text_data_path = self.create_sample_text_data(num_samples=30)
                text_labeled_path = self.test_data_labeling(data_path=text_data_path, is_text=True)
                
                model_text, config_text = self.test_semi_supervised_training(
                    labeled_data_path=text_labeled_path,
                    model_type='text'
                )
                
                # Test 3: Data Synthesis (Text)
                # Parquet (default, recommended)
                self.test_data_synthesis(
                    model_path=model_text,
                    config_path=config_text,
                    model_type='text',
                    num_samples=10,
                    output_format='parquet'
                )
                
                # PyTorch tensor format
                self.test_data_synthesis(
                    model_path=model_text,
                    config_path=config_text,
                    model_type='text',
                    num_samples=10,
                    output_format='pt'
                )
                
                # Test 4: Anomaly Detection (Text)
                self.test_anomaly_detection(
                    model_path=model_text,
                    config_path=config_text,
                    model_type='text'
                )
                
            except Exception as e:
                print(f"\n⚠ Text model tests skipped: {e}")
                print("  (This is expected if transformers library is not available)")
            
            # Test 5: Model Zoo - Advanced Generative Models
            print("\n" + "="*60)
            print("TESTING MODEL ZOO")
            print("="*60)
            self.test_model_zoo_models()
            
            # Print summary
            print("\n" + "="*60)
            print("TEST SUMMARY")
            print("="*60)
            for test_name, result in self.test_results.items():
                status = "✓ PASS" if result.get('success', False) else "✗ FAIL"
                print(f"{status}: {test_name}")
            
            all_passed = all(r.get('success', False) for r in self.test_results.values())
            print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
            
            return self.test_results
            
        except Exception as e:
            print(f"\n✗ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return self.test_results


if __name__ == '__main__':
    # Run tests
    tester = DataFoundryTester()
    try:
        results = tester.run_all_tests()
    finally:
        # Optionally cleanup (comment out to keep test files)
        # tester.cleanup()
        pass

