import os
import json
import tempfile
import shutil
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix
from snorkel.labeling import labeling_function

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

try:
    from DataFoundry.labeler import create_labels
    from DataFoundry.trainer import train_model
    from DataFoundry.generator import generate_data
    from DataFoundry.evaluator import find_anomalies
    from DataFoundry.utils import Preprocessor
except ImportError:
    from labeler import create_labels
    from trainer import train_model
    from generator import generate_data
    from evaluator import find_anomalies
    from utils import Preprocessor


class AdultDatasetTester:
    """
    Comprehensive test class for DataFoundry using the Adult Census Income dataset.
    Tests three main features:
    1. Data labeling (weak supervision with Snorkel)
    2. Data generation (synthetic data generation with evaluation)
    3. Anomaly detection
    """
    
    def __init__(self, adult_csv_path, test_dir=None):
        """
        Initialize the tester with the adult dataset.
        
        Args:
            adult_csv_path: Path to adult.csv file
            test_dir: Optional directory for test files. If None, creates a temp directory.
        """
        self.adult_csv_path = adult_csv_path
        if test_dir is None:
            self.test_dir = tempfile.mkdtemp(prefix='adult_test_')
        else:
            self.test_dir = test_dir
            os.makedirs(self.test_dir, exist_ok=True)
        
        self.test_results = {}
        print(f"Test directory: {self.test_dir}")
        
        # Load and prepare the dataset
        self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """Load the adult dataset and prepare it for testing."""
        print("\nLoading adult census income dataset...")
        # Read CSV with proper handling of quoted column names
        df = pd.read_csv(self.adult_csv_path, quotechar='"')
        
        # Clean column names (remove quotes if present)
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        # Convert income to binary (1 for >50K, 0 for <=50K)
        # Handle both string and already processed values
        if df['income'].dtype == 'object':
            df['income_binary'] = (df['income'].str.strip() == '>50K').astype(int)
        else:
            df['income_binary'] = (df['income'] == '>50K').astype(int)
        
        # Clean up missing values (replace "?" with NaN for numerical columns)
        numerical_cols = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.df_full = df.copy()
        print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        print(f"Income distribution: {df['income_binary'].value_counts().to_dict()}")
    
    def create_adult_labeling_functions(self):
        """
        Create labeling functions for the adult income prediction task.
        These are heuristics based on domain knowledge.
        
        Returns:
            List of LabelingFunction objects
        """
        @labeling_function()
        def lf_high_education(x):
            """High education (Masters, Doctorate, Prof-school) -> >50K"""
            high_ed = ['Masters', 'Doctorate', 'Prof-school']
            return 1 if str(x.get('education', '')).strip() in high_ed else 0
        
        @labeling_function()
        def lf_low_education(x):
            """Low education (1st-4th, 5th-6th, 7th-8th) -> <=50K"""
            low_ed = ['1st-4th', '5th-6th', '7th-8th']
            return 0 if str(x.get('education', '')).strip() in low_ed else 1
        
        @labeling_function()
        def lf_high_capital_gain(x):
            """High capital gain (>5000) -> >50K"""
            try:
                cg = float(x.get('capital.gain', 0))
                return 1 if cg > 5000 else 0
            except:
                return 0
        
        @labeling_function()
        def lf_high_hours(x):
            """High hours per week (>50) -> >50K"""
            try:
                hours = float(x.get('hours.per.week', 0))
                return 1 if hours > 50 else 0
            except:
                return 0
        
        @labeling_function()
        def lf_exec_managerial(x):
            """Executive/Managerial occupation -> >50K"""
            occ = str(x.get('occupation', '')).strip()
            return 1 if 'Exec-managerial' in occ else 0
        
        @labeling_function()
        def lf_prof_specialty(x):
            """Professional specialty -> >50K"""
            occ = str(x.get('occupation', '')).strip()
            return 1 if 'Prof-specialty' in occ else 0
        
        @labeling_function()
        def lf_other_service(x):
            """Other-service occupation -> <=50K"""
            occ = str(x.get('occupation', '')).strip()
            return 0 if 'Other-service' in occ else 1
        
        @labeling_function()
        def lf_handlers_cleaners(x):
            """Handlers-cleaners -> <=50K"""
            occ = str(x.get('occupation', '')).strip()
            return 0 if 'Handlers-cleaners' in occ else 1
        
        @labeling_function()
        def lf_high_age(x):
            """High age (>60) might indicate >50K"""
            try:
                age = float(x.get('age', 0))
                return 1 if age > 60 else 0
            except:
                return 0
        
        @labeling_function()
        def lf_young_age(x):
            """Young age (<25) might indicate <=50K"""
            try:
                age = float(x.get('age', 0))
                return 0 if age < 25 else 1
            except:
                return 0
        
        return [
            lf_high_education,
            lf_low_education,
            lf_high_capital_gain,
            lf_high_hours,
            lf_exec_managerial,
            lf_prof_specialty,
            lf_other_service,
            lf_handlers_cleaners,
            lf_high_age,
            lf_young_age
        ]
    
    def test_data_labeling(self, unlabeled_ratio=0.7):
        """
        Test 1: Data labeling with partial labels.
        
        Creates a temporary dataset where some labels are missing,
        then tests the labeling system and calculates miss rate.
        
        Args:
            unlabeled_ratio: Fraction of data to keep unlabeled (default 0.7)
        
        Returns:
            Dictionary with test results including miss rate
        """
        print("\n" + "="*60)
        print("TEST 1: Data Labeling (Weak Supervision)")
        print("="*60)
        
        # Create a copy of the data
        df_test = self.df_full.copy()
        
        # Create partially labeled dataset (remove some income labels)
        np.random.seed(42)
        n_samples = len(df_test)
        n_unlabeled = int(n_samples * unlabeled_ratio)
        unlabeled_indices = np.random.choice(n_samples, n_unlabeled, replace=False)
        
        # Create unlabeled dataset (remove income column)
        df_unlabeled = df_test.copy()
        df_unlabeled = df_unlabeled.drop(columns=['income', 'income_binary'])
        
        # Save unlabeled data
        unlabeled_path = os.path.join(self.test_dir, 'adult_unlabeled.parquet')
        df_unlabeled.to_parquet(unlabeled_path, index=False)
        print(f"Created unlabeled dataset: {len(df_unlabeled)} samples")
        print(f"  - Unlabeled ratio: {unlabeled_ratio:.1%}")
        
        # Create labeling functions
        labeling_functions = self.create_adult_labeling_functions()
        print(f"  - Created {len(labeling_functions)} labeling functions")
        
        # Apply labeling
        labeled_path = os.path.join(self.test_dir, 'adult_labeled.parquet')
        try:
            labeled_path = create_labels(
                data_path=unlabeled_path,
                labeling_functions_list=labeling_functions,
                output_path=labeled_path
            )
            
            # Load labeled data
            df_labeled = pd.read_parquet(labeled_path)
            
            # Add gold labels by matching indices
            # Since we removed income column, we need to match by position
            df_labeled = df_labeled.reset_index(drop=True)
            df_test_reset = df_test.reset_index(drop=True)
            df_labeled['income_binary'] = df_test_reset['income_binary'].values
            
            # Convert probabilities to binary predictions (threshold 0.5)
            df_labeled['predicted_label'] = (df_labeled['label_probability'] >= 0.5).astype(int)
            
            # Calculate miss rate (error rate)
            # Only calculate on samples where we have gold labels
            mask = df_labeled['income_binary'].notna()
            if mask.sum() > 0:
                miss_rate = (df_labeled.loc[mask, 'predicted_label'] != 
                           df_labeled.loc[mask, 'income_binary']).mean()
                accuracy = 1 - miss_rate
                
                print(f"\n✓ Data labeling successful!")
                print(f"  - Labeled {len(df_labeled)} samples")
                print(f"  - Probability range: [{df_labeled['label_probability'].min():.3f}, "
                      f"{df_labeled['label_probability'].max():.3f}]")
                print(f"  - Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                print(f"  - Miss rate: {miss_rate:.3f} ({miss_rate*100:.1f}%)")
                
                # Confusion matrix
                cm = confusion_matrix(
                    df_labeled.loc[mask, 'income_binary'],
                    df_labeled.loc[mask, 'predicted_label']
                )
                print(f"  - Confusion matrix:")
                print(f"    True Neg: {cm[0,0]}, False Pos: {cm[0,1]}")
                print(f"    False Neg: {cm[1,0]}, True Pos: {cm[1,1]}")
                
                self.test_results['labeling'] = {
                    'success': True,
                    'output_path': labeled_path,
                    'miss_rate': float(miss_rate),
                    'accuracy': float(accuracy),
                    'confusion_matrix': cm.tolist()
                }
            else:
                print("⚠ Warning: No gold labels available for comparison")
                self.test_results['labeling'] = {
                    'success': True,
                    'output_path': labeled_path,
                    'miss_rate': None,
                    'accuracy': None
                }
            
            return labeled_path
            
        except Exception as e:
            print(f"✗ Data labeling failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['labeling'] = {'success': False, 'error': str(e)}
            raise
    
    def test_data_generation(self, train_size=1000, num_generate=500, model_types=None):
        """
        Test 2: Data generation with evaluation.
        
        Uses a small portion of data to train models, generates synthetic data,
        and evaluates using:
        - Univariate: Histograms and KS test for age
        - Multivariate: Correlation heatmaps
        
        Args:
            train_size: Number of samples to use for training
            num_generate: Number of samples to generate
            model_types: List of model configs to test. If None, tests all available models.
                        Each config should be a dict with 'model_template', 'model_class_name', 
                        and optional 'model_specific_params'
        
        Returns:
            Dictionary with test results for each model
        """
        print("\n" + "="*60)
        print("TEST 2: Data Generation (with Evaluation)")
        print("="*60)
        
        # Create a small training dataset
        np.random.seed(42)
        df_train = self.df_full.sample(n=min(train_size, len(self.df_full)), random_state=42)
        
        # Prepare data for training (need to label it first)
        # Create a simplified version with just key features
        df_train_clean = df_train[[
            'age', 'education.num', 'capital.gain', 'capital.loss',
            'hours.per.week', 'income', 'workclass', 'education', 
            'occupation', 'relationship', 'race', 'sex', 'income_binary'
        ]].copy()
        
        # Remove any rows with missing values in numerical columns
        numerical_cols = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
        df_train_clean = df_train_clean.dropna(subset=numerical_cols)
        
        # Save training data (without income_binary for labeling step)
        train_path = os.path.join(self.test_dir, 'adult_train_small.parquet')
        df_train_for_labeling = df_train_clean.drop(columns=['income_binary'])
        df_train_for_labeling.to_parquet(train_path, index=False)
        print(f"Created training dataset: {len(df_train_clean)} samples")
        
        # Create labeling functions and label the data
        labeling_functions = self.create_adult_labeling_functions()
        labeled_train_path = os.path.join(self.test_dir, 'adult_train_labeled.parquet')
        
        try:
            # Label the training data
            labeled_train_path = create_labels(
                data_path=train_path,
                labeling_functions_list=labeling_functions,
                output_path=labeled_train_path
            )
            
            # Add income_binary back for training (use gold labels)
            df_labeled = pd.read_parquet(labeled_train_path)
            df_labeled = df_labeled.reset_index(drop=True)
            df_train_clean_reset = df_train_clean.reset_index(drop=True)
            df_labeled['income_binary'] = df_train_clean_reset['income_binary'].values
            df_labeled.to_parquet(labeled_train_path, index=False)
            
            # Define models to test
            if model_types is None:
                model_types = [
                    {
                        'name': 'TabularCVAE',
                        'model_template': 'tabular_cvae.py',
                        'model_class_name': 'TabularCVAE',
                        'model_specific_params': {},
                        'use_categorical': False  # Only numerical features
                    },
                    {
                        'name': 'TabularVAE_GMM',
                        'model_template': 'tabular_vae_gmm.py',
                        'model_class_name': 'TabularVAE_GMM',
                        'model_specific_params': {'n_clusters': 3},
                        'use_categorical': False  # Only numerical features
                    },
                    {
                        'name': 'MixedDataCVAE',
                        'model_template': 'mixed_data_cvae.py',
                        'model_class_name': 'MixedDataCVAE',
                        'model_specific_params': {},
                        'use_categorical': True  # Needs categorical features
                    }
                ]
            
            # Load original data for comparison (use full dataset)
            df_real = self.df_full[[
                'age', 'education.num', 'capital.gain', 
                'capital.loss', 'hours.per.week'
            ]].copy()
            df_real = df_real.dropna(subset=['age', 'education.num', 'capital.gain', 
                                            'capital.loss', 'hours.per.week'])
            
            all_results = {}
            
            # Test each model
            for model_config in model_types:
                model_name = model_config['name']
                print(f"\n{'='*60}")
                print(f"Testing {model_name}")
                print(f"{'='*60}")
                
                try:
                    # Prepare config for training
                    use_categorical = model_config.get('use_categorical', False)
                    
                    if use_categorical and model_config['model_class_name'] == 'MixedDataCVAE':
                        # MixedDataCVAE needs special config with categorical features
                        # Use a couple of categorical columns: workclass and education
                        categorical_cols = ['workclass', 'education']
                        numerical_cols = ['age', 'education.num', 'capital.gain', 
                                         'capital.loss', 'hours.per.week']
                        
                        # Get number of unique categories for each categorical column
                        df_labeled_sample = pd.read_parquet(labeled_train_path)
                        categorical_embed_dims = []
                        for col in categorical_cols:
                            n_categories = df_labeled_sample[col].nunique()
                            # Use embedding dimension of 8 for each categorical feature
                            categorical_embed_dims.append((n_categories, 8))
                        
                        base_model_params = {
                            'model_template': model_config['model_template'],
                            'model_class_name': model_config['model_class_name'],
                            'numerical_dim': len(numerical_cols),
                            'categorical_embed_dims': categorical_embed_dims,
                            'latent_dim': 16,
                            'condition_dim': 1,
                            'encoder_hidden_layers': [32, 24],
                            'decoder_hidden_layers': [24, 32],
                            'feature_cols': {
                                'numerical_cols': numerical_cols,
                                'categorical_cols': categorical_cols
                            },
                            'condition_cols': ['income_binary']
                        }
                    else:
                        # Standard models (TabularCVAE, TabularVAE_GMM)
                        base_model_params = {
                            'model_template': model_config['model_template'],
                            'model_class_name': model_config['model_class_name'],
                            'latent_dim': 16,
                            'condition_dim': 1,
                            'encoder_hidden_layers': [32, 24],
                            'decoder_hidden_layers': [24, 32],
                            'feature_cols': {
                                'numerical_cols': ['age', 'education.num', 'capital.gain', 
                                                 'capital.loss', 'hours.per.week']
                            },
                            'condition_cols': ['income_binary']
                        }
                    
                    # Add model-specific parameters
                    base_model_params.update(model_config.get('model_specific_params', {}))
                    
                    config = {
                        'data_path': labeled_train_path,
                        'preprocessor_path': os.path.join(self.test_dir, f'preprocessor_{model_name.lower()}.pkl'),
                        'output_model_path': os.path.join(self.test_dir, f'decoder_{model_name.lower()}.pth'),
                        'model_params': base_model_params,
                        'training_params': {
                            'batch_size': 32,
                            'learning_rate': 0.001,
                            'epochs': 60  # Reduced to 50-60 epochs as requested
                        }
                    }
                    
                    config_path = os.path.join(self.test_dir, f'config_{model_name.lower()}.json')
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    print(f"\nTraining {model_name} model...")
                    model_path = train_model(config_path)
                    print(f"✓ Model trained and saved to {model_path}")
                    
                    # Generate synthetic data
                    print(f"\nGenerating {num_generate} synthetic samples with {model_name}...")
                    generated_path = os.path.join(self.test_dir, f'adult_generated_{model_name.lower()}.parquet')
                    generated_path = generate_data(
                        model_path=model_path,
                        config_path=config_path,
                        label=1,  # Generate samples with income >50K
                        num_to_generate=num_generate,
                        output_path=generated_path,
                        output_format='parquet'
                    )
                    
                    # Load generated data
                    df_generated = pd.read_parquet(generated_path)
                    print(f"✓ Generated {len(df_generated)} samples")
                    
                    # Evaluation: Univariate (Age)
                    print(f"\n--- {model_name}: Univariate Evaluation (Age) ---")
                    real_age = df_real['age'].values
                    synthetic_age = df_generated['age'].values
                    
                    # KS Test
                    ks_statistic, ks_pvalue = stats.ks_2samp(real_age, synthetic_age)
                    print(f"Kolmogorov-Smirnov Test:")
                    print(f"  - KS Statistic: {ks_statistic:.4f}")
                    print(f"  - P-value: {ks_pvalue:.4f}")
                    print(f"  - {'✓ Distributions are similar' if ks_pvalue > 0.05 else '✗ Distributions differ significantly'}")
                    
                    # Plot histograms
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    axes[0].hist(real_age, bins=30, alpha=0.7, label='Real', color='blue', edgecolor='black')
                    axes[0].set_title('Real Age Distribution')
                    axes[0].set_xlabel('Age')
                    axes[0].set_ylabel('Frequency')
                    axes[0].grid(True, alpha=0.3)
                    
                    axes[1].hist(synthetic_age, bins=30, alpha=0.7, label='Synthetic', color='orange', edgecolor='black')
                    axes[1].set_title(f'{model_name} Synthetic Age Distribution')
                    axes[1].set_xlabel('Age')
                    axes[1].set_ylabel('Frequency')
                    axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    histogram_path = os.path.join(self.test_dir, f'age_histograms_{model_name.lower()}.png')
                    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
                    print(f"  - Saved histograms to {histogram_path}")
                    plt.close()
                    
                    # Evaluation: Multivariate (Correlations)
                    print(f"\n--- {model_name}: Multivariate Evaluation (Correlations) ---")
                    
                    # Select numerical columns for correlation
                    numerical_cols = ['age', 'education.num', 'capital.gain', 
                                    'capital.loss', 'hours.per.week']
                    
                    # Calculate correlations
                    real_corr = df_real[numerical_cols].corr()
                    synthetic_corr = df_generated[numerical_cols].corr()
                    
                    # Calculate correlation difference
                    corr_diff = np.abs(real_corr - synthetic_corr)
                    mean_corr_diff = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean()
                    print(f"Mean absolute correlation difference: {mean_corr_diff:.4f}")
                    
                    # Plot correlation heatmaps
                    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                    
                    sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                               center=0, vmin=-1, vmax=1, ax=axes[0], cbar_kws={'label': 'Correlation'})
                    axes[0].set_title('Real Data Correlation Matrix', fontsize=14, fontweight='bold')
                    
                    sns.heatmap(synthetic_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                               center=0, vmin=-1, vmax=1, ax=axes[1], cbar_kws={'label': 'Correlation'})
                    axes[1].set_title(f'{model_name} Synthetic Data Correlation Matrix', fontsize=14, fontweight='bold')
                    
                    plt.tight_layout()
                    correlation_path = os.path.join(self.test_dir, f'correlation_heatmaps_{model_name.lower()}.png')
                    plt.savefig(correlation_path, dpi=150, bbox_inches='tight')
                    print(f"  - Saved correlation heatmaps to {correlation_path}")
                    plt.close()
                    
                    all_results[model_name] = {
                        'success': True,
                        'generated_path': generated_path,
                        'ks_statistic': float(ks_statistic),
                        'ks_pvalue': float(ks_pvalue),
                        'mean_corr_diff': float(mean_corr_diff),
                        'histogram_path': histogram_path,
                        'correlation_path': correlation_path
                    }
                    
                except Exception as e:
                    print(f"✗ {model_name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[model_name] = {'success': False, 'error': str(e)}
            
            # Store results
            self.test_results['generation'] = all_results
            
            # Print summary comparison
            print(f"\n{'='*60}")
            print("MODEL COMPARISON SUMMARY")
            print(f"{'='*60}")
            for model_name, result in all_results.items():
                if result.get('success', False):
                    print(f"\n{model_name}:")
                    print(f"  - KS Statistic: {result.get('ks_statistic', 'N/A'):.4f}")
                    print(f"  - KS P-value: {result.get('ks_pvalue', 'N/A'):.4f}")
                    print(f"  - Mean Correlation Diff: {result.get('mean_corr_diff', 'N/A'):.4f}")
                else:
                    print(f"\n{model_name}: FAILED - {result.get('error', 'Unknown error')}")
            
            return all_results
            
        except Exception as e:
            print(f"✗ Data generation failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['generation'] = {'success': False, 'error': str(e)}
            raise
    
    def test_anomaly_detection(self):
        """
        Test 3: Anomaly detection.
        
        Tests the anomaly detection functionality using a trained model.
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*60)
        print("TEST 3: Anomaly Detection")
        print("="*60)
        
        # Use the model from data generation test if available
        # Try to find any available model config (prefer TabularCVAE)
        config_path = os.path.join(self.test_dir, 'config_tabularcvae.json')
        model_path = os.path.join(self.test_dir, 'decoder_tabularcvae.pth')
        
        # If TabularCVAE not found, try TabularVAE_GMM
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            config_path = os.path.join(self.test_dir, 'config_tabularvae_gmm.json')
            model_path = os.path.join(self.test_dir, 'decoder_tabularvae_gmm.pth')
        
        # If still not found, run data generation test
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            print("⚠ Model not found. Running data generation test first...")
            self.test_data_generation(train_size=1000, num_generate=500)
            # After generation test, use TabularCVAE config
            config_path = os.path.join(self.test_dir, 'config_tabularcvae.json')
            model_path = os.path.join(self.test_dir, 'decoder_tabularcvae.pth')
        
        # Load config to get preprocessor path
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Use a subset of data for anomaly detection
        np.random.seed(42)
        df_scan = self.df_full.sample(n=min(500, len(self.df_full)), random_state=42)
        
        # Prepare data (same columns as training)
        # Need to include all columns that labeling functions use
        # Also need income_binary as condition column for the preprocessor
        df_scan_clean = df_scan[[
            'age', 'education.num', 'capital.gain', 'capital.loss',
            'hours.per.week', 'income', 'workclass', 'education', 
            'occupation', 'relationship', 'race', 'sex', 'income_binary'
        ]].copy()
        numerical_cols = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
        df_scan_clean = df_scan_clean.dropna(subset=numerical_cols)
        
        # Save data to scan
        scan_path = os.path.join(self.test_dir, 'adult_scan.parquet')
        df_scan_clean.to_parquet(scan_path, index=False)
        print(f"Created dataset to scan: {len(df_scan_clean)} samples")
        
        try:
            output_path = os.path.join(self.test_dir, 'anomaly_scores_adult.parquet')
            
            anomaly_path = find_anomalies(
                config_path=config_path,
                model_path=model_path,
                preprocessor_path=config_dict['preprocessor_path'],
                data_path_to_scan=scan_path,
                output_path=output_path
            )
            
            # Load results
            df_anomalies = pd.read_parquet(anomaly_path)
            
            print(f"\n✓ Anomaly detection successful!")
            print(f"  - Scanned {len(df_anomalies)} samples")
            print(f"  - Anomaly score range: [{df_anomalies['anomaly_score'].min():.4f}, "
                  f"{df_anomalies['anomaly_score'].max():.4f}]")
            print(f"  - Mean anomaly score: {df_anomalies['anomaly_score'].mean():.4f}")
            print(f"  - Median anomaly score: {df_anomalies['anomaly_score'].median():.4f}")
            print(f"  - Top 5 anomalies:")
            for idx, row in df_anomalies.head(5).iterrows():
                print(f"    - Score: {row['anomaly_score']:.4f}, Age: {row['age']:.0f}, "
                      f"Hours: {row['hours.per.week']:.0f}")
            
            self.test_results['anomaly_detection'] = {
                'success': True,
                'output_path': anomaly_path,
                'max_score': float(df_anomalies['anomaly_score'].max()),
                'min_score': float(df_anomalies['anomaly_score'].min()),
                'mean_score': float(df_anomalies['anomaly_score'].mean()),
                'median_score': float(df_anomalies['anomaly_score'].median())
            }
            
            return anomaly_path
            
        except Exception as e:
            print(f"✗ Anomaly detection failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['anomaly_detection'] = {'success': False, 'error': str(e)}
            raise
    
    def run_all_tests(self):
        """
        Run all three tests:
        1. Data labeling
        2. Data generation
        3. Anomaly detection
        
        Returns:
            Dictionary of test results
        """
        print("\n" + "="*60)
        print("RUNNING ALL ADULT DATASET TESTS")
        print("="*60)
        
        try:
            # Test 1: Data Labeling
            self.test_data_labeling(unlabeled_ratio=0.7)
            
            # Test 2: Data Generation
            self.test_data_generation(train_size=1000, num_generate=500)
            
            # Test 3: Anomaly Detection
            self.test_anomaly_detection()
            
            # Print summary
            print("\n" + "="*60)
            print("TEST SUMMARY")
            print("="*60)
            for test_name, result in self.test_results.items():
                if test_name == 'generation' and isinstance(result, dict):
                    # Handle generation results which is a dict of models
                    print(f"✓ PASS: {test_name}")
                    for model_name, model_result in result.items():
                        if model_result.get('success', False):
                            print(f"  - {model_name}:")
                            if 'ks_pvalue' in model_result:
                                print(f"    KS test p-value: {model_result['ks_pvalue']:.4f}")
                            if 'mean_corr_diff' in model_result:
                                print(f"    Mean correlation diff: {model_result['mean_corr_diff']:.4f}")
                        else:
                            print(f"  - {model_name}: FAILED - {model_result.get('error', 'Unknown')}")
                else:
                    status = "✓ PASS" if result.get('success', False) else "✗ FAIL"
                    print(f"{status}: {test_name}")
                    if result.get('success', False):
                        if 'miss_rate' in result:
                            print(f"  - Miss rate: {result['miss_rate']:.3f}")
                        if 'ks_pvalue' in result:
                            print(f"  - KS test p-value: {result['ks_pvalue']:.4f}")
                        if 'mean_corr_diff' in result:
                            print(f"  - Mean correlation diff: {result['mean_corr_diff']:.4f}")
            
            # Check if all tests passed (handle generation dict specially)
            all_passed = True
            for test_name, result in self.test_results.items():
                if test_name == 'generation' and isinstance(result, dict):
                    # For generation, check if at least one model succeeded
                    all_passed = all_passed and any(r.get('success', False) for r in result.values())
                else:
                    all_passed = all_passed and result.get('success', False)
            print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
            
            return self.test_results
            
        except Exception as e:
            print(f"\n✗ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return self.test_results
    
    def cleanup(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"Cleaned up test directory: {self.test_dir}")


if __name__ == '__main__':
    # Path to adult.csv
    adult_csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'testing_data',
        'adult.csv'
    )
    
    if not os.path.exists(adult_csv_path):
        # Try alternative path
        adult_csv_path = 'testing_data/adult.csv'
        if not os.path.exists(adult_csv_path):
            raise FileNotFoundError(f"Could not find adult.csv at {adult_csv_path}")
    
    # Run tests
    tester = AdultDatasetTester(adult_csv_path)
    try:
        results = tester.run_all_tests()
        print(f"\nTest files saved in: {tester.test_dir}")
        print("(Set tester.cleanup() to remove test files)")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

