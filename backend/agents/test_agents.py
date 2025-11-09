import warnings
import json
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import requests

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add labeling_pipeline to path
labeling_pipeline_path = os.path.join(os.path.dirname(__file__), 'labeling_pipeline')
if labeling_pipeline_path not in sys.path:
    sys.path.insert(0, labeling_pipeline_path)

# Import labeling pipeline tool
try:
    from labeling_pipeline.labeling_tool import run_labeling_pipeline
except ImportError:
    try:
        from agents.labeling_pipeline.labeling_tool import run_labeling_pipeline
    except ImportError:
        labeling_dir = os.path.join(os.path.dirname(__file__), 'labeling_pipeline')
        if labeling_dir not in sys.path:
            sys.path.insert(0, labeling_dir)
        from labeling_tool import run_labeling_pipeline

# Import training pipeline tool
try:
    from training_pipeline.training_tool import run_training_pipeline
except ImportError:
    try:
        from agents.training_pipeline.training_tool import run_training_pipeline
    except ImportError:
        training_dir = os.path.join(os.path.dirname(__file__), 'training_pipeline')
        if training_dir not in sys.path:
            sys.path.insert(0, training_dir)
        from training_tool import run_training_pipeline

# Import generation pipeline tool
try:
    from generation_pipeline.generation_tool import run_generation_pipeline
except ImportError:
    try:
        from agents.generation_pipeline.generation_tool import run_generation_pipeline
    except ImportError:
        generation_dir = os.path.join(os.path.dirname(__file__), 'generation_pipeline')
        if generation_dir not in sys.path:
            sys.path.insert(0, generation_dir)
        from generation_tool import run_generation_pipeline

# Import anomaly pipeline tool
try:
    from anomaly_pipeline.anomaly_tool import run_anomaly_pipeline
except ImportError:
    try:
        from agents.anomaly_pipeline.anomaly_tool import run_anomaly_pipeline
    except ImportError:
        anomaly_dir = os.path.join(os.path.dirname(__file__), 'anomaly_pipeline')
        if anomaly_dir not in sys.path:
            sys.path.insert(0, anomaly_dir)
        from anomaly_tool import run_anomaly_pipeline


# ============================================================================
# NAT Server Helper Functions for Orchestrator Tests
# ============================================================================

def get_nat_url():
    """Get NAT server base URL from environment or use default."""
    return os.getenv("NAT_URL", "http://localhost:8000")


def call_nat_workflow(user_message: str, nat_url: str = None, timeout: int = 600) -> dict:
    """
    Call the NAT server's /generate endpoint with a user message.
    
    Based on NVIDIA NeMo Agent Toolkit documentation:
    https://docs.nvidia.com/nemo/agent-toolkit/latest/workflows/run-workflows.html
    
    Args:
        user_message: The natural language instruction for the orchestrator
        nat_url: Base URL of NAT server (default: from env or http://localhost:8000)
        timeout: Request timeout in seconds (default: 600s for long pipelines)
    
    Returns:
        Response dictionary from the NAT server
    """
    if nat_url is None:
        nat_url = get_nat_url()
    
    endpoint = f"{nat_url}/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"input_message": user_message}
    
    print(f"Calling NAT endpoint: {endpoint}")
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        print(f"NAT Response received (status {response.status_code})")
        return result
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Failed to connect to NAT server at {nat_url}. "
            f"Make sure the server is running with: nat serve --config_file backend/agents/workflow.yml"
        ) from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            f"NAT server request timed out after {timeout}s. "
            f"The orchestrator may be taking longer than expected."
        ) from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"NAT server returned error {response.status_code}: {response.text[:500]}"
        ) from e


def extract_path_from_response(response: dict, key_hints: list) -> str:
    """
    Extract a file path from NAT response by searching various possible locations.
    
    Now supports the new structured format with 'file_paths' dictionary.
    
    Args:
        response: Response dictionary from NAT server
        key_hints: List of possible key names to search for (e.g., ['labeled_output_path', 'output_path'])
    
    Returns:
        Extracted file path or None if not found
    """
    
    # Try common response structures
    if isinstance(response, dict):
        # NEW: Check structured file_paths dictionary first (most reliable)
        if "file_paths" in response and isinstance(response["file_paths"], dict):
            for key in key_hints:
                if key in response["file_paths"] and response["file_paths"][key]:
                    path = response["file_paths"][key]
                    if path and path != "None" and path.lower() != "null":
                        return path
        
        # Direct keys (backward compatibility)
        for key in key_hints:
            if key in response and response[key]:
                return response[key]
        
        # Nested in 'output'
        if "output" in response and isinstance(response["output"], dict):
            for key in key_hints:
                if key in response["output"] and response["output"][key]:
                    return response["output"][key]
        
        # Nested in 'result'
        if "result" in response and isinstance(response["result"], dict):
            for key in key_hints:
                if key in response["result"] and response["result"][key]:
                    return response["result"][key]
        
        # Try to parse as JSON string in 'output' or 'result'
        for response_key in ["output", "result", "message"]:
            if response_key in response and isinstance(response[response_key], str):
                try:
                    parsed = json.loads(response[response_key])
                    if isinstance(parsed, dict):
                        for key in key_hints:
                            if key in parsed and parsed[key]:
                                return parsed[key]
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Try to extract path from natural language text in 'output' or 'result'
        for response_key in ["output", "result", "message"]:
            if response_key in response and isinstance(response[response_key], str):
                text = response[response_key]
                # Look for patterns like "labeled_output_path: /path/to/file"
                for key in key_hints:
                    pattern = rf'{key}:\s*([/\w\-_.]+(?:\.parquet|\.pth|\.json|\.joblib|\.csv|\.pt))'
                    match = re.search(pattern, text)
                    if match:
                        return match.group(1)
                
                # Also try finding any file path that looks valid
                path_pattern = r'(/tmp/[^\s]+(?:\.parquet|\.pth|\.json|\.joblib|\.csv|\.pt))'
                match = re.search(path_pattern, text)
                if match:
                    return match.group(1)
    
    return None


class DataFoundryAgentTester:
    """
    Comprehensive test class for DataFoundry agents and orchestrator pipeline.
    Currently tests the labeling pipeline, but designed to be extended for:
    - Training pipeline
    - Anomaly detection pipeline
    - Generation pipeline
    - Full orchestrator workflow
    """
    
    def __init__(self, test_dir=None):
        """
        Initialize the tester with a temporary directory for test files.
        
        Args:
            test_dir: Optional directory for test files. If None, creates a temp directory.
        """
        if test_dir is None:
            self.test_dir = tempfile.mkdtemp(prefix='datafoundry_agent_test_')
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
    
    def prepare_test_data(self, adult_csv_path, train_ratio=0.7):
        """
        Prepare test data from adult.csv:
        - Split into raw unlabeled data and hand-labeled ground truth
        - Create ground truth file with income labels
        
        Args:
            adult_csv_path: Path to adult.csv file
            train_ratio: Ratio of data to use as "unlabeled" (rest is ground truth)
        
        Returns:
            Tuple of (raw_data_path, ground_truth_path)
        """
        print("\nPreparing test data...")
        
        # Load adult dataset
        df = pd.read_csv(adult_csv_path, quotechar='"')
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        # Convert income to binary (1 for >50K, 0 for <=50K)
        df['income_binary'] = (df['income'].str.strip() == '>50K').astype(int)
        
        # Clean numerical columns
        numerical_cols = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=numerical_cols)
        
        # Split into "unlabeled" and "ground truth" sets
        # In real use, ground truth is a small manually-labeled subset of unlabeled data
        np.random.seed(42)
        n_samples = len(df)
        n_train = int(n_samples * train_ratio)
        train_indices = np.random.choice(n_samples, n_train, replace=False)
        
        # Create "unlabeled" data (remove income columns)
        df_unlabeled = df.iloc[train_indices].copy()
        df_unlabeled = df_unlabeled.drop(columns=['income', 'income_binary'])
        
        # Create ground truth from a small subset of the unlabeled data
        # This simulates manually labeling a small portion
        n_ground_truth = min(500, len(df_unlabeled))  # Use up to 500 samples for ground truth
        ground_truth_indices = np.random.choice(len(df_unlabeled), n_ground_truth, replace=False)
        df_ground_truth_full = df.iloc[train_indices[ground_truth_indices]].copy()
        
        # Create ground truth file with common columns + income label
        # Include common columns for proper alignment
        common_cols = ['age', 'education.num', 'hours.per.week']
        cols_to_keep = [col for col in common_cols if col in df_ground_truth_full.columns]
        df_ground_truth = df_ground_truth_full[cols_to_keep].copy()
        # Add income column (1 for >50K, 0 for <=50K) for evaluation
        df_ground_truth['income'] = df_ground_truth_full['income_binary'].copy()
        
        # Save files
        raw_data_path = os.path.join(self.test_dir, 'raw_unlabeled_data.parquet')
        ground_truth_path = os.path.join(self.test_dir, 'ground_truth_labels.parquet')
        
        df_unlabeled.to_parquet(raw_data_path, index=False)
        df_ground_truth.to_parquet(ground_truth_path, index=False)
        
        print(f"  - Created unlabeled data: {len(df_unlabeled)} samples")
        print(f"  - Created ground truth: {len(df_ground_truth)} samples")
        
        return raw_data_path, ground_truth_path
    
    def test_labeling_pipeline(self, adult_csv_path=None, user_goal=None, 
                               target_auc=0.75, max_attempts=2):
        """
        Test the labeling pipeline agent.
        
        Args:
            adult_csv_path: Path to adult.csv. If None, tries to find it.
            user_goal: Natural language goal for labeling. If None, uses default.
            target_auc: Target AUC score (default 0.75, lower for testing)
            max_attempts: Maximum number of attempts (default 2 for faster testing)
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*60)
        print("TEST: Labeling Pipeline Agent")
        print("="*60)
        
        # Find adult.csv if not provided
        if adult_csv_path is None:
            adult_csv_path = os.path.join(project_root, 'testing_data', 'adult.csv')
            if not os.path.exists(adult_csv_path):
                adult_csv_path = 'testing_data/adult.csv'
                if not os.path.exists(adult_csv_path):
                    raise FileNotFoundError(f"Could not find adult.csv at {adult_csv_path}")
        
        # Set default user goal
        if user_goal is None:
            user_goal = "Label the data to predict if a person's income is greater than $50K per year"
        
        try:
            # Prepare test data
            raw_data_path, ground_truth_path = self.prepare_test_data(adult_csv_path)
            
            print(f"\nUser Goal: {user_goal}")
            print(f"Target AUC: {target_auc}")
            print(f"Max Attempts: {max_attempts}")
            
            # Run the labeling pipeline
            print("\n--- Running Labeling Pipeline ---")
            try:
                labeled_file_path = run_labeling_pipeline(
                    user_goal=user_goal,
                    raw_data_path=raw_data_path,
                    hand_labeled_examples_path=ground_truth_path,
                    target_auc_score=target_auc,
                    max_attempts=max_attempts
                )
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "Unauthorized" in error_msg or "API key" in error_msg:
                    print("\n⚠ API Key Error: Please set NVIDIA_API_KEY in your .env file")
                    print("   The labeling pipeline requires a valid NVIDIA API key to use NIM models.")
                    raise Exception("NVIDIA_API_KEY_ERROR: Please set NVIDIA_API_KEY in .env file") from e
                elif "Syntax error" in error_msg or "invalid" in error_msg.lower():
                    print("\n⚠ Code Generation Error: The LLM generated invalid Python code.")
                    print("   This can happen if the model response format is unexpected.")
                    print("   Try running again - the model may generate better code on retry.")
                    raise Exception("CODE_GENERATION_ERROR: Invalid code generated by LLM") from e
                else:
                    raise
            
            # Verify output
            if not labeled_file_path or not os.path.exists(labeled_file_path):
                print(f"\n⚠ Warning: Labeled file not found: {labeled_file_path}")
                print("   This may indicate the pipeline failed to generate labels.")
                raise FileNotFoundError(f"Labeled file not found: {labeled_file_path}")
            
            # Load and verify labeled data
            df_labeled = pd.read_parquet(labeled_file_path)
            
            assert 'label_probability' in df_labeled.columns, "Missing label_probability column"
            assert len(df_labeled) > 0, "Labeled data is empty"
            assert df_labeled['label_probability'].between(0, 1).all(), "Invalid probabilities"
            
            # Evaluate final result
            try:
                from labeling_pipeline.utils import evaluate_labels
            except ImportError:
                try:
                    from agents.labeling_pipeline.utils import evaluate_labels
                except ImportError:
                    # Direct import from labeling_pipeline directory
                    labeling_dir = os.path.join(os.path.dirname(__file__), 'labeling_pipeline')
                    if labeling_dir not in sys.path:
                        sys.path.insert(0, labeling_dir)
                    from utils import evaluate_labels
            final_eval = evaluate_labels(labeled_file_path, ground_truth_path)
            final_auc = final_eval.get('roc_auc', 0.0)
            
            print(f"\n✓ Labeling pipeline successful!")
            print(f"  - Labeled {len(df_labeled)} samples")
            print(f"  - Probability range: [{df_labeled['label_probability'].min():.3f}, "
                  f"{df_labeled['label_probability'].max():.3f}]")
            print(f"  - Final AUC: {final_auc:.3f}")
            print(f"  - Output file: {labeled_file_path}")
            
            self.test_results['labeling_pipeline'] = {
                'success': True,
                'output_path': labeled_file_path,
                'final_auc': float(final_auc),
                'num_samples': len(df_labeled),
                'probability_range': [
                    float(df_labeled['label_probability'].min()),
                    float(df_labeled['label_probability'].max())
                ]
            }
            
            self.test_results['labeling_pipeline'] = {
                'success': True,
                'output_path': labeled_file_path,
                'final_auc': float(final_auc),
                'num_samples': len(df_labeled),
                'probability_range': [
                    float(df_labeled['label_probability'].min()),
                    float(df_labeled['label_probability'].max())
                ]
            }
            
            return self.test_results['labeling_pipeline']
            
        except Exception as e:
            print(f"\n✗ Labeling pipeline test failed: {e}")
            self.test_results['labeling_pipeline'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    def test_training_pipeline(self, labeled_data_path=None, holdout_test_path=None,
                               target_utility_pct=0.85, max_attempts=2):
        """
        Test the training pipeline agent.
        """
        print("\n" + "="*60)
        print("TEST: Training Pipeline Agent")
        print("="*60)
        
        # If no paths provided, use the output from labeling pipeline test
        if labeled_data_path is None:
            # First run labeling pipeline to get labeled data
            print("\nRunning labeling pipeline first to generate labeled data...")
            labeling_result = self.test_labeling_pipeline(max_attempts=1)
            if not labeling_result.get('success', False):
                print("\n✗ Cannot test training pipeline: Labeling failed")
                return {'success': False, 'error': 'Labeling pipeline failed'}
            
            labeled_data_path = labeling_result['output_path']
        
        # Create holdout test set if not provided
        if holdout_test_path is None:
            print("\nPreparing holdout test set...")
            adult_csv_path = os.path.join(project_root, 'testing_data', 'adult.csv')
            if not os.path.exists(adult_csv_path):
                adult_csv_path = 'testing_data/adult.csv'
            
            df = pd.read_csv(adult_csv_path)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
            df['income_binary'] = (df['income'].str.strip() == '>50K').astype(int)
            df = df.drop(columns=['income'])
            
            # Take a small holdout set (different from training)
            holdout_df = df.sample(n=500, random_state=123)
            holdout_test_path = os.path.join(self.test_dir, 'holdout_test.parquet')
            holdout_df.to_parquet(holdout_test_path, index=False)
            print(f"  - Created holdout test set: {len(holdout_df)} samples")
        
        try:
            print(f"\nTarget Quality Score: {target_utility_pct:.0%} (based on reconstruction error)")
            print(f"Max Attempts: {max_attempts}")
            
            # Run the training pipeline
            print("\n--- Running Training Pipeline ---")
            config_path, model_path, preprocessor_path = run_training_pipeline(
                labeled_data_path=labeled_data_path,
                holdout_test_path=holdout_test_path,
                target_utility_pct=target_utility_pct,
                max_attempts=max_attempts
            )
            
            # Verify outputs
            if not config_path or not os.path.exists(config_path):
                print(f"\n⚠ Warning: Config file not found: {config_path}")
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            if not model_path or not os.path.exists(model_path):
                print(f"\n⚠ Warning: Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Read config to get details
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract model type from nested structure
            model_type = config.get('model_params', {}).get('model_class_name', 'Unknown')
            
            print(f"\n✓ Training pipeline successful!")
            print(f"  - Model type: {model_type}")
            print(f"  - Config: {config_path}")
            print(f"  - Model: {model_path}")
            print(f"  - Preprocessor: {preprocessor_path}")
            
            self.test_results['training_pipeline'] = {
                'success': True,
                'config_path': config_path,
                'model_path': model_path,
                'preprocessor_path': preprocessor_path,
                'model_type': model_type,
                'message': "Training pipeline completed successfully."
            }
            return self.test_results['training_pipeline']
            
        except Exception as e:
            print(f"\n✗ Training pipeline test failed: {e}")
            self.test_results['training_pipeline'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    def test_generation_pipeline(self, config_path=None, model_path=None, 
                                 preprocessor_path=None, label=1.0, num_samples=100):
        """
        Test the generation pipeline agent.
        """
        print("\n" + "="*60)
        print("TEST: Generation Pipeline Agent")
        print("="*60)
        
        # If no paths provided, use the output from training pipeline test
        if config_path is None or model_path is None or preprocessor_path is None:
            # Check if training pipeline was already run
            if 'training_pipeline' in self.test_results and self.test_results['training_pipeline'].get('success', False):
                config_path = self.test_results['training_pipeline']['config_path']
                model_path = self.test_results['training_pipeline']['model_path']
                preprocessor_path = self.test_results['training_pipeline']['preprocessor_path']
            else:
                print("\nRunning training pipeline first to get model...")
                training_result = self.test_training_pipeline(max_attempts=1)
                if not training_result.get('success', False):
                    print("\n✗ Cannot test generation pipeline: Training failed")
                    return {'success': False, 'error': 'Training pipeline failed'}
                
                config_path = training_result['config_path']
                model_path = training_result['model_path']
                preprocessor_path = training_result['preprocessor_path']
        
        try:
            print(f"\nGenerating {num_samples} samples for label={label}")
            print(f"Using model: {os.path.basename(model_path)}")
            
            # Run the generation pipeline
            print("\n--- Running Generation Pipeline ---")
            synthetic_data_path = run_generation_pipeline(
                config_path=config_path,
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                label=label,
                num_to_generate=num_samples,
                output_format='csv'  # Use CSV for easy inspection
            )
            
            # Verify output
            if not synthetic_data_path or not os.path.exists(synthetic_data_path):
                print(f"\n⚠ Warning: Synthetic data file not found: {synthetic_data_path}")
                raise FileNotFoundError(f"Synthetic data file not found: {synthetic_data_path}")
            
            # Load and validate the synthetic data
            synthetic_df = pd.read_csv(synthetic_data_path)
            
            print(f"\n✓ Generation pipeline successful!")
            print(f"  - Generated {len(synthetic_df)} samples")
            print(f"  - Columns: {list(synthetic_df.columns)}")
            print(f"  - Output file: {synthetic_data_path}")
            
            # Show sample statistics
            print(f"\nSample statistics:")
            print(synthetic_df.describe())
            
            self.test_results['generation_pipeline'] = {
                'success': True,
                'output_path': synthetic_data_path,
                'num_samples': len(synthetic_df),
                'num_features': len(synthetic_df.columns),
                'message': "Generation pipeline completed successfully."
            }
            return self.test_results['generation_pipeline']
            
        except Exception as e:
            print(f"\n✗ Generation pipeline test failed: {e}")
            self.test_results['generation_pipeline'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    def test_anomaly_pipeline(self, config_path=None, model_path=None, 
                              preprocessor_path=None, data_to_scan_path=None):
        """
        Test the anomaly detection pipeline agent.
        """
        print("\n" + "="*60)
        print("TEST: Anomaly Detection Pipeline Agent")
        print("="*60)
        
        # If no paths provided, use the output from training pipeline test
        if config_path is None or model_path is None or preprocessor_path is None:
            # Check if training pipeline was already run
            if 'training_pipeline' in self.test_results and self.test_results['training_pipeline'].get('success', False):
                config_path = self.test_results['training_pipeline']['config_path']
                model_path = self.test_results['training_pipeline']['model_path']
                preprocessor_path = self.test_results['training_pipeline']['preprocessor_path']
            else:
                print("\nRunning training pipeline first to get model...")
                training_result = self.test_training_pipeline(max_attempts=1)
                if not training_result.get('success', False):
                    print("\n✗ Cannot test anomaly pipeline: Training failed")
                    return {'success': False, 'error': 'Training pipeline failed'}
                
                config_path = training_result['config_path']
                model_path = training_result['model_path']
                preprocessor_path = training_result['preprocessor_path']
        
        # If no data to scan provided, prepare some test data
        if data_to_scan_path is None:
            print("\nPreparing test data to scan for anomalies...")
            adult_csv_path = os.path.join(project_root, 'testing_data', 'adult.csv')
            if not os.path.exists(adult_csv_path):
                adult_csv_path = 'testing_data/adult.csv'
            
            df = pd.read_csv(adult_csv_path)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
            df['income_binary'] = (df['income'].str.strip() == '>50K').astype(int)
            
            # Add label_probability column (the conditioning variable needed by the model)
            # Use income_binary as a proxy for label_probability
            df['label_probability'] = df['income_binary'].astype(float)
            
            # Take a small sample to scan
            scan_df = df.sample(n=200, random_state=456)
            data_to_scan_path = os.path.join(self.test_dir, 'data_to_scan.parquet')
            scan_df.to_parquet(data_to_scan_path, index=False)
            print(f"  - Created test data: {len(scan_df)} samples")
        
        try:
            print(f"\nScanning data: {os.path.basename(data_to_scan_path)}")
            print(f"Using model: {os.path.basename(model_path)}")
            
            # Run the anomaly pipeline
            print("\n--- Running Anomaly Detection Pipeline ---")
            anomaly_report_path = run_anomaly_pipeline(
                config_path=config_path,
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                data_to_scan_path=data_to_scan_path
            )
            
            # Verify output
            if not anomaly_report_path or not os.path.exists(anomaly_report_path):
                print(f"\n⚠ Warning: Anomaly report not found: {anomaly_report_path}")
                raise FileNotFoundError(f"Anomaly report not found: {anomaly_report_path}")
            
            # Load and validate the anomaly report
            report_df = pd.read_parquet(anomaly_report_path)
            
            # Count anomalies (assuming anomaly_score column exists)
            if 'anomaly_score' in report_df.columns:
                num_anomalies = (report_df['anomaly_score'] > report_df['anomaly_score'].quantile(0.95)).sum()
            else:
                num_anomalies = 0
            
            print(f"\n✓ Anomaly detection pipeline successful!")
            print(f"  - Scanned {len(report_df)} samples")
            print(f"  - Detected ~{num_anomalies} potential anomalies (top 5%)")
            print(f"  - Report columns: {list(report_df.columns)}")
            print(f"  - Output file: {anomaly_report_path}")
            
            # Show anomaly score statistics
            if 'anomaly_score' in report_df.columns:
                print(f"\nAnomaly score statistics:")
                print(report_df['anomaly_score'].describe())
            
            self.test_results['anomaly_pipeline'] = {
                'success': True,
                'report_path': anomaly_report_path,
                'num_scanned': len(report_df),
                'num_anomalies': num_anomalies,
                'message': "Anomaly detection pipeline completed successfully."
            }
            return self.test_results['anomaly_pipeline']
            
        except Exception as e:
            print(f"\n✗ Anomaly detection pipeline test failed: {e}")
            self.test_results['anomaly_pipeline'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    def test_orchestrator_single_labeling(self, adult_csv_path=None):
        """
        Test orchestrator with a single labeling task via NAT server.
        
        This tests the orchestrator's ability to:
        - Understand natural language instructions
        - Select and call the correct pipeline tool (run_labeling_pipeline)
        - Return the labeled dataset path
        
        Requires: NAT server running with: nat serve --config_file backend/agents/workflow.yml
        """
        print("\n" + "="*60)
        print("TEST: Orchestrator - Single Task (Labeling)")
        print("="*60)
        
        # Find adult.csv
        if adult_csv_path is None:
            adult_csv_path = os.path.join(project_root, 'testing_data', 'adult.csv')
            if not os.path.exists(adult_csv_path):
                adult_csv_path = 'testing_data/adult.csv'
                if not os.path.exists(adult_csv_path):
                    raise FileNotFoundError(f"Could not find adult.csv")
        
        # Prepare test data
        raw_data_path, ground_truth_path = self.prepare_test_data(adult_csv_path)
        
        # Craft a natural language instruction for the orchestrator
        user_message = f"""Please label this dataset using weak supervision to predict income > $50K.

Here are the file paths:
- Unlabeled data: {raw_data_path}
- Ground truth examples: {ground_truth_path}

Target AUC: 0.70
Max attempts: 1

Return the path to the labeled output file."""
        
        print(f"\nSending instruction to orchestrator...")
        print(f"Raw data: {raw_data_path}")
        print(f"Ground truth: {ground_truth_path}")
        
        try:
            # Call NAT server
            response = call_nat_workflow(user_message, timeout=300)
            print(f"\nOrchestrator response received")
            print(f"Response structure: {json.dumps(response, indent=2)[:800]}...")
            
            # Validate new structured response format
            if "file_paths" not in response:
                print("⚠ Warning: Response missing 'file_paths' field (may be old format)")
            else:
                print(f"\n✓ Structured response format detected")
                print(f"  File paths: {json.dumps(response.get('file_paths', {}), indent=4)}")
                print(f"  Steps completed: {response.get('steps_completed', [])}")
            
            # Extract labeled file path from response (uses new file_paths structure)
            labeled_path = extract_path_from_response(
                response, 
                ['labeled_output_path', 'output_path', 'labeled_file_path', 'result_path']
            )
            
            if not labeled_path:
                raise ValueError(f"Could not find labeled output path in response: {json.dumps(response, indent=2)[:500]}")
            
            if not os.path.exists(labeled_path):
                raise FileNotFoundError(f"Labeled file not found at: {labeled_path}")
            
            # Validate the labeled data
            df_labeled = pd.read_parquet(labeled_path)
            assert 'label_probability' in df_labeled.columns, "Missing label_probability column"
            assert len(df_labeled) > 0, "Labeled data is empty"
            
            # Validate steps_completed includes labeling
            steps_completed = response.get('steps_completed', [])
            if 'labeling' not in steps_completed:
                print("⚠ Warning: 'labeling' not in steps_completed list")
            
            print(f"\n✓ Orchestrator labeling task successful!")
            print(f"  - Labeled {len(df_labeled)} samples")
            print(f"  - Output: {labeled_path}")
            print(f"  - Steps completed: {steps_completed}")
            
            self.test_results['orchestrator_single_labeling'] = {
                'success': True,
                'labeled_path': labeled_path,
                'num_samples': len(df_labeled),
                'steps_completed': steps_completed,
                'has_structured_response': 'file_paths' in response
            }
            return self.test_results['orchestrator_single_labeling']
            
        except Exception as e:
            print(f"\n✗ Orchestrator labeling test failed: {e}")
            self.test_results['orchestrator_single_labeling'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    def test_orchestrator_multi_pipeline(self, adult_csv_path=None, num_generate=50):
        """
        Test orchestrator with multi-pipeline end-to-end workflow via NAT server.
        
        This tests the orchestrator's ability to:
        - Chain multiple pipeline tools together (Label → Train → Generate)
        - Manage file path dependencies between steps
        - Execute a complex, multi-step plan
        
        Workflow: Unlabeled data → Label → Train model → Generate synthetic data
        
        Requires: NAT server running with: nat serve --config_file backend/agents/workflow.yml
        """
        print("\n" + "="*60)
        print("TEST: Orchestrator - Multi-Pipeline (Label → Train → Generate)")
        print("="*60)
        
        # Find adult.csv
        if adult_csv_path is None:
            adult_csv_path = os.path.join(project_root, 'testing_data', 'adult.csv')
            if not os.path.exists(adult_csv_path):
                adult_csv_path = 'testing_data/adult.csv'
                if not os.path.exists(adult_csv_path):
                    raise FileNotFoundError(f"Could not find adult.csv")
        
        # Prepare test data and holdout
        raw_data_path, ground_truth_path = self.prepare_test_data(adult_csv_path)
        
        # Create holdout test set
        df = pd.read_csv(adult_csv_path)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
        df['income_binary'] = (df['income'].str.strip() == '>50K').astype(int)
        df = df.drop(columns=['income'])
        holdout_df = df.sample(n=500, random_state=123)
        holdout_test_path = os.path.join(self.test_dir, 'holdout_test.parquet')
        holdout_df.to_parquet(holdout_test_path, index=False)
        
        # Craft a comprehensive natural language instruction for the orchestrator
        user_message = f"""Execute the complete DataFoundry pipeline with these steps:

STEP 1: LABEL the unlabeled dataset using weak supervision
- Unlabeled data path: {raw_data_path}
- Ground truth examples path: {ground_truth_path}
- Target AUC: 0.70
- Max attempts: 1

STEP 2: TRAIN a generative model on the labeled data
- Use the labeled data from Step 1
- Holdout test data path: {holdout_test_path}
- Target utility: 80% of baseline
- Max attempts: 1

STEP 3: GENERATE synthetic data using the trained model
- Label value: 1.0
- Number of samples: {num_generate}
- Output format: CSV

Please execute all three steps in sequence and return a JSON object with these keys:
- labeled_data_path
- config_path
- model_path
- preprocessor_path
- synthetic_data_path

DO NOT include any explanations or markdown formatting - ONLY return the JSON object."""
        
        print(f"\nSending multi-pipeline instruction to orchestrator...")
        print(f"Expected workflow: Label → Train → Generate")
        
        try:
            # Call NAT server with longer timeout for multi-step workflow
            response = call_nat_workflow(user_message, timeout=900)
            print(f"\nOrchestrator response received")
            print(f"Response preview: {json.dumps(response, indent=2)[:800]}...")
            
            # Validate new structured response format
            if "file_paths" not in response:
                print("⚠ Warning: Response missing 'file_paths' field (may be old format)")
            else:
                print(f"\n✓ Structured response format detected")
                print(f"  File paths available: {list(response.get('file_paths', {}).keys())}")
                print(f"  Steps completed: {response.get('steps_completed', [])}")
            
            # Extract all required paths using new structured format
            paths = {}
            required_keys = {
                'labeled_data_path': ['labeled_output_path', 'labeled_data_path', 'labeled_file_path'],
                'config_path': ['config_path', 'model_config_path'],
                'model_path': ['model_path', 'trained_model_path'],
                'preprocessor_path': ['preprocessor_path', 'preproc_path'],
                'synthetic_data_path': ['synthetic_output_path', 'synthetic_data_path', 'generated_data_path']
            }
            
            for key, hints in required_keys.items():
                path = extract_path_from_response(response, hints)
                if not path or not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Required file '{key}' not found. "
                        f"Searched for: {hints}. "
                        f"Response file_paths: {json.dumps(response.get('file_paths', {}), indent=2)}"
                    )
                paths[key] = path
            
            # Validate steps_completed includes expected steps
            steps_completed = response.get('steps_completed', [])
            expected_steps = ['labeling', 'training', 'generation']
            missing_steps = [step for step in expected_steps if step not in steps_completed]
            if missing_steps:
                print(f"⚠ Warning: Expected steps not in steps_completed: {missing_steps}")
            else:
                print(f"✓ All expected steps completed: {steps_completed}")
            
            # Validate each output
            df_labeled = pd.read_parquet(paths['labeled_data_path'])
            assert 'label_probability' in df_labeled.columns
            
            with open(paths['config_path'], 'r') as f:
                config = json.load(f)
            model_type = config.get('model_params', {}).get('model_class_name', 'Unknown')
            
            # Validate synthetic data
            synthetic_df = pd.read_csv(paths['synthetic_data_path'])
            
            print(f"\n✓ Orchestrator multi-pipeline successful!")
            print(f"  - Labeled: {len(df_labeled)} samples → {os.path.basename(paths['labeled_data_path'])}")
            print(f"  - Trained: {model_type} model → {os.path.basename(paths['model_path'])}")
            print(f"  - Generated: {len(synthetic_df)} samples → {os.path.basename(paths['synthetic_data_path'])}")
            print(f"  - Steps completed: {steps_completed}")
            
            self.test_results['orchestrator_multi_pipeline'] = {
                'success': True,
                **paths,
                'num_labeled': len(df_labeled),
                'model_type': model_type,
                'num_generated': len(synthetic_df),
                'steps_completed': steps_completed,
                'has_structured_response': 'file_paths' in response
            }
            return self.test_results['orchestrator_multi_pipeline']
            
        except Exception as e:
            print(f"\n✗ Orchestrator multi-pipeline test failed: {e}")
            self.test_results['orchestrator_multi_pipeline'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    def run_all_tests(self):
        """
        Run all agent tests.
        
        Returns:
            Dictionary of test results
        """
        print("\n" + "="*60)
        print("RUNNING ALL AGENT TESTS")
        print("="*60)
        
        try:
            # Test 1: Labeling Pipeline
            self.test_labeling_pipeline(
                target_auc=0.75,  # Lower target for testing
                max_attempts=2     # Fewer attempts for faster testing
            )
            
            # Test 2: Orchestrator Workflow (placeholder)
            # self.test_orchestrator_workflow()
            
            # Print summary
            print("\n" + "="*60)
            print("TEST SUMMARY")
            print("="*60)
            for test_name, result in self.test_results.items():
                status = "✓ PASS" if result.get('success', False) else "✗ FAIL"
                print(f"{status}: {test_name}")
                if result.get('success', False):
                    if 'final_auc' in result:
                        print(f"  - Final AUC: {result['final_auc']:.3f}")
                    if 'num_samples' in result:
                        print(f"  - Samples labeled: {result['num_samples']}")
                else:
                    print(f"  - Error: {result.get('error', 'Unknown error')}")
            
            all_passed = all(r.get('success', False) for r in self.test_results.values())
            print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
            
            return self.test_results
            
        except Exception as e:
            print(f"\n✗ Test suite failed: {e}")
            return self.test_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DataFoundry Agent Pipelines')
    parser.add_argument('--test', type=str, 
                       choices=['all', 'labeling', 'training', 'generation', 'anomaly', 
                               'orchestrator-single', 'orchestrator-multi', 'orchestrator-all'], 
                       default='all',
                       help='Which test(s) to run (default: all)')
    parser.add_argument('--labeled-data', type=str, default=None,
                       help='Path to pre-labeled data (for training pipeline)')
    parser.add_argument('--holdout-test', type=str, default=None,
                       help='Path to holdout test data (for training pipeline)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model config (for generation/anomaly pipelines)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (for generation/anomaly pipelines)')
    parser.add_argument('--preprocessor', type=str, default=None,
                       help='Path to preprocessor (for generation/anomaly pipelines)')
    parser.add_argument('--data-to-scan', type=str, default=None,
                       help='Path to data to scan (for anomaly pipeline)')
    parser.add_argument('--max-attempts', type=int, default=2,
                       help='Maximum attempts per pipeline (default: 2)')
    parser.add_argument('--target-auc', type=float, default=0.75,
                       help='Target AUC for labeling pipeline (default: 0.75)')
    parser.add_argument('--target-utility', type=float, default=0.85,
                       help='Target quality score (0-1) for training pipeline based on reconstruction error (default: 0.85)')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to generate (for generation pipeline, default: 100)')
    parser.add_argument('--label', type=float, default=1.0,
                       help='Label value for generation (default: 1.0)')
    parser.add_argument('--nat-url', type=str, 
                       default=os.getenv('NAT_URL', 'http://localhost:8000'),
                       help='NAT server base URL (default: env NAT_URL or http://localhost:8000)')
    
    args = parser.parse_args()
    
    # Set NAT_URL environment variable if provided via CLI
    if args.nat_url:
        os.environ['NAT_URL'] = args.nat_url
    
    # Run tests
    tester = DataFoundryAgentTester()
    try:
        if args.test == 'labeling' or args.test == 'all':
            print("\n" + "="*60)
            print("RUNNING LABELING PIPELINE TEST")
            print("="*60)
            tester.test_labeling_pipeline(
                target_auc=args.target_auc,
                max_attempts=args.max_attempts
            )
        
        if args.test == 'training' or args.test == 'all':
            print("\n" + "="*60)
            print("RUNNING TRAINING PIPELINE TEST")
            print("="*60)
            tester.test_training_pipeline(
                labeled_data_path=args.labeled_data,
                holdout_test_path=args.holdout_test,
                target_utility_pct=args.target_utility,
                max_attempts=args.max_attempts
            )
        
        if args.test == 'generation' or args.test == 'all':
            print("\n" + "="*60)
            print("RUNNING GENERATION PIPELINE TEST")
            print("="*60)
            tester.test_generation_pipeline(
                config_path=args.config,
                model_path=args.model,
                preprocessor_path=args.preprocessor,
                label=args.label,
                num_samples=args.num_samples
            )
        
        if args.test == 'anomaly' or args.test == 'all':
            print("\n" + "="*60)
            print("RUNNING ANOMALY DETECTION PIPELINE TEST")
            print("="*60)
            tester.test_anomaly_pipeline(
                config_path=args.config,
                model_path=args.model,
                preprocessor_path=args.preprocessor,
                data_to_scan_path=args.data_to_scan
            )
        
        # Orchestrator tests (require NAT server running)
        if args.test == 'orchestrator-single' or args.test == 'orchestrator-all':
            print("\n" + "="*60)
            print("RUNNING ORCHESTRATOR - SINGLE TASK (LABELING) TEST")
            print("="*60)
            print(f"NAT Server URL: {get_nat_url()}")
            print("NOTE: This test requires NAT server to be running:")
            print("  cd /home/justin/Projects/HackUTD2025")
            print("  nat serve --config_file backend/agents/workflow.yml")
            print()
            tester.test_orchestrator_single_labeling()
        
        if args.test == 'orchestrator-multi' or args.test == 'orchestrator-all':
            print("\n" + "="*60)
            print("RUNNING ORCHESTRATOR - MULTI-PIPELINE (LABEL→TRAIN→GENERATE) TEST")
            print("="*60)
            print(f"NAT Server URL: {get_nat_url()}")
            print("NOTE: This test requires NAT server to be running:")
            print("  cd /home/justin/Projects/HackUTD2025")
            print("  nat serve --config_file backend/agents/workflow.yml")
            print()
            tester.test_orchestrator_multi_pipeline(num_generate=args.num_samples)
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        for test_name, result in tester.test_results.items():
            status = "✓ PASS" if result.get('success', False) else "✗ FAIL"
            print(f"{status}: {test_name}")
            if result.get('success', False):
                if test_name == 'labeling_pipeline':
                    print(f"  - Final AUC: {result.get('final_auc', 0):.3f}")
                    print(f"  - Samples labeled: {result.get('num_samples', 0)}")
                elif test_name == 'training_pipeline':
                    print(f"  - Model type: {result.get('model_type', 'Unknown')}")
                    print(f"  - Model saved: {os.path.basename(result.get('model_path', ''))}")
                elif test_name == 'generation_pipeline':
                    print(f"  - Generated {result.get('num_samples', 0)} samples")
                    print(f"  - Features: {result.get('num_features', 0)}")
                    print(f"  - Output: {os.path.basename(result.get('output_path', ''))}")
                elif test_name == 'anomaly_pipeline':
                    print(f"  - Scanned {result.get('num_scanned', 0)} samples")
                    print(f"  - Anomalies detected: {result.get('num_anomalies', 0)}")
                    print(f"  - Report: {os.path.basename(result.get('report_path', ''))}")
                elif test_name == 'orchestrator_single_labeling':
                    print(f"  - Labeled {result.get('num_samples', 0)} samples via orchestrator")
                    print(f"  - Output: {os.path.basename(result.get('labeled_path', ''))}")
                elif test_name == 'orchestrator_multi_pipeline':
                    print(f"  - Labeled: {result.get('num_labeled', 0)} samples")
                    print(f"  - Trained: {result.get('model_type', 'Unknown')} model")
                    print(f"  - Generated: {result.get('num_generated', 0)} samples")
            else:
                print(f"  - Error: {result.get('error', 'Unknown error')}")
        
        all_passed = all(r.get('success', False) for r in tester.test_results.values())
        print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        
        print(f"\nTest files saved in: {tester.test_dir}")
        print("(Call tester.cleanup() to remove test files)")
    except Exception as e:
        print(f"Error: {e}")

