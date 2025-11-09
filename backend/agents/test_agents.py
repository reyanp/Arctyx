import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np

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
    # Try alternative import path (when running as module or direct script)
    try:
        from agents.labeling_pipeline.labeling_tool import run_labeling_pipeline
    except ImportError:
        # Last resort: direct import from labeling_pipeline directory
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
            print(f"\nTarget Utility: {target_utility_pct:.0%} of baseline")
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
            import json
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
    
    def test_orchestrator_workflow(self):
        """
        Test the full orchestrator workflow (to be implemented).
        This will test the complete end-to-end pipeline orchestration.
        """
        print("\n" + "="*60)
        print("TEST: Orchestrator Workflow")
        print("="*60)
        print("⚠ Not yet implemented - will test full orchestrator pipeline")
        # TODO: Implement orchestrator workflow test
        pass
    
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
                       choices=['all', 'labeling', 'training', 'generation', 'anomaly'], 
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
                       help='Target utility percentage for training pipeline (default: 0.85)')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to generate (for generation pipeline, default: 100)')
    parser.add_argument('--label', type=float, default=1.0,
                       help='Label value for generation (default: 1.0)')
    
    args = parser.parse_args()
    
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
            else:
                print(f"  - Error: {result.get('error', 'Unknown error')}")
        
        all_passed = all(r.get('success', False) for r in tester.test_results.values())
        print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        
        print(f"\nTest files saved in: {tester.test_dir}")
        print("(Call tester.cleanup() to remove test files)")
    except Exception as e:
        print(f"Error: {e}")

