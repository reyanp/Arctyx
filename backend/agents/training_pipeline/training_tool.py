"""
Training Pipeline Tool

This is a self-correcting agentic tool for model training.
It orchestrates a team of sub-agents to generate a config,
train, evaluate, and tune the model.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# Add parent directory to path to access shared config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Add DataFoundry to path
datafoundry_path = os.path.join(os.path.dirname(__file__), '..', '..')
if datafoundry_path not in sys.path:
    sys.path.insert(0, datafoundry_path)

# Import our DataFoundry library's "worker" function
import DataFoundry.trainer as trainer

# Import our clients and Architect agent
from config import (
    NEMOTRON_HIGH_COST_CLIENT,
    NEMOTRON_LOW_COST_CLIENT,
    NEMOTRON_ARCHITECT_AGENT
)

# Import from local utils (same directory)
try:
    from training_pipeline.utils import evaluate_model_utility
except ImportError:
    from utils import evaluate_model_utility


def run_training_pipeline(labeled_data_path: str,
                          holdout_test_path: str,
                          target_utility_pct: float = 0.90,
                          max_attempts: int = 3) -> tuple:
    """
    This is a self-correcting agentic tool for model training.
    It orchestrates a team of sub-agents to generate a config,
    train, evaluate, and tune the model.
    
    Args:
        labeled_data_path: Path to the labeled training data
        holdout_test_path: Path to the holdout test data
        target_utility_pct: Target utility as percentage of baseline (0.0-1.0)
        max_attempts: Maximum number of training attempts
    
    Returns:
        Tuple: (best_config_path, best_model_path, best_preprocessor_path)
    """
    print(f"--- Starting Training Pipeline ---")
    print(f"Goal: Achieve {target_utility_pct:.0%} of baseline F1 score")
    
    best_config_path = ""
    best_model_path = ""
    best_preprocessor_path = ""
    best_utility_score = -1.0
    baseline_score = 0.85  # Will be updated on first evaluation
    utility_history = []
    
    # Get schema info for the Architect's prompt
    try:
        df = pd.read_parquet(labeled_data_path)
        schema_info = str(df.dtypes.to_dict())
        num_samples = len(df)
        print(f"Loaded training data: {num_samples} samples")
    except Exception as e:
        print(f"Failed to load training data: {e}")
        return "", "", ""
    
    # Prepare output directory
    output_dir = os.path.dirname(labeled_data_path) if os.path.dirname(labeled_data_path) else '.'
    
    previous_config_summary = "This is the first attempt. Start with good defaults."
    
    # --- The Self-Correcting Loop ---
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Attempt {attempt} of {max_attempts} ---")
        
        # --- 1. Architect Agent ---
        print("Agent 1 (Architect): Designing model configuration...")
        
        # Get conditioning column and numerical columns
        condition_col = 'label_probability' if 'label_probability' in df.columns else 'income_binary'
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove the condition column from features
        if condition_col in numerical_cols:
            numerical_cols.remove(condition_col)
        
        # Build the complete config directly (don't rely on LLM for critical structure)
        config_data = {
            "data_path": labeled_data_path,
            "output_model_path": os.path.join(output_dir, f'model_attempt_{attempt}.pth'),
            "preprocessor_path": os.path.join(output_dir, f'preprocessor_attempt_{attempt}.joblib'),
            "model_params": {
                "model_template": "tabular_cvae.py",
                "model_class_name": "TabularCVAE",
                "latent_dim": 64 if attempt == 1 else (32 if attempt == 2 else 128),
                "condition_dim": 1,
                "encoder_hidden_layers": [128, 64],
                "decoder_hidden_layers": [64, 128],
                "feature_cols": {
                    "numerical_cols": numerical_cols
                },
                "condition_cols": [condition_col]
            },
            "training_params": {
                "batch_size": 128,  # Larger batch = faster
                "learning_rate": 0.001 if attempt == 1 else 0.0001,
                "epochs": 5  # Much fewer epochs for testing
            }
        }
        
        current_config_json = json.dumps(config_data)
        current_config_path = os.path.join(output_dir, f"config_attempt_{attempt}.json")

        try:
            # Write config directly (programmatic approach for reliability)
            with open(current_config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"Generated config: {current_config_path}")
            print(f"  Model: {config_data['model_params']['model_class_name']}")
            print(f"  Latent dim: {config_data['model_params']['latent_dim']}")
            print(f"  Learning rate: {config_data['training_params']['learning_rate']}")
            print(f"  Features: {len(config_data['model_params']['feature_cols']['numerical_cols'])} numerical cols")
                
        except Exception as e:
            print(f"Error: Failed to create config: {e}")
            break
        
        # --- 2. Trainer (Programmatic Worker) ---
        print("Agent 2 (Trainer): Running model training...")
        try:
            # Call the DataFoundry trainer
            trainer.train_model(config_path=current_config_path)
            
            # Get output paths from config
            current_model_path = config_data.get('output_model_path', f'{output_dir}/model.pth')
            current_preprocessor_path = config_data.get('preprocessor_path', f'{output_dir}/preprocessor.joblib')
            
            # Verify files were created
            if not os.path.exists(current_model_path):
                print(f"Error: Model file not found: {current_model_path}")
                continue
                
            print(f"Training complete: {current_model_path}")
            
        except Exception as e:
            print(f"Error: Training worker failed: {e}")
            continue  # Try again
        
        # --- 3. Model Evaluator (Programmatic Worker) ---
        print("Agent 3 (Evaluator): Checking model utility...")
        eval_results = evaluate_model_utility(
            current_config_path,
            current_model_path,
            current_preprocessor_path,
            holdout_test_path
        )
        
        current_utility_score = eval_results.get("utility_f1", 0.0)
        baseline_score = eval_results.get("baseline_f1", baseline_score)
        target_score = baseline_score * target_utility_pct
        
        print(f"Attempt {attempt} Utility F1: {current_utility_score:.4f} (Target: {target_score:.4f})")
        
        utility_history.append(current_utility_score)
        
        # Update best model if improved
        if current_utility_score > best_utility_score:
            best_utility_score = current_utility_score
            best_config_path = current_config_path
            best_model_path = current_model_path
            best_preprocessor_path = current_preprocessor_path
        
        # --- 4. HyperTuner (LLM-based Agent / Loop Controller) ---
        print(f"Agent 4 (HyperTuner): Analyzing score: {current_utility_score:.4f}")
        
        # Check for immediate success
        if current_utility_score >= target_score:
            print("Target score achieved! Pipeline successful.")
            break
        
        # If not success and not max attempts, ask the Tuner agent
        if attempt < max_attempts:
            # Build utility history string
            utility_history_str = ", ".join([f"Attempt {i+1}: {score:.4f}" for i, score in enumerate(utility_history)])
            
            tuner_prompt = f"""You are a HyperTuner agent. Your job is to decide if we should continue training.

Current State:
- Attempt: {attempt} of {max_attempts}
- Baseline F1: {baseline_score:.4f}
- Target F1: {target_score:.4f}
- Current F1: {current_utility_score:.4f}
- Best F1 So Far: {best_utility_score:.4f}
- Utility History: {utility_history_str}

Analyze this. Consider:
1. Is the current score close enough to target to be acceptable?
2. Is the score improving with each attempt?
3. Is improvement stalling or getting worse?

Your response must be ONLY one of these two options:
- 'RETRY' if the score has good potential to improve with different hyperparameters
- 'STOP_FAILURE' if the score is low and improvement seems stalled

Response:"""
            
            try:
                decision = NEMOTRON_LOW_COST_CLIENT.invoke(tuner_prompt).content.strip()
                print(f"Tuner Agent decision: {decision}")
                
                if decision == "STOP_FAILURE":
                    print("Tuner decided improvement is stalled. Stopping loop.")
                    break
                elif decision == "RETRY":
                    print("Tuner decided to retry. Architect will generate new config...")
                    # Update summary for next iteration
                    previous_config_summary = f"Attempt {attempt}: {config_data.get('model_type')} with latent_dim={config_data.get('latent_dim')}, lr={config_data.get('learning_rate')} → F1={current_utility_score:.4f}"
                    continue
                else:
                    # Default to retry
                    continue
            
            except Exception as e:
                print(f"Error: Tuner Agent failed: {e}. Defaulting to retry.")
                continue
        
        else:
            print("Max attempts reached. Returning best-effort model.")
            break
    
    print(f"--- Training Pipeline Complete ---")
    print(f"Best Utility F1 achieved: {best_utility_score:.4f}")
    print(f"Best model saved to: {best_model_path}")
    
    return best_config_path, best_model_path, best_preprocessor_path

