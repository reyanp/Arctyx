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
    from training_pipeline.utils import evaluate_model_quality
except ImportError:
    from utils import evaluate_model_quality


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
    print(f"Goal: Train a generative model with quality score >= {target_utility_pct:.0%}")
    
    best_config_path = ""
    best_model_path = ""
    best_preprocessor_path = ""
    best_quality_score = -1.0
    quality_history = []
    
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
        
        # Infer columns - separate numerical and categorical (matching Flask API logic)
        all_cols = df.columns.tolist()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Categorical columns are non-numerical columns that aren't label columns
        label_cols = ['label', 'label_probability', 'income_binary', 'income']
        categorical_cols = [col for col in all_cols 
                          if col not in numerical_cols and col not in label_cols]
        
        # Find condition column
        condition_col = None
        for col in label_cols:
            if col in numerical_cols:
                condition_col = col
                numerical_cols.remove(col)
                break
        
        if not condition_col:
            # Try to find label_probability or income_binary
            if 'label_probability' in all_cols:
                condition_col = 'label_probability'
            elif 'income_binary' in all_cols:
                condition_col = 'income_binary'
            elif numerical_cols:
                condition_col = numerical_cols[-1]  # Use last numerical column as fallback
                numerical_cols = numerical_cols[:-1]
            else:
                print(f"Error: No suitable condition column found")
                continue
        
        # Available models mapping
        available_models = {
            'tabular_cvae': ('tabular_cvae.py', 'TabularCVAE', 'Conditional VAE for tabular numerical data'),
            'mixed_data_cvae': ('mixed_data_cvae.py', 'MixedDataCVAE', 'Conditional VAE for mixed numerical and categorical data'),
            'tabular_vae_gmm': ('tabular_vae_gmm.py', 'TabularVAEGMM', 'VAE with Gaussian Mixture Model for tabular data'),
            'tabular_ctgan': ('tabular_ctgan.py', 'TabularCTGAN', 'CTGAN for tabular data generation'),
            'text_cvae': ('text_cvae.py', 'TextCVAE', 'Conditional VAE for text data')
        }
        
        # Have LLM choose a model (unless categorical columns force MixedDataCVAE)
        if categorical_cols:
            # Force MixedDataCVAE when categorical columns exist
            model_key = 'mixed_data_cvae'
            model_template, model_class, model_desc = available_models[model_key]
            print(f"  Detected {len(categorical_cols)} categorical columns. Forcing MixedDataCVAE")
        else:
            # Let LLM choose from available models with retry logic
            model_list = "\n".join([f"- {key}: {desc}" for key, (_, _, desc) in available_models.items()])
            
            max_llm_retries = 3
            retry_count = 0
            model_key = None
            
            while retry_count < max_llm_retries and not model_key:
                try:
                    base_prompt = f"""You are an Architect agent designing a generative model configuration.

Dataset Information:
- Number of samples: {num_samples}
- Number of numerical features: {len(numerical_cols)}
- Number of categorical features: {len(categorical_cols)}
- Numerical columns: {numerical_cols[:5]}{'...' if len(numerical_cols) > 5 else ''}
- Data types: {schema_info}

Previous attempts: {previous_config_summary}

Available Models:
{model_list}

Based on the dataset characteristics, choose the most appropriate model for generating synthetic data.
Your response must be ONLY one of these model keys: {', '.join(available_models.keys())}

Response (model key only):"""
                    
                    # Add aggressive feedback on retries
                    if retry_count > 0:
                        error_feedback = "\n\n" + "="*60 + "\n"
                        error_feedback += f"ERROR: Your previous response (attempt {retry_count}) was INVALID.\n"
                        error_feedback += f"You MUST respond with ONLY one of these exact model keys: {', '.join(available_models.keys())}\n"
                        error_feedback += "DO NOT include any explanations, thinking, or additional text.\n"
                        if retry_count >= 2:
                            error_feedback += "CRITICAL: This is your FINAL attempt. Return ONLY the model key NOW.\n"
                        error_feedback += "="*60 + "\n\n"
                        architect_prompt = error_feedback + base_prompt
                    else:
                        architect_prompt = base_prompt
                    
                    if retry_count > 0:
                        print(f"  Retry attempt {retry_count}/{max_llm_retries - 1} with stricter prompt...")
                    
                    llm_response = NEMOTRON_ARCHITECT_AGENT.invoke(architect_prompt)
                    
                    # Extract content from response
                    response_text = None
                    if hasattr(llm_response, 'content'):
                        content = llm_response.content
                        if isinstance(content, str):
                            response_text = content.strip()
                        elif hasattr(content, 'text'):
                            response_text = content.text.strip()
                        else:
                            response_text = str(content).strip()
                    
                    if not response_text:
                        retry_count += 1
                        if retry_count < max_llm_retries:
                            print(f"  Error: Model returned no usable response. Retrying...")
                            continue
                        else:
                            print(f"  Error: Model returned no usable response after {max_llm_retries} attempts. Defaulting to tabular_cvae")
                            model_key = 'tabular_cvae'
                            break
                    
                    print(f"  Architect agent response: {response_text}")
                    
                    # Extract model key from response (handle cases where LLM adds extra text)
                    for key in available_models.keys():
                        if key.lower() in response_text.lower():
                            model_key = key
                            break
                    
                    if not model_key:
                        retry_count += 1
                        if retry_count < max_llm_retries:
                            print(f"  Error: Could not parse model choice from response. Retrying...")
                            continue
                        else:
                            print(f"  Warning: Could not parse model choice after {max_llm_retries} attempts. Defaulting to tabular_cvae")
                            model_key = 'tabular_cvae'
                    
                    if model_key:
                        if retry_count > 0:
                            print(f"  ✓ Successfully parsed model choice on retry {retry_count}")
                        break
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_llm_retries:
                        print(f"  Error: Architect agent failed: {e}. Retrying...")
                        continue
                    else:
                        print(f"  Error: Architect agent failed after {max_llm_retries} attempts: {e}. Defaulting to tabular_cvae")
                        model_key = 'tabular_cvae'
                        break
            
            model_template, model_class, model_desc = available_models[model_key]
            print(f"  Selected model: {model_class} ({model_desc})")
        
        # Build feature_cols dict - include both numerical and categorical if they exist
        feature_cols = {}
        if numerical_cols:
            feature_cols['numerical_cols'] = numerical_cols
        if categorical_cols:
            feature_cols['categorical_cols'] = categorical_cols
        
        # Build the complete config directly (don't rely on LLM for critical structure)
        model_params = {
            "model_template": model_template,
            "model_class_name": model_class,
            "latent_dim": 64 if attempt == 1 else (32 if attempt == 2 else 128),
            "condition_dim": 1,
            "encoder_hidden_layers": [128, 64],
            "decoder_hidden_layers": [64, 128],
            "feature_cols": feature_cols,
            "condition_cols": [condition_col]
        }
        
        # If using MixedDataCVAE with categorical columns, calculate embedding dimensions
        if model_class == 'MixedDataCVAE' and categorical_cols:
            categorical_embed_dims = []
            for col in categorical_cols:
                n_categories = df[col].nunique()
                embed_dim = min(8, max(4, n_categories // 2))  # Embedding dim between 4-8
                categorical_embed_dims.append((n_categories, embed_dim))
            model_params['categorical_embed_dims'] = categorical_embed_dims
            model_params['numerical_dim'] = len(numerical_cols) if numerical_cols else 0
        
        # Calculate input_dim for models that need it (TabularCVAE, TabularVAEGMM, TabularCTGAN)
        if model_class in ['TabularCVAE', 'TabularVAEGMM', 'TabularCTGAN']:
            num_numerical = len(numerical_cols) if numerical_cols else 0
            num_categorical = len(categorical_cols) if categorical_cols else 0
            model_params['input_dim'] = num_numerical + num_categorical
        
        config_data = {
            "data_path": labeled_data_path,
            "output_model_path": os.path.join(output_dir, f'model_attempt_{attempt}.pth'),
            "preprocessor_path": os.path.join(output_dir, f'preprocessor_attempt_{attempt}.joblib'),
            "model_params": model_params,
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
            num_numerical = len(config_data['model_params']['feature_cols'].get('numerical_cols', []))
            num_categorical = len(config_data['model_params']['feature_cols'].get('categorical_cols', []))
            print(f"  Features: {num_numerical} numerical cols, {num_categorical} categorical cols")
                
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
        print("Agent 3 (Evaluator): Checking model quality...")
        eval_results = evaluate_model_quality(
            current_config_path,
            current_model_path,
            current_preprocessor_path,
            holdout_test_path
        )
        
        current_quality_score = eval_results.get("quality_score", 0.0)
        reconstruction_error = eval_results.get("reconstruction_error", float('inf'))
        target_score = target_utility_pct  # Direct target (0-1 scale)
        
        print(f"Attempt {attempt} Quality Score: {current_quality_score:.4f} (Target: {target_score:.4f})")
        print(f"  Reconstruction Error: {reconstruction_error:.6f}")
        
        quality_history.append(current_quality_score)
        
        # Update best model if improved
        if current_quality_score > best_quality_score:
            best_quality_score = current_quality_score
            best_config_path = current_config_path
            best_model_path = current_model_path
            best_preprocessor_path = current_preprocessor_path
        
        # --- 4. HyperTuner (LLM-based Agent / Loop Controller) ---
        print(f"Agent 4 (HyperTuner): Analyzing quality score: {current_quality_score:.4f}")
        
        # Check for immediate success
        if current_quality_score >= target_score:
            print("Target quality score achieved! Pipeline successful.")
            break
        
        # If not success and not max attempts, ask the Tuner agent
        if attempt < max_attempts:
            # Build quality history string
            quality_history_str = ", ".join([f"Attempt {i+1}: {score:.4f}" for i, score in enumerate(quality_history)])
            
            tuner_prompt = f"""You are a HyperTuner agent. Your job is to decide if we should continue training.

Current State:
- Attempt: {attempt} of {max_attempts}
- Target Quality Score: {target_score:.4f}
- Current Quality Score: {current_quality_score:.4f}
- Best Quality Score So Far: {best_quality_score:.4f}
- Reconstruction Error: {reconstruction_error:.6f}
- Quality History: {quality_history_str}

Analyze this. Consider:
1. Is the current quality score close enough to target to be acceptable?
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
                    previous_config_summary = f"Attempt {attempt}: {config_data.get('model_type')} with latent_dim={config_data.get('latent_dim')}, lr={config_data.get('learning_rate')} → Quality={current_quality_score:.4f}"
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
    print(f"Best Quality Score achieved: {best_quality_score:.4f}")
    print(f"Best model saved to: {best_model_path}")
    
    return best_config_path, best_model_path, best_preprocessor_path

