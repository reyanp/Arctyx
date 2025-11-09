import re
import os
import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel

# Import our clients and helper functions
import sys
import os
# Add parent directory to path to access shared config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    NEMOTRON_HIGH_COST_CLIENT,
    NEMOTRON_LOW_COST_CLIENT,
    NEMOTRON_WRITER_AGENT
)

# Import from local utils (same directory)
from utils import evaluate_labels


def _load_lfs_from_string(lf_code_string: str, current_lfs: list) -> list:
    """
    Dynamically executes generated Python code to get new LF functions.
    Handles code that may be embedded in thinking tags or other text.
    """
    new_lfs_found = []
    
    # Handle escaped newlines from JSON (\\n -> \n)
    # This is common when code comes from tool call arguments
    if '\\n' in lf_code_string and '\n' not in lf_code_string:
        lf_code_string = lf_code_string.replace('\\n', '\n')
    
    # Remove leading whitespace from each line (dedent)
    import textwrap
    lf_code_string = textwrap.dedent(lf_code_string)
    
    # Clean up the code string - remove thinking tags and extract just the code
    # Look for code after thinking tags or in code blocks
    if '<think>' in lf_code_string:
        # Try to find where thinking ends
        think_end = lf_code_string.find('</think>')
        if think_end > 0:
            lf_code_string = lf_code_string[think_end + len('</think>'):].strip()
    
    # Remove markdown code blocks if present
    if '```python' in lf_code_string:
        start = lf_code_string.find('```python') + len('```python')
        end = lf_code_string.find('```', start)
        if end > start:
            lf_code_string = lf_code_string[start:end].strip()
    elif '```' in lf_code_string:
        start = lf_code_string.find('```') + 3
        end = lf_code_string.find('```', start)
        if end > start:
            lf_code_string = lf_code_string[start:end].strip()
    
    # Use a simple regex to find function names
    lf_names = re.findall(r"@labeling_function\(\)\s*def (\w+)\(x\):", lf_code_string)
    
    if not lf_names:
        return current_lfs
    
    # Execute the code in a temp namespace
    temp_namespace = {}
    exec_globals = {
        "labeling_function": labeling_function,
        "pd": pd,
        "np": __import__("numpy")
    }
    try:
        exec(lf_code_string, exec_globals, temp_namespace)
    except Exception as e:
        return current_lfs
    
    for name in lf_names:
        if name in temp_namespace:
            new_lfs_found.append(temp_namespace[name])
    
    # Avoid adding duplicate functions
    existing_names = {lf.name for lf in current_lfs}
    final_lfs = list(current_lfs)
    for new_lf in new_lfs_found:
        if new_lf.name not in existing_names:
            final_lfs.append(new_lf)
            
    return final_lfs


def run_labeling_pipeline(user_goal: str, 
                          raw_data_path: str, 
                          hand_labeled_examples_path: str, 
                          target_auc_score: float = 0.85, 
                          max_attempts: int = 3) -> str:
    """
    This is a self-correcting agentic tool for data labeling.
    It orchestrates a team of sub-agents to generate and refine labels.
    """
    print(f"--- Starting Labeling Pipeline: Goal='{user_goal}' ---")
    
    current_lfs = []
    best_auc = 0.0
    best_labeled_file = ""
    auc_history = []  # Track AUC scores across attempts for the Tuner
    
    # Load raw data *once*
    try:
        df = pd.read_parquet(raw_data_path)
        schema_info = str(df.info())
    except Exception as e:
        print(f"Failed to load raw data: {e}")
        return ""
        
    # --- The Self-Correcting Loop ---
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Attempt {attempt} of {max_attempts} ---")
        
        # --- 1. Heuristic Writer Agent ---
        print("Agent 1 (Heuristic Writer): Generating new labeling functions...")
        existing_lf_names = [lf.name for lf in current_lfs]
        
        writer_prompt = f"""Write Python labeling functions NOW. NO thinking, NO explanations - ONLY code.

TASK: Create 3-5 NEW Python functions for: {user_goal}

COLUMNS AVAILABLE:
{schema_info}

ALREADY WRITTEN: {existing_lf_names}

FORMAT (copy this exactly):

@labeling_function()
def lf_NAME_HERE(x):
    try:
        # Use x.get('column', default)
        value = x.get('age', 0)
        if value > 40:
            return 1
        return 0
    except:
        return -1

NOW WRITE YOUR CODE (ONLY CODE, NO TEXT):"""
        
        try:
            # Call the agent with the tool-binding
            response = NEMOTRON_WRITER_AGENT.invoke(writer_prompt)
            
            # Extract code from either tool call or content
            if response.tool_calls:
                # Model used tool calling (preferred path)
                lf_code_string = response.tool_calls[0]['args']['code']
                current_lfs = _load_lfs_from_string(lf_code_string, current_lfs)
            else:
                # Fallback: Model put code in content instead of tool call
                if hasattr(response, 'content') and response.content:
                    lf_code_string = response.content
                    current_lfs = _load_lfs_from_string(lf_code_string, current_lfs)
                    if not current_lfs:
                        print(f"Error: Failed to generate labeling functions from model response.")
                        break
                else:
                    print(f"Error: Model returned no usable response.")
                    break
            
            if not current_lfs:
                print(f"Error: No labeling functions generated. Aborting.")
                break
                
        except Exception as e:
            print(f"Error: Heuristic Writer failed: {e}")
            break

        # --- 2. Labeler (Programmatic Worker) ---
        print("Agent 2 (Labeler): Applying LFs and running Snorkel...")
        # Save in the same directory as the raw data for organization
        output_dir = os.path.dirname(raw_data_path) if os.path.dirname(raw_data_path) else '.'
        labeled_file_path = os.path.join(output_dir, f"labeled_attempt_{attempt}.parquet")
        try:
            # Apply all LFs to the dataframe
            applier = PandasLFApplier(lfs=current_lfs)
            L_train = applier.apply(df=df)
            
            # Train the Snorkel Label Model
            label_model = LabelModel(cardinality=2, verbose=False)
            label_model.fit(L_train=L_train, n_epochs=500)
            
            # Get probabilities and save
            probs = label_model.predict_proba(L=L_train)
            df_labeled = df.copy()  # Start with the original data
            df_labeled['label_probability'] = probs[:, 1]
            df_labeled.to_parquet(labeled_file_path, index=True)  # Save WITH index
            
        except Exception as e:
            print(f"Error: Labeling worker failed: {e}")
            continue  # Try again

        # --- 3. Label Evaluator (Programmatic Worker) ---
        print("Agent 3 (Evaluator): Checking score against ground truth...")
        eval_results = evaluate_labels(labeled_file_path, hand_labeled_examples_path)
        current_auc = eval_results.get("roc_auc", 0.0)
        print(f"Attempt {attempt} AUC: {current_auc:.4f} (Target: {target_auc_score})")
        
        # Track AUC history for the Tuner
        auc_history.append(current_auc)
        
        if current_auc > best_auc:
            best_auc = current_auc
            best_labeled_file = labeled_file_path

        # --- 4. Label Tuner (LLM-based Agent / Loop Controller) ---
        print(f"Agent 4 (Tuner): Analyzing score: {current_auc:.4f}")
        
        # Check for immediate success first
        if current_auc >= target_auc_score:
            print("Target score achieved! Pipeline successful.")
            break
        
        # If not success, and not max attempts, ask the Tuner agent
        if attempt < max_attempts:
            # Build AUC history string for context
            auc_history_str = ", ".join([f"Attempt {i+1}: {auc:.4f}" for i, auc in enumerate(auc_history)])
            
            tuner_prompt = f"""
You are a Label Tuner agent. Your job is to decide if we should continue trying to improve our data labels.

Current State:
- Attempt: {attempt} of {max_attempts}
- Current AUC Score: {current_auc:.4f}
- Target AUC Score: {target_auc_score}
- Best AUC So Far: {best_auc:.4f}
- AUC History: {auc_history_str}
- Number of Labeling Functions: {len(current_lfs)}

Analyze this. Consider:
1. Is the current score close enough to target to be acceptable?
2. Is the score improving with each attempt?
3. Is improvement stalling or getting worse?

Your response must be ONLY one of these three options:
- 'STOP_SUCCESS' if the score is good enough (even if below target)
- 'RETRY_FOR_MORE_LFS' if the score has good potential to improve with more labeling functions
- 'STOP_FAILURE' if the score is low and improvement seems stalled

Response:"""
            
            # Call the low-cost brain for decision-making
            try:
                decision = NEMOTRON_LOW_COST_CLIENT.invoke(tuner_prompt).content.strip()
                print(f"Tuner Agent decision: {decision}")
                
                if decision == "STOP_SUCCESS":
                    print("Tuner decided the score is good enough.")
                    break
                elif decision == "STOP_FAILURE":
                    print("Tuner decided the improvement is stalled. Stopping loop.")
                    break
                elif decision == "RETRY_FOR_MORE_LFS":
                    print("Tuner decided to retry. Requesting more LFs for next loop...")
                    continue  # This will go to the next loop iteration
                else:
                    # Default to retry if response is unexpected
                    continue
            
            except Exception as e:
                print(f"Error: Tuner Agent failed: {e}. Defaulting to retry.")
                continue
        
        else:
            print("Max attempts reached. Returning best-effort file.")
            break
        
        # The loop will now automatically re-run the Heuristic Writer (Step 1)
        # and it will see the updated list of `existing_lf_names`.
        
    print(f"--- Labeling Pipeline Complete ---")
    print(f"Best AUC achieved: {best_auc:.4f}")
    print(f"Best labeled file: {best_labeled_file}")
    
    return best_labeled_file
