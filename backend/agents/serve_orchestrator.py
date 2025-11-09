"""
DataFoundry Orchestrator Server using NVIDIA NeMo Agent Toolkit Python API.

This script creates a ReAct agent that can orchestrate our custom Python pipeline tools
(labeling, training, generation, anomaly detection) using NAT's Python API instead of YAML configuration.

Usage:
    python serve_orchestrator.py
"""

import asyncio
import os
import re
import sys
import uvicorn
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

# Add agents directory to path
agents_dir = Path(__file__).parent
sys.path.insert(0, str(agents_dir))

# Load environment variables
load_dotenv()

# Import the pipeline functions
from labeling_pipeline.labeling_tool import run_labeling_pipeline
from training_pipeline.training_tool import run_training_pipeline
from generation_pipeline.generation_tool import run_generation_pipeline
from anomaly_pipeline.anomaly_tool import run_anomaly_pipeline


# Define FastAPI app
app = FastAPI(title="DataFoundry Orchestrator")


# Wrap our pipeline functions as LangChain tools
@tool
def label_dataset(user_goal: str, raw_data_path: str, hand_labeled_examples_path: str = "", 
                   target_auc_score: float = 0.75, max_attempts: int = 2) -> str:
    """
    Label an unlabeled dataset using weak supervision. 
    
    **IMPORTANT**: Only use this tool if the user EXPLICITLY requests weak supervision labeling.
    Most workflows should skip this and go directly to train_model instead.
    
    Args:
        user_goal: Natural language description of the labeling goal
        raw_data_path: Path to unlabeled parquet file
        hand_labeled_examples_path: Path to ground truth parquet file (REQUIRED for evaluation)
        target_auc_score: Target AUC score (default 0.75)
        max_attempts: Maximum attempts (default 2)
    
    Returns:
        Path to the labeled parquet file, or empty string if labeling fails
    """
    return run_labeling_pipeline(
        user_goal=user_goal,
        raw_data_path=raw_data_path,
        hand_labeled_examples_path=hand_labeled_examples_path,
        target_auc_score=target_auc_score,
        max_attempts=max_attempts
    )


@tool
def train_model(labeled_data_path: str, holdout_test_path: str = "",
                target_utility_pct: float = 0.85, max_attempts: int = 2) -> dict:
    """
    Train a generative model on labeled data.
    
    Args:
        labeled_data_path: Path to labeled parquet file (REQUIRED)
        holdout_test_path: Path to holdout test parquet file (optional, defaults to empty for no testing)
        target_utility_pct: Target utility percentage (default 0.85)
        max_attempts: Maximum attempts (default 2)
    
    Returns:
        Dictionary with keys: config_path, model_path, preprocessor_path
    """
    if not labeled_data_path:
        error_msg = "ERROR: labeled_data_path is empty! Cannot proceed with training."
        print(error_msg)
        return error_msg
    
    config_path, model_path, preprocessor_path = run_training_pipeline(
        labeled_data_path=labeled_data_path,
        holdout_test_path=holdout_test_path,
        target_utility_pct=target_utility_pct,
        max_attempts=max_attempts
    )
    
    return (
        f"config_path: {config_path}\n"
        f"model_path: {model_path}\n"
        f"preprocessor_path: {preprocessor_path}"
    )


@tool
def generate_synthetic_data(config_path: str, model_path: str, preprocessor_path: str,
                            label: float = 1.0, num_to_generate: int = 100,
                            output_format: str = 'csv') -> str:
    """
    Generate synthetic data from a trained model.
    
    Args:
        config_path: Path to model config JSON
        model_path: Path to trained model PTH file
        preprocessor_path: Path to preprocessor joblib file
        label: Label value for generation (default 1.0)
        num_to_generate: Number of samples to generate (default 100)
        output_format: Output format 'csv' or 'pt' (default 'csv')
    
    Returns:
        Path to generated synthetic data file
    """
    if not all([config_path, model_path, preprocessor_path]):
        error_msg = (
            "ERROR: Missing required paths for generation! "
            f"config={config_path}, model={model_path}, preprocessor={preprocessor_path}"
        )
        print(error_msg)
        return error_msg
    
    result = run_generation_pipeline(
        config_path=config_path,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        label=label,
        num_to_generate=num_to_generate,
        output_format=output_format
    )
    
    return f"synthetic_output_path: {result}"


@tool
def detect_anomalies(config_path: str, model_path: str, preprocessor_path: str,
                     data_to_scan_path: str) -> str:
    """
    Detect anomalies in a dataset using a trained model.
    
    Args:
        config_path: Path to model config JSON
        model_path: Path to trained model PTH file
        preprocessor_path: Path to preprocessor joblib file
        data_to_scan_path: Path to data parquet file to scan
    
    Returns:
        Path to anomaly report parquet file
    """
    return run_anomaly_pipeline(
        config_path=config_path,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        data_to_scan_path=data_to_scan_path
    )


# Create the LLM (high-cost brain for orchestration)
llm = ChatNVIDIA(
    base_url="https://integrate.api.nvidia.com/v1/",
    model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
    temperature=0.7
)

# System prompt for the orchestrator
SYSTEM_PROMPT = """You are a master Orchestrator for the DataFoundry platform.
Your job is to take a high-level, natural-language user goal and break it down
into a step-by-step execution plan using your available tools.

Available tools:
- label_dataset: Label unlabeled data using weak supervision (ONLY use if explicitly requested)
- train_model: Train a generative model on data with existing labels
- generate_synthetic_data: Generate synthetic data from a trained model
- detect_anomalies: Find anomalies in a dataset using a trained model

CRITICAL WORKFLOW RULES:
1. **DEFAULT WORKFLOW** (99% of requests): SKIP labeling entirely!
   - Users provide data that ALREADY HAS labels or condition columns
   - Go DIRECTLY to: train_model → generate_synthetic_data → STOP
   
2. **ONLY use label_dataset if**:
   - User explicitly says "use weak supervision" or "create labeling functions"
   - User says "I need to label my unlabeled data"
   - User provides a hand_labeled_examples_path parameter
   
3. **NEVER use label_dataset if**:
   - User just says "generate synthetic data" (this means use existing labels!)
   - User says "generate a label/condition" (this means train on existing column!)
   - No hand_labeled_examples_path is provided
   - User wants to "create synthetic samples" (this is generation, not labeling!)

4. When calling tools, extract file paths from their return values
5. Pass file paths between tools (e.g., training output → generation input)
6. After completing the workflow, STOP and report the results
7. At the end, you MUST return file paths using these EXACT patterns:
   - For training: "config_path: /path/to/config.json" and "model_path: /path/to/model.pth" and "preprocessor_path: /path/to/preprocessor.joblib"
   - For generation: "synthetic_output_path: /path/to/synthetic.csv"

MANDATORY WORKFLOW (ALWAYS follow this):
1. FIRST: Call train_model with the provided dataset_path as labeled_data_path
   - This will return config_path, model_path, preprocessor_path
2. SECOND: Call generate_synthetic_data with the paths from train_model output
   - Use the config_path, model_path, preprocessor_path returned from training
3. THIRD: STOP and report results with file paths

DO NOT skip training! Training is REQUIRED first!
DO NOT call generate_synthetic_data without training first!
DO NOT repeatedly call the same tool. If a tool succeeds, move to the next step.
DO NOT try to fix errors by retrying - report the error and stop.
DO NOT use label_dataset unless the user EXPLICITLY requests weak supervision labeling.

Example response:
"I have trained the model and generated synthetic data:
config_path: /tmp/config.json
model_path: /tmp/model.pth
preprocessor_path: /tmp/preprocessor.joblib
synthetic_output_path: /tmp/synthetic.csv"

Plan your steps, execute them ONCE each, and confirm the final output with the file paths."""

# Create the ReAct agent with system prompt and recursion limit
tools = [label_dataset, train_model, generate_synthetic_data, detect_anomalies]

# Wrap LLM with system prompt
llm_with_prompt = llm.bind(system=SYSTEM_PROMPT)

# Create agent with max iterations to prevent infinite loops
agent = create_react_agent(llm_with_prompt, tools)


# Request/Response models
class GenerateRequest(BaseModel):
    input_message: str


class GenerateResponse(BaseModel):
    output: str
    file_paths: Dict[str, Optional[str]]
    steps_completed: List[str]


def extract_file_paths_from_response(response_text: str) -> Dict[str, Optional[str]]:
    """
    Extract file paths from the agent's response text.
    Looks for patterns like "key: /path/to/file" or "key: path/to/file"
    """
    file_paths = {
        "labeled_output_path": None,
        "config_path": None,
        "model_path": None,
        "preprocessor_path": None,
        "synthetic_output_path": None,
        "anomaly_report_path": None
    }
    
    # Patterns to match various formats
    patterns = {
        "labeled_output_path": [
            r"labeled_output_path:\s*([^\s\n]+)",
            r"labeled_data_path:\s*([^\s\n]+)",
            r"labeled.*?path:\s*([^\s\n]+)",
        ],
        "config_path": [
            r"config_path:\s*([^\s\n]+)",
            r"config.*?path:\s*([^\s\n]+)",
        ],
        "model_path": [
            r"model_path:\s*([^\s\n]+)",
            r"model.*?path:\s*([^\s\n]+)",
        ],
        "preprocessor_path": [
            r"preprocessor_path:\s*([^\s\n]+)",
            r"preprocessor.*?path:\s*([^\s\n]+)",
        ],
        "synthetic_output_path": [
            r"synthetic_output_path:\s*([^\s\n]+)",
            r"synthetic_data_path:\s*([^\s\n]+)",
            r"synthetic.*?path:\s*([^\s\n]+)",
        ],
        "anomaly_report_path": [
            r"anomaly_report_path:\s*([^\s\n]+)",
            r"anomaly.*?path:\s*([^\s\n]+)",
        ]
    }
    
    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                path = match.group(1).strip()
                # Remove trailing punctuation that might have been captured
                path = path.rstrip('.,;!?')
                if path and os.path.exists(path):
                    file_paths[key] = path
                    break
    
    return file_paths


def extract_steps_completed(response_text: str, agent_messages: List) -> List[str]:
    """
    Determine which steps were completed by examining tool calls in the agent's execution.
    """
    steps = []
    
    # Check agent messages for tool calls
    for message in agent_messages:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.get('name', '')
                if tool_name == 'label_dataset':
                    steps.append('labeling')
                elif tool_name == 'train_model':
                    steps.append('training')
                elif tool_name == 'generate_synthetic_data':
                    steps.append('generation')
                elif tool_name == 'detect_anomalies':
                    steps.append('anomaly_detection')
    
    # Also check response text for keywords
    response_lower = response_text.lower()
    if 'label' in response_lower and 'labeled' in response_lower:
        if 'labeling' not in steps:
            steps.append('labeling')
    if 'train' in response_lower or 'model' in response_lower:
        if 'training' not in steps:
            steps.append('training')
    if 'generat' in response_lower or 'synthetic' in response_lower:
        if 'generation' not in steps:
            steps.append('generation')
    if 'anomal' in response_lower:
        if 'anomaly_detection' not in steps:
            steps.append('anomaly_detection')
    
    return list(set(steps))  # Remove duplicates


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Main endpoint that receives natural language instructions and runs the orchestrator.
    
    Returns structured JSON with:
    - output: The natural language response from the agent
    - file_paths: Dictionary of file paths for each step
    - steps_completed: List of steps that were executed
    """
    try:
        # Strict fallback: If a dataset path is provided, run training -> generation programmatically
        # This prevents the LLM from looping or skipping required steps
        ds_match = re.search(r"Dataset path:\s*(.+)", request.input_message)
        if ds_match:
            ds_path = ds_match.group(1).strip()
            if os.path.exists(ds_path):
                # Extract number of samples to generate (default 100)
                num_match = re.search(r"(\d+)\s+(?:sample|samples)", request.input_message, re.IGNORECASE)
                num_to_generate = int(num_match.group(1)) if num_match else 100
                # Default label if not specified
                label = 1.0

                # Step 1: Train model (skip evaluation if no holdout provided)
                config_path, model_path, preprocessor_path = run_training_pipeline(
                    labeled_data_path=ds_path,
                    holdout_test_path="",
                    target_utility_pct=0.85,
                    max_attempts=2
                )

                if all([config_path, model_path, preprocessor_path]) and os.path.exists(model_path):
                    # Step 2: Generate synthetic data (CSV by default for frontend usability)
                    synth_path = run_generation_pipeline(
                        config_path=config_path,
                        model_path=model_path,
                        preprocessor_path=preprocessor_path,
                        label=label,
                        num_to_generate=num_to_generate,
                        output_format="csv"
                    )

                    final_message = (
                        "I have trained the model and generated synthetic data:\n"
                        f"config_path: {config_path}\n"
                        f"model_path: {model_path}\n"
                        f"preprocessor_path: {preprocessor_path}\n"
                        f"synthetic_output_path: {synth_path}"
                    )

                    file_paths = {
                        "labeled_output_path": None,
                        "config_path": config_path,
                        "model_path": model_path,
                        "preprocessor_path": preprocessor_path,
                        "synthetic_output_path": synth_path,
                        "anomaly_report_path": None,
                    }
                    steps_completed = ["training", "generation"]

                    return GenerateResponse(
                        output=final_message,
                        file_paths=file_paths,
                        steps_completed=steps_completed,
                    )
                else:
                    pass  # Fall back to agent

        # Run the agent with recursion limit to prevent infinite loops
        result = await agent.ainvoke(
            {"messages": [("user", request.input_message)]},
            config={"recursion_limit": 15}  # Limit to 15 steps max
        )
        
        # Extract the final response
        messages = result.get("messages", [])
        final_message = messages[-1].content if messages else "No response generated"
        
        # Extract file paths from tool return values in messages (most reliable)
        file_paths = {}
        steps_completed = []
        
        # Look through messages for tool return values
        for i, message in enumerate(messages):
            # Check if this is a tool message (contains tool return value)
            if hasattr(message, 'content'):
                content = message.content
                
                # If content is a string, it might be a file path from a tool
                if isinstance(content, str) and os.path.exists(content):
                    # This is likely a file path from label_dataset, generate_synthetic_data, or detect_anomalies
                    if content.endswith('.parquet'):
                        if 'labeled' in content.lower() or 'label' in content.lower():
                            file_paths['labeled_output_path'] = content
                            if 'labeling' not in steps_completed:
                                steps_completed.append('labeling')
                        elif 'anomaly' in content.lower():
                            file_paths['anomaly_report_path'] = content
                            if 'anomaly_detection' not in steps_completed:
                                steps_completed.append('anomaly_detection')
                    elif content.endswith('.csv'):
                        file_paths['synthetic_output_path'] = content
                        if 'generation' not in steps_completed:
                            steps_completed.append('generation')
                
                # If content is a dict, it might be from train_model
                elif isinstance(content, dict):
                    if 'config_path' in content:
                        file_paths['config_path'] = content.get('config_path')
                        file_paths['model_path'] = content.get('model_path')
                        file_paths['preprocessor_path'] = content.get('preprocessor_path')
                        if 'training' not in steps_completed:
                            steps_completed.append('training')
        
        # Also extract from response text as fallback (for paths not captured above)
        text_paths = extract_file_paths_from_response(final_message)
        for key, value in text_paths.items():
            if value and (key not in file_paths or not file_paths[key]):
                file_paths[key] = value
        
        # Also check response text for steps
        text_steps = extract_steps_completed(final_message, messages)
        for step in text_steps:
            if step not in steps_completed:
                steps_completed.append(step)
        
        # Ensure all expected file_path keys exist (set to None if missing)
        expected_keys = {
            "labeled_output_path": None,
            "config_path": None,
            "model_path": None,
            "preprocessor_path": None,
            "synthetic_output_path": None,
            "anomaly_report_path": None
        }
        for key in expected_keys:
            if key not in file_paths:
                file_paths[key] = None
        
        return GenerateResponse(
            output=final_message,
            file_paths=file_paths,
            steps_completed=steps_completed
        )
    
    except RecursionError as e:
        return GenerateResponse(
            output="Error: Agent exceeded maximum steps (recursion limit). The workflow was too complex or the agent got stuck in a loop. Try simplifying your request or use Manual Mode for fine-grained control.",
            file_paths={
                "labeled_output_path": None,
                "config_path": None,
                "model_path": None,
                "preprocessor_path": None,
                "synthetic_output_path": None,
                "anomaly_report_path": None
            },
            steps_completed=[]
        )
    except Exception as e:
        error_msg = str(e)
        if "recursion" in error_msg.lower():
            error_msg = "Agent exceeded maximum steps. Try simplifying your request or use Manual Mode."
        
        return GenerateResponse(
            output=f"Error: {error_msg}",
            file_paths={
                "labeled_output_path": None,
                "config_path": None,
                "model_path": None,
                "preprocessor_path": None,
                "synthetic_output_path": None,
                "anomaly_report_path": None
            },
            steps_completed=[]
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Start the FastAPI server."""
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("="*60)
    print("DataFoundry Orchestrator Server")
    print("="*60)
    print(f"Starting server on {host}:{port}")
    print("Endpoints:")
    print(f"  - POST http://{host}:{port}/generate")
    print(f"  - GET  http://{host}:{port}/health")
    print("\nAvailable tools:")
    print("  - label_dataset")
    print("  - train_model")
    print("  - generate_synthetic_data")
    print("  - detect_anomalies")
    print("="*60)
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

