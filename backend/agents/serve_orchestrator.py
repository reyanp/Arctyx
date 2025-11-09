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
def label_dataset(user_goal: str, raw_data_path: str, hand_labeled_examples_path: str, 
                   target_auc_score: float = 0.75, max_attempts: int = 2) -> str:
    """
    Label an unlabeled dataset using weak supervision.
    
    Args:
        user_goal: Natural language description of the labeling goal
        raw_data_path: Path to unlabeled parquet file
        hand_labeled_examples_path: Path to ground truth parquet file
        target_auc_score: Target AUC score (default 0.75)
        max_attempts: Maximum attempts (default 2)
    
    Returns:
        Path to the labeled parquet file
    """
    return run_labeling_pipeline(
        user_goal=user_goal,
        raw_data_path=raw_data_path,
        hand_labeled_examples_path=hand_labeled_examples_path,
        target_auc_score=target_auc_score,
        max_attempts=max_attempts
    )


@tool
def train_model(labeled_data_path: str, holdout_test_path: str,
                target_utility_pct: float = 0.85, max_attempts: int = 2) -> dict:
    """
    Train a generative model on labeled data.
    
    Args:
        labeled_data_path: Path to labeled parquet file
        holdout_test_path: Path to holdout test parquet file
        target_utility_pct: Target utility percentage (default 0.85)
        max_attempts: Maximum attempts (default 2)
    
    Returns:
        Dictionary with keys: config_path, model_path, preprocessor_path
    """
    config_path, model_path, preprocessor_path = run_training_pipeline(
        labeled_data_path=labeled_data_path,
        holdout_test_path=holdout_test_path,
        target_utility_pct=target_utility_pct,
        max_attempts=max_attempts
    )
    return {
        "config_path": config_path,
        "model_path": model_path,
        "preprocessor_path": preprocessor_path
    }


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
    return run_generation_pipeline(
        config_path=config_path,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        label=label,
        num_to_generate=num_to_generate,
        output_format=output_format
    )


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
- label_dataset: Label unlabeled data using weak supervision
- train_model: Train a generative model on labeled data
- generate_synthetic_data: Generate synthetic data from a trained model
- detect_anomalies: Find anomalies in a dataset using a trained model

CRITICAL INSTRUCTIONS:
1. You MUST carefully manage the file paths and dependencies between steps.
2. When you call tools, extract the file paths from their return values.
3. At the end, you MUST return file paths in your response using these EXACT patterns:
   - For labeling: "labeled_output_path: /path/to/file.parquet"
   - For training: "config_path: /path/to/config.json" and "model_path: /path/to/model.pth" and "preprocessor_path: /path/to/preprocessor.joblib"
   - For generation: "synthetic_output_path: /path/to/synthetic.csv"
   - For anomaly: "anomaly_report_path: /path/to/report.parquet"

Example response format:
"I have completed the labeling task. labeled_output_path: /tmp/labeled_data.parquet"

Or for multi-step:
"I have completed all steps:
labeled_data_path: /tmp/labeled.parquet
config_path: /tmp/config.json  
model_path: /tmp/model.pth
preprocessor_path: /tmp/preprocessor.joblib
synthetic_data_path: /tmp/synthetic.csv"

Plan your steps, execute them one by one, and confirm the final output with the file paths."""

# Create the ReAct agent with system prompt
tools = [label_dataset, train_model, generate_synthetic_data, detect_anomalies]

# Wrap LLM with system prompt
llm_with_prompt = llm.bind(system=SYSTEM_PROMPT)
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
        # Run the agent
        result = await agent.ainvoke({"messages": [("user", request.input_message)]})
        
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
    
    except Exception as e:
        return GenerateResponse(
            output=f"Error: {str(e)}",
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

