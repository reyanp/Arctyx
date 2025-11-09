"""
DataFoundry Orchestrator Server using NVIDIA NeMo Agent Toolkit Python API.

This script creates a ReAct agent that can orchestrate our custom Python pipeline tools
(labeling, training, generation, anomaly detection) using NAT's Python API instead of YAML configuration.

Usage:
    python serve_orchestrator.py
"""

import asyncio
import os
import sys
import uvicorn
from pathlib import Path

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


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Main endpoint that receives natural language instructions and runs the orchestrator.
    
    Compatible with NAT's /generate endpoint format.
    """
    try:
        # Run the agent
        result = await agent.ainvoke({"messages": [("user", request.input_message)]})
        
        # Extract the final response
        final_message = result["messages"][-1].content
        
        return GenerateResponse(output=final_message)
    
    except Exception as e:
        return GenerateResponse(output=f"Error: {str(e)}")


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

