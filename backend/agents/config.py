import os

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Load the .env file
load_dotenv()

# Set the API key - The ChatNVIDIA client will automatically find this environment variable
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# This is the default base URL for the NVIDIA API catalog
# We don't need to set it in the .env, but we can be explicit.
NIM_API_BASE_URL = "https://integrate.api.nvidia.com/v1/"

# --- Define Our Agent Brains ---

# High-Cost Brain (for complex generation and planning)
NEMOTRON_HIGH_COST_CLIENT = ChatNVIDIA(
    base_url=NIM_API_BASE_URL,
    model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
    temperature=0.7
)

# Low-Cost Brain (for constrained, repetitive tasks like tuning)
NEMOTRON_LOW_COST_CLIENT = ChatNVIDIA(
    base_url=NIM_API_BASE_URL,
    model="nvidia/nvidia-nemotron-nano-9b-v2"
)

# --- Define the Code Submission Tool ---
# This tool allows the LLM to submit generated Python code

CODE_SUBMISSION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_labeling_functions",
        "description": "Submit the generated Python labeling functions code",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The complete Python code with all labeling functions"
                }
            },
            "required": ["code"]
        }
    }
}

# --- Writer Agent (High-Cost Brain bound with tool) ---
# This agent generates labeling functions (code generation task) and submits them via tool call
# Uses high-cost model because writing code requires complex reasoning and creativity
NEMOTRON_WRITER_AGENT = NEMOTRON_HIGH_COST_CLIENT.bind(tools=[CODE_SUBMISSION_TOOL])

# --- Define the Training Config Submission Tool ---
# This tool allows the LLM to submit generated JSON configurations for training
TRAINING_CONFIG_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_training_config",
        "description": "Submit the generated JSON configuration for a generative model",
        "parameters": {
            "type": "object",
            "properties": {
                "config_json": {
                    "type": "string",
                    "description": "A string containing the complete, valid JSON configuration"
                }
            },
            "required": ["config_json"]
        }
    }
}

# --- Architect Agent (High-Cost Brain bound with training config tool) ---
# This agent designs model configurations and submits them via tool call
# Uses high-cost model because architecture design requires complex reasoning
NEMOTRON_ARCHITECT_AGENT = NEMOTRON_HIGH_COST_CLIENT.bind(tools=[TRAINING_CONFIG_TOOL])

print("NIM clients, Writer, and Architect agents initialized.")

