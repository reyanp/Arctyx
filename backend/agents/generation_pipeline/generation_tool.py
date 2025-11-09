"""
Simple worker tool for generating synthetic data.
This is a direct wrapper around DataFoundry's generator module.
"""

import sys
import os

# Add DataFoundry to path
datafoundry_path = os.path.join(os.path.dirname(__file__), '..', '..')
if datafoundry_path not in sys.path:
    sys.path.insert(0, datafoundry_path)

import DataFoundry.generator as generator


def run_generation_pipeline(
    config_path: str, 
    model_path: str, 
    preprocessor_path: str, 
    label: float, 
    num_to_generate: int, 
    output_format: str = 'pt'
) -> str:
    """
    This is a simple "worker" tool that calls the DataFoundry generator.
    
    It takes a trained model and user parameters to generate new
    synthetic data.
    
    Args:
        config_path: Path to the config.json file.
        model_path: Path to the trained model.pth file.
        preprocessor_path: Path to the preprocessor.joblib file.
        label: The condition to generate for (e.g., 1.0 for '>50K').
        num_to_generate: How many samples to create.
        output_format: 'pt' for a PyTorch file or 'csv'.
        
    Returns:
        The path to the newly generated synthetic data file.
    """
    print(f"--- Starting Generation Pipeline ---")
    print(f"   - Generating {num_to_generate} samples for label '{label}'")
    
    try:
        # Define a logical output path
        output_dir = os.path.dirname(config_path) if os.path.dirname(config_path) else '.'
        output_path = os.path.join(
            output_dir,
            f"synthetic_data_label_{label}_count_{num_to_generate}.{output_format}"
        )
        
        # Call the core library function
        # Note: The generator doesn't need preprocessor_path as it's embedded in the model
        generated_file_path = generator.generate_data(
            model_path=model_path,
            config_path=config_path,
            label=label,
            num_to_generate=num_to_generate,
            output_path=output_path,
            output_format=output_format
        )
        
        print(f"--- Generation Pipeline Complete ---")
        print(f"   - Synthetic data saved to: {generated_file_path}")
        
        return generated_file_path
        
    except Exception as e:
        print(f"Error during data generation: {e}")
        raise

