"""
Simple worker tool for anomaly detection.
This is a direct wrapper around DataFoundry's evaluator module.
"""

import sys
import os

# Add DataFoundry to path
datafoundry_path = os.path.join(os.path.dirname(__file__), '..', '..')
if datafoundry_path not in sys.path:
    sys.path.insert(0, datafoundry_path)

import DataFoundry.evaluator as evaluator


def run_anomaly_pipeline(
    config_path: str, 
    model_path: str, 
    preprocessor_path: str, 
    data_to_scan_path: str
) -> str:
    """
    This is a simple "worker" tool that calls the DataFoundry evaluator
    to find anomalies in a dataset.
    
    Args:
        config_path: Path to the config.json file.
        model_path: Path to the trained model.pth file.
        preprocessor_path: Path to the preprocessor.joblib file.
        data_to_scan_path: Path to the .parquet file to be scored.
        
    Returns:
        The path to the anomaly report file.
    """
    print(f"--- Starting Anomaly Detection Pipeline ---")
    print(f"   - Scanning file: {data_to_scan_path}")
    
    try:
        # Define a logical output path in the same directory as the data
        output_dir = os.path.dirname(data_to_scan_path) if os.path.dirname(data_to_scan_path) else '.'
        output_path = os.path.join(output_dir, "anomaly_report.parquet")
        
        # Call the core library function
        report_path = evaluator.find_anomalies(
            config_path=config_path,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            data_path_to_scan=data_to_scan_path,
            output_path=output_path
        )
        
        print(f"--- Anomaly Pipeline Complete ---")
        print(f"   - Anomaly report saved to: {report_path}")
        
        return report_path
        
    except Exception as e:
        print(f"Error during anomaly detection: {e}")
        raise

