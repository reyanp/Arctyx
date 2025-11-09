#!/usr/bin/env python3
"""
Test script to verify agent endpoint output format.
This ensures the Flask proxy correctly formats the orchestrator's response for the frontend.
"""

import json
import requests

def test_agent_response_format():
    """Test that the agent endpoint returns the expected format."""
    
    print("="*60)
    print("Testing Agent Endpoint Response Format")
    print("="*60)
    
    # Test 1: Connection Error (orchestrator not running)
    print("\nTest 1: Orchestrator Not Running")
    print("-" * 40)
    
    try:
        response = requests.post(
            'http://localhost:5000/api/agent/generate',
            json={
                'input_message': 'Generate 100 samples',
                'dataset_path': 'testing_data/adult.csv'
            },
            timeout=5
        )
        
        data = response.json()
        print(f"Status Code: {response.status_code}")
        print(f"Response Keys: {list(data.keys())}")
        
        # Verify required fields
        required_fields = ['output', 'file_paths', 'steps_completed']
        optional_fields = ['error']
        
        for field in required_fields:
            if field in data:
                print(f"✓ {field}: present")
            else:
                print(f"✗ {field}: MISSING")
        
        if 'file_paths' in data:
            required_paths = [
                'labeled_output_path',
                'config_path',
                'model_path',
                'preprocessor_path',
                'synthetic_output_path',
                'anomaly_report_path'
            ]
            
            print("\nFile Paths:")
            for path_key in required_paths:
                if path_key in data['file_paths']:
                    value = data['file_paths'][path_key]
                    print(f"  ✓ {path_key}: {value}")
                else:
                    print(f"  ✗ {path_key}: MISSING")
        
        print(f"\nFull Response:")
        print(json.dumps(data, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == '__main__':
    test_agent_response_format()

