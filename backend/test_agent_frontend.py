"""
Test script to emulate frontend calls to the Flask API and Orchestrator.

This script tests:
1. Flask API /api/agent/generate endpoint
2. Orchestrator service on port 8000
3. Full workflow: CSV upload -> Training -> Generation

Usage:
    # First, start the services:
    # Terminal 1: python backend/flask_api.py
    # Terminal 2: python backend/agents/serve_orchestrator.py
    
    # Then run this test:
    python backend/test_agent_frontend.py
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
FLASK_URL = "http://localhost:5000"
ORCHESTRATOR_URL = "http://localhost:8000"
BACKEND_DIR = Path(__file__).parent
TEST_DATA_PATH = BACKEND_DIR / "testing_data" / "adult.csv"


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def check_services():
    """Check if Flask API and Orchestrator are running."""
    print_section("CHECKING SERVICES")
    
    # Check Flask API
    try:
        response = requests.get(f"{FLASK_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Flask API is running on {FLASK_URL}")
            print(f"  Features: {json.dumps(data.get('features', {}), indent=4)}")
            flask_ok = True
        else:
            print(f"✗ Flask API returned status {response.status_code}")
            flask_ok = False
    except requests.exceptions.ConnectionError:
        print(f"✗ Flask API not running on {FLASK_URL}")
        print("  Start it with: python backend/flask_api.py")
        flask_ok = False
    except Exception as e:
        print(f"✗ Flask API error: {e}")
        flask_ok = False
    
    # Check Orchestrator
    try:
        response = requests.get(f"{ORCHESTRATOR_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ Orchestrator is running on {ORCHESTRATOR_URL}")
            orchestrator_ok = True
        else:
            print(f"✗ Orchestrator returned status {response.status_code}")
            orchestrator_ok = False
    except requests.exceptions.ConnectionError:
        print(f"✗ Orchestrator not running on {ORCHESTRATOR_URL}")
        print("  Start it with: python backend/agents/serve_orchestrator.py")
        orchestrator_ok = False
    except Exception as e:
        print(f"✗ Orchestrator error: {e}")
        orchestrator_ok = False
    
    return flask_ok and orchestrator_ok


def test_direct_orchestrator_call():
    """Test calling the orchestrator directly (bypassing Flask)."""
    print_section("TEST 1: Direct Orchestrator Call")
    
    if not TEST_DATA_PATH.exists():
        print(f"✗ Test data not found: {TEST_DATA_PATH}")
        return False
    
    # The orchestrator expects parquet files, so convert CSV to Parquet first
    print(f"Converting CSV to Parquet for orchestrator...")
    try:
        import pandas as pd
        df = pd.read_csv(TEST_DATA_PATH)
        parquet_path = TEST_DATA_PATH.parent / (TEST_DATA_PATH.stem + '_test.parquet')
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        print(f"  ✓ Converted to: {parquet_path}")
        test_data_path = parquet_path
    except Exception as e:
        print(f"  ✗ Conversion failed: {e}")
        return False
    
    test_message = f"""Generate 100 synthetic samples.

Dataset path: {test_data_path.absolute()}"""
    
    print(f"\nRequest to {ORCHESTRATOR_URL}/generate:")
    print(f"  Message: {test_message[:100]}...")
    
    try:
        response = requests.post(
            f"{ORCHESTRATOR_URL}/generate",
            json={"input_message": test_message},
            timeout=300
        )
        
        print(f"\nResponse Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✓ Success!")
            print(f"\nOutput:\n{data.get('output', 'No output')[:500]}...")
            print(f"\nFile Paths:")
            for key, value in data.get('file_paths', {}).items():
                print(f"  {key}: {value}")
            print(f"\nSteps Completed: {data.get('steps_completed', [])}")
            return True
        else:
            print(f"✗ Failed with status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out (> 300 seconds)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flask_agent_endpoint():
    """Test calling Flask API /api/agent/generate endpoint (emulates frontend)."""
    print_section("TEST 2: Flask API Agent Endpoint (Frontend Emulation)")
    
    if not TEST_DATA_PATH.exists():
        print(f"✗ Test data not found: {TEST_DATA_PATH}")
        return False
    
    # This emulates what the frontend sends
    request_data = {
        "input_message": "Generate 100 synthetic samples with label 1.0",
        "dataset_path": str(TEST_DATA_PATH.absolute())  # Absolute path
    }
    
    print(f"Request to {FLASK_URL}/api/agent/generate:")
    print(f"  input_message: {request_data['input_message']}")
    print(f"  dataset_path: {request_data['dataset_path']}")
    
    try:
        response = requests.post(
            f"{FLASK_URL}/api/agent/generate",
            json=request_data,
            timeout=300
        )
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        # Try to get the response text first
        response_text = response.text
        print(f"\nRaw Response (first 1000 chars):\n{response_text[:1000]}")
        
        # Try to parse as JSON
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"\n✗ Failed to parse JSON response!")
            print(f"  JSON Error: {e}")
            print(f"  Response was: {response_text[:500]}")
            return False
        
        # Check for error field
        if 'error' in data:
            print(f"\n✗ Error in response: {data['error']}")
            if 'traceback' in data:
                print(f"\nTraceback:\n{data['traceback']}")
            return False
        
        # Check response structure
        print("\n✓ Response Structure:")
        print(f"  Keys: {list(data.keys())}")
        
        # Validate expected fields
        expected_fields = ['output', 'file_paths', 'steps_completed']
        missing_fields = [f for f in expected_fields if f not in data]
        
        if missing_fields:
            print(f"\n⚠ Missing expected fields: {missing_fields}")
        
        # Display output
        if 'output' in data:
            print(f"\nOutput:\n{data['output'][:500]}...")
        
        # Display file paths
        if 'file_paths' in data:
            print(f"\nFile Paths:")
            for key, value in data['file_paths'].items():
                status = "✓" if value else "○"
                print(f"  {status} {key}: {value}")
                
                # Verify file exists
                if value and os.path.exists(value):
                    file_size = os.path.getsize(value)
                    print(f"      → File exists ({file_size:,} bytes)")
                elif value:
                    print(f"      → ⚠ File does not exist!")
        
        # Display steps
        if 'steps_completed' in data:
            print(f"\nSteps Completed: {data['steps_completed']}")
        
        if response.status_code == 200:
            print("\n✓ Test passed!")
            return True
        else:
            print(f"\n✗ Unexpected status code: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out (> 300 seconds)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_upload_and_generate():
    """Test full workflow: upload CSV -> generate via agent."""
    print_section("TEST 3: Full Upload + Generate Workflow")
    
    if not TEST_DATA_PATH.exists():
        print(f"✗ Test data not found: {TEST_DATA_PATH}")
        return False
    
    # Step 1: Upload file (emulates frontend file upload)
    print("\nStep 1: Uploading CSV file...")
    
    try:
        with open(TEST_DATA_PATH, 'rb') as f:
            files = {'file': (TEST_DATA_PATH.name, f, 'text/csv')}
            response = requests.post(
                f"{FLASK_URL}/api/upload",  # Correct endpoint
                files=files,
                timeout=30
            )
        
        if response.status_code != 200:
            print(f"✗ Upload failed with status {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return False
        
        upload_data = response.json()
        
        if not upload_data.get('success'):
            print(f"✗ Upload unsuccessful: {upload_data.get('message')}")
            return False
        
        uploaded_file_path = upload_data.get('file_path')
        print(f"✓ File uploaded: {uploaded_file_path}")
        
    except Exception as e:
        print(f"✗ Upload error: {e}")
        return False
    
    # Step 2: Generate synthetic data using agent
    print("\nStep 2: Generating synthetic data via agent...")
    
    request_data = {
        "input_message": "Generate 50 synthetic samples",
        "dataset_path": uploaded_file_path
    }
    
    try:
        response = requests.post(
            f"{FLASK_URL}/api/agent/generate",
            json=request_data,
            timeout=300
        )
        
        print(f"Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"✗ Generation failed with status {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return False
        
        data = response.json()
        
        if 'error' in data:
            print(f"✗ Error: {data['error']}")
            return False
        
        print(f"\n✓ Generation completed!")
        print(f"Output: {data.get('output', '')[:300]}...")
        
        synthetic_path = data.get('file_paths', {}).get('synthetic_output_path')
        if synthetic_path and os.path.exists(synthetic_path):
            print(f"\n✓ Synthetic data created: {synthetic_path}")
            file_size = os.path.getsize(synthetic_path)
            print(f"  Size: {file_size:,} bytes")
            return True
        else:
            print(f"\n⚠ Synthetic data path not found or file doesn't exist")
            print(f"  Path from response: {synthetic_path}")
            return False
            
    except Exception as e:
        print(f"✗ Generation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  DataFoundry Frontend Emulation Test Suite".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    # Check services first
    if not check_services():
        print("\n" + "!"*80)
        print("! SERVICES NOT RUNNING - Please start them first:")
        print("!   Terminal 1: python backend/flask_api.py")
        print("!   Terminal 2: python backend/agents/serve_orchestrator.py")
        print("!"*80)
        return
    
    # Run tests
    results = {
        "Direct Orchestrator": test_direct_orchestrator_call(),
        "Flask Agent Endpoint": test_flask_agent_endpoint(),
        "Upload + Generate": test_upload_and_generate(),
    }
    
    # Summary
    print_section("TEST SUMMARY")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("  ✓ ALL TESTS PASSED!")
    else:
        print("  ✗ SOME TESTS FAILED - Check output above for details")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
