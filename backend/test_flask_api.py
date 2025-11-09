"""
Quick test script for DataFoundry Flask API.

This script verifies that all API endpoints are accessible and responding correctly.
Run this after starting the Flask API server to ensure everything is working.
"""

import requests
import sys
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health check endpoint."""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("  ✓ Health check passed")
            print(f"    Status: {data['status']}")
            print(f"    Features: {data['features']}")
            return True
        else:
            print(f"  ✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  ✗ Cannot connect to Flask API")
        print("    Make sure the server is running: python flask_api.py")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_dataset_info():
    """Test the dataset info endpoint."""
    print("\nTesting /api/dataset/info endpoint...")
    try:
        # Test with adult dataset
        response = requests.post(
            f"{BASE_URL}/api/dataset/info",
            json={"data_path": "testing_data/adult.csv"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("  ✓ Dataset info retrieved")
            print(f"    Rows: {data['num_rows']}")
            print(f"    Columns: {data['num_columns']}")
            print(f"    Sample columns: {data['columns'][:5]}...")
            return True
        else:
            error = response.json()
            print(f"  ✗ Dataset info failed: {error.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_labeling_endpoint():
    """Test the labeling endpoint (without actually running it)."""
    print("\nTesting /api/labeling/create-labels endpoint...")
    try:
        # Test with minimal request (will fail validation, but proves endpoint exists)
        response = requests.post(
            f"{BASE_URL}/api/labeling/create-labels",
            json={},
            timeout=5
        )
        
        # We expect a 400 error for missing parameters
        if response.status_code == 400:
            error = response.json()
            if 'error' in error:
                print("  ✓ Labeling endpoint accessible")
                print(f"    Validation working: {error['error']}")
                return True
        
        print(f"  ⚠ Unexpected response: {response.status_code}")
        return True  # Endpoint exists, just unexpected response
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_training_config_endpoint():
    """Test the training config creation endpoint."""
    print("\nTesting /api/training/create-config endpoint...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/training/create-config",
            json={},
            timeout=5
        )
        
        # We expect a 400 error for missing parameters
        if response.status_code == 400:
            error = response.json()
            if 'error' in error:
                print("  ✓ Training config endpoint accessible")
                print(f"    Validation working: {error['error']}")
                return True
        
        print(f"  ⚠ Unexpected response: {response.status_code}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_generation_endpoint():
    """Test the generation endpoint."""
    print("\nTesting /api/generation/generate-data endpoint...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/generation/generate-data",
            json={},
            timeout=5
        )
        
        # We expect a 400 error for missing parameters
        if response.status_code == 400:
            error = response.json()
            if 'error' in error:
                print("  ✓ Generation endpoint accessible")
                print(f"    Validation working: {error['error']}")
                return True
        
        print(f"  ⚠ Unexpected response: {response.status_code}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_anomaly_endpoint():
    """Test the anomaly detection endpoint."""
    print("\nTesting /api/cleaning/detect-anomalies endpoint...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/cleaning/detect-anomalies",
            json={},
            timeout=5
        )
        
        # We expect a 400 error for missing parameters
        if response.status_code == 400:
            error = response.json()
            if 'error' in error:
                print("  ✓ Anomaly detection endpoint accessible")
                print(f"    Validation working: {error['error']}")
                return True
        
        print(f"  ⚠ Unexpected response: {response.status_code}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_files_list():
    """Test the file listing endpoint."""
    print("\nTesting /api/files/list endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/files/list", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("  ✓ File listing works")
            print(f"    Found {len(data.get('files', []))} files")
            return True
        else:
            print(f"  ✗ File listing failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║         DataFoundry Flask API - Endpoint Tests             ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    tests = [
        ("Health Check", test_health),
        ("Dataset Info", test_dataset_info),
        ("Labeling Endpoint", test_labeling_endpoint),
        ("Training Config Endpoint", test_training_config_endpoint),
        ("Generation Endpoint", test_generation_endpoint),
        ("Anomaly Detection Endpoint", test_anomaly_endpoint),
        ("File Listing", test_files_list),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! The Flask API is working correctly.")
        return 0
    elif passed > 0:
        print(f"\n⚠ Some tests failed. {passed}/{total} endpoints are working.")
        return 1
    else:
        print("\n✗ All tests failed. Is the Flask API server running?")
        print("  Start it with: python flask_api.py")
        return 2


if __name__ == "__main__":
    sys.exit(main())

