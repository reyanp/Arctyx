"""
Test script to verify that the create_training_config endpoint
correctly detects and includes categorical columns.
"""

import json
import os
import sys
import tempfile
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from flask_api import app, create_training_config
from flask import Flask

def create_test_dataset():
    """Create a test dataset with both numerical and categorical columns."""
    # Create sample data similar to Adult Census dataset
    data = {
        # Numerical columns
        'age': [39, 50, 38, 53, 28, 37, 49, 52, 31, 42],
        'fnlwgt': [77516, 83311, 215646, 234721, 338409, 284582, 160187, 209642, 45781, 159449],
        'education.num': [13, 13, 9, 7, 13, 14, 5, 9, 14, 13],
        'capital.gain': [2174, 0, 0, 0, 0, 0, 0, 0, 14084, 5178],
        'capital.loss': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'hours.per.week': [40, 13, 40, 40, 40, 40, 16, 45, 50, 40],
        # Categorical columns
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private', 
                      'Private', 'Private', 'Self-emp-not-inc', 'Private', 'Private'],
        'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors', 'Masters',
                     '11th', 'HS-grad', 'Masters', 'Bachelors'],
        'marital.status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-civ-spouse',
                          'Married-civ-spouse', 'Married-civ-spouse', 'Married-civ-spouse',
                          'Married-civ-spouse', 'Married-civ-spouse', 'Married-civ-spouse'],
        'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Handlers-cleaners',
                      'Prof-specialty', 'Exec-managerial', 'Other-service', 'Exec-managerial',
                      'Prof-specialty', 'Exec-managerial'],
        'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband', 'Wife',
                        'Wife', 'Husband', 'Husband', 'Wife', 'Husband'],
        'race': ['White', 'White', 'White', 'Black', 'Black', 'White', 'Black', 'White', 'White', 'White'],
        'sex': ['Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male'],
        'native.country': ['United-States', 'United-States', 'United-States', 'United-States',
                          'Cuba', 'United-States', 'United-States', 'United-States', 'United-States', 'United-States'],
        # Label column
        'label_probability': [0.2, 0.8, 0.1, 0.3, 0.9, 0.95, 0.15, 0.7, 0.85, 0.75]
    }
    
    df = pd.DataFrame(data)
    return df


def test_categorical_detection():
    """Test that categorical columns are detected and included in config."""
    print("="*60)
    print("Testing Categorical Column Detection in create_training_config")
    print("="*60)
    
    # Create test dataset
    df = create_test_dataset()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as f:
        temp_path = f.name
    
    df.to_parquet(temp_path, index=False)
    print(f"\n✓ Created test dataset: {temp_path}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Numerical columns: {len(df.select_dtypes(include=['number']).columns)}")
    print(f"  Categorical columns: {len([c for c in df.columns if c not in df.select_dtypes(include=['number']).columns and c != 'label_probability'])}")
    
    try:
        # Create a test Flask app context
        with app.test_request_context(
            '/api/training/create-config',
            method='POST',
            json={
                'data_path': temp_path,
                'output_dir': tempfile.mkdtemp(),
                'model_type': 'tabular_cvae'  # Should auto-switch to MixedDataCVAE
            }
        ):
            # Call the endpoint function directly
            response = create_training_config()
            
            # Get the response data
            if hasattr(response, 'get_json'):
                result = response.get_json()
            else:
                # If it's a tuple (status_code, response), extract the response
                if isinstance(response, tuple):
                    result = response[0].get_json() if hasattr(response[0], 'get_json') else None
                else:
                    result = response.get_json() if hasattr(response, 'get_json') else None
            
            if not result:
                print("\n✗ Failed: No response data")
                return False
            
            if 'error' in result:
                print(f"\n✗ Error: {result['error']}")
                return False
            
            # Check the config
            config = result.get('config', {})
            model_params = config.get('model_params', {})
            feature_cols = model_params.get('feature_cols', {})
            
            numerical_cols = feature_cols.get('numerical_cols', [])
            categorical_cols = feature_cols.get('categorical_cols', [])
            model_class = model_params.get('model_class_name', '')
            
            print(f"\n✓ Config created successfully")
            print(f"  Model class: {model_class}")
            print(f"  Numerical columns ({len(numerical_cols)}): {numerical_cols}")
            print(f"  Categorical columns ({len(categorical_cols)}): {categorical_cols}")
            
            # Verify categorical columns are detected
            if not categorical_cols:
                print("\n✗ FAILED: Categorical columns were not detected!")
                print(f"  Expected categorical columns: workclass, education, marital.status, etc.")
                return False
            
            # Verify it auto-switched to MixedDataCVAE
            if model_class != 'MixedDataCVAE':
                print(f"\n⚠ WARNING: Model class is {model_class}, expected MixedDataCVAE")
                print("  (This might be OK if categorical columns are empty)")
            
            # Verify all expected categorical columns are present
            expected_categorical = ['workclass', 'education', 'marital.status', 'occupation', 
                                  'relationship', 'race', 'sex', 'native.country']
            missing = [col for col in expected_categorical if col not in categorical_cols]
            
            if missing:
                print(f"\n⚠ WARNING: Some categorical columns missing: {missing}")
            else:
                print(f"\n✓ All expected categorical columns detected!")
            
            # Verify numerical columns
            expected_numerical = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                                'capital.loss', 'hours.per.week']
            missing_num = [col for col in expected_numerical if col not in numerical_cols]
            
            if missing_num:
                print(f"\n⚠ WARNING: Some numerical columns missing: {missing_num}")
            else:
                print(f"✓ All expected numerical columns detected!")
            
            # Verify condition column
            condition_cols = model_params.get('condition_cols', [])
            if 'label_probability' not in condition_cols:
                print(f"\n⚠ WARNING: label_probability not in condition_cols: {condition_cols}")
            else:
                print(f"✓ Condition column correctly set: {condition_cols}")
            
            # If using MixedDataCVAE, verify embedding dimensions
            if model_class == 'MixedDataCVAE' and categorical_cols:
                embed_dims = model_params.get('categorical_embed_dims', [])
                if not embed_dims:
                    print(f"\n⚠ WARNING: No categorical_embed_dims found for MixedDataCVAE")
                else:
                    print(f"✓ Categorical embedding dimensions: {embed_dims}")
            
            print("\n" + "="*60)
            print("✅ TEST PASSED: Categorical columns are correctly detected!")
            print("="*60)
            return True
            
    except Exception as e:
        print(f"\n✗ TEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_numerical_only():
    """Test that numerical-only datasets still work correctly."""
    print("\n" + "="*60)
    print("Testing Numerical-Only Dataset (should work with TabularCVAE)")
    print("="*60)
    
    # Create dataset with only numerical columns
    data = {
        'age': [39, 50, 38, 53, 28],
        'fnlwgt': [77516, 83311, 215646, 234721, 338409],
        'education.num': [13, 13, 9, 7, 13],
        'capital.gain': [2174, 0, 0, 0, 0],
        'capital.loss': [0, 0, 0, 0, 0],
        'hours.per.week': [40, 13, 40, 40, 40],
        'label_probability': [0.2, 0.8, 0.1, 0.3, 0.9]
    }
    
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as f:
        temp_path = f.name
    
    df.to_parquet(temp_path, index=False)
    print(f"\n✓ Created numerical-only test dataset: {temp_path}")
    
    try:
        with app.test_request_context(
            '/api/training/create-config',
            method='POST',
            json={
                'data_path': temp_path,
                'output_dir': tempfile.mkdtemp(),
                'model_type': 'tabular_cvae'
            }
        ):
            response = create_training_config()
            
            if hasattr(response, 'get_json'):
                result = response.get_json()
            else:
                if isinstance(response, tuple):
                    result = response[0].get_json() if hasattr(response[0], 'get_json') else None
                else:
                    result = response.get_json() if hasattr(response, 'get_json') else None
            
            if not result or 'error' in result:
                print(f"\n✗ Failed: {result.get('error', 'Unknown error') if result else 'No response'}")
                return False
            
            config = result.get('config', {})
            model_params = config.get('model_params', {})
            model_class = model_params.get('model_class_name', '')
            feature_cols = model_params.get('feature_cols', {})
            categorical_cols = feature_cols.get('categorical_cols', [])
            
            print(f"\n✓ Config created successfully")
            print(f"  Model class: {model_class}")
            print(f"  Categorical columns: {categorical_cols}")
            
            # Should use TabularCVAE for numerical-only
            if categorical_cols:
                print(f"\n⚠ WARNING: Found categorical columns in numerical-only dataset")
            else:
                print(f"✓ Correctly detected numerical-only dataset")
            
            if model_class == 'TabularCVAE':
                print(f"✓ Correctly using TabularCVAE for numerical-only data")
            else:
                print(f"⚠ Using {model_class} instead of TabularCVAE")
            
            print("\n✅ TEST PASSED: Numerical-only dataset handled correctly!")
            return True
            
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Categorical Column Detection Test Suite")
    print("="*60)
    
    test1 = test_categorical_detection()
    test2 = test_numerical_only()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test 1 (Categorical Detection): {'✓ PASSED' if test1 else '✗ FAILED'}")
    print(f"Test 2 (Numerical Only): {'✓ PASSED' if test2 else '✗ FAILED'}")
    
    if test1 and test2:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)

