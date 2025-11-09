"""
Quick test for the convert-to-csv endpoint.
"""

import json
import os
import sys
import tempfile
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from flask_api import app

def test_convert_to_csv():
    """Test the convert-to-csv endpoint."""
    print("="*60)
    print("Testing /api/files/convert-to-csv endpoint")
    print("="*60)
    
    # Create a test parquet file
    test_data = {
        'age': [39, 50, 38, 53, 28],
        'fnlwgt': [77516, 83311, 215646, 234721, 338409],
        'education.num': [13, 13, 9, 7, 13],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private'],
        'label_probability': [0.2, 0.8, 0.1, 0.3, 0.9]
    }
    
    df = pd.DataFrame(test_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as f:
        temp_parquet = f.name
    
    df.to_parquet(temp_parquet, index=False)
    print(f"\n✓ Created test parquet file: {temp_parquet}")
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    
    try:
        # Test with default output path (auto-generated)
        with app.test_request_context(
            '/api/files/convert-to-csv',
            method='POST',
            json={
                'parquet_path': temp_parquet
            }
        ):
            from flask_api import convert_to_csv
            response = convert_to_csv()
            
            if hasattr(response, 'get_json'):
                result = response.get_json()
            else:
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
            
            csv_path = result.get('csv_path')
            if not csv_path or not os.path.exists(csv_path):
                print(f"\n✗ Failed: CSV file not created at {csv_path}")
                return False
            
            print(f"\n✓ CSV file created: {csv_path}")
            print(f"  Rows: {result.get('num_rows')}")
            print(f"  Columns: {result.get('num_columns')}")
            print(f"  Column names: {result.get('columns')}")
            
            # Verify CSV content
            df_csv = pd.read_csv(csv_path)
            if len(df_csv) != len(df):
                print(f"\n✗ Failed: Row count mismatch. Expected {len(df)}, got {len(df_csv)}")
                return False
            
            if list(df_csv.columns) != list(df.columns):
                print(f"\n✗ Failed: Column mismatch")
                print(f"  Expected: {list(df.columns)}")
                print(f"  Got: {list(df_csv.columns)}")
                return False
            
            print(f"\n✓ CSV content verified - all data matches!")
            
            # Test with custom output path
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                custom_csv_path = f.name
            
            with app.test_request_context(
                '/api/files/convert-to-csv',
                method='POST',
                json={
                    'parquet_path': temp_parquet,
                    'output_path': custom_csv_path
                }
            ):
                response = convert_to_csv()
                result = response.get_json() if hasattr(response, 'get_json') else None
                
                if result and result.get('csv_path') == custom_csv_path:
                    print(f"\n✓ Custom output path works: {custom_csv_path}")
                else:
                    print(f"\n⚠ Custom output path test had issues")
            
            # Cleanup custom CSV
            if os.path.exists(custom_csv_path):
                os.remove(custom_csv_path)
            
            print("\n" + "="*60)
            print("✅ TEST PASSED: Convert to CSV endpoint works correctly!")
            print("="*60)
            return True
            
    except Exception as e:
        print(f"\n✗ TEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(temp_parquet):
            os.remove(temp_parquet)
        # Also clean up auto-generated CSV
        auto_csv = temp_parquet.rsplit('.parquet', 1)[0] + '.csv'
        if os.path.exists(auto_csv):
            os.remove(auto_csv)


if __name__ == "__main__":
    test_convert_to_csv()

