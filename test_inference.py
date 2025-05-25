#!/usr/bin/env python3
"""
Test script for the inference.py functionality.
This script tests the data processing pipeline without requiring trained models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the current directory to the path so we can import our modules
sys.path.append('.')

from feature_engineering import MODWTFeatureEngineer
from metrics import *

def test_data_loading():
    """Test loading and basic preprocessing of CAMELS data."""
    print("=" * 60)
    print("Testing Data Loading and Preprocessing")
    print("=" * 60)
    
    # Find a CSV file to test with
    data_dir = Path("./data")
    csv_files = list(data_dir.glob("*_camels.csv"))
    
    if not csv_files:
        print("No CSV files found in data directory!")
        return False
    
    # Use the first available CSV file
    test_file = csv_files[0]
    station_id = test_file.stem.split("_")[0]
    
    print(f"Testing with station: {station_id}")
    print(f"CSV file: {test_file}")
    
    # Load data
    df = pd.read_csv(test_file, parse_dates=['date'])
    print(f"Loaded {len(df)} rows of data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Check required columns
    required_columns = ['date', 'Q', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 
                       'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    print("✓ All required columns present")
    
    # Check for missing values
    missing_counts = df[required_columns].isnull().sum()
    print(f"Missing values per column:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count}")
    
    return True, df, station_id

def test_preprocessing():
    """Test the preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("Testing Preprocessing Pipeline")
    print("=" * 60)
    
    success, df, station_id = test_data_loading()
    if not success:
        return False
    
    # Test basic preprocessing
    df_processed = df.copy()
    df_processed.ffill(inplace=True)
    
    features = ['Q', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 
                'tmax(C)', 'tmin(C)', 'vp(Pa)']
    
    df_processed = df_processed[['date'] + features]
    df_processed.sort_values('date', inplace=True)
    df_processed.reset_index(drop=True, inplace=True)
    
    # Add timestamp
    df_processed['timestamp'] = pd.to_datetime(df_processed['date']).astype(int) / 10**9
    
    print(f"✓ Basic preprocessing completed")
    print(f"  Shape after preprocessing: {df_processed.shape}")
    
    return True, df_processed, station_id

def test_modwt_features():
    """Test MODWT feature engineering."""
    print("\n" + "=" * 60)
    print("Testing MODWT Feature Engineering")
    print("=" * 60)
    
    success, df, station_id = test_preprocessing()
    if not success:
        return False
    
    # Test MODWT feature engineering
    wavelet_filter = "db1"
    max_level = 6
    
    features = ['Q', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 
                'tmax(C)', 'tmin(C)', 'vp(Pa)']
    
    try:
        modwt_fe = MODWTFeatureEngineer(
            wavelet=wavelet_filter, 
            v_levels=[max_level], 
            w_levels=list(range(1, max_level+1))
        )
        
        print(f"✓ MODWT Feature Engineer initialized with {wavelet_filter}")
        
        df_with_modwt = modwt_fe.transform(df, features)
        
        print(f"✓ MODWT features computed")
        print(f"  Shape before MODWT: {df.shape}")
        print(f"  Shape after MODWT: {df_with_modwt.shape}")
        
        # Remove NaN rows
        df_with_modwt.dropna(inplace=True)
        print(f"  Shape after removing NaN: {df_with_modwt.shape}")
        
        # Show some of the new feature columns
        new_columns = [col for col in df_with_modwt.columns if col not in df.columns]
        print(f"  New MODWT features added: {len(new_columns)}")
        print(f"  Example features: {new_columns[:5]}...")
        
        return True, df_with_modwt, station_id
        
    except Exception as e:
        print(f"✗ Error in MODWT feature engineering: {str(e)}")
        return False

def test_sequence_creation():
    """Test sequence creation for LSTM input."""
    print("\n" + "=" * 60)
    print("Testing Sequence Creation")
    print("=" * 60)
    
    success, df, station_id = test_modwt_features()
    if not success:
        return False
    
    # Test sequence creation
    input_window = 270
    forecast_horizon = 1
    
    total_samples = len(df) - input_window - forecast_horizon + 1
    
    if total_samples <= 0:
        print(f"✗ Not enough data for sequences. Need at least {input_window + forecast_horizon} days, have {len(df)}")
        return False
    
    print(f"✓ Can create {total_samples} sequences")
    
    # Create a few test sequences
    sequences = []
    for i in range(min(5, total_samples)):  # Just test first 5 sequences
        seq_input = df.iloc[i : i + input_window]
        seq_output = df.iloc[i + input_window + forecast_horizon - 1]
        sequences.append((seq_input, seq_output))
    
    print(f"✓ Created {len(sequences)} test sequences")
    print(f"  Input sequence shape: {sequences[0][0].shape}")
    print(f"  Output target: {sequences[0][1]['Q']}")
    
    return True

def test_metrics():
    """Test evaluation metrics."""
    print("\n" + "=" * 60)
    print("Testing Evaluation Metrics")
    print("=" * 60)
    
    # Create some test data
    np.random.seed(42)
    y_true = np.random.rand(100) * 100 + 50  # Random values between 50-150
    y_pred = y_true + np.random.normal(0, 10, 100)  # Add some noise
    
    try:
        # Test all metrics
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mase = mean_absolute_scaled_error(y_true, y_pred)
        nse = nash_sutcliffe_efficiency(y_true, y_pred)
        kge = kling_gupta_efficiency(y_true, y_pred)
        
        print(f"✓ All metrics computed successfully")
        print(f"  MAPE: {mape:.4f}")
        print(f"  MASE: {mase:.4f}")
        print(f"  NSE: {nse:.4f}")
        print(f"  KGE: {kge:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error computing metrics: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("Wavelet-LSTM CAMELS Inference Pipeline Test")
    print("=" * 60)
    
    tests = [
        test_data_loading,
        test_preprocessing, 
        test_modwt_features,
        test_sequence_creation,
        test_metrics
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            if isinstance(result, tuple):
                results.append(result[0])
            else:
                results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    test_names = [
        "Data Loading",
        "Preprocessing", 
        "MODWT Features",
        "Sequence Creation",
        "Metrics"
    ]
    
    for name, result in zip(test_names, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nThe inference pipeline is ready to use!")
        print("You can now run inference.py with trained models.")
    else:
        print("\nPlease fix the failing tests before using inference.py")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 