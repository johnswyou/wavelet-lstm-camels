#!/usr/bin/env python3
"""
Example script demonstrating how to use the inference.py script for 
wavelet-LSTM streamflow forecasting inference and evaluation.

This script shows various usage patterns for the inference script.
"""

import subprocess
import sys
from pathlib import Path

def run_inference_example():
    """
    Run example inference commands to demonstrate the script usage.
    """
    
    # Example 1: Basic inference with both wavelet and baseline models
    print("=" * 60)
    print("Example 1: Basic inference with both models")
    print("=" * 60)
    
    cmd1 = [
        "python", "inference.py",
        "--station_id", "01013500",
        "--forecast_horizon", "1", 
        "--wavelet_filter", "db1",
        "--model_type", "both",
        "--verbose"
    ]
    
    print("Command:", " ".join(cmd1))
    print("This will:")
    print("- Load both wavelet and baseline models for station 01013500")
    print("- Use 1-day forecast horizon with db1 wavelet")
    print("- Evaluate on the entire available dataset")
    print("- Save results to inference_results/ directory")
    print()
    
    # Example 2: Inference with specific date range
    print("=" * 60)
    print("Example 2: Inference with specific date range")
    print("=" * 60)
    
    cmd2 = [
        "python", "inference.py",
        "--station_id", "01013500",
        "--forecast_horizon", "3",
        "--wavelet_filter", "sym4", 
        "--model_type", "wavelet",
        "--start_date", "2010-01-01",
        "--end_date", "2015-12-31",
        "--output_dir", "custom_results"
    ]
    
    print("Command:", " ".join(cmd2))
    print("This will:")
    print("- Use only the wavelet model")
    print("- 3-day forecast horizon with sym4 wavelet")
    print("- Evaluate only on data from 2010-2015")
    print("- Save results to custom_results/ directory")
    print()
    
    # Example 3: Baseline model only with different wavelet
    print("=" * 60)
    print("Example 3: Baseline model with different settings")
    print("=" * 60)
    
    cmd3 = [
        "python", "inference.py",
        "--station_id", "02096846",
        "--forecast_horizon", "5",
        "--wavelet_filter", "coif2",
        "--model_type", "baseline",
        "--max_level", "6"
    ]
    
    print("Command:", " ".join(cmd3))
    print("This will:")
    print("- Use only the baseline model")
    print("- 5-day forecast horizon")
    print("- Use coif2 wavelet filter settings")
    print("- Use max_level=6 (should match training)")
    print()
    
    # Show available stations
    print("=" * 60)
    print("Available stations (examples from data directory):")
    print("=" * 60)
    
    data_dir = Path("wavelet-lstm-camels/data")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*_camels.csv"))
        station_ids = [f.stem.split("_")[0] for f in csv_files[:10]]  # Show first 10
        print("Station IDs:", ", ".join(station_ids))
        if len(csv_files) > 10:
            print(f"... and {len(csv_files) - 10} more stations")
    else:
        print("Data directory not found. Please check the path.")
    print()
    
    # Show available wavelets
    print("=" * 60)
    print("Available wavelet filters:")
    print("=" * 60)
    
    wavelets = [
        "bl7", "coif1", "coif2", "db1", "db2", "db3", "db4", "db5", 
        "db6", "db7", "fk4", "fk6", "fk8", "fk14", "han2_3", "han3_3",
        "han4_5", "han5_5", "mb4_2", "mb8_2", "mb8_3", "mb8_4", 
        "mb10_3", "mb12_3", "mb14_3", "sym4", "sym5", "sym6", "sym7", 
        "la8", "la10", "la12", "la14"
    ]
    
    print("Wavelet filters:", ", ".join(wavelets))
    print()
    
    print("=" * 60)
    print("Usage Notes:")
    print("=" * 60)
    print("1. Make sure the station_id matches both:")
    print("   - A CSV file: wavelet-lstm-camels/data/{station_id}_camels.csv")
    print("   - A model directory: mnt/correct_output/{station_id}/leadtime_{horizon}/{wavelet}/")
    print()
    print("2. The script will output:")
    print("   - Evaluation metrics (NSE, KGE, RMSE, MAE, MAPE, MASE, R²)")
    print("   - A pickle file with detailed results")
    print("   - A CSV file with predictions and true values")
    print()
    print("3. Use --verbose for detailed logging")
    print("4. Use --start_date and --end_date to limit evaluation period")
    print("5. Use --model_type to choose wavelet, baseline, or both models")

if __name__ == "__main__":
    run_inference_example() 