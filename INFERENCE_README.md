# Wavelet-LSTM CAMELS Inference Script

This document describes how to use the `inference.py` script to perform inference and evaluation using trained wavelet and baseline LSTM models for streamflow forecasting.

## Overview

The `inference.py` script allows you to:
- Load pre-trained wavelet and baseline LSTM models
- Perform inference on CAMELS streamflow data
- Evaluate model performance using standard hydrological metrics
- Save predictions and evaluation results

## Requirements

- Python 3.7+
- TensorFlow/Keras
- scikit-learn
- pandas
- numpy
- The same dependencies as the training script (`main.py`)

## Usage

### Basic Command Structure

```bash
python inference.py --station_id STATION_ID --forecast_horizon HORIZON --wavelet_filter WAVELET [OPTIONS]
```

### Required Arguments

- `--station_id`: Station ID (e.g., "01013500"). Must match both the CSV filename and model directory.
- `--forecast_horizon`: Forecast horizon in days (1, 3, or 5).
- `--wavelet_filter`: Wavelet filter name (e.g., "db1", "sym4", "coif2").

### Optional Arguments

- `--model_type`: Which model(s) to use ("wavelet", "baseline", or "both"). Default: "both"
- `--max_level`: Maximum decomposition level for MODWT. Default: 6
- `--model_base_path`: Path to trained models directory. Default: "mnt/correct_output"
- `--data_base_path`: Path to CAMELS CSV data directory. Default: "wavelet-lstm-camels/data"
- `--start_date`: Start date for evaluation (YYYY-MM-DD). Optional.
- `--end_date`: End date for evaluation (YYYY-MM-DD). Optional.
- `--output_dir`: Directory to save results. Default: "inference_results"
- `--verbose`: Enable verbose logging.

## Examples

### Example 1: Basic Inference with Both Models

```bash
python inference.py \
    --station_id 01013500 \
    --forecast_horizon 1 \
    --wavelet_filter db1 \
    --model_type both \
    --verbose
```

This will:
- Load both wavelet and baseline models for station 01013500
- Use 1-day forecast horizon with db1 wavelet
- Evaluate on the entire available dataset
- Save results to `inference_results/` directory

### Example 2: Inference with Specific Date Range

```bash
python inference.py \
    --station_id 01013500 \
    --forecast_horizon 3 \
    --wavelet_filter sym4 \
    --model_type wavelet \
    --start_date 2010-01-01 \
    --end_date 2015-12-31 \
    --output_dir custom_results
```

This will:
- Use only the wavelet model
- 3-day forecast horizon with sym4 wavelet
- Evaluate only on data from 2010-2015
- Save results to `custom_results/` directory

### Example 3: Baseline Model Only

```bash
python inference.py \
    --station_id 02096846 \
    --forecast_horizon 5 \
    --wavelet_filter coif2 \
    --model_type baseline
```

This will:
- Use only the baseline model
- 5-day forecast horizon
- Use coif2 wavelet filter settings (for consistency with training)

## Input Requirements

### File Structure

The script expects the following file structure:

```
├── wavelet-lstm-camels/data/
│   └── {station_id}_camels.csv
├── mnt/correct_output/
│   └── {station_id}/
│       └── leadtime_{forecast_horizon}/
│           └── {wavelet_filter}/
│               ├── model.keras
│               ├── baseline_model.keras
│               ├── feature_scaler.pkl
│               ├── baseline_feature_scaler.pkl
│               ├── q_scaler.pkl
│               ├── baseline_q_scaler.pkl
│               └── ea_cmi_tol_005_selected_feature_names.pkl
```

### CSV Data Format

The CSV files should contain the following columns:
- `date`: Date in YYYY-MM-DD format
- `Q`: Streamflow (target variable)
- `dayl(s)`: Daylength in seconds
- `prcp(mm/day)`: Precipitation in mm/day
- `srad(W/m2)`: Solar radiation in W/m²
- `swe(mm)`: Snow water equivalent in mm
- `tmax(C)`: Maximum temperature in °C
- `tmin(C)`: Minimum temperature in °C
- `vp(Pa)`: Vapor pressure in Pa

## Output

The script generates the following outputs:

### 1. Console Output
- Evaluation metrics for each model:
  - NSE (Nash-Sutcliffe Efficiency)
  - KGE (Kling-Gupta Efficiency)
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  - MASE (Mean Absolute Scaled Error)
  - R² (Coefficient of Determination)

### 2. Pickle File
- `{station_id}_leadtime_{horizon}_{wavelet}_results.pkl`
- Contains detailed results including predictions, true values, dates, and metrics

### 3. CSV File
- `{station_id}_leadtime_{horizon}_{wavelet}_predictions.csv`
- Contains predictions and true values with dates in tabular format

## Available Stations and Wavelets

### Station IDs
Station IDs are extracted from the CSV filenames in the data directory. Examples include:
- 01013500, 01022500, 01030500, etc.

### Wavelet Filters
Available wavelet filters include:
- Daubechies: db1, db2, db3, db4, db5, db6, db7
- Symlets: sym4, sym5, sym6, sym7
- Coiflets: coif1, coif2
- Biorthogonal: bl7
- Fejér-Korovkin: fk4, fk6, fk8, fk14
- Haar: han2_3, han3_3, han4_5, han5_5
- Meyer: mb4_2, mb8_2, mb8_3, mb8_4, mb10_3, mb12_3, mb14_3
- Least Asymmetric: la8, la10, la12, la14

## Data Processing Pipeline

The script follows the same data processing pipeline as the training script:

1. **Data Loading**: Load CSV data and parse dates
2. **Preprocessing**: Handle missing values, feature selection
3. **MODWT Feature Engineering**: Apply wavelet decomposition (for wavelet models)
4. **Sequence Creation**: Create input-output sequences with 270-day input window
5. **Feature Scaling**: Apply pre-fitted scalers from training
6. **Feature Selection**: Use pre-selected features from training
7. **Inference**: Make predictions using trained models
8. **Evaluation**: Calculate metrics and save results

## Error Handling

The script includes error handling for common issues:
- Missing CSV files or model directories
- Insufficient data for the specified date range
- Model loading errors
- Feature scaling/selection mismatches

## Performance Considerations

- The script processes sequences in batches for memory efficiency
- MODWT feature engineering can be computationally intensive for large datasets
- Use date range filtering to limit evaluation period if needed

## Troubleshooting

### Common Issues

1. **File Not Found Errors**
   - Ensure station_id matches both CSV filename and model directory
   - Check that all required model files exist

2. **Feature Mismatch Errors**
   - Ensure max_level matches the training configuration
   - Verify that the same wavelet filter was used in training

3. **Insufficient Data Errors**
   - Check that the date range provides enough data (need 270+ days)
   - Verify that the CSV file contains the required columns

4. **Memory Issues**
   - Use date range filtering to process smaller chunks
   - Consider processing one model type at a time

### Getting Help

Run the example script to see usage patterns:
```bash
python example_inference.py
```

Use verbose logging for detailed information:
```bash
python inference.py --verbose [other arguments]
``` 