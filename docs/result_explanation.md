# Result Directory Structure

The `correct_output/` directory is approximately **412 GB** in size and contains **923,180 items**. This data is no longer stored in S3 and is available upon request. Please contact John You at johnswyou@gmail.com.

To make use of the trained outputs for streamflow forecasting, see the [wavelet-streamflow-forecast](https://github.com/johnswyou/wavelet-streamflow-forecast) repository.

## Directory Structure

```
correct_output/
├── [STATION_ID]/                    # e.g., 01013500, 01022500, etc.
│   ├── leadtime_1/
│   │   ├── bl7/
│   │   ├── coif1/
│   │   ├── coif2/
│   │   ├── db1/
│   │   ├── db2/
│   │   ├── db3/
│   │   ├── db4/
│   │   ├── db5/
│   │   ├── db6/
│   │   ├── db7/
│   │   ├── fk14/
│   │   ├── fk4/
│   │   ├── fk6/
│   │   ├── fk8/
│   │   ├── han2_3/
│   │   ├── han3_3/
│   │   ├── han4_5/
│   │   ├── han5_5/
│   │   ├── la10/
│   │   ├── la12/
│   │   ├── la14/
│   │   ├── la8/
│   │   ├── mb10_3/
│   │   ├── mb12_3/
│   │   ├── mb14_3/
│   │   ├── mb4_2/
│   │   ├── mb8_2/
│   │   ├── mb8_3/
│   │   ├── mb8_4/
│   │   ├── sym4/
│   │   ├── sym5/
│   │   ├── sym6/
│   │   └── sym7/
│   ├── leadtime_3/
│   │   └── [same wavelet directories as leadtime_1]
│   └── leadtime_5/
│       └── [same wavelet directories as leadtime_1]
```

**At the deepest level** (e.g., `correct_output/01013500/leadtime_1/db1/`), each directory contains these files:

- `baseline_feature_scaler.pkl`
- `baseline_history.pkl`
- `baseline_model.keras`
- `baseline_pred_label_df.pkl`
- `baseline_q_scaler.pkl`
- `baseline_test_metrics_dict.pkl`
- `ea_cmi_tol_005_selected_feature_names.pkl`
- `feature_scaler.pkl`
- `history.pkl`
- `model.keras`
- `pred_label_df.pkl`
- `q_scaler.pkl`
- `test_metrics_dict.pkl`
- `timings.pkl`

Where:

- **Station IDs** represent different streamflow gauge locations
- **Lead times** (1, 3, 5) represent different forecast horizons
- **Wavelet names** (db1, db2, sym4, etc.) represent different wavelets used for feature engineering
- **Files** contain trained models, scalers, predictions, metrics, and timing information for each experiment configuration

---

## Detailed File Descriptions

### Scaler Files

#### `feature_scaler.pkl` & `baseline_feature_scaler.pkl`
- **Type**: `sklearn.preprocessing.MinMaxScaler` objects
- **Purpose**: Fitted scalers for input features
- **Content**:
  - `feature_scaler.pkl`: Scaler fitted on MODWT-transformed features (wavelet coefficients + original features)
  - `baseline_feature_scaler.pkl`: Scaler fitted on baseline features: ["Q", "timestamp", "dayl(s)", "prcp(mm/day)", "srad(W/m2)", "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]
- **Usage**: Used to transform input features to [0,1] range during inference

#### `q_scaler.pkl` & `baseline_q_scaler.pkl`
- **Type**: `sklearn.preprocessing.MinMaxScaler` objects
- **Purpose**: Fitted scalers for target variable 'Q' (streamflow)
- **Content**: Scalers fitted on the target variable during training
- **Usage**: Used to scale target values during training and inverse-transform predictions back to original units

### Feature Selection File

#### `ea_cmi_tol_005_selected_feature_names.pkl`
- **Type**: Dictionary
- **Content Structure**:
```python
{
    "selected_feature_names": list,           # Selected wavelet feature names
    "selected_feature_indices": np.array,     # 0-based indices of selected features
    "selected_feature_scores": np.array,      # Feature importance scores
    "baseline_selected_feature_names": list,  # Selected baseline feature names
    "baseline_selected_feature_indices": np.array,  # Baseline feature indices
    "baseline_selected_feature_scores": np.array    # Baseline feature scores
}
```
- **Purpose**: Results from hydroIVS R package feature selection using "ea_cmi_tol" method with tolerance 0.05
- **Usage**: Determines which features to use for model training and inference

### Training History Files

#### `history.pkl` & `baseline_history.pkl`
- **Type**: Dictionary (from `keras.callbacks.History.history`)
- **Content Structure**:
```python
{
    'loss': [epoch1_loss, epoch2_loss, ...],
    'val_loss': [epoch1_val_loss, epoch2_val_loss, ...],
    'mae': [epoch1_mae, epoch2_mae, ...],
    'val_mae': [epoch1_val_mae, epoch2_val_mae, ...],
    'mse': [epoch1_mse, epoch2_mse, ...],
    'val_mse': [epoch1_val_mse, epoch2_val_mse, ...],
    'mape': [epoch1_mape, epoch2_mape, ...],
    'val_mape': [epoch1_val_mape, epoch2_val_mape, ...],
    'r2_keras': [epoch1_r2, epoch2_r2, ...],
    'val_r2_keras': [epoch1_val_r2, epoch2_val_r2, ...],
    'nse': [epoch1_nse, epoch2_nse, ...],
    'val_nse': [epoch1_val_nse, epoch2_val_nse, ...],
    'kge': [epoch1_kge, epoch2_kge, ...],
    'val_kge': [epoch1_val_kge, epoch2_val_kge, ...]
}
```
- **Purpose**: Complete training history including losses and metrics for each epoch
- **Usage**: Analyzing model training progression, plotting learning curves

### Test Metrics Files

#### `test_metrics_dict.pkl` & `baseline_test_metrics_dict.pkl`
- **Type**: Dictionary
- **Content Structure**:
```python
{
    "nse": float,    # Nash-Sutcliffe Efficiency
    "kge": float,    # Kling-Gupta Efficiency
    "rmse": float,   # Root Mean Square Error
    "mae": float,    # Mean Absolute Error
    "mape": float,   # Mean Absolute Percentage Error
    "mase": float,   # Mean Absolute Scaled Error
    "r2": float      # R-squared (coefficient of determination)
}
```
- **Purpose**: Final evaluation metrics computed on the test set
- **Usage**: Model performance comparison and reporting

### Prediction Files

#### `pred_label_df.pkl`
- **Type**: Pandas DataFrame
- **Content Structure**:
```python
DataFrame with columns:
- 'date': pd.Timestamp      # Prediction dates
- 'y_pred': float           # Wavelet model predictions
- 'y_true': float           # True streamflow values
```

#### `baseline_pred_label_df.pkl`
- **Type**: Pandas DataFrame
- **Content Structure**:
```python
DataFrame with columns:
- 'date': pd.Timestamp        # Prediction dates
- 'y_pred': float             # Wavelet model predictions
- 'y_true': float             # True streamflow values
- 'baseline_y_pred': float    # Baseline model predictions
- 'baseline_y_true': float    # Baseline true values (same as y_true)
```
- **Purpose**: Complete prediction results for both models with corresponding dates
- **Usage**: Time series analysis, plotting predictions vs observations

### Timing File

#### `timings.pkl`
- **Type**: Dictionary
- **Content Structure**:
```python
{
    'modwt': float,        # Time for MODWT feature engineering (seconds)
    'ea_cmi_tol': float,   # Time for feature selection (seconds)
    'lstm': float          # Time for LSTM model training (seconds)
}
```
- **Purpose**: Performance benchmarking of different processing steps
- **Usage**: Computational efficiency analysis

---

## Key Notes

1. **Consistency**: Each directory has identical file structure but different content based on the specific station, lead time, and wavelet filter combination.

2. **Data Flow**: The scalers are fitted during training and used during inference to ensure consistent data preprocessing.

3. **Model Types**: Files distinguish between "wavelet" models (using MODWT features) and "baseline" models (using standard meteorological features).

4. **Direct Forecasting**: All models use direct forecasting (predicting a single future value) rather than recursive forecasting.

5. **Feature Selection**: The `ea_cmi_tol_005_selected_feature_names.pkl` file is crucial as it determines which subset of available features each model actually uses.

This structure allows for complete reproducibility of experiments and comprehensive analysis of model performance across different configurations.
