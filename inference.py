#!/usr/bin/env python3
"""
Inference script for wavelet-based and baseline LSTM streamflow forecasting models.

This script loads pre-trained models and makes predictions for a specified station,
lead time, and wavelet filter. It can also evaluate model performance when target
data is available.

Usage:
    python inference.py --station_id 01013500 --leadtime 1 --wavelet_filter db1 
                       --start_date 2010-01-01 --end_date 2010-12-31
"""

import argparse
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# Import custom modules
from feature_engineering import MODWTFeatureEngineer
from metrics import (
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    nash_sutcliffe_efficiency,
    kling_gupta_efficiency
)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def r2_keras(y_true, y_pred):
    """R² metric for Keras models."""
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - SS_res/(SS_tot + K.epsilon())
    return r2

class NashSutcliffeEfficiency(tf.keras.metrics.Metric):
    """Nash-Sutcliffe Efficiency metric for Keras models."""
    def __init__(self, name='nse', **kwargs):
        super(NashSutcliffeEfficiency, self).__init__(name=name, **kwargs)
        self.sum_squared_errors = self.add_weight(name='sum_squared_errors', initializer='zeros')
        self.sum_squared_total = self.add_weight(name='sum_squared_total', initializer='zeros')
        self.sum_observed = self.add_weight(name='sum_observed', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        self.sum_observed.assign_add(tf.reduce_sum(y_true))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
        
        squared_errors = tf.square(y_true - y_pred)
        self.sum_squared_errors.assign_add(tf.reduce_sum(squared_errors))
        
        sum_y_true_sq = tf.reduce_sum(tf.square(y_true))
        self.sum_squared_total.assign_add(sum_y_true_sq)
        
    def result(self):
        mean_observed = self.sum_observed / self.count
        sum_squared_total = self.sum_squared_total - (tf.square(self.sum_observed) / self.count)
        nse = 1 - (self.sum_squared_errors / (sum_squared_total + K.epsilon()))
        return nse
    
    def reset_state(self):
        self.sum_squared_errors.assign(0.0)
        self.sum_squared_total.assign(0.0)
        self.sum_observed.assign(0.0)
        self.count.assign(0.0)

class KlingGuptaEfficiency(tf.keras.metrics.Metric):
    """Kling-Gupta Efficiency metric for Keras models."""
    def __init__(self, name='kge', **kwargs):
        super(KlingGuptaEfficiency, self).__init__(name=name, **kwargs)
        self.sum_true = self.add_weight(name='sum_true', initializer='zeros')
        self.sum_pred = self.add_weight(name='sum_pred', initializer='zeros')
        self.sum_true_sq = self.add_weight(name='sum_true_sq', initializer='zeros')
        self.sum_pred_sq = self.add_weight(name='sum_pred_sq', initializer='zeros')
        self.sum_true_pred = self.add_weight(name='sum_true_pred', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        self.sum_true.assign_add(tf.reduce_sum(y_true))
        self.sum_pred.assign_add(tf.reduce_sum(y_pred))
        self.sum_true_sq.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.sum_pred_sq.assign_add(tf.reduce_sum(tf.square(y_pred)))
        self.sum_true_pred.assign_add(tf.reduce_sum(y_true * y_pred))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
        
    def result(self):
        mean_true = self.sum_true / self.count
        mean_pred = self.sum_pred / self.count
        
        var_true = (self.sum_true_sq / self.count) - tf.square(mean_true)
        var_pred = (self.sum_pred_sq / self.count) - tf.square(mean_pred)
        
        std_true = tf.sqrt(var_true + K.epsilon())
        std_pred = tf.sqrt(var_pred + K.epsilon())
        
        cov = (self.sum_true_pred / self.count) - (mean_true * mean_pred)
        r = cov / (std_true * std_pred + K.epsilon())
        
        alpha = std_pred / (std_true + K.epsilon())
        beta = mean_pred / (mean_true + K.epsilon())
        
        kge = 1 - tf.sqrt(tf.square(r - 1) + tf.square(alpha - 1) + tf.square(beta - 1))
        return kge
    
    def reset_state(self):
        self.sum_true.assign(0.0)
        self.sum_pred.assign(0.0)
        self.sum_true_sq.assign(0.0)
        self.sum_pred_sq.assign(0.0)
        self.sum_true_pred.assign(0.0)
        self.count.assign(0.0)

def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess CAMELS data."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path, parse_dates=['date'])
    
    # Handle missing values
    df.ffill(inplace=True)
    
    # Feature selection
    features = ['Q', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 
                'tmax(C)', 'tmin(C)', 'vp(Pa)']
    df = df[['date'] + features]
    
    # Sort by date
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Add timestamp
    df['timestamp'] = pd.to_datetime(df['date']).astype(int) / 10**9
    
    return df

def create_sequences(df: pd.DataFrame, input_window: int = 270, 
                    forecast_horizon: int = 1) -> List[Tuple]:
    """Create input-output sequences for prediction."""
    logger = logging.getLogger(__name__)
    
    total_samples = len(df) - input_window - forecast_horizon + 1
    if total_samples <= 0:
        logger.error(f"Not enough data to create sequences. Need at least {input_window + forecast_horizon} days.")
        return []
    
    sequences = []
    for i in range(total_samples):
        seq_input = df.iloc[i : i + input_window]
        seq_output = df.iloc[i + input_window + forecast_horizon - 1]
        sequences.append((seq_input, seq_output))
    
    logger.info(f"Created {len(sequences)} sequences")
    return sequences

def filter_sequences_by_date_range(sequences: List[Tuple], start_date: str, 
                                 end_date: str, forecast_horizon: int) -> List[Tuple]:
    """Filter sequences to only include those with target dates in the specified range."""
    logger = logging.getLogger(__name__)
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    filtered_sequences = []
    for seq_input, seq_output in sequences:
        # The target date is the date of seq_output
        target_date = pd.to_datetime(seq_output['date'])
        if start_date <= target_date <= end_date:
            filtered_sequences.append((seq_input, seq_output))
    
    logger.info(f"Filtered to {len(filtered_sequences)} sequences in date range {start_date.date()} to {end_date.date()}")
    return filtered_sequences

def scale_sequences(sequences: List[Tuple], x_scaler: MinMaxScaler, 
                   y_scaler: MinMaxScaler, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Scale input sequences using pre-fitted scalers."""
    X = []
    y = []
    
    for seq_input, seq_output in sequences:
        scaled_input = x_scaler.transform(seq_input[feature_cols])
        
        # Handle both Series and scalar cases for seq_output['Q']
        try:
            scaled_q = y_scaler.transform(seq_output['Q'].values.reshape(-1, 1)).flatten()
        except:
            scaled_q = y_scaler.transform(np.array([[seq_output["Q"]]])).flatten()
        
        X.append(scaled_input)
        y.append(scaled_q)
    
    return np.array(X), np.array(y)

def create_model_architecture(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Create the LSTM model architecture as defined in main.py."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
    
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(256, activation='tanh', return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='linear'))
    
    return model

def load_compatible_model(model_path: Path, custom_objects: Dict, input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Load a Keras model with compatibility fixes for different TensorFlow versions."""
    logger = logging.getLogger(__name__)
    
    logger.info("Reconstructing model architecture and loading weights...")
    
    try:
        # Create the model architecture manually
        model = create_model_architecture(input_shape)
        
        # Load weights from the saved model
        # For .keras files, we need to extract weights differently
        import tempfile
        import zipfile
        import json
        
        # .keras files are zip archives
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract the .keras file
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            # Look for weights file
            weights_file = temp_path / "model.weights.h5"
            if weights_file.exists():
                model.load_weights(weights_file)
                logger.info("Successfully loaded weights from extracted .keras file")
                return model
            else:
                # Try to find any .h5 file in the extracted content
                h5_files = list(temp_path.glob("*.h5"))
                if h5_files:
                    model.load_weights(h5_files[0])
                    logger.info(f"Successfully loaded weights from {h5_files[0].name}")
                    return model
                else:
                    raise FileNotFoundError("No weights file found in .keras archive")
                    
    except Exception as e:
        logger.error(f"Failed to reconstruct model: {e}")
        raise e

def load_model_artifacts(model_dir: Path) -> Dict:
    """Load all model artifacts from the specified directory."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model artifacts from {model_dir}")
    
    artifacts = {}
    
    # Load feature selection info
    with open(model_dir / "ea_cmi_tol_005_selected_feature_names.pkl", 'rb') as f:
        feature_info = pickle.load(f)
        artifacts['selected_feature_names'] = feature_info['selected_feature_names']
        artifacts['selected_feature_indices'] = feature_info['selected_feature_indices']
        artifacts['baseline_selected_feature_names'] = feature_info['baseline_selected_feature_names']
        artifacts['baseline_selected_feature_indices'] = feature_info['baseline_selected_feature_indices']
    
    # Load scalers
    with open(model_dir / "feature_scaler.pkl", 'rb') as f:
        artifacts['feature_scaler'] = pickle.load(f)
    
    with open(model_dir / "q_scaler.pkl", 'rb') as f:
        artifacts['q_scaler'] = pickle.load(f)
    
    with open(model_dir / "baseline_feature_scaler.pkl", 'rb') as f:
        artifacts['baseline_feature_scaler'] = pickle.load(f)
    
    with open(model_dir / "baseline_q_scaler.pkl", 'rb') as f:
        artifacts['baseline_q_scaler'] = pickle.load(f)
    
    # Load models with custom objects
    custom_objects = {
        'r2_keras': r2_keras,
        'NashSutcliffeEfficiency': NashSutcliffeEfficiency,
        'KlingGuptaEfficiency': KlingGuptaEfficiency
    }
    
    # Determine input shapes based on selected features
    wavelet_input_shape = (270, len(artifacts['selected_feature_indices']))
    baseline_input_shape = (270, len(artifacts['baseline_selected_feature_indices']))
    
    # Load models using the compatible loader
    artifacts['wavelet_model'] = load_compatible_model(
        model_dir / "model.keras", custom_objects, wavelet_input_shape
    )
    artifacts['baseline_model'] = load_compatible_model(
        model_dir / "baseline_model.keras", custom_objects, baseline_input_shape
    )
    
    logger.info("Successfully loaded all model artifacts")
    return artifacts

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'mase': mean_absolute_scaled_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'nse': nash_sutcliffe_efficiency(y_true, y_pred),
        'kge': kling_gupta_efficiency(y_true, y_pred)
    }
    
    return metrics

def run_inference(station_id: str, leadtime: int, wavelet_filter: str,
                 start_date: str, end_date: str, max_level: int = 6,
                 output_file: Optional[str] = None) -> pd.DataFrame:
    """Run inference for the specified configuration."""
    logger = logging.getLogger(__name__)
    
    # Define paths
    data_path = Path(f"data/{station_id}_camels.csv")
    model_dir = Path(f"../mnt/correct_output/{station_id}/leadtime_{leadtime}/{wavelet_filter}")
    
    # Validate paths
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load data
    df = load_data(data_path)
    
    # Apply MODWT feature engineering
    logger.info(f"Applying MODWT feature engineering with {wavelet_filter} wavelet")
    features = ['Q', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 
                'tmax(C)', 'tmin(C)', 'vp(Pa)']
    
    modwt_fe = MODWTFeatureEngineer(
        wavelet=wavelet_filter, 
        v_levels=[max_level], 
        w_levels=list(range(1, max_level+1))
    )
    df = modwt_fe.transform(df, features)
    
    # Remove NaN values from boundary effects
    df.dropna(inplace=True)
    
    # Get feature columns
    feature_columns = list(df.columns)
    feature_columns.remove("date")
    
    # Create sequences
    sequences = create_sequences(df, input_window=270, forecast_horizon=leadtime)
    
    if not sequences:
        raise ValueError("No sequences could be created from the data")
    
    # Filter by date range if specified
    if start_date and end_date:
        sequences = filter_sequences_by_date_range(sequences, start_date, end_date, leadtime)
        
        if not sequences:
            raise ValueError(f"No sequences found in date range {start_date} to {end_date}")
    
    # Load model artifacts
    artifacts = load_model_artifacts(model_dir)
    
    # Prepare baseline features
    baseline_feature_names = [
        "Q", "timestamp", "dayl(s)", "prcp(mm/day)", "srad(W/m2)", 
        "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"
    ]
    
    # Scale sequences
    X_wavelet, y_true_scaled = scale_sequences(
        sequences, artifacts['feature_scaler'], artifacts['q_scaler'], feature_columns
    )
    
    X_baseline, _ = scale_sequences(
        sequences, artifacts['baseline_feature_scaler'], artifacts['baseline_q_scaler'], 
        baseline_feature_names
    )
    
    # Select features
    X_wavelet = X_wavelet[:, :, artifacts['selected_feature_indices']]
    X_baseline = X_baseline[:, :, artifacts['baseline_selected_feature_indices']]
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred_wavelet_scaled = artifacts['wavelet_model'].predict(X_wavelet, verbose=0)
    y_pred_baseline_scaled = artifacts['baseline_model'].predict(X_baseline, verbose=0)
    
    # Inverse transform predictions
    y_pred_wavelet = artifacts['q_scaler'].inverse_transform(y_pred_wavelet_scaled)
    y_pred_baseline = artifacts['baseline_q_scaler'].inverse_transform(y_pred_baseline_scaled)
    y_true = artifacts['q_scaler'].inverse_transform(y_true_scaled.reshape(-1, 1))
    
    # Create results DataFrame
    delta = pd.Timedelta(days=leadtime)
    dates = [pd.to_datetime(seq_output["date"]) for seq_input, seq_output in sequences]
    
    results_df = pd.DataFrame({
        'date': dates,
        'y_true': y_true.flatten(),
        'y_pred_wavelet': y_pred_wavelet.flatten(),
        'y_pred_baseline': y_pred_baseline.flatten()
    })
    
    # Compute metrics
    wavelet_metrics = compute_metrics(y_true, y_pred_wavelet)
    baseline_metrics = compute_metrics(y_true, y_pred_baseline)
    
    # Print results
    print(f"\n=== Inference Results for Station {station_id} ===")
    print(f"Lead time: {leadtime} days")
    print(f"Wavelet filter: {wavelet_filter}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Number of predictions: {len(results_df)}")
    
    print(f"\n=== Wavelet Model Performance ===")
    for metric, value in wavelet_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print(f"\n=== Baseline Model Performance ===")
    for metric, value in baseline_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print(f"\n=== Selected Features ===")
    print(f"Wavelet model features ({len(artifacts['selected_feature_names'])}):")
    for i, feature in enumerate(artifacts['selected_feature_names']):
        print(f"  {i+1}. {feature}")
    
    print(f"\nBaseline model features ({len(artifacts['baseline_selected_feature_names'])}):")
    for i, feature in enumerate(artifacts['baseline_selected_feature_names']):
        print(f"  {i+1}. {feature}")
    
    # Save results if requested
    if output_file:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    
    return results_df

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run inference with pre-trained wavelet and baseline LSTM models"
    )
    
    parser.add_argument(
        '--station_id', 
        type=str, 
        required=True,
        help='Station ID (e.g., 01013500)'
    )
    
    parser.add_argument(
        '--leadtime', 
        type=int, 
        required=True,
        choices=[1, 3, 5],
        help='Forecast lead time in days (1, 3, or 5)'
    )
    
    parser.add_argument(
        '--wavelet_filter', 
        type=str, 
        required=True,
        help='Wavelet filter name (e.g., db1, db2, sym4, etc.)'
    )
    
    parser.add_argument(
        '--start_date', 
        type=str, 
        required=True,
        help='Start date for inference (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end_date', 
        type=str, 
        required=True,
        help='End date for inference (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--max_level', 
        type=int, 
        default=6,
        help='Maximum MODWT decomposition level (default: 6)'
    )
    
    parser.add_argument(
        '--output_file', 
        type=str,
        help='Output CSV file path (optional)'
    )
    
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Run inference
        results_df = run_inference(
            station_id=args.station_id,
            leadtime=args.leadtime,
            wavelet_filter=args.wavelet_filter,
            start_date=args.start_date,
            end_date=args.end_date,
            max_level=args.max_level,
            output_file=args.output_file
        )
        
        print(f"\nInference completed successfully!")
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 