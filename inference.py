import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

# Set TensorFlow logging level before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K

from feature_engineering import MODWTFeatureEngineer
from metrics import *

def setup_logging(log_level: str) -> None:
    """
    Sets up logging configuration.

    Args:
        log_level (str): Logging level as a string (e.g., 'DEBUG', 'INFO').
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}")
        sys.exit(1)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Performs inference and evaluation using trained wavelet and baseline LSTM models for CAMELS streamflow forecasting."
    )
    parser.add_argument(
        '--station_id',
        type=str,
        required=True,
        help='Station ID for the catchment (e.g., 01013500). Must match both CSV filename and model directory.'
    )
    parser.add_argument(
        '--forecast_horizon',
        type=int,
        required=True,
        choices=[1, 3, 5],
        help='Forecast horizon in days (1, 3, or 5).'
    )
    parser.add_argument(
        '--wavelet_filter',
        type=str,
        required=True,
        help='Wavelet filter name (e.g., db1, sym4, etc.).'
    )
    parser.add_argument(
        '--max_level',
        type=int,
        required=False,
        default=6,
        help='Maximum decomposition level used for MODWT (should match training).'
    )
    parser.add_argument(
        '--model_base_path',
        type=Path,
        required=False,
        default=Path("mnt/correct_output"),
        help='Path to the base directory where trained models are stored.'
    )
    parser.add_argument(
        '--data_base_path',
        type=Path,
        required=False,
        default=Path("wavelet-lstm-camels/data"),
        help='Path to the base directory where CAMELS CSV data is stored.'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        required=False,
        default='both',
        choices=['wavelet', 'baseline', 'both'],
        help='Which model(s) to use for inference: wavelet, baseline, or both.'
    )
    parser.add_argument(
        '--start_date',
        type=str,
        required=False,
        help='Start date for inference period (YYYY-MM-DD). If not provided, uses entire dataset.'
    )
    parser.add_argument(
        '--end_date',
        type=str,
        required=False,
        help='End date for inference period (YYYY-MM-DD). If not provided, uses entire dataset.'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=False,
        default=Path("inference_results"),
        help='Directory to save inference results.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )
    return parser.parse_args()

def r2_keras(y_true, y_pred):
    """
    Calculates the coefficient of determination (R²) for Keras models.
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - SS_res/(SS_tot + K.epsilon())
    return r2

class NashSutcliffeEfficiency(tf.keras.metrics.Metric):
    def __init__(self, name='nse', **kwargs):
        super(NashSutcliffeEfficiency, self).__init__(name=name, **kwargs)
        self.sum_squared_errors = self.add_weight(name='sum_squared_errors', initializer='zeros')
        self.sum_squared_total = self.add_weight(name='sum_squared_total', initializer='zeros')
        self.mean_observed = self.add_weight(name='mean_observed', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.sum_observed = self.add_weight(name='sum_observed', initializer='zeros')
        
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

def load_model_artifacts(model_path: Path, model_type: str = 'wavelet'):
    """
    Load model artifacts including the trained model, scalers, and selected features.
    
    Args:
        model_path: Path to the model directory
        model_type: 'wavelet' or 'baseline'
    
    Returns:
        dict: Dictionary containing loaded artifacts
    """
    artifacts = {}
    
    # Load model
    if model_type == 'wavelet':
        model_file = model_path / "model.keras"
        scaler_file = model_path / "feature_scaler.pkl"
        q_scaler_file = model_path / "q_scaler.pkl"
    else:  # baseline
        model_file = model_path / "baseline_model.keras"
        scaler_file = model_path / "baseline_feature_scaler.pkl"
        q_scaler_file = model_path / "baseline_q_scaler.pkl"
    
    # Load the model with custom objects
    custom_objects = {
        'r2_keras': r2_keras,
        'NashSutcliffeEfficiency': NashSutcliffeEfficiency,
        'KlingGuptaEfficiency': KlingGuptaEfficiency
    }
    
    artifacts['model'] = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
    
    # Load scalers
    with open(scaler_file, 'rb') as f:
        artifacts['feature_scaler'] = pickle.load(f)
    
    with open(q_scaler_file, 'rb') as f:
        artifacts['q_scaler'] = pickle.load(f)
    
    # Load selected features
    selected_features_file = model_path / "ea_cmi_tol_005_selected_feature_names.pkl"
    with open(selected_features_file, 'rb') as f:
        selected_features_dict = pickle.load(f)
    
    if model_type == 'wavelet':
        artifacts['selected_feature_names'] = selected_features_dict['selected_feature_names']
        artifacts['selected_feature_indices'] = selected_features_dict['selected_feature_indices']
    else:  # baseline
        artifacts['selected_feature_names'] = selected_features_dict['baseline_selected_feature_names']
        artifacts['selected_feature_indices'] = selected_features_dict['baseline_selected_feature_indices']
    
    return artifacts

def preprocess_data(df: pd.DataFrame, wavelet_filter: str, max_level: int, model_type: str = 'wavelet'):
    """
    Preprocess the data following the same pipeline as training.
    
    Args:
        df: Input DataFrame
        wavelet_filter: Wavelet filter name
        max_level: Maximum decomposition level
        model_type: 'wavelet' or 'baseline'
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Handle missing values
    df = df.copy()
    df.ffill(inplace=True)
    
    # Feature Selection and Engineering
    features = ['Q', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 
                'tmax(C)', 'tmin(C)', 'vp(Pa)']
    
    # Keep only required columns
    df = df[['date'] + features]
    
    # Sort by date
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Convert date to Unix timestamp
    df['timestamp'] = pd.to_datetime(df['date']).astype(int) / 10**9
    
    if model_type == 'wavelet':
        # Add MODWT features
        modwt_fe = MODWTFeatureEngineer(
            wavelet=wavelet_filter, 
            v_levels=[max_level], 
            w_levels=list(range(1, max_level+1))
        )
        df = modwt_fe.transform(df, features)
        
        # Remove rows with NaN due to boundary coefficients
        df.dropna(inplace=True)
    
    return df

def create_sequences(df: pd.DataFrame, forecast_horizon: int, input_window: int = 270, 
                    start_date: str = None, end_date: str = None):
    """
    Create sequences for inference.
    
    Args:
        df: Preprocessed DataFrame
        forecast_horizon: Forecast horizon in days
        input_window: Input window size in days
        start_date: Start date for inference period
        end_date: End date for inference period
    
    Returns:
        tuple: (sequences, dates) where sequences is list of (input, output) tuples
               and dates is list of prediction dates
    """
    # Filter by date range if provided
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    total_samples = len(df) - input_window - forecast_horizon + 1
    if total_samples <= 0:
        raise ValueError(f"Not enough data for inference. Need at least {input_window + forecast_horizon} days.")
    
    sequences = []
    prediction_dates = []
    
    for i in range(total_samples):
        seq_input = df.iloc[i : i + input_window]
        seq_output = df.iloc[i + input_window + forecast_horizon - 1]
        sequences.append((seq_input, seq_output))
        
        # Calculate prediction date
        pred_date = pd.to_datetime(seq_input['date'].iloc[-1]) + pd.Timedelta(days=forecast_horizon)
        prediction_dates.append(pred_date)
    
    return sequences, prediction_dates

def scale_sequences(sequences, feature_scaler, q_scaler, feature_columns, selected_feature_indices):
    """
    Scale sequences using the provided scalers and select features.
    
    Args:
        sequences: List of (input, output) sequence tuples
        feature_scaler: Fitted feature scaler
        q_scaler: Fitted target scaler
        feature_columns: List of feature column names
        selected_feature_indices: Indices of selected features
    
    Returns:
        tuple: (X, y) scaled arrays
    """
    X = []
    y = []
    
    for seq_input, seq_output in sequences:
        # Scale input features
        scaled_input = feature_scaler.transform(seq_input[feature_columns])
        
        # Select only the features that were selected during training
        scaled_input = scaled_input[:, selected_feature_indices]
        
        # Scale target
        try:
            scaled_q = q_scaler.transform(seq_output['Q'].values.reshape(-1, 1)).flatten()
        except:
            scaled_q = q_scaler.transform(np.array([[seq_output["Q"]]])).flatten()
        
        X.append(scaled_input)
        y.append(scaled_q)
    
    return np.array(X), np.array(y)

def evaluate_predictions(y_true, y_pred):
    """
    Calculate evaluation metrics for predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Convert to numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate metrics
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    metrics['mase'] = mean_absolute_scaled_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['nse'] = nash_sutcliffe_efficiency(y_true, y_pred)
    metrics['kge'] = kling_gupta_efficiency(y_true, y_pred)
    
    return metrics

def main(args: argparse.Namespace) -> int:
    """
    Main function for inference and evaluation.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        int: Exit status code
    """
    logger = logging.getLogger(__name__)
    logger.info("Inference script started.")
    
    # Construct paths
    csv_file = args.data_base_path / f"{args.station_id}_camels.csv"
    model_dir = args.model_base_path / args.station_id / f"leadtime_{args.forecast_horizon}" / args.wavelet_filter
    
    # Check if files exist
    if not csv_file.exists():
        logger.error(f"CSV file does not exist: {csv_file}")
        return 1
    
    if not model_dir.exists():
        logger.error(f"Model directory does not exist: {model_dir}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file, parse_dates=['date'])
    
    results = {}
    
    # Process each model type
    model_types = ['wavelet', 'baseline'] if args.model_type == 'both' else [args.model_type]
    
    for model_type in model_types:
        logger.info(f"Processing {model_type} model...")
        
        try:
            # Load model artifacts
            artifacts = load_model_artifacts(model_dir, model_type)
            
            # Preprocess data
            processed_df = preprocess_data(df, args.wavelet_filter, args.max_level, model_type)
            
            # Get feature columns based on model type
            if model_type == 'wavelet':
                feature_columns = list(processed_df.columns)
                feature_columns.remove("date")
            else:  # baseline
                feature_columns = [
                    "Q", "timestamp", "dayl(s)", "prcp(mm/day)", "srad(W/m2)", 
                    "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"
                ]
            
            # Create sequences
            sequences, prediction_dates = create_sequences(
                processed_df, args.forecast_horizon, 
                start_date=args.start_date, end_date=args.end_date
            )
            
            # Scale sequences
            X, y_true_scaled = scale_sequences(
                sequences, 
                artifacts['feature_scaler'], 
                artifacts['q_scaler'],
                feature_columns,
                artifacts['selected_feature_indices']
            )
            
            # Make predictions
            logger.info(f"Making predictions with {model_type} model...")
            y_pred_scaled = artifacts['model'].predict(X, verbose=0)
            
            # Inverse transform predictions and true values
            y_pred = artifacts['q_scaler'].inverse_transform(y_pred_scaled)
            y_true = artifacts['q_scaler'].inverse_transform(y_true_scaled.reshape(-1, 1))
            
            # Flatten arrays
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
            
            # Evaluate predictions
            metrics = evaluate_predictions(y_true, y_pred)
            
            # Store results
            results[model_type] = {
                'predictions': y_pred,
                'true_values': y_true,
                'prediction_dates': prediction_dates,
                'metrics': metrics
            }
            
            # Print metrics
            logger.info(f"{model_type.capitalize()} Model Evaluation Metrics:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"  {metric_name.upper()}: {metric_value:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing {model_type} model: {str(e)}")
            continue
    
    # Save results
    output_file = args.output_dir / f"{args.station_id}_leadtime_{args.forecast_horizon}_{args.wavelet_filter}_results.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Create and save predictions DataFrame
    if results:
        pred_df_data = {'date': results[list(results.keys())[0]]['prediction_dates']}
        
        for model_type, result in results.items():
            pred_df_data[f'{model_type}_pred'] = result['predictions']
            pred_df_data[f'{model_type}_true'] = result['true_values']
        
        pred_df = pd.DataFrame(pred_df_data)
        csv_output_file = args.output_dir / f"{args.station_id}_leadtime_{args.forecast_horizon}_{args.wavelet_filter}_predictions.csv"
        pred_df.to_csv(csv_output_file, index=False)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Predictions saved to {csv_output_file}")
    
    logger.info("Inference script completed successfully.")
    return 0

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set logging level based on verbosity
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    
    exit_code = main(args)
    sys.exit(exit_code) 