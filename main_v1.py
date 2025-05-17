import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Set TensorFlow logging level before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from rpy2.robjects import r

from feature_engineering import MODWTFeatureEngineer
from metrics import (
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    nash_sutcliffe_efficiency,
    kling_gupta_efficiency
)

# --- Configuration Constants ---
INPUT_WINDOW = 270
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# TEST_RATIO is implicitly 1.0 - TRAIN_RATIO - VAL_RATIO

FORECAST_HORIZONS = [1, 3, 5]
POSSIBLE_FILTERS = [
    "bl7", "coif1", "coif2", "db1", "db2", "db3", "db4", "db5",
    "db6", "db7", "fk4", "fk6", "fk8", "fk14", "han2_3", "han3_3",
    "han4_5", "han5_5", "mb4_2", "mb8_2", "mb8_3", "mb8_4", "mb10_3", "mb12_3",
    "mb14_3", "sym4", "sym5", "sym6", "sym7", "la8",
    "la10", "la12", "la14"
]

BASE_FEATURES_FOR_MODWT = ['Q', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']
BASELINE_MODEL_FEATURES = ["Q", "timestamp", "dayl(s)", "prcp(mm/day)", "srad(W/m2)", "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]
TARGET_COLUMN_NAME = 'Q'

LSTM_UNITS = 256
LSTM_ACTIVATION = 'tanh'
LSTM_DROPOUT = 0.4
LEARNING_RATE = 0.001
EPOCHS = 1000  # Max epochs, early stopping will determine actual
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = 'val_nse' # Custom metric name

# Rpy2 Feature Selection Config
IVS_METHOD = "ea_cmi_tol"
IVS_THRESHOLD = 0.05

# --- Logging Setup ---
def setup_logging(log_level: str) -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}")
        sys.exit(1)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# --- Argument Parsing ---
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trains wavelet and baseline LSTM models for streamflow forecasting (Version 1)."
    )
    parser.add_argument(
        '--csv_filename', type=Path, required=True,
        help='Filename of the CSV file for the catchment (e.g., 02096846_camels.csv).'
    )
    parser.add_argument(
        '--max_level', type=int, default=6,
        help='Maximum decomposition level for MODWT.'
    )
    parser.add_argument(
        '--base_save_path', type=Path, default=Path("output"),
        help='Base directory for saving all outputs.'
    )
    parser.add_argument(
        '--base_csv_path', type=Path, default=Path("data"),
        help='Base directory where CAMELS CSV data is stored.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose (DEBUG level) logging.'
    )
    return parser.parse_args()

# --- Keras Custom Metrics ---
def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

class NashSutcliffeEfficiencyMetric(tf.keras.metrics.Metric):
    def __init__(self, name='nse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_squared_errors = self.add_weight(name='sum_squared_errors', initializer='zeros')
        self.sum_y_true_sq = self.add_weight(name='sum_y_true_sq', initializer='zeros')
        self.sum_y_true = self.add_weight(name='sum_y_true', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        self.sum_y_true.assign_add(tf.reduce_sum(y_true_f))
        self.count.assign_add(tf.cast(tf.size(y_true_f), tf.float32))
        self.sum_squared_errors.assign_add(tf.reduce_sum(tf.square(y_true_f - y_pred_f)))
        self.sum_y_true_sq.assign_add(tf.reduce_sum(tf.square(y_true_f)))

    def result(self):
        mean_observed = self.sum_y_true / self.count
        denominator = self.sum_y_true_sq - (tf.square(self.sum_y_true) / self.count) # Sum of (y_true - mean(y_true))^2
        nse = 1 - (self.sum_squared_errors / (denominator + K.epsilon()))
        return nse

    def reset_state(self):
        self.sum_squared_errors.assign(0.0)
        self.sum_y_true_sq.assign(0.0)
        self.sum_y_true.assign(0.0)
        self.count.assign(0.0)

class KlingGuptaEfficiencyMetric(tf.keras.metrics.Metric):
    def __init__(self, name='kge', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_true = self.add_weight(name='sum_true', initializer='zeros')
        self.sum_pred = self.add_weight(name='sum_pred', initializer='zeros')
        self.sum_true_sq = self.add_weight(name='sum_true_sq', initializer='zeros')
        self.sum_pred_sq = self.add_weight(name='sum_pred_sq', initializer='zeros')
        self.sum_true_pred = self.add_weight(name='sum_true_pred', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        self.sum_true.assign_add(tf.reduce_sum(y_true_f))
        self.sum_pred.assign_add(tf.reduce_sum(y_pred_f))
        self.sum_true_sq.assign_add(tf.reduce_sum(tf.square(y_true_f)))
        self.sum_pred_sq.assign_add(tf.reduce_sum(tf.square(y_pred_f)))
        self.sum_true_pred.assign_add(tf.reduce_sum(y_true_f * y_pred_f))
        self.count.assign_add(tf.cast(tf.size(y_true_f), tf.float32))

    def result(self):
        mean_true = self.sum_true / self.count
        mean_pred = self.sum_pred / self.count
        var_true = (self.sum_true_sq / self.count) - tf.square(mean_true)
        var_pred = (self.sum_pred_sq / self.count) - tf.square(mean_pred)
        std_true = tf.sqrt(var_true + K.epsilon())
        std_pred = tf.sqrt(var_pred + K.epsilon())
        cov = (self.sum_true_pred / self.count) - (mean_true * mean_pred)
        
        r = cov / (std_true * std_pred + K.epsilon())
        alpha = std_pred / (std_true + K.epsilon()) # Ratio of std deviations
        beta = mean_pred / (mean_true + K.epsilon()) # Ratio of means
        
        kge = 1 - tf.sqrt(tf.square(r - 1) + tf.square(alpha - 1) + tf.square(beta - 1))
        return kge

    def reset_state(self):
        self.sum_true.assign(0.0); self.sum_pred.assign(0.0)
        self.sum_true_sq.assign(0.0); self.sum_pred_sq.assign(0.0)
        self.sum_true_pred.assign(0.0); self.count.assign(0.0)

# --- Helper Functions ---
def create_output_directory(base_path: Path, catchment_id: str, forecast_horizon: int, filter_name: Optional[str] = None) -> Path:
    catchment_dir = base_path / catchment_id
    horizon_dir = catchment_dir / f"leadtime_{forecast_horizon}"
    current_run_dir = horizon_dir / filter_name if filter_name else horizon_dir
    current_run_dir.mkdir(parents=True, exist_ok=True)
    return current_run_dir

def load_and_preprocess_data(file_path: Path, features_to_use: List[str]) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.debug(f"Reading CAMELS CSV file: {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV {file_path}: {e}")
        raise
        
    df.ffill(inplace=True)

    # Create 'timestamp' column immediately after loading and ffill, if 'date' exists.
    if 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date']).astype(int) / 10**9 
    else:
        logger.warning("Could not find 'date' column in the CSV. 'timestamp' column will not be created.")

    # Determine the final set of columns to keep.
    # This includes 'date', all `features_to_use`, and 'timestamp' if it was created or part of features_to_use.
    final_feature_set = set(['date'] + features_to_use)

    # If 'timestamp' is needed for baseline models (as indicated by BASELINE_MODEL_FEATURES)
    # and it isn't already in features_to_use, ensure it's part of the set we want to keep.
    # It should have been created above if 'date' was present in the CSV.
    if 'timestamp' in BASELINE_MODEL_FEATURES:
        final_feature_set.add('timestamp')

    # Select only the columns that are actually present in the DataFrame at this point.
    columns_to_keep = [col for col in list(final_feature_set) if col in df.columns]
    
    missing_columns_from_final_set = [col for col in list(final_feature_set) if col not in df.columns]
    if missing_columns_from_final_set:
        logger.warning(
            f"The following requested or derived columns are not available in the DataFrame "
            f"and will be omitted from the final selection: {missing_columns_from_final_set}"
        )
        if 'timestamp' in missing_columns_from_final_set and 'timestamp' in final_feature_set:
             logger.warning(
                 "Specifically, the 'timestamp' column was expected but is missing. "
                 "This might be due to a missing 'date' column in the input CSV, "
                 "or 'timestamp' itself was expected in the CSV but was absent."
            )

    df = df[columns_to_keep].copy() # Use .copy() to avoid SettingWithCopyWarning on subsequent operations

    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def engineer_modwt_features(df: pd.DataFrame, modwt_engineer: MODWTFeatureEngineer, features_for_modwt: List[str]) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.debug(f"Applying MODWT with wavelet: {modwt_engineer.wavelet_filter[:5]}... (showing first 5 coeffs)") # Example, actual wavelet name is in engineer
    df_modwt = modwt_engineer.transform(df.copy(), features_for_modwt)
    df_modwt.dropna(inplace=True) # Remove NaNs from MODWT boundary effects
    return df_modwt

def generate_sequences(df: pd.DataFrame, input_window: int, forecast_horizon: int, target_col: str) -> List[Tuple[pd.DataFrame, pd.Series]]:
    sequences = []
    total_samples = len(df) - input_window - forecast_horizon + 1
    for i in range(total_samples):
        seq_input_df = df.iloc[i : i + input_window]
        # Direct forecasting: target is a single value
        seq_output_series = df.iloc[i + input_window + forecast_horizon - 1]
        sequences.append((seq_input_df, seq_output_series))
    return sequences

def split_sequences(all_sequences: list, train_r: float, val_r: float) -> Tuple[list, list, list]:
    n = len(all_sequences)
    train_size = int(n * train_r)
    val_size = int(n * val_r)
    
    train_seq = all_sequences[:train_size]
    val_seq = all_sequences[train_size : train_size + val_size]
    test_seq = all_sequences[train_size + val_size :]
    return train_seq, val_seq, test_seq

def get_training_block_for_scaling_selection(
    df_full_features: pd.DataFrame, train_sequences: list, 
    forecast_horizon: int, target_col_name: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepares a continuous block of training data for fitting scalers and feature selection."""
    if not train_sequences:
        raise ValueError("Training sequences list cannot be empty.")
    
    earliest_date = train_sequences[0][0]['date'].iloc[0]
    latest_date = train_sequences[-1][0]['date'].iloc[-1]
    
    train_block_df = df_full_features[
        (df_full_features['date'] >= earliest_date) & (df_full_features['date'] <= latest_date)
    ].copy()
    
    train_block_df[f'{target_col_name}_target'] = train_block_df[target_col_name].shift(-forecast_horizon)
    train_block_df.dropna(subset=[f'{target_col_name}_target'], inplace=True) # Drop rows where target is NaN due to shift
    
    train_target_series = train_block_df[f'{target_col_name}_target']
    # Features exclude the target itself and date
    train_features_df = train_block_df.drop(columns=['date', f'{target_col_name}_target']) 
    # Also remove original target_col_name if it's different from target_col_name_target and present
    if target_col_name in train_features_df.columns and target_col_name != f'{target_col_name}_target':
        train_features_df = train_features_df.drop(columns=[target_col_name])
        
    return train_features_df, train_target_series

def fit_and_scale_training_block(
    train_features_df: pd.DataFrame, train_target_series: pd.Series, 
    feature_cols_to_scale: List[str]
) -> Tuple[MinMaxScaler, MinMaxScaler, pd.DataFrame, pd.Series]:
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_train_features_df = pd.DataFrame(
        feature_scaler.fit_transform(train_features_df[feature_cols_to_scale]),
        columns=feature_cols_to_scale, index=train_features_df.index
    )
    scaled_train_target_series = pd.Series(
        target_scaler.fit_transform(train_target_series.values.reshape(-1, 1)).flatten(),
        name=train_target_series.name + "_scaled", index=train_target_series.index
    )
    return feature_scaler, target_scaler, scaled_train_features_df, scaled_train_target_series

def perform_feature_selection_ivs(
    scaled_train_features_df: pd.DataFrame, scaled_train_target_series: pd.Series,
    method: str, threshold: float
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    logger = logging.getLogger(__name__)
    logger.debug(f"Performing feature selection with hydroIVS method: {method}, threshold: {threshold}")
    
    hydroIVS = None # Initialize to None
    try:
        hydroIVS = importr('hydroIVS')
        logger.info("hydroIVS package found and imported.")
    except Exception as e_import_hydroivs: # Catch generic Exception
        logger.info(f"Failed to import hydroIVS (Error: {e_import_hydroivs}). Assuming it's not installed. Attempting to install...")
        
        devtools = None # Initialize to None
        try:
            devtools = importr('devtools')
            logger.info("devtools package found and imported.")
        except Exception as e_import_devtools: # Catch generic Exception
            logger.info(f"devtools package not found (Error: {e_import_devtools}). Attempting to install devtools...")
            try:
                utils = importr('utils')
                utils.install_packages('devtools', repos='http://cran.us.r-project.org')
                devtools = importr('devtools') # Try importing again after install
                logger.info("devtools installed and imported successfully.")
            except Exception as e_install_devtools:
                logger.error(f"Failed to install devtools: {e_install_devtools}")
                raise # Re-raise; hydroIVS cannot be installed without devtools
        
        if devtools: # Proceed only if devtools was successfully imported or installed and imported
            try:
                logger.info("Installing hydroIVS from GitHub using devtools::install_github('johnswyou/hydroIVS')...")
                r('devtools::install_github("johnswyou/hydroIVS", upgrade="never")')
                hydroIVS = importr('hydroIVS') # Try importing again after install
                logger.info("hydroIVS installed and imported successfully.")
            except Exception as e_install_hydroivs:
                logger.error(f"Failed to install hydroIVS using devtools: {e_install_hydroivs}")
                raise # Re-raise as the process failed
        else:
            logger.error("devtools is not available. Cannot install hydroIVS.")
            raise Exception("devtools not available, hydroIVS installation failed.")

    if not hydroIVS: # If after all attempts, hydroIVS is still None
        logger.error("hydroIVS package could not be loaded or installed. Aborting feature selection.")
        raise ImportError("hydroIVS package is required but could not be loaded or installed.")

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_train_features = robjects.conversion.py2rpy(scaled_train_features_df)
        r_train_target = robjects.conversion.py2rpy(scaled_train_target_series)

    ivsIOData_func = hydroIVS.ivsIOData
    
    r_selected_indices, r_selected_names, r_selected_scores = ivsIOData_func(
        r_train_target, r_train_features, method, threshold
    )

    with localconverter(robjects.default_converter + pandas2ri.converter):
        selected_indices = np.array(robjects.conversion.rpy2py(r_selected_indices), dtype=int) - 1 # R to Python 0-based indexing
        selected_names = list(robjects.conversion.rpy2py(r_selected_names))
        selected_scores = np.array(robjects.conversion.rpy2py(r_selected_scores))
        
    logger.debug(f"Selected {len(selected_names)} features: {selected_names}")
    return selected_indices, selected_names, selected_scores

def convert_sequences_to_numpy(
    sequences: list, feature_cols: List[str], target_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for seq_input_df, seq_output_series in sequences:
        X_list.append(seq_input_df[feature_cols].values)
        y_list.append(seq_output_series[target_name])
    return np.array(X_list), np.array(y_list)

def scale_and_subset_data(
    X_full: np.ndarray, y_orig: np.ndarray,
    feature_scaler: MinMaxScaler, target_scaler: MinMaxScaler,
    selected_feature_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Reshape X for scaling (samples * timesteps, features), then reshape back
    num_samples, timesteps, num_all_features = X_full.shape
    X_scaled_full = feature_scaler.transform(X_full.reshape(-1, num_all_features)).reshape(num_samples, timesteps, num_all_features)
    
    y_scaled = target_scaler.transform(y_orig.reshape(-1, 1)).flatten()
    
    X_selected_scaled = X_scaled_full[:, :, selected_feature_indices]
    return X_selected_scaled, y_scaled, y_orig # y_orig is returned for evaluation (y_test_original)

def build_lstm_model(
    input_timesteps: int, num_selected_features: int,
    lstm_units: int, dropout_rate: float, learning_rate: float,
    custom_metrics: list
) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(input_timesteps, num_selected_features)))
    model.add(LSTM(lstm_units, activation=LSTM_ACTIVATION, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear')) # Single step direct forecasting
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse', 'mape', r2_keras] + custom_metrics
    )
    # model.summary(print_fn=logging.getLogger(__name__).info) # Needs a logger that accepts multi-line
    return model

def train_lstm_model(
    model: Sequential, X_train: np.ndarray, y_train_scaled: np.ndarray,
    X_val: np.ndarray, y_val_scaled: np.ndarray, epochs: int, batch_size: int,
    early_stop_monitor: str, early_stop_patience: int
) -> Dict:
    logger = logging.getLogger(__name__)
    logger.info("Training LSTM model...")
    early_stopping = EarlyStopping(
        monitor=early_stop_monitor, patience=early_stop_patience,
        restore_best_weights=True, mode='max' # Assuming higher is better for NSE/KGE
    )
    history = model.fit(
        X_train, y_train_scaled,
        epochs=epochs, batch_size=batch_size,
        validation_data=(X_val, y_val_scaled),
        callbacks=[early_stopping],
        verbose=2 # Or 1 for less output, or 0 for silent
    )
    return history.history

def evaluate_model_predictions(
    model: Sequential, X_test: np.ndarray, y_test_original: np.ndarray, target_scaler: MinMaxScaler
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    logger = logging.getLogger(__name__)
    y_pred_scaled = model.predict(X_test)
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled).flatten()
    
    # y_test_original is already 1D from scale_and_subset_data
    
    metrics = {
        "mape": mean_absolute_percentage_error(y_test_original, y_pred_original),
        "rmse": np.sqrt(mean_squared_error(y_test_original, y_pred_original)),
        "mae": mean_absolute_error(y_test_original, y_pred_original),
        "mase": mean_absolute_scaled_error(y_test_original, y_pred_original), # Requires y_train for naive forecast
        "r2": r2_score(y_test_original, y_pred_original),
        "nse": nash_sutcliffe_efficiency(y_test_original, y_pred_original),
        "kge": kling_gupta_efficiency(y_test_original, y_pred_original)
    }
    # MASE requires historical training data for naive scaling, simplified here or needs y_train passed.
    # For simplicity, if mase calculation is complex due to lack of y_train here, consider removing or adapting.
    # The original metrics.py MASE uses y_true[1:] - y_true[:-1] which is on test data, so it's fine.
    
    logger.info(f"Test Metrics: {metrics}")
    
    # Create DataFrame for predictions and actuals
    # Need dates for this. Dates should be extracted from test_seq.
    # This function currently doesn't have test_seq.
    # For now, return raw preds and trues. Plotting df to be assembled higher up.
    
    return y_pred_original, y_test_original, metrics


def save_artifacts(
    output_dir: Path, model_type_prefix: str, model: Sequential,
    feature_scaler: MinMaxScaler, target_scaler: MinMaxScaler,
    selected_features_info: Dict, history: Dict, test_metrics: Dict,
    timings: Dict
):
    logger = logging.getLogger(__name__)
    logger.info(f"Saving artifacts for {model_type_prefix}...")

    model.save(output_dir / f"{model_type_prefix}_model.keras")
    with open(output_dir / f"{model_type_prefix}_feature_scaler.pkl", 'wb') as f: pickle.dump(feature_scaler, f)
    with open(output_dir / f"{model_type_prefix}_target_scaler.pkl", 'wb') as f: pickle.dump(target_scaler, f)
    with open(output_dir / f"{model_type_prefix}_selected_features.pkl", 'wb') as f: pickle.dump(selected_features_info, f)
    with open(output_dir / f"{model_type_prefix}_history.pkl", 'wb') as f: pickle.dump(history, f)
    with open(output_dir / f"{model_type_prefix}_test_metrics.pkl", 'wb') as f: pickle.dump(test_metrics, f)
    with open(output_dir / f"{model_type_prefix}_timings.pkl", 'wb') as f: pickle.dump(timings, f)


# --- Main Workflow ---
def run_experiment_iteration(
    args: argparse.Namespace,
    catchment_id: str,
    forecast_horizon: int,
    filter_shortname: Optional[str], # None for baseline-only run if we adapt
    df_processed_common: pd.DataFrame, # Data after initial load & preprocess
    all_runs_summary_list: List[Dict]
):
    logger = logging.getLogger(__name__)
    current_iter_timings = {}
    
    run_output_dir = create_output_directory(args.base_save_path, catchment_id, forecast_horizon, filter_shortname if filter_shortname else "baseline_only")

    # --- MODWT Path (Wavelet Model) ---
    if filter_shortname: # This block is for the wavelet-based model
        logger.info(f"--- Processing Wavelet Model: Filter '{filter_shortname}' ---")
        model_prefix = "wavelet"
        
        # 1. MODWT Feature Engineering
        modwt_fe = MODWTFeatureEngineer(
            wavelet=filter_shortname,
            v_levels=[args.max_level],
            w_levels=list(range(1, args.max_level + 1))
        )
        start_time = time.perf_counter()
        df_wavelet_features = engineer_modwt_features(df_processed_common.copy(), modwt_fe, BASE_FEATURES_FOR_MODWT)
        current_iter_timings[f'{model_prefix}_modwt_duration'] = time.perf_counter() - start_time
        
        wavelet_feature_cols_all = [col for col in df_wavelet_features.columns if col not in ['date']]

        # 2. Create sequences
        sequences_wavelet = generate_sequences(df_wavelet_features, INPUT_WINDOW, forecast_horizon, TARGET_COLUMN_NAME)
        train_seq_w, val_seq_w, test_seq_w = split_sequences(sequences_wavelet, TRAIN_RATIO, VAL_RATIO)

        # 3. Prepare data for scaler fitting and feature selection
        train_block_feats_w, train_block_target_w = get_training_block_for_scaling_selection(
            df_wavelet_features, train_seq_w, forecast_horizon, TARGET_COLUMN_NAME
        )
        
        # These are the features on which scaler will be fit and IVS will run
        # Original script used all columns from df_modwt.drop(["Q_target", "date"], axis=1)
        # This includes 'Q' and all wavelet features.
        cols_for_wavelet_scaling_ivs = [col for col in train_block_feats_w.columns if col in wavelet_feature_cols_all]

        fscaler_w, tscaler_w, scaled_train_block_feats_w, scaled_train_block_target_w = fit_and_scale_training_block(
            train_block_feats_w, train_block_target_w, cols_for_wavelet_scaling_ivs
        )

        # 4. Feature Selection
        start_time = time.perf_counter()
        sel_indices_w, sel_names_w, sel_scores_w = perform_feature_selection_ivs(
            scaled_train_block_feats_w[cols_for_wavelet_scaling_ivs], scaled_train_block_target_w, IVS_METHOD, IVS_THRESHOLD
        )
        current_iter_timings[f'{model_prefix}_ivs_duration'] = time.perf_counter() - start_time
        selected_features_info_w = {"names": sel_names_w, "indices_original": sel_indices_w, "scores": sel_scores_w, 
                                    "feature_set_columns_for_ivs": cols_for_wavelet_scaling_ivs}

        # 5. Prepare (X,y) Numpy arrays and scale them
        X_train_w_full, y_train_w_orig = convert_sequences_to_numpy(train_seq_w, cols_for_wavelet_scaling_ivs, TARGET_COLUMN_NAME)
        X_val_w_full,   y_val_w_orig   = convert_sequences_to_numpy(val_seq_w,   cols_for_wavelet_scaling_ivs, TARGET_COLUMN_NAME)
        X_test_w_full,  y_test_w_orig  = convert_sequences_to_numpy(test_seq_w,  cols_for_wavelet_scaling_ivs, TARGET_COLUMN_NAME)
        
        X_train_w, y_train_w_scaled, _ = scale_and_subset_data(X_train_w_full, y_train_w_orig, fscaler_w, tscaler_w, sel_indices_w)
        X_val_w,   y_val_w_scaled,   _ = scale_and_subset_data(X_val_w_full,   y_val_w_orig,   fscaler_w, tscaler_w, sel_indices_w)
        X_test_w,  y_test_w_scaled, y_test_w_original = scale_and_subset_data(X_test_w_full,  y_test_w_orig,  fscaler_w, tscaler_w, sel_indices_w)

        # 6. Build and Train Model
        if X_train_w.shape[2] == 0: # No features selected
             logger.warning(f"No features selected for wavelet model with filter {filter_shortname}. Skipping training and evaluation.")
             wavelet_metrics = {metric: np.nan for metric in ["nse", "kge", "rmse", "mae", "mape", "mase", "r2"]}
             wavelet_model_path_str = "skipped_no_features"
             wavelet_history = {}
             y_pred_w_original = np.full_like(y_test_w_original, np.nan) if y_test_w_original is not None else []
        else:
            model_w = build_lstm_model(INPUT_WINDOW, X_train_w.shape[2], LSTM_UNITS, LSTM_DROPOUT, LEARNING_RATE, [NashSutcliffeEfficiencyMetric(name='nse'), KlingGuptaEfficiencyMetric(name='kge')])
            model_w.summary(print_fn=logger.info)
            start_time = time.perf_counter()
            wavelet_history = train_lstm_model(model_w, X_train_w, y_train_w_scaled, X_val_w, y_val_w_scaled, EPOCHS, BATCH_SIZE, EARLY_STOPPING_MONITOR, EARLY_STOPPING_PATIENCE)
            current_iter_timings[f'{model_prefix}_train_duration'] = time.perf_counter() - start_time
            
            # 7. Evaluate
            y_pred_w_original, _, wavelet_metrics = evaluate_model_predictions(model_w, X_test_w, y_test_w_original, tscaler_w)
            
            # 8. Save Wavelet Artifacts
            save_artifacts(run_output_dir, model_prefix, model_w, fscaler_w, tscaler_w, selected_features_info_w, wavelet_history, wavelet_metrics, current_iter_timings)
            wavelet_model_path_str = str(run_output_dir / f"{model_prefix}_model.keras")

    else: # Should not happen with current main loop structure, but for completeness
        wavelet_metrics = {metric: np.nan for metric in ["nse", "kge", "rmse", "mae", "mape", "mase", "r2"]}
        y_pred_w_original = [] # Placeholder
        y_test_w_original = [] # Placeholder
        wavelet_model_path_str = "N/A"
        selected_features_info_w = {} # Placeholder

    # --- Baseline Model Path ---
    # Baseline model runs for every wavelet filter iteration to use aligned data
    logger.info(f"--- Processing Baseline Model (related to filter '{filter_shortname if filter_shortname else 'N/A'}') ---")
    model_prefix_b = "baseline"
    
    # Data for baseline: use df_wavelet_features if filter_shortname is present to ensure temporal alignment after MODWT's dropna.
    # If filter_shortname is None (e.g. a standalone baseline run), use df_processed_common.
    df_for_baseline = df_wavelet_features if filter_shortname and 'df_wavelet_features' in locals() else df_processed_common.copy()

    # If 'timestamp' is not in df_for_baseline but is in BASELINE_MODEL_FEATURES, it needs to be added from 'date'
    if 'timestamp' not in df_for_baseline.columns and 'timestamp' in BASELINE_MODEL_FEATURES:
        df_for_baseline['timestamp'] = pd.to_datetime(df_for_baseline['date']).astype(int) / 10**9


    # 1. Create sequences (using the potentially NaN-trimmed df_for_baseline)
    sequences_baseline = generate_sequences(df_for_baseline, INPUT_WINDOW, forecast_horizon, TARGET_COLUMN_NAME)
    train_seq_b, val_seq_b, test_seq_b = split_sequences(sequences_baseline, TRAIN_RATIO, VAL_RATIO)

    # 2. Prepare data for scaler fitting and feature selection
    train_block_feats_b_full, train_block_target_b = get_training_block_for_scaling_selection(
        df_for_baseline, train_seq_b, forecast_horizon, TARGET_COLUMN_NAME
    )
    # Select only defined BASELINE_MODEL_FEATURES from the block
    cols_for_baseline_scaling_ivs = [col for col in BASELINE_MODEL_FEATURES if col in train_block_feats_b_full.columns]
    train_block_feats_b = train_block_feats_b_full[cols_for_baseline_scaling_ivs]


    fscaler_b, tscaler_b, scaled_train_block_feats_b, scaled_train_block_target_b = fit_and_scale_training_block(
        train_block_feats_b, train_block_target_b, cols_for_baseline_scaling_ivs # Pass only baseline cols here
    )

    # 3. Feature Selection for Baseline
    start_time = time.perf_counter()
    sel_indices_b, sel_names_b, sel_scores_b = perform_feature_selection_ivs(
        scaled_train_block_feats_b, scaled_train_block_target_b, IVS_METHOD, IVS_THRESHOLD
    )
    current_iter_timings[f'{model_prefix_b}_ivs_duration'] = time.perf_counter() - start_time
    selected_features_info_b = {"names": sel_names_b, "indices_original": sel_indices_b, "scores": sel_scores_b,
                                "feature_set_columns_for_ivs": cols_for_baseline_scaling_ivs}


    # 4. Prepare (X,y) Numpy arrays and scale
    X_train_b_full, y_train_b_orig = convert_sequences_to_numpy(train_seq_b, cols_for_baseline_scaling_ivs, TARGET_COLUMN_NAME)
    X_val_b_full,   y_val_b_orig   = convert_sequences_to_numpy(val_seq_b,   cols_for_baseline_scaling_ivs, TARGET_COLUMN_NAME)
    X_test_b_full,  y_test_b_orig  = convert_sequences_to_numpy(test_seq_b,  cols_for_baseline_scaling_ivs, TARGET_COLUMN_NAME)

    X_train_b, y_train_b_scaled, _ = scale_and_subset_data(X_train_b_full, y_train_b_orig, fscaler_b, tscaler_b, sel_indices_b)
    X_val_b,   y_val_b_scaled,   _ = scale_and_subset_data(X_val_b_full,   y_val_b_orig,   fscaler_b, tscaler_b, sel_indices_b)
    X_test_b,  y_test_b_scaled, y_test_b_original = scale_and_subset_data(X_test_b_full,  y_test_b_orig,  fscaler_b, tscaler_b, sel_indices_b)

    # 5. Build and Train Baseline Model
    if X_train_b.shape[2] == 0: # No features selected
        logger.warning(f"No features selected for baseline model with filter {filter_shortname}. Skipping training and evaluation.")
        baseline_metrics = {metric: np.nan for metric in ["nse", "kge", "rmse", "mae", "mape", "mase", "r2"]}
        baseline_model_path_str = "skipped_no_features"
        baseline_history = {}
        y_pred_b_original = np.full_like(y_test_b_original, np.nan) if y_test_b_original is not None else []
    else:
        model_b = build_lstm_model(INPUT_WINDOW, X_train_b.shape[2], LSTM_UNITS, LSTM_DROPOUT, LEARNING_RATE, [NashSutcliffeEfficiencyMetric(name='nse'), KlingGuptaEfficiencyMetric(name='kge')])
        model_b.summary(print_fn=logger.info)
        start_time = time.perf_counter()
        baseline_history = train_lstm_model(model_b, X_train_b, y_train_b_scaled, X_val_b, y_val_b_scaled, EPOCHS, BATCH_SIZE, EARLY_STOPPING_MONITOR, EARLY_STOPPING_PATIENCE)
        current_iter_timings[f'{model_prefix_b}_train_duration'] = time.perf_counter() - start_time
        
        # 6. Evaluate Baseline
        y_pred_b_original, _, baseline_metrics = evaluate_model_predictions(model_b, X_test_b, y_test_b_original, tscaler_b)
        
        # 7. Save Baseline Artifacts
        save_artifacts(run_output_dir, model_prefix_b, model_b, fscaler_b, tscaler_b, selected_features_info_b, baseline_history, baseline_metrics, current_iter_timings) # Pass relevant part of timings
        baseline_model_path_str = str(run_output_dir / f"{model_prefix_b}_model.keras")


    # --- Consolidate Predictions and Save ---
    # Extract test dates from test_seq_b (should be same as test_seq_w if aligned)
    if test_seq_b: # If sequences were generated
        # Calculate the date for each prediction in the test set
        # The date of a prediction corresponds to 'date' of the last day of its input window + forecast_horizon
        pred_dates = [pd.Timestamp(seq[0]['date'].iloc[-1]) + pd.Timedelta(days=forecast_horizon) for seq in test_seq_b]
        
        pred_label_df = pd.DataFrame({
            'date': pred_dates,
            'y_true': y_test_b_original if y_test_b_original is not None and len(y_test_b_original)>0 else np.nan, # True values are same for both
            'wavelet_y_pred': y_pred_w_original if filter_shortname and y_pred_w_original is not None and len(y_pred_w_original)>0 else np.nan,
            'baseline_y_pred': y_pred_b_original if y_pred_b_original is not None and len(y_pred_b_original)>0 else np.nan,
        })
        pred_label_df.to_pickle(run_output_dir / "predictions_and_true_values.pkl")
        pred_label_df.to_csv(run_output_dir / "predictions_and_true_values.csv", index=False)


    # --- Append to Summary ---
    wavelet_summary_metrics = {f"wavelet_{k}": v for k,v in wavelet_metrics.items()} if filter_shortname and wavelet_metrics else \
                              {f"wavelet_{k}": np.nan for k in ["nse", "kge", "rmse", "mae", "mape", "mase", "r2"]}
    baseline_summary_metrics = {f"baseline_{k}": v for k,v in baseline_metrics.items()} if baseline_metrics else \
                               {f"baseline_{k}": np.nan for k in ["nse", "kge", "rmse", "mae", "mape", "mase", "r2"]}

    summary_entry = {
        "catchment_id": catchment_id,
        "forecast_horizon": forecast_horizon,
        "filter_name": filter_shortname if filter_shortname else "N/A",
        "wavelet_model_path": wavelet_model_path_str if filter_shortname else "N/A",
        "baseline_model_path": baseline_model_path_str,
        **wavelet_summary_metrics,
        **baseline_summary_metrics,
        **current_iter_timings
    }
    all_runs_summary_list.append(summary_entry)


def main():
    args = parse_arguments()
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting main workflow (Version 1)...")

    # Ensure rpy2 can find R packages if not in default libPath (might need R_HOME or other setup)
    # For now, assume 'hydroIVS' is installable/findable by Rpy2's R instance.

    input_csv_full_path = args.base_csv_path / args.csv_filename
    if not input_csv_full_path.exists():
        logger.error(f"Input CSV file does not exist: {input_csv_full_path}")
        sys.exit(1)

    catchment_id = args.csv_filename.stem.split("_")[0]
    logger.info(f"Processing catchment: {catchment_id}")
    
    # Create base save directory for the catchment
    catchment_base_save_dir = args.base_save_path / catchment_id
    catchment_base_save_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    df_initial = load_and_preprocess_data(input_csv_full_path, BASE_FEATURES_FOR_MODWT)

    all_runs_summary = []

    for horizon in FORECAST_HORIZONS:
        logger.info(f"===== Processing Forecast Horizon: {horizon} days =====")
        for f_filter in POSSIBLE_FILTERS:
            logger.info(f"--- Processing Filter: {f_filter} for Horizon: {horizon} ---")
            try:
                run_experiment_iteration(
                    args, catchment_id, horizon, f_filter,
                    df_initial.copy(), # Pass a copy to avoid modification issues
                    all_runs_summary
                )
            except Exception as e:
                logger.error(f"ERROR during iteration: Horizon {horizon}, Filter {f_filter}. Error: {e}", exc_info=True)
                # Add error entry to summary
                error_wavelet_metrics = {f"wavelet_{m}": np.nan for m in ["nse", "kge", "rmse", "mae", "mape", "mase", "r2"]}
                error_baseline_metrics = {f"baseline_{m}": np.nan for m in ["nse", "kge", "rmse", "mae", "mape", "mase", "r2"]}
                error_entry = {
                    "catchment_id": catchment_id, "forecast_horizon": horizon, "filter_name": f_filter,
                    "status": "error", "error_message": str(e),
                    "wavelet_model_path": "error", 
                    "baseline_model_path": "error",
                    **error_wavelet_metrics,
                    **error_baseline_metrics
                }
                all_runs_summary.append(error_entry)


    # Save summary of all runs
    summary_df = pd.DataFrame(all_runs_summary)
    summary_file_path = catchment_base_save_dir / "experiment_summary_results.csv"
    summary_df.to_csv(summary_file_path, index=False)
    logger.info(f"Experiment summary saved to: {summary_file_path}")
    logger.info("Script completed successfully.")
    return 0

if __name__ == "__main__":
    # Initialize R environment for rpy2
    pandas2ri.activate() # Enables automatic conversion between pandas and R objects
    
    # Set R library paths if necessary - this is often a point of failure for rpy2
    # You might need to configure .Renviron or set R_LIBS_USER environment variable
    # For example, you could try to add the user's default R library path if known
    # if "R_LIBS_USER" not in os.environ:
    #     user_lib_path = robjects.r('Sys.getenv("R_LIBS_USER")')[0]
    #     if user_lib_path and os.path.isdir(user_lib_path):
    #          robjects.r(f'.libPaths(new = "{user_lib_path}")')
    #          print(f"Added {user_lib_path} to R .libPaths()")
    #     else:
    #         print("R_LIBS_USER not found or not a directory. Using default .libPaths().")
    # print("Current R .libPaths():", robjects.r('.libPaths()'))


    exit_code = main()
    sys.exit(exit_code) 