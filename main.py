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
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter # For rpy2 3.1.x
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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
            # Uncomment the next line to log to a file
            # logging.FileHandler('script.log'),
        ]
    )

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Trains and saves wavelet and baseline LSTM forecasting models and associated metadata for a given catchment in CAMELS."
    )
    parser.add_argument(
        '--csv_filename',
        type=Path,
        required=True,
        help='Path to the CSV file for the catchment.'
    )
    parser.add_argument(
        '--max_level',
        type=int,
        required=False,
        default=6,
        help='Maximum decomposition level to use for MODWT.'
    )
    parser.add_argument(
        '--base_save_path',
        type=Path,
        required=False,
        default=Path("/home/jswyou/scratch"),
        help='Path to the base directory where all output data will be saved.'
    )
    parser.add_argument(
        '--base_csv_path',
        type=Path,
        required=False,
        default=Path("/home/jswyou/projects/def-quiltyjo/jswyou/wavelet-lstm-camels/data"),
        help='Path to the base directory where all CAMELS data is stored as CSV files.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )
    return parser.parse_args()

# def baseline(args: argparse.Namespace) -> int:
#     """
#     Baseline function of the script.

#     Args:
#         args (argparse.Namespace): Parsed command-line arguments.

#     Returns:
#         int: Exit status code.
#     """

def r2_keras(y_true, y_pred):
    """
    Calculates the coefficient of determination (R²) for Keras models.

    Args:
        y_true (tensor): Ground truth (correct) target values.
        y_pred (tensor): Estimated target values.

    Returns:
        tensor: R² metric.
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - SS_res/(SS_tot + K.epsilon())
    return r2

class NashSutcliffeEfficiency(tf.keras.metrics.Metric):
    def __init__(self, name='nse', **kwargs):
        super(NashSutcliffeEfficiency, self).__init__(name=name, **kwargs)
        # Initialize state variables
        self.sum_squared_errors = self.add_weight(name='sum_squared_errors', initializer='zeros')
        self.sum_squared_total = self.add_weight(name='sum_squared_total', initializer='zeros')
        self.mean_observed = self.add_weight(name='mean_observed', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.sum_observed = self.add_weight(name='sum_observed', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten the tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Update sum of observed values and count for mean calculation
        self.sum_observed.assign_add(tf.reduce_sum(y_true))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
        
        # Update sum of squared errors
        squared_errors = tf.square(y_true - y_pred)
        self.sum_squared_errors.assign_add(tf.reduce_sum(squared_errors))
        
        # Update sum of squared total (variance)
        # To compute sum((y_true - mean(y_true))^2), we need to compute the mean incrementally
        # However, for simplicity, we'll compute the sum of squares of y_true and the sum of y_true,
        # then later compute sum_squared_total = sum(y_true^2) - (sum(y_true)^2) / count
        sum_y_true_sq = tf.reduce_sum(tf.square(y_true))
        self.sum_squared_total.assign_add(sum_y_true_sq)
        
    def result(self):
        # Compute mean_observed
        mean_observed = self.sum_observed / self.count
        
        # Compute sum_squared_total = sum(y_true^2) - (sum(y_true)^2) / N
        sum_squared_total = self.sum_squared_total - (tf.square(self.sum_observed) / self.count)
        
        # Compute NSE
        nse = 1 - (self.sum_squared_errors / (sum_squared_total + K.epsilon()))
        return nse
    
    def reset_state(self):
        # Reset all state variables
        self.sum_squared_errors.assign(0.0)
        self.sum_squared_total.assign(0.0)
        self.sum_observed.assign(0.0)
        self.count.assign(0.0)

class KlingGuptaEfficiency(tf.keras.metrics.Metric):
    def __init__(self, name='kge', **kwargs):
        super(KlingGuptaEfficiency, self).__init__(name=name, **kwargs)
        # Initialize state variables
        self.sum_true = self.add_weight(name='sum_true', initializer='zeros')
        self.sum_pred = self.add_weight(name='sum_pred', initializer='zeros')
        self.sum_true_sq = self.add_weight(name='sum_true_sq', initializer='zeros')
        self.sum_pred_sq = self.add_weight(name='sum_pred_sq', initializer='zeros')
        self.sum_true_pred = self.add_weight(name='sum_true_pred', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten the tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Update sums
        self.sum_true.assign_add(tf.reduce_sum(y_true))
        self.sum_pred.assign_add(tf.reduce_sum(y_pred))
        self.sum_true_sq.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.sum_pred_sq.assign_add(tf.reduce_sum(tf.square(y_pred)))
        self.sum_true_pred.assign_add(tf.reduce_sum(y_true * y_pred))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
        
    def result(self):
        # Compute mean values
        mean_true = self.sum_true / self.count
        mean_pred = self.sum_pred / self.count
        
        # Compute standard deviations
        var_true = (self.sum_true_sq / self.count) - tf.square(mean_true)
        var_pred = (self.sum_pred_sq / self.count) - tf.square(mean_pred)
        
        std_true = tf.sqrt(var_true + K.epsilon())
        std_pred = tf.sqrt(var_pred + K.epsilon())
        
        # Compute covariance
        cov = (self.sum_true_pred / self.count) - (mean_true * mean_pred)
        
        # Compute correlation coefficient
        r = cov / (std_true * std_pred + K.epsilon())
        
        # Compute alpha and beta
        alpha = std_pred / (std_true + K.epsilon())
        beta = mean_pred / (mean_true + K.epsilon())
        
        # Compute KGE
        kge = 1 - tf.sqrt(tf.square(r - 1) + tf.square(alpha - 1) + tf.square(beta - 1))
        return kge
    
    def reset_state(self):
        # Reset all state variables
        self.sum_true.assign(0.0)
        self.sum_pred.assign(0.0)
        self.sum_true_sq.assign(0.0)
        self.sum_pred_sq.assign(0.0)
        self.sum_true_pred.assign(0.0)
        self.count.assign(0.0)

def main(args: argparse.Namespace) -> int:
    """
    Main function of the script.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        int: Exit status code.
    """
    logger = logging.getLogger(__name__)
    logger.info("Script started.")
    
    # Check if input CSV file exists
    if not (args.base_csv_path / args.csv_filename).exists():
        logger.error(f"Input CSV file does not exist: {args.base_csv_path / args.csv_filename}")
        return 1

    # Ensure catchment_id directory exists in args.base_save_path
    catchment_id = args.csv_filename.stem.split("_")[0]
    logger.debug(f"Create {catchment_id} subdirectory under {args.base_save_path}")
    os.makedirs(args.base_save_path / catchment_id, exist_ok=True)

    possible_filters = ["bl7", "coif1", "coif2", "db1", "db2", "db3", "db4", "db5", 
    "db6", "db7", "fk4", "fk6", "fk8", "fk14", "han2_3", "han3_3",
    "han4_5", "han5_5", "mb4_2", "mb8_2", "mb8_3", "mb8_4", "mb10_3", "mb12_3",
    "mb14_3", "sym4", "sym5", "sym6", "sym7", "la8", 
    "la10", "la12", "la14"]

    for forecast_horizon in [1, 3, 5]:
        os.makedirs(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}", exist_ok=True)
        for filter_shortname in possible_filters:
            # For the present catchment_id directory, ensure the filter_shortname subdir is present
            os.makedirs(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname, exist_ok=True)

            # -----------------------
            # Step 1: Load the dataset
            # -----------------------
            logger.debug(f"Read CAMELS csv file: {args.csv_filename}")
            df = pd.read_csv(args.base_csv_path / args.csv_filename, parse_dates=['date'])

            # ------------------------------
            # Step 2: Data Preprocessing
            # ------------------------------

            # Handle missing values
            df.ffill(inplace=True)

            # Feature Selection and Engineering
            features = ['Q', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 
                        'tmax(C)', 'tmin(C)', 'vp(Pa)']

            # Optionally, drop the 'flag' column if it's not useful or handle it appropriately
            df = df[['date'] + features]

            # Sort the data by date to ensure chronological order
            df.sort_values('date', inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Convert the date to a Unix timestamp (seconds since January 1, 1970)
            df['timestamp'] = pd.to_datetime(df['date']).astype(int) / 10**9

            # ------------------------------
            # Step 2.5: MODWT features
            # ------------------------------

            # Try to import 'remotes'; install if not available
            utils = importr('utils')

            try:
                remotes = importr('remotes')
            except:
                print("Installing 'remotes' package from CRAN...")
                utils.install_packages('remotes')
                remotes = importr('remotes')

            # Install hydroIVS
            # github_repo = 'johnswyou/hydroIVS'
            # remotes.install_github(github_repo)

            start_section_modwt = time.perf_counter()

            # Add MODWT features
            modwt_fe = MODWTFeatureEngineer(wavelet=filter_shortname, v_levels=[args.max_level], w_levels=list(range(1, args.max_level+1)))
            df = modwt_fe.transform(df, features)

            end_section_modwt = time.perf_counter()
            timings['modwt'] = end_section_modwt - start_section_modwt

            # At this point, df has NaN values in the first args.max_level rows due to removing bouncary coefficients.
            # df still has the non-numeric `date` column.

            # Remove rows with NaN due to removing boundary coefficients
            df.dropna(inplace=True)

            feature_columns = list(df.columns)
            feature_columns.remove("date")

            # ------------------------------
            # Step 3: Splitting the Dataset
            # ------------------------------

            input_window = 270    # t days of historical data

            total_samples = len(df) - input_window - forecast_horizon + 1
            sequences = []

            for i in range(total_samples):
                seq_input = df.iloc[i : i + input_window]
                # seq_output = df.iloc[i + input_window : i + input_window + forecast_horizon] # multi-output
                seq_output = df.iloc[i + input_window + forecast_horizon - 1] # direct (single forecast horizon)
                sequences.append((seq_input, seq_output))

            # Determine split sizes
            train_size = int(0.7 * len(sequences))
            val_size = int(0.15 * len(sequences))
            test_size = len(sequences) - train_size - val_size

            # Split the sequences
            train_seq = sequences[:train_size]
            val_seq = sequences[train_size : train_size + val_size]
            test_seq = sequences[train_size + val_size :]

            # ----------------------------
            # Step 4: Scale Features
            # ----------------------------

            # # THE PROCESS BELOW IS WRONG

            # # Initialize scaler for input features
            # scaler = MinMaxScaler()
            # train_features = pd.concat([seq[0][feature_columns] for seq in train_seq], axis=0)
            # scaler.fit(train_features)

            # # Initialize and fit scaler for target variable 'Q'
            # q_scaler = MinMaxScaler()
            # train_q = pd.concat([seq[0]['Q'] for seq in train_seq], axis=0).values.reshape(-1, 1) # 2D array (column vector)
            # q_scaler.fit(train_q)

            # # THE PROCESS ABOVE IS WRONG

            # Function to scale input sequences
            def scale_sequences(sequences, x_scaler, y_scaler, feature_cols):
                X = []
                y = []
                for seq_input, seq_output in sequences:
                    scaled_input = x_scaler.transform(seq_input[feature_cols])
                    # Note: scaled_q is a 1D np.array (in both casese below, in the second case the np.array is 1D and has a single value in it)
                    try:
                        scaled_q = y_scaler.transform(seq_output['Q'].values.reshape(-1, 1)).flatten() # if seq_output['Q'] is a pd.Series
                    except:
                        scaled_q = y_scaler.transform(np.array([[seq_output["Q"]]])).flatten() # if seq_output['Q'] is an integer
                    X.append(scaled_input)
                    y.append(scaled_q)
                return np.array(X), np.array(y)

            # # Scale the input features
            # X_train, y_train_scaled = scale_sequences(train_seq, scaler, q_scaler, feature_columns)
            # X_val, y_val_scaled = scale_sequences(val_seq, scaler, q_scaler, feature_columns)
            # X_test, y_test_scaled = scale_sequences(test_seq, scaler, q_scaler, feature_columns)

            # # Save the input scaler
            # with open(args.base_save_path + catchment_id + '/feature_scaler.pkl', 'wb') as f:
            #     pickle.dump(scaler, f)

            # # Save the target scaler
            # with open(args.base_save_path + catchment_id + '/q_scaler.pkl', 'wb') as f:
            #     pickle.dump(q_scaler, f)

            # ----------------------------
            # Step 4 (updated): Feature Selection
            # ----------------------------

            # -----------
            # Fit scaler
            # -----------

            earliest_training_date = train_seq[0][0]['date'].iloc[0]
            last_training_date = train_seq[len(train_seq)-1][0]['date'].iloc[-1]

            # We will use a copy of df, and only the training portion, to select features
            train_df = df.copy()

            # Add target
            train_df['Q_target'] = train_df['Q'].shift(-forecast_horizon)
            # train_df.dropna(inplace=True) # we don't really need this

            # Subset train_df between earliest training date and last training date
            train_df = train_df[(train_df['date'] >= earliest_training_date) & (train_df['date'] <= last_training_date)]

            # Separate features, Q, and Q_target for scaling
            # train_df_q = train_df["Q"]
            train_df_q_target = train_df["Q_target"]
            # train_df_features = train_df.drop(["Q", "Q_target", "date"], axis=1)
            train_df_features = train_df.drop(["Q_target", "date"], axis=1)

            # Initialize and fit scalers
            scaler = MinMaxScaler()
            q_scaler = MinMaxScaler()

            # Scale features, Q, and Q_target
            train_df_features = pd.DataFrame(scaler.fit_transform(train_df_features), columns=train_df_features.columns)
            # train_df_q = q_scaler.fit_transform(train_df_q.values.reshape(-1, 1)).flatten() # 1D numpy array
            # train_df_q_target = q_scaler.transform(train_df_q_target.values.reshape(-1, 1)).flatten() # 1D numpy array
            train_df_q_target = q_scaler.fit_transform(train_df_q_target.values.reshape(-1, 1)).flatten() # 1D numpy array

            # Concatenate train_df_features and train_df_q
            # train_df_features['Q'] = train_df_q

            # Needs to be a pandas series
            train_df_q_target = pd.Series(train_df_q_target, name = "Q_target")

            # *********
            # BASELINE
            # *********

            # Initialize and fit baseline scalers
            baseline_scaler = MinMaxScaler()
            baseline_q_scaler = MinMaxScaler()

            # Get baseline features and target
            baseline_train_df_q_target = train_df["Q_target"]
            baseline_train_df_features = train_df.drop(["Q_target", "date"], axis=1)
            baseline_feature_names = [
                "Q",
                "timestamp",
                "dayl(s)",
                "prcp(mm/day)",
                "srad(W/m2)",
                "swe(mm)",
                "tmax(C)",
                "tmin(C)",
                "vp(Pa)"
            ]
            baseline_train_df_features = baseline_train_df_features.loc[:, baseline_feature_names]
            baseline_train_df_features = pd.DataFrame(baseline_scaler.fit_transform(baseline_train_df_features), columns=baseline_feature_names)
            baseline_train_df_q_target = baseline_q_scaler.fit_transform(baseline_train_df_q_target.values.reshape(-1, 1)).flatten() # 1D numpy array
            baseline_train_df_q_target = pd.Series(baseline_train_df_q_target, name = "Q_target")

            # --------------------------
            # Perform feature selection
            # --------------------------

            # with (robjects.default_converter + pandas2ri.converter).context(): # For rpy2 3.5.x
            with localconverter(robjects.default_converter + pandas2ri.converter): # For rpy2 3.1.x
                train_df_features_r = robjects.conversion.py2rpy(train_df_features)
                train_df_q_target_r = robjects.conversion.py2rpy(train_df_q_target)
                
                # BASELINE
                baseline_train_df_features_r = robjects.conversion.py2rpy(baseline_train_df_features)
                baseline_train_df_q_target_r = robjects.conversion.py2rpy(baseline_train_df_q_target)

            # robjects.globalenv['train_df_features'] =  train_df_features_r
            # robjects.globalenv['train_df_q_target'] =  train_df_q_target_r

            try:
                hydroIVS = importr('hydroIVS')
            except:
                print("Installing 'hydroIVS' package from CRAN...")
                github_repo = 'johnswyou/hydroIVS'
                remotes.install_github(github_repo)
                hydroIVS = importr('hydroIVS')

            ivsIOData = hydroIVS.ivsIOData
            start_section_ea = time.perf_counter()
            selected_feature_indices, selected_feature_names, selected_feature_scores = ivsIOData(train_df_q_target_r, train_df_features_r, "ea_cmi_tol", 0.05)
            end_section_ea = time.perf_counter()
            timings['ea_cmi_tol'] = end_section_ea - start_section_ea

            # BASELINE
            baseline_selected_feature_indices, baseline_selected_feature_names, baseline_selected_feature_scores = ivsIOData(baseline_train_df_q_target_r, baseline_train_df_features_r, "ea_cmi_tol", 0.05)
            
            # Convert r outputs back to python
            # with (robjects.default_converter + pandas2ri.converter).context(): # For rpy2 3.5.x
            with localconverter(robjects.default_converter + pandas2ri.converter): # For rpy2 3.1.x
                selected_feature_indices = robjects.conversion.rpy2py(selected_feature_indices) # np.array
                selected_feature_names = robjects.conversion.rpy2py(selected_feature_names)
                selected_feature_scores = robjects.conversion.rpy2py(selected_feature_scores) # np.array

                # BASELINE
                baseline_selected_feature_indices = robjects.conversion.rpy2py(baseline_selected_feature_indices) # np.array
                baseline_selected_feature_names = robjects.conversion.rpy2py(baseline_selected_feature_names)
                baseline_selected_feature_scores = robjects.conversion.rpy2py(baseline_selected_feature_scores) # np.array

            selected_feature_names = list(selected_feature_names) # selected_feature_names has trouble converting back to python so we help it
            selected_feature_indices = selected_feature_indices - 1 # account for the 0 based indexing in python

            # BASELINE
            baseline_selected_feature_names = list(baseline_selected_feature_names)
            baseline_selected_feature_indices = baseline_selected_feature_indices - 1

            # ----------------------------
            # Step 4.5: Scale Features
            # ----------------------------

            X_train, y_train_scaled = scale_sequences(train_seq, scaler, q_scaler, train_df_features.columns)
            X_val, y_val_scaled = scale_sequences(val_seq, scaler, q_scaler, train_df_features.columns)
            X_test, y_test_scaled = scale_sequences(test_seq, scaler, q_scaler, train_df_features.columns)

            # BASELINE
            baseline_X_train, baseline_y_train_scaled = scale_sequences(train_seq, baseline_scaler, baseline_q_scaler, baseline_feature_names)
            baseline_X_val, baseline_y_val_scaled = scale_sequences(val_seq, baseline_scaler, baseline_q_scaler, baseline_feature_names)
            baseline_X_test, baseline_y_test_scaled = scale_sequences(test_seq, baseline_scaler, baseline_q_scaler, baseline_feature_names)

            # Subset the columns of X_train, X_val and X_test acccording to selected_feature_indices
            X_train = X_train[:, :, selected_feature_indices]
            X_val = X_val[:, :, selected_feature_indices]
            X_test = X_test[:, :, selected_feature_indices]

            # BASELINE
            baseline_X_train = baseline_X_train[:, :, baseline_selected_feature_indices]
            baseline_X_val = baseline_X_val[:, :, baseline_selected_feature_indices]
            baseline_X_test = baseline_X_test[:, :, baseline_selected_feature_indices]

            # Save the input scaler
            with open(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "feature_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)

            # BASELINE
            with open(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "baseline_feature_scaler.pkl", 'wb') as f:
                pickle.dump(baseline_scaler, f)

            # Save the target scaler
            with open(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "q_scaler.pkl", 'wb') as f:
                pickle.dump(q_scaler, f)

            # BASELINE
            with open(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "baseline_q_scaler.pkl", 'wb') as f:
                pickle.dump(baseline_q_scaler, f)

            # Save the selected feature names, indices and scores
            # selected_features_dict = {"selected_feature_names": selected_feature_names,
            #                           "selected_feature_indices": selected_feature_indices,
            #                           "selected_feature_scores": selected_feature_scores}

            selected_features_dict = {
                "selected_feature_names": selected_feature_names,
                "selected_feature_indices": selected_feature_indices,
                "selected_feature_scores": selected_feature_scores,
                "baseline_selected_feature_names": baseline_selected_feature_names,
                "baseline_selected_feature_indices": baseline_selected_feature_indices,
                "baseline_selected_feature_scores": baseline_selected_feature_scores
            }

            with open(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "ea_cmi_tol_005_selected_feature_names.pkl", 'wb') as f:
                pickle.dump(selected_features_dict, f)

            # -------------------------
            # Step 5: Building the Model
            # -------------------------

            timesteps = input_window
            num_features = X_train.shape[2]
            # n_ahead = forecast_horizon

            model = Sequential()
            model.add(Input(shape=(timesteps, num_features)))
            # model.add(LSTM(64, activation='relu', return_sequences=True))
            # model.add(Dropout(0.2))
            # model.add(LSTM(32, activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(64, activation='relu'))
            # model.add(Dense(n_ahead))  # Output layer for n-day forecast
            model.add(LSTM(256, activation='tanh', return_sequences=False))
            model.add(Dropout(0.4))
            # model.add(Dense(n_ahead, activation='linear'))
            model.add(Dense(1, activation='linear'))

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=[
                    'mae',
                    'mse',
                    'mape',
                    r2_keras,
                    NashSutcliffeEfficiency(),
                    KlingGuptaEfficiency()
                ]
            )

            model.summary()

            # -------------------------
            # Step 5.5: Baseline Model
            # -------------------------

            baseline_num_features = baseline_X_train.shape[2]

            baseline_model = Sequential()
            baseline_model.add(Input(shape=(timesteps, baseline_num_features)))
            baseline_model.add(LSTM(256, activation='tanh', return_sequences=False))
            baseline_model.add(Dropout(0.4))
            # baseline_model.add(Dense(n_ahead, activation='linear'))
            baseline_model.add(Dense(1, activation='linear'))

            baseline_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=[
                    'mae',
                    'mse',
                    'mape',
                    r2_keras,
                    NashSutcliffeEfficiency(),
                    KlingGuptaEfficiency()
                ]
            )

            print()
            print("Baseline model structure:")
            print()

            baseline_model.summary()

            # ----------------------------
            # Step 6: Training the Model
            # ----------------------------

            early_stop = EarlyStopping(monitor='val_nse', patience=10, restore_best_weights=True, mode='max')

            start_section_lstm = time.perf_counter()
            history = model.fit(
                X_train, y_train_scaled,
                epochs=1000,
                batch_size=32,
                validation_data=(X_val, y_val_scaled),
                callbacks=[early_stop],
                verbose=2
            )
            end_section_lstm = time.perf_counter()
            timings['lstm'] = end_section_lstm - start_section_lstm

            # BASELINE
            baseline_early_stop = EarlyStopping(monitor='val_nse', patience=10, restore_best_weights=True, mode='max')
            baseline_history = baseline_model.fit(
                baseline_X_train, baseline_y_train_scaled,
                epochs=1000,
                batch_size=32,
                validation_data=(baseline_X_val, baseline_y_val_scaled),
                callbacks=[baseline_early_stop],
                verbose=2
            )

            with open(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "timings.pkl", 'wb') as f:
                pickle.dump(timings, f)

            with open(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "history.pkl", 'wb') as f:
                pickle.dump(history.history, f)

            # BASELINE
            with open(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "baseline_history.pkl", 'wb') as f:
                pickle.dump(baseline_history.history, f)

            # ----------------------------
            # Step 7: Plotting Loss Curves
            # ----------------------------

            # plt.figure(figsize=(12,6))
            # plt.plot(history.history['loss'], label='Training Loss')
            # plt.plot(history.history['val_loss'], label='Validation Loss')
            # plt.title('LSTM Model Loss')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss (MSE)')
            # plt.legend()
            # plt.show()

            # --------------------------
            # Step 8: Evaluating the Model
            # --------------------------

            # Predict on test data
            y_pred_scaled = model.predict(X_test)

            # Inverse transform predictions and true values
            y_pred = q_scaler.inverse_transform(y_pred_scaled)
            y_true = q_scaler.inverse_transform(y_test_scaled)

            # Calculate evaluation metrics
            mape = mean_absolute_percentage_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mase = mean_absolute_scaled_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            nse = nash_sutcliffe_efficiency(y_true, y_pred)
            kge = kling_gupta_efficiency(y_true, y_pred)

            print(f'Test NSE: {nse:.2f}')
            print(f'Test KGE: {kge:.2f}')
            print(f'Test RMSE: {rmse:.2f}')
            print(f'Test MAE: {mae:.2f}')
            print(f'Test MAPE: {mape:.2f}')
            print(f'Test MASE: {mase:.2f}')
            print(f'Test R²: {r2:.2f}')

            test_metrics_dict = {"nse": nse,
                                 "kge": kge,
                                 "rmse": rmse,
                                 "mae": mae,
                                 "mape": mape,
                                 "mase": mase,
                                 "r2": r2}
            
            with open(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "test_metrics_dict.pkl", "wb") as f:
                pickle.dump(test_metrics_dict, f)

            # Create a prediction and label data frame
            delta = pd.Timedelta(days=forecast_horizon)
            earliest_test_date = pd.Timestamp(test_seq[0][0]['date'].iloc[-1]) + delta
            last_test_date = pd.Timestamp(test_seq[len(test_seq)-1][0]['date'].iloc[-1]) + delta
            date_range = pd.date_range(start=earliest_test_date, end=last_test_date, freq='D')
            pred_label_df = pd.DataFrame({'date': date_range})

            pred_label_df["y_pred"] = y_pred
            pred_label_df["y_true"] = y_true

            # Save pred_label_df
            pred_label_df.to_pickle(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "pred_label_df.pkl")

            # --------------------------
            # Step 8.5: Evaluating the Baseline Model
            # --------------------------

            # Predict on test data
            baseline_y_pred_scaled = baseline_model.predict(baseline_X_test)

            # Inverse transform predictions and true values
            baseline_y_pred = baseline_q_scaler.inverse_transform(baseline_y_pred_scaled)
            baseline_y_true = baseline_q_scaler.inverse_transform(baseline_y_test_scaled)

            # Calculate evaluation metrics
            baseline_mape = mean_absolute_percentage_error(baseline_y_true, baseline_y_pred)
            baseline_rmse = np.sqrt(mean_squared_error(baseline_y_true, baseline_y_pred))
            baseline_mae = mean_absolute_error(baseline_y_true, baseline_y_pred)
            baseline_mase = mean_absolute_scaled_error(baseline_y_true, baseline_y_pred)
            baseline_r2 = r2_score(baseline_y_true, baseline_y_pred)
            baseline_nse = nash_sutcliffe_efficiency(baseline_y_true, baseline_y_pred)
            baseline_kge = kling_gupta_efficiency(baseline_y_true, baseline_y_pred)

            print(f'Baseline Test NSE: {baseline_nse:.2f}')
            print(f'Baseline Test KGE: {baseline_kge:.2f}')
            print(f'Baseline Test RMSE: {baseline_rmse:.2f}')
            print(f'Baseline Test MAE: {baseline_mae:.2f}')
            print(f'Baseline Test MAPE: {baseline_mape:.2f}')
            print(f'Baseline Test MASE: {baseline_mase:.2f}')
            print(f'Baseline Test R²: {baseline_r2:.2f}')

            baseline_test_metrics_dict = {
                "nse": baseline_nse,
                "kge": baseline_kge,
                "rmse": baseline_rmse,
                "mae": baseline_mae,
                "mape": baseline_mape,
                "mase": baseline_mase,
                "r2": baseline_r2
            }
            
            with open(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "baseline_test_metrics_dict.pkl", "wb") as f:
                pickle.dump(baseline_test_metrics_dict, f)

            # Create a prediction and label data frame

            pred_label_df["baseline_y_pred"] = baseline_y_pred
            pred_label_df["baseline_y_true"] = baseline_y_true

            # Save pred_label_df
            pred_label_df.to_pickle(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "baseline_pred_label_df.pkl")

            # --------------------------------
            # Step 9: Visualizing Predictions
            # --------------------------------

            # # Select a sample from the test set
            # sample_idx = 0  # Change as needed
            # n_steps = forecast_horizon

            # plt.figure(figsize=(12,6))
            # plt.plot(range(1, n_steps + 1), y_true[sample_idx], marker='o', label='True')
            # plt.plot(range(1, n_steps + 1), y_pred[sample_idx], marker='x', label='Predicted')
            # plt.title(f'Sample {sample_idx} - True vs Predicted Streamflow')
            # plt.xlabel('Forecast Horizon (Days Ahead)')
            # plt.ylabel('Streamflow (Q)')
            # plt.legend()
            # plt.show()

            # -------------------------------
            # Step 10: Making Future Predictions
            # -------------------------------

            # The following needs to be modified for single step (direct) forecasting

            # # Prepare the input sequence (last `input_window` days from the dataset)
            # last_sequence_input = df.iloc[-input_window:][feature_columns]
            # scaled_last_sequence_input = scaler.transform(last_sequence_input)
            # scaled_last_sequence_input = scaled_last_sequence_input.reshape((1, input_window, num_features))

            # # Predict
            # future_pred_scaled = model.predict(scaled_last_sequence_input)
            # future_pred = q_scaler.inverse_transform(future_pred_scaled)

            # # Create future dates
            # last_date = df['date'].iloc[-1]
            # future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

            # # Create a DataFrame for predictions
            # predictions_df = pd.DataFrame({
            #     'date': future_dates,
            #     'Q_pred': future_pred.flatten()
            # })

            # print(predictions_df)

            # -------------------------------
            # Step 11: Saving the Model
            # -------------------------------

            # Save the entire model in TensorFlow SavedModel format
            # model.save('saved_model/my_lstm_model')

            # Alternatively, save in HDF5 format
            # model.save('/my_lstm_model.h5')

            try:
                model.save(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "model.keras")
                logger.info(f"Trained LSTM-{filter_shortname} for h={forecast_horizon} has been saved.")
            except Exception as e:
                logger.exception(f"Failed to save trained LSTM-{filter_shortname} for h={forecast_horizon}.")
                return 1
            
            try:
                baseline_model.save(args.base_save_path / catchment_id / f"leadtime_{forecast_horizon}" / filter_shortname / "baseline_model.keras")
                logger.info(f"Trained baseline LSTM for h={forecast_horizon} has been saved.")
            except Exception as e:
                logger.exception(f"Failed to save trained baseline LSTM for h={forecast_horizon}.")
                return 1
    
    # # Example: Write to output file
    # try:
    #     with args.output.open('w') as f:
    #         f.write("Hello, World!\n")
    #     logger.info(f"Output written to: {args.output}")
    # except Exception as e:
    #     logger.exception("Failed to write to output file.")
    #     return 1

    logger.info("Script completed successfully.")
    return 0

if __name__ == "__main__":
    args = parse_arguments()

    # Initialize an empty dictionary to store timings
    timings = {}
    
    # Set logging level based on verbosity
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    
    exit_code = main(args)
    sys.exit(exit_code)
