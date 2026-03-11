# Explanation of `main.py`

The `main.py` script is designed to train and evaluate wavelet-based LSTM (Long Short-Term Memory) neural network models for streamflow forecasting, using data from the CAMELS (Catchment Attributes and MEteorology for Large-sample Studies) dataset. It also trains a baseline LSTM model for comparison.

Here's a step-by-step walkthrough:

1.  **Imports and Setup:**
    *   Standard Python libraries like `argparse`, `logging`, `os`, `pickle`, `sys`, `time`, and `pathlib` are imported for general operations.
    *   `numpy` and `pandas` are imported for numerical operations and data manipulation.
    *   `rpy2` is used to interface with R, specifically to call the `hydroIVS` package for feature selection. `rpy2.robjects.packages.importr` is used to import R packages, and `rpy2.robjects.pandas2ri` along with `rpy2.robjects.conversion.localconverter` are used to convert data between pandas DataFrames and R objects.
    *   `tensorflow` and `keras` are imported for building and training the LSTM models. Various Keras components like `Dense`, `Dropout`, `Input`, `LSTM`, `Sequential`, `Adam` (optimizer), and `EarlyStopping` (callback) are used.
    *   `sklearn.metrics` and `sklearn.preprocessing` are used for evaluation metrics (like MAE, MSE, R²) and feature scaling (`MinMaxScaler`).
    *   Custom modules `feature_engineering` (containing `MODWTFeatureEngineer`) and `metrics` (containing custom evaluation metric functions like `nash_sutcliffe_efficiency`, `kling_gupta_efficiency`, etc.) are imported.
    *   TensorFlow's C++ log level is set to '1' to reduce verbosity.

2.  **Logging Setup (`setup_logging` function):**
    *   This function configures the logging for the script. It sets the logging level (e.g., DEBUG, INFO) based on user input and formats the log messages to include timestamp, logger name, level, and message. Logs are directed to standard output.

3.  **Argument Parsing (`parse_arguments` function):**
    *   This function defines and parses command-line arguments that control the script's behavior:
        *   `--csv_filename`: (Required) Path to the input CSV file for a specific catchment. As per `docs/data_outline.md`, these CSV files are expected to be in the `data/` directory and contain time-series data like streamflow (`Q`), precipitation, temperature, etc.
        *   `--max_level`: (Optional, default: 6) Maximum decomposition level for the MODWT (Maximal Overlap Discrete Wavelet Transform).
        *   `--base_save_path`: (Optional, default: `/home/jswyou/scratch`) Base directory where all output (models, scalers, results) will be saved.
        *   `--base_csv_path`: (Optional, default: `/home/jswyou/projects/def-quiltyjo/jswyou/oct_2024/wavelet-lstm-camels/data`) Base directory where the CAMELS CSV data is stored.
        *   `-v` or `--verbose`: (Optional flag) Enables verbose (DEBUG level) logging.

4.  **Custom Keras Metrics:**
    *   `r2_keras`: A function to calculate the R² (coefficient of determination) metric within a Keras model.
    *   `NashSutcliffeEfficiency`: A custom Keras metric class to calculate the Nash-Sutcliffe Efficiency. This class maintains state variables to compute NSE incrementally across batches.
    *   `KlingGuptaEfficiency`: A custom Keras metric class to calculate the Kling-Gupta Efficiency. This class maintains state variables to compute KGE incrementally across batches.

5.  **Main Function (`main` function):**
    *   This is the core of the script. It takes the parsed arguments as input.
    *   **Initialization:**
        *   A logger instance is created.
        *   It checks if the input CSV file exists.
        *   It extracts a `catchment_id` from the CSV filename (e.g., "02096846" from `02096846_camels.csv`).
        *   It creates a subdirectory for this `catchment_id` under `args.base_save_path` if it doesn't exist.
        *   A list of `possible_filters` (wavelet names like 'bl7', 'coif1', 'db1', etc.) is defined. These names correspond to the keys found in `filters/wavelet_dict.pkl` and `filters/scaling_dict.pkl` as described in `docs/data_outline.md`, which store the wavelet and scaling filter coefficients.
    *   **Outer Loops:** The script then enters nested loops:
        *   It iterates through `forecast_horizon` values: `[1, 3, 5]` (days).
        *   Inside that, it iterates through each `filter_shortname` from `possible_filters`.
        *   For each combination of `forecast_horizon` and `filter_shortname`, it creates a specific output directory (e.g., `/home/jswyou/scratch/02096846/leadtime_1/db1/`).
    *   **Inside the Loops (for each forecast horizon and wavelet filter):**
        *   **Step 1: Load Dataset:**
            *   The specified CAMELS CSV file (e.g., `data/02096846_camels.csv`) is loaded into a pandas DataFrame. Dates are parsed.
        *   **Step 2: Data Preprocessing:**
            *   Missing values in the DataFrame are filled using forward fill (`ffill`).
            *   A predefined list of `features` (including 'Q', 'prcp(mm/day)', 'tmax(C)', etc.) is selected. The DataFrame is subsetted to keep only the 'date' column and these features.
            *   Data is sorted by date.
            *   A 'timestamp' column (Unix timestamp) is created from the 'date' column.
        *   **Step 2.5: MODWT Feature Engineering:**
            *   An instance of `MODWTFeatureEngineer` (from `feature_engineering.py`) is created. This class uses the specified `filter_shortname` (wavelet) and `args.max_level`. The `MODWTFeatureEngineer` itself loads wavelet and scaling filter coefficients from `filters/wavelet_dict.pkl` and `filters/scaling_dict.pkl`.
            *   The `transform` method of `MODWTFeatureEngineer` is called to compute MODWT wavelet (W) and scaling (V) coefficients for each of the specified `features`. These new coefficient series are added as new columns to the DataFrame (e.g., `Q_W1`, `Q_V6`, `prcp(mm/day)_W1`, etc.).
            *   Rows with NaN values (generated at the beginning of the series due to MODWT boundary effects) are dropped.
            *   `feature_columns` is updated to include these new MODWT features.
            *   Timing for the MODWT process is recorded in the `timings` dictionary.
        *   **Step 3: Splitting the Dataset:**
            *   The data is prepared into sequences for the LSTM model. An `input_window` of 270 days is used.
            *   For each sample, the input is a sequence of `input_window` days of all features, and the output is the 'Q' value at the specified `forecast_horizon` days ahead (direct forecasting).
            *   The sequences are split into training (70%), validation (15%), and test (15%) sets.
        *   **Step 4: Feature Selection and Scaling:**
            *   **Scaler Fitting (on Training Data):**
                *   A copy of the DataFrame (`train_df`) is made, focusing on the date range covered by the training sequences.
                *   A target column `Q_target` is created by shifting the 'Q' column by the negative `forecast_horizon`.
                *   Two `MinMaxScaler` instances are initialized: `scaler` for input features and `q_scaler` for the target variable ('Q').
                *   These scalers are *fitted* only on the training portion of the data (`train_df_features` and `train_df_q_target`).
                *   Separate scalers (`baseline_scaler`, `baseline_q_scaler`) are also fitted for the baseline model features. The baseline features are a predefined subset: "Q", "timestamp", "dayl(s)", "prcp(mm/day)", "srad(W/m2)", "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)".
            *   **Feature Selection using `hydroIVS`:**
                *   The scaled training features (`train_df_features_r`) and scaled target (`train_df_q_target_r`) are converted to R objects using `rpy2`. The same is done for baseline features and target.
                *   The `hydroIVS` R package is imported.
                *   The `ivsIOData` function from `hydroIVS` is called. Based on the `hydroIVS` GitHub README, this function performs input variable selection. The method used here is `"ea_cmi_tol"` with a parameter of `0.05`. The README describes `ea_cmi_tol` as "Edgeworth Approximation (EA) based Shannon Conditional Mutual Information (CMI) Input Variable Selection (IVS) using ratio of CMI over Mutual Information (MI) to identify significant inputs." The `0.05` likely represents this ratio threshold.
                *   This selection is done for both the MODWT features and the baseline features.
                *   The results (selected feature indices, names, and scores) are converted back to Python objects. Indices are adjusted for 0-based Python indexing.
                *   Timing for the feature selection process is recorded in the `timings` dictionary.
        *   **Step 4.5: Scale Features (Applying Scalers):**
            *   The `scale_sequences` helper function is defined to apply the *fitted* scalers (`scaler` and `q_scaler` for MODWT features, `baseline_scaler` and `baseline_q_scaler` for baseline) to the training, validation, and test sequences. This produces `X_train`, `y_train_scaled`, `X_val`, `y_val_scaled`, etc.
            *   The input feature arrays (`X_train`, `X_val`, `X_test`) are then subsetted to include only the features selected by `hydroIVS` (using `selected_feature_indices`). The same is done for the baseline model's input arrays.
            *   The fitted scalers (`scaler`, `q_scaler`, `baseline_scaler`, `baseline_q_scaler`) and the selected feature information (names, indices, scores for both MODWT and baseline) are saved to pickle files in the specific output directory.
        *   **Step 5: Building the Model (Wavelet LSTM):**
            *   A Keras `Sequential` model is defined.
            *   It consists of an `Input` layer, an `LSTM` layer (256 units, tanh activation), a `Dropout` layer (0.4 rate), and a `Dense` output layer (1 unit, linear activation for direct single-step forecasting).
            *   The model is compiled with the `Adam` optimizer, 'mse' (mean squared error) loss, and several metrics including 'mae', 'mse', 'mape', and the custom `r2_keras`, `NashSutcliffeEfficiency`, and `KlingGuptaEfficiency`.
            *   The model summary is printed.
        *   **Step 5.5: Building the Baseline Model:**
            *   A similar Keras `Sequential` model is defined for the baseline, using the `baseline_num_features` (number of selected baseline features).
            *   It has the same architecture and compilation settings as the wavelet LSTM.
            *   The baseline model summary is printed.
        *   **Step 6: Training the Model:**
            *   `EarlyStopping` callbacks are defined for both models, monitoring `val_nse` (validation Nash-Sutcliffe Efficiency) with a patience of 10 epochs and restoring the best weights. The mode is 'max' because higher NSE is better.
            *   The wavelet LSTM model is trained using `model.fit` with `X_train` and `y_train_scaled`, validating on `(X_val, y_val_scaled)`.
            *   The baseline LSTM model is trained similarly with its respective data.
            *   Timing for the LSTM training is recorded in the `timings` dictionary.
            *   The `timings` dictionary and the training history (losses, metrics per epoch) for both models are saved to pickle files.
        *   **Step 8: Evaluating the Model (Wavelet LSTM):**
            *   Predictions (`y_pred_scaled`) are made on the test set (`X_test`).
            *   These scaled predictions and the true scaled test values (`y_test_scaled`) are inverse-transformed back to their original scale using the fitted `q_scaler`.
            *   Various evaluation metrics are calculated using functions from `sklearn.metrics` and the custom `metrics.py` module: MAPE, RMSE, MAE, MASE (Mean Absolute Scaled Error), R², NSE, and KGE.
            *   These test metrics are printed and saved to `test_metrics_dict.pkl`.
            *   A DataFrame (`pred_label_df`) is created containing the test dates, true values (`y_true`), and predicted values (`y_pred`). This DataFrame is saved to a pickle file.
        *   **Step 8.5: Evaluating the Baseline Model:**
            *   The same evaluation process is repeated for the baseline model using `baseline_X_test`, `baseline_y_test_scaled`, and `baseline_q_scaler`.
            *   Baseline test metrics are printed and saved to `baseline_test_metrics_dict.pkl`.
            *   The baseline predictions (`baseline_y_pred`) and true values (`baseline_y_true`) are added to the `pred_label_df`, which is then re-saved as `baseline_pred_label_df.pkl`.
        *   **Step 11: Saving the Model:**
            *   The trained wavelet LSTM model (`model`) is saved in Keras format (`.keras`) to the specific output directory.
            *   The trained baseline LSTM model (`baseline_model`) is also saved.
    *   **Completion:**
        *   A "Script completed successfully" message is logged.
        *   The function returns an exit code (0 for success, 1 for failure).

6.  **Script Execution (`if __name__ == "__main__":`)**
    *   Command-line arguments are parsed.
    *   The `timings` dictionary is initialized.
    *   Logging is set up based on the `--verbose` flag.
    *   The `main` function is called, and its exit code is used to terminate the script.

## Key Files Generated

For each combination of catchment, forecast horizon, and wavelet filter, the script generates the following files in the directory structure `{base_save_path}/{catchment_id}/leadtime_{forecast_horizon}/{filter_shortname}/`:

**Wavelet LSTM Model Files:**
- `model.keras` - Trained wavelet LSTM model
- `feature_scaler.pkl` - MinMaxScaler for input features
- `q_scaler.pkl` - MinMaxScaler for target variable
- `test_metrics_dict.pkl` - Test evaluation metrics
- `history.pkl` - Training history
- `pred_label_df.pkl` - Predictions and true values with dates

**Baseline LSTM Model Files:**
- `baseline_model.keras` - Trained baseline LSTM model
- `baseline_feature_scaler.pkl` - MinMaxScaler for baseline input features
- `baseline_q_scaler.pkl` - MinMaxScaler for baseline target variable
- `baseline_test_metrics_dict.pkl` - Baseline test evaluation metrics
- `baseline_history.pkl` - Baseline training history
- `baseline_pred_label_df.pkl` - Combined predictions dataframe with both models

**Shared Files:**
- `ea_cmi_tol_005_selected_feature_names.pkl` - Selected features for both models
- `timings.pkl` - Timing information for different processing steps

**Note**: The structure and content of the above files (specifically, the `.pkl` files shown above) are described in greater detail in `docs/result_explanation.md`.

In summary, `main.py` automates a complex workflow for hydrological forecasting. For each specified catchment, forecast horizon, and wavelet type, it performs:
1.  Data loading and extensive preprocessing.
2.  Wavelet decomposition to generate additional features.
3.  Feature selection using an R package (`hydroIVS`) that employs conditional mutual information techniques.
4.  Training and evaluation of a specialized LSTM model using these selected wavelet-based features.
5.  Training and evaluation of a baseline LSTM model using a standard set of features (also selected via `hydroIVS`).
6.  Saving of all relevant artifacts: processed data, scalers, selected feature information, trained models, training histories, evaluation metrics, and predictions.
