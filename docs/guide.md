# Repository Guide

This document explains the `wavelet-lstm-camels` repository in detail. It is the
training codebase for the thesis *"Improving Short-term Streamflow Forecasting with
Wavelet Transforms: A Large-Sample Evaluation"* (You, 2024, University of Waterloo).

The guide starts with a high-level overview and progressively drills into the
details of every component.

---

## 1. What This Repository Does

This repository trains LSTM neural networks to forecast daily streamflow (river
discharge) across 620 catchments in the contiguous United States. The central
research question is:

> **Does augmenting an LSTM with wavelet-engineered features improve streamflow
> forecasts, and if so, under what conditions?**

For each catchment, two models are trained side by side:

- **W-LSTM (Wavelet LSTM):** Receives the original meteorological features *plus*
  MODWT-decomposed wavelet and scaling coefficients as input.
- **B-LSTM (Baseline LSTM):** Receives only the original meteorological features
  (no wavelet features).

Both models share an identical LSTM architecture so that any performance difference
is attributable to the wavelet features alone.

### Scale of the Experiment

```
620 catchments  x  3 lead times (1, 3, 5 days)  x  33 wavelet filters
= 61,380 W-LSTM models   +   corresponding B-LSTM baselines
```

All models were trained on the Digital Alliance of Canada's HPC cluster using
SLURM array jobs with GPU acceleration.

---

## 2. The Big Picture

The end-to-end pipeline, from raw data to saved model artifacts, is summarized
below. Each box is a stage; each arrow is a data flow.

```
 CAMELS CSV file (one per catchment)
 e.g. 01013500_camels.csv
           |
           v
 +---------------------+
 | 1. Load & Preprocess |  Parse dates, forward-fill missing values,
 |    (main.py)         |  select 8 features, add Unix timestamp
 +---------------------+
           |
           v
 +---------------------+
 | 2. MODWT Feature    |  Decompose each feature into wavelet (W)
 |    Engineering       |  and scaling (V) coefficients using one of
 | (feature_engineering |  33 wavelet filters at levels 1..6.
 |      .py)            |  Then drop boundary NaN rows.
 +---------------------+
           |
           v
 +---------------------+
 | 3. Sequence Creation |  Sliding window: 270-day input window,
 |    & Data Splitting  |  single target value at forecast_horizon
 |                      |  days ahead. Split 70/15/15.
 +---------------------+
           |
           v
 +---------------------+
 | 4. Scaling & Feature |  Fit MinMaxScaler on training data.
 |    Selection         |  Call hydroIVS (R) for EA-CMI feature
 |                      |  selection. Subset features by selected
 |                      |  indices.
 +---------------------+
           |
           v
 +-------------------+     +-------------------+
 | 5a. Train W-LSTM  |     | 5b. Train B-LSTM  |
 |  (wavelet feats)  |     |  (original feats)  |
 |  LSTM(256,tanh)   |     |  LSTM(256,tanh)   |
 |  Dropout(0.4)     |     |  Dropout(0.4)     |
 |  Dense(1,linear)  |     |  Dense(1,linear)  |
 +-------------------+     +-------------------+
           |                         |
           v                         v
 +---------------------+   +---------------------+
 | 6a. Evaluate W-LSTM |   | 6b. Evaluate B-LSTM |
 |  on test set        |   |  on test set        |
 |  (NSE, KGE, RMSE,  |   |  (NSE, KGE, RMSE,  |
 |   MAE, MAPE, MASE, |   |   MAE, MAPE, MASE, |
 |   R^2)              |   |   R^2)              |
 +---------------------+   +---------------------+
           |                         |
           +----------- + -----------+
                        |
                        v
              +-------------------+
              | 7. Save Artifacts |  14 files per configuration:
              |  model.keras,     |  models, scalers, metrics,
              |  scalers, metrics,|  predictions, feature names,
              |  predictions, ... |  training history, timings
              +-------------------+
```

This entire pipeline runs inside a nested loop in `main.py`:

```
for forecast_horizon in [1, 3, 5]:
    for filter_shortname in [33 wavelet filters]:
        ... full pipeline above ...
```

On the HPC cluster, each SLURM array task processes one catchment (one CSV
file), executing all 3 x 33 = 99 model pairs.

---

## 3. Repository Structure

```
wavelet-lstm-camels/
|
|-- main.py                     Core training script (the pipeline above)
|-- inference.py                Load trained models and run predictions
|-- feature_engineering.py      MODWT wavelet decomposition implementation
|-- metrics.py                  Hydrological evaluation metrics (NSE, KGE, etc.)
|-- utils.py                    Helpers: pickle loading, data validation, plotting
|
|-- requirements.txt            Python dependencies
|-- csv_filenames.txt           List of the 620 CAMELS CSV files used
|-- data.zip                    Compressed CAMELS dataset (621 CSV files)
|
|-- filters/
|   |-- scaling_dict.pkl        128 scaling (low-pass) filter coefficient arrays
|   |-- wavelet_dict.pkl        128 wavelet (high-pass) filter coefficient arrays
|
|-- docs/                       Documentation, guides, and installation
|   |-- data_outline.md
|   |-- main_script_explanation.md
|   |-- result_explanation.md
|   |-- guide.md
|   |-- installing_r.md
|   |-- r_packages.md
|   |-- python_version.md
|
|-- job_submission_scripts/     HPC (SLURM) job scripts
|   |-- run_camels_v2.sh
|   |-- run_camels_v2_onetime.txt
|
|-- naive_baseline/             Naive persistence baseline for comparison
|   |-- run_naive_baseline.py
|
|-- one_time_scripts/           Utility scripts for setup and validation
    |-- make_csv_filenames_txt.py
    |-- verify_directory_structure.py
    |-- inspect_pickles.py
    |-- check_slurm_outputs.py
    |-- add_camels_suffix.py
    |-- diff_text_files.py
    |-- delete_camels_dirs.sh
```

---

## 4. The Data: CAMELS Dataset

The CAMELS (Catchment Attributes and MEteorology for Large-sample Studies)
dataset provides daily hydrometeorological time series for 671 catchments across the
CONUS, spanning 1980-01-01 to 2014-12-31.

### 4.1 Features in Each CSV

Each CSV file (e.g. `01013500_camels.csv`) contains these columns:

| Column         | Description                     | Unit     |
|----------------|---------------------------------|----------|
| `date`         | Calendar date                   | YYYY-MM-DD |
| `Q`            | Streamflow (discharge)          | mm/day   |
| `dayl(s)`      | Day length                      | seconds  |
| `prcp(mm/day)` | Precipitation                   | mm/day   |
| `srad(W/m2)`   | Solar radiation                 | W/m^2    |
| `swe(mm)`      | Snow water equivalent           | mm       |
| `tmax(C)`      | Maximum air temperature         | C        |
| `tmin(C)`      | Minimum air temperature         | C        |
| `vp(Pa)`       | Vapor pressure                  | Pa       |

### 4.2 Catchment Filtering

Of the 671 catchments in CAMELS, 620 are used. The other 51 were excluded due
to missing records or insufficient data (fewer than 30 years). The 620 filenames are
listed in `csv_filenames.txt`.

---

## 5. MODWT Feature Engineering

The Maximal Overlap Discrete Wavelet Transform (MODWT) is the core
feature engineering method. It decomposes each input time series into
sub-series that capture variability at different timescales.

### 5.1 Why MODWT (Not DWT)?

Two properties make MODWT suitable for forecasting while the standard DWT is
not:

1. **No down-sampling.** The DWT produces coefficients shorter than the input.
   MODWT produces coefficients of the same length, so they can be used directly
   as features in a regression model.

2. **No look-ahead bias.** The standard DWT (and DWT-MRA) requires knowledge of
   future values. The MODWT implementation in this repo uses causal boundary
   handling: coefficient W_j[t] depends only on values at times t, t-1, ...,
   t-L_j+1. No future data is accessed.

### 5.2 Decomposition Diagram

Given a single feature (e.g. precipitation) and max level J=6, the MODWT
produces 7 new time series (6 wavelet + 1 scaling):

```
                     Original time series X
                              |
          MODWT decomposition (J = 6)
                              |
    +----+----+----+----+----+----+----+
    |    |    |    |    |    |    |    |
   W1   W2   W3   W4   W5   W6   V6
    |    |    |    |    |    |    |
   ~1d  ~2d  ~4d  ~8d ~16d ~32- smooth
                            64d  (>64d)

   W_j = wavelet coefficients at level j
         (captures fluctuations at timescale ~2^(j-1) days)

   V_6 = scaling coefficients at level 6
         (captures slow trends / periodicities > 64 days)
```

With 8 input features (Q + 7 meteorological variables), MODWT adds 8 x 7 = 56
new columns per catchment-filter combination. Together with the original 8 features
and the timestamp column, the W-LSTM candidate feature set has up to 65 columns
before feature selection.

### 5.3 Implementation Details (`feature_engineering.py`)

The `MODWTFeatureEngineer` class:

1. **Loads filter coefficients** from `filters/scaling_dict.pkl` and
   `filters/wavelet_dict.pkl`. These are pre-computed numpy arrays for 128
   wavelet types.

2. **Constructs equivalent filters** for each decomposition level j. For level
   j > 1, this involves iteratively convolving the base scaling filter with
   zero-upsampled versions of itself, then convolving with the wavelet filter.

3. **Computes MODWT coefficients** at each level using the equivalent filters,
   with causal boundary handling:

```
For each time index t:
    if t < L_j - 1:
        W_j[t] = NaN     (boundary coefficient -- discarded later)
        V_j[t] = NaN
    else:
        W_j[t] = sum( h_tilde_j[l] * X[t-l] )  for l = 0..L_j-1
        V_j[t] = sum( g_tilde_j[l] * X[t-l] )  for l = 0..L_j-1

where:
    h_tilde_j = equivalent wavelet filter at level j, normalized by 2^(j/2)
    g_tilde_j = equivalent scaling filter at level j, normalized by 2^(j/2)
    L_j = (2^j - 1)(L - 1) + 1   (equivalent filter length)
    L = base filter length
```

4. **Appends columns** named `{feature}_W{j}` and `{feature}_V{j}` to the
   DataFrame.

After MODWT, `df.dropna()` removes the leading rows that contain boundary NaN
values. The number of rows removed depends on the filter length and the max
decomposition level.

### 5.4 The 33 Wavelet Filters

| Family                    | Filters                                                     |
|---------------------------|-------------------------------------------------------------|
| Daubechies                | db1, db2, db3, db4, db5, db6, db7                           |
| Symlets                   | sym4, sym5, sym6, sym7                                      |
| Coiflets                  | coif1, coif2                                                |
| Least Asymmetric          | la8, la10, la12, la14                                       |
| Fejer-Korovkin            | fk4, fk6, fk8, fk14                                        |
| Best-localized Daubechies | bl7                                                         |
| Morris minimum-bandwidth  | mb4_2, mb8_2, mb8_3, mb8_4, mb10_3, mb12_3, mb14_3         |
| Han linear-phase moments  | han2_3, han3_3, han4_5, han5_5                              |

All 33 are orthogonal filters with lengths up to 14, selected to limit the number
of boundary coefficients removed.

---

## 6. Feature Selection via hydroIVS

After MODWT, the augmented feature set can contain 60+ columns. Not all are
useful. The project uses a nonlinear, information-theoretic feature selection
algorithm to pick the most informative subset.

### 6.1 The Algorithm: EA-CMI with Threshold

The feature selection is performed by the `hydroIVS` R package (called from
Python via `rpy2`). The method is **Edgeworth Approximation-based Conditional Mutual
Information (EA-CMI)** with a tolerance (threshold) stopping criterion.

```
Algorithm: Feature Selection using CMI with Threshold
------------------------------------------------------
Input: Feature set D, target variable Y, threshold tau
Output: Selected feature subset S

1.  S = {} (empty set)
2.  WHILE True:
3.      For each candidate feature f not in S:
4.          Compute CMI(f, Y | S)   -- conditional mutual information
5.                                     of f and Y given already-selected S
6.      best_feature = argmax CMI
7.      max_cmi = max CMI
8.      mi = MI(best_feature, Y)   -- mutual information
9.      ratio = max_cmi / mi
10.     IF ratio < tau:
11.         BREAK                   -- stop: adding more features is not
12.                                    worth the diminishing information gain
13.     Add best_feature to S
14. RETURN S
```

In the code, the threshold (tau) is 0.05 and the method string is `"ea_cmi_tol"`.

### 6.2 Why EA-CMI?

- **Nonlinear:** Unlike correlation-based methods, CMI captures nonlinear
  dependencies between features and the target variable.
- **Parameterless estimation:** The Edgeworth Approximation approach to estimating
  probability density functions does not require hyper-parameters (unlike KNN or
  KDE estimators).
- **Efficient:** Computationally cheaper than the popular PMIS method, making it
  feasible for a large-sample study (620 catchments x 99 configurations).

### 6.3 How It Fits Into the Pipeline

```
                   Training data (scaled)
                          |
          +---------------+---------------+
          |                               |
   All MODWT + original           Original 9 features
   features (~65 cols)            (Q, timestamp, 7 meteo)
          |                               |
   hydroIVS EA-CMI               hydroIVS EA-CMI
   (tau = 0.05)                  (tau = 0.05)
          |                               |
   Selected indices              Selected indices
   for W-LSTM                    for B-LSTM
          |                               |
   Subset X_train,              Subset X_train,
   X_val, X_test                X_val, X_test
```

Feature selection runs on MinMax-scaled training data. The selected feature
indices are saved and later used to subset the 3D input arrays
`X[samples, timesteps, features]` along the features axis.

---

## 7. Sequence Creation and Data Splitting

### 7.1 Direct Forecasting Strategy

The model uses a **direct forecasting** approach: for each forecast horizon h
(1, 3, or 5 days), a separate model is trained to predict Q at time t+h given
features at times t-269 through t.

```
         Input window (270 days)                Target
  |<------------------------------------>|
  t-269  t-268  ...  t-1    t              t+h
  [____________________________________]    Q(t+h)
  Feature values at each day                  |
  (8 original + MODWT columns)               single
                                             scalar
```

### 7.2 Data Split

Sequences are created by sliding the window one day at a time. They are split
chronologically (no shuffling):

```
|<--------- 70% train --------->|<-- 15% val -->|<-- 15% test -->|
 earliest sequences                                 latest sequences
```

---

## 8. LSTM Model Architecture

Both W-LSTM and B-LSTM share the same architecture, based on
Kratzert et al. (2019) who used it for streamflow simulation across CAMELS:

```
         Input
   (270, n_features)
          |
          v
  +----------------+
  |  LSTM Layer    |   256 hidden units
  |  activation:   |   return_sequences=False
  |    tanh        |   (only final hidden state is output)
  +----------------+
          |
          v
  +----------------+
  |  Dropout       |   rate = 0.4
  +----------------+
          |
          v
  +----------------+
  |  Dense Layer   |   1 unit, linear activation
  +----------------+
          |
          v
     Q_predicted
     (1 scalar)
```

The only difference between the two models is `n_features`:
- **W-LSTM:** Number of features selected by EA-CMI from the MODWT-augmented set
- **B-LSTM:** Number of features selected by EA-CMI from the 9 original features

### 8.1 Training Configuration

| Parameter          | Value                           |
|--------------------|---------------------------------|
| Optimizer          | Adam (learning rate = 0.001)    |
| Loss function      | Mean Squared Error (MSE)        |
| Batch size         | 32                              |
| Max epochs         | 1000                            |
| Early stopping     | Monitor `val_nse`, patience=10, restore best weights, mode=max |
| Feature scaling    | MinMaxScaler (fitted on training set only)      |
| Target scaling     | MinMaxScaler on Q (fitted on training set only) |

---

## 9. Evaluation Metrics

Models are evaluated on the held-out test set after inverse-transforming
predictions back to the original scale. Seven metrics are computed
(`metrics.py` + sklearn):

| Metric | Formula / Description | Range |
|--------|----------------------|-------|
| **NSE** (Nash-Sutcliffe Efficiency) | 1 - SS_res / SS_tot | (-inf, 1]; 1 = perfect |
| **KGE** (Kling-Gupta Efficiency)    | 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2) | (-inf, 1] |
| **RMSE** | sqrt(mean((y_true - y_pred)^2)) | [0, inf) |
| **MAE**  | mean(\|y_true - y_pred\|) | [0, inf) |
| **MAPE** | mean(\|y_true - y_pred\| / \|y_true\|) x 100 | [0, inf) |
| **MASE** | MAE / mean(\|y_true[i] - y_true[i-1]\|) | [0, inf) |
| **R^2**  | 1 - SS_res / SS_tot (sklearn version) | (-inf, 1] |

NSE is the primary metric. The thesis uses `NSE > 0.4` as a minimum performance
threshold to filter catchments for detailed analysis.

Custom Keras metric classes for NSE and KGE are implemented in `main.py` so
that early stopping can monitor `val_nse` during training.

---

## 10. Output Artifacts

For each combination of catchment, lead time, and wavelet filter, 14 files are
saved:

```
{base_save_path}/{catchment_id}/leadtime_{h}/{filter}/
|
|-- model.keras                          Trained W-LSTM model
|-- baseline_model.keras                 Trained B-LSTM model
|
|-- feature_scaler.pkl                   MinMaxScaler for W-LSTM input features
|-- baseline_feature_scaler.pkl          MinMaxScaler for B-LSTM input features
|-- q_scaler.pkl                         MinMaxScaler for W-LSTM target (Q)
|-- baseline_q_scaler.pkl               MinMaxScaler for B-LSTM target (Q)
|
|-- history.pkl                          W-LSTM training history (loss, metrics per epoch)
|-- baseline_history.pkl                 B-LSTM training history
|
|-- test_metrics_dict.pkl                W-LSTM test metrics {nse, kge, rmse, mae, ...}
|-- baseline_test_metrics_dict.pkl       B-LSTM test metrics
|
|-- pred_label_df.pkl                    DataFrame: date, y_pred, y_true (W-LSTM)
|-- baseline_pred_label_df.pkl           DataFrame: date, y_pred, y_true (both models)
|
|-- ea_cmi_tol_005_selected_feature_names.pkl
|       Dict with selected feature names, indices, and scores
|       for both W-LSTM and B-LSTM
|
|-- timings.pkl                          Dict: {modwt, ea_cmi_tol, lstm} durations in seconds
```

### Full directory tree (one catchment)

```
01013500/
|-- leadtime_1/
|   |-- db1/         (14 files)
|   |-- db2/         (14 files)
|   |-- ...
|   |-- sym7/        (14 files)
|   +-- (33 directories total)
|
|-- leadtime_3/
|   +-- (33 directories, 14 files each)
|
|-- leadtime_5/
    +-- (33 directories, 14 files each)
```

Across all 620 catchments: 620 x 3 x 33 = 61,380 directories, each containing
14 files.

---

## 11. HPC Job Submission

Training was performed on Digital Alliance of Canada's Graham cluster.

### 11.1 SLURM Array Job (`run_camels_v2.sh`)

```
#SBATCH --job-name=wavelet_camels
#SBATCH --time=15:59:00          Wall clock limit per task
#SBATCH --ntasks=1               One task (no MPI)
#SBATCH --mem=12G                RAM per task
#SBATCH --gpus-per-node=1        One GPU for LSTM training
#SBATCH --array=1-6              Array tasks (one CSV per task)
```

Each array task:
1. Loads modules: `gcc/12.3`, `python/3.11`, `r/4.4`
2. Installs R packages into `$SLURM_TMPDIR` from pre-built tarballs
3. Creates a Python virtualenv and installs TensorFlow, rpy2, etc.
4. Reads its CSV filename from `csv_filenames_v5.txt` using `sed -n "${SLURM_ARRAY_TASK_ID}p"`
5. Runs `python main.py --csv_filename "$CSV_FILE" --verbose`

### 11.2 One-Time Setup (`run_camels_v2_onetime.txt`)

Before submitting the job, run these commands once on the login node:
1. Download CRAN package source tarballs (Rcpp, RcppEigen, ranger, Boruta, RANN, RRF)
2. Clone and build the `hydroIVS` R package from GitHub
3. Place all `.tar.gz` files in the shared project directory

### 11.3 Retry Workflow

If some SLURM tasks fail:
1. Run `one_time_scripts/check_slurm_outputs.py` to scan `slurm-*.out` files
   for tasks that did not finish successfully.
2. The script writes the failed CSV filenames to `csv_filenames_v5.txt`.
3. Edit the `--array` line in the job script and resubmit.

---

## 12. Inference Pipeline

To make use of the trained outputs for streamflow forecasting, see the
[wavelet-streamflow-forecast](https://github.com/johnswyou/wavelet-streamflow-forecast) repository.

This repository also includes a legacy `inference.py` script that loads
pre-trained models from a local `correct_output/` directory and makes predictions
for a specified station, lead time, and wavelet filter. The result data is
available upon request (contact John You at johnswyou@gmail.com).

```
python inference.py \
    --station_id 01013500 \
    --leadtime 1 \
    --wavelet_filter db1 \
    --start_date 2010-01-01 \
    --end_date 2010-12-31
```

Steps:
1. Load and preprocess the CAMELS CSV (same as training)
2. Apply MODWT with the specified wavelet filter
3. Create sequences and filter by date range
4. Load saved scalers, feature indices, and model weights
5. Scale inputs, select features, and run predictions for both W-LSTM and B-LSTM
6. Inverse-transform predictions and compute metrics
7. Print a comparison table and optionally save to CSV

The model weights are loaded by extracting the `.keras` zip archive and loading
`model.weights.h5` into a freshly constructed model architecture.

---

## 13. File-by-File Reference

### `main.py` -- Training Pipeline

The core script. Key components:

- **`parse_arguments()`**: CLI args for csv_filename, base_save_path, base_csv_path,
  max_level, verbose.
- **`r2_keras()`**: Custom R^2 metric for use in Keras `model.compile()`.
- **`NashSutcliffeEfficiency`**: Stateful Keras metric class that accumulates sums
  across batches to compute NSE at epoch end.
- **`KlingGuptaEfficiency`**: Stateful Keras metric class for KGE.
- **`main()`**: Contains the full nested loop over forecast horizons and wavelet
  filters. Each iteration runs the complete pipeline from data loading to model
  saving.
- **`scale_sequences()`**: Nested function that applies pre-fitted scalers to
  sequences of (input_df, output_row) pairs, returning 3D numpy arrays.

Note: The `timings` dictionary is initialized at module level and populated inside
`main()` with durations for the MODWT, EA-CMI, and LSTM training stages.

### `feature_engineering.py` -- MODWT

- **`MODWTFeatureEngineer.__init__()`**: Loads filter dictionaries, validates levels,
  computes boundary coefficient counts (L_J).
- **`MODWTFeatureEngineer.modwt()`**: Runs the MODWT on a 1D signal. Contains nested
  helper functions `insert_zeros_between_elements()`, `equivalent_filter()`, and
  `modwt_level_j()`.
- **`MODWTFeatureEngineer.transform()`**: Applies MODWT to each specified column of
  a DataFrame and appends the resulting coefficient columns.

### `metrics.py` -- Evaluation Metrics

Four functions, each taking `y_true` and `y_pred` as numpy arrays:
- `mean_absolute_percentage_error()`
- `mean_absolute_scaled_error()`
- `nash_sutcliffe_efficiency()`
- `kling_gupta_efficiency()`

### `utils.py` -- Utilities

- `load_pickle()`: Deserialize a `.pkl` file.
- `get_csv_filename_without_extension()`: Extract station ID from path.
- `create_directory_if_not_exists()`: Safe directory creation.
- `check_csv_order_and_continuity()`: Validate date ordering and continuity of
  all CSVs in a folder.
- `plot_wavelet_stacks()`: Visualize original feature, scaling, and wavelet
  coefficient time series in a vertical stack plot.

### `inference.py` -- Inference

- `load_data()`: Same preprocessing as training (parse dates, ffill, select
  features, add timestamp).
- `create_sequences()`: Same sliding-window logic as training.
- `filter_sequences_by_date_range()`: Subset sequences by target date.
- `scale_sequences()`: Apply saved scalers to input sequences.
- `create_model_architecture()`: Reconstruct the LSTM architecture.
- `load_compatible_model()`: Extract weights from `.keras` zip and load them
  into the reconstructed model.
- `load_model_artifacts()`: Load all 6+ artifacts (scalers, feature info, models)
  from a model directory.
- `compute_metrics()`: Compute all 8 evaluation metrics.
- `run_inference()`: Orchestrates the full inference pipeline.

### `naive_baseline/run_naive_baseline.py`

Computes a naive persistence baseline: forecast Q(t+h) = Q(t). Reports NSE
across all catchments for lead times 1, 3, 5, 7, and 14 days.

---

## 14. Dependencies

### Python (3.11.12 recommended)

Key packages (see `requirements.txt` for full list):

| Package         | Version  | Purpose                        |
|-----------------|----------|--------------------------------|
| tensorflow      | 2.15.0   | LSTM model                     |
| keras           | 2.15.0   | High-level model API           |
| rpy2            | 3.1.0    | Python-R bridge for hydroIVS   |
| scikit-learn    | 1.6.1    | MinMaxScaler, sklearn metrics  |
| pandas          | 1.5.3    | Data manipulation              |
| numpy           | 1.24.3   | Numerical operations           |
| matplotlib      | 3.10.3   | Visualization                  |
| scipy           | 1.15.3   | Scientific computing           |

### R (4.4)

| Package    | Purpose                                      |
|------------|----------------------------------------------|
| hydroIVS   | EA-CMI feature selection (`ivsIOData()`)     |
| Rcpp       | Dependency for hydroIVS                      |
| RcppEigen  | Dependency                                   |
| ranger     | Dependency                                   |
| Boruta     | Dependency                                   |
| RANN       | Dependency                                   |
| RRF        | Dependency                                   |

---

## 15. Key Findings (from the thesis)

For readers interested in the research outcomes:

- **W-LSTM outperforms B-LSTM** in 97% of catchments for both 1-day and 3-day
  forecast horizons (across all 620 catchments).
- When restricting to catchments where B-LSTM achieves `NSE > 0.4`:
  - 1-day ahead: W-LSTM improves upon B-LSTM in **60%** of catchments.
  - 3-day ahead: W-LSTM improves upon B-LSTM in **70%** of catchments.
- The **Morris Minimum Bandwidth 4.2 (mb4_2)** filter outperforms B-LSTM in over
  50% of filtered catchments for both forecast horizons.
- Wavelet transforms provide the greatest improvement in **D (snowy) and B (dry)**
  Koppen climate regions.
