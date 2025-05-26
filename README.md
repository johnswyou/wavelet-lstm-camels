# Wavelet-LSTM Streamflow Forecasting with CAMELS Dataset

## Overview

This repository contains the research code for training wavelet-enhanced LSTM models for streamflow forecasting using the CAMELS (Catchment Attributes and Meteorology for Large-sample Studies) dataset. The project investigates whether incorporating MODWT (Maximal Overlap Discrete Wavelet Transform) features improves LSTM-based streamflow forecasting performance across multiple catchments.

The research involved training **61,380 LSTM models** in total:
- 620 catchments × 3 lead times × 33 wavelet/scaling filters = 61,380 wavelet LSTM models
- Plus baseline LSTM models (without wavelet features) for comparison

## Key Features

- **Multi-horizon Forecasting:** 1, 3, and 5 days ahead streamflow prediction
- **Wavelet Feature Engineering:** 33 different wavelet and scaling filters using MODWT
- **Large-scale Evaluation:** 620 catchments from the CAMELS dataset (51 catchments excluded due to data quality issues)
- **Advanced Feature Selection:** Uses the `hydroIVS` R package for input variable selection
- **HPC-ready:** Designed for parallel execution on SLURM-based clusters
- **Comprehensive Evaluation:** Includes custom hydrological metrics (NSE, KGE)

## Project Structure

```
wavelet-lstm-camels/
├── main.py                          # Main training script
├── inference.py                     # Inference script for model evaluation
├── feature_engineering.py           # MODWT feature extraction
├── metrics.py                       # Custom evaluation metrics
├── utils.py                         # Utility functions
├── requirements.txt                 # Python dependencies
├── csv_filenames.txt               # List of 620 CSV files used
├── data.zip                        # Compressed CAMELS dataset (621 files)
├── context/                        # Documentation and explanations
│   ├── main_script_explanation.md
│   ├── result_explanation.md
│   └── data_outline.md
├── docs/                           # Installation guides
│   ├── installing_r.md            # R installation for WSL Ubuntu
│   ├── r_packages.md              # Required R packages
│   └── python_version.md          # Recommended Python version (3.11.12)
├── job_submission_scripts/         # HPC job submission scripts
│   ├── run_camels_v2.sh          # SLURM array job script
│   └── run_camels_v2_onetime.txt # One-time setup commands
├── filters/                        # Wavelet filter specifications
├── naive_baseline/                 # Naive baseline model
└── one_time_scripts/              # Utility scripts
```

## Experimental Design

### Models Trained

For each of the 620 catchments:
- **Lead times:** 1, 3, and 5 days ahead
- **Wavelet filters:** 33 different filters including:
  - Daubechies (db1-db7)
  - Symlets (sym4-sym7)
  - Coiflets (coif1-coif2)
  - Fejer-Korovkin (fk4, fk6, fk8, fk14)
  - Least Asymmetric (la8, la10, la12, la14)
  - Morris minimum-bandwidth (mb4_2, mb8_2, mb8_3, mb8_4, mb10_3, mb12_3, mb14_3)
  - Han filters (han2_3, han3_3, han4_5, han5_5)
  - Best-localized Daubechies (bl7)

### Input Variable Selection

The project employs a niche feature selection strategy using:
- Mutual information-based selection via `hydroIVS` R package
- Tolerance threshold of 0.005 for EA-CMI (Edgeworth Approximation of Conditional Mutual Information)

## Installation

### Prerequisites

- Python 3.11.12 (recommended)
- R installation (for `hydroIVS` package)
- WSL Ubuntu (if on Windows)

### Setup Instructions

**Note**: Make sure you have `git-lfs` installed. See [this link](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/johnswyou/wavelet-lstm-camels.git
   cd wavelet-lstm-camels
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install R and required packages:**
   See `docs/installing_r.md` for detailed R installation instructions on WSL Ubuntu.
   Required R packages are listed in `docs/r_packages.md`.

4. **Extract the data:**
   ```bash
   unzip data.zip -d data/
   ```

## Usage

### Training Models

The main training script `main.py` trains both wavelet-enhanced and baseline LSTM models:

```bash
python main.py --csv_filename <FILENAME> --base_save_path <OUTPUT_PATH> --base_csv_path <DATA_PATH> [options]
```

**Arguments:**
- `--csv_filename`: Name of the CSV file (e.g., "01013500_camels.csv")
- `--base_save_path`: Base directory for saving results
- `--base_csv_path`: Directory containing CAMELS CSV files
- `--max_level`: Maximum MODWT decomposition level (default: 6)
- `--verbose`: Enable verbose logging

### HPC Job Submission

For running on SLURM-based clusters (e.g., Graham cluster in Waterloo, Ontario):

1. **One-time setup** (on login node):
   ```bash
   # Execute commands from job_submission_scripts/run_camels_v2_onetime.txt
   ```

2. **Submit array job:**
   ```bash
   sbatch job_submission_scripts/run_camels_v2.sh
   ```

The array job processes all CSV files listed in `csv_filenames.txt` in parallel.

**Note**: You will likely need to edit the job submission script (lines 7, 20, and 61).

### Inference

**Note**: At present, the only person who can access the s3 bucket (`s3://modwt-lstm-results`) is the author. This will change once a more permanent location is found for the results data.

The `inference.py` script accesses trained models from an S3 bucket mounted locally:

```bash
# Mount S3 bucket (modwt-lstm-results) to ../mnt
# Then run inference
python inference.py
```

Results are organized in the S3 bucket as:
```
mnt/correct_output/
└── [STATION_ID]/
    └── leadtime_[1,3,5]/
        └── [WAVELET_FILTER]/
            ├── model.keras
            ├── baseline_model.keras
            ├── test_metrics_dict.pkl
            ├── baseline_test_metrics_dict.pkl
            └── ... (other artifacts)
```

## Data

### CAMELS Dataset

- **Total catchments in CAMELS:** 671
- **Catchments used:** 620 (51 excluded due to data quality issues)
- **Data file:** `data.zip` contains 621 CSV files (one extra file `09535100_camels.csv` to be removed)
- **Active files:** Listed in `csv_filenames.txt` (620 in total)

### Features

Each CSV file contains:
- Streamflow (Q)
- Meteorological forcing data:
  - Precipitation (prcp)
  - Temperature (tmax, tmin)
  - Solar radiation (srad)
  - Vapor pressure (vp)
  - Day length (dayl)

## Results Storage

Trained models and results are stored in an S3 bucket (`modwt-lstm-results`) with the following structure:

- Models (`.keras` files)
- Scalers (`.pkl` files)
- Feature names and selection results
- Performance metrics
- Predictions and labels
- Training histories
- Timing information

See `context/result_explanation.md` for detailed structure.

## Citation

If you use this code in your research, please cite:

```
[Citation information to be added]
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- CAMELS dataset providers
- `hydroIVS` R package developers
- Compute Canada (Digital Alliance) for HPC resources

---

For questions or issues, please open an issue on this repository.
