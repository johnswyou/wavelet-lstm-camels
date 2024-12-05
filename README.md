# Wavelet-Coupled LSTM Forecasting with CAMELS Data

## Overview

This repository houses the complete codebase and datasets utilized in my Master's research thesis titled **"Improving Short-term Streamflow Forecasting with Wavelet Transforms: A Large-Sample Evaluation"**. The project focuses on developing advanced forecasting models by integrating Wavelet Transform techniques with Long Short-Term Memory (LSTM) neural networks, leveraging the comprehensive CAMELS (Catchment Attributes and Meteorology for Large-sample Studies) dataset.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Evaluation](#evaluation)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Wavelet Transform Integration:** Enhances model performance by decomposing time-series data into different frequency components.
- **LSTM Neural Networks:** Utilizes LSTM layers for capturing temporal dependencies in hydrological data.
- **Custom Metrics:** Implements specialized metrics like Nash-Sutcliffe Efficiency (NSE) and Kling-Gupta Efficiency (KGE) for model evaluation.
- **Feature Engineering:** Employs MODWT (Maximal Overlap Discrete Wavelet Transform) for feature extraction.
- **Baseline Models:** Includes baseline LSTM models for performance comparison.
- **Automated Feature Selection:** Uses hydroIVS for feature importance and selection.

## Architecture

The main workflow consists of the following steps:

1. **Data Loading and Preprocessing:** Handles missing values, feature selection, and engineering using MODWT.
2. **Scaling and Normalization:** Applies Min-Max scaling to input features and target variables.
3. **Feature Selection:** Selects significant features using hydroIVS based on mutual information.
4. **Model Building:** Constructs an LSTM with dropout regularization.
5. **Training:** Trains the model with early stopping based on validation set metrics.
6. **Evaluation:** Assesses model performance using custom and standard metrics.
7. **Saving Models and Results:** Persists trained models and evaluation metrics for future use.

## Data

### CAMELS Dataset

The CAMELS dataset provides a wide range of hydrological and meteorological attributes for 671 catchments in the United States. This repository uses 620 CSV files from the CAMELS dataset, each corresponding to a unique catchment.

- **CSV Filenames:** Listed in [`csv_filenames.txt`](./csv_filenames.txt)
- **Data Attributes:** Include streamflow (`Q`), precipitation (`prcp(mm/day)`), temperature (`tmax(C)`, `tmin(C)`), solar radiation (`srad(W/m²)`), and more.

**Note**: Presently, there are 621 CSV files stored in `data.zip`. We are only interested in the 620 csv files listed in `csv_filenames.txt`. In the future, `data.zip` will be modified to drop the one additional CSV file (`09535100_camels.csv`) that does not get used.

**Note**: We drop 51 catchments (CSV files) from the original 671 CAMELS dataset due to missing data, data quality, and/or data quantity issues.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/wavelet-lstm-camels.git
   cd wavelet-lstm-camels
   ```

2. **Create a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install --upgrade pip

   pip install tensorflow
   pip install keras
   pip install rpy2==3.1.0
   pip install scikit-learn
   pip install scipy
   pip install pandas==1.5.3
   ```

   *If you encounter issues with R dependencies, ensure that R is installed on your system and that necessary R packages (`hydroIVS`, `remotes`, etc.) are available. Furthermore, ensure your `R_HOME` environment variable is set.*

## Usage

### Training Models

Execute the main training script with the required arguments:

```bash
python main.py --csv_filename PATH_TO_CSV --base_save_path PATH_TO_SAVE --base_csv_path PATH_TO_CAMELS_DATA [--max_level MAX_LEVEL] [--verbose]
```

**Arguments:**

- `--csv_filename`: Path to the CAMELS CSV file for the specific catchment.
- `--base_save_path`: Directory where all outputs (models, metrics, scalers) will be saved.
- `--base_csv_path`: Directory containing all CAMELS CSV files.
- `--max_level`: (Optional) Maximum decomposition level for MODWT. Default is `6`.
- `--verbose`: (Optional) Enable verbose logging.

**Example:**

```bash
python main.py --csv_filename 12358500_camels.csv --base_save_path ./output --base_csv_path ./data/camels --max_level 5 --verbose
```

### Evaluation

After training, evaluation metrics are saved in the specified `base_save_path` under the corresponding catchment and lead time directories. Additionally, prediction and true values are stored for further analysis.

## Directory Structure

```
wavelet-lstm-camels/
├── data/
│   └── camels/
│       ├── 12358500_camels.csv
│       ├── 06934000_camels.csv
│       └── ... (other CSV files)
├── output/
│   └── {catchment_id}/
│       └── leadtime_{h}/
│           └── {filter_shortname}/
│               ├── model.keras
│               ├── baseline_model.keras
│               ├── feature_scaler.pkl
│               ├── baseline_feature_scaler.pkl
│               ├── q_scaler.pkl
│               ├── baseline_q_scaler.pkl
│               ├── ea_cmi_tol_005_selected_feature_names.pkl
│               ├── timings.pkl
│               ├── history.pkl
│               ├── baseline_history.pkl
│               ├── test_metrics_dict.pkl
│               ├── baseline_test_metrics_dict.pkl
│               ├── pred_label_df.pkl
│               └── baseline_pred_label_df.pkl
├── main.py
├── metrics.py
├── utils.py
├── csv_filenames.txt
└── README.md
```

## Dependencies

- **Python Libraries:**
  - `argparse`
  - `logging`
  - `numpy`
  - `pandas`
  - `rpy2`
  - `tensorflow`
  - `scikit-learn`
  - `feature_engineering`
  - `hydroIVS`
  
- **R Packages:**
  - `hydroIVS`
  - `remotes`
  - `utils`

*Ensure that R is installed and properly configured on your system to handle `rpy2` dependencies.*

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

---

*For any inquiries or issues, please open an [issue](https://github.com/johnswyou/wavelet-lstm-camels/issues) on this repository.*
