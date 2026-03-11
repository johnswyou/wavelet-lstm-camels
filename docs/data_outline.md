# LLM Context for wavelet-lstm-camels Repository

This document provides an overview of the structure and content of key files within the `wavelet-lstm-camels` repository to aid an LLM in understanding the project.

## 1. CSV Files in `data/`

All CSV files located in the `data/` directory share the same column structure. These files likely contain time-series data related to hydrological and meteorological observations for different CAMELS (Catchment Attributes and MEteorology for Large-sample Studies) basins.

**Common Column Structure:**

Based on a sample file (`data/02096846_camels.csv`), the columns are:

*   `date`: The date of the observation (YYYY-MM-DD).
*   `Q`: Likely streamflow or discharge (flow rate). Units are not specified in the header but are often m³/s or cfs.
*   `flag`: A flag associated with the `Q` value, possibly indicating data quality or estimation method (e.g., "A:e", "A").
*   `dayl(s)`: Daylength in seconds.
*   `prcp(mm/day)`: Precipitation in millimeters per day.
*   `srad(W/m2)`: Solar radiation in Watts per square meter.
*   `swe(mm)`: Snow water equivalent in millimeters.
*   `tmax(C)`: Maximum air temperature in Celsius.
*   `tmin(C)`: Minimum air temperature in Celsius.
*   `vp(Pa)`: Vapor pressure in Pascals.

**Example:**
The `data/` directory contains numerous CSV files, each named with a basin identifier followed by `_camels.csv` (e.g., `02096846_camels.csv`, `07263295_camels.csv`).

## 2. Pickle Files in `filters/`

The `filters/` directory contains the following pickle files:

*   `wavelet_dict.pkl`
*   `scaling_dict.pkl`

Pickle files (`.pkl`) are Python-specific binary files used for serializing and de-serializing Python object structures. Inspection of these files (using a helper script that loads them with `pickle.load()`) reveals the following structure:

**`filters/wavelet_dict.pkl`**
*   **Type:** Python dictionary.
*   **Number of Keys:** 128.
*   **Keys:** Strings representing standard wavelet names (e.g., 'bl7', 'bl9', 'beyl', 'coif1', 'db1' through 'db45', 'fk4' through 'fk22', 'sym2' through 'sym45', 'vaid', 'la8' through 'la20', etc.).
*   **Values:** Each key maps to a NumPy array. For example, the key 'bl7' maps to a NumPy array of shape (14,). These arrays likely store the **wavelet filter coefficients** for the corresponding wavelet type.

**`filters/scaling_dict.pkl`**
*   **Type:** Python dictionary.
*   **Number of Keys:** 128.
*   **Keys:** Strings representing standard wavelet names, identical to those in `wavelet_dict.pkl`.
*   **Values:** Each key maps to a NumPy array. For example, the key 'bl7' maps to a NumPy array of shape (14,). These arrays likely store the **scaling function filter coefficients** for the corresponding wavelet type.

The purpose of these files is to store pre-defined wavelet and scaling function coefficients, which are essential for performing wavelet transforms on the time-series data.

## 3. Text File `csv_filenames.txt`

The `csv_filenames.txt` file is a plain text file that contains a list of CSV filenames.

**Structure:**

*   Each line in the file corresponds to the filename of a CSV file that is presumably located in the `data/` directory.
*   The filenames follow the pattern `[basin_identifier]_camels.csv`.

**Purpose:**

This file likely serves as a manifest or an input list for scripts that process multiple CAMELS dataset files. It allows for easy iteration over a specific set of basins/CSV files.

**Example Content:**
```
12358500_camels.csv
06934000_camels.csv
09081600_camels.csv
...
``` 