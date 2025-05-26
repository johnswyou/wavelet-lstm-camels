Here's the consistent directory structure that all directories under `mnt/correct_output` follow:

```
mnt/correct_output/
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

**At the deepest level** (e.g., `mnt/correct_output/01013500/leadtime_1/db1/`), each directory contains these files:

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

where:

- **Station IDs** represent different streamflow gauge locations
- **Lead times** (1, 3, 5) represent different forecast horizons
- **Wavelet names** (db1, db2, sym4, etc.) represent different wavelets used for feature engineering
- **Files** contain trained models, scalers, predictions, metrics, and timing information for each experiment configuration

**Note**: The structure and content of the above files (specifically, the `.pkl` files shown above) are described in greater detail in `./context/result_explanation_1.md`.

**Note**: `mnt` is a directory that was mounted via s3 mountpoint. The folder simulates the "feeling" of accessing the s3 bucket (`s3://modwt-lstm-results`) locally, as though the bucket's contents were available locally.